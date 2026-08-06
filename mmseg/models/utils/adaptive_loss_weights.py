"""
Adaptive Loss Weight Balancing Module

基于合理公式动态调整各个损失的权重，实现自适应平衡。

核心思想：
1. 基于损失值的相对大小动态调整权重
2. 基于训练进度调整权重（curriculum learning）
3. 基于类别难度调整权重
4. 确保各损失对总梯度的贡献相对平衡
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class AdaptiveLossWeightBalancer(nn.Module):
    """自适应损失权重平衡器
    
    使用多种策略动态调整损失权重：
    1. 梯度归一化（Gradient Normalization）
    2. 不确定性加权（Uncertainty Weighting）
    3. 课程学习（Curriculum Learning）
    4. 损失值归一化（Loss Magnitude Balancing）
    
    Args:
        loss_names: 损失名称列表
        strategy: 权重策略 ['gradient_norm', 'uncertainty', 'curriculum', 'magnitude', 'hybrid']
        base_weights: 基础权重（可选）
        total_iters: 总迭代次数（用于curriculum）
        warmup_iters: warmup迭代次数
        alpha: 平滑系数（EMA）
    """
    
    def __init__(
        self,
        loss_names: list,
        strategy: str = 'hybrid',
        base_weights: Optional[Dict[str, float]] = None,
        total_iters: int = 80000,
        warmup_iters: int = 4500,
        alpha: float = 0.9,
    ):
        super().__init__()
        
        self.loss_names = loss_names
        self.strategy = strategy
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.alpha = alpha
        
        # 基础权重（如果未提供，使用均匀权重）
        if base_weights is None:
            base_weights = {name: 1.0 for name in loss_names}
        self.base_weights = base_weights
        
        # 可学习的不确定性参数（用于uncertainty策略）
        # 注意：ParameterDict的key不能包含"."，所以替换为"_"
        self.log_vars = nn.ParameterDict({
            name.replace('.', '_'): nn.Parameter(torch.zeros(1))
            for name in loss_names
        })
        
        # 损失值的移动平均（用于magnitude策略）
        self.register_buffer('loss_ema', torch.ones(len(loss_names)))
        
        # 当前迭代
        self.register_buffer('current_iter', torch.tensor(0))
    
    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        iter_num: Optional[int] = None
    ) -> Dict[str, float]:
        """计算自适应权重
        
        Args:
            losses: 损失字典 {loss_name: loss_value}
            iter_num: 当前迭代次数（可选）
        
        Returns:
            weights: 权重字典 {loss_name: weight}
        """
        if iter_num is not None:
            self.current_iter.fill_(iter_num)
        
        if self.strategy == 'gradient_norm':
            return self._gradient_norm_weights(losses)
        elif self.strategy == 'uncertainty':
            return self._uncertainty_weights(losses)
        elif self.strategy == 'curriculum':
            return self._curriculum_weights(losses)
        elif self.strategy == 'magnitude':
            return self._magnitude_weights(losses)
        elif self.strategy == 'hybrid':
            return self._hybrid_weights(losses)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _gradient_norm_weights(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """基于梯度范数的权重
        
        思想：让每个损失对总梯度的贡献相对平衡
        公式：w_i = 1 / ||∇L_i||
        """
        weights = {}
        grad_norms = {}
        
        # 计算每个损失的梯度范数
        for name, loss in losses.items():
            if loss.requires_grad:
                grad = torch.autograd.grad(
                    loss, loss, 
                    retain_graph=True, 
                    create_graph=False
                )[0]
                grad_norms[name] = grad.norm().item()
            else:
                grad_norms[name] = 1.0
        
        # 归一化权重
        total_grad_norm = sum(grad_norms.values())
        for name in self.loss_names:
            if name in grad_norms:
                weights[name] = total_grad_norm / (grad_norms[name] * len(grad_norms))
            else:
                weights[name] = self.base_weights.get(name, 1.0)
        
        return weights
    
    def _uncertainty_weights(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """基于不确定性的权重（Multi-Task Learning Using Uncertainty）
        
        论文：Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
        公式：L_total = Σ (1/(2σ²_i)) * L_i + log(σ_i)
        权重：w_i = 1/(2σ²_i)
        """
        weights = {}
        
        for name in self.loss_names:
            # 将"."替换为"_"以匹配ParameterDict的key
            param_name = name.replace('.', '_')
            if param_name in self.log_vars:
                # σ = exp(log_var)
                # w = 1/(2σ²) = exp(-2*log_var) / 2
                precision = torch.exp(-2 * self.log_vars[param_name])
                weights[name] = (precision / 2).item()
            else:
                weights[name] = self.base_weights.get(name, 1.0)
        
        return weights
    
    def _curriculum_weights(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """基于课程学习的权重
        
        思想：训练初期关注简单损失，后期关注复杂损失
        
        损失难度排序（从易到难）：
        1. Dice Loss（直接优化IoU，最直观）
        2. Focal Loss（关注困难样本）
        3. Lovasz Loss（IoU的凸松弛）
        4. Tversky Loss（FP/FN权衡）
        5. Contrastive Loss（特征表示，最抽象）
        
        公式：w_i(t) = w_base_i * schedule(t, difficulty_i)
        """
        progress = min(self.current_iter.item() / self.total_iters, 1.0)
        weights = {}
        
        # 定义损失难度（0-1，越大越难）
        difficulty = {
            'loss_dice': 0.0,
            'loss_focal': 0.25,
            'loss_lovasz': 0.5,
            'loss_tversky': 0.75,
            'loss_contrastive': 1.0,
        }
        
        for name in self.loss_names:
            base_weight = self.base_weights.get(name, 1.0)
            diff = difficulty.get(name, 0.5)
            
            # 课程调度函数：sigmoid曲线
            # 简单损失：早期高权重
            # 困难损失：后期高权重
            if progress < 0.1:  # warmup阶段
                schedule = 1.0
            else:
                # sigmoid: 1 / (1 + exp(-k*(t - t0)))
                k = 10  # 陡峭度
                t0 = diff  # 中心点（与难度对齐）
                schedule = 1.0 / (1.0 + math.exp(-k * (progress - t0)))
            
            weights[name] = base_weight * schedule
        
        return weights
    
    def _magnitude_weights(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """基于损失值大小的权重
        
        思想：平衡不同损失的数值范围
        公式：w_i = mean(L) / (L_i + ε)
        """
        weights = {}
        loss_values = []
        
        # 收集损失值
        for name in self.loss_names:
            if name in losses:
                loss_values.append(losses[name].item())
            else:
                loss_values.append(1.0)
        
        # 更新EMA
        if self.training:
            current_losses = torch.tensor(loss_values, device=self.loss_ema.device)
            self.loss_ema = self.alpha * self.loss_ema + (1 - self.alpha) * current_losses
        
        # 计算权重
        mean_loss = self.loss_ema.mean()
        for i, name in enumerate(self.loss_names):
            base_weight = self.base_weights.get(name, 1.0)
            # 归一化权重：让所有损失的加权值接近
            weights[name] = base_weight * (mean_loss / (self.loss_ema[i] + 1e-6)).item()
        
        return weights
    
    def _hybrid_weights(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """混合策略：结合多种方法
        
        公式：w_i = w_base_i * w_curriculum_i * w_magnitude_i
        """
        # 1. 课程学习权重
        curriculum_weights = self._curriculum_weights(losses)
        
        # 2. 损失值平衡权重
        magnitude_weights = self._magnitude_weights(losses)
        
        # 3. 组合权重
        weights = {}
        for name in self.loss_names:
            w_curr = curriculum_weights.get(name, 1.0)
            w_mag = magnitude_weights.get(name, 1.0)
            
            # 几何平均（避免某个权重过小导致总权重过小）
            weights[name] = math.sqrt(w_curr * w_mag)
        
        return weights
    
    def get_uncertainty_loss(self) -> torch.Tensor:
        """获取不确定性正则化损失
        
        用于uncertainty策略，防止σ过大
        L_reg = Σ log(σ_i) = Σ log_var_i
        """
        if self.strategy != 'uncertainty':
            return torch.tensor(0.0)
        
        return sum(self.log_vars.values())


def create_adaptive_loss_balancer(
    num_classes: int = 3,
    use_contrastive: bool = True,
    strategy: str = 'hybrid',
    total_iters: int = 80000,
) -> AdaptiveLossWeightBalancer:
    """创建自适应损失平衡器的工厂函数
    
    Args:
        num_classes: 类别数
        use_contrastive: 是否使用对比损失
        strategy: 权重策略
        total_iters: 总迭代次数
    
    Returns:
        balancer: 损失平衡器
    """
    # 定义损失名称
    loss_names = [
        'decode.loss_dice',
        'decode.loss_focal',
        'decode.loss_lovasz',
        'decode.loss_tversky',
    ]
    
    if use_contrastive:
        loss_names.append('decode.loss_contrastive')
    
    # 辅助头损失
    loss_names.extend([
        'aux.loss_dice',
        'aux.loss_focal',
        'aux.loss_lovasz',
    ])
    
    # 基础权重（相对重要性）
    base_weights = {
        # Decode head（主要）
        'decode.loss_dice': 3.0,      # 最重要：直接优化IoU
        'decode.loss_focal': 2.0,     # 重要：困难样本
        'decode.loss_lovasz': 2.0,    # 重要：IoU理论保证
        'decode.loss_tversky': 1.5,   # 中等：FP/FN权衡
        'decode.loss_contrastive': 0.5,  # 辅助但重要：特征表示
        
        # Auxiliary head（辅助）
        'aux.loss_dice': 2.0,
        'aux.loss_focal': 1.2,
        'aux.loss_lovasz': 1.0,
    }
    
    balancer = AdaptiveLossWeightBalancer(
        loss_names=loss_names,
        strategy=strategy,
        base_weights=base_weights,
        total_iters=total_iters,
        warmup_iters=4500,
        alpha=0.9,
    )
    
    return balancer


# ============================================================================
# 权重计算公式说明
# ============================================================================

"""
## 1. 梯度归一化策略（Gradient Normalization）

**公式**：
```
w_i = mean(||∇L_j||) / ||∇L_i||
```

**思想**：让每个损失对总梯度的贡献相对平衡

**优点**：
- 自动平衡不同损失的梯度尺度
- 避免某个损失主导训练

**缺点**：
- 计算开销大（需要计算梯度）
- 可能不稳定


## 2. 不确定性加权策略（Uncertainty Weighting）

**公式**：
```
L_total = Σ (1/(2σ²_i)) * L_i + log(σ_i)
w_i = 1/(2σ²_i) = exp(-2*log_var_i) / 2
```

**思想**：学习每个损失的不确定性，不确定性高的损失权重低

**优点**：
- 端到端学习权重
- 理论支撑（贝叶斯推断）

**缺点**：
- 增加可学习参数
- 可能过拟合


## 3. 课程学习策略（Curriculum Learning）

**公式**：
```
w_i(t) = w_base_i * sigmoid(k * (t - difficulty_i))

其中：
- t: 训练进度 [0, 1]
- difficulty_i: 损失难度 [0, 1]
- k: 陡峭度参数
```

**难度排序**：
```
Dice (0.0) < Focal (0.25) < Lovasz (0.5) < Tversky (0.75) < Contrastive (1.0)
```

**思想**：
- 训练初期：关注简单损失（Dice）
- 训练后期：关注复杂损失（Contrastive）

**优点**：
- 符合人类学习规律
- 训练更稳定

**缺点**：
- 需要手动定义难度


## 4. 损失值平衡策略（Magnitude Balancing）

**公式**：
```
w_i = w_base_i * mean(L_j) / L_i

其中L_i使用EMA平滑：
L_i^(t) = α * L_i^(t-1) + (1-α) * L_i^(t)
```

**思想**：平衡不同损失的数值范围

**优点**：
- 简单有效
- 自动适应损失尺度

**缺点**：
- 可能放大噪声


## 5. 混合策略（Hybrid）⭐ 推荐

**公式**：
```
w_i = sqrt(w_curriculum_i * w_magnitude_i)
```

**思想**：结合课程学习和损失值平衡

**优点**：
- 综合多种策略的优点
- 训练稳定且效果好

**缺点**：
- 稍微复杂


## 权重演化示例

### 训练初期（iter 0-10000）

```
Dice:        4.0 * 1.0 * 1.2 = 4.8  ← 高权重（简单+重要）
Focal:       2.5 * 0.8 * 1.0 = 2.0
Lovasz:      2.0 * 0.5 * 0.9 = 0.9
Tversky:     1.5 * 0.2 * 0.8 = 0.24
Contrastive: 0.1 * 0.0 * 1.1 = 0.0  ← 几乎为0（困难）
```

### 训练中期（iter 10000-40000）

```
Dice:        4.0 * 1.0 * 1.0 = 4.0
Focal:       2.5 * 0.9 * 1.0 = 2.25
Lovasz:      2.0 * 0.8 * 1.0 = 1.6
Tversky:     1.5 * 0.6 * 1.0 = 0.9
Contrastive: 0.1 * 0.5 * 1.0 = 0.05  ← 逐渐增加
```

### 训练后期（iter 40000-80000）

```
Dice:        4.0 * 1.0 * 1.0 = 4.0
Focal:       2.5 * 1.0 * 1.0 = 2.5
Lovasz:      2.0 * 1.0 * 1.0 = 2.0
Tversky:     1.5 * 0.9 * 1.0 = 1.35
Contrastive: 0.1 * 1.0 * 1.0 = 0.1  ← 达到最大值
```

## 总结

**推荐使用混合策略（hybrid）**：
- 结合课程学习和损失值平衡
- 训练初期关注简单损失
- 自动平衡损失尺度
- 训练稳定且效果好
"""
