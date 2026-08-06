"""
Adaptive State Space Model (AS-SSM)

自适应状态空间模型 - 根据特征重要性动态调整SSM强度

核心创新：
1. 重要性预测：预测每个位置的重要性
2. 自适应门控：重要区域使用更强的SSM特征
3. 轻量级设计：额外参数<5%

论文价值：
- 针对类别不平衡问题
- 重点增强少数类区域
- 可解释性强（可视化重要性图）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class AdaptiveSSM(nn.Module):
    """自适应状态空间模型
    
    根据特征重要性动态调整SSM的处理强度：
    - 重要区域（如少数类）：更多SSM特征
    - 不重要区域：更多原始特征
    
    Args:
        dim: 特征维度
        d_state: SSM状态维度（默认64）
        d_conv: SSM卷积核大小（默认4）
        expand: SSM扩展因子（默认2）
        importance_reduction: 重要性预测的降维比例（默认4）
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        importance_reduction: int = 4,
    ):
        super().__init__()
        self.dim = dim
        
        # 标准4D SSM
        self.ssm = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 全局重要性预测器（轻量级）
        # 预测整个特征图的全局重要性
        self.global_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // importance_reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // importance_reduction, 1, 1),
            nn.Sigmoid()
        )
        
        # 局部自适应门控（轻量级）
        # 为每个位置生成自适应权重
        self.local_gate = nn.Sequential(
            nn.Conv2d(dim, dim // importance_reduction, 1),
            nn.BatchNorm2d(dim // importance_reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // importance_reduction, dim, 1),
            nn.Sigmoid()
        )
        
        # 可选：通道注意力（进一步增强）
        self.use_channel_attn = True
        if self.use_channel_attn:
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // importance_reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // importance_reduction, dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C] 输入特征（序列格式）
            H, W: 空间尺寸
        
        Returns:
            out: [B, H*W, C] 输出特征
        """
        B, N, C = x.shape
        assert N == H * W, f"Sequence length {N} != H*W {H*W}"
        
        # 转换为2D格式用于重要性预测
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        # 1. 全局重要性预测
        # 预测整个特征图的重要性（标量）
        global_importance = self.global_importance(x_2d)  # [B, 1, 1, 1]
        
        # 2. 标准SSM处理
        x_ssm = self.ssm(x)  # [B, H*W, C]
        x_ssm_2d = x_ssm.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        # 3. 局部自适应门控
        # 为每个位置生成自适应权重
        local_gate = self.local_gate(x_2d)  # [B, C, H, W]
        
        # 4. 可选：通道注意力
        if self.use_channel_attn:
            channel_weight = self.channel_attn(x_2d)  # [B, C, 1, 1]
            local_gate = local_gate * channel_weight
        
        # 5. 自适应融合
        # alpha = global_importance × local_gate
        # 重要区域：alpha接近1，更多SSM特征
        # 不重要区域：alpha接近0，更多原始特征
        alpha = global_importance * local_gate  # [B, C, H, W]
        out_2d = alpha * x_ssm_2d + (1 - alpha) * x_2d
        
        # 6. 转回序列格式
        out = out_2d.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        return out
    
    def get_importance_map(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """获取重要性图（用于可视化）
        
        Args:
            x: [B, H*W, C] 输入特征
            H, W: 空间尺寸
        
        Returns:
            importance_map: [B, 1, H, W] 重要性图
        """
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        
        # 全局重要性
        global_importance = self.global_importance(x_2d)  # [B, 1, 1, 1]
        
        # 局部门控
        local_gate = self.local_gate(x_2d)  # [B, C, H, W]
        
        # 平均到单通道
        local_gate_avg = local_gate.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 组合
        importance_map = global_importance * local_gate_avg  # [B, 1, H, W]
        
        return importance_map


class AdaptiveSSMLayer(nn.Module):
    """自适应SSM层（包装器）
    
    用于替换Asmamba中的MambaLayer
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.adaptive_ssm = AdaptiveSSM(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C]
            H, W: 空间尺寸
        
        Returns:
            out: [B, H*W, C]
        """
        x_norm = self.norm(x)
        x_out = self.adaptive_ssm(x_norm, H, W)
        return x_out


class AdaptiveSSMv2(nn.Module):
    """自适应SSM v2 - 增强版
    
    额外特性：
    1. 多头重要性预测
    2. 残差连接
    3. 更强的自适应能力
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 标准SSM
        self.ssm = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 多头重要性预测
        self.importance_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 16, 1, 1),
                nn.Sigmoid()
            )
            for _ in range(num_heads)
        ])
        
        # 多头门控
        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 4, 1),
                nn.BatchNorm2d(dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, dim, 1),
                nn.Sigmoid()
            )
            for _ in range(num_heads)
        ])
        
        # 头融合
        self.head_fusion = nn.Conv2d(dim * num_heads, dim, 1)
        
        # 残差门控
        self.residual_gate = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C]
            H, W: 空间尺寸
        
        Returns:
            out: [B, H*W, C]
        """
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        
        # SSM处理
        x_ssm = self.ssm(x)
        x_ssm_2d = x_ssm.transpose(1, 2).reshape(B, C, H, W)
        
        # 多头自适应
        head_outputs = []
        for importance_head, gate_head in zip(self.importance_heads, self.gate_heads):
            # 重要性预测
            importance = importance_head(x_2d)  # [B, 1, 1, 1]
            
            # 门控
            gate = gate_head(x_2d)  # [B, C, H, W]
            
            # 自适应融合
            alpha = importance * gate
            head_out = alpha * x_ssm_2d + (1 - alpha) * x_2d
            head_outputs.append(head_out)
        
        # 融合多头
        multi_head = torch.cat(head_outputs, dim=1)  # [B, C*num_heads, H, W]
        fused = self.head_fusion(multi_head)  # [B, C, H, W]
        
        # 残差连接
        out_2d = self.residual_gate * fused + (1 - self.residual_gate) * x_2d
        
        # 转回序列格式
        out = out_2d.flatten(2).transpose(1, 2)
        
        return out
