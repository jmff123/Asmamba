"""
Adaptive Weight Logger Hook

记录自适应损失权重的变化，用于监控和可视化
"""

from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class AdaptiveWeightLoggerHook(Hook):
    """记录自适应损失权重的Hook
    
    在训练过程中记录每个损失的权重变化，
    方便监控和后续可视化分析。
    
    Args:
        log_interval: 记录间隔（默认50次迭代）
    """
    
    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
    
    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None
    ) -> None:
        """训练迭代后记录权重"""
        
        # 只在指定间隔记录
        if runner.iter % self.log_interval != 0:
            return
        
        # 获取模型
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # 检查是否有decode_head且启用了自适应权重
        if not hasattr(model, 'decode_head'):
            return
        
        decode_head = model.decode_head
        if not hasattr(decode_head, 'use_adaptive_weights'):
            return
        
        if not decode_head.use_adaptive_weights:
            return
        
        if not hasattr(decode_head, 'loss_balancer'):
            return
        
        loss_balancer = decode_head.loss_balancer
        
        # 获取当前损失值（从outputs中）
        if outputs is None or 'loss' not in outputs:
            return
        
        # 模拟损失字典（用于计算权重）
        # 注意：这里我们无法获取实际的损失值，所以使用占位符
        # 实际权重会在forward时计算
        losses = {}
        for name in loss_balancer.loss_names:
            losses[name] = runner.message_hub.get_scalar(f'train/{name}').current()
            if losses[name] is None:
                losses[name] = 1.0  # 占位符
        
        # 计算当前权重
        try:
            import torch
            losses_tensor = {k: torch.tensor(v) for k, v in losses.items()}
            weights = loss_balancer(losses_tensor, runner.iter)
            
            # 记录权重到message_hub
            for name, weight in weights.items():
                runner.message_hub.update_scalar(
                    f'train/{name}_weight',
                    weight,
                    runner.iter
                )
        except Exception as e:
            # 如果计算失败，静默跳过
            pass
