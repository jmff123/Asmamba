# Copyright (c) OpenMMLab. All rights reserved.
"""
DCBS迭代更新Hook

用于在每次迭代时更新IterBasedDCBSSampler的迭代数，
确保α_t能够正确随训练进度动态变化。
"""

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DCBSIterHook(Hook):
    """DCBS迭代更新Hook
    
    在每次迭代前更新采样器的迭代数，使α_t能够正确衰减。
    
    Args:
        interval: 更新间隔（默认每次迭代都更新）
    """
    
    def __init__(self, interval: int = 1):
        self.interval = interval
    
    def before_train_iter(self, runner, batch_idx: int, data_batch=None):
        """在每次训练迭代前更新采样器的迭代数"""
        # 获取当前迭代数
        current_iter = runner.iter
        
        # 只在指定间隔更新
        if current_iter % self.interval != 0:
            return
        
        # 获取数据加载器的采样器
        dataloader = runner.train_dataloader
        if hasattr(dataloader, 'sampler'):
            sampler = dataloader.sampler
            
            # 检查是否是IterBasedDCBSSampler
            if hasattr(sampler, 'set_iter'):
                sampler.set_iter(current_iter)
                
                # 每1000次迭代打印一次α_t信息
                if current_iter % 1000 == 0 and hasattr(sampler, '_get_alpha_t'):
                    alpha_t = sampler._get_alpha_t()
                    runner.logger.info(
                        f"DCBS更新: iter={current_iter}, α_t={alpha_t:.6f}"
                    )
