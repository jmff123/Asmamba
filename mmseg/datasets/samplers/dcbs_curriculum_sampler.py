# Copyright (c) OpenMMLab. All rights reserved.
"""
课程学习增强的DCBS采样器 (DCBS with Curriculum Learning)

核心思路：
结合互补检测和课程学习，分阶段训练：
1. 前期：专注简单样本（少量类别1）
2. 中期：引入中等样本（适量类别1）
3. 后期：全面学习（所有样本 + 互补检测增强）

优势：
- 训练更稳定（从易到难）
- 收敛更快（避免早期被困难样本干扰）
- 效果更好（后期互补检测精准增强）
"""

import numpy as np
from typing import Iterator, Optional
import mmcv

from .dcbs_complement_sampler import IterBasedDCBSComplementSampler
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DCBSCurriculumSampler(IterBasedDCBSComplementSampler):
    """课程学习增强的DCBS采样器
    
    在互补检测DCBS基础上，添加课程学习机制。
    
    Args:
        dataset: 数据集对象
        num_classes: 类别数量，默认3
        alpha_0: DCBS初始调整系数，默认0.01
        total_iters: 总训练迭代数
        samples_per_gpu: 每个GPU的batch size
        seed: 随机种子
        shuffle: 是否打乱
        
        # 互补检测参数
        complement_weight: 互补检测权重β，默认0.5
        high_precision_threshold: 启动阈值，默认0.65
        complement_start_iter: 互补检测启动迭代，默认10%
        target_class: 目标类别，默认1
        reference_classes: 参考类别，默认[0, 2]
        
        # 课程学习参数（新增）
        curriculum_stages: 课程阶段配置，默认3阶段
        enable_curriculum: 是否启用课程学习，默认True
    """
    
    def __init__(
        self,
        dataset,
        num_classes: int = 3,
        alpha_0: float = 0.01,
        total_iters: int = 80000,
        samples_per_gpu: int = 1,
        seed: Optional[int] = None,
        shuffle: bool = True,
        # 互补检测参数
        complement_weight: float = 0.5,
        high_precision_threshold: float = 0.65,
        complement_start_iter: Optional[int] = None,
        target_class: int = 1,
        reference_classes: Optional[list] = None,
        # 课程学习参数
        curriculum_stages: Optional[list] = None,
        enable_curriculum: bool = True,
    ):
        # 课程学习配置
        self.enable_curriculum = enable_curriculum
        
        # 默认3阶段课程
        if curriculum_stages is None:
            self.curriculum_stages = [
                {
                    'name': 'easy',
                    'start_iter': 0,
                    'end_iter': int(total_iters * 0.05),  # 0-5%
                    'target_ratio_range': (0.0, 0.05),  # 类别1占比<5%
                    'description': '简单样本：建立基础'
                },
                {
                    'name': 'medium',
                    'start_iter': int(total_iters * 0.05),
                    'end_iter': int(total_iters * 0.10),  # 5-10%
                    'target_ratio_range': (0.0, 0.15),  # 类别1占比<15%
                    'description': '中等样本：逐步引入'
                },
                {
                    'name': 'hard',
                    'start_iter': int(total_iters * 0.10),
                    'end_iter': total_iters,  # 10-100%
                    'target_ratio_range': (0.0, 1.0),  # 所有样本
                    'description': '困难样本：全面学习 + 互补检测'
                }
            ]
        else:
            self.curriculum_stages = curriculum_stages
        
        # 预计算每个样本的目标类别占比（用于课程学习）
        self.sample_target_ratios = self._compute_target_ratios(dataset, target_class)
        
        # 调用父类初始化
        super().__init__(
            dataset=dataset,
            num_classes=num_classes,
            alpha_0=alpha_0,
            total_iters=total_iters,
            samples_per_gpu=samples_per_gpu,
            seed=seed,
            shuffle=shuffle,
            complement_weight=complement_weight,
            high_precision_threshold=high_precision_threshold,
            complement_start_iter=complement_start_iter,
            target_class=target_class,
            reference_classes=reference_classes,
        )
        
        print(f"\n{'='*70}")
        print("课程学习增强DCBS采样器初始化")
        print(f"{'='*70}")
        print(f"课程学习: {'启用' if enable_curriculum else '禁用'}")
        if enable_curriculum:
            print(f"\n课程阶段配置:")
            for i, stage in enumerate(self.curriculum_stages):
                print(f"  阶段{i+1} ({stage['name']}): "
                      f"Iter {stage['start_iter']}-{stage['end_iter']}")
                print(f"    目标占比范围: {stage['target_ratio_range'][0]:.1%}-{stage['target_ratio_range'][1]:.1%}")
                print(f"    描述: {stage['description']}")
        print(f"{'='*70}\n")
    
    def _compute_target_ratios(self, dataset, target_class: int) -> np.ndarray:
        """预计算每个样本的目标类别占比
        
        Args:
            dataset: 数据集对象
            target_class: 目标类别
        
        Returns:
            target_ratios: shape (num_samples,)
        """
        print("正在计算样本目标类别占比（用于课程学习）...")
        
        target_ratios = np.zeros(len(dataset), dtype=np.float32)
        
        for idx in range(len(dataset)):
            data_info = dataset.get_data_info(idx)
            seg_map_path = data_info.get('seg_map_path', None)
            
            if seg_map_path is None:
                continue
            
            try:
                seg_map = mmcv.imread(seg_map_path, flag='unchanged')
                target_ratios[idx] = np.sum(seg_map == target_class) / seg_map.size
            except Exception as e:
                continue
            
            if (idx + 1) % 500 == 0:
                print(f"  已处理 {idx + 1}/{len(dataset)} 张图像")
        
        print(f"✓ 目标类别占比计算完成")
        print(f"  占比分布: min={target_ratios.min():.3f}, "
              f"max={target_ratios.max():.3f}, "
              f"mean={target_ratios.mean():.3f}")
        
        return target_ratios
    
    def _get_current_stage(self) -> dict:
        """获取当前训练阶段
        
        Returns:
            stage: 当前阶段配置
        """
        if not self.enable_curriculum:
            # 禁用课程学习，返回最后阶段（所有样本）
            return self.curriculum_stages[-1]
        
        for stage in self.curriculum_stages:
            if stage['start_iter'] <= self.current_iter < stage['end_iter']:
                return stage
        
        # 超出范围，返回最后阶段
        return self.curriculum_stages[-1]
    
    def _compute_sample_weights(self) -> np.ndarray:
        """计算样本权重（课程学习增强版）
        
        在互补检测的基础上，添加课程学习过滤。
        
        Returns:
            sample_weights: shape (num_samples,)
        """
        # 获取当前阶段
        current_stage = self._get_current_stage()
        
        # 打印阶段信息
        if self.rank == 0 and self.enable_curriculum:
            print(f"\n[课程学习] 当前阶段: {current_stage['name']}")
            print(f"  Iter {self.current_iter}/{self.total_iters} "
                  f"({self.current_iter/self.total_iters*100:.1f}%)")
            print(f"  目标占比范围: {current_stage['target_ratio_range'][0]:.1%}-"
                  f"{current_stage['target_ratio_range'][1]:.1%}")
        
        # 调用父类方法获取基础权重（包含互补检测）
        base_weights = super()._compute_sample_weights()
        
        # 应用课程学习过滤
        if self.enable_curriculum:
            min_ratio, max_ratio = current_stage['target_ratio_range']
            
            # 创建课程掩码
            curriculum_mask = (
                (self.sample_target_ratios >= min_ratio) &
                (self.sample_target_ratios <= max_ratio)
            )
            
            # 过滤样本：不符合当前阶段的样本权重设为0
            filtered_weights = base_weights * curriculum_mask
            
            # 重新归一化（只在有效样本上）
            valid_sum = filtered_weights.sum()
            if valid_sum > 0:
                filtered_weights = filtered_weights / valid_sum * len(self.dataset)
            else:
                # 如果没有有效样本，回退到基础权重
                filtered_weights = base_weights
            
            # 统计过滤效果
            if self.rank == 0:
                n_valid = curriculum_mask.sum()
                n_total = len(self.dataset)
                print(f"  有效样本: {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)")
                print(f"  权重范围: [{filtered_weights[filtered_weights>0].min():.4f}, "
                      f"{filtered_weights.max():.4f}]")
            
            return filtered_weights
        else:
            return base_weights
