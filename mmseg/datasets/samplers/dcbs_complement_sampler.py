# Copyright (c) OpenMMLab. All rights reserved.
"""
互补检测增强的DCBS采样器 (DCBS with Complement Detection)

核心思路：
利用类别0和2的高准确率，反向推断可能被忽略的类别1区域，
并在训练中动态调整对这些区域的关注度。

原理：
1. 当类别0和2的预测置信度超过阈值时，它们"未识别"的区域很可能是类别1
2. 计算互补检测置信度 = (P_class0 + P_class2) / 2
3. 估计被忽略的类别1区域 = 1 - max(recall_class0, recall_class2)
4. 动态调整样本权重，增加对这些"潜在类别1"区域的关注

公式：
- 互补置信度: C_comp = (precision_0 + precision_2) / 2
- 潜在类别1区域: R_potential = 1 - max(recall_0, recall_2)
- 增强权重: W_enhanced = W_direct + β × C_comp × R_potential
  其中 β 是互补检测权重系数，随训练进度动态调整
"""

import math
import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional, Dict
import mmcv
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS

from .dcbs_sampler import DCBSSampler


@DATA_SAMPLERS.register_module()
class DCBSComplementSampler(DCBSSampler):
    """互补检测增强的DCBS采样器
    
    在标准DCBS基础上，利用类别0和2的高准确率反向推断类别1区域。
    
    Args:
        dataset: 数据集对象
        num_classes: 类别数量（默认3）
        alpha_0: DCBS初始调整系数，默认0.01
        total_epochs: 总训练epoch数
        samples_per_gpu: 每个GPU的batch size
        seed: 随机种子
        shuffle: 是否打乱
        
        # 互补检测相关参数
        complement_weight: 互补检测权重系数β，默认0.5
        high_precision_threshold: 高准确率阈值，默认0.65（放宽条件）
        complement_start_epoch: 开始使用互补检测的epoch，默认总epoch的10%
        target_class: 目标类别（需要增强的类别），默认1
        reference_classes: 参考类别（用于反向推断），默认[0, 2]
    """
    
    def __init__(
        self,
        dataset,
        num_classes: int = 3,
        alpha_0: float = 0.01,
        total_epochs: int = 50,
        samples_per_gpu: int = 1,
        seed: Optional[int] = None,
        shuffle: bool = True,
        # 互补检测参数
        complement_weight: float = 0.5,
        high_precision_threshold: float = 0.65,  # 放宽阈值
        complement_start_epoch: Optional[int] = None,
        target_class: int = 1,
        reference_classes: Optional[list] = None,
    ):
        # 互补检测参数
        self.complement_weight = complement_weight
        self.high_precision_threshold = high_precision_threshold
        self.complement_start_epoch = complement_start_epoch or int(total_epochs * 0.1)  # 10%就启动
        self.target_class = target_class
        self.reference_classes = reference_classes or [0, 2]
        
        # 存储类别性能指标（在训练过程中更新）
        self.class_metrics = {
            'precision': np.ones(num_classes) * 0.5,  # 初始假设0.5
            'recall': np.ones(num_classes) * 0.5,
        }
        
        # 调用父类初始化
        super().__init__(
            dataset=dataset,
            num_classes=num_classes,
            alpha_0=alpha_0,
            total_epochs=total_epochs,
            samples_per_gpu=samples_per_gpu,
            seed=seed,
            shuffle=shuffle
        )
        
        print(f"\n{'='*70}")
        print("互补检测增强DCBS采样器初始化")
        print(f"{'='*70}")
        print(f"目标类别: {self.target_class}")
        print(f"参考类别: {self.reference_classes}")
        print(f"互补权重β: {self.complement_weight}")
        print(f"高准确率阈值: {self.high_precision_threshold}")
        print(f"互补检测启动epoch: {self.complement_start_epoch}")
        print(f"{'='*70}\n")
    
    def update_class_metrics(self, metrics: Dict[str, np.ndarray]):
        """更新类别性能指标
        
        在训练过程中，通过Hook定期调用此方法更新各类别的precision和recall。
        
        Args:
            metrics: 包含 'precision' 和 'recall' 的字典
                    每个值为 shape (num_classes,) 的numpy数组
        """
        if 'precision' in metrics:
            self.class_metrics['precision'] = metrics['precision']
        if 'recall' in metrics:
            self.class_metrics['recall'] = metrics['recall']
        
        # 打印更新后的指标（显示迭代数而不是epoch）
        if self.rank == 0:
            # 使用迭代数显示
            iter_info = f"Iter {self.current_iter}" if hasattr(self, 'current_iter') else f"Epoch {self.epoch}"
            print(f"\n[互补检测] 类别指标已更新 ({iter_info}):")
            for k in range(self.num_classes):
                print(f"  类别 {k}: Precision={self.class_metrics['precision'][k]:.4f}, "
                      f"Recall={self.class_metrics['recall'][k]:.4f}")
    
    def _compute_complement_confidence(self) -> float:
        """计算互补检测置信度
        
        基于参考类别（0和2）的precision计算。
        当它们的precision高时，说明它们的预测很可靠，
        因此它们"未识别"的区域更可能是目标类别（1）。
        
        Returns:
            complement_confidence: 互补检测置信度 [0, 1]
        """
        precisions = [self.class_metrics['precision'][c] for c in self.reference_classes]
        return np.mean(precisions)
    
    def _compute_complement_potential(self) -> float:
        """计算互补检测潜力
        
        估计参考类别"未识别"的区域比例，这些区域可能是目标类别。
        
        Returns:
            complement_potential: 潜在目标类别区域比例 [0, 1]
        """
        recalls = [self.class_metrics['recall'][c] for c in self.reference_classes]
        max_recall = np.max(recalls)
        
        # 未被参考类别识别的区域比例
        potential = max(0.0, 1.0 - max_recall)
        return potential
    
    def _should_use_complement(self) -> bool:
        """判断是否应该使用互补检测
        
        放宽的条件（满足任一即可）：
        1. 当前epoch >= complement_start_epoch 且
        2. 参考类别的平均precision >= high_precision_threshold (0.65)
           或 至少有一个参考类别的precision >= 0.75
        
        Returns:
            是否启用互补检测
        """
        if self.epoch < self.complement_start_epoch:
            return False
        
        # 计算平均precision
        complement_confidence = self._compute_complement_confidence()
        
        # 检查单个类别的precision
        max_precision = max([self.class_metrics['precision'][c] for c in self.reference_classes])
        
        # 放宽条件：平均达标 或 单个类别表现优秀
        return (complement_confidence >= self.high_precision_threshold or 
                max_precision >= 0.75)

    
    def _compute_sample_weights(self) -> np.ndarray:
        """计算样本权重（增强版）
        
        在标准DCBS权重基础上，加入互补检测增强：
        1. 标准权重：基于样本包含的类别频率
        2. 互补增强：对于包含目标类别的样本，根据互补检测结果增加权重
        
        Returns:
            sample_weights: shape (num_samples,)
        """
        # 先计算标准DCBS权重
        base_weights = super()._compute_sample_weights()
        
        # 如果不使用互补检测，直接返回基础权重
        if not self._should_use_complement():
            return base_weights
        
        print(f"\n[互补检测] 正在计算增强权重 (Epoch {self.epoch})...")
        
        # 计算互补检测参数
        complement_confidence = self._compute_complement_confidence()
        complement_potential = self._compute_complement_potential()
        
        print(f"  互补置信度: {complement_confidence:.4f}")
        print(f"  潜在区域比例: {complement_potential:.4f}")
        
        # 计算动态β系数（随训练进度调整）
        # 前期：β较大，强调互补检测
        # 后期：β减小，回归标准DCBS
        
        # 优先使用迭代数计算progress（IterBased版本）
        if hasattr(self, 'total_iters') and self.total_iters > 0:
            if hasattr(self, 'current_iter'):
                progress = self.current_iter / self.total_iters
            else:
                # 如果current_iter未设置，使用epoch估算
                progress = (self.epoch * len(self.dataset)) / self.total_iters
        else:
            # EpochBased版本
            progress = self.epoch / self.total_epochs if self.total_epochs > 0 else 0
        
        # 确保progress在[0, 1]范围内
        progress = max(0.0, min(1.0, progress))
        
        dynamic_beta = self.complement_weight * (1.0 - progress * 0.5)  # 后期衰减到50%
        
        # 确保beta不为负
        dynamic_beta = max(0.0, dynamic_beta)
        
        print(f"  训练进度: {progress:.2%}")
        print(f"  动态β系数: {dynamic_beta:.4f}")
        
        # 增强权重
        enhanced_weights = base_weights.copy()
        
        for idx in range(len(self.dataset)):
            data_info = self.dataset.get_data_info(idx)
            seg_map_path = data_info.get('seg_map_path', None)
            
            if seg_map_path is None:
                continue
            
            try:
                seg_map = mmcv.imread(seg_map_path, flag='unchanged')
                
                # 检查是否包含目标类别
                if self.target_class in seg_map:
                    # 计算目标类别的像素比例
                    target_ratio = np.sum(seg_map == self.target_class) / seg_map.size
                    
                    # 互补增强：目标类别比例越高，增强越多
                    # 改进公式：增加基础增强系数，避免增强过小
                    # 基础增强 = β × 置信度 × (1 + 潜在区域)
                    # 样本增强 = 基础增强 × sqrt(目标比例) （使用sqrt让小比例也有明显增强）
                    base_boost = dynamic_beta * complement_confidence * (1.0 + complement_potential)
                    sample_boost = base_boost * np.sqrt(target_ratio)
                    
                    enhanced_weights[idx] += sample_boost
            
            except Exception as e:
                continue
        
        # 在归一化前计算增强幅度（真实的增强效果）
        avg_boost_before = (enhanced_weights / (base_weights + 1e-10)).mean()
        
        # 不进行归一化！保持增强效果
        # PyTorch的WeightedRandomSampler会自动处理权重归一化
        # 我们只需要保持相对权重比例即可
        
        # 归一化后的增强幅度（应该保持不变）
        avg_boost_after = (enhanced_weights / (base_weights + 1e-10)).mean()
        
        print(f"  增强权重范围: [{enhanced_weights.min():.4f}, {enhanced_weights.max():.4f}]")
        print(f"  归一化前增强幅度: {avg_boost_before:.4f}x")
        print(f"  归一化后增强幅度: {avg_boost_after:.4f}x")
        
        return enhanced_weights
    
    def __iter__(self) -> Iterator[int]:
        """生成采样索引（增强版）"""
        # 重新计算样本权重（考虑最新的类别指标）
        self.sample_weights = self._compute_sample_weights()
        
        # 调用父类的迭代方法
        return super().__iter__()


@DATA_SAMPLERS.register_module()
class IterBasedDCBSComplementSampler(DCBSComplementSampler):
    """基于迭代的互补检测增强DCBS采样器
    
    适配MMSegmentation的IterBasedTrainLoop
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
        high_precision_threshold: float = 0.65,  # 放宽阈值
        complement_start_iter: Optional[int] = None,
        target_class: int = 1,
        reference_classes: Optional[list] = None,
    ):
        # 先设置迭代相关属性（在调用父类__init__之前）
        self.total_iters = total_iters
        self.current_iter = 0
        self.complement_start_iter = complement_start_iter or int(total_iters * 0.1)  # 10%就启动
        
        # 将迭代数转换为epoch数
        total_epochs = max(1, total_iters // len(dataset))
        complement_start_epoch = None
        if complement_start_iter is not None:
            complement_start_epoch = complement_start_iter // len(dataset)
        
        super().__init__(
            dataset=dataset,
            num_classes=num_classes,
            alpha_0=alpha_0,
            total_epochs=total_epochs,
            samples_per_gpu=samples_per_gpu,
            seed=seed,
            shuffle=shuffle,
            complement_weight=complement_weight,
            high_precision_threshold=high_precision_threshold,
            complement_start_epoch=complement_start_epoch,
            target_class=target_class,
            reference_classes=reference_classes,
        )
    
    def _get_alpha_t(self) -> float:
        """基于迭代数计算α_t"""
        progress = min(self.current_iter / self.total_iters, 1.0)
        alpha_t = self.alpha_0 * (1.0 - progress)
        return max(alpha_t, 1e-6)
    
    def _should_use_complement(self) -> bool:
        """判断是否应该使用互补检测（基于迭代数）
        
        放宽的条件（满足任一即可）：
        1. 当前iter >= complement_start_iter 且
        2. 参考类别的平均precision >= high_precision_threshold (0.65)
           或 至少有一个参考类别的precision >= 0.75
        """
        if self.current_iter < self.complement_start_iter:
            return False
        
        # 计算平均precision
        complement_confidence = self._compute_complement_confidence()
        
        # 检查单个类别的precision
        max_precision = max([self.class_metrics['precision'][c] for c in self.reference_classes])
        
        # 放宽条件：平均达标 或 单个类别表现优秀
        return (complement_confidence >= self.high_precision_threshold or 
                max_precision >= 0.75)
    
    def set_iter(self, iter_num: int) -> None:
        """设置当前迭代数"""
        self.current_iter = iter_num
        self.epoch = iter_num // len(self.dataset)
