# Copyright (c) OpenMMLab. All rights reserved.
"""
动态类别平衡采样器 (Dynamic Class-Balanced Sampling, DCBS)

基于论文实现，通过动态调整类别采样概率来解决类别不平衡问题。

核心原理:
1. 统计训练集中各类别的像素频率
2. 使用softmax计算类别采样概率（少样本类别获得更高概率）
3. 动态调整系数α_t随训练进程变化，避免过度偏向

公式:
- 类别频率: f_k = Σ[y_ij == k] / (N_S × H × W)
- 采样概率: P_k = exp((1-f_k)/α_t) / Σexp((1-f_k')/α_t)
- 动态系数: α_t = α_0 × (1 - t/T)
"""

import math
import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional, List
import mmcv
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DCBSSampler(Sampler):
    """动态类别平衡采样器
    
    Args:
        dataset: 数据集对象
        num_classes: 类别数量
        alpha_0: 初始调整系数，默认0.01（论文推荐值）
        total_epochs: 总训练epoch数
        samples_per_gpu: 每个GPU的batch size
        seed: 随机种子
        shuffle: 是否打乱
    """
    
    def __init__(
        self,
        dataset,
        num_classes: int = 3,
        alpha_0: float = 0.01,
        total_epochs: int = 50,
        samples_per_gpu: int = 1,
        seed: Optional[int] = None,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.num_classes = num_classes
        self.alpha_0 = alpha_0
        self.total_epochs = total_epochs
        self.samples_per_gpu = samples_per_gpu
        self.shuffle = shuffle
        
        # 分布式训练相关
        self.rank, self.world_size = get_dist_info()
        self.seed = sync_random_seed() if seed is None else seed
        
        # 当前epoch
        self.epoch = 0
        
        # 计算类别频率
        self.class_frequencies = self._compute_class_frequencies()
        
        # 为每个样本计算采样权重
        self.sample_weights = self._compute_sample_weights()
        
        print(f"\n{'='*70}")
        print("DCBS采样器初始化")
        print(f"{'='*70}")
        print(f"类别数量: {self.num_classes}")
        print(f"初始α_0: {self.alpha_0}")
        print(f"总epoch数: {self.total_epochs}")
        print(f"\n类别频率统计:")
        for k in range(self.num_classes):
            print(f"  类别 {k}: {self.class_frequencies[k]:.4f}")
        print(f"{'='*70}\n")
    
    def _compute_class_frequencies(self) -> np.ndarray:
        """计算各类别的像素频率
        
        公式: f_k = Σ[y_ij == k] / (N_S × H × W)
        
        Returns:
            class_frequencies: shape (num_classes,)
        """
        print("正在统计类别频率...")
        
        class_pixel_counts = np.zeros(self.num_classes, dtype=np.int64)
        total_pixels = 0
        
        # 遍历数据集统计
        for idx in range(len(self.dataset)):
            # 获取标注
            data_info = self.dataset.get_data_info(idx)
            seg_map_path = data_info.get('seg_map_path', None)
            
            if seg_map_path is None:
                continue
            
            # 读取标注图像
            try:
                seg_map = mmcv.imread(seg_map_path, flag='unchanged')
                
                # 统计各类别像素数
                for k in range(self.num_classes):
                    class_pixel_counts[k] += np.sum(seg_map == k)
                
                total_pixels += seg_map.size
                
                # 每100张图像打印一次进度
                if (idx + 1) % 100 == 0:
                    print(f"  已处理 {idx + 1}/{len(self.dataset)} 张图像")
            
            except Exception as e:
                print(f"  警告: 无法读取 {seg_map_path}: {e}")
                continue
        
        # 计算频率
        class_frequencies = class_pixel_counts / total_pixels
        
        # 避免频率为0（添加平滑）
        class_frequencies = np.maximum(class_frequencies, 1e-6)
        
        print(f"✓ 类别频率统计完成")
        
        return class_frequencies
    
    def _compute_sample_weights(self) -> np.ndarray:
        """为每个样本计算采样权重
        
        基于样本中包含的类别来计算权重，包含更多少样本类别的图像获得更高权重
        
        Returns:
            sample_weights: shape (num_samples,)
        """
        print("正在计算样本权重...")
        
        sample_weights = np.zeros(len(self.dataset), dtype=np.float32)
        
        for idx in range(len(self.dataset)):
            data_info = self.dataset.get_data_info(idx)
            seg_map_path = data_info.get('seg_map_path', None)
            
            if seg_map_path is None:
                sample_weights[idx] = 1.0
                continue
            
            try:
                seg_map = mmcv.imread(seg_map_path, flag='unchanged')
                
                # 统计该样本包含的类别
                unique_classes = np.unique(seg_map)
                
                # 计算该样本的权重（包含的类别频率越低，权重越高）
                weight = 0.0
                for k in unique_classes:
                    if k < self.num_classes:
                        # 使用 1 - f_k 作为权重基础
                        weight += (1.0 - self.class_frequencies[k])
                
                sample_weights[idx] = weight / len(unique_classes)
            
            except Exception as e:
                sample_weights[idx] = 1.0
                continue
        
        # 归一化权重
        sample_weights = sample_weights / sample_weights.sum() * len(self.dataset)
        
        print(f"✓ 样本权重计算完成")
        print(f"  权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
        
        return sample_weights
    
    def _compute_class_sampling_probs(self, alpha_t: float) -> np.ndarray:
        """计算当前epoch的类别采样概率
        
        修正公式以实现"后期趋向频率分布"：
        当α_t较大时：使用 exp((1-f_k)/α_t) 偏向少样本
        当α_t较小时：逐渐过渡到频率分布 f_k
        
        混合公式：P_k = (1-w) × f_k + w × softmax((1-f_k)/α_t)
        其中 w = α_t / α_0（权重随α_t衰减）
        
        Args:
            alpha_t: 当前epoch的动态调整系数
        
        Returns:
            class_probs: shape (num_classes,)
        """
        # 计算基于DCBS的采样概率
        numerators = np.exp((1.0 - self.class_frequencies) / alpha_t)
        dcbs_probs = numerators / numerators.sum()
        
        # 计算混合权重（α_t越小，越接近频率分布）
        weight = alpha_t / self.alpha_0
        
        # 混合：前期主要用DCBS，后期主要用频率分布
        class_probs = weight * dcbs_probs + (1.0 - weight) * self.class_frequencies
        
        # 归一化
        class_probs = class_probs / class_probs.sum()
        
        return class_probs
    
    def _get_alpha_t(self) -> float:
        """计算当前epoch的动态调整系数
        
        论文公式: α_t = α_0 × (1 - t/T)
        
        训练前期（t小）：α_t ≈ α_0，采样概率分布尖锐，少样本类别采样概率极高
        训练后期（t接近T）：α_t → 0，采样概率接近类别频率分布
        
        注意：严格按照论文公式实现
        
        Returns:
            alpha_t: 当前的调整系数
        """
        progress = self.epoch / self.total_epochs
        alpha_t = self.alpha_0 * (1.0 - progress)
        
        # 确保α_t不会完全为0（避免除零错误）
        alpha_t = max(alpha_t, 1e-6)
        
        return alpha_t
    
    def __iter__(self) -> Iterator[int]:
        """生成采样索引"""
        # 计算当前epoch的α_t
        alpha_t = self._get_alpha_t()
        
        # 计算类别采样概率
        class_probs = self._compute_class_sampling_probs(alpha_t)
        
        # 打印当前的采样信息（使用current_iter而不是epoch）
        if self.rank == 0:
            progress = self.current_iter / self.total_iters * 100
            print(f"\nIter {self.current_iter}/{self.total_iters} ({progress:.1f}%): α_t = {alpha_t:.6f}")
            print("类别采样概率:")
            for k in range(self.num_classes):
                print(f"  类别 {k}: P={class_probs[k]:.4f} (频率={self.class_frequencies[k]:.4f})")
        
        # 设置随机种子
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 根据样本权重进行采样
        if self.shuffle:
            # 使用样本权重进行加权采样
            # 确保 NumPy 数组正确转换为 Tensor
            import numpy as np
            weights = np.asarray(self.sample_weights, dtype=np.float32)
            indices = torch.multinomial(
                torch.from_numpy(weights),
                num_samples=len(self.dataset),
                replacement=True,
                generator=g
            ).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # 分布式训练：每个rank只处理部分数据
        indices = indices[self.rank::self.world_size]
        
        return iter(indices)
    
    def __len__(self) -> int:
        """返回采样器长度"""
        return len(self.dataset) // self.world_size
    
    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch
        
        Args:
            epoch: 当前epoch编号
        """
        self.epoch = epoch


@DATA_SAMPLERS.register_module()
class IterBasedDCBSSampler(DCBSSampler):
    """基于迭代的DCBS采样器
    
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
        shuffle: bool = True
    ):
        # 将迭代数转换为epoch数（假设每个epoch遍历一次数据集）
        total_epochs = max(1, total_iters // len(dataset))
        
        super().__init__(
            dataset=dataset,
            num_classes=num_classes,
            alpha_0=alpha_0,
            total_epochs=total_epochs,
            samples_per_gpu=samples_per_gpu,
            seed=seed,
            shuffle=shuffle
        )
        
        self.total_iters = total_iters
        self.current_iter = 0
    
    def _get_alpha_t(self) -> float:
        """基于迭代数计算α_t
        
        论文公式: α_t = α_0 × (1 - iter/total_iters)
        
        训练前期（iter小）：α_t ≈ α_0，采样概率分布尖锐，少样本类别采样概率极高
        训练后期（iter接近total_iters）：α_t → 0，采样概率接近类别频率分布
        
        注意：严格按照论文公式实现
        """
        progress = min(self.current_iter / self.total_iters, 1.0)
        alpha_t = self.alpha_0 * (1.0 - progress)
        
        # 确保α_t不会完全为0（避免除零错误）
        alpha_t = max(alpha_t, 1e-6)
        
        return alpha_t
    
    def set_iter(self, iter_num: int) -> None:
        """设置当前迭代数
        
        Args:
            iter_num: 当前迭代编号
        """
        self.current_iter = iter_num
        # 同时更新epoch（用于打印）
        self.epoch = iter_num // len(self.dataset)
