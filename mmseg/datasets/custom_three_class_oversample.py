# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
from PIL import Image
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CustomThreeClassOversampleDataset(BaseSegDataset):
    """Custom Three Class Dataset with oversampling for minority class.

    对包含类别1（红色区域）的样本进行过采样，解决类别不平衡问题。
    """
    METAINFO = dict(
        classes=('background', 'red_region', 'green_region'),
        palette=[[0, 0, 0], [200, 5, 5], [5, 128, 10]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 oversample_ratio=3,  # 类别1样本重复次数
                 **kwargs) -> None:
        self.oversample_ratio = oversample_ratio
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self):
        """加载数据列表，对包含类别1的样本进行过采样"""
        # 先调用父类方法获取原始数据列表
        data_list = super().load_data_list()

        # 找出包含类别1的样本
        samples_with_class1 = []
        samples_without_class1 = []

        for data_info in data_list:
            seg_map_path = data_info.get('seg_map_path', None)
            if seg_map_path and os.path.exists(seg_map_path):
                try:
                    mask = np.array(Image.open(seg_map_path))
                    if 1 in np.unique(mask):
                        samples_with_class1.append(data_info)
                    else:
                        samples_without_class1.append(data_info)
                except:
                    samples_without_class1.append(data_info)
            else:
                samples_without_class1.append(data_info)

        # 对包含类别1的样本进行过采样
        oversampled_list = samples_without_class1.copy()
        for _ in range(self.oversample_ratio):
            oversampled_list.extend(samples_with_class1)

        print(f"[CustomThreeClassOversampleDataset] 原始样本: {len(data_list)}")
        print(f"  - 包含类别1: {len(samples_with_class1)}")
        print(f"  - 不包含类别1: {len(samples_without_class1)}")
        print(f"  - 过采样后总数: {len(oversampled_list)}")

        return oversampled_list