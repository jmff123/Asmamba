# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CustomThreeClassDataset(BaseSegDataset):
    """Custom Three Class Dataset.

    A custom dataset with three classes:
    - Class 0: Background (Black: 0,0,0)
    - Class 1: Red region (0,118,0) to (10,138,19)
    - Class 2: Green region (118,0,0) to (255,10,10)

    The ``img_suffix`` and ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background', 'red_region', 'green_region'),
        palette=[[0, 0, 0], [200, 5, 5], [5, 128, 10]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)