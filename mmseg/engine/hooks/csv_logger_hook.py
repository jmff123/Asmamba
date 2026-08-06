# Copyright (c) OpenMMLab. All rights reserved.
import os
import csv
from typing import Optional, Sequence, Union
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class CSVLoggerHook(Hook):
    """将训练指标记录到CSV文件的Hook

    Args:
        log_dir: CSV文件保存目录，默认为work_dir
        filename: CSV文件名，默认为'training_metrics.csv'
        interval: 记录间隔（迭代次数），默认100
    """

    def __init__(self,
                 log_dir: Optional[str] = None,
                 filename: str = 'training_metrics.csv',
                 interval: int = 100):
        self.log_dir = log_dir
        self.filename = filename
        self.interval = interval
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        self.header_written = False

        # 存储验证指标
        self.val_metrics = {}

    def before_run(self, runner):
        """训练开始前初始化CSV文件"""
        if self.log_dir is None:
            self.log_dir = runner.work_dir

        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, self.filename)

        # 打开CSV文件
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 预定义表头顺序 - 包含每个类别的详细指标
        self.header_order = [
            'iter', 'loss', 
            'decode.loss_dice', 'decode.loss_lovasz', 'decode.acc_seg',
            'aux.loss_dice', 'aux.loss_lovasz', 'aux.acc_seg',
            'lr', 'data_time', 'time',
            # 总体指标
            'val_aAcc', 'val_mIoU', 'val_mDice', 'val_mFscore',
            # 各类别IoU
            'val_IoU_background', 'val_IoU_red_region', 'val_IoU_green_region',
            # 各类别Dice
            'val_Dice_background', 'val_Dice_red_region', 'val_Dice_green_region',
            # 各类别准确率
            'val_Acc_background', 'val_Acc_red_region', 'val_Acc_green_region',
            # 各类别Fscore
            'val_Fscore_background', 'val_Fscore_red_region', 'val_Fscore_green_region',
            # 各类别精确率
            'val_Precision_background', 'val_Precision_red_region', 'val_Precision_green_region',
            # 各类别召回率
            'val_Recall_background', 'val_Recall_red_region', 'val_Recall_green_region'
        ]
        
        # 类别名称映射
        self.class_names = ['background', 'red_region', 'green_region']

        runner.logger.info(f'CSV Logger: 指标将保存到 {self.csv_path}')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None):
        """每次训练迭代后记录指标"""
        cur_iter = runner.iter + 1

        if cur_iter % self.interval != 0:
            return

        # 构建记录行
        row_data = {'iter': cur_iter}

        # 从message_hub获取日志指标
        try:
            # 获取所有标量日志
            log_dict = runner.message_hub.get_scalar('train')
            
            # 提取所有指标
            for key, value in log_dict.items():
                try:
                    # 尝试获取当前值
                    if hasattr(value, 'current'):
                        val = value.current()
                    elif hasattr(value, 'value'):
                        val = value.value
                    else:
                        val = float(value)
                    
                    row_data[key] = f'{val:.6f}'
                except (TypeError, ValueError, AttributeError):
                    # 如果无法转换为数值，跳过
                    continue
                    
        except Exception as e:
            runner.logger.warning(f'CSV Logger: 无法获取训练指标: {e}')

        # 添加验证指标（如果有）
        row_data.update(self.val_metrics)

        # 写入CSV
        if len(row_data) > 1:  # 确保除了iter外还有其他数据
            self._write_row(row_data)
        else:
            runner.logger.warning(f'CSV Logger: iter {cur_iter} 没有可记录的指标')

    def after_val_epoch(self, runner, metrics=None):
        """验证epoch结束后记录验证指标"""
        if metrics is None:
            return

        # 存储验证指标供下次训练迭代使用
        self.val_metrics = {}

        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.val_metrics[f'val_{key}'] = f'{value:.4f}'
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            self.val_metrics[f'val_{key}_{sub_key}'] = f'{sub_value:.4f}'
                elif isinstance(value, (list, tuple)):
                    # 处理per_class指标数组
                    if len(value) == len(self.class_names):
                        for idx, class_name in enumerate(self.class_names):
                            try:
                                self.val_metrics[f'val_{key}_{class_name}'] = f'{float(value[idx]):.4f}'
                            except (TypeError, ValueError, IndexError):
                                continue
                    else:
                        # 如果长度不匹配，尝试用索引记录
                        for idx, val in enumerate(value):
                            try:
                                self.val_metrics[f'val_{key}_{idx}'] = f'{float(val):.4f}'
                            except (TypeError, ValueError):
                                continue
            
            runner.logger.info(f'CSV Logger: 记录了 {len(self.val_metrics)} 个验证指标')
        except Exception as e:
            runner.logger.warning(f'CSV Logger: 处理验证指标时出错: {e}')

    def _write_row(self, row_data: dict):
        """写入一行数据到CSV"""
        if not self.header_written:
            # 写入表头 - 使用预定义顺序 + 动态发现的新列
            headers = []
            
            # 首先添加预定义顺序中存在的列
            for h in self.header_order:
                if h in row_data:
                    headers.append(h)
            
            # 然后添加预定义顺序中没有的新列
            for key in sorted(row_data.keys()):
                if key not in self.header_order and key not in headers:
                    headers.append(key)
            
            if not headers:
                return  # 没有数据，不写入
            
            self.csv_writer.writerow(headers)
            self.header_written = True
            self.current_headers = headers
            self.csv_file.flush()

        # 按照表头顺序写入数据
        row_values = [row_data.get(h, '') for h in self.current_headers]
        self.csv_writer.writerow(row_values)
        self.csv_file.flush()  # 立即写入磁盘

    def after_run(self, runner):
        """训练结束后关闭文件"""
        if self.csv_file:
            self.csv_file.close()
            runner.logger.info(f'CSV Logger: 训练指标已保存到 {self.csv_path}')