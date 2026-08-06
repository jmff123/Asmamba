# Copyright (c) OpenMMLab. All rights reserved.
"""
互补检测DCBS Hook

功能：
1. 定期在验证集上评估各类别的precision和recall
2. 将指标更新到DCBSComplementSampler中
3. 记录互补检测的效果
"""

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from typing import Optional


@HOOKS.register_module()
class DCBSComplementHook(Hook):
    """互补检测DCBS Hook
    
    定期评估类别性能并更新采样器的互补检测参数。
    
    Args:
        update_interval: 更新间隔（迭代数），默认1000
        num_classes: 类别数量，默认3
        log_interval: 日志打印间隔，默认1000
    """
    
    # 设置Hook优先级，确保在验证后执行
    priority = 'VERY_LOW'
    
    def __init__(
        self,
        update_interval: int = 1000,
        num_classes: int = 3,
        log_interval: int = 1000,
    ):
        self.update_interval = update_interval
        self.num_classes = num_classes
        self.log_interval = log_interval
        
        # 存储历史指标
        self.metrics_history = []
    
    def _compute_class_metrics(self, runner) -> dict:
        """从验证结果中计算各类别的precision和recall
        
        Args:
            runner: 训练器对象
        
        Returns:
            metrics: 包含precision和recall的字典
        """
        # 初始化
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        
        # 方法1: 从val_evaluator的results获取
        if hasattr(runner, 'val_evaluator'):
            evaluator = runner.val_evaluator
            
            # 尝试从evaluator的内部状态获取混淆矩阵
            if hasattr(evaluator, 'results') and evaluator.results:
                try:
                    # IoUMetric会存储混淆矩阵或类别指标
                    results = evaluator.results
                    
                    # 尝试获取每个类别的IoU（可以推算precision和recall）
                    for k in range(self.num_classes):
                        iou_key = f'IoU.{k}'
                        if iou_key in results:
                            # 有IoU，但我们需要precision和recall
                            # 暂时使用IoU作为近似
                            iou = results[iou_key]
                            # 粗略估计：假设precision ≈ recall ≈ sqrt(IoU)
                            precision[k] = np.sqrt(iou) if iou > 0 else 0.5
                            recall[k] = np.sqrt(iou) if iou > 0 else 0.5
                    
                    if np.any(precision > 0):
                        runner.logger.info("[互补检测] 从IoU推算precision/recall")
                        return {'precision': precision, 'recall': recall}
                except Exception as e:
                    runner.logger.debug(f"[互补检测] 从results获取失败: {e}")
            
            # 尝试从混淆矩阵计算
            if hasattr(evaluator, 'confusion_matrix'):
                try:
                    cm = evaluator.confusion_matrix
                    if cm is not None and cm.size > 0:
                        # 从混淆矩阵计算precision和recall
                        for k in range(min(self.num_classes, cm.shape[0])):
                            tp = cm[k, k]
                            fp = cm[:, k].sum() - tp
                            fn = cm[k, :].sum() - tp
                            
                            precision[k] = tp / (tp + fp + 1e-10)
                            recall[k] = tp / (tp + fn + 1e-10)
                        
                        runner.logger.info("[互补检测] 成功从混淆矩阵计算指标")
                        return {'precision': precision, 'recall': recall}
                except Exception as e:
                    runner.logger.debug(f"[互补检测] 从混淆矩阵计算失败: {e}")
        
        # 方法2: 从message_hub获取
        if hasattr(runner, 'message_hub'):
            message_hub = runner.message_hub
            
            # 尝试获取各类别的指标
            for k in range(self.num_classes):
                # 尝试多种可能的指标名称格式
                precision_keys = [
                    f'val/Precision_class{k}',
                    f'val/Class_{k}_Precision',
                    f'Precision_class{k}',
                    f'Class_{k}_Precision',
                ]
                recall_keys = [
                    f'val/Recall_class{k}',
                    f'val/Class_{k}_Recall',
                    f'Recall_class{k}',
                    f'Class_{k}_Recall',
                ]
                
                # 查找precision
                for key in precision_keys:
                    try:
                        val = message_hub.get_scalar(key)
                        if val is not None:
                            precision[k] = val.current()
                            break
                    except:
                        continue
                
                # 查找recall
                for key in recall_keys:
                    try:
                        val = message_hub.get_scalar(key)
                        if val is not None:
                            recall[k] = val.current()
                            break
                    except:
                        continue
            
            # 检查是否成功获取
            if np.any(precision > 0) or np.any(recall > 0):
                runner.logger.info("[互补检测] 成功从message_hub获取验证指标")
                return {'precision': precision, 'recall': recall}
        
        # 如果无法获取，返回None（调用方会使用默认值）
        runner.logger.warning("[互补检测] 无法从验证结果获取类别指标")
        return None
    
    def _estimate_metrics_from_batch(self, runner, data_batch) -> dict:
        """从当前batch估计类别指标（快速近似）
        
        当无法获取完整验证指标时，使用训练batch进行快速估计。
        
        Args:
            runner: 训练器对象
            data_batch: 当前训练batch
        
        Returns:
            metrics: 估计的precision和recall
        """
        model = runner.model
        was_training = model.training
        model.eval()
        
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        
        try:
            with torch.no_grad():
                # 获取输入数据
                if isinstance(data_batch, dict):
                    inputs = data_batch.get('inputs', data_batch)
                else:
                    inputs = data_batch
                
                # 前向传播获取预测
                if hasattr(model, 'module'):
                    # 分布式训练
                    outputs = model.module.predict(inputs, data_batch)
                else:
                    outputs = model.predict(inputs, data_batch)
                
                # 计算每个类别的precision和recall
                for k in range(self.num_classes):
                    tp = fp = fn = 0
                    
                    for i, output in enumerate(outputs):
                        # 获取预测和真值
                        if hasattr(output, 'pred_sem_seg'):
                            pred = output.pred_sem_seg.data.cpu().numpy()
                        else:
                            pred = output['pred_sem_seg'].cpu().numpy()
                        
                        if hasattr(output, 'gt_sem_seg'):
                            gt = output.gt_sem_seg.data.cpu().numpy()
                        else:
                            # 从data_batch获取gt
                            if isinstance(data_batch, dict) and 'data_samples' in data_batch:
                                gt = data_batch['data_samples'][i].gt_sem_seg.data.cpu().numpy()
                            else:
                                continue
                        
                        # 计算TP, FP, FN
                        tp += np.sum((pred == k) & (gt == k))
                        fp += np.sum((pred == k) & (gt != k))
                        fn += np.sum((pred != k) & (gt == k))
                    
                    # 计算precision和recall
                    precision[k] = tp / (tp + fp + 1e-10)
                    recall[k] = tp / (tp + fn + 1e-10)
        
        except Exception as e:
            runner.logger.warning(f"[互补检测] batch估计失败: {e}")
            # 返回保守的默认值
            precision = np.ones(self.num_classes) * 0.6
            recall = np.ones(self.num_classes) * 0.6
        
        finally:
            if was_training:
                model.train()
        
        return {'precision': precision, 'recall': recall}
    
    def before_train_iter(self, runner, batch_idx: int, data_batch=None):
        """在训练迭代前更新采样器的迭代数"""
        current_iter = runner.iter
        
        # 只更新采样器的迭代数
        # 注意：类别指标的更新已经在after_val_iter中完成，这里不再重复
        dataloader = runner.train_dataloader
        if hasattr(dataloader, 'sampler'):
            sampler = dataloader.sampler
            if hasattr(sampler, 'set_iter'):
                sampler.set_iter(current_iter)
    
    def after_val_epoch(self, runner, metrics=None):
        """验证epoch结束后调用（所有验证batch完成后）
        
        此时evaluator已经完成了所有batch的汇总，可以获取准确的指标。
        """
        runner.logger.info(f"[互补检测] 验证完成，准备更新类别指标 (Iter {runner.iter})")
        
        # 尝试从验证结果中获取真实指标
        class_metrics = self._compute_class_metrics(runner)
        
        # 如果无法获取，使用基于mIoU的估计
        if class_metrics is None or np.all(class_metrics['precision'] == 0.5):
            runner.logger.warning("[互补检测] 无法从验证结果获取指标，使用mIoU估计")
            
            try:
                if hasattr(runner, 'message_hub'):
                    miou = runner.message_hub.get_scalar('val/mIoU')
                    if miou is not None:
                        miou_val = miou.current()
                        estimated_pr = np.sqrt(miou_val / 100.0)
                        
                        class_metrics = {
                            'precision': np.array([
                                estimated_pr * 1.2,  # 类别0
                                estimated_pr * 0.8,  # 类别1
                                estimated_pr * 1.0,  # 类别2
                            ]),
                            'recall': np.array([
                                estimated_pr * 1.0,  # 类别0
                                estimated_pr * 0.9,  # 类别1
                                estimated_pr * 1.3,  # 类别2
                            ]),
                        }
                        runner.logger.info(f"[互补检测] 基于mIoU={miou_val:.2f}%估计类别指标")
                    else:
                        raise ValueError("无法获取mIoU")
            except Exception as e:
                # 最后的fallback
                class_metrics = {
                    'precision': np.array([0.75, 0.50, 0.60]),
                    'recall': np.array([0.60, 0.55, 0.80]),
                }
                runner.logger.warning(f"[互补检测] 使用默认值: {e}")
        
        # 更新采样器
        self._update_sampler(runner, class_metrics)
    
    def _update_sampler(self, runner, class_metrics: dict):
        """更新采样器的类别指标"""
        dataloader = runner.train_dataloader
        if hasattr(dataloader, 'sampler'):
            sampler = dataloader.sampler
            if hasattr(sampler, 'update_class_metrics'):
                sampler.update_class_metrics(class_metrics)
                
                runner.logger.info(
                    f"\n[互补检测] 验证后更新类别指标 (Iter {runner.iter}):"
                )
                for k in range(self.num_classes):
                    runner.logger.info(
                        f"  类别 {k}: "
                        f"Precision={class_metrics['precision'][k]:.4f}, "
                        f"Recall={class_metrics['recall'][k]:.4f}"
                    )
