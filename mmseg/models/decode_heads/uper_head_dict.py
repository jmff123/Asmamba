# Copyright (c) OpenMMLab. All rights reserved.
"""
UPerHead with Dynamic Dictionary Enhancement

在标准UPerHead基础上，在FPN特征融合后插入动态字典模块，
利用字典查询机制增强类别表示能力。
"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from ..utils.dynamic_dictionary import DynamicDictionary, compute_contrastive_loss
from ..utils.adaptive_loss_weights import AdaptiveLossWeightBalancer
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@MODELS.register_module()
class UPerHeadDict(BaseDecodeHead):
    """UPerNet with Dynamic Dictionary Enhancement
    
    在标准UPerHead的FPN特征融合后，插入动态字典模块：
    1. FPN特征 → 动态字典生成类别查询
    2. 类别查询与特征交互 → 增强的类别表示
    3. 最终分割输出
    
    Args:
        pool_scales: PSP模块的池化尺度
        use_dynamic_dict: 是否使用动态字典（默认True）
        dict_embed_dim: 字典嵌入维度（默认256）
        dict_query_ratio: 每个类别的候选查询数量（默认4）
        dict_num_heads: 字典交互的注意力头数（默认8）
        use_contrastive_loss: 是否使用对比损失（默认True）
        contrastive_weight: 对比损失权重（默认0.1）
    """
    
    def __init__(
        self,
        pool_scales=(1, 2, 3, 6),
        use_dynamic_dict=True,
        dict_embed_dim=256,
        dict_query_ratio=4,
        dict_num_heads=8,
        use_contrastive_loss=True,
        use_adaptive_weights=True,  # 是否使用自适应权重
        weight_strategy='hybrid',  # 权重策略：'hybrid', 'curriculum', 'magnitude', 'uncertainty'
        total_iters=80000,  # 总迭代次数（用于课程学习）
        **kwargs
    ):
        super().__init__(input_transform='multiple_select', **kwargs)
        
        self.use_dynamic_dict = use_dynamic_dict
        self.use_contrastive_loss = use_contrastive_loss
        self.use_adaptive_weights = use_adaptive_weights
        self._contrastive_loss = None  # 用于存储对比损失
        
        # 自适应权重平衡器
        if self.use_adaptive_weights:
            # 定义损失名称和基础权重
            loss_names = ['loss_dice', 'loss_focal', 'loss_lovasz', 'loss_tversky']
            if use_contrastive_loss:
                loss_names.append('loss_contrastive')
            
            # 基础权重（相对重要性，会被自适应调整）
            # 改进：缩小权重差距，从40倍降到6倍，使各损失更平衡
            base_weights = {
                'loss_dice': 3.0,      # 最重要：直接优化IoU
                'loss_focal': 2.0,     # 重要：困难样本
                'loss_lovasz': 2.0,    # 重要：IoU理论保证
                'loss_tversky': 1.5,   # 中等：FP/FN权衡
                'loss_contrastive': 0.5,  # 辅助但重要：特征表示（从0.1提升到0.5）
            }
            
            self.loss_balancer = AdaptiveLossWeightBalancer(
                loss_names=loss_names,
                strategy=weight_strategy,
                base_weights=base_weights,
                total_iters=total_iters,
                warmup_iters=4500,
                alpha=0.9,
            )
        else:
            self.loss_balancer = None
        
        # ============ 标准UPerNet组件 ============
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners
        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        # ============ 动态字典增强 ============
        if self.use_dynamic_dict:
            # 特征投影到字典维度
            self.dict_proj = ConvModule(
                self.channels,
                dict_embed_dim,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            
            # 动态字典模块
            self.dynamic_dict = DynamicDictionary(
                num_classes=self.num_classes,
                embed_dim=dict_embed_dim,
                query_ratio=dict_query_ratio,
                feature_dim=self.channels  # 使用FPN融合后的特征维度
            )
            
            # 字典-特征交互层
            self.dict_interaction = nn.ModuleList([
                _DictInteractionBlock(dict_embed_dim, dict_num_heads)
                for _ in range(2)  # 2层交互
            ])
            
            # 字典输出投影
            self.dict_output = nn.Sequential(
                nn.LayerNorm(dict_embed_dim),
                nn.Linear(dict_embed_dim, dict_embed_dim // 4),
                nn.GELU(),
            )
            
            # 融合字典增强和原始特征
            self.fusion = ConvModule(
                self.channels + dict_embed_dim // 4,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
    
    def psp_forward(self, inputs):
        """PSP模块前向传播"""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output
    
    def _forward_feature(self, inputs):
        """特征提取（标准UPerNet FPN）"""
        inputs = self._transform_inputs(inputs)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners
            )
        
        # Build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])
        
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        
        return feats
    
    def _dict_enhance(self, feats):
        """动态字典增强
        
        Args:
            feats: FPN融合特征 [B, C, H, W]
        
        Returns:
            enhanced_feats: 增强后的特征 [B, C, H, W]
            contrastive_loss: 对比损失（训练时）
        """
        b, c, h, w = feats.shape
        
        # 1. 生成动态字典查询
        dynamic_queries, _ = self.dynamic_dict(
            feats,
            return_static=False
        )  # [B, num_classes, embed_dim]
        
        # 2. 投影特征到字典维度
        dict_feats = self.dict_proj(feats)  # [B, dict_embed_dim, H, W]
        keys = dict_feats.flatten(2).permute(0, 2, 1)  # [B, H*W, dict_embed_dim]
        
        # 3. 字典-特征交互
        queries = dynamic_queries
        for layer in self.dict_interaction:
            queries, keys = layer(queries, keys)
        
        # 4. 字典输出投影
        dict_out = self.dict_output(queries)  # [B, num_classes, dict_embed_dim//4]
        
        # 5. 将字典表示广播到空间维度
        dict_spatial = dict_out.unsqueeze(-1).unsqueeze(-1)  # [B, num_classes, dim, 1, 1]
        dict_spatial = dict_spatial.expand(-1, -1, -1, h, w)  # [B, num_classes, dim, H, W]
        
        # 6. 聚合所有类别的字典表示
        dict_spatial = dict_spatial.sum(dim=1)  # [B, dict_embed_dim//4, H, W]
        
        # 7. 融合原始特征和字典增强（带残差连接）
        fused = torch.cat([feats, dict_spatial], dim=1)
        dict_enhanced = self.fusion(fused)
        
        # 残差连接：保留原始特征信息
        enhanced_feats = feats + dict_enhanced
        
        # 8. 计算对比损失（训练时）
        contrastive_loss = None
        if self.training and self.use_contrastive_loss:
            contrastive_loss = compute_contrastive_loss(dynamic_queries)
        
        return enhanced_feats, contrastive_loss
    
    def forward(self, inputs):
        """前向传播
        
        Args:
            inputs: 多尺度backbone特征
        
        Returns:
            output: 分割输出 [B, num_classes, H, W]
        """
        # 1. 标准UPerNet特征提取
        feats = self._forward_feature(inputs)
        
        # 2. 动态字典增强（可选）
        if self.use_dynamic_dict:
            feats, contrastive_loss = self._dict_enhance(feats)
            # 保存对比损失，在loss_by_feat中使用
            if self.training and contrastive_loss is not None:
                self._contrastive_loss = contrastive_loss
        
        # 3. 分类输出
        output = self.cls_seg(feats)
        
        return output
    
    def loss_by_feat(self, seg_logits, batch_data_samples):
        """计算损失（重写以添加对比损失和自适应权重）
        
        Args:
            seg_logits: 分割logits
            batch_data_samples: 数据样本
        
        Returns:
            losses: 损失字典（只包含tensor）
        """
        # 调用父类方法计算标准分割损失
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        
        # 添加对比损失
        if self.training and self.use_contrastive_loss and self._contrastive_loss is not None:
            losses['loss_contrastive'] = self._contrastive_loss
            # 清空缓存
            self._contrastive_loss = None
        
        # 应用自适应权重
        if self.training and self.use_adaptive_weights and self.loss_balancer is not None:
            # 获取当前迭代次数（从batch_data_samples中提取）
            current_iter = None
            if hasattr(batch_data_samples[0], 'metainfo'):
                current_iter = batch_data_samples[0].metainfo.get('iter', None)
            
            # 计算自适应权重
            adaptive_weights = self.loss_balancer(losses, current_iter)
            
            # 直接应用自适应权重（配置文件中的loss_weight已设为1.0）
            weighted_losses = {}
            for loss_name, loss_value in losses.items():
                if loss_name in adaptive_weights:
                    weight = adaptive_weights[loss_name]
                    weighted_losses[loss_name] = loss_value * weight
                else:
                    weighted_losses[loss_name] = loss_value
            
            # 如果使用不确定性策略，添加正则化损失
            if self.loss_balancer.strategy == 'uncertainty':
                uncertainty_reg = self.loss_balancer.get_uncertainty_loss()
                weighted_losses['loss_uncertainty_reg'] = uncertainty_reg
            
            return weighted_losses
        
        return losses


class _DictInteractionBlock(nn.Module):
    """字典-特征交互块"""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, queries, keys):
        """
        Args:
            queries: 字典查询 [B, num_classes, embed_dim]
            keys: 图像特征 [B, H*W, embed_dim]
        
        Returns:
            queries: 更新后的查询
            keys: 更新后的特征
        """
        # 交叉注意力：字典查询图像特征
        q_out, _ = self.cross_attn(queries, keys, keys)
        queries = self.norm1(queries + q_out)
        
        # 自注意力：字典内部交互
        q_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + q_out)
        
        # FFN
        q_out = self.ffn(queries)
        queries = self.norm3(queries + q_out)
        
        return queries, keys
