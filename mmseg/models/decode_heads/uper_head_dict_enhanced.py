"""
UPerHead with Enhanced Dynamic Dictionary

在标准UPerHead基础上，集成增强版动态字典模块：
1. 可学习的类别关系矩阵: 显式建模类别间的关系
2. 难样本挖掘的对比损失: 自适应关注难区分类别
"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from ..utils.dynamic_dictionary_enhanced import (
    EnhancedDynamicDictionary,
    compute_hard_contrastive_loss
)
from ..utils.adaptive_loss_weights import AdaptiveLossWeightBalancer
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@MODELS.register_module()
class UPerHeadDictEnhanced(BaseDecodeHead):
    """UPerNet with Enhanced Dynamic Dictionary
    
    创新点：
    1. 可学习的类别关系矩阵：显式建模类别间关系
    2. 难样本挖掘的对比损失：自适应关注难区分类别
    
    Args:
        pool_scales: PSP模块的池化尺度
        use_enhanced_dict: 是否使用增强版动态字典
        dict_embed_dim: 字典嵌入维度
        dict_query_ratio: 每个类别的候选查询数量
        dict_num_heads: 字典交互的注意力头数
        use_relation_matrix: 是否使用关系矩阵
        relation_init: 关系矩阵初始化类型
        use_contrastive_loss: 是否使用对比损失
        use_adaptive_weights: 是否使用自适应权重
    """
    
    def __init__(
        self,
        pool_scales=(1, 2, 3, 6),
        use_enhanced_dict=True,
        dict_embed_dim=256,
        dict_query_ratio=4,
        dict_num_heads=8,
        use_relation_matrix=True,
        relation_init='identity',
        use_contrastive_loss=True,
        use_adaptive_weights=True,
        weight_strategy='hybrid',
        total_iters=90000,
        **kwargs
    ):
        super().__init__(input_transform='multiple_select', **kwargs)
        
        self.use_enhanced_dict = use_enhanced_dict
        self.use_contrastive_loss = use_contrastive_loss
        self.use_adaptive_weights = use_adaptive_weights
        self._contrastive_loss = None
        self._confusion_matrix = None  # 用于可视化
        
        # 自适应权重平衡器
        if self.use_adaptive_weights:
            loss_names = ['loss_dice', 'loss_focal', 'loss_lovasz', 'loss_tversky']
            if use_contrastive_loss:
                loss_names.append('loss_contrastive')
            
            base_weights = {
                'loss_dice': 3.0,
                'loss_focal': 2.0,
                'loss_lovasz': 2.0,
                'loss_tversky': 1.5,
                'loss_contrastive': 0.5,
            }
            
            self.loss_balancer = AdaptiveLossWeightBalancer(
                loss_names=loss_names,
                strategy=weight_strategy,
                base_weights=base_weights,
                total_iters=total_iters,
                warmup_iters=int(total_iters * 0.05625),
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
        
        # ============ 增强版动态字典 ============
        if self.use_enhanced_dict:
            # 增强版动态字典模块
            self.enhanced_dict = EnhancedDynamicDictionary(
                num_classes=self.num_classes,
                embed_dim=dict_embed_dim,
                query_ratio=dict_query_ratio,
                feature_dim=self.channels,  # 使用FPN融合后的特征维度
                use_relation_matrix=use_relation_matrix,
                relation_init=relation_init
            )
            
            # 字典-特征交互层
            self.dict_interaction = nn.ModuleList([
                _DictInteractionBlock(dict_embed_dim, dict_num_heads)
                for _ in range(2)
            ])
            
            # 字典输出投影
            self.dict_output = nn.Sequential(
                nn.LayerNorm(dict_embed_dim),
                nn.Linear(dict_embed_dim, dict_embed_dim // 4),
                nn.GELU(),
            )
            
            # 融合字典增强和FPN特征
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
    
    def _dict_enhance(self, fpn_feats):
        """增强版动态字典增强
        
        Args:
            fpn_feats: FPN融合特征 [B, C, H, W]
        
        Returns:
            enhanced_feats: 增强后的特征 [B, C, H, W]
            contrastive_loss: 对比损失
            confusion_matrix: 混淆度矩阵
        """
        b, c, h, w = fpn_feats.shape
        
        # 1. 从FPN特征生成动态字典查询（带关系矩阵精炼）
        dynamic_queries, _ = self.enhanced_dict(
            fpn_feats,
            return_static=False
        )  # [B, num_classes, embed_dim]
        
        # 2. 投影FPN特征到字典维度
        # 使用FPN特征作为keys进行交互
        dict_embed_dim = dynamic_queries.shape[-1]
        
        # 简单投影
        fpn_proj = nn.Conv2d(
            c, dict_embed_dim, 1,
            device=fpn_feats.device
        )(fpn_feats)  # [B, dict_embed_dim, H, W]
        
        keys = fpn_proj.flatten(2).permute(0, 2, 1)  # [B, H*W, dict_embed_dim]
        
        # 3. 字典-特征交互
        queries = dynamic_queries
        for layer in self.dict_interaction:
            queries, keys = layer(queries, keys)
        
        # 4. 字典输出投影
        dict_out = self.dict_output(queries)  # [B, num_classes, dict_embed_dim//4]
        
        # 5. 将字典表示广播到空间维度
        dict_spatial = dict_out.unsqueeze(-1).unsqueeze(-1)
        dict_spatial = dict_spatial.expand(-1, -1, -1, h, w)
        dict_spatial = dict_spatial.sum(dim=1)  # [B, dict_embed_dim//4, H, W]
        
        # 6. 融合原始特征和字典增强
        fused = torch.cat([fpn_feats, dict_spatial], dim=1)
        dict_enhanced = self.fusion(fused)
        
        # 残差连接
        enhanced_feats = fpn_feats + dict_enhanced
        
        # 7. 计算对比损失和混淆度矩阵
        contrastive_loss = None
        confusion_matrix = None
        if self.training and self.use_contrastive_loss:
            contrastive_loss, confusion_matrix = compute_hard_contrastive_loss(dynamic_queries)
        
        return enhanced_feats, contrastive_loss, confusion_matrix
    
    def forward(self, inputs):
        """前向传播
        
        Args:
            inputs: 多尺度backbone特征 [C1, C2, C3, C4]
        
        Returns:
            output: 分割输出 [B, num_classes, H, W]
        """
        # 1. 标准UPerNet特征提取
        fpn_feats = self._forward_feature(inputs)
        
        # 2. 增强版动态字典增强
        if self.use_enhanced_dict:
            fpn_feats, contrastive_loss, confusion_matrix = self._dict_enhance(fpn_feats)
            
            # 保存损失和混淆度矩阵
            if self.training:
                if contrastive_loss is not None:
                    self._contrastive_loss = contrastive_loss
                if confusion_matrix is not None:
                    self._confusion_matrix = confusion_matrix
        
        # 3. 分类输出
        output = self.cls_seg(fpn_feats)
        
        return output
    
    def loss_by_feat(self, seg_logits, batch_data_samples):
        """计算损失"""
        # 调用父类方法计算标准分割损失
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        
        # 添加对比损失
        if self.training and self.use_contrastive_loss and self._contrastive_loss is not None:
            losses['loss_contrastive'] = self._contrastive_loss
            self._contrastive_loss = None
        
        # 应用自适应权重
        if self.training and self.use_adaptive_weights and self.loss_balancer is not None:
            current_iter = None
            if hasattr(batch_data_samples[0], 'metainfo'):
                current_iter = batch_data_samples[0].metainfo.get('iter', None)
            
            adaptive_weights = self.loss_balancer(losses, current_iter)
            
            weighted_losses = {}
            for loss_name, loss_value in losses.items():
                if loss_name in adaptive_weights:
                    weight = adaptive_weights[loss_name]
                    weighted_losses[loss_name] = loss_value * weight
                else:
                    weighted_losses[loss_name] = loss_value
            
            if self.loss_balancer.strategy == 'uncertainty':
                uncertainty_reg = self.loss_balancer.get_uncertainty_loss()
                weighted_losses['loss_uncertainty_reg'] = uncertainty_reg
            
            return weighted_losses
        
        return losses
    
    def get_relation_matrix(self):
        """获取关系矩阵（用于可视化）"""
        if self.use_enhanced_dict:
            return self.enhanced_dict.get_relation_matrix()
        return None
    
    def get_confusion_matrix(self):
        """获取最近一次的混淆度矩阵（用于可视化）"""
        return self._confusion_matrix


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
        # 交叉注意力
        q_out, _ = self.cross_attn(queries, keys, keys)
        queries = self.norm1(queries + q_out)
        
        # 自注意力
        q_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + q_out)
        
        # FFN
        q_out = self.ffn(queries)
        queries = self.norm3(queries + q_out)
        
        return queries, keys
