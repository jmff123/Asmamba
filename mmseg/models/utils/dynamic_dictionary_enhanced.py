# Copyright (c) OpenMMLab. All rights reserved.
"""
增强版动态字典模块

创新点:
1. 可学习的类别关系矩阵: 显式建模类别间的关系
2. 难样本挖掘的对比损失: 自适应关注难区分类别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class RelationMatrix(nn.Module):
    """可学习的类别关系矩阵
    
    显式建模类别间的关系，用于精炼查询表示
    
    Args:
        num_classes: 类别数量
        init_type: 初始化类型 ('identity', 'random', 'prior')
    """
    
    def __init__(self, num_classes: int, init_type: str = 'identity'):
        super().__init__()
        self.num_classes = num_classes
        
        # 初始化关系矩阵
        if init_type == 'identity':
            # 单位矩阵: 初始时类别独立
            init_matrix = torch.eye(num_classes)
        elif init_type == 'random':
            # 随机初始化
            init_matrix = torch.randn(num_classes, num_classes) * 0.01
            init_matrix += torch.eye(num_classes)
        elif init_type == 'prior':
            # 先验知识: 前景类之间更相关
            init_matrix = torch.eye(num_classes)
            # 假设类别0是背景，类别1-2是前景
            init_matrix[1, 2] = 0.3
            init_matrix[2, 1] = 0.3
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        # 可学习的关系矩阵
        self.relation = nn.Parameter(init_matrix)
    
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, num_classes, embed_dim]
        
        Returns:
            refined_queries: [B, num_classes, embed_dim]
        """
        # 用关系矩阵精炼查询
        # queries: [B, C, D]
        # relation: [C, C]
        # refined: [B, C, D]
        refined = torch.einsum('bcd,ce->bed', queries, self.relation)
        return refined
    
    def get_relation_matrix(self) -> torch.Tensor:
        """获取关系矩阵用于可视化"""
        return self.relation.detach()





def compute_hard_contrastive_loss(x, temperature=0.5):
    """计算难样本挖掘的对比损失
    
    对混淆度高的类别对增加权重
    
    Args:
        x: [batch, num_classes, dim] 字典表示
        temperature: 温度参数
    
    Returns:
        loss: 对比损失值
        confusion_matrix: 混淆度矩阵（用于可视化）
    """
    batch_size, num_classes, dim = x.shape
    
    # L2归一化
    x_norm = F.normalize(x, p=2, dim=2)  # [batch, num_classes, dim]
    
    # 计算类别中心
    class_means = torch.mean(x_norm, dim=0)  # [num_classes, dim]
    class_means = F.normalize(class_means, p=2, dim=1)
    
    # 类内损失
    similarity_intra = torch.sum(x_norm * class_means.unsqueeze(0), dim=2)
    intra_loss = torch.mean(1.0 - similarity_intra)
    
    # 类间损失（带难样本挖掘）
    similarity_inter = torch.mm(class_means, class_means.t())  # [num_classes, num_classes]
    
    # 计算混淆度（相似度的绝对值）
    confusion = torch.abs(similarity_inter)
    
    # 对混淆度高的类别对增加权重
    triu_indices = torch.triu_indices(num_classes, num_classes, offset=1, device=x.device)
    inter_similarities = similarity_inter[triu_indices[0], triu_indices[1]]
    confusion_weights = confusion[triu_indices[0], triu_indices[1]]
    
    # 加权的hinge loss
    weighted_inter_loss = confusion_weights * F.relu(inter_similarities + 0.5)
    inter_loss = torch.mean(weighted_inter_loss)
    
    # 总损失
    loss = (intra_loss + inter_loss) / temperature
    
    return loss, confusion.detach()


class EnhancedDynamicDictionary(nn.Module):
    """增强版动态字典
    
    创新点:
    1. 可学习的类别关系矩阵: 显式建模类别间的关系
    2. 难样本挖掘的对比损失: 自适应关注难区分类别
    
    基于原始DynamicDictionary，添加关系矩阵精炼查询
    
    Args:
        num_classes: 类别数量
        embed_dim: 嵌入维度
        query_ratio: 每个类别的候选查询数量
        feature_dim: 输入特征维度
        use_relation_matrix: 是否使用关系矩阵
        relation_init: 关系矩阵初始化类型
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        query_ratio: int = 4,
        feature_dim: int = 1024,
        use_relation_matrix: bool = True,
        relation_init: str = 'identity'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.query_ratio = query_ratio
        self.use_relation_matrix = use_relation_matrix
        
        # 静态基础字典
        self.basic_queries = nn.Embedding(num_classes, embed_dim)
        nn.init.normal_(self.basic_queries.weight, std=0.02)
        
        # 将基础查询扩展为多个候选
        self.token_mlp = nn.Linear(embed_dim, query_ratio * embed_dim)
        
        # 特征调制器（使用原始实现）
        self.modulator = _Modulator(
            num_classes=num_classes,
            embed_dim=embed_dim,
            query_ratio=query_ratio,
            feature_dim=feature_dim
        )
        
        # 可学习的类别关系矩阵（核心创新）
        if use_relation_matrix:
            self.relation_matrix = RelationMatrix(
                num_classes=num_classes,
                init_type=relation_init
            )
    
    def forward(
        self,
        image_feature: torch.Tensor,
        return_static: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            image_feature: 图像特征 [B, C, H, W]
            return_static: 是否返回静态字典
        
        Returns:
            dynamic_queries: [B, num_classes, embed_dim]
            static_queries: [B, num_classes, embed_dim] (可选)
        """
        batch_size = image_feature.shape[0]
        
        # 1. 从图像特征生成动态权重
        weights = self.modulator(image_feature)  # [B, num_classes, query_ratio]
        
        # 2. 扩展基础查询
        basic = self.basic_queries.weight.unsqueeze(0)  # [1, num_classes, embed_dim]
        expanded = self.token_mlp(basic)  # [1, num_classes, query_ratio * embed_dim]
        expanded = expanded.view(self.num_classes * self.query_ratio, self.embed_dim)
        
        # 3. 动态加权组合
        dynamic_queries = []
        for b in range(batch_size):
            weighted = F.conv1d(
                expanded.unsqueeze(0),
                weights[b].unsqueeze(-1),
                groups=self.num_classes
            )
            dynamic_queries.append(weighted)
        
        dynamic_queries = torch.stack(dynamic_queries, dim=0).squeeze(1)  # [B, num_classes, embed_dim]
        
        # 4. 用关系矩阵精炼查询（核心创新）
        if self.use_relation_matrix:
            dynamic_queries = self.relation_matrix(dynamic_queries)
        
        if return_static:
            static_queries = basic.expand(batch_size, -1, -1)
            return dynamic_queries, static_queries
        
        return dynamic_queries, None
    
    def get_relation_matrix(self) -> Optional[torch.Tensor]:
        """获取关系矩阵用于可视化"""
        if self.use_relation_matrix:
            return self.relation_matrix.get_relation_matrix()
        return None


class _Modulator(nn.Module):
    """特征调制器（内部使用）
    
    从图像特征生成动态权重
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        query_ratio: int,
        feature_dim: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.query_ratio = query_ratio
        
        # 特征投影
        self.proj = nn.Linear(feature_dim, embed_dim)
        
        # 双分支通道注意力
        self.gmp_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=1),
        )
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=1),
        )
        
        # 权重生成
        self.weight_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, num_classes * query_ratio),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像特征 [B, C, H, W]
        
        Returns:
            weights: 动态权重 [B, num_classes, query_ratio]
        """
        bs, c, h, w = x.shape
        
        # 投影到统一维度
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.proj(x)  # [B, H*W, embed_dim]
        x = x.transpose(1, 2).reshape(bs, -1, h, w)  # [B, embed_dim, H, W]
        
        # 双分支通道注意力
        x1, x2 = x.chunk(2, dim=1)
        max_attn = self.gmp_branch(x1)
        avg_attn = self.gap_branch(x2)
        
        # 融合并生成权重
        fused = torch.cat([max_attn, avg_attn], dim=1).flatten(1)
        weights = self.weight_mlp(fused)
        weights = weights.view(bs, self.num_classes, self.query_ratio)
        weights = F.softmax(weights, dim=-1)
        
        return weights

