# Copyright (c) OpenMMLab. All rights reserved.
"""
独立的动态字典与静态字典模块
可单独使用或集成到其他分割网络中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_contrastive_loss(x, margin=10.0, temperature=0.5):
    """计算对比损失，鼓励类内紧凑、类间分离
    
    使用改进的对比损失公式：
    L = intra_loss + inter_loss
    
    其中：
    - intra_loss: 类内紧凑性（最小化类内方差）
    - inter_loss: 类间分离性（使用hinge loss确保最小间距）
    
    Args:
        x: [batch, num_classes, dim] 字典表示
        margin: 类间距离的最小期望值（默认10.0）
        temperature: 温度参数，用于缩放损失（默认0.5，从0.1调整以增加损失贡献）
    
    Returns:
        loss: 对比损失值
    """
    batch_size, num_classes, dim = x.shape
    
    # L2归一化，使得比较更稳定
    x_norm = F.normalize(x, p=2, dim=2)  # [batch, num_classes, dim]
    
    # 计算每个类别在batch内的均值（已归一化）
    class_means = torch.mean(x_norm, dim=0)  # [num_classes, dim]
    class_means = F.normalize(class_means, p=2, dim=1)  # 再次归一化
    
    # 类内损失：最小化同类样本到类中心的余弦距离
    # 余弦相似度: cos(θ) = x·y / (||x|| ||y||)
    # 余弦距离: 1 - cos(θ)
    similarity_intra = torch.sum(x_norm * class_means.unsqueeze(0), dim=2)  # [batch, num_classes]
    intra_loss = torch.mean(1.0 - similarity_intra)  # 最小化余弦距离
    
    # 类间损失：使用 hinge loss 确保类间余弦距离足够大
    # 计算所有类别对之间的余弦相似度
    similarity_inter = torch.mm(class_means, class_means.t())  # [num_classes, num_classes]
    
    # 只考虑上三角（避免重复计算和对角线）
    triu_indices = torch.triu_indices(num_classes, num_classes, offset=1, device=x.device)
    inter_similarities = similarity_inter[triu_indices[0], triu_indices[1]]  # [num_pairs]
    
    # Hinge loss: 如果相似度太高（距离太小），则惩罚
    # 我们希望余弦相似度 < 0（即余弦距离 > 1）
    inter_loss = torch.mean(F.relu(inter_similarities + 0.5))  # 惩罚相似度 > -0.5 的类别对
    
    # 总损失 = 类内紧凑 + 类间分离
    loss = (intra_loss + inter_loss) / temperature
    
    return loss


class StaticDictionary(nn.Module):
    """静态字典模块
    
    使用可学习的嵌入作为固定的类别查询向量
    
    Args:
        num_classes: 类别数量（字典长度）
        embed_dim: 嵌入维度
    """
    
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 可学习的类别嵌入
        self.class_embeddings = nn.Embedding(num_classes, embed_dim)
        
        # 初始化
        nn.init.normal_(self.class_embeddings.weight, std=0.02)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Args:
            batch_size: 批次大小
        
        Returns:
            queries: [batch_size, num_classes, embed_dim]
        """
        queries = self.class_embeddings.weight.unsqueeze(0)
        return queries.expand(batch_size, -1, -1)
    
    def get_dictionary(self) -> torch.Tensor:
        """获取原始字典权重 [num_classes, embed_dim]"""
        return self.class_embeddings.weight


class DynamicDictionary(nn.Module):
    """动态字典模块
    
    根据输入图像特征动态生成类别查询向量
    
    Args:
        num_classes: 类别数量（字典长度）
        embed_dim: 嵌入维度
        query_ratio: 每个类别的候选查询数量
        feature_dim: 输入特征维度（用于 Modulator）
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        query_ratio: int = 4,
        feature_dim: int = 1024
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.query_ratio = query_ratio
        
        # 静态基础字典
        self.basic_queries = nn.Embedding(num_classes, embed_dim)
        
        # 将基础查询扩展为多个候选
        self.token_mlp = nn.Linear(embed_dim, query_ratio * embed_dim)
        
        # 特征调制器：从图像特征生成动态权重
        self.modulator = _Modulator(
            num_classes=num_classes,
            embed_dim=embed_dim,
            query_ratio=query_ratio,
            feature_dim=feature_dim
        )
    
    def forward(
        self,
        image_feature: torch.Tensor,
        return_static: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            image_feature: 图像特征 [B, C, H, W]
            return_static: 是否同时返回静态字典
        
        Returns:
            dynamic_queries: 动态字典 [B, num_classes, embed_dim]
            static_queries: 静态字典 [B, num_classes, embed_dim] (可选)
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
        
        dynamic_queries = torch.stack(dynamic_queries, dim=0).squeeze(1)
        
        if return_static:
            static_queries = basic.expand(batch_size, -1, -1)
            return dynamic_queries, static_queries
        
        return dynamic_queries, None
    
    def get_static_dictionary(self) -> torch.Tensor:
        """获取静态基础字典 [num_classes, embed_dim]"""
        return self.basic_queries.weight


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


class DictionaryDecoder(nn.Module):
    """基于字典的解码器
    
    将字典查询与图像特征交互生成分割结果
    
    Args:
        num_classes: 类别数量
        embed_dim: 嵌入维度
        num_layers: 交互层数
        num_heads: 注意力头数
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        output_scale: int = 4  # 相对于输入特征的上采样倍数
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 交互层
        self.layers = nn.ModuleList([
            _InteractionBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # 上采样
        self.upsampler = nn.Sequential(
            nn.PixelShuffle(2),
            nn.LayerNorm([embed_dim // 4]),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.GELU(),
        ) if output_scale == 4 else nn.Identity()
        
        # 输出投影
        out_dim = embed_dim // 16 if output_scale == 4 else embed_dim
        self.output_proj = nn.Linear(embed_dim, out_dim)
    
    def forward(
        self,
        image_features: torch.Tensor,
        queries: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_features: 图像特征 [B, C, H, W]
            queries: 字典查询 [B, num_classes, embed_dim]
        
        Returns:
            output: 分割输出 [B, num_classes, H*scale, W*scale]
        """
        b, c, h, w = image_features.shape
        
        # 展平图像特征
        keys = image_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # 字典-图像交互
        for layer in self.layers:
            queries, keys = layer(queries, keys)
        
        queries = self.final_norm(queries)
        
        # 上采样图像特征
        upscaled = self.upsampler(image_features)  # [B, C//16, H*4, W*4]
        
        # 生成分割 mask
        hyper = self.output_proj(queries)  # [B, num_classes, C//16]
        b, c_up, h_up, w_up = upscaled.shape
        output = hyper @ upscaled.view(b, c_up, -1)  # [B, num_classes, H*W]
        output = output.view(b, self.num_classes, h_up, w_up)
        
        return output


class _InteractionBlock(nn.Module):
    """交互块：实现字典与图像特征的双向注意力"""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.cross_attn_q2k = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.cross_attn_k2q = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Query → Key 交叉注意力
        q_out, _ = self.cross_attn_q2k(queries, keys, keys)
        queries = self.norm1(queries + q_out)
        
        # Key → Query 交叉注意力
        k_out, _ = self.cross_attn_k2q(keys, queries, queries)
        keys = self.norm2(keys + k_out)
        
        return queries, keys
