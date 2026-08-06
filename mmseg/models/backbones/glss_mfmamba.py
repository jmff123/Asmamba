"""
完整的 GLSS (Global-Local State Space) 模块
基于 MF-Mamba 项目实现

结构:
- Local分支: 两个不同kernel size的深度可分离卷积 (3x3, 5x5)
- Global分支: 8D SSM (使用项目现有的SS2D_8D)
- 三分支特征融合

输入 [B, C, H, W]
     ↓
fc1 (1x1 Conv + BN + ReLU6)
     ↓
┌──────────────┬──────────────┬──────────────┐
│ conv1 (3x3)  │ conv2 (5x5)  │  SS2D_8D     │
│ Local细粒度  │ Local中粒度   │  Global全局  │
└──────────────┴──────────────┴──────────────┘
     ↓ 三者相加
fc2 (1x1 Conv + BN) + ReLU6
     ↓
输出 [B, C, H, W]
"""

import torch
import torch.nn as nn

# 导入项目现有的8D SSM
from .ss2d_8d import SS2D_8D


# ==================== 基础卷积模块 ====================

class ConvBNReLU(nn.Sequential):
    """Conv + BatchNorm + ReLU6"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                     dilation=dilation, stride=stride,
                     padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    """Conv + BatchNorm"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                     dilation=dilation, stride=stride,
                     padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class DepthWiseConv(nn.Module):
    """
    深度可分离卷积
    
    Args:
        in_channel: 输入通道数
        out_channel: 输出通道数
        kernel: 卷积核大小
    """
    
    def __init__(self, in_channel, out_channel, kernel):
        super(DepthWiseConv, self).__init__()
        # 逐通道卷积 (Depthwise)
        self.depth_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=kernel,
            stride=1,
            padding=(kernel - 1) // 2,
            groups=in_channel
        )
        # 逐点卷积 (Pointwise)
        self.point_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
    
    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


# ==================== 完整 GLSS 模块 ====================

class GLSS_MFMamba(nn.Module):
    """
    完整的 GLSS (Global-Local State Space) 模块
    基于 MF-Mamba 实现，使用项目现有的 SS2D_8D
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层特征维度，默认等于in_features
        out_features: 输出特征维度，默认等于in_features
        kernel_sizes: Local分支的两个卷积核大小，默认[3, 5]
        act_layer: 激活函数，默认ReLU6
        drop: dropout率
        d_state: SSM的状态维度
        d_conv: SSM的卷积核大小
        expand: SSM的扩展因子
    """
    
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 kernel_sizes=[3, 5],
                 act_layer=nn.ReLU6,
                 drop=0.,
                 d_state=16,
                 d_conv=3,
                 expand=2):
        super(GLSS_MFMamba, self).__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 输入投影
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        
        # Local分支: 两个不同kernel的深度可分离卷积
        self.conv1 = DepthWiseConv(in_channel=hidden_features, out_channel=hidden_features, kernel=kernel_sizes[0])
        self.conv2 = DepthWiseConv(in_channel=hidden_features, out_channel=hidden_features, kernel=kernel_sizes[1])
        
        # Global分支: 使用项目现有的 SS2D_8D
        self.attn = SS2D_8D(
            d_model=hidden_features,
            d_state=d_state,
            d_conv=d_conv,
            expand=1,  # 外层已经有hidden_features的扩展
            scan=8,
        )
        
        # 输出投影
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, H, W]
        
        Returns:
            输出张量，形状为 [B, C, H, W]
        """
        # 输入投影
        x = self.fc1(x)  # [B, C, H, W] -> [B, hidden, H, W]
        
        # Local分支
        x1 = self.conv1(x)  # 3x3 深度可分离卷积
        x2 = self.conv2(x)  # 5x5 深度可分离卷积
        
        # Global分支
        x3 = self.attn(x)   # 8D SSM
        
        # 三分支等权重融合（MF-Mamba方式）
        x = self.fc2(x1 + x2 + x3)
        
        # 激活和dropout
        x = self.act(x)
        x = self.drop(x)
        
        return x


# ==================== 简化版 GLSS（推荐用于Asmamba）====================

class GLSS_Simple(nn.Module):
    """
    简化版 GLSS 模块
    
    改进：
    - 去掉fc1/fc2的维度变换，直接在原始维度上工作
    - Global分支为主，Local分支为辅
    - 加权融合，避免过度混合
    
    Args:
        dim: 特征维度
        kernel_sizes: Local分支的两个卷积核大小
        local_weight: Local分支的权重（0-1），Global分支权重为1-local_weight
        d_state: SSM的状态维度
        d_conv: SSM的卷积核大小
        expand: SSM的扩展因子
    """
    
    def __init__(self,
                 dim,
                 kernel_sizes=[3, 5],
                 local_weight=0.3,
                 d_state=16,
                 d_conv=3,
                 expand=2):
        super(GLSS_Simple, self).__init__()
        
        self.local_weight = local_weight
        
        # Global分支：8D SSM
        self.global_branch = SS2D_8D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            scan=8,
        )
        
        # Local分支：轻量级深度卷积
        self.local_conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_sizes[0], 
                                     padding=kernel_sizes[0]//2, groups=dim)
        self.local_conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_sizes[1], 
                                     padding=kernel_sizes[1]//2, groups=dim)
        
        # 融合层
        self.fusion = nn.Conv2d(dim, dim, kernel_size=1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, H, W]
        
        Returns:
            输出张量，形状为 [B, C, H, W]
        """
        # Global分支：8D SSM
        x_global = self.global_branch(x)
        
        # Local分支：多尺度深度卷积
        x_local1 = self.local_conv1(x)
        x_local2 = self.local_conv2(x)
        x_local = x_local1 + x_local2
        
        # 加权融合：Global为主，Local为辅
        x_out = self.fusion(
            (1 - self.local_weight) * x_global + 
            self.local_weight * x_local
        )
        
        return x_out
