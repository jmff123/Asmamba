"""
GLSS (Global-Local State Space) 模块

结合Local (多尺度深度可分离卷积) 和 Global (8D SSM) 分支
Local部分提取自 MF-Mamba 项目

Local部分使用两个不同kernel size的深度可分离卷积来捕获局部特征:
- conv1: kernel_size=3 捕获细粒度局部特征
- conv2: kernel_size=5 捕获稍大范围的局部特征
"""

import torch
import torch.nn as nn


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


class GLSSLocal(nn.Module):
    """
    GLSS模块的Local部分
    
    使用两个不同kernel size的深度可分离卷积捕获多尺度局部特征
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层特征维度，默认等于in_features
        out_features: 输出特征维度，默认等于in_features
        kernel_sizes: 两个卷积的kernel size，默认[3, 5]
        act_layer: 激活函数，默认ReLU6
        drop: dropout率
    """
    
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 kernel_sizes=[3, 5],
                 act_layer=nn.ReLU6,
                 drop=0.):
        super(GLSSLocal, self).__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 输入投影
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        
        # 两个不同kernel size的深度可分离卷积
        self.conv1 = DepthWiseConv(in_channel=hidden_features, out_channel=hidden_features, kernel=kernel_sizes[0])
        self.conv2 = DepthWiseConv(in_channel=hidden_features, out_channel=hidden_features, kernel=kernel_sizes[1])
        
        # 输出投影
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, H, W]
        
        Returns:
            输出张量，形状为 [B, C, H, W]
        """
        x = self.fc1(x)
        
        # Local: 多尺度深度可分离卷积
        x1 = self.conv1(x)  # kernel=3
        x2 = self.conv2(x)  # kernel=5
        
        # 融合
        x = self.fc2(x1 + x2)
        x = self.act(x)
        
        return x


class GLSS(nn.Module):
    """
    完整的GLSS模块 (Global-Local State Space)
    
    结合Local (多尺度深度可分离卷积) 和 Global (8D SSM) 分支
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层特征维度
        out_features: 输出特征维度
        kernel_sizes: Local分支的kernel sizes
        act_layer: 激活函数
        drop: dropout率
        atten_drop: attention dropout率
        ssm_module: 外部传入的SSM模块类 (如SS2D_8D)，若为None则只使用Local
    """
    
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 kernel_sizes=[3, 5],
                 act_layer=nn.ReLU6,
                 drop=0.,
                 atten_drop=0.,
                 ssm_module=None):
        super(GLSS, self).__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 输入投影
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        
        # Local分支: 多尺度深度可分离卷积
        self.conv1 = DepthWiseConv(in_channel=hidden_features, out_channel=hidden_features, kernel=kernel_sizes[0])
        self.conv2 = DepthWiseConv(in_channel=hidden_features, out_channel=hidden_features, kernel=kernel_sizes[1])
        
        # Global分支: SSM (可选)
        self.use_ssm = ssm_module is not None
        if self.use_ssm:
            self.attn = ssm_module(d_model=hidden_features, dropout=atten_drop, d_state=16)
        
        # 输出投影
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, H, W]
        
        Returns:
            输出张量，形状为 [B, C, H, W]
        """
        x = self.fc1(x)
        
        # Local分支
        x1 = self.conv1(x)  # kernel=3
        x2 = self.conv2(x)  # kernel=5
        
        # Global分支 (如果有)
        if self.use_ssm:
            x3 = self.attn(x)
            x = self.fc2(x1 + x2 + x3)
        else:
            x = self.fc2(x1 + x2)
        
        x = self.act(x)
        
        return x
