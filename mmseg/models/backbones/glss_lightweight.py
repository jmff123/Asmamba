"""
轻量级GLSS模块：4D SSM + 轻量Local分支

设计理念：
- Global: 使用原始4D SSM（已验证有效）
- Local: 添加轻量级深度卷积（增强局部细节）
- 融合: 简单加权求和，无额外投影

优势：
- 保留4D SSM的稳定性
- 添加局部细节捕获能力
- 参数增加少，训练稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class GLSS_Lightweight(nn.Module):
    """轻量级GLSS：4D SSM + 简单Local分支
    
    Args:
        dim: 输入维度
        d_state: SSM状态维度
        local_kernel: Local分支的卷积核大小（默认3）
        global_weight: Global分支权重（默认0.7）
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        local_kernel: int = 3,
        global_weight: float = 0.7,
    ):
        super().__init__()
        self.dim = dim
        self.global_weight = global_weight
        self.local_weight = 1.0 - global_weight
        
        # Global分支：原始4D SSM
        self.global_branch = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        
        # Local分支：轻量级深度可分离卷积
        # 只用一个卷积，保持简单
        padding = local_kernel // 2
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=local_kernel,
                padding=padding,
                groups=dim,  # 深度卷积
                bias=False
            ),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        
        # 可学习的融合权重（可选）
        self.use_learnable_weights = False
        if self.use_learnable_weights:
            self.fusion_weights = nn.Parameter(
                torch.tensor([global_weight, self.local_weight])
            )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C] 输入特征
            H, W: 空间尺寸
        
        Returns:
            out: [B, H*W, C] 输出特征
        """
        B, N, C = x.shape
        
        # Global分支：4D SSM
        x_global = self.global_branch(x)  # [B, H*W, C]
        
        # Local分支：深度卷积
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        x_local = self.local_branch(x_2d)  # [B, C, H, W]
        x_local = x_local.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 融合：加权求和
        if self.use_learnable_weights:
            weights = F.softmax(self.fusion_weights, dim=0)
            out = weights[0] * x_global + weights[1] * x_local
        else:
            out = self.global_weight * x_global + self.local_weight * x_local
        
        return out


class GLSS_Lightweight_MultiScale(nn.Module):
    """轻量级GLSS：4D SSM + 多尺度Local分支
    
    使用两个不同kernel size的卷积捕获不同尺度的局部特征
    
    Args:
        dim: 输入维度
        d_state: SSM状态维度
        local_kernels: Local分支的卷积核大小列表（默认[3, 5]）
        global_weight: Global分支权重（默认0.6）
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        local_kernels: list = [3, 5],
        global_weight: float = 0.6,
    ):
        super().__init__()
        self.dim = dim
        self.global_weight = global_weight
        self.num_local = len(local_kernels)
        self.local_weight = (1.0 - global_weight) / self.num_local
        
        # Global分支：原始4D SSM
        self.global_branch = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        
        # Local分支：多尺度深度可分离卷积
        self.local_branches = nn.ModuleList()
        for kernel in local_kernels:
            padding = kernel // 2
            branch = nn.Sequential(
                nn.Conv2d(
                    dim, dim,
                    kernel_size=kernel,
                    padding=padding,
                    groups=dim,  # 深度卷积
                    bias=False
                ),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
            self.local_branches.append(branch)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C] 输入特征
            H, W: 空间尺寸
        
        Returns:
            out: [B, H*W, C] 输出特征
        """
        B, N, C = x.shape
        
        # Global分支：4D SSM
        x_global = self.global_branch(x)  # [B, H*W, C]
        
        # Local分支：多尺度卷积
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        x_locals = []
        for branch in self.local_branches:
            x_local = branch(x_2d)  # [B, C, H, W]
            x_local = x_local.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x_locals.append(x_local)
        
        # 融合：加权求和
        out = self.global_weight * x_global
        for x_local in x_locals:
            out = out + self.local_weight * x_local
        
        return out


class GLSS_Lightweight_Adaptive(nn.Module):
    """轻量级GLSS：4D SSM + 自适应Local分支
    
    使用通道注意力动态调整Global和Local的权重
    
    Args:
        dim: 输入维度
        d_state: SSM状态维度
        local_kernel: Local分支的卷积核大小（默认3）
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        local_kernel: int = 3,
    ):
        super().__init__()
        self.dim = dim
        
        # Global分支：原始4D SSM
        self.global_branch = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        
        # Local分支：轻量级深度可分离卷积
        padding = local_kernel // 2
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=local_kernel,
                padding=padding,
                groups=dim,
                bias=False
            ),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        
        # 自适应权重生成器（通道注意力）
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 2, 1),  # 输出2个权重
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C] 输入特征
            H, W: 空间尺寸
        
        Returns:
            out: [B, H*W, C] 输出特征
        """
        B, N, C = x.shape
        
        # Global分支：4D SSM
        x_global = self.global_branch(x)  # [B, H*W, C]
        
        # Local分支：深度卷积
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        x_local = self.local_branch(x_2d)  # [B, C, H, W]
        
        # 生成自适应权重
        weights = self.weight_generator(x_2d)  # [B, 2, 1, 1]
        w_global = weights[:, 0:1, :, :]  # [B, 1, 1, 1]
        w_local = weights[:, 1:2, :, :]   # [B, 1, 1, 1]
        
        # 融合
        x_fused = w_global * x_global.transpose(1, 2).reshape(B, C, H, W) + \
                  w_local * x_local
        
        # 转回序列格式
        out = x_fused.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        return out
