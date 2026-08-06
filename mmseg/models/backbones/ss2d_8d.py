# Copyright (c) OpenMMLab. All rights reserved.
"""
8D SSM (State Space Model) Module
提取自 MF-Mamba 项目

8个扫描方向:
- 0: 水平方向 (左→右)
- 1: 垂直方向 (上→下)  
- 2: 水平反向 (右→左)
- 3: 垂直反向 (下→上)
- 4: 对角线方向
- 5: 反对角线方向
- 6: 对角线反向
- 7: 反对角线反向
"""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# ==================== 辅助函数：斜向扫描 ====================

def antidiagonal_gather(tensor):
    """取出矩阵所有反斜向的元素并拼接"""
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_gather(tensor):
    """取出矩阵所有斜向的元素并拼接"""
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_scatter(tensor_flat, original_shape):
    """把斜向元素拼接起来的一维向量还原为最初的矩阵形式"""
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


def antidiagonal_scatter(tensor_flat, original_shape):
    """把反斜向元素拼接起来的一维向量还原为最初的矩阵形式"""
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


# ==================== 8方向扫描类 ====================

class CrossScan(torch.autograd.Function):
    """8方向扫描：将2D特征图展平为8个方向的1D序列"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 8, C, H * W))
        
        # 水平和垂直扫描
        xs[:, 0] = x.flatten(2, 3)  # 水平
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # 垂直
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # 反向
        
        # 斜向和反斜向扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        
        # 处理水平和垂直方向
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)
        
        # 处理斜向和反斜向方向
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))
        
        return y_rb + y_da


class CrossMerge(torch.autograd.Function):
    """8方向合并：将8个方向的1D序列合并回2D特征图"""
    
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        
        # 处理水平和垂直方向
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)
        
        # 处理斜向和反斜向方向
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))
        
        return (y_rb + y_da).view(B, D, -1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 8, C, L))
        
        # 水平和垂直扫描
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        
        # 斜向和反斜向扫描
        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        
        return xs.view(B, 8, C, H, W)


# ==================== 核心模块：8D SS2D ====================

class SS2D_8D(nn.Module):
    """8方向状态空间模型 (8D State Space Model)
    
    Args:
        d_model: 输入特征维度
        d_state: SSM状态维度，默认16
        d_conv: 卷积核大小，默认3
        expand: 扩展因子，默认2
        dt_rank: delta时间步的秩，默认"auto"
        dropout: dropout率，默认0
        scan: 扫描方向数量，8为8方向扫描
    """
    
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        scan=8,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan = scan
        
        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # 深度可分离卷积
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        
        # 8方向的投影权重
        self.x_proj = []
        for i in range(scan):
            self.x_proj.append(
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        
        # Delta时间步投影
        self.dt_projs = []
        for i in range(scan):
            self.dt_projs.append(
                self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max,
                             dt_init_floor, **factory_kwargs)
            )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        # SSM参数
        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=scan, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=scan, merge=True)
        
        # 输出层
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
    
    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        
        return dt_proj
    
    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log
    
    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def forward_core(self, x: torch.Tensor):
        """8方向SSM核心前向传播"""
        B, C, H, W = x.shape
        L = H * W
        K = self.scan
        
        # 8方向扫描
        xs = CrossScan.apply(x)
        
        # 计算B, C, delta
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        # 准备SSM输入
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        # Selective Scan
        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, H, W)
        
        # 8方向合并
        out_y = CrossMerge.apply(out_y)
        
        return out_y
    
    def forward(self, x: torch.Tensor, **kwargs):
        """前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, H, W]
        
        Returns:
            输出张量，形状为 [B, C, H, W]
        """
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        B, H, W, C = x.shape
        
        # 输入投影并分割
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]
        x = self.act(self.conv2d(x))
        
        # 8方向SSM
        y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        
        # 输出处理
        y = self.out_norm(y)
        y = y * F.silu(z)  # 门控
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = out.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        return out
