"""
SS2D_4D: 4方向状态空间模型

4个扫描方向:
1. 水平 →     : 从左到右
2. 垂直 ↓     : 从上到下
3. 水平反向 ← : 从右到左
4. 垂直反向 ↑ : 从下到上

相比8D版本:
- 参数量减半
- 计算量减半
- 仍然能捕获主要的空间依赖关系
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# ==================== 4方向扫描类 ====================

class CrossScan4D(torch.autograd.Function):
    """4方向扫描：将2D特征图展平为4个方向的1D序列"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]
        
        Returns:
            xs: [B, 4, C, L] 其中L=H*W
        """
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        
        xs = x.new_empty((B, 4, C, H * W))
        
        # 1. 水平扫描 (从左到右，逐行)
        xs[:, 0] = x.flatten(2, 3)  # [B, C, H*W]
        
        # 2. 垂直扫描 (从上到下，逐列)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # [B, C, W*H]
        
        # 3. 水平反向扫描 (从右到左)
        xs[:, 2] = torch.flip(xs[:, 0], dims=[-1])
        
        # 4. 垂直反向扫描 (从下到上)
        xs[:, 3] = torch.flip(xs[:, 1], dims=[-1])
        
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        """
        Args:
            ys: [B, 4, C, L]
        
        Returns:
            y: [B, C, H, W]
        """
        B, C, H, W = ctx.shape
        L = H * W
        
        # 处理水平方向 (正向 + 反向)
        y_horizontal = ys[:, 0] + ys[:, 2].flip(dims=[-1])
        y_horizontal = y_horizontal.view(B, C, H, W)
        
        # 处理垂直方向 (正向 + 反向)
        y_vertical = ys[:, 1] + ys[:, 3].flip(dims=[-1])
        y_vertical = y_vertical.view(B, C, W, H).transpose(dim0=2, dim1=3)
        
        # 融合两个方向
        y = y_horizontal + y_vertical
        
        return y


class CrossMerge4D(torch.autograd.Function):
    """4方向合并：将4个方向的输出合并回2D特征图"""
    
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        """
        Args:
            ys: [B, 4, D, L] 其中D是输出维度
        
        Returns:
            y: [B, D, H, W]
        """
        B, K, D, L = ys.shape
        H = W = int(math.sqrt(L))
        ctx.shape = (H, W)
        
        # 处理水平方向
        y_horizontal = ys[:, 0] + ys[:, 2].flip(dims=[-1])
        y_horizontal = y_horizontal.view(B, D, H, W)
        
        # 处理垂直方向
        y_vertical = ys[:, 1] + ys[:, 3].flip(dims=[-1])
        y_vertical = y_vertical.view(B, D, W, H).transpose(dim0=2, dim1=3)
        
        # 融合
        y = y_horizontal + y_vertical
        
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, H, W = x.shape
        L = H * W
        
        xs = x.new_empty((B, 4, C, L))
        
        # 水平
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 2] = xs[:, 0].flip(dims=[-1])
        
        # 垂直
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 3] = xs[:, 1].flip(dims=[-1])
        
        return xs


# ==================== 核心模块：4D SS2D ====================

class SS2D_4D(nn.Module):
    """4方向状态空间模型 (4D State Space Model)
    
    相比标准Mamba:
    - 4个扫描方向 vs 1个方向
    - 更好的空间建模能力
    - 参数量增加约2倍
    
    相比8D版本:
    - 参数量减半
    - 计算量减半
    - 保留主要方向的空间依赖
    
    Args:
        d_model: 模型维度
        d_state: SSM状态维度，默认16
        d_conv: SSM卷积核大小，默认3
        expand: 扩展因子，默认2
        dt_rank: delta时间步的秩，默认"auto"
        dropout: dropout率，默认0
        scan: 扫描方向数量，固定为4
    """
    
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dropout=0.,
        scan=4,  # 固定为4方向
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan = 4  # 固定4方向
        
        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False, **factory_kwargs)
        
        # 1D卷积
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        
        # 4方向的投影权重
        self.x_proj = []
        for i in range(self.scan):
            self.x_proj.append(
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        
        # Delta时间步投影
        self.dt_projs = []
        for i in range(self.scan):
            self.dt_projs.append(self._dt_init(self.dt_rank, self.d_inner, **factory_kwargs))
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        # SSM参数
        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=self.scan, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=self.scan, merge=True)
        
        # 输出层
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    
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
    
    def forward(self, x: torch.Tensor, H: int, W: int):
        """
        Args:
            x: [B, L, C] 其中L=H*W
            H, W: 空间维度
        
        Returns:
            out: [B, L, C]
        """
        B, L, C = x.shape
        assert L == H * W, f"Sequence length {L} != H*W {H*W}"
        K = self.scan  # 4
        
        # 1. 输入投影
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # 各 [B, L, d_inner]
        
        # 2. 转换为2D格式进行卷积
        x = x.transpose(1, 2).view(B, self.d_inner, H, W)  # [B, d_inner, H, W]
        x = self.conv2d(x)  # [B, d_inner, H, W]
        x = self.act(x)
        
        # 3. 4方向扫描
        xs = CrossScan4D.apply(x)  # [B, 4, d_inner, L]
        
        # 4. 计算B, C, delta
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        # 5. 准备SSM输入
        xs = xs.float().view(B, -1, L)  # [B, K*d_inner, L]
        dts = dts.contiguous().float().view(B, -1, L)  # [B, K*d_inner, L]
        Bs = Bs.float().view(B, K, -1, L)  # [B, K, d_state, L]
        Cs = Cs.float().view(B, K, -1, L)  # [B, K, d_state, L]
        
        As = -torch.exp(self.A_logs.float())  # [K*d_inner, d_state]
        Ds = self.Ds.float()  # [K*d_inner]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # [K*d_inner]
        
        # 6. Selective Scan
        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        
        # 7. 合并4个方向
        out_y = out_y.view(B, K, -1, L)  # [B, K, d_inner, L]
        y = CrossMerge4D.apply(out_y)  # [B, d_inner, H, W]
        
        # 8. 转回序列格式
        y = y.flatten(2).transpose(1, 2)  # [B, L, d_inner]
        
        # 9. 门控和输出投影
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = self.dropout(out)
        
        return out


# ==================== MambaLayer包装器 ====================

class MambaLayer_4D(nn.Module):
    """4D SSM版本的MambaLayer
    
    使用SS2D_4D替换原始Mamba，支持4方向扫描
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # 使用4D SSM
        self.mamba_4d = SS2D_4D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            scan=4,  # 4方向扫描
        )
    
    def forward(self, x, H, W):
        """
        Args:
            x: [B, L, C] 输入特征
            H, W: 空间维度
        
        Returns:
            x_mamba: [B, L, C] 输出特征
        """
        B, L, C = x.shape
        assert L == H * W, f"Sequence length {L} != H*W {H*W}"
        
        x_norm = self.norm(x)
        x_mamba = self.mamba_4d(x_norm, H, W)
        
        return x_mamba


if __name__ == '__main__':
    # 测试4D SSM
    print("="*60)
    print("测试 SS2D_4D (4方向扫描)")
    print("="*60)
    
    B, H, W, C = 2, 32, 32, 64
    L = H * W
    
    # 创建模型
    model = SS2D_4D(d_model=C, d_state=16, d_conv=3, expand=2)
    
    # 创建输入
    x = torch.randn(B, L, C)
    
    # 前向传播
    print(f"\n输入: {x.shape}")
    y = model(x, H, W)
    print(f"输出: {y.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 对比标准Mamba
    from mamba_ssm import Mamba
    mamba_1d = Mamba(d_model=C, d_state=16, d_conv=4, expand=2)
    mamba_params = sum(p.numel() for p in mamba_1d.parameters())
    print(f"标准Mamba参数量: {mamba_params:,} ({mamba_params/1e6:.2f}M)")
    print(f"参数增加: {(total_params/mamba_params - 1)*100:.1f}%")
    
    print("\n" + "="*60)
    print("✓ 测试完成！")
    print("="*60)
