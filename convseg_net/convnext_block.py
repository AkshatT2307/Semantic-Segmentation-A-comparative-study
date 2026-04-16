"""
ConvNeXt Block — used in encoder stages 1 & 2.

Implements the modernized ConvNeXt design:
    Y = X + γ · Conv1x1( GELU( Conv1x1( LN( DWConv7x7(X) ) ) ) )

where γ is a learnable per-channel scale initialised to 1e-6.

Reference: A ConvNet for the 2020s (Liu et al., 2022)
"""

import torch
import torch.nn as nn


class LayerNormChannels(nn.Module):
    """LayerNorm applied on the channel dimension for (B, C, H, W) tensors.

    Equivalent to permuting to (B, H, W, C), applying LayerNorm, then back.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DropPath(nn.Module):
    """Stochastic depth — drops the entire residual branch with probability p."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # shape: (B, 1, 1, 1) for broadcasting
        mask = torch.empty(
            x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype
        ).bernoulli_(keep)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block.

    Architecture:
        7×7 Depthwise Conv → LayerNorm → 1×1 Conv (expand 4×) → GELU
        → 1×1 Conv (project back) → layer_scale γ → DropPath → residual add

    Args:
        dim:                Number of input/output channels.
        drop_path:          Stochastic depth rate.
        layer_scale_init:   Initial value for per-channel scale γ.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        # 7×7 depthwise conv — large receptive field, minimal parameters
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNormChannels(dim)
        # Point-wise expansion and projection (inverted bottleneck)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        # Learnable per-channel scale
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(1, dim, 1, 1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = self.drop_path(x)
        return residual + x
