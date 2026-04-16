"""
ConvSeg-Net Encoder — 4-stage hierarchical feature pyramid.

    Stage 1  (CNN):          H/4  × W/4  × C1   — ConvNeXt blocks
    Stage 2  (CNN):          H/8  × W/8  × C2   — ConvNeXt blocks
    Stage 3  (Transformer):  H/16 × W/16 × C3   — Transformer blocks with SRA
    Stage 4  (Transformer):  H/32 × W/32 × C4   — Transformer blocks with SRA

Returns a list of 4 feature maps [F1, F2, F3, F4] in (B, C, H, W) format.
"""

import torch
import torch.nn as nn

from .convnext_block import ConvNeXtBlock, LayerNormChannels
from .transformer_block import TransformerBlock


class PatchEmbedStem(nn.Module):
    """Initial patch embedding: 4×4 conv with stride 4 + LayerNorm.

    Maps (B, 3, H, W) → (B, C1, H/4, W/4).
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=4, stride=4,
        )
        self.norm = LayerNormChannels(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class DownsampleLayer(nn.Module):
    """Spatial downsampling between stages: 2×2 conv stride 2 + LayerNorm.

    Maps (B, C_in, H, W) → (B, C_out, H/2, W/2).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = LayerNormChannels(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class ConvSegEncoder(nn.Module):
    """Hierarchical 4-stage encoder mixing CNN and Transformer blocks.

    Args:
        channels:       List of 4 channel dims [C1, C2, C3, C4].
        depths:         List of 4 block counts per stage.
        sr_ratios:      Spatial reduction ratios for Transformer stages.
                        Index 0-1 are ignored (CNN stages).
        num_heads:      Attention heads for Transformer stages.
                        Index 0-1 are ignored (CNN stages).
        drop_path_rate: Maximum stochastic depth rate (linearly distributed).
        drop_rate:      Dropout rate inside Transformer blocks.
        qkv_bias:       Bias in Q/K/V projections.
        in_channels:    Number of input image channels.
    """

    def __init__(
        self,
        channels: list[int] = (64, 128, 320, 512),
        depths: list[int] = (3, 3, 6, 3),
        sr_ratios: list[int] = (8, 4, 2, 1),
        num_heads: list[int] = (1, 2, 5, 8),
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.0,
        qkv_bias: bool = True,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_stages = 4

        # Compute linearly increasing drop-path rates across all blocks
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        block_idx = 0

        # ── Stem: image → H/4 ──
        self.stem = PatchEmbedStem(in_channels, channels[0])

        # ── Stage 1: CNN (H/4) ──
        stage1_blocks = []
        for i in range(depths[0]):
            stage1_blocks.append(ConvNeXtBlock(channels[0], drop_path=dpr[block_idx]))
            block_idx += 1
        self.stage1 = nn.Sequential(*stage1_blocks)

        # ── Downsample 1→2 ──
        self.down1 = DownsampleLayer(channels[0], channels[1])

        # ── Stage 2: CNN (H/8) ──
        stage2_blocks = []
        for i in range(depths[1]):
            stage2_blocks.append(ConvNeXtBlock(channels[1], drop_path=dpr[block_idx]))
            block_idx += 1
        self.stage2 = nn.Sequential(*stage2_blocks)

        # ── Downsample 2→3 ──
        self.down2 = DownsampleLayer(channels[1], channels[2])

        # ── Stage 3: Transformer (H/16) ──
        stage3_blocks = []
        for i in range(depths[2]):
            stage3_blocks.append(TransformerBlock(
                dim=channels[2],
                num_heads=num_heads[2],
                sr_ratio=sr_ratios[2],
                drop=drop_rate,
                drop_path=dpr[block_idx],
                qkv_bias=qkv_bias,
            ))
            block_idx += 1
        self.stage3 = nn.ModuleList(stage3_blocks)
        self.norm3 = nn.LayerNorm(channels[2])

        # ── Downsample 3→4 ──
        self.down3 = DownsampleLayer(channels[2], channels[3])

        # ── Stage 4: Transformer (H/32) ──
        stage4_blocks = []
        for i in range(depths[3]):
            stage4_blocks.append(TransformerBlock(
                dim=channels[3],
                num_heads=num_heads[3],
                sr_ratio=sr_ratios[3],
                drop=drop_rate,
                drop_path=dpr[block_idx],
                qkv_bias=qkv_bias,
            ))
            block_idx += 1
        self.stage4 = nn.ModuleList(stage4_blocks)
        self.norm4 = nn.LayerNorm(channels[3])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image.
        Returns:
            List of 4 feature maps, each (B, C_i, H_i, W_i).
        """
        outs = []

        # Stage 1: CNN
        x = self.stem(x)           # (B, C1, H/4, W/4)
        x = self.stage1(x)
        outs.append(x)

        # Stage 2: CNN
        x = self.down1(x)          # (B, C2, H/8, W/8)
        x = self.stage2(x)
        outs.append(x)

        # Stage 3: Transformer
        x = self.down2(x)          # (B, C3, H/16, W/16)
        B, C, H3, W3 = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
        for blk in self.stage3:
            x = blk(x, H3, W3)
        x = self.norm3(x)
        x = x.permute(0, 2, 1).reshape(B, C, H3, W3)  # back to (B, C, H, W)
        outs.append(x)

        # Stage 4: Transformer
        x = self.down3(x)          # (B, C4, H/32, W/32)
        B, C, H4, W4 = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        for blk in self.stage4:
            x = blk(x, H4, W4)
        x = self.norm4(x)
        x = x.permute(0, 2, 1).reshape(B, C, H4, W4)
        outs.append(x)

        return outs
