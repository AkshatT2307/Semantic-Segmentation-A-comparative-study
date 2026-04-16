"""
ConvSeg-Net — End-to-end hybrid CNN-Transformer for semantic segmentation.

Combines:
    - Encoder: 2 ConvNeXt stages (local boundaries) + 2 Transformer stages
      with SRA (global semantics).
    - Decoder: Boundary-Aware MLP Fusion with Cross-Attention Boundary Gate.

Outputs (B, num_classes, H, W) logits at full input resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvSegEncoder
from .decoder import ConvSegDecoder


# ── Variant configurations ──────────────────────────────────────────────────
CONVSEG_VARIANTS = {
    'tiny': dict(
        channels=[32, 64, 160, 256],
        depths=[2, 2, 4, 2],
        sr_ratios=[8, 4, 2, 1],
        num_heads=[1, 2, 5, 8],
        embed_dim=256,
    ),
    'small': dict(
        channels=[64, 128, 320, 512],
        depths=[3, 3, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        num_heads=[1, 2, 5, 8],
        embed_dim=256,
    ),
    'base': dict(
        channels=[64, 128, 320, 512],
        depths=[3, 3, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        num_heads=[1, 2, 5, 8],
        embed_dim=256,
    ),
    'large': dict(
        channels=[96, 192, 384, 768],
        depths=[3, 3, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        num_heads=[1, 2, 5, 8],
        embed_dim=256,
    ),
}


class ConvSegNet(nn.Module):
    """ConvSeg-Net: A Hybrid CNN-Transformer for Semantic Segmentation.

    Args:
        num_classes:     Number of segmentation classes.
        channels:        List of 4 encoder stage channel dims.
        depths:          List of 4 block counts per encoder stage.
        sr_ratios:       Spatial reduction ratios for Transformer stages.
        num_heads:       Attention heads for Transformer stages.
        embed_dim:       Unified decoder embedding dimension.
        drop_path_rate:  Maximum stochastic depth rate (linearly distributed).
        drop_rate:       Dropout rate in Transformer blocks.
        decoder_dropout: Spatial dropout before final classifier.
        in_channels:     Number of input image channels.
        gate_sr_ratio:   Spatial reduction for the decoder boundary gate K/V.
    """

    def __init__(
        self,
        num_classes: int = 19,
        channels: list[int] = (64, 128, 320, 512),
        depths: list[int] = (3, 3, 6, 3),
        sr_ratios: list[int] = (8, 4, 2, 1),
        num_heads: list[int] = (1, 2, 5, 8),
        embed_dim: int = 256,
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.0,
        decoder_dropout: float = 0.1,
        in_channels: int = 3,
        gate_sr_ratio: int = 1,
    ):
        super().__init__()

        self.encoder = ConvSegEncoder(
            channels=channels,
            depths=depths,
            sr_ratios=sr_ratios,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            in_channels=in_channels,
        )

        self.decoder = ConvSegDecoder(
            in_channels_list=list(channels),
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
            gate_sr_ratio=gate_sr_ratio,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Weight initialisation following ConvNeXt / SegFormer conventions."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image.
        Returns:
            (B, num_classes, H, W) segmentation logits at full resolution.
        """
        input_size = x.shape[2:]

        # Encoder: 4 multi-scale feature maps
        features = self.encoder(x)

        # Decoder: fuse features → logits at H/4 × W/4
        logits = self.decoder(features)

        # Upsample ×4 to original resolution
        logits = F.interpolate(
            logits, size=input_size, mode='bilinear', align_corners=False,
        )

        return logits


# ── Variant factory functions ────────────────────────────────────────────────

def convseg_t(num_classes: int = 19, **kwargs) -> ConvSegNet:
    """ConvSeg-T (Tiny) — ~6M params, for mobile / edge."""
    cfg = {**CONVSEG_VARIANTS['tiny'], **kwargs}
    return ConvSegNet(num_classes=num_classes, **cfg)


def convseg_s(num_classes: int = 19, **kwargs) -> ConvSegNet:
    """ConvSeg-S (Small) — ~25M params, standard benchmark variant."""
    cfg = {**CONVSEG_VARIANTS['small'], **kwargs}
    return ConvSegNet(num_classes=num_classes, **cfg)


def convseg_b(num_classes: int = 19, **kwargs) -> ConvSegNet:
    """ConvSeg-B (Base) — ~45M params, high accuracy."""
    cfg = {**CONVSEG_VARIANTS['base'], **kwargs}
    return ConvSegNet(num_classes=num_classes, **cfg)


def convseg_l(num_classes: int = 19, **kwargs) -> ConvSegNet:
    """ConvSeg-L (Large) — ~85M params, SOTA push."""
    cfg = {**CONVSEG_VARIANTS['large'], **kwargs}
    return ConvSegNet(num_classes=num_classes, **cfg)
