"""
ConvSeg-Net — A Hybrid CNN-Transformer Architecture for Semantic Segmentation.

Pure PyTorch implementation with no external dependencies beyond torch.

Usage:
    from convseg_net import ConvSegNet, convseg_s

    # Using a variant factory
    model = convseg_s(num_classes=19)

    # Using the base class directly
    model = ConvSegNet(num_classes=19, channels=[64,128,320,512], depths=[3,3,6,3])

    # Forward pass
    logits = model(images)  # (B, num_classes, H, W)
"""

from .model import ConvSegNet, convseg_t, convseg_s, convseg_b, convseg_l, CONVSEG_VARIANTS

__all__ = [
    'ConvSegNet',
    'convseg_t',
    'convseg_s',
    'convseg_b',
    'convseg_l',
    'CONVSEG_VARIANTS',
]
