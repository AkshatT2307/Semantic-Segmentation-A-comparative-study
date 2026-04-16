import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    """Standalone Encoder-Decoder framework for semantic segmentation."""
    def __init__(self, backbone, decode_head, align_corners=False):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.align_corners = align_corners

    def forward(self, inputs):
        """
        Forward function for inference.
        Args:
            inputs (Tensor): Input images of shape (N, C, H, W)
        Returns:
            Tensor: Semantic segmentation predictions of shape (N, num_classes, H, W)
        """
        x = self.backbone(inputs)
        out = self.decode_head(x)
        out = F.interpolate(
            out,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
