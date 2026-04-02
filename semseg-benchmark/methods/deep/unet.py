"""
UNet wrapper using segmentation_models_pytorch (SMP).

Transfer learning strategy:
  - Encoder (ResNet34 by default) is loaded with ImageNet pretrained weights automatically.
  - Decoder and segmentation head are randomly initialised and fine-tuned.
  - A different encoder can be chosen via the `encoder_name` argument.
"""
import segmentation_models_pytorch as smp


def UNet(n_channels: int = 3,
         n_classes: int = 19,
         bilinear: bool = False,   # kept for API compatibility, ignored by SMP
         pretrained_path: str = None,  # kept for API compatibility, ignored (SMP loads from hub)
         encoder_name: str = 'resnet34',
         encoder_weights: str = 'imagenet'):
    """
    Return an SMP UNet with a pretrained encoder.

    Args:
        n_channels:      Number of input image channels (default 3 for RGB).
        n_classes:       Number of output segmentation classes.
        bilinear:        Ignored (kept for backward compatibility).
        pretrained_path: Ignored (kept for backward compatibility).
        encoder_name:    Encoder backbone, e.g. 'resnet34', 'resnet50', 'efficientnet-b4'.
        encoder_weights: Pretrained weights source, e.g. 'imagenet'.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=n_channels,
        classes=n_classes,
    )
    return model
