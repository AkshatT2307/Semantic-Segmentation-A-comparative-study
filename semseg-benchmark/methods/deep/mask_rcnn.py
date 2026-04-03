"""
FPN (Feature Pyramid Network) wrapper using segmentation_models_pytorch (SMP),
often used in Mask R-CNN like architectures.

Transfer learning strategy:
  - Encoder (ResNet34 by default) is loaded with ImageNet pretrained weights automatically.
  - Decoder and segmentation head are randomly initialised and fine-tuned.
  - A different encoder can be chosen via the `encoder_name` argument.
"""
import segmentation_models_pytorch as smp


def FPN(n_channels: int = 3,
        n_classes: int = 19,
        pretrained_path: str = None,   # kept for API compatibility, ignored by SMP
        freeze_encoder: bool = False,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet'):
    """
    Return an SMP FPN with a pretrained encoder.

    Args:
        n_channels:      Number of input image channels (default 3 for RGB).
        n_classes:       Number of output segmentation classes.
        pretrained_path: Ignored (kept for backward compatibility).
        freeze_encoder:  If True, freeze encoder weights so only the
                         decoder and segmentation head are trained.
        encoder_name:    Encoder backbone, e.g. 'resnet34', 'resnet50'.
        encoder_weights: Pretrained weights source, e.g. 'imagenet'.
    """
    model = smp.FPN(
        encoder_name=encoder_name,
        encoder_depth=5,
        encoder_weights=encoder_weights,
        decoder_pyramid_channels=256,
        decoder_segmentation_channels=128,
        decoder_merge_policy='add',
        decoder_dropout=0.2,
        in_channels=n_channels,
        classes=n_classes,
        activation=None,
        upsampling=4,
        aux_params=None
    )
    
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model

# Alias for backwards compatibility with the file name
MaskRCNN = FPN
