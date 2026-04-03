"""
SegFormer wrapper using segmentation_models_pytorch (SMP).

Transfer learning strategy:
  - Encoder (ResNet34 by default) is loaded with ImageNet pretrained weights.
  - Decoder and segmentation head are randomly initialised and fine-tuned.
  - A different encoder can be chosen via the `encoder_name` argument.
  - Pass freeze_encoder=True to freeze encoder weights so only the decoder and segmentation head are trained.
"""
import segmentation_models_pytorch as smp


def SegFormer(n_channels: int = 3,
              n_classes: int = 19,
              pretrained_path: str = None,   # kept for API compatibility
              freeze_encoder: bool = True,
              encoder_name: str = 'resnet34',
              encoder_weights: str = 'imagenet'):
    """
    Return an SMP Segformer with a pretrained encoder, ready for fine-tuning.

    Args:
        n_channels:      Number of input image channels (default 3 for RGB).
        n_classes:       Number of output segmentation classes.
        pretrained_path: Ignored (kept for backward compatibility).
        freeze_encoder:  If True (default), freeze encoder weights so only the
                         decoder and segmentation head are trained.
        encoder_name:    Encoder backbone, e.g. 'resnet34', 'resnet50'.
        encoder_weights: Pretrained weights source, e.g. 'imagenet'.
    """
    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=n_channels,
        classes=n_classes,
    )

    # Freeze / unfreeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = not freeze_encoder

    return model