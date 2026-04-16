import torch
import torch.nn.functional as F
import numpy as np
import warnings

from .backbones.resnet import ResNetV1c
from .backbones.mit import MixVisionTransformer
from .decode_heads.fcn_head import FCNHead
from .decode_heads.aspp_head import ASPPHead
from .decode_heads.segformer_head import SegformerHead
from .segmentors.encoder_decoder import EncoderDecoder

def _load_checkpoint(model, checkpoint_path, device):
    """Loads an mmsegmentation checkpoint and maps the state dict."""
    if checkpoint_path is None:
        return model
        
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    # Remove auxiliary head weights since we don't use them for inference
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('auxiliary_head.')}
    
    # Handle DeepLabV3 ASSP name mismatch (if any)
    # Handle FCNHead name match
    
    msg = model.load_state_dict(state_dict, strict=False)
    if len(msg.missing_keys) > 0:
        warnings.warn(f"Missing keys when loading checkpoint: {msg.missing_keys}")
    
    model = model.to(device)
    model.eval()
    return model

def inference_model(model, img_path_or_bgr):
    import cv2
    if isinstance(img_path_or_bgr, str):
        img = cv2.imread(img_path_or_bgr)
    else:
        img = img_path_or_bgr
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalization (Cityscapes / ADE20K standard mmseg values)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img_norm = (img_rgb.astype(np.float32) - mean) / std
    
    # HWC to CHW
    img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
    
    device = next(model.parameters()).device
    img_t = img_t.to(device)
    
    with torch.no_grad():
        out = model(img_t)
        
    # Resize output back to original image size if needed
    h, w = img_rgb.shape[:2]
    if out.shape[2:] != (h, w):
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred

def build_fcn(num_classes=19, checkpoint=None, device='cuda:0'):
    norm_cfg = dict(type='BN', requires_grad=True)
    backbone = ResNetV1c(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    )
    decode_head = FCNHead(
        in_channels=2048,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=num_classes,
        in_index=3
    )
    model = EncoderDecoder(backbone, decode_head, align_corners=False)
    return _load_checkpoint(model, checkpoint, device)

def build_deeplabv3(num_classes=19, checkpoint=None, device='cuda:0'):
    norm_cfg = dict(type='BN', requires_grad=True)
    backbone = ResNetV1c(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    )
    decode_head = ASPPHead(
        in_channels=2048,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=num_classes,
        in_index=3
    )
    model = EncoderDecoder(backbone, decode_head, align_corners=False)
    return _load_checkpoint(model, checkpoint, device)

def build_segformer(num_classes=19, checkpoint=None, device='cuda:0'):
    backbone = MixVisionTransformer(
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[2, 2, 2, 2], # for MiT-B1
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    )
    decode_head = SegformerHead(
        in_channels=[64, 128, 320, 512],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        in_index=(0, 1, 2, 3)
    )
    model = EncoderDecoder(backbone, decode_head, align_corners=False)
    return _load_checkpoint(model, checkpoint, device)
