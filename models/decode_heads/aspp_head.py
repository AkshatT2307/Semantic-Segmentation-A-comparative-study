import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

class ASPPModule(nn.ModuleList):
    def __init__(self, dilations, in_channels, channels, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.norm_cfg = norm_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    norm_cfg=self.norm_cfg))

    def forward(self, x):
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))
        return aspp_outs

class ASPPHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, dilations=(1, 12, 24, 36), dropout_ratio=0.1, in_index=-1):
        super().__init__()
        self.in_index = in_index
        self.num_classes = num_classes
        norm_cfg = dict(type='BN', requires_grad=True)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(in_channels, channels, 1, norm_cfg=norm_cfg))

        self.aspp_modules = ASPPModule(dilations, in_channels, channels, norm_cfg=norm_cfg)

        self.bottleneck = ConvModule(
            (len(dilations) + 1) * channels,
            channels,
            3,
            padding=1,
            norm_cfg=norm_cfg)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs[self.in_index]
        aspp_outs = [
            F.interpolate(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=False)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        
        if self.dropout is not None:
            output = self.dropout(output)
            
        return self.conv_seg(output)
