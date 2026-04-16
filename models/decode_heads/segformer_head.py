import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

class SegformerHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, dropout_ratio=0.1, in_index=(0, 1, 2, 3)):
        super().__init__()
        self.in_index = in_index
        self.num_classes = num_classes
        self.channels = channels
        
        self.convs = nn.ModuleList()
        for in_chan in in_channels:
            self.convs.append(
                ConvModule(
                    in_channels=in_chan,
                    out_channels=channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=dict(type='BN', requires_grad=True)))
            
        self.fusion_conv = ConvModule(
            in_channels=channels * len(in_channels),
            out_channels=channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
            
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        # inputs should be out from transformer backbone
        x = [inputs[i] for i in self.in_index]
        
        outs = []
        for i, (conv, feat) in enumerate(zip(self.convs, x)):
            feat_out = conv(feat)
            if i != 0:
                feat_out = F.interpolate(
                    feat_out,
                    size=x[0].shape[2:],
                    mode='bilinear',
                    align_corners=False)
            outs.append(feat_out)

        out = self.fusion_conv(torch.cat(outs, dim=1))
        
        if self.dropout is not None:
            out = self.dropout(out)
            
        return self.conv_seg(out)
