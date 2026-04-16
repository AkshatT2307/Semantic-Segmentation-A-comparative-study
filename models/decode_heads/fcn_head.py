import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

class FCNHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, num_convs=2, concat_input=True, dropout_ratio=0.1, in_index=-1):
        super().__init__()
        self.in_index = in_index
        self.concat_input = concat_input
        self.num_classes = num_classes
        
        convs = []
        convs.append(
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True)))
        
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True)))
            
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
            
        if self.concat_input:
            self.conv_cat = ConvModule(
                in_channels + channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True))
            
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
            
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        # inputs is a tuple of features from the backbone
        x = inputs[self.in_index]
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
            
        if self.dropout is not None:
            output = self.dropout(output)
            
        return self.conv_seg(output)
