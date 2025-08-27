# coding=utf-8

import torch
import torch.nn as nn
from torch import Tensor

# Base block
# Partial Convolution
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # For training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

# CAM Block
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# Light CNN designed
class lcnn(nn.Module):
    def __init__(self, in_channels):
        super(lcnn, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1, stride=1, ),  # The in_channels should be changed
            # According to the number of bands in the image, e.g. in_channels=8 when the image has four bands.
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, stride=1, ),  # The in_channels change
            # is the same as above.
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.PConv = nn.Sequential(
            Partial_conv3(dim=64, n_div=4, forward='split_cat'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.pool1 = nn.Sequential(nn.AvgPool2d(kernel_size=2, ))
        self.cam = ChannelAttention(in_planes=64, ratio=16)
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.FCN = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)
        x_conv12 = torch.cat([x_conv1, x_conv2], dim=1)
        x_conv3 = self.conv3(x_conv12)
        x_pc1 = self.PConv(x_conv3)
        x_pool1 = self.pool1(x_pc1)
        x_cam = self.cam(x_pool1) * x_pool1
        x_pc2 = self.PConv(x_cam)
        x_conv4 = self.conv4(x_pc2)
        x_gap = self.GAP(x_conv4)
        x_gap = x_gap.view(x_gap.size(0), -1)
        out = self.FCN(x_gap)
        return out