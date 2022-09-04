import torch
from torch import nn

class GBlockUp(nn.Module):  ### for upsampling in GRU
    def __init__(self, in_channels, out_channels):
        super(GBlockUp, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.conv1(x1)
        x2 = self.BN(x)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.conv3_1(x2)
        x2 = self.BN(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        out = x1 + x2
        return out
