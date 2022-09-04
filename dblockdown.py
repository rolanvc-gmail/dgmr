import torch
from torch import nn
from spectral_norm import SpectralNorm


class DBlockDown(nn.Module):  ### for sampler ####
    def __init__(self, in_channels, out_channels):
        super(DBlockDown, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm( nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 3,stride = 1,padding = 1))
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1))
        self.maxpool = nn.MaxPool2d(2, stride = 2, return_indices = False, ceil_mode = False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)
        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out

class DBlockDownFirst(nn.Module):  ### for sampler #### (only difference is Relu in forward ?)
    def __init__(self, in_channels, out_channels):
        super(DBlockDownFirst, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 3,stride = 1,padding = 1))
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1))
        self.maxpool = nn.MaxPool2d(2, stride = 2, return_indices = False, ceil_mode = False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)
        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out
