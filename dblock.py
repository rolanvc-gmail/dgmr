import torch
from torch import nn
from spectral_norm import SpectralNorm


class DBlock(nn.Module):  ### for both spatial and temporal discriminators
    def __init__(self, in_channels, out_channels):
        super(DBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(x)
        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)
        out = x1 + x2
        return out
