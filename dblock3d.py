import torch
from torch import nn
from spectral_norm import SpectralNorm


class DBlock3D_1(nn.Module):  ### for temporal discriminators
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_1, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)
        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2
        return out


class DBlock3D_2(nn.Module):  ### for temporal discriminators ### (only difference is Relu in forward ?)
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_2, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)
        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2
        return out
