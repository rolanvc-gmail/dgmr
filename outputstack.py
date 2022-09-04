import torch
from torch import nn
from spectral_norm import SpectralNorm
from depth_to_space import depth_to_space


class outputStack(nn.Module):
    def __init__(self,):
        super(outputStack, self).__init__()
        self.BN = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(48, 4, 1))

    def forward(self, x):
        x = self.BN(x)
        x = self.relu(x)
        x = self.conv1(x)
        out = depth_to_space(x, 2)
        return out
