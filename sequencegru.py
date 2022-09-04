import torch
from torch import nn
from spectral_norm import SpectralNorm
from gblock import GBlock
from gblockup import GBlockUp


class SequenceGRU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = SpectralNorm(nn.Conv2d(input_size, input_size, kernel_size=3, padding=1, stride=1))
        self.GBlock = GBlock(input_size, input_size)
        self.GBlockUp = GBlockUp(input_size, input_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.GBlock(x)
        out = self.GBlockUp(x)
        return out
