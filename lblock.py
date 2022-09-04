import torch
from torch import nn

class LBlock(nn.Module):  ### for latent conditioning
    def __init__(self, in_channels, out_channels):
        super(LBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels,out_channels-in_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)

    def forward(self, x):
        x1 = self.relu(x)
        x1 = self.conv3_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv3_2(x1)
        x2 = self.conv1(x)
        x3 = x
        x23 = torch.cat([x2,x3],axis = 1)
        out = x1 + x23
        return out
