import torch
from torch import nn
from lblock import LBlock


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        mask = self.sigmoid(x)
        return mask


class LCStack(nn.Module):
    def __init__(self, ):
        super(LCStack, self).__init__()
        self.conv3_1 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.LBlock_1 = LBlock(8, 24)
        self.LBlock_2 = LBlock(24, 48)
        self.LBlock_3 = LBlock(48, 192)
        self.LBlock_4 = LBlock(192, 768)
        self.mask = SpatialAttention()

    def forward(self, x):
        x = self.conv3_1(x)
        x = self.LBlock_1(x)
        x = self.LBlock_2(x)
        x = self.LBlock_3(x)
        mask = self.mask(x)
        x = x * mask
        out = self.LBlock_4(x)
        return out

