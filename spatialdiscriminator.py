import torch
from torch import nn
from dblockdown import DBlockDownFirst, DBlockDown
from dblock import DBlock
from spectral_norm import SpectralNorm
from space_to_depth import space_to_depth


class spaDiscriminator(nn.Module):
    def __init__(self, ):
        super(spaDiscriminator, self).__init__()
        self.DBlockDown_1 = DBlockDownFirst(4, 48)
        self.DBlockDown_2 = DBlockDown(48, 96)
        self.DBlockDown_3 = DBlockDown(96, 192)
        self.DBlockDown_4 = DBlockDown(192, 384)
        self.DBlockDown_5 = DBlockDown(384, 768)
        self.DBlock_6 = DBlock(768, 768)
        self.sum_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)  # I ADDED THIS
        self.linear = SpectralNorm(nn.Linear(in_features=768 * 1 * 1, out_features=1))

    def forward(self, x):
        x = self.avgPool(x)  # used avg pool instead of random crop sampling

        for i in range(x.shape[1]):
            x_temp = x[:, i]
            x_temp = x_temp.view(x_temp.shape[0], 1, x_temp.shape[1], x_temp.shape[2])
            x_temp = space_to_depth(x_temp, 2)
            x_temp = torch.squeeze(x_temp)
            x_temp = self.DBlockDown_1(x_temp)
            x_temp = self.DBlockDown_2(x_temp)
            x_temp = self.DBlockDown_3(x_temp)
            x_temp = self.DBlockDown_4(x_temp)
            x_temp = self.DBlockDown_5(x_temp)
            x_temp = self.DBlock_6(x_temp)
            x_temp = self.sum_pool(x_temp)
            x_temp = x_temp.view(x_temp.shape[0], x_temp.shape[1])
            x_temp = x_temp * 4
            out = self.linear(x_temp)

            if i == 0:
                data = out
            else:
                data = data + out

        data = torch.squeeze((data))
        return data
