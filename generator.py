import torch
from torch import nn
from conditioningstack import conditioningStack
from lcstack import LCStack
from outputstack import outputStack
from convgru import ConvGRU


class Generator(nn.Module):
    def __init__(self, input_channel):
        super(Generator, self).__init__()
        self.conditioningStack = conditioningStack(input_channel)
        self.LCStack = LCStack()
        self.ConvGRU = ConvGRU(x_dim=[768, 384, 192, 96],
                               h_dim=[384, 192, 96, 48],
                               kernel_sizes=3,
                               num_layers=4,
                               gb_hidden_size=[384, 192, 96, 48])
        self.outputStack = outputStack()

    def forward(self, cd_input, lcs_input):
        cd_input = torch.unsqueeze(cd_input, 2)
        lcs_output = self.LCStack(lcs_input)
        cd_output = self.conditioningStack(cd_input)
        cd_output.reverse()  # to make the largest first
        lcs_output = torch.unsqueeze(lcs_output, 1)
        lcs_outputs = [lcs_output] * 18

        for i in range(len(lcs_outputs)):
            if i == 0:
                lcs_outputs_data = lcs_outputs[i]
            else:
                lcs_outputs_data = torch.cat((lcs_outputs_data, lcs_outputs[i]), 1)  # create list of Z from latent conditioning stack

        gru_output = self.ConvGRU(lcs_outputs_data, cd_output)

        for i in range(gru_output.shape[1]):
            out = gru_output[:, i]
            out = self.outputStack(out)
            if i == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), dim=1)
        return pred
