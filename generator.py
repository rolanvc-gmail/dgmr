import torch
from torch import nn
from conditioningstack import conditioningStack
from lcstack import LCStack
from outputstack import outputStack
from convgru import ConvGRU

class generator(nn.Module):
    def __init__(self, input_channel):
        super(generator, self).__init__()
        self.conditioningStack = conditioningStack(input_channel)
        self.LCStack = LCStack()
        self.ConvGRU = ConvGRU(x_dim=[768, 384, 192, 96],
                               h_dim=[384, 192, 96, 48],
                               kernel_sizes=3,
                               num_layers=4,
                               gb_hidden_size=[384, 192, 96, 48])
        self.outputStack = outputStack()

    def forward(self, CD_input, LCS_input):
        CD_input = torch.unsqueeze(CD_input, 2)
        LCS_output = self.LCStack(LCS_input)
        CD_output = self.conditioningStack(CD_input)
        CD_output.reverse()  # to make the largest first
        LCS_output = torch.unsqueeze(LCS_output, 1)
        LCS_outputs = [LCS_output] * 18

        for i in range(len(LCS_outputs)):
            if i == 0:
                LCS_outputs_data = LCS_outputs[i]
            else:
                LCS_outputs_data = torch.cat((LCS_outputs_data, LCS_outputs[i]), 1)  # create list of Z from latent conditioning stack

        gru_output = self.ConvGRU(LCS_outputs_data, CD_output)

        for i in range(gru_output.shape[1]):
            out = gru_output[:, i]
            out = self.outputStack(out)
            if i == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), dim=1)
        return pred
