import torch
from torch import nn
from convgrucell import ConvGRUCell
from sequencegru import SequenceGRU


class ConvGRU(nn.Module):
    def __init__(self, x_dim, h_dim, kernel_sizes, num_layers, gb_hidden_size):  # ls_dim is [768, 384, 192, 96]; cs_dim is [384, 192, 96, 48]
        super().__init__()

        if type(x_dim) != list:
            self.x_dim = [x_dim] * num_layers
        else:
            assert len(x_dim) == num_layers
            self.x_dim = x_dim

        if type(h_dim) != list:
            self.h_dim = [h_dim] * num_layers
        else:
            assert len(h_dim) == num_layers
            self.h_dim = h_dim

        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes] * num_layers
        else:
            assert len(kernel_sizes) == num_layers
            self.kernel_sizes = kernel_sizes

        self.n_layers = num_layers  # 4 layers
        cells = nn.ModuleList()
        squenceCells = nn.ModuleList()

        for i in range(self.n_layers):
            cell = ConvGRUCell(self.x_dim[i], self.h_dim[i], 3)
            cells.append(cell)
        self.cells = cells

        for i in range(self.n_layers):
            squenceCell = SequenceGRU(gb_hidden_size[i])
            squenceCells.append(squenceCell)
        self.squenceCells = squenceCells

    def forward(self, x, h):  # x is from latent conditioning stack, h is from conditioning stack
        seq_len = x.size(1)  # 18
        prev_state_list = []

        for t in range(seq_len):
            if t == 0:
                curr_state_list = []
                for layer_idx in range(self.n_layers):
                    cell = self.cells[layer_idx]
                    h_input = h[layer_idx]  # hidden is from conditioning stack: bs x 384 x 8 x 8; bs x 192 x 16 x 16; bs x 96 x 32 x 32; bs x 48 x 64 x 64
                    squenceCell = self.squenceCells[layer_idx]

                    if layer_idx == 0:
                        x_input = x[:, t, :, :, :]  # a is from latent conditioning stack: bs x 1 x 768 x 8 x 8
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)
                    else:
                        x_input = upd_new_state  # get lower upscaled hidden state
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)

                upd_new_state = torch.unsqueeze(upd_new_state, dim=1)
                output = upd_new_state  # get upper output at t = 0
                prev_state_list = curr_state_list

            else:
                curr_state_list = []
                for layer_idx in range(self.n_layers):
                    cell = self.cells[layer_idx]
                    h_input = prev_state_list[layer_idx]
                    squenceCell = self.squenceCells[layer_idx]

                    if layer_idx == 0:
                        x_input = x[:, t, :, :, :]  # a is from latent conditioning stack: bs x 1 x 768 x 8 x 8
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)
                    else:
                        x_input = upd_new_state  # get lower upscaled hidden state
                        new_state = cell(x_input, h_input)
                        upd_new_state = squenceCell(new_state)
                        curr_state_list.append(new_state)

                upd_new_state = torch.unsqueeze(upd_new_state, dim=1)
                output = torch.cat((output, upd_new_state), dim=1)  # get upper output
                prev_state_list = curr_state_list
        return output
