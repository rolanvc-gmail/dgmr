import torch
from torch import nn
import numpy as np
from spectral_norm import SpectralNorm


class ConvGRUCell(nn.Module):  # modified GRU cell from original code
    def __init__(self, x_dim, h_dim, kernel_size, activation=torch.sigmoid):
        super().__init__()
        padding = kernel_size // 2
        self.x_dim = x_dim  # [768, 384, 192, 96],
        self.h_dim = h_dim  # [384, 192, 96, 48]
        self.activation = activation
        self.reset_gate_x = nn.Conv2d(self.x_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # x input reset gate
        self.reset_gate_h = nn.Conv2d(self.h_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # h input reset gate
        self.update_gate_x = nn.Conv2d(self.x_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # x input update gate
        self.update_gate_h = nn.Conv2d(self.h_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # h input update gate
        self.new_gate_x = nn.Conv2d(self.x_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # x input update gate
        self.new_gate_h = nn.Conv2d(self.h_dim, self.h_dim, kernel_size, padding=padding, stride=1)  # h input update gate

        self.sqrt_k = np.sqrt(1 / self.h_dim)

        nn.init.uniform_(self.reset_gate_x.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.reset_gate_h.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.update_gate_x.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.update_gate_h.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.new_gate_x.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.uniform_(self.new_gate_h.weight, - self.sqrt_k, + self.sqrt_k)
        nn.init.constant_(self.reset_gate_x.bias, 0.)
        nn.init.constant_(self.reset_gate_h.bias, 0.)
        nn.init.constant_(self.update_gate_x.bias, 0.)
        nn.init.constant_(self.update_gate_h.bias, 0.)
        nn.init.constant_(self.new_gate_x.bias, 0.)
        nn.init.constant_(self.new_gate_h.bias, 0.)

        self.reset_gate_x = SpectralNorm(self.reset_gate_x)
        self.reset_gate_h = SpectralNorm(self.reset_gate_h)
        self.update_gate_x = SpectralNorm(self.update_gate_x)
        self.update_gate_h = SpectralNorm(self.update_gate_h)
        self.new_gate_x = SpectralNorm(self.new_gate_x)
        self.new_gate_h = SpectralNorm(self.new_gate_h)

    def forward(self, x, prev_state=None):  # prev_state: bs x 768 x 8 x 8; x : bs x 384 x 8 x 8
        if prev_state is None:
            batch_size = x.data.size()[0]  # number of samples
            spatial_size = x.data.size()[2:]  # width x height --> 8 x 8, 16 x 16, 32 x 32, 64 x 64
            state_size = [batch_size, self.cs_dim] + list(spatial_size)  # [batch size, hidden size, height, width]
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size).cuda()

        r_t = self.activation(self.reset_gate_x(x) + self.reset_gate_h(prev_state))
        z_t = self.activation(self.update_gate_x(x) + self.update_gate_h(prev_state))
        n_t = torch.tanh(self.new_gate_x(x) + (r_t * self.new_gate_h(prev_state)))
        h_t = ((1 - z_t) * n_t) + (z_t * prev_state)
        return h_t
