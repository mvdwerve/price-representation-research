import torch.nn as nn
import torch
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


class GELU(nn.Module):
    def forward(self, x):
        return gelu(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, kernel_sizes, strides, padding):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden = hidden
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding

        assert (
            len(kernel_sizes) == len(strides) == len(padding)
        ), "Inconsistent size of network parameters (kernels, strides and padding)"

        self.model = nn.Sequential()

        for idx in range(len(kernel_sizes)):
            self.model.add_module(
                "layer %d" % idx,
                self.new_block(
                    input_dim,
                    self.hidden,
                    kernel_sizes[idx],
                    strides[idx],
                    padding[idx],
                ),
            )
            input_dim = self.hidden

    def new_block(self, in_dim, out_dim, kernel_size, stride, padding):
        new_block = nn.Sequential(
            nn.Conv1d(
                in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            GELU(),
        )
        return new_block

    def forward(self, x):
        # calculate z
        z = self.model(x)

        assert (z == z).all(), "NaN in Encoder.forward output"

        return z
