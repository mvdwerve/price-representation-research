import torch
import torch.nn as nn

from GreedyInfoMax.utils import utils


class Autoregressor(nn.Module):
    def __init__(self, opt, input_size, hidden_dim):
        super(Autoregressor, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=self.input_size, hidden_size=self.hidden_dim, batch_first=True
        )

        self.opt = opt

    def forward(self, input):

        cur_device = utils.get_device(self.opt, input)

        regress_hidden_state = torch.zeros(
            1, input.size(0), self.hidden_dim, device=cur_device
        )
        self.gru.flatten_parameters()
        output, regress_hidden_state = self.gru(input, regress_hidden_state)

        return output
