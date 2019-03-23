import torch
import torch.nn.functional

from .HyperParam import HyperParam


class PositionwiseFeedForward(torch.nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, param: HyperParam):
        super().__init__()
        d_in = param.model_dimension
        d_hid = param.hidden_layer_dimension
        self.w_1 = torch.nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = torch.nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = torch.nn.LayerNorm(d_in)
        self.dropout = torch.nn.Dropout(param.dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(torch.nn.functional.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
