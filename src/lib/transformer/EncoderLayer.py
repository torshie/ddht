import torch

from .HyperParam import HyperParam
from .MultiHeadAttention import MultiHeadAttention
from .PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(torch.nn.Module):
    ''' Compose with two layers '''
    def __init__(self, param: HyperParam):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(param)
        self.pos_ffn = PositionwiseFeedForward(param)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
