"""
Author : Chiyeong Heo
Date : 2021-08-13
"""

from torch.nn.modules import dropout
from model.layer.layernorm import LayerNormalization
from model.layer.multi_head_attention import MultiHeadAttention
from model.layer.positionwise_feedforward import PositionwiseFeedforward

import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, drop_rate):
        super(Encoder, self).__init__()

        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.lnorm1 = LayerNormalization(d_model = d_model)
        self.dropout1 = nn.Dropout(p=drop_rate)

        self.ffn = PositionwiseFeedforward(d_model=d_model, d_hidden=d_hidden, drop_rate=drop_rate)
        self.lnorm2 = LayerNormalization(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_rate)

    def forward(self, x, pad_mask):
        # x : Positional Encoding + Embedding
        # Multi-Head Attetion -> Add&Norm
        x_ = x
        x = self.attn(Q=x, K=x, V=x, mask=pad_mask)
        x = self.lnorm1(x + x_)
        x = self.dropout1(x)

        # Feed Forward -> Add&Norm
        x_ = x
        x = self.ffn(x)
        x = self.lnorm2(x + x_)
        x = self.dropout2(x)

        return x