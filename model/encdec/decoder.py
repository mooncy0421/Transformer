"""
Author : Chiyeong Heo
Date : 2021-08-16
"""

from model.layer.multi_head_attention import MultiHeadAttention
from model.layer.layernorm import LayerNormalization
from model.layer.positionwise_feedforward import PositionwiseFeedforward

import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, drop_rate):
        super(Decoder, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.lnorm1 = LayerNormalization(d_model=d_model)
        self.drop1 = nn.Dropout(p=drop_rate)

        self.encdec_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.lnorm2 = LayerNormalization(d_model=d_model)
        self.drop2 = nn.Dropout(p=drop_rate)

        self.ffn = PositionwiseFeedforward(d_model=d_model, d_hidden=d_hidden, drop_rate=drop_rate)
        self.lnorm3 = LayerNormalization(d_model=d_model)
        self.drop3 = nn.Dropout(p=drop_rate)

    def forward(self, enc_res, dec_in, pad_mask, dec_mask):
        x_ = dec_in
        x = self.self_attn(Q=dec_in, K=dec_in, V=dec_in, mask=dec_mask)
        x = self.lnorm1(x_ + x)
        x = self.drop1(x)

        if enc_res is not None:
            x_ = x
            x = self.encdec_attn(Q=enc_res, K=x, V=x, mask=pad_mask)
            x = self.lnorm2(x_ + x)
            x = self.drop2(x)

        x_ = x
        x = self.ffn(x)
        x = self.lnorm3(x_ + x)
        x = self.drop3(x)

        return x