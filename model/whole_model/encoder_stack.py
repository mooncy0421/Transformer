"""
Author : Chiyeong Heo
Date : 2021-08-16
"""

import torch
from torch import nn

from model.encdec.encoder import Encoder
from model.embedding.TransformerEmbedding import TransformerEmbedding

class EncoderStack(nn.Module):
    def __init__(self, enc_voc_size, d_model, max_len, drop_rate, num_heads, d_hidden, num_layers, device):
        super(EncoderStack, self).__init__()

        self.emb = TransformerEmbedding(vocab_size=enc_voc_size,
                                        d_model=d_model,
                                        max_seq_len=max_len, 
                                        drop_rate=drop_rate,
                                        device=device)
        self.encoders = nn.ModuleList([Encoder(d_model=d_model,
                                               num_heads=num_heads,
                                               d_hidden=d_hidden,
                                               drop_rate=drop_rate)
                                               for _ in range(num_layers)])

    def forward(self, x, pad_mask):
        x = self.emb(x)

        for enc in self.encoders:
            x = enc(x, pad_mask)

        return x