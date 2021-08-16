"""
Author : Chiyeong Heo
Date : 2021-08-16
"""

import torch
from torch import nn

from model.encdec.decoder import Decoder
from model.embedding.TransformerEmbedding import TransformerEmbedding

class DecoderStack(nn.Module):
    def __init__(self, dec_voc_size, d_model, max_len, drop_rate, num_heads, d_hidden, num_layers, device):
        super(DecoderStack, self).__init__()

        self.emb = TransformerEmbedding(vocab_size=dec_voc_size,
                                        d_model=d_model,
                                        max_seq_len=max_len,
                                        drop_rate=drop_rate,
                                        device=device)

        self.decoders = nn.ModuleList([Decoder(d_model=d_model, 
                                               num_heads=num_heads,
                                               d_hidden=d_hidden, 
                                               drop_rate=drop_rate)
                                               for _ in range(num_layers)])
        self.lin = nn.Linear(d_model, dec_voc_size)

    def forward(self, enc_in, dec_in, pad_mask, dec_mask):
        x = self.emb(dec_in)

        for dec in self.deocders:
            x = dec(enc_in, x, pad_mask, dec_mask)

        x = self.lin(x)
        
        return x
