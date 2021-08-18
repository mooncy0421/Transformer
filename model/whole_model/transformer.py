"""
Author : Chiyeong Heo
Date : 2021-08-17
"""

import torch
from torch import nn

from model.whole_model.encoder_stack import EncoderStack
from model.whole_model.decoder_stack import DecoderStack

class Transformer(nn.Module):
    def __init__(self, enc_voc_size, dec_voc_size, d_model, max_seq_len, drop_rate, n_heads, d_hidden, n_layers, device):
        super(Transformer, self).__init__()

        self.device = device

        self.encoder = EncoderStack(enc_voc_size = enc_voc_size, 
                                    d_model = d_model, 
                                    max_len = max_seq_len, 
                                    drop_rate = drop_rate, 
                                    num_heads = n_heads, 
                                    d_hidden = d_hidden, 
                                    num_layers = n_layers, 
                                    device=device)

        self.decoder = DecoderStack(dec_voc_size = dec_voc_size, 
                                    d_model = d_model, 
                                    max_len = max_seq_len, 
                                    drop_rate = drop_rate, 
                                    num_heads = n_heads, 
                                    d_hidden = d_hidden, 
                                    num_layers = n_layers, 
                                    device=device)

    def forward(self, src, tgt, enc_mask, dec_mask):
        enc_res = self.encoder(src, enc_mask)
        dec_res = self.decoder(enc_res, tgt, enc_mask, dec_mask)


    def make_padding(self, Q, K):
        len_q = Q.size(0), K.size(0)

    def make_cheat_mask(self, Q, K):
        len_q, len_k = Q.size(0), K.size(0)

        mask=  torch.ones(len_q, len_k).tril().type(torch.BoolTensor).to(self.device)

        return mask