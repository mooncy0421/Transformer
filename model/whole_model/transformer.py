"""
Author : Chiyeong Heo
Date : 2021-08-17
"""

import torch
from torch import nn

from model.whole_model.encoder_stack import EncoderStack
from model.whole_model.decoder_stack import DecoderStack

class Transformer(nn.Module):
    def __init__(self, pad, enc_voc_size, dec_voc_size, d_model, max_seq_len, drop_rate, n_heads, d_hidden, n_layers, device):
        super(Transformer, self).__init__()

        self.pad = pad
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

    def forward(self, src, tgt):
        """
        src, tgt : Sequence, not a word embedding made by TransformerEmbedding.
                   src = [batch_size, src_seq_len]
                   tgt = [batch_size, tgt_seq_len]
        """
        
        # Making mask
        src_pad = self.make_padding(src, src)                                   # enc mask
        src_tgt_pad = self.make_apdding(src, tgt)                               # enc-dec mask
        tgt_mask = self.make_cheat_mask(tgt,tgt) & self.make_padding(tgt, tgt)  # dec mask : no-peeking mask + padding mask

        # Encoder - Decoder
        enc_res = self.encoder(src, src_pad)
        dec_res = self.decoder(enc_res, tgt, src_tgt_pad, tgt_mask)
        
        return dec_res


    def make_padding(self, Q: torch.Tensor, K: torch.Tensor):
        # Q, K : [batch_size, seq_len]
        len_q, len_k = Q.size(1), K.size(1)

        k_pad = K.ne(self.pad).unsqueeze(1).unsqueeze(2)    # k_pad : [batch_size, 1, 1, len_k]
        k_pad = k_pad.repeat(1, 1, len_q, 1)                # k_pad : [batch_size, 1, len_q, len_k]

        q_pad = Q.ne(self.pad).unsqueeze(1).unsqueeze(3)    # q_pad : [batch_size, 1, len_q, 1]
        q_pad = q_pad.repeat(1, 1, 1, len_k)                # q_pad : [batch_size, 1, len_q, len_k]

        padding_mask = k_pad & q_pad                        # padding_mask : [batch_size, 1, len_q, len_k]

        return padding_mask


    def make_cheat_mask(self, Q, K):
        # Q, K : [batch_size, seq_len]
        len_q, len_k = Q.size(0), K.size(0)

        cheat_mask =  torch.ones(len_q, len_k).tril().type(torch.BoolTensor).to(self.device)  # cheat_mask : [len_q, len_k]

        return cheat_mask