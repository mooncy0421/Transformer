"""
Author : Chiyeong Heo
Reference : http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
"""
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, device):

        super(PositionalEncoding, self).__init__()

        self.PE = torch.zeros(max_seq_len, d_model, device)
        self.PE.requires_grad = False
        pos = torch.arange(0, max_seq_len, device=device).float().unsqueeze(1)
        PE_step = torch.arange(0, d_model, step = 2, device=device).float()

        # Compute Sinusoid Positional Encoding
        self.PE[:,0::2] = torch.sin(pos/(10000**(PE_step/d_model)))
        self.PE[:,1::2] = torch.cos(pos/(10000**(PE_step/d_model)))

    def forward(self, x):       # x : batch (batch_size, seq_len)
        return self.PE[:x.size(1), :]   # PE를 seq_len 만큼만 잘라서 return