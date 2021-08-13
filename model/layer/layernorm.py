"""
Author : Chiyeong Heo
Date : 2021-08-12
"""

import torch
from torch import nn

class LayerNormalization(nn.Module):
    def __init__(self, d_model):
        super(LayerNormalization, self).__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std_dev = x.std(-1, keepdim=True)

        out = self.gain * (x-mean) / (std_dev + 1e-12) + self.bias  # 1e-12 for zero divide problem
        return out