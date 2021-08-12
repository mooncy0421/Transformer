"""
Author : Chiyeong Heo
Date : 2021-08-12
"""

import math
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, Q, K, V, mask=None):
        # For Multi-Head + Batch => 4D Tensor (Q,K,V)
        # [batch_size, num_haeds, seq_len, d_vector]
        batch_size, num_heads, seq_len, d_vector = K.size()

        K_T = K.view(batch_size, num_heads, d_vector, seq_len)      # Transpose K
        attn_weight = (Q @ K_T) / math.sqrt(d_vector)        # Scaled Dot-Product

        # Masking
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask==0, -1e-12)

        attn_weight = self.softmax(attn_weight)       # [0,1] range

        V = attn_weight @ V     # Weighted Sum of Value

        return V, attn_weight
        
