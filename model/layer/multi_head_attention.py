"""
Author : Chiyeong Heo
Date : 2021-08-12
"""

from torch import nn
from model.layer.scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0
        
        self.d_K = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn = ScaledDotProductAttention()

        # d_K = d_V = d_model/h
        self.W_Q = nn.Linear(self.d_model, self.d_model)      # W_Q = [d_model, d_K]
        self.W_K = nn.Linear(self.d_model, self.d_model)      # W_K = [d_model, d_K]
        self.W_V = nn.Linear(self.d_model, self.d_model)      # W_V = [d_model, d_V]
        self.W_O = nn.Linear(self.d_model, self.d_model)      # W_O = [hd_V, d_model]

    def forward(self, Q, K, V, mask=None):
        # Linear Project Q, K, V respectively
        # Q, K, V = [batch_size, seq_length, d_K]
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)

        batch_size, len = Q.size(0), Q.size(1)
        # Multi-Head [batch_size, num_heads, seq_len, d_K(==d_vector)]
        Q = Q.view(batch_size, self.num_heads, len, self.d_K)
        K = K.view(batch_size, self.num_heads, len, self.d_K)
        V = V.view(batch_size, self.num_heads, len, self.d_K)

        heads, attention = self.attn(Q, K, V, mask=mask)
        heads = heads.view(batch_size, len, self.d_model)
        out = self.W_O(heads)

        return out