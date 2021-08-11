"""
Author : Chiyeong Heo
Date : 2021-08-11
"""

import math
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, Q, K, V, mask=None, e=1e-12):

