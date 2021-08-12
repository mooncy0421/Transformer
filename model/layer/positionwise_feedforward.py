"""
Author : Chiyeong Heo
Date : 2021-08-12
"""

from torch import nn

class PositionwiseFeedforward(nn.Module):
    def ___init__(self, d_model, d_hidden, drop_rate=0.1):
        super(PositionwiseFeedforward, self).__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.lin1 = nn.Linear(d_model, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return x