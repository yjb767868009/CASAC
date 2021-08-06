import torch.nn as nn
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
