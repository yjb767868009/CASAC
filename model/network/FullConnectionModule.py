import torch.nn as nn

from model.utils import activation_layer


class FullConnectionModule(nn.Module):
    def __init__(self, n_layers, dims, activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(n_layers):
            self.layers.append(nn.Sequential(nn.Dropout(dropout),
                                             nn.Linear(dims[i], dims[i + 1]),
                                             activation_layer(activation)))

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)
        return x
