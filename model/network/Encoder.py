import os
import torch
import torch.nn as nn

from ..utils.activation_layer import activation_layer


class Encoder(torch.nn.Module):
    def __init__(self, encoder_dims, encoder_activations, encoder_dropout):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.encoder_activations = encoder_activations
        self.encoder_dropout = encoder_dropout
        self.layer_nums = len(encoder_dims) - 1

        self.layer1 = nn.Sequential(nn.Dropout(encoder_dropout, ),
                                    nn.Linear(encoder_dims[0], encoder_dims[1]),
                                    activation_layer(encoder_activations[0]))
        self.layer2 = nn.Sequential(nn.Dropout(encoder_dropout, ),
                                    nn.Linear(encoder_dims[1], encoder_dims[2]),
                                    activation_layer(encoder_activations[1]))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
