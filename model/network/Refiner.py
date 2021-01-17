import os
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from ..utils.activation_layer import activation_layer


class Refiner(nn.Module):
    """
    Refiner Posture,Trajectory,Goal
    """

    def __init__(self, refiner_dims, refiner_activations, refiner_dropout):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Dropout(refiner_dropout),
                                 nn.Linear(refiner_dims[0], refiner_dims[1]),
                                 activation_layer(refiner_activations[0]), )
        self.lstm = nn.LSTM(refiner_dims[1], refiner_dims[2], batch_first=True, )
        self.fc2 = nn.Sequential(nn.Dropout(refiner_dropout),
                                 nn.Linear(refiner_dims[2], refiner_dims[3]),
                                 activation_layer(refiner_activations[1]))

    def forward(self, x, x_length):
        x = self.fc1(x)
        x = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True)
        x, _ = self.lstm(x)
        x, x_length = rnn_utils.pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = self.fc2(x)
        return x

    def save_network(self, save_path):
        pass
