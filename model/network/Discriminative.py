import os
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from ..utils.activation_layer import activation_layer


class Discriminative(nn.Module):
    def __init__(self, discriminative_dims, discriminative_activations, discriminative_dropout):
        super().__init__()
        self.lstm_hidden_size = discriminative_dims[2]
        self.fc1 = nn.Sequential(nn.Linear(discriminative_dims[0], discriminative_dims[1]),
                                 activation_layer(discriminative_activations[0]))
        self.lstm = nn.LSTM(discriminative_dims[1], discriminative_dims[2], num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc2 = nn.Sequential(nn.Linear(discriminative_dims[2] * 2, discriminative_dims[3]),
                                 activation_layer(discriminative_activations[1]))
        self.fc3 = nn.Sequential(nn.Linear(discriminative_dims[3], discriminative_dims[4]),
                                 activation_layer(discriminative_activations[2]))

    def forward(self, x, x_length):
        x = self.fc1(x)
        x = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True)
        x, _ = self.lstm(x)
        x, x_length = rnn_utils.pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.squeeze(-1)
        if torch.cuda.is_available():
            x_length = x_length.cuda()
        x_length = x_length - 1
        x_length = x_length.unsqueeze(1)
        z = torch.gather(x, 1, x_length)
        return z
