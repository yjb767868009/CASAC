import torch.nn as nn

from .bert import BERT
import torch.nn.utils.rnn as rnn_utils


class BERTPredictionModel(nn.Module):
    """
    BERT Prediction Model
    Last Frame Prediction Model
    """

    def __init__(self, input_dim=1280, hidden=1024, output_dim=618, dropout=0.1):
        super().__init__()
        self.last_frame = LastFramePrediction(input_dim, hidden, output_dim, dropout)

    def forward(self, x, data_length):
        return self.last_frame(x, data_length)


class LastFramePrediction(nn.Module):

    def __init__(self, input_dim, hidden, output_dim=618, dropout=0.1):
        super().__init__()
        self.start_layer = nn.Sequential(nn.Linear(input_dim, hidden), nn.ELU(), )
        self.end_layer = nn.Sequential(nn.Linear(hidden, output_dim))

    def forward(self, x, x_length):
        x = self.start_layer(x)
        x = self.end_layer(x)
        return x

