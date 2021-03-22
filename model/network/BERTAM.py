import torch.nn as nn

from .bert import BERT
import torch.nn.utils.rnn as rnn_utils


class BERTAM(nn.Module):
    """
    BERT Animation Model
    Masked Language Model
    Last Frame Prediction Model
    """

    def __init__(self, bert: BERT, input_dim=1280, hidden=1024, dropout=0.1):
        """
        :param bert: BERT model which should be trained
        """

        super().__init__()
        self.bert = bert
        self.last_frame = LastFramePrediction(input_dim, hidden, 618, dropout)
        self.mask_lm = MaskedLanguageModel(input_dim, hidden, 5307, dropout)

    def forward(self, x, data_length, train=True):
        x = self.bert(x, data_length)
        if train:
            return self.mask_lm(x, data_length)
        else:
            return self.last_frame(x, data_length)


class LastFramePrediction(nn.Module):

    def __init__(self, input_dim, hidden, output_dim=618, dropout=0.1):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.start_layer = nn.Sequential(nn.Linear(input_dim, hidden), nn.ELU(), )
        self.end_layer = nn.Sequential(nn.Linear(hidden, output_dim))

    def forward(self, x, x_length):
        x = self.start_layer(x)
        x = self.end_layer(x)
        return x


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    """

    def __init__(self, input_dim, hidden, output_dim=5307, dropout=0.1):
        """
        :param hidden: output size of BERT model
        """
        super().__init__()
        self.start_layer = nn.Sequential(nn.Linear(input_dim, hidden), nn.ELU(), )
        self.end_layer = nn.Sequential(nn.Linear(hidden, output_dim), )

    def forward(self, x, x_length):
        x = self.start_layer(x)
        x = self.end_layer(x)
        return x
