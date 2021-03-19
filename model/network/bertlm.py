import torch.nn as nn

from .bert import BERT
import torch.nn.utils.rnn as rnn_utils


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, input_dim=1280, hidden=1024, dropout=0.1):
        """
        :param bert: BERT model which should be trained
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(input_dim, hidden, 618, dropout)
        self.mask_lm = MaskedLanguageModel(input_dim, hidden, 5307, dropout)

    def forward(self, x, data_length, train=True):
        x = self.bert(x, data_length)
        if train:
            return self.mask_lm(x, data_length)
        else:
            return self.next_sentence(x, data_length)


class NextSentencePrediction(nn.Module):

    def __init__(self, input_dim, hidden, output_dim, dropout):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.start_layer = nn.Sequential(nn.Linear(input_dim, hidden), nn.ELU(), )
        # self.lstm1 = nn.LSTM(hidden, hidden, dropout=dropout, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden, hidden, dropout=dropout, batch_first=True)
        self.end_layer = nn.Sequential(nn.Linear(hidden, output_dim))

    def forward(self, x, x_length):
        x = self.start_layer(x)
        # x = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True)
        # x, (h_1, c_1) = self.lstm1(x)
        # x, (h_1, c_1) = self.lstm2(x)
        # x, x_length = rnn_utils.pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = self.end_layer(x)
        return x


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, input_dim, hidden, output_dim, dropout):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.start_layer = nn.Sequential(nn.Linear(input_dim, hidden), nn.ELU(), )
        # self.lstm1 = nn.LSTM(hidden, hidden, dropout=dropout, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden, hidden, dropout=dropout, batch_first=True)
        self.end_layer = nn.Sequential(nn.Linear(hidden, output_dim), )

    def forward(self, x, x_length):
        x = self.start_layer(x)
        # x = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True)
        # x, (h_1, c_1) = self.lstm1(x)
        # x, (h_1, c_1) = self.lstm2(x)
        # x, x_length = rnn_utils.pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = self.end_layer(x)
        return x
