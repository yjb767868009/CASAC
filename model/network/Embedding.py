import torch
import torch.nn as nn

from model.network import Encoder


class Embedding(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.n_layer = parameter["encoder_nums"]
        self.segmentation = parameter["segmentation"]
        self.encoder_blocks = nn.ModuleList(
            [Encoder(parameter["encoder_dims"][i],
                     parameter["encoder_activations"][i],
                     parameter["encoder_dropout"])
             for i in range(self.n_layer)])

    def forward(self, x):
        y = []
        for i in range(self.n_layer):
            y.append(self.encoder_blocks[i].forward(x[:, :, self.segmentation[i]: self.segmentation[i + 1]]))
        y = torch.cat(y, 2)
        if torch.cuda.is_available():
            y = y.cuda()
        return y
