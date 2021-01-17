import torch
import torch.nn as nn

from model.network import Encoder


class Embedding(nn.Module):
    def __init__(self, n_layers, dims, activations, dropout, segmentation):
        super().__init__()
        self.n_layer = n_layers
        self.segmentation = segmentation
        self.encoder_blocks = nn.ModuleList([Encoder(dims[i], activations[i], dropout) for i in range(n_layers)])

    def forward(self, x):
        y = []
        for i in range(self.n_layer):
            y.append(self.encoder_blocks[i].forward(x[:, :, self.segmentation[i]: self.segmentation[i + 1]]))
        y = torch.cat(y, 2)
        if torch.cuda.is_available():
            y = y.cuda()
        return y
