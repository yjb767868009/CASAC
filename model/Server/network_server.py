import os
from datetime import datetime

import numpy as np
import torch

from model.bert.model import Model
from model.utils import data2gpu
from model.utils.data_preprocess import get_norm


class Server(object):
    def __init__(self):
        model_path = 'E:/NSM/trained20210625/'
        self.model = Model(None, None, None, 1, 100, 0.0001)
        self.model.load_param(model_path)
        self.data = torch.empty(0, 5307)
        self.full = False
        self.input_mean, self.input_std = get_norm("E:/NSM/data/InputNorm.txt")
        self.output_mean, self.output_std = get_norm("E:/NSM/data/OutputNorm.txt")

    def forward(self, x):
        x = torch.tensor(x)
        x = (x - self.input_mean) / self.input_std
        x = x.unsqueeze(0)

        if self.data.size(0) == 0:
            self.data = x.repeat(10, 1)
        else:
            self.data = torch.cat((self.data, x), 0)
            self.data = self.data[1:]

        t = data2gpu(self.data.unsqueeze(0))
        data = self.model.forward(t)
        data = data[0][-1].cpu().detach()
        data = data * self.output_std + self.output_mean
        data = data.numpy().tolist()
        return data
