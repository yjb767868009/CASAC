import os
from datetime import datetime

import numpy as np
import torch

from model.bert.model import Model
from model.utils import data2gpu
from model.utils.data_preprocess import get_norm

write_file = False


class Server(object):
    def __init__(self):
        model_path = 'E:/NSM/trained20210630/'
        self.model = Model(None, None, None, 1, 100, 0.0001)
        self.model.load_param(model_path)
        self.data = torch.empty(0, 5307)
        self.full = False
        self.input_mean, self.input_std = get_norm("E:/NSM/data/InputNorm.txt")
        self.output_mean, self.output_std = get_norm("E:/NSM/data/OutputNorm.txt")
        if write_file:
            self.input_file = open('E:/NSM/our_data/tmp/Input.txt', 'w')
            self.output_file = open('E:/NSM/our_data/tmp/Output.txt', 'w')

    def forward(self, x):
        if write_file:
            x_str = ""
            for i in x:
                x_str += str(i) + " "
            self.input_file.write(x_str[:-1] + '\n')

        x = torch.tensor(x)
        x = (x - self.input_mean) / self.input_std
        x = x.unsqueeze(0)

        self.data = torch.cat((self.data, x), 0)
        if self.full is True:
            self.data = self.data[1:]
            data_length = 10
        else:
            data_length = self.data.size(0)
            if data_length >= 10:
                self.full = True

        t = data2gpu(self.data.unsqueeze(0))
        data = self.model.forward(t)
        data = data.mean(dim=1)[0].cpu().detach()
        data = data * self.output_std + self.output_mean
        data = data.numpy().tolist()

        if write_file:
            data_str = ""
            for i in data:
                data_str += str(i) + " "
            self.output_file.write(data_str[:-1] + '\n')

        return data
