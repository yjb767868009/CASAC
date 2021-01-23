import os

import numpy as np
import torch

from model.utils.data_preprocess import get_norm
from model.utils.initialization import initialization


class Server(object):
    def __init__(self, model_path):
        self.model = initialization(100, 1, None, model_path, model_path, train=False, unity=True)
        self.model.load_param()
        self.data = torch.empty(0, 5307)
        self.full = False
        self.input_mean, self.input_std = get_norm("E:/NSM/data2/InputNorm.txt")
        self.output_mean, self.output_std = get_norm("E:/NSM/data2/OutputNorm.txt")

    def forward(self, x):
        x = np.array(x)
        x = (x - self.input_mean) / self.input_std
        x = torch.FloatTensor([x])
        self.data = torch.cat((self.data, x), 0)
        if self.full is True:
            self.data = self.data[1:]
            data_length = 100
        else:
            data_length = self.data.size(0)
            if data_length >= 100:
                self.full = True
        data = self.model.forward(self.data.unsqueeze(0), [data_length], train=False)
        data = data[0][-1].cpu().detach().numpy()
        data = data * self.output_std + self.output_mean
        return data.tolist()
