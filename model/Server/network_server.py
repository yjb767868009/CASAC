import os
from datetime import datetime

import numpy as np
import torch
import csv

from model.utils.data_preprocess import get_norm
from model.utils.initialization import initialization


class Server(object):
    def __init__(self, model_path):
        self.model = initialization(100, 1, None, model_path, model_path, train=False, unity=True)
        self.model.load_param(model_path)
        self.data = torch.empty(0, 5307)
        self.full = False
        self.input_mean, self.input_std = get_norm("E:/NSM/data/InputNorm.txt")
        self.output_mean, self.output_std = get_norm("E:/NSM/data/OutputNorm.txt")
        # self.csv_writer = csv.writer(open('test.csv', 'w', newline=""))
        # self.csv_writer.writerow([i for i in range(618)])

    def forward(self, x):
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
        # t1=datetime.now()
        data = self.model.forward(self.data.unsqueeze(0), [data_length])
        # t2=datetime.now()
        # print(t2-t1)
        data = data[0][-1].cpu().detach()
        data = data * self.output_std + self.output_mean
        # self.csv_writer.writerow(data)
        return data.numpy().tolist()
