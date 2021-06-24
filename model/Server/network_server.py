import os
from datetime import datetime

import numpy as np
import torch
import csv

from model.utils.data_preprocess import get_norm
from model.utils.initialization import initialization


class Server(object):
    def __init__(self):
        model_path = 'E:/NSM/trained20210525/'
        self.model = initialization(100, 1, None, model_path, model_path, train=False, unity=True)
        self.model.load_param(model_path)
        self.model.bert_prediction.test_init()
        self.data = torch.empty(0, 5307)
        self.full = False
        self.input_mean, self.input_std = get_norm("E:/NSM/data/InputNorm.txt")
        self.output_mean, self.output_std = get_norm("E:/NSM/data/OutputNorm.txt")
        # self.input_writer = csv.writer(open('Input.csv', 'w', newline=""))
        # self.input_writer.writerow([i for i in range(5307)])
        # self.output_writer = csv.writer(open('Output.csv', 'w', newline=""))
        # self.output_writer.writerow([i for i in range(618)])

    def forward(self, x):
        x = torch.tensor(x)
        # self.input_writer.writerow(x.numpy().tolist())
        x = (x - self.input_mean) / self.input_std
        x = x.unsqueeze(0)

        if self.data.size(0) == 0:
            self.data = x.repeat(10, 1)
        else:
            self.data = torch.cat((self.data, x), 0)
            self.data = self.data[1:]
        # t1=datetime.now()
        # data = self.model.forward(self.data.unsqueeze(0), [data_length])
        data, _ = self.model.forward(self.data.unsqueeze(0))
        # t2=datetime.now()
        # print(t2-t1)
        data = data[0][-1].cpu().detach()
        data = data * self.output_std + self.output_mean
        data = data.numpy().tolist()
        #self.output_writer.writerow(data)
        return data
