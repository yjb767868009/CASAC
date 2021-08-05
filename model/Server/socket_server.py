import os
from datetime import datetime

import numpy as np
import torch

from model.bert.model import Model
from model.utils import data2gpu
from model.utils.data_preprocess import get_norm
import socket
import sys
import struct


class Server():
    def __init__(self):
        self.write_file = False
        self.model_path = 'E:/NSM/trained20210805/'
        self.model = Model(None, 0.0001)
        self.model.load_param(self.model_path)
        self.data = torch.empty(0, 5307)
        self.full = False
        self.input_mean, self.input_std = get_norm("E:/NSM/data/InputNorm.txt")
        self.output_mean, self.output_std = get_norm("E:/NSM/data/OutputNorm.txt")
        if self.write_file:
            self.input_file = open('E:/NSM/our_data/tmp/Input.txt', 'w')
            self.output_file = open('E:/NSM/our_data/tmp/Output.txt', 'w')
        self.index = 0
        self.data_len = 2

    def predict(self, x):
        if self.write_file:
            x_str = ""
            for i in x:
                x_str += str(i) + " "
            self.input_file.write(x_str[:-1] + '\n')

        x = torch.tensor(x)
        x = (x - self.input_mean) / self.input_std
        x = x.unsqueeze(0)

        if not self.full:
            self.data = x.repeat(self.data_len, 1)
            self.full = True
        else:
            self.data = torch.cat((self.data, x), 0)
            self.data = self.data[1:]

        t = data2gpu(self.data.unsqueeze(0))
        data = self.model.forward(t)
        data = data.mean(dim=1)[0].cpu().detach()
        data = data * self.output_std + self.output_mean
        data = data.numpy().tolist()

        if self.write_file:
            data_str = ""
            for i in data:
                data_str += str(i) + " "
            self.output_file.write(data_str[:-1] + '\n')

        return data

    def startServer(self):
        address = ('127.0.0.1', 31500)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(address)
        s.listen(5)
        print('start')
        while True:
            ss, addr = s.accept()
            self.full = False
            self.data = torch.empty(0, 5307)
            self.index = 0
            print('got connected from', addr)
            while True:
                try:
                    ra = ss.recv(40960)
                    input = list(struct.unpack('5307f', ra))

                    output = self.predict(input)

                    output = struct.pack('618f', *output)

                    ss.send(output)
                except Exception as e:
                    print(e)
                    break
            print('close')
            ss.close()
        s.close()


if __name__ == '__main__':
    s = Server()
    s.startServer()
