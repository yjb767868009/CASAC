import random

import torch
import torch.utils.data as tordata
import numpy as np
import os
from tqdm import tqdm

from model.utils.ReadDataProcess import readData


class DataSet(tordata.Dataset):

    def __init__(self, input_dir, label_dir, cache, ):
        self.input_data_dir = input_dir
        assert os.path.exists(self.input_data_dir), 'No Input dir'
        self.label_data_dir = label_dir
        assert os.path.exists(self.label_data_dir), 'No label dir'

        self.data_size = len(os.listdir(self.input_data_dir))
        assert self.data_size == len(os.listdir(self.label_data_dir)), 'input size != label size'

        self.input_data = [None for _ in range(self.data_size + 1)]
        self.input_data_random = [None for _ in range(self.data_size + 1)]
        self.label_data = [None for _ in range(self.data_size + 1)]

        self.cache = cache

    def load_data(self, index):
        return self.__getitem__(index)

    def load_all_data(self):
        self.input_data = readData(self.input_data_dir, worker_nums=5)
        self.label_data = readData(self.label_data_dir, worker_nums=5)
        self.random_all_data(self.input_data)

    def __len__(self):
        return self.data_size

    def __loader__(self, path):
        return torch.load(os.path.join(path))

    def __getitem__(self, item):
        if not self.cache:
            input_data = self.__loader__(os.path.join(self.input_data_dir, str(item) + '.txt'))
            label_data = self.__loader__(os.path.join(self.label_data_dir, str(item) + '.txt'))
            input_data_random = self.random_word(input_data)
        elif self.input_data[item] is None or self.label_data[item] is None:
            input_data = self.__loader__(os.path.join(self.input_data_dir, str(item) + '.txt'))
            label_data = self.__loader__(os.path.join(self.label_data_dir, str(item) + '.txt'))
            input_data_random = self.random_word(input_data)
            self.input_data[item] = input_data
            self.input_data_random[item] = input_data_random
            self.label_data[item] = label_data
        else:
            input_data = self.input_data[item]
            input_data_random = self.input_data_random[item]
            label_data = self.label_data[item]
        return input_data, input_data_random, label_data

    def random_word(self, data):
        for i in range(data.size(0)):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    data[i] = torch.zeros(5307)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    data[i] = data[random.randrange(data.size(0))]
        return data

    def random_all_data(self, data):
        for i in range(self.data_size):
            self.input_data_random[i] = self.random_word(data[i])
