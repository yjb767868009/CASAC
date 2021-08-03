import random

import torch
import torch.utils.data as tordata
import os

debug = False


class DataManager(object):
    def __init__(self, data_root, batch_size, data_size=0, data_len=10):
        self.batch_size = batch_size
        self.data_source = DataSet(data_root, data_size, data_len)

    def load_data(self):
        self.data_source.get_new_data()
        data_iter = tordata.DataLoader(
            dataset=self.data_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        return data_iter


class DataSet(tordata.Dataset):

    def __init__(self, data_root, data_size=0, data_len=10):
        super().__init__()
        self.data_len = data_len
        self.input_root = torch.load(os.path.join(data_root, "input.pth"))
        self.label_root = torch.load(os.path.join(data_root, "output.pth"))
        self.breakpoints = torch.load(os.path.join(data_root, "breakpoints.pth"))
        if debug:
            print('input', self.input_root.size(0))
            print('label', self.label_root.size(0))
            print('breakpoints', self.breakpoints)
        self.max_size = self.breakpoints[-1]
        self.input_data = []
        self.label_data = []
        if data_size == 0:
            self.data_size = self.max_size // (2 * self.data_len)

    def __len__(self):
        return self.data_size

    def get_new_data(self):
        start_list = [random.randint(0, self.max_size - self.data_len * 2) for _ in range(self.data_size)]
        start_list.sort()
        if debug:
            print('start_old', start_list)
        index = 0
        self.input_data = []
        self.label_data = []
        for i in range(self.data_size):
            if i != 0:
                while start_list[i] <= start_list[i - 1]:
                    start_list[i] += 1
                while start_list[i] < self.breakpoints[index] < start_list[i] + self.data_len:
                    start_list[i] += 1
                if start_list[i] > self.breakpoints[index]:
                    index += 1
            ir = self.input_root[start_list[i]:start_list[i] + self.data_len]
            lr = self.label_root[start_list[i]:start_list[i] + self.data_len]
            if ir.size(0) != 10 or lr.size(0) != 10:
                print(ir.size(0), lr.size(0), start_list[i])
            self.input_data.append(ir)
            self.label_data.append(lr)
        if debug:
            print('start_new', start_list)

    def __getitem__(self, item):
        input_data = self.input_data[item]
        label_data = self.label_data[item]
        return input_data, label_data


if __name__ == '__main__':
    d = DataSet("D:/NSM/data/Train", 20, 100)
    d.get_new_data()
