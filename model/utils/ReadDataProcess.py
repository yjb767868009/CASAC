import multiprocessing
import os
from datetime import datetime

import numpy as np
import torch


def not_none_num(input_data):
    nums = 0
    for x in input_data:
        nums += 0 if x is None else 1
    return nums


def read_data(input_data, start_index: int, end_index: int, data_root):
    # print("thread id " + str(start_index) + " start")
    for index in range(start_index, end_index):
        input_data_line = torch.load(os.path.join(data_root, "%s.pth" % index))
        input_data[index] = input_data_line


def readData(data_root, worker_nums):
    data_size = len(os.listdir(data_root))
    input_data = multiprocessing.Manager().list([None for _ in range(data_size + 1)])
    gap = round(data_size / worker_nums)
    processes = []
    for i in range(worker_nums):
        if i == worker_nums - 1:
            processes.append(
                multiprocessing.Process(target=read_data, args=(input_data, i * gap, data_size, data_root)))
        else:
            processes.append(
                multiprocessing.Process(target=read_data, args=(input_data, i * gap, (i + 1) * gap, data_root)))
    # t1 = datetime.now()
    for thread in processes:
        thread.start()
    for thread in processes:
        thread.join()
    # t2 = datetime.now()
    # print(t2 - t1)
    # print(not_none_num(input_data))
    print("Finish Load Data")
    return input_data


if __name__ == '__main__':
    data_root = "E:/NSM/data3/Train/Input"
    readData(data_root,5)
