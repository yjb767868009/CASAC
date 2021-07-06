import multiprocessing
import os
from datetime import datetime

import numpy as np
import torch


def read_data_thread(input_data, start_index: int, end_index: int, data_root):
    """
    读取数据线程
    :param input_data: 读取到的数据位置
    :param start_index: 该线程读取数据开始位置
    :param end_index: 该线程读取数据结束位置
    :param data_root: 数据地址
    :return: None
    """
    for index in range(start_index, end_index):
        input_data_line = torch.load(os.path.join(data_root, "%s.pth" % index))
        input_data[index] = input_data_line
    print("Thead " + str(start_index) + " finish load data")


def read_data(data_root, worker_nums):
    """
    多线程读取数据集配合分片数据预处理食用
    :param data_root: 数据集位置
    :param worker_nums: 最大进程数
    :return: 数据集
    """
    data_size = len(os.listdir(data_root))
    input_data = multiprocessing.Manager().list([None for _ in range(data_size + 1)])
    gap = round(data_size / worker_nums)
    processes = []
    for i in range(worker_nums):
        if i == worker_nums - 1:
            processes.append(
                multiprocessing.Process(target=read_data_thread, args=(input_data, i * gap, data_size, data_root)))
        else:
            processes.append(
                multiprocessing.Process(target=read_data_thread, args=(input_data, i * gap, (i + 1) * gap, data_root)))
    for thread in processes:
        thread.start()
    for thread in processes:
        thread.join()
    return input_data


if __name__ == '__main__':
    data_root = "E:/NSM/data3/Train/Input"
    read_data(data_root, 5)
