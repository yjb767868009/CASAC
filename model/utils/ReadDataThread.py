import os
import math
import numpy as np
import torch
import threading

input_data = []


def not_none_num():
    nums = 0
    for i in range(data_size):
        nums += 0 if input_data[i] is None else 1
    return nums


class ReadDataThread(threading.Thread):
    def __init__(self, threadID, name, start_index: int, end_index: int, data_root):
        super().__init__()
        self.threadId = threadID
        self.name = name
        self.start_index = start_index
        self.end_index = end_index
        self.data_root = data_root

    def run(self):
        print("thread id " + str(self.threadId) + " start")
        for index in range(self.start_index, self.end_index):
            input_data_line = torch.FloatTensor(np.float32(np.loadtxt(os.path.join(data_root, "%s.txt" % i))))
            input_data[index] = input_data_line
            print("thread input not none: " + str(not_none_num()))


if __name__ == '__main__':
    data_root = "E:/NSM/data3/Train/Input"
    input_nums = 0
    thread_nums = 10
    # data_size = len(os.listdir(data_root))
    data_size = 100
    gap = math.ceil(data_size / thread_nums)
    input_data = [None for _ in range(data_size + 1)]
    threads = []
    for i in range(thread_nums):
        if i == thread_nums - 1:
            threads.append(ReadDataThread(i, "Thread-%s" % i, i * gap, data_size, data_root))
        else:
            threads.append(ReadDataThread(i, "Thread-%s" % i, i * gap, (i + 1) * gap, data_root))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
