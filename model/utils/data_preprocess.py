import argparse
import os
import random
import shutil
import torch
import numpy as np
from tqdm import tqdm


def get_norm(file_path):
    norm = torch.tensor(np.loadtxt(file_path)).float()
    mean = norm[0]
    std = norm[1]
    for i in range(std.size(0)):
        if std[i] == 0:
            std[i] = 1
    return mean, std


def normalize(x, norm_path):
    mean, std = get_norm(norm_path)
    return (x - mean) / std


def save_data(file_path, norm_path, train_save_path, test_save_path, train_list, test_list):
    data = torch.tensor(np.loadtxt(file_path)).float()
    data = normalize(data, norm_path)
    train_data = data[train_list]
    test_data = data[test_list]
    torch.save(train_data, train_save_path)
    torch.save(test_data, test_save_path)


def data_len2breakpoint(data_len_list):
    breakpoint_list = [0]
    index = 0
    for i in data_len_list:
        index += i
        breakpoint_list.append(index)
    return breakpoint_list


def data_preprocess(data_root, output_root, scale):
    train_data_len_list = []
    test_data_len_list = []
    sequences_file = open(os.path.join(data_root, "Sequences.txt"), 'r')
    # sequences_file = open(os.path.join(data_root, "tmp.txt"), 'r')
    sequences_list = []
    i = 0
    data_index = 0
    data_len = 0
    train_list = []
    test_list = []
    test_or_train = False
    while True:
        sequences_data = sequences_file.readline()
        if sequences_data == '':
            break
        sequences_index = int(sequences_data) - 1
        test_or_train = (data_index + 1) % int(1 / scale) == 0
        if data_index != sequences_index:
            data_index += 1
            if test_or_train:
                test_data_len_list.append(data_len)
            else:
                train_data_len_list.append(data_len)
            data_len = 0
        sequences_list.append(sequences_index)
        train_list.append(not test_or_train)
        test_list.append(test_or_train)
        i += 1
        data_len += 1
    if test_or_train:
        test_data_len_list.append(data_len)
    else:
        train_data_len_list.append(data_len)

    train_breakpoint_list = data_len2breakpoint(train_data_len_list)
    test_breakpoint_list = data_len2breakpoint(test_data_len_list)

    train_dir = os.path.join(output_root, "Train")
    test_dir = os.path.join(output_root, "Test")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    save_data(os.path.join(data_root, "Input.txt"),
              os.path.join(data_root, "InputNorm.txt"),
              os.path.join(train_dir, "input.pth"),
              os.path.join(test_dir, "input.pth"),
              train_list, test_list)

    save_data(os.path.join(data_root, "Output.txt"),
              os.path.join(data_root, "OutputNorm.txt"),
              os.path.join(train_dir, "output.pth"),
              os.path.join(test_dir, "output.pth"),
              train_list, test_list)

    torch.save(train_breakpoint_list, os.path.join(train_dir, "breakpoints.pth"))
    torch.save(test_breakpoint_list, os.path.join(test_dir, "breakpoints.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--data_root", type=str, help="data file root dir")
    parser.add_argument("--output_root", type=str, help="output file root dir")
    parser.add_argument("--data_length", type=int, help="data time length")
    parser.add_argument("--scale", type=float, help="train and test scale")
    args = parser.parse_args()
    data_preprocess(args.data_root, args.output_root, args.scale)
