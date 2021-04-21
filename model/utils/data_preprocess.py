import argparse
import os
import random
import shutil
import torch

from tqdm import tqdm


def get_norm(file_path):
    with open(file_path, "r") as f:
        lines = [list(map(float, line[:-1].split(" ")))
                 for line in tqdm(f, desc="Loading Norm")]
    normalize_data = torch.tensor(lines)
    mean = normalize_data[0]
    std = normalize_data[1]
    for i in range(std.size(0)):
        if std[i] == 0:
            std[i] = 1
    return mean, std


def data_preprocess(root_dir, output_root, data_length):
    input_dir = os.path.join(output_root, "Input")
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    output_dir = os.path.join(output_root, "Label")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    def save_data(i, write_input, write_output):
        print("Preprocess Data " + str(i) + " ......")
        torch.save(write_input, os.path.join(input_dir, str(i) + '.pth'))
        torch.save(write_output, os.path.join(output_dir, str(i) + '.pth'))

    sequences_file = open(os.path.join(root_dir, "Sequences.txt"), 'r')
    input_file = open(os.path.join(root_dir, "Input.txt"), 'r')
    output_file = open(os.path.join(root_dir, "Output.txt"), 'r')

    input_mean, input_std = get_norm(os.path.join(root_dir, "InputNorm.txt"))
    output_mean, output_std = get_norm(os.path.join(root_dir, "OutputNorm.txt"))

    index = 0
    write_input_list = torch.zeros((0, 5307))
    write_output_list = torch.zeros((0, 618))
    length = 0
    save_index = 0
    while True:
        sequences_data = sequences_file.readline()
        if sequences_data == '':
            break
        sequences_index = int(sequences_data) - 1
        if length >= int(data_length * 1.5):
            print("OVERSIZE SPILT")
            save_data(save_index, write_input_list[:data_length], write_output_list[:data_length])
            save_index += 1
            write_input_list = write_input_list[data_length:]
            write_output_list = write_output_list[data_length:]
            length = length - data_length
        if index != sequences_index:
            save_data(save_index, write_input_list, write_output_list)
            save_index += 1
            write_input_list = torch.zeros((0, 5307))
            write_output_list = torch.zeros((0, 618))
            index = sequences_index
            length = 0
        input_data_str = input_file.readline()
        input_data = [[float(x) for x in input_data_str.split(' ')]]
        input_data = torch.tensor(input_data)
        input_data = (input_data - input_mean) / input_std
        output_data_str = output_file.readline()
        output_data = [[float(x) for x in output_data_str.split(' ')]]
        output_data = torch.tensor(output_data)
        output_data = (output_data - output_mean) / output_std
        write_input_list = torch.cat((write_input_list, input_data), 0)
        write_output_list = torch.cat((write_output_list, output_data), 0)
        length += 1
    save_data(save_index, write_input_list, write_output_list)
    print("Preprocess Data Complete")


def divide_train_test(root_dir, scale):
    input_dir = os.path.join(root_dir, "Input")
    output_dir = os.path.join(root_dir, "Label")
    train_dir = os.path.join(root_dir, "Train")
    test_dir = os.path.join(root_dir, "Test")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    train_input_dir = os.path.join(train_dir, "Input")
    train_output_dir = os.path.join(train_dir, "Label")
    test_input_dir = os.path.join(test_dir, "Input")
    test_output_dir = os.path.join(test_dir, "Label")
    if not os.path.exists(train_input_dir):
        os.mkdir(train_input_dir)
    if not os.path.exists(train_output_dir):
        os.mkdir(train_output_dir)
    if not os.path.exists(test_input_dir):
        os.mkdir(test_input_dir)
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)
    input_list = os.listdir(input_dir)
    data_size = len(input_list)
    assert data_size == len(os.listdir(output_dir)), "输入和输出文件数量不一致"
    random_list = random.sample(input_list, int(data_size * scale))
    random_size = 0
    for i in tqdm(range(data_size), ncols=100):
        file = input_list[i]
        test_file = str(random_size) + '.pth'
        train_file = str(i - random_size) + '.pth'
        if file in random_list:
            shutil.copy(os.path.join(input_dir, file), os.path.join(test_input_dir, test_file))
            shutil.copy(os.path.join(output_dir, file), os.path.join(test_output_dir, test_file))
            random_size += 1
        else:
            shutil.copy(os.path.join(input_dir, file), os.path.join(train_input_dir, train_file))
            shutil.copy(os.path.join(output_dir, file), os.path.join(train_output_dir, train_file))


def read_input_file(root_dir):
    input_file = open(os.path.join(root_dir, "Input.txt"), 'r')
    while True:
        input_data_str = input_file.readline()
        input_data = [float(x) for x in input_data_str.split(' ')]


def read_output_file(root_dir):
    output_file = open(os.path.join(root_dir, "Output.txt"), 'r')
    while True:
        output_data_str = output_file.readline()
        output_data = [float(x) for x in output_data_str.split(' ')]
        print(output_data[611:618])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--data_root", type=str, help="data file root dir")
    parser.add_argument("--output_root", type=str, help="output file root dir")
    parser.add_argument("--data_length", type=int, help="data time length")
    parser.add_argument("--scale", type=float, help="train and test scale")
    args = parser.parse_args()
    data_preprocess(args.data_root, args.output_root, args.data_length)
    divide_train_test(args.output_root, args.scale)
