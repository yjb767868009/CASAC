import os
import random
import shutil

import numpy
from tqdm import tqdm


def get_norm(file_path):
    normalize_data = numpy.float32(numpy.loadtxt(file_path))
    mean = normalize_data[0]
    std = normalize_data[1]
    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1
    return mean, std


def data_preprocess(root_dir):
    input_dir = os.path.join(root_dir, "Input")
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    output_dir = os.path.join(root_dir, "Label")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    def save_data(i, write_input, write_output):
        print("Preprocess Data " + str(i) + " ......")
        numpy.savetxt(os.path.join(input_dir, str(i) + '.txt'), write_input, fmt="%.8f")
        numpy.savetxt(os.path.join(output_dir, str(i) + '.txt'), write_output, fmt="%.8f")

    sequences_file = open(os.path.join(root_dir, "Sequences.txt"), 'r')
    input_file = open(os.path.join(root_dir, "Input.txt"), 'r')
    output_file = open(os.path.join(root_dir, "Output.txt"), 'r')

    input_mean, input_std = get_norm(os.path.join(root_dir, "InputNorm.txt"))
    output_mean, output_std = get_norm(os.path.join(root_dir, "OutputNorm.txt"))

    index = 0
    write_input_list = numpy.empty(shape=[0, 5307]).astype('float32')
    write_output_list = numpy.empty(shape=[0, 618]).astype('float32')
    length = 0
    save_index = 0
    while True:
        sequences_data = sequences_file.readline()
        if sequences_data == '':
            break
        sequences_index = int(sequences_data) - 1
        if length >= 150:
            print("OVERSIZE SPILT")
            save_data(save_index, write_input_list[:100], write_output_list[:100])
            save_index += 1
            write_input_list = write_input_list[100:]
            write_output_list = write_output_list[100:]
            length = length - 100
        if index != sequences_index:
            save_data(save_index, write_input_list, write_output_list)
            save_index += 1
            write_input_list = numpy.empty(shape=[0, 5307])
            write_output_list = numpy.empty(shape=[0, 618])
            index = sequences_index
            length = 0
        input_data_str = input_file.readline()
        input_data = [[float(x) for x in input_data_str.split(' ')]]
        input_data = numpy.array(input_data).astype('float32')
        input_data = (input_data - input_mean) / input_std
        output_data_str = output_file.readline()
        output_data = [[float(x) for x in output_data_str.split(' ')]]
        output_data = numpy.array(output_data).astype('float32')
        output_data = (output_data - output_mean) / output_std
        write_input_list = numpy.append(write_input_list, input_data, axis=0)
        write_output_list = numpy.append(write_output_list, output_data, axis=0)
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
        test_file = str(random_size) + '.txt'
        train_file = str(i - random_size) + '.txt'
        if file in random_list:
            shutil.copy(os.path.join(input_dir, file), os.path.join(test_input_dir, test_file))
            shutil.copy(os.path.join(output_dir, file), os.path.join(test_output_dir, test_file))
            random_size += 1
        else:
            shutil.copy(os.path.join(input_dir, file), os.path.join(train_input_dir, train_file))
            shutil.copy(os.path.join(output_dir, file), os.path.join(train_output_dir, train_file))


if __name__ == '__main__':
    # data_preprocess("/home/yujubo/disk/data")
    divide_train_test("/home/yujubo/disk/data", 0.1)
    # divide_train_test("E:/NSM/data", 0.1)
