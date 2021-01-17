import os
import numpy


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
    while True:
        sequences_data = sequences_file.readline()
        if sequences_data == '':
            break
        sequences_index = int(sequences_data) - 1
        if index != sequences_index:
            save_data(index, write_input_list, write_output_list)
            write_input_list = numpy.empty(shape=[0, 5307])
            write_output_list = numpy.empty(shape=[0, 618])
            index = sequences_index
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
    save_data(index, write_input_list, write_output_list)
    print("Preprocess Data Complete")


if __name__ == '__main__':
    data_preprocess("E:/NSM/data2")
