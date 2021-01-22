import torch
import torch.nn.utils.rnn as rnn_utils


def collate_fn(data):
    batch_size = len(data)
    input_data = [data[i][0] for i in range(batch_size)]
    input_data_random = [data[i][1] for i in range(batch_size)]
    output_data = [data[i][2] for i in range(batch_size)]
    input_data.sort(key=lambda x: len(x), reverse=True)
    input_data_random.sort(key=lambda x: len(x), reverse=True)
    output_data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in input_data]
    input_data = rnn_utils.pad_sequence(input_data, batch_first=True, padding_value=0)
    input_data_random = rnn_utils.pad_sequence(input_data_random, batch_first=True, padding_value=0)
    output_data = rnn_utils.pad_sequence(output_data, batch_first=True, padding_value=0)
    return [input_data, input_data_random, output_data], data_length
