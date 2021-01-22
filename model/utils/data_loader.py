import os
from .seq_data_set import DataSet


def load_data(data_root, cache=True):
    input_dir = os.path.join(data_root, 'Input')
    label_dir = os.path.join(data_root, 'Label')
    data_source = DataSet(input_dir, label_dir, cache)
    if cache is True:
        print("Loading cache")
        data_source.load_all_data()
        print("Loading finish")
    return data_source



