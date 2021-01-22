import os
from model.utils import load_data
from model.bert.model import Model
from model.bert.config import conf


def initialization(epoch, batch_size, data_root, save_path, load_path, cache=True, train=True):
    print("Initializing model...")
    conf['epoch'] = epoch
    conf['batch_size'] = batch_size
    conf['save_path'] = save_path
    conf['load_path'] = load_path
    if train:
        train_source = load_data(os.path.join(data_root, "Train"), cache=cache)
        test_source = load_data(os.path.join(data_root, "Test"), cache=cache)
    else:
        train_source = load_data(data_root, cache=cache)
        test_source = train_source
    conf['train_source'] = train_source
    conf['test_source'] = test_source
    model = Model(**conf)
    print("Model initialization complete.")
    return model
