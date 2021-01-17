import os
from model.utils import load_data
from model.bert.model import Model
from model.bert.config import conf


def initialization(epoch, batch_size, data_root, save_path, load_path, cache=True):
    print("Initializing model...")
    train_source, test_source = load_data(data_root, cache=cache)
    conf['epoch'] = epoch
    conf['batch_size'] = batch_size
    conf['train_source'] = train_source
    conf['test_source'] = test_source
    conf['save_path'] = save_path
    conf['load_path'] = load_path
    model = Model(**conf)
    print("Model initialization complete.")
    return model
