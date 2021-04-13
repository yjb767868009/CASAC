import os
import torch
import torch.nn as nn

from model.utils import model2gpu
from model.utils import mask_loss


class BaseModel(object):
    """
    Base model
    """
    def __init__(self):
        self.lr=0

    def train_init(self):
        pass

    def test_init(self):
        pass

    def save(self, save_path):
        pass

    def load_param(self, load_path):
        pass

    def update_lr(self):
        pass

    def train(self, data_iter):
        pass

    def test(self, data_iter):
        pass
