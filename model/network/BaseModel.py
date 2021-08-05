import os
import torch
import torch.nn as nn

from model.utils import model2gpu
from model.utils import base_loss


class BaseModel(object):
    """
    Base model
    """

    def __init__(self):
        self.lr = 0.001

    def train_init(self):
        pass

    def test_init(self):
        pass

    def save(self):
        pass

    def load_param(self, load_path):
        pass

    def update_lr(self):
        pass

    def ep(self, data_iter, epoch, train):
        pass
