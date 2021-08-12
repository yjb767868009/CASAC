import logging
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils

from model.network import *
from model.network.ATM import ATM


class Model(object):
    """
    模型训练框架
    """

    def __init__(self, save_path, lr, ):

        self.save_path = save_path

        self.atm = ATM(self.save_path, lr)

    def load_param(self, load_path):
        print("Loading Param")
        self.atm.test_init()
        self.atm.load_param(load_path)
        print("Load Finish")

    def forward(self, x=None):
        return self.atm.forward(x)

    def step_train(self, model: BaseModel, train_data_manager, test_data_manager, epoch):
        for e in range(epoch):
            # if (e + 1) % 30 == 0:
            #     model.update_lr()
            train_data_iter = train_data_manager.load_data()
            model.ep(train_data_iter, e, train=True)
            test_data_iter = test_data_manager.load_data()
            model.ep(test_data_iter, e, train=False)
            if (e + 1) % 10 == 0:
                print("saving")
                model.save()

    def train(self, train_data_manager, test_data_manager, epoch, load_path):
        if load_path != "":
            self.load_param(load_path)
        self.atm.train_init()
        self.step_train(self.atm, train_data_manager, test_data_manager, epoch)

    def test(self, data_manager, load_path=""):
        print("Testing")
        if load_path != "":
            self.load_param(load_path)
        self.atm.test_init()
        loss = self.atm.ep(data_manager, 1, train=False)
        message = 'Loss = {:.5f} '.format(loss)
        print(message)

    def view_attention(self, data_manager, load_path=""):
        print("View Attention")
        self.load_param(load_path)
        self.atm.test_init()
        data_iter = data_manager.load_data()
        self.atm.view_attention(data_iter)
