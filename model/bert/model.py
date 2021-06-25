import logging
import os
import numpy as np
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.cpp_extension
import torch.utils.data as tordata
import torch.nn.utils.rnn as rnn_utils
from model.network import *
from model.network.ATM import ATM
from model.utils.GPU_tools import *
from model.utils.Loss import *


class Model(object):
    def __init__(self,
                 # For Date information
                 train_source, test_source, save_path,
                 # For Model base information
                 epoch, batch_size,
                 ):
        self.epoch = epoch
        self.batch_size = batch_size

        self.train_source = train_source
        self.test_source = test_source
        self.save_path = save_path

        self.atm = ATM()

    def load_param(self, load_path):
        self.atm.load_param(load_path)

    def forward(self, x=None):
        return self.atm.forward(x)

    def step_train(self, model: BaseModel, train_data_iter, test_data_iter):
        for e in range(self.epoch):
            if (e + 1) % 30 == 0:
                model.update_lr()
            loss = model.ep(train_data_iter, train=True)
            test_loss = model.ep(test_data_iter, train=False)
            train_message = 'Epoch {} : '.format(e + 1) + \
                            'Train Loss = {:.5f} '.format(loss) + \
                            'Test Loss = {:.5f} '.format(test_loss) + \
                            'lr = {} '.format(model.lr)
            logging.info(train_message)
            print(train_message)
            if (e + 1) % 10 == 0:
                print("saving")
                model.save(self.save_path)

    def train(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

        train_data_iter = tordata.DataLoader(
            dataset=self.train_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
        )
        test_data_iter = tordata.DataLoader(
            dataset=self.test_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
        )

        self.step_train(self.atm, train_data_iter, test_data_iter)

    def test(self, load_path=""):
        print("Testing")
        if load_path != "":
            self.load_param(load_path)
        data_iter = tordata.DataLoader(
            dataset=self.train_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        self.atm.test_init()
        loss = self.atm.test(data_iter)
        message = 'Loss = {:.5f} '.format(loss)
        print(message)
