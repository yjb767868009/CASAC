import logging
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data as tordata

from model.network import *
from model.network.ATM import ATM


class Model(object):
    """
    模型训练框架
    """

    def __init__(self,
                 # For Date information
                 save_path,
                 # For Model base information
                 epoch, batch_size, lr,
                 ):
        self.epoch = epoch
        self.batch_size = batch_size

        self.save_path = save_path

        self.atm = ATM(self.save_path, lr)

    def load_param(self, load_path):
        self.atm.test_init()
        self.atm.load_param(load_path)

    def forward(self, x=None):
        return self.atm.forward(x)

    def step_train(self, model: BaseModel, train_data_iter, test_data_iter):
        for e in range(self.epoch):
            if (e + 1) % 30 == 0:
                model.update_lr()
            loss = model.ep(train_data_iter, e, train=True)
            test_loss = model.ep(test_data_iter, e, train=False)

            train_message = 'Epoch {} : '.format(e + 1) + \
                            'Train Loss = {:.5f} '.format(loss) + \
                            'Test Loss = {:.5f} '.format(test_loss) + \
                            'lr = {} '.format(model.lr)
            logging.info(train_message)
            print(train_message)
            if (e + 1) % 10 == 0:
                print("saving")
                model.save(self.save_path)

    def train(self, train_source, test_source):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

        train_data_iter = tordata.DataLoader(
            dataset=train_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
        )
        test_data_iter = tordata.DataLoader(
            dataset=test_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
        )

        self.step_train(self.atm, train_data_iter, test_data_iter)

    def test(self, test_source, load_path=""):
        print("Testing")
        if load_path != "":
            self.load_param(load_path)
        data_iter = tordata.DataLoader(
            dataset=test_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        self.atm.test_init()
        loss = self.atm.ep(data_iter, 1, train=False)
        message = 'Loss = {:.5f} '.format(loss)
        print(message)

    def view_ateention(self, test_source, load_path=""):
        print("Testing")
        self.load_param(load_path)
        data_iter = tordata.DataLoader(
            dataset=test_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        self.atm.test_init()
        self.atm.view_attention(data_iter)
