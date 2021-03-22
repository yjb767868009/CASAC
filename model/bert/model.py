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
from model.utils.Loss import *
from model.network.bert import BERT
from model.network.BERTAM import BERTAM
from model.utils.collate_fn import collate_fn


class Model(object):
    def __init__(self,
                 # For Date information
                 train_source, test_source, save_path, load_path,
                 # For encoder network information
                 encoder_nums, encoder_dims, encoder_activations, encoder_dropout,
                 # For BERT network information
                 mid, hidden, n_layers, attn_heads, bert_dropout, lm_dropout,
                 # For Model base information
                 epoch, batch_size, segmentation, lr,
                 ):
        self.epoch = epoch
        self.batch_size = batch_size
        self.segmentation = segmentation

        self.train_source = train_source
        self.test_source = test_source
        self.save_path = save_path
        self.load_path = load_path

        self.bert = BERT(encoder_nums, encoder_dims, encoder_activations, encoder_dropout, segmentation,
                         mid, n_layers, attn_heads, bert_dropout, )
        self.model = BERTAM(self.bert, mid, hidden, lm_dropout)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        # build optimizer
        self.base_lr = lr
        self.lr = lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        # build loss function
        self.min_loss = 99

    def load_param(self):
        print('Loading parm...')
        # Load Model
        self.model.load_state_dict(torch.load(os.path.join(self.load_path, 'model.pth')))
        # Load optimizer
        self.optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'optimizer.pth')))
        print('Loading param complete')

    def forward(self, x, x_length=None, pre_train=True):
        return self.model(x, x_length, pre_train)

    def step_train(self, train_data_iter, test_data_iter, pre_train=False):
        self.lr = self.base_lr
        for e in range(self.epoch):
            if e % 50 == 0:
                self.lr = self.lr / 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            loss = self.iteration(train_data_iter, pre_train=pre_train, train=True)
            test_loss = self.iteration(test_data_iter, pre_train=pre_train, train=False)
            train_message = 'Epoch {} : '.format(e + 1) + \
                            'Train Loss = {:.5f} '.format(loss) + \
                            'Test Loss = {:.5f} '.format(test_loss) + \
                            'lr = {} '.format(self.lr)
            logging.info(train_message)
            print(train_message)
            if (e + 1) % 10 == 0 and loss < self.min_loss:
                self.save()

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.save_path, "optimizer.pth"))

    def train(self, train=True, pretrain=True):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))
        logging.info(self.model)
        self.model.train()

        train_data_iter = tordata.DataLoader(
            dataset=self.train_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
        )
        test_data_iter = tordata.DataLoader(
            dataset=self.test_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
        )
        if pretrain:
            print("Pretraining START")
            logging.info("Pretraining START")
            self.step_train(train_data_iter, test_data_iter, pre_train=True)
            print("Pretraining COMPLETE")
            logging.info("Pretraining COMPLETE")
        if train:
            print("Training START")
            logging.info("Training START")
            self.step_train(train_data_iter, test_data_iter, pre_train=False)
            logging.info("Training COMPLETE")
            print("Training COMPLETE")

    def test(self):
        print("Testing")
        data_iter = tordata.DataLoader(
            dataset=self.test_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
        )
        loss = self.iteration(data_iter, pre_train=False, train=False)
        message = 'Loss = {:.5f} '.format(loss)
        print(message)

    def iteration(self, data_iter, pre_train=True, train=True):
        loss_list = []
        for (input, input_random, label), data_length in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                input_random = input_random.cuda()
                label = label.cuda()
            self.optimizer.zero_grad()

            # loss
            if pre_train:
                output = self.model(input_random, data_length, pre_train)
                loss = mask_loss(output, input, data_length)
            else:
                output = self.model(input, data_length, pre_train)
                loss = mask_last_loss(output, label, data_length)

            loss_list.append(loss.item())

            if train:
                loss.backward()
                self.optimizer.step()

        avg_loss = np.asarray(loss_list).mean()
        return avg_loss
