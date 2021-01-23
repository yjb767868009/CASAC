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
from model.network.bert import BERT
from model.network.bertlm import BERTLM
from model.utils.collate_fn import collate_fn


class Model(object):
    def __init__(self,
                 # For Date information
                 train_source, test_source, save_path, load_path,
                 # For encoder network information
                 encoder_nums, encoder_dims, encoder_activations, encoder_dropout,
                 # For BERT network information
                 mid, hidden, n_layers, attn_heads, bert_dropout,
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
        self.model = BERTLM(self.bert, mid, hidden)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        # build optimizer
        self.lr = lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        # build loss function
        self.loss_function = self.mask_loss
        self.min_loss = 99

    def mask_loss(self, x, y, data_length):
        mask = torch.zeros_like(x).float()
        for i in range(len(mask)):
            mask[i][:data_length[i]] = 1
        x = x * mask
        loss = torch.mean(torch.pow((x - y), 2))
        return loss

    def load_param(self):
        print('Loading parm...')
        # Load Model
        self.model.load_state_dict(torch.load(os.path.join(self.load_path, 'model.pth')))
        # Load optimizer
        self.optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'optimizer.pth')))
        print('Loading param complete')

    def forward(self, x, x_length=None, train=True):
        return self.model(x, x_length, train)

    def train(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))
        logging.info(self.model)
        print("Training")
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
        for e in range(self.epoch):
            if e % 50 == 0:
                self.lr = self.lr / 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            loss, next_loss, mask_loss = self.iteration(train_data_iter, True)
            test_loss, test_next_loss, test_mask_loss = self.iteration(test_data_iter, False)
            train_message = 'Epoch {} : '.format(e + 1) + \
                            'Train Loss = {:.5f} '.format(loss) + \
                            'Train Next Loss = {:.5f} '.format(next_loss) + \
                            'Train Mask Loss = {:.5f} '.format(mask_loss) + \
                            'Test Loss = {:.5f} '.format(test_loss) + \
                            'Test Next Loss = {:.5f} '.format(test_next_loss) + \
                            'Test Mask Loss = {:.5f} '.format(test_mask_loss) + \
                            'lr = {} '.format(self.lr)
            logging.info(train_message)
            print(train_message)
            if e % 10 == 0 and loss < self.min_loss:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pth"))
                torch.save(self.optimizer.state_dict(), os.path.join(self.save_path, "optimizer.pth"))

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
        loss, next_loss, mask_loss = self.iteration(data_iter, False)
        message = 'Train Loss = {:.5f} '.format(loss) + \
                  'Train Next Loss = {:.5f} '.format(next_loss) + \
                  'Train Mask Loss = {:.5f} '.format(mask_loss)
        print(message)

    def iteration(self, data_iter, train=True):
        loss_list = []
        next_loss_list = []
        mask_loss_list = []
        for (input, input_random, label), data_length in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                input_random = input_random.cuda()
                label = label.cuda()
            self.optimizer.zero_grad()

            next_sent_output, mask_lm_output = self.model(input_random, data_length)

            # loss
            next_loss = self.mask_loss(next_sent_output, label, data_length)
            mask_loss = self.mask_loss(mask_lm_output, input, data_length)
            loss = next_loss + mask_loss

            next_loss_list.append(next_loss.item())
            mask_loss_list.append(mask_loss.item())
            loss_list.append(loss.item())

            if train:
                loss.backward()
                self.optimizer.step()

        avg_loss = np.asarray(loss_list).mean()
        avg_next_loss = np.asarray(next_loss_list).mean()
        avg_mask_loss = np.asarray(mask_loss_list).mean()
        return avg_loss, avg_next_loss, avg_mask_loss
