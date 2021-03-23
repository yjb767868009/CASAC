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
from model.utils.GPU_tools import model2gpu
from model.utils.Loss import *
from model.network.bert import BERT
from model.network.BERTPredictionModel import BERTPredictionModel
from model.utils.collate_fn import collate_fn


class Model(object):
    def __init__(self,
                 # For Date information
                 train_source, test_source, save_path, load_path,
                 # For encoder network information
                 encoder_nums, encoder_dims, encoder_activations, encoder_dropout,
                 # For BERT network information
                 mid_dims, n_layers, attn_heads, bert_dropout,
                 # For Phase Prediction network information
                 # phase_n_layers, phase_dims, phase_activations, phase_dropout,
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
                         mid_dims, n_layers, attn_heads, bert_dropout, )
        self.pretrain_model = BERTPretrainModel()
        self.prediction_model = BERTPredictionModel()
        # self.phase_prediction_model = FullConnectionModule(phase_n_layers, phase_dims, phase_activations, phase_dropout)

        self.bert = model2gpu(self.bert)
        self.pretrain_model = model2gpu(self.pretrain_model)
        self.prediction_model = model2gpu(self.prediction_model)

        # build optimizer
        self.base_lr = lr
        self.lr = lr
        self.bert_optimizer = optim.AdamW(self.bert.parameters(), lr=self.lr)
        self.pretrain_optimizer = optim.AdamW(self.pretrain_model.parameters(), lr=self.lr)
        self.prediction_optimizer = optim.AdamW(self.prediction_model.parameters(), lr=self.lr)

    def load_param(self):
        print('Loading parm...')
        # Load Model
        self.bert.load_state_dict(torch.load(os.path.join(self.load_path, 'bert.pth')))
        self.pretrain_model.load_state_dict(torch.load(os.path.join(self.load_path, 'pretrain_model.pth')))
        self.prediction_model.load_state_dict(torch.load(os.path.join(self.load_path, 'prediction_model.pth')))
        # Load optimizer
        self.bert_optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'bert_optimizer.pth')))
        self.pretrain_model.load_state_dict(torch.load(os.path.join(self.load_path, 'pretrain_model.pth')))
        self.prediction_model.load_state_dict(torch.load(os.path.join(self.load_path, 'prediction_model.pth')))
        print('Loading param complete')

    def save(self):
        # Save Model
        torch.save(self.bert.state_dict(), os.path.join(self.save_path, "bert.pth"))
        torch.save(self.pretrain_model.state_dict(), os.path.join(self.save_path, "pretrain_model.pth"))
        torch.save(self.prediction_model.state_dict(), os.path.join(self.save_path, "prediction_model.pth"))
        # Save optimizer
        torch.save(self.bert_optimizer.state_dict(), os.path.join(self.save_path, "bert_optimizer.pth"))
        torch.save(self.pretrain_model.state_dict(), os.path.join(self.save_path, "pretrain_model.pth"))
        torch.save(self.prediction_model.state_dict(), os.path.join(self.save_path, "prediction_model.pth"))

    def forward(self, x, x_length=None, pre_train=True):
        x = self.bert(x, x_length)
        if pre_train:
            return self.pretrain_model(x, x_length, pre_train)
        else:
            return self.prediction_model(x, x_length, pre_train)

    def step_train(self, train_data_iter, test_data_iter, pre_train=False):
        self.lr = self.base_lr
        for e in range(self.epoch):
            if e % 50 == 0:
                self.lr = self.lr / 10
                for param_group in self.bert_optimizer.param_groups:
                    param_group['lr'] = self.lr
                if pre_train:
                    for param_group in self.pretrain_optimizer.param_groups:
                        param_group['lr'] = self.lr
                else:
                    for param_group in self.prediction_optimizer.param_groups:
                        param_group['lr'] = self.lr
            loss = self.iteration(train_data_iter, pre_train=pre_train, train=True)
            test_loss = self.iteration(test_data_iter, pre_train=pre_train, train=False)
            train_message = 'Epoch {} : '.format(e + 1) + \
                            'Train Loss = {:.5f} '.format(loss) + \
                            'Test Loss = {:.5f} '.format(test_loss) + \
                            'lr = {} '.format(self.lr)
            logging.info(train_message)
            print(train_message)
            if (e + 1) % 10 == 0:
                self.save()

    def train(self, train=True, pretrain=True):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))
        logging.info(self.bert)
        logging.info(self.pretrain_model)
        logging.info(self.prediction_model)

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

        self.bert.train()
        self.pretrain_model.train()
        self.prediction_model.train()

        if pretrain:
            self.step_train(train_data_iter, test_data_iter, pre_train=True)
        if train:
            self.step_train(train_data_iter, test_data_iter, pre_train=False)

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
            if pre_train:
                self.bert_optimizer.zero_grad()
                self.pretrain_optimizer.zero_grad()
            else:
                self.bert_optimizer.zero_grad()
                self.prediction_model.zero_grad()

            # loss
            if pre_train:
                output = self.bert(input_random, data_length)
                output = self.pretrain_model(output, data_length)
                loss = mask_loss(output, input, data_length)
            else:
                output = self.bert(input, data_length)
                output = self.prediction_model(output, data_length)
                loss = mask_last_loss(output, label, data_length)

            loss_list.append(loss.item())

            if train:
                loss.backward()
                if pre_train:
                    self.bert_optimizer.step()
                    self.pretrain_optimizer.step()
                else:
                    self.bert_optimizer.step()
                    self.prediction_optimizer.step()

        avg_loss = np.asarray(loss_list).mean()
        return avg_loss
