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
from model.utils.GPU_tools import *
from model.utils.Loss import *
from model.network import KeyBERT,MotionBERT
from model.utils.collate_fn import collate_fn


class Model(object):
    def __init__(self,
                 # For Date information
                 train_source, test_source, save_path, load_path,
                 # For encoder network information
                 encoder_nums, encoder_dims, encoder_activations, encoder_dropout,
                 # For BERT network information
                 key_bert_hidden,motion_bert_hidden, n_layers, attn_heads, bert_dropout,
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

        self.embedding = Embedding(encoder_nums, encoder_dims, encoder_activations, encoder_dropout, segmentation)
        self.key_bert = KeyBERT(self.embedding, key_bert_hidden, n_layers, attn_heads, bert_dropout, )
        self.motion_bert = MotionBERT(self.embedding, motion_bert_hidden, n_layers, attn_heads, bert_dropout, )

        self.key_bert_pretrain = KeyBertPretrain(self.key_bert, lr)
        self.key_bert_prediction = KeyBertPrediction(self.key_bert, lr)
        self.motion_bert_pretrain = MotionBertPretrain(self.motion_bert, lr)
        self.motion_bert_prediction = MotionBertPrediction(self.motion_bert, lr)

    def forward(self, x, x_length=None):
        key = self.key_bert_prediction.forward(x, x_length)
        return self.motion_bert_prediction.forward(key, x, x_length)

    def step_train(self, model: BaseModel, train_data_iter, test_data_iter):
        for e in range(self.epoch):
            if (e+1) % 50 == 0:
                model.update_lr()
            loss = model.train(train_data_iter)
            test_loss = model.test(test_data_iter)
            train_message = 'Epoch {} : '.format(e + 1) + \
                            'Train Loss = {:.5f} '.format(loss) + \
                            'Test Loss = {:.5f} '.format(test_loss) + \
                            'lr = {} '.format(model.lr)
            logging.info(train_message)
            print(train_message)
            if (e + 1) % 10 == 0:
                print("saving")
                model.save(self.save_path)

    def train(self, key_train=True, motion_train=True):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

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

        if key_train:
            #self.key_bert_pretrain.train_init()
            #self.step_train(self.key_bert_pretrain, train_data_iter, test_data_iter)
            self.key_bert_prediction.train_init()
            self.step_train(self.key_bert_prediction, train_data_iter, test_data_iter)
        if motion_train:
            #self.motion_bert_pretrain.train_init()
            #self.step_train(self.motion_bert_pretrain, train_data_iter, test_data_iter)
            self.motion_bert_prediction.train_init()
            self.step_train(self.motion_bert_prediction, train_data_iter, test_data_iter)

    def test(self, load_path=""):
        print("Testing")
        if load_path != "":
            self.key_bert_prediction.load_param(load_path)
            self.motion_bert_prediction.load_param(load_path)
        data_iter = tordata.DataLoader(
            dataset=self.test_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.key_bert_prediction.test_init()
        key_loss = self.key_bert_prediction.test(data_iter)
        self.motion_bert_prediction.test_init()
        motion_loss = self.motion_bert_prediction.test(data_iter)
        message = 'Key Loss = {:.5f} '.format(key_loss) + \
                  'Motion Loss = {:.5f} '.format(motion_loss)
        print(message)
