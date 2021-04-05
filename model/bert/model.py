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
        self.phase_prediction_model = PhasePredictionModel()
        self.contact_prediction_model = ContactPredictionModel()

        self.bert = model2gpu(self.bert)
        self.pretrain_model = model2gpu(self.pretrain_model)
        self.prediction_model = model2gpu(self.prediction_model)
        self.phase_prediction_model = model2gpu(self.phase_prediction_model)
        self.contact_prediction_model = model2gpu(self.contact_prediction_model)

        # build optimizer
        self.base_lr = lr
        self.lr = lr
        self.bert_optimizer = optim.AdamW(self.bert.parameters(), lr=self.lr)
        self.pretrain_optimizer = optim.AdamW(self.pretrain_model.parameters(), lr=self.lr)
        self.prediction_optimizer = optim.AdamW(self.prediction_model.parameters(), lr=self.lr)
        self.phase_prediction_optimizer = optim.AdamW(self.phase_prediction_model.parameters(), lr=self.lr)
        self.contact_prediction_optimizer = optim.AdamW(self.contact_prediction_model.parameters(), lr=self.lr)

    def load_param(self, load_type="test"):
        print('Loading param...')
        if load_type == "test":
            # Load Model
            self.bert.load_state_dict(torch.load(os.path.join(self.load_path, 'bert.pth')))
            self.prediction_model.load_state_dict(torch.load(os.path.join(self.load_path, 'prediction_model.pth')))
            self.phase_prediction_model.load_state_dict(
                torch.load(os.path.join(self.load_path, 'phase_prediction_model.pth')))
            self.contact_prediction_model.load_state_dict(
                torch.load(os.path.join(self.load_path, 'contact_prediction_model.pth')))
            # Load optimizer
            self.bert_optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'bert_optimizer.pth')))
            self.prediction_optimizer.load_state_dict(
                torch.load(os.path.join(self.load_path, 'prediction_optimizer.pth')))
            self.phase_prediction_optimizer.load_state_dict(
                torch.load(os.path.join(self.load_path, 'phase_prediction_optimizer.pth')))
            self.contact_prediction_optimizer.load_state_dict(
                torch.load(os.path.join(self.load_path, 'contact_prediction_optimizer.pth')))
        if load_type == "train":
            # Load Model
            self.bert.load_state_dict(torch.load(os.path.join(self.load_path, 'bert.pth')))
            self.pretrain_model.load_state_dict(torch.load(os.path.join(self.load_path, 'pretrain_model.pth')))
            # Load optimizer
            self.bert_optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'bert_optimizer.pth')))
            self.pretrain_optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'pretrain_optimizer.pth')))
        if load_type == "extra_train":
            # Load Model
            self.bert.load_state_dict(torch.load(os.path.join(self.load_path, 'bert.pth')))
            self.prediction_model.load_state_dict(torch.load(os.path.join(self.load_path, 'prediction_model.pth')))
            # Load optimizer
            self.bert_optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'bert_optimizer.pth')))
            self.prediction_optimizer.load_state_dict(
                torch.load(os.path.join(self.load_path, 'prediction_optimizer.pth')))
        print('Loading param complete')

    def forward(self, x, x_length=None):
        x = self.bert(x, x_length)
        y = self.prediction_model(x, x_length)
        phase = self.phase_prediction_model(x, x_length)
        contact = self.contact_prediction_model(x, x_length)
        output = torch.cat([y[:606], contact, phase])
        return output

    def step_train(self, train_data_iter, test_data_iter, train_type):
        self.lr = self.base_lr
        for e in range(self.epoch):
            if e % 50 == 0:
                self.lr = self.lr / 10
                if train_type == "pretrain":
                    for param_group in self.bert_optimizer.param_groups:
                        param_group['lr'] = self.lr
                    for param_group in self.pretrain_optimizer.param_groups:
                        param_group['lr'] = self.lr
                elif train_type == "prediction":
                    for param_group in self.bert_optimizer.param_groups:
                        param_group['lr'] = self.lr
                    for param_group in self.pretrain_optimizer.param_groups:
                        param_group['lr'] = self.lr
                elif train_type == "phase_prediction":
                    for param_group in self.phase_prediction_optimizer.param_groups:
                        param_group['lr'] = self.lr
                elif train_type == "contact_prediction":
                    for param_group in self.contact_prediction_optimizer.param_groups:
                        param_group['lr'] = self.lr
                else:
                    print("no this train type, update lr fail")
                    exit()
            loss = self.iteration(train_data_iter, train_type)
            test_loss = self.iteration(test_data_iter, train_type)
            train_message = 'Epoch {} : '.format(e + 1) + \
                            'Train Loss = {:.5f} '.format(loss) + \
                            'Test Loss = {:.5f} '.format(test_loss) + \
                            'lr = {} '.format(self.lr)
            logging.info(train_message)
            print(train_message)
            if (e + 1) % 10 == 0:
                if train_type == "pretrain":
                    # Save Model
                    torch.save(self.bert.state_dict(), os.path.join(self.save_path, "bert.pth"))
                    torch.save(self.pretrain_model.state_dict(), os.path.join(self.save_path, "pretrain_model.pth"))
                    # Save optimizer
                    torch.save(self.bert_optimizer.state_dict(), os.path.join(self.save_path, "bert_optimizer.pth"))
                    torch.save(self.pretrain_optimizer.state_dict(),
                               os.path.join(self.save_path, "pretrain_optimizer.pth"))
                elif train_type == "prediction":
                    # Save Model
                    torch.save(self.bert.state_dict(), os.path.join(self.save_path, "bert.pth"))
                    torch.save(self.prediction_model.state_dict(), os.path.join(self.save_path, "prediction_model.pth"))
                    # Save optimizer
                    torch.save(self.bert_optimizer.state_dict(), os.path.join(self.save_path, "bert_optimizer.pth"))
                    torch.save(self.prediction_optimizer.state_dict(),
                               os.path.join(self.save_path, "prediction_optimizer.pth"))
                elif train_type == "phase_prediction":
                    # Save Model
                    torch.save(self.phase_prediction_model.state_dict(),
                               os.path.join(self.save_path, "phase_prediction_model.pth"))
                    # Save optimizer
                    torch.save(self.phase_prediction_optimizer.state_dict(),
                               os.path.join(self.save_path, "phase_prediction_optimizer.pth"))
                elif train_type == "contact_prediction":
                    # Save Model
                    torch.save(self.contact_prediction_model.state_dict(),
                               os.path.join(self.save_path, "contact_prediction_model.pth"))
                    # Save optimizer
                    torch.save(self.contact_prediction_optimizer.state_dict(),
                               os.path.join(self.save_path, "contact_prediction_optimizer.pth"))
                else:
                    print("no this train type, save fail")
                    exit()

    def train(self, train=True, pretrain=True, extra_train=True):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))
        logging.info(self.bert)
        logging.info(self.pretrain_model)
        logging.info(self.prediction_model)
        logging.info(self.contact_prediction_model)
        logging.info(self.phase_prediction_model)

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
            self.bert.train()
            self.pretrain_model.train()
            logging.info("pretrain")
            self.step_train(train_data_iter, test_data_iter, train_type="pretrain")
        if train:
            if not pretrain:
                self.load_param("train")
            self.bert.train()
            self.prediction_model.train()
            logging.info("base train")
            self.step_train(train_data_iter, test_data_iter, train_type="prediction")
        if extra_train:
            if not train:
                self.load_param("extra_train")
            # phase prediction train
            self.bert.eval()
            self.phase_prediction_model.train()
            logging.info("phase prediction train")
            self.step_train(train_data_iter, test_data_iter, train_type="phase_prediction")
            # contact prediction train
            self.bert.eval()
            self.contact_prediction_model.train()
            logging.info("contact prediction train")
            self.step_train(train_data_iter, test_data_iter, train_type="contact_prediction")

    def test(self):
        self.load_param("test")
        self.bert.eval()
        self.prediction_model.eval()
        self.contact_prediction_model.eval()
        self.phase_prediction_model.eval()
        print("Testing")
        data_iter = tordata.DataLoader(
            dataset=self.test_source,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
        )
        base_loss = self.iteration(data_iter, train_type="base_test")
        contact_loss = self.iteration(data_iter, train_type="contact_test")
        phase_loss = self.iteration(data_iter, train_type="phase_test")
        loss = self.iteration(data_iter, train_type="test")
        message = 'base Loss = {:.5f} '.format(base_loss) + \
                  'contact loss Loss = {:.5f} '.format(contact_loss) + \
                  'phase loss Loss = {:.5f} '.format(phase_loss) + \
                  'loss Loss = {:.5f} '.format(loss)
        print(message)

    def iteration(self, data_iter, train_type):
        loss_list = []
        for (input, input_random, label), data_length in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                input_random = input_random.cuda()
                label = label.cuda()
            if train_type == "pretrain":
                self.bert_optimizer.zero_grad()
                self.pretrain_optimizer.zero_grad()

                output = self.bert(input_random, data_length)
                output = self.pretrain_model(output, data_length)

                loss = mask_loss(output, input, data_length)
                loss_list.append(loss.item())
                loss.backward()

                self.bert_optimizer.step()
                self.pretrain_optimizer.step()
            elif train_type == "prediction":
                self.bert_optimizer.zero_grad()
                self.prediction_model.zero_grad()

                output = self.bert(input, data_length)
                output = self.prediction_model(output, data_length)

                loss = mask_last_loss(output, label, data_length)
                loss_list.append(loss.item())
                loss.backward()

                self.bert_optimizer.step()
                self.prediction_optimizer.step()

            elif train_type == "phase_prediction":
                self.phase_prediction_model.zero_grad()

                output = self.bert(input, data_length)
                output = self.phase_prediction_model(output)

                loss = mask_last_loss(output, label[:, :, 611:618], data_length)
                loss_list.append(loss.item())
                loss.backward()

                self.phase_prediction_optimizer.step()
            elif train_type == "contact_prediction":
                self.contact_prediction_optimizer.zero_grad()

                output = self.bert(input, data_length)
                output = self.contact_prediction_model(output)

                loss = mask_last_loss(output, label[:, :, 606:611], data_length)
                loss_list.append(loss.item())
                loss.backward()

                self.contact_prediction_optimizer.step()
            elif train_type == "test":
                output = self.bert(input, data_length)
                base = self.prediction_model(output, data_length)
                contact = self.contact_prediction_model(output)
                phase = self.phase_prediction_model(output)
                output = torch.cat([base[:, :, :606], contact, phase], 2)

                loss = mask_last_loss(output, label, data_length)
                loss_list.append(loss.item())
            elif train_type == "base_test":
                output = self.bert(input, data_length)
                base = self.prediction_model(output, data_length)
                loss = mask_last_loss(base, label, data_length)
                loss_list.append(loss.item())
            elif train_type == "contact_test":
                output = self.bert(input, data_length)
                contact = self.contact_prediction_model(output)
                loss = mask_last_loss(contact, label[:, :, 606:611], data_length)
                loss_list.append(loss.item())
            elif train_type == "phase_test":
                output = self.bert(input, data_length)
                phase = self.phase_prediction_model(output)
                loss = mask_last_loss(phase, label[:, :, 611:618], data_length)
                loss_list.append(loss.item())
            else:
                print("no this train type, train fail")
                exit()
        avg_loss = np.asarray(loss_list).mean()
        return avg_loss
