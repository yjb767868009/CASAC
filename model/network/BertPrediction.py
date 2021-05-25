import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .BaseModel import BaseModel
from model.utils import model2gpu
from model.utils import *


class BertPrediction(BaseModel):
    """
    Motion Bert Train Model
    Prediction Animation motion feature
    """

    def __init__(self, bert, lr):
        super().__init__()
        self.bert = model2gpu(bert)
        self.prediction = model2gpu(Prediction())
        self.rebuild = model2gpu(Rebuild())
        self.lr = lr
        self.bert_optimizer = torch.optim.AdamW(self.bert.parameters(), lr=self.lr)
        self.prediction_optimizer = torch.optim.AdamW(self.prediction.parameters(), lr=self.lr)
        self.rebuild_optimizer = torch.optim.AdamW(self.rebuild.parameters(), lr=self.lr)

    def train_init(self):
        self.bert.train()
        self.prediction.train()
        self.rebuild.train()

    def test_init(self):
        self.bert.eval()
        self.prediction.eval()
        self.rebuild.eval()

    def save(self, save_path):
        # Save Model
        torch.save(self.bert.state_dict(), os.path.join(save_path, "bert.pth"))
        torch.save(self.prediction.state_dict(), os.path.join(save_path, "prediction.pth"))
        torch.save(self.rebuild.state_dict(), os.path.join(save_path, "rebuild.pth"))
        # Save optimizer
        torch.save(self.bert_optimizer.state_dict(), os.path.join(save_path, "bert_optimizer.pth"))
        torch.save(self.prediction_optimizer.state_dict(),
                   os.path.join(save_path, "prediction_optimizer.pth"))
        torch.save(self.rebuild_optimizer.state_dict(), os.path.join(save_path, "rebuild_optimizer.pth"))

    def load_param(self, load_path):
        # Load Model
        self.bert.load_state_dict(torch.load(os.path.join(load_path, 'bert.pth')))
        self.prediction.load_state_dict(torch.load(os.path.join(load_path, 'prediction.pth')))
        self.rebuild.load_state_dict(torch.load(os.path.join(load_path, 'rebuild.pth')))
        # Load optimizer
        # self.bert_optimizer.load_state_dict(torch.load(os.path.join(load_path, 'bert_optimizer.pth')))
        self.prediction_optimizer.load_state_dict(
            torch.load(os.path.join(load_path, 'prediction_optimizer.pth')))
        self.rebuild_optimizer.load_state_dict(torch.load(os.path.join(load_path, 'rebuild_optimizer.pth')))

    def update_lr(self):
        self.lr /= 2
        for param_group in self.bert_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.prediction_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.rebuild_optimizer.param_groups:
            param_group['lr'] = self.lr

    def train(self, data_iter):
        prediction_loss = []
        rebuild_loss = []
        loss_list = []
        for (input, input_random, label) in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                # input_random = input_random.cuda()
                label = label.cuda()
            self.bert_optimizer.zero_grad()
            self.prediction_optimizer.zero_grad()
            self.rebuild_optimizer.zero_grad()

            output = self.bert(input)
            output = self.prediction(output)
            loss1 = last_loss(output, label)

            rebuild_output = self.bert(input)
            rebuild_output = self.rebuild(rebuild_output)
            loss2 = base_loss(rebuild_output, input)

            loss = loss1 + loss2

            prediction_loss.append(loss1.item())
            rebuild_loss.append(loss2.item())
            loss_list.append(loss.item())
            loss.backward()

            self.bert_optimizer.step()
            self.prediction_optimizer.step()
            self.rebuild_optimizer.step()

        avg_loss = np.asarray(loss_list).mean()
        prediction_loss = np.asarray(prediction_loss).mean()
        rebuild_loss = np.asarray(rebuild_loss).mean()
        print("prediction_loss:" + str(prediction_loss) + " rebuild loss:" + str(rebuild_loss))
        return avg_loss

    def test(self, data_iter):
        loss_list = []
        for (input, input_random, label) in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                # input_random = input_random.cuda()
                label = label.cuda()

            prediction_output, rebuild_output = self.forward(input)
            loss1 = last_loss(prediction_output, label)
            loss2 = base_loss(rebuild_output, input)

            loss = loss1 + loss2

            loss_list.append(loss.item())

        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def forward(self, x):
        output = self.bert(x)
        prediction_output = self.prediction(output)
        rebuild_output = self.rebuild(output)
        return prediction_output, rebuild_output


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(1152,618),
                                   )

    def forward(self, x):
        x = self.layer(x)
        return x


class Rebuild(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(1152,5307),
                                   )

    def forward(self, x):
        x = self.layer(x)
        return x
