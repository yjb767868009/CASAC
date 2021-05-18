import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .BaseModel import BaseModel
from model.utils import model2gpu
from model.utils import mask_last_loss


class BertPrediction(BaseModel):
    """
    Motion Bert Train Model
    Prediction Animation motion feature
    """

    def __init__(self, bert, lr):
        super().__init__()
        self.bert = model2gpu(bert)
        self.prediction = model2gpu(Prediction())
        self.lr = lr
        self.bert_optimizer = torch.optim.AdamW(self.bert.parameters(), lr=self.lr)
        self.prediction_optimizer = torch.optim.AdamW(self.prediction.parameters(), lr=self.lr)

    def train_init(self):
        self.bert.train()
        self.prediction.train()

    def test_init(self):
        self.bert.eval()
        self.prediction.eval()

    def save(self, save_path):
        # Save Model
        torch.save(self.bert.state_dict(), os.path.join(save_path, "bert.pth"))
        torch.save(self.prediction.state_dict(), os.path.join(save_path, "prediction.pth"))
        # Save optimizer
        torch.save(self.bert_optimizer.state_dict(), os.path.join(save_path, "bert_optimizer.pth"))
        torch.save(self.prediction_optimizer.state_dict(),
                   os.path.join(save_path, "prediction_optimizer.pth"))

    def load_param(self, load_path):
        # Load Model
        self.bert.load_state_dict(torch.load(os.path.join(load_path, 'bert.pth')))
        self.prediction.load_state_dict(torch.load(os.path.join(load_path, 'prediction.pth')))
        # Load optimizer
        self.bert_optimizer.load_state_dict(torch.load(os.path.join(load_path, 'bert_optimizer.pth')))
        self.prediction_optimizer.load_state_dict(
            torch.load(os.path.join(load_path, 'prediction_optimizer.pth')))

    def update_lr(self):
        self.lr /= 2
        for param_group in self.bert_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.prediction_optimizer.param_groups:
            param_group['lr'] = self.lr

    def train(self, data_iter):
        loss_list = []
        for (input, input_random, label) in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                # input_random = input_random.cuda()
                label = label.cuda()
            self.bert_optimizer.zero_grad()
            self.prediction_optimizer.zero_grad()

            output = self.forward(input)
            loss = mask_last_loss(output, label)
            loss_list.append(loss.item())
            loss.backward()

            self.bert_optimizer.step()
            self.prediction_optimizer.step()
        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def test(self, data_iter):
        loss_list = []
        for (input, input_random, label) in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                # input_random = input_random.cuda()
                label = label.cuda()

            output = self.forward(input)
            loss = mask_last_loss(output, label)
            loss_list.append(loss.item())

        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def forward(self,x):
        output = self.bert(x)
        output = self.prediction(output)
        return output


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(1792, 1024),
                                   nn.ELU(),
                                   nn.Linear(1024, 1024),
                                   nn.ELU(),
                                   nn.Linear(1024, 1024),
                                   nn.ELU(),
                                   nn.Linear(1024, 618),
                                   )

    def forward(self, x):
        x = self.layer(x)
        return x
