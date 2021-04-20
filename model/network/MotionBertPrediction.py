import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .BaseModel import BaseModel
from model.utils import model2gpu
from model.utils import mask_last_loss


class MotionBertPrediction(BaseModel):
    """
    Motion Bert Train Model
    Prediction Animation motion feature
    """

    def __init__(self, motion_bert, lr):
        super().__init__()
        self.motion_bert = model2gpu(motion_bert)
        self.motion_prediction = model2gpu(MotionPrediction())
        self.lr = lr
        self.motion_bert_optimizer = torch.optim.AdamW(self.motion_bert.parameters(), lr=self.lr)
        self.motion_prediction_optimizer = torch.optim.AdamW(self.motion_prediction.parameters(), lr=self.lr)

    def train_init(self):
        self.motion_bert.train()
        self.motion_prediction.train()

    def test_init(self):
        self.motion_bert.eval()
        self.motion_prediction.eval()

    def save(self, save_path):
        # Save Model
        torch.save(self.motion_bert.state_dict(), os.path.join(save_path, "motion_bert.pth"))
        torch.save(self.motion_prediction.state_dict(), os.path.join(save_path, "motion_prediction.pth"))
        # Save optimizer
        torch.save(self.motion_bert_optimizer.state_dict(), os.path.join(save_path, "motion_bert_optimizer.pth"))
        torch.save(self.motion_prediction_optimizer.state_dict(),
                   os.path.join(save_path, "motion_prediction_optimizer.pth"))

    def load_param(self, load_path):
        # Load Model
        self.motion_bert.load_state_dict(torch.load(os.path.join(load_path, 'motion_bert.pth')))
        self.motion_prediction.load_state_dict(torch.load(os.path.join(load_path, 'motion_prediction.pth')))
        # Load optimizer
        self.motion_bert_optimizer.load_state_dict(torch.load(os.path.join(load_path, 'motion_bert_optimizer.pth')))
        self.motion_prediction_optimizer.load_state_dict(
            torch.load(os.path.join(load_path, 'motion_prediction_optimizer.pth')))

    def update_lr(self):
        self.lr /= 2
        for param_group in self.motion_bert_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.motion_prediction_optimizer.param_groups:
            param_group['lr'] = self.lr

    def train(self, data_iter):
        loss_list = []
        for (input, input_random, label), data_length in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                # input_random = input_random.cuda()
                label = label.cuda()
            self.motion_bert_optimizer.zero_grad()
            self.motion_prediction_optimizer.zero_grad()

            output = self.forward(label[:, :, 606:], input, data_length)
            loss = mask_last_loss(output, label[:, :, :606], data_length)
            loss_list.append(loss.item())
            loss.backward()

            self.motion_bert_optimizer.step()
            self.motion_prediction_optimizer.step()
        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def test(self, data_iter):
        loss_list = []
        for (input, input_random, label), data_length in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                # input_random = input_random.cuda()
                label = label.cuda()

            output = self.forward(label[:, :, 606:], input, data_length)
            loss = mask_last_loss(output, label[:, :, :606], data_length)
            loss_list.append(loss.item())
            loss.backward()

        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def forward(self, key, x, x_length):
        output = self.motion_bert(key, x, x_length)
        output = self.motion_prediction(output, x_length)
        return output


class MotionPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1296, 606)

    def forward(self, x, x_length):
        x = self.layer(x)
        return x