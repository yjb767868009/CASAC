import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .BaseModel import BaseModel
from model.utils import model2gpu
from model.utils import mask_last_loss


class KeyBertPrediction(BaseModel):
    """
    Key Bert Train Model
    Prediction Animation key feature
    """

    def __init__(self, key_bert, lr):
        super().__init__()
        self.key_bert = model2gpu(key_bert)
        self.key_prediction = model2gpu(KeyPrediction())
        self.lr = lr
        self.key_bert_optimizer = torch.optim.AdamW(self.key_bert.parameters(), lr=self.lr)
        self.key_prediction_optimizer = torch.optim.AdamW(self.key_prediction.parameters(), lr=self.lr)

    def train_init(self):
        self.key_bert.train()
        self.key_prediction.train()

    def test_init(self):
        self.key_bert.eval()
        self.key_prediction.eval()

    def save(self, save_path):
        # Save Model
        torch.save(self.key_bert.state_dict(), os.path.join(save_path, "key_bert.pth"))
        torch.save(self.key_prediction.state_dict(), os.path.join(save_path, "key_prediction.pth"))
        # Save optimizer
        torch.save(self.key_bert_optimizer.state_dict(), os.path.join(save_path, "key_bert_optimizer.pth"))
        torch.save(self.key_prediction_optimizer.state_dict(),
                   os.path.join(save_path, "key_prediction_optimizer.pth"))

    def load_param(self, load_path):
        # Load Model
        self.key_bert.load_state_dict(torch.load(os.path.join(load_path, 'key_bert.pth')))
        self.key_prediction.load_state_dict(torch.load(os.path.join(load_path, 'key_prediction.pth')))
        # Load optimizer
        self.key_bert_optimizer.load_state_dict(torch.load(os.path.join(load_path, 'key_bert_optimizer.pth')))
        self.key_prediction_optimizer.load_state_dict(
            torch.load(os.path.join(load_path, 'key_prediction_optimizer.pth')))

    def update_lr(self):
        self.lr /= 10
        for param_group in self.key_bert_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.key_prediction_optimizer.param_groups:
            param_group['lr'] = self.lr

    def train(self, data_iter):
        loss_list = []
        for (input, input_random, label), data_length in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                # input_random = input_random.cuda()
                label = label.cuda()
            self.key_bert_optimizer.zero_grad()
            self.key_prediction_optimizer.zero_grad()

            output = self.forward(input, data_length)
            loss = mask_last_loss(output, label[:, :, 606:618], data_length)
            loss.backward()

            self.key_bert_optimizer.step()
            self.key_prediction_optimizer.step()
        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def forward(self, x, x_length):
        output = self.key_bert(x, x_length)
        output = self.key_prediction(output, x_length)
        return output


class KeyPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_layer = nn.Sequential(nn.Linear(1280, 512), nn.ELU(), )
        self.mid_layer = nn.Sequential(nn.Linear(512, 64), nn.ELU(), )
        self.end_layer = nn.Sequential(nn.Linear(64, 12), )

    def forward(self, x, x_length):
        x = self.start_layer(x)
        x = self.mid_layer(x)
        x = self.end_layer(x)
        return x
