import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .BaseModel import BaseModel
from model.utils import model2gpu
from model.utils import mask_loss


class KeyBertPretrain(BaseModel):
    """
    key Bert pretrain Model
    pretrain Bert model's key feature
    """

    def __init__(self, key_bert, lr):
        super().__init__()
        self.key_bert = model2gpu(key_bert)
        self.key_pretrain = model2gpu(KeyPretrain())
        self.lr = lr
        self.key_bert_optimizer = torch.optim.AdamW(self.key_bert.parameters(), lr=self.lr)
        self.key_pretrain_optimizer = torch.optim.AdamW(self.key_pretrain.parameters(), lr=self.lr)

    def train_init(self):
        self.key_bert.train()
        self.key_pretrain.train()

    def test_init(self):
        self.key_bert.eval()
        self.key_pretrain.eval()

    def save(self, save_path):
        # Save Model
        torch.save(self.key_bert.state_dict(), os.path.join(save_path, "key_bert.pth"))
        torch.save(self.key_pretrain.state_dict(), os.path.join(save_path, "key_pretrain.pth"))
        # Save optimizer
        torch.save(self.key_bert_optimizer.state_dict(), os.path.join(save_path, "key_bert_optimizer.pth"))
        torch.save(self.key_pretrain_optimizer.state_dict(),
                   os.path.join(save_path, "key_pretrain_optimizer.pth"))

    def load_param(self, load_path):
        # Load Model
        self.key_bert.load_state_dict(torch.load(os.path.join(load_path, 'key_bert.pth')))
        self.key_pretrain.load_state_dict(torch.load(os.path.join(load_path, 'key_pretrain.pth')))
        # Load optimizer
        self.key_bert_optimizer.load_state_dict(torch.load(os.path.join(load_path, 'key_bert_optimizer.pth')))
        self.key_pretrain_optimizer.load_state_dict(torch.load(os.path.join(load_path, 'key_pretrain_optimizer.pth')))

    def update_lr(self):
        self.lr /= 10
        for param_group in self.key_bert_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.key_pretrain_optimizer.param_groups:
            param_group['lr'] = self.lr

    def test(self, data_iter):
        loss_list = []
        for (input, input_random, label), data_length in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                input_random = input_random.cuda()
                # label = label.cuda()

            output = self.forward(input_random, data_length)

            loss = mask_loss(output, input, data_length)
            loss_list.append(loss.item())

        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def train(self, data_iter):
        loss_list = []
        for (input, input_random, label), data_length in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                input_random = input_random.cuda()
                # label = label.cuda()
            self.key_bert_optimizer.zero_grad()
            self.key_pretrain_optimizer.zero_grad()

            output = self.forward(input_random, data_length)

            loss = mask_loss(output, input, data_length)
            loss_list.append(loss.item())
            loss.backward()

            self.key_bert_optimizer.step()
            self.key_pretrain_optimizer.step()
        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def forward(self, x, x_length):
        output = self.key_bert(x, x_length)
        output = self.key_pretrain(output, x_length)
        return output


class KeyPretrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Dropout(0.1), nn.Linear(1280, 5307), nn.ELU(), )

    def forward(self, x, x_length):
        x = self.layer(x)
        return x