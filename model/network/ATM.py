import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .AT import AT
from .BaseModel import BaseModel
from model.utils import model2gpu
from model.utils import last_loss


class ATM(BaseModel):
    """
    Animation Transformer Model
    """

    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.at = model2gpu(AT())
        self.at_optimizer = torch.optim.AdamW(self.at.parameters(), lr=self.lr)

    def train_init(self):
        self.at.train()

    def test_init(self):
        self.at.eval()

    def save(self, save_path):
        # Save Model
        torch.save(self.at.state_dict(), os.path.join(save_path, "at.pth"))
        # Save optimizer
        torch.save(self.at_optimizer.state_dict(), os.path.join(save_path, "at_optimizer.pth"))

    def load_param(self, load_path):
        # Load Model
        self.at.load_state_dict(torch.load(os.path.join(load_path, 'at.pth')))
        # Load optimizer
        self.at_optimizer.load_state_dict(torch.load(os.path.join(load_path, 'at_optimizer.pth')))

    def update_lr(self):
        self.lr /= 2
        for param_group in self.at_optimizer.param_groups:
            param_group['lr'] = self.lr

    def ep(self, data_iter, train):
        loss_list = []
        for input, label in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()
            if train:
                self.at_optimizer.zero_grad()

            output = self.forward(input)
            loss = last_loss(output, label)
            loss_list.append(loss.item())
            loss.backward()
            if train:
                self.at_optimizer.step()
        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def forward(self, x):
        output = self.at(x)
        return output
