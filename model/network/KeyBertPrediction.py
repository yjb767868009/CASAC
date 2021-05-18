import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .BaseModel import BaseModel
from model.utils import model2gpu
from model.utils import mask_last_loss
from .attention.gelu import GELU


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
        self.lr /= 2
        for param_group in self.key_bert_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.key_prediction_optimizer.param_groups:
            param_group['lr'] = self.lr

    def test(self, data_iter):
        loss_list = []
        for (input, input_random, label) in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                # input_random = input_random.cuda()
                label = label.cuda()

            output = self.forward(input)
            loss = mask_last_loss(output, label[:, :, 606:618])
            loss_list.append(loss.item())

        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def train(self, data_iter):
        loss_list = []
        for (input, input_random, label) in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                # input_random = input_random.cuda()
                label = label.cuda()
            self.key_bert_optimizer.zero_grad()
            self.key_prediction_optimizer.zero_grad()

            output = self.forward(input)
            loss = mask_last_loss(output, label[:, :, 606:618])
            loss_list.append(loss.item())
            loss.backward()

            self.key_bert_optimizer.step()
            self.key_prediction_optimizer.step()
        avg_loss = np.asarray(loss_list).mean()
        return avg_loss

    def forward(self, x):
        output = self.key_bert(x)
        output = self.key_prediction(output)
        return output


class KeyPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(1792, 12),
                                   )

    def forward(self, x):
        x = self.layer(x)
        return x


"""
class LinearBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act=GELU,
                 norm=nn.LayerNorm, n_tokens=197):  # 197 = 16**2 + 1
        super().__init__()
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop = nn.Dropout(drop)
        # FF over features
        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act, drop=drop)
        self.norm1 = norm(dim)
        # FF over patches
        self.mlp2 = Mlp(in_features=n_tokens, hidden_features=int(n_tokens * mlp_ratio), act_layer=act, drop=drop)
        self.norm2 = norm(n_tokens)

    def forward(self, x):
        x = x + self.drop(self.mlp1(self.norm1(x)))
        x = x.transpose(-2, -1)
        x = x + self.drop(self.mlp2(self.norm2(x)))
        x = x.transpose(-2, -1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
"""
