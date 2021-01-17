import os

import numpy as np
import torch

from model.c_rnn_gan import conf
from model.utils.data_preprocess import get_norm
from model.utils.initialization import initialize_model


class Server(object):
    def __init__(self):
        self.base_model = initialize_model("BaseModel", conf['base_model'], (None, None))
        self.gan_model = initialize_model("GANModel", conf['gan_model'], (None, None))
        self.base_model.load_param()
        self.gan_model.load_param()
        self.data = torch.empty(0, 5307)
        self.full = False
        self.input_mean, self.input_std = get_norm("E:/NSM/data2/InputNorm.txt")
        self.output_mean, self.output_std = get_norm("E:/NSM/data2/OutputNorm.txt")

    def forward(self, x):
        x = np.array(x)
        x = (x - self.input_mean) / self.input_std
        x = torch.FloatTensor([x])
        self.data = torch.cat((self.data, x), 0)
        if self.full is True:
            self.data = self.data[1:]
            data_length = 200
        else:
            data_length = self.data.size(0)
            if data_length >= 200:
                self.full = True
        data = self.base_model.forward(self.data.unsqueeze(0), [data_length])
        data = self.gan_model.forward(data, [data_length])
        data = data[0][-1].cpu().detach().numpy()
        data = data * self.output_std + self.output_mean
        return data.tolist()
