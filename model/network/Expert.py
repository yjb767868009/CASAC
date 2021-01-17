import numpy as np
import torch
import torch.nn as nn

from ..utils.activation_layer import activation_layer
import os


class Expert(nn.Module):
    def __init__(self, expert_nums, expert_dims, expert_activations, expert_dropout):
        super().__init__()
        self.expert_dims = expert_dims
        self.expert_activations = expert_activations
        self.expert_dropout = expert_dropout
        self.expert_nums = expert_nums
        self.layer_nums = len(expert_dims) - 1

        self.W = nn.ParameterList()
        self.B = nn.ParameterList()
        self.D = []
        self.A = []
        for i in range(self.layer_nums):
            w = self.init_weight((self.expert_nums, self.expert_dims[i + 1], self.expert_dims[i]))
            if torch.cuda.is_available():
                w = w.cuda()
            self.W.append(nn.Parameter(w))
            b = torch.zeros(self.expert_nums, self.expert_dims[i + 1], 1)
            if torch.cuda.is_available():
                b = b.cuda()
            self.B.append(nn.Parameter(b))
            self.D.append(nn.Dropout(p=expert_dropout))
            self.A.append(activation_layer(self.expert_activations[i]))

    def forward(self, weight_blend, x):
        for i in range(self.layer_nums):
            x = self.D[i](x)
            x = x.unsqueeze(-1)
            weight = self.get_wb(self.W[i], weight_blend)
            t = torch.bmm(weight, x)
            bias = self.get_wb(self.B[i], weight_blend)
            x = torch.add(t, bias)
            x = x.squeeze(-1)
            if self.A[i]:
                x = self.A[i](x)
        return x

    def init_weight(self, shape):
        a = np.sqrt(6. / np.prod(shape[-2:]))
        w = np.asarray(
            np.random.uniform(low=-a, high=a, size=shape),
            dtype=np.float32)
        return torch.tensor(w)

    def get_wb(self, x, weight_blend):
        """
        put weight blend in weight or bias

        :param x: weight or bis
        :param weight_blend: from last expert's weight blend
        :return: new weight or bias
        """
        batch_nums = weight_blend.size()[0]
        c = weight_blend.unsqueeze(-1).unsqueeze(-1)
        x_size = x.size()
        x = x.unsqueeze(0).expand(batch_nums, x_size[0], x_size[1], x_size[2])
        x = c * x
        return x.sum(dim=1)

    def save_network(self, expert_index, save_path):
        """
        save expert weight and bias for unity playing

        :param expert_index: this expert's index of all expert
        :param save_path: the root of save path
        :return: None
        """
        for i in range(self.layer_nums):
            for j in range(self.expert_nums):
                self.W[i][j, :, :].cpu().detach().numpy().tofile(
                    os.path.join(save_path, 'wc%0i%0i%0i_w.bin' % (expert_index, i, j)))
                self.B[i][j, :, :].cpu().detach().numpy().tofile(
                    os.path.join(save_path, 'wc%0i%0i%0i_b.bin' % (expert_index, i, j))
                )
