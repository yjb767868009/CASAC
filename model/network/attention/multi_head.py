import torch.nn as nn
from model.network.attention.single import Attention
from torch.utils.tensorboard import SummaryWriter

import torchvision


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, id, save_path, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.id = id
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

        self.writer = SummaryWriter(save_path + '/log')

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, dropout=self.dropout)

        # attn = attn.transpose(0, 1)
        # img_grid = torchvision.utils.make_grid(attn, normalize=True, scale_each=True, nrow=4)
        # self.writer.add_image('Attention/attention_%s' % self.id, img_grid, global_step=0)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
