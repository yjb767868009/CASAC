import torch.nn as nn

from model.network.attention.multi_head import MultiHeadedAttention
from .sublayer import SublayerConnection
from .feed_forward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, id, save_path, hidden_dim, attention_head_nums, feed_forward_hidden, dropout):
        """
        :param id: the id of layer
        :param hidden_dim: hidden size of transformer
        :param attention_head_nums: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(id=id, save_path=save_path, attention_head_nums=attention_head_nums, hidden_dim=hidden_dim)
        self.feed_forward = PositionwiseFeedForward(dim=hidden_dim, hidden_dim=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden_dim, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout, )

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
