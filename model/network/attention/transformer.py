import torch.nn as nn

from model.network.attention.multi_head import MultiHeadedAttention
from .sublayer import SublayerConnection
from .feed_forward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, id, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param id: the id of layer
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(id=id, h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout, )

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
