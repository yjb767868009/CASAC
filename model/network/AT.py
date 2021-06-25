import torch
import torch.nn as nn

from model.network.Embedding import Embedding
from model.network.attention.transformer import TransformerBlock
from model.bert.config import conf


class AT(nn.Module):
    """
    Animation Transformer
    """

    def __init__(self):
        super().__init__()
        self.embedding = Embedding(conf["embedding"])
        self.bert = BERT(conf["hidden_dim"], dropout=conf["bert_dropout"])
        self.prediction = Prediction(conf["hidden_dim"])

    def forward(self, x):
        y = self.embedding(x)
        y = self.bert(y)
        y = self.prediction(y)
        return y


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden=1280, n_layers=16, attn_heads=16, dropout=0.1):
        """
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, None)
        return x


class Prediction(nn.Module):
    def __init__(self, hidden_dim, output_dim=618, ):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer(x)
