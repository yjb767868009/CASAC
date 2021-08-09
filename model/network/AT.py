import torch
import torch.nn as nn

from model.network.Embedding import Embedding
from model.network.attention.transformer import TransformerBlock
from model.bert.config import conf


class AT(nn.Module):
    """
    Animation Transformer
    """

    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.embedding = Embedding(conf["embedding"])
        self.bert = BERT(save_path, hidden=conf["hidden_dim"],
                         attention_head_nums=conf["attention_head_nums"], dropout=conf["bert_dropout"])
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

    def __init__(self, save_path, hidden=1280, transformer_nums=1, attention_head_nums=16, dropout=0.1):
        """
        :param hidden: BERT model hidden size
        :param transformer_nums: numbers of Transformer blocks(layers)
        :param attention_head_nums: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.save_path = save_path
        self.hidden = hidden
        self.n_layers = transformer_nums
        self.attn_heads = attention_head_nums

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(i, save_path, hidden, attention_head_nums, hidden * 4, dropout) for i in
             range(transformer_nums)])

    def forward(self, x):
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)
        return x


class Prediction(nn.Module):
    def __init__(self, hidden_dim, output_dim=618, ):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer(x)
