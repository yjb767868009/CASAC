import torch
import torch.nn as nn

from model.network.Embedding import Embedding
from model.network.attention.transformer import TransformerBlock


class MotionBERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, encoder_nums, encoder_dims, encoder_activations, encoder_dropout, segmentation,
                 hidden=1296, n_layers=16, attn_heads=16, dropout=0.1):
        """
        :param input_size: input_size
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

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = Embedding(encoder_nums, encoder_dims, encoder_activations, encoder_dropout, segmentation)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        self.key_layer=nn.Linear(12,16)

    def forward(self, key, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)

        assert (key.size(0) == x.size(0)) and (key.size(1) == x.size(1)), 'key size not equal x size'

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        key = self.key_layer(key)
        x = torch.cat([x, key], 2)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, None)

        return x
