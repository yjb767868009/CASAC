import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, dropout=None):
        """
        query,key,value:[batch_size,attention_heads_nums,seq_length,attention_head_size]
        scores:[batch_size,attention_heads_nums,seq_length,seq_length]
        对于scores[:,:,i,j]表示j对i的attention权重
        math.sqrt(query.size(-1)):点积模型的值通常有比较大方差，从而导致 softmax 函数的梯度会比较小。因此，缩放点积模型可以较好地解决这一问题
        return [batch_size,attention_heads_nums,seq_length,attention_head_size],scores.size()
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
