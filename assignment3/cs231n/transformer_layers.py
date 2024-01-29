import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # 使用torch.arange函数生成一个从0到max_len-1的序列。
        # torch.arange的第一个参数是起始值（inclusive），第二个参数是结束值（exclusive）
        # 第三个参数是步长（默认为1）。
        # dtype=torch.float表示生成的张量的数据类型为浮点型。
        # .unsqueeze(1)将生成的张量在维度1上添加一个维度，
        # 将其形状从(max_len,)变为(max_len, 1)。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
  
        # 这行代码使用torch.arange函数生成一个从0到embed_dim-1的序列，
        # 并且步长为2。然后，.float()将生成的张量转换为浮点型。
        # (-math.log(10000.0) / embed_dim)计算出一个常数值。
        # 最后，将序列乘以该常数值，并使用torch.exp对结果进行指数运算，
        # 得到一个包含位置编码的张量。
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        # 2的倍数切片操作0::2和1::2
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        output = self.dropout(x + self.pe[:,:S, :D])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads # H
        self.emd_dim = embed_dim # E
        self.head_dim = self.emd_dim // self.n_head #E/H

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # self.n_head = num_heads # H
        # self.emd_dim = embed_dim # E
        # self.head_dim = self.emd_dim // self.n_head #E/H
        
        query = self.query(query)
        # (N, S, H, E/H)
        query = query.view(N, S, self.n_head, self.head_dim)
        # (N, H, S, E/H)
        query = query.permute((0, 2, 1, 3)) 

        key = self.key(key)
        # (N, T, H, E/H)
        key = key.view(N, T, self.n_head, self.head_dim)
        # (N, H, E/H, T)
        key = key.permute((0, 2, 3, 1)) 

        value = self.value(value)
        # (N, T, H, E/H)
        value = value.view(N, T, self.n_head, self.head_dim)
        # (N, H, T, E/H)
        value = value.permute((0, 2, 1, 3)) 

        # Calculate attention weights
        # 注意公式里的d，代码中是E，d/h即E/H，为self.head_dim
        # (N, H, S, E/H)*(N, H, E/H, T) = (N, H, S, T)
        score = torch.matmul(query, key)/math.sqrt(self.head_dim)

        # 如果提供了注意力掩码（attn_mask），则修改得分以防止某些值影响输出。
        # 这通过使用 PyTorch 的 masked_fill 函数实现，
        # 其中对应于掩码位置的任何得分都被替换为一个非常小的值：float('-inf')
        if attn_mask is not None:
          score = score.masked_fill(attn_mask.view(1,1,*attn_mask.size())==0, float('-inf'))

        # (N, H, S, T)
        # 沿着最后一个维度（dim=-1）对注意力得分进行归一化，得到注意力权重
        attention_weights = F.softmax(score , dim=-1) 
        # dropout
        attention_weights = self.attn_drop(attention_weights)
        
        # (N, H, S, T)*(N, H, T, E/H) = (N, H, S, E/H)
        output = torch.matmul(attention_weights, value)
        # (N, S, H, E/H) -> (N, S, E)
        output = output.permute(0, 2, 1, 3).reshape(N, S, E)
        output = self.proj(output)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


