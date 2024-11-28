import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class AttentionHead(nn.Module):

    def __init__(self, dim_emb, dim_k, dim_v, max_seq_len, dropout=0.0):
        """
        Implementation of a single attention head non-optimized version
        from Attention is All You Need paper

        dim_emb: embedding dimension
        dim_k: dimension of the queries and keys matrices
        dim_v: dimension of the values matrices
        max_seq_len: maximum sequence length or context length/window
        dropout: attention dropout rate
        """
        super(AttentionHead, self).__init__()
        self.dim_emb = dim_emb

        self.K = nn.Linear(dim_emb, dim_k, bias=False)
        self.Q = nn.Linear(dim_emb, dim_k, bias=False)
        self.V = nn.Linear(dim_emb, dim_v, bias=False)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.att_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x : (batch_size, seq_len, dim_emb)
        seq_len = x.shape[1]
        queries = self.Q(x)  # (batch_size, seq_len, dim_k)
        keys = self.K(x)  # (batch_size, seq_len, dim_k)
        values = self.V(x)  # (batch_size, seq_len, dim_v)

        qk = torch.einsum("bqd,bkd->bqk", queries, keys) / (
            self.dim_emb**0.5
        )  # (batch_size, seq_len, seq_len)
        masked_qk = qk.masked_fill(self.mask[:seq_len, :seq_len] == 0, -torch.inf)
        softmax_masked_qk = F.softmax(masked_qk, dim=-1)
        att = self.att_dropout(softmax_masked_qk)
        return att @ values  # (batch_size, seq_len, dim_v)


class MultiAttentionHead(nn.Module):

    def __init__(
        self, n_heads, dim_emb, max_seq_len, att_dropout=0.0, resid_dropout=0.0
    ):
        super().__init__()
        """ 
        This is a rather inefficient implementation of multi-head attention.
        """
        assert dim_emb % n_heads == 0
        dim_k_and_v = dim_emb // n_heads
        self.att_heads = nn.ModuleList(
            [
                AttentionHead(
                    dim_emb, dim_k_and_v, dim_k_and_v, max_seq_len, att_dropout
                )
                for _ in range(n_heads)
            ]
        )
        self.projector = nn.Linear(dim_emb, dim_emb)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x):
        # x : (batch_size, seq_len, dim_emb)
        concat_heads = torch.cat(
            [head(x) for head in self.att_heads], dim=-1
        )  # (batch_size, seq_len, dim_v * n_heads = dim_emb)
        return self.resid_dropout(
            self.projector(concat_heads)
        )  # (batch_size, seq_len, dim_emb)


class MultiHeadSelfAttention(nn.Module):
    """
    A more efficient implementation of multi-head attention.
    """

    def __init__(
        self, n_heads, dim_emb, max_seq_len, att_dropout=0.0, resid_dropout=0.0
    ):
        super().__init__()
        assert dim_emb % n_heads == 0

        # key, query, value projections for all heads, but in a batch
        self.att_weights = nn.Linear(dim_emb, 3 * dim_emb, bias=False)
        # output projection
        self.output_proj = nn.Linear(dim_emb, dim_emb)
        # regularization
        self.attn_dropout = nn.Dropout(att_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.n_heads = n_heads
        self.dim_emb = dim_emb
        self.dropout = nn.Dropout(att_dropout)
        self.flash = hasattr(F, "scaled_dot_product_attention")

        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                    1, 1, max_seq_len, max_seq_len
                ),
            )

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size()[:-2] + (self.hidden_dim,))
        return x

    def forward(self, x):
        seq_len = x.shape[1]
        q, k, v = self.att_weights(x).split(self.dim_emb, dim=2)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, -torch.inf)
            att = self.softmax(att)
            att = self.attn_dropout(att)
            y = torch.matmul(att, v)

        y = self.merge_heads(y)
        # output projection
        y = self.resid_dropout(self.output_proj(y))
        return y


# if __name__ == "__main__":
#     att_layer = EfficientMultiAttentionHead(8, 512, 100, 0.1, 0.1)
#     x = torch.randn(32, 100, 512)
#     y = att_layer(x)
#     print(y.shape)
