import torch
import torch.nn as nn
from torch.nn import functional as F
from model.attention import MultiAttentionHead, MultiHeadSelfAttention
from model.misc import MLP
from tqdm import tqdm


class TransformerBlock(nn.Module):

    def __init__(
        self,
        n_heads,
        dim_emb,
        max_seq_len,
        ff_factor=4,
        att_dropout=0.0,
        resid_dropout=0.0,
        fast=True,
    ):
        super().__init__()
        if fast:
            self.attention = MultiHeadSelfAttention(
                n_heads, dim_emb, max_seq_len, att_dropout, resid_dropout
            )
        else:
            self.attention = MultiAttentionHead(
                n_heads, dim_emb, max_seq_len, att_dropout
            )
        self.norm1 = nn.LayerNorm(dim_emb)
        self.mlp = MLP(dim_emb, ff_factor, resid_dropout)
        self.norm2 = nn.LayerNorm(dim_emb)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT2(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        dim_emb,
        n_heads,
        n_layers,
        emb_dropout_freq=0.0,
        fast=True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim_emb)
        self.pos_emb = nn.Embedding(max_seq_len, dim_emb)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(n_heads, dim_emb, max_seq_len, fast=fast)
                for _ in range(n_layers)
            ]
        )
        self.emb_drop = nn.Dropout(emb_dropout_freq)
        # Linear projection on vocab size to have most likely next token
        self.final_linear = nn.Linear(dim_emb, vocab_size)

    def forward(self, x):
        assert (
            x.shape[1] <= self.max_seq_len
        ), f"Cannot forward sequence of length {x.shape[0]}, block size is only {self.max_seq_len}"
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        x = self.emb_drop(token_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        return self.final_linear(x)

    @torch.no_grad()
    def generate(self, x, generation_length, temperature=1.0, top_k=None):
        for _ in tqdm(range(generation_length)):
            logits = self(x[:, -self.max_seq_len :])[:, -1]
            # scale the logits with temperature
            logits /= temperature
            # optionally crop the logits to only show top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # convert logits to probabilities
            prob = F.softmax(logits, dim=-1)
            # sample from the distribution
            cur_tokens = torch.multinomial(prob, num_samples=1)
            x = torch.cat([x, cur_tokens], dim=-1)
        return x
