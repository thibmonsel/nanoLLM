import math

import torch
import torch.nn as nn


class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input, forget, and output gates parameters
        self.w_i = nn.Linear(input_size, hidden_size)
        self.w_f = nn.Linear(input_size, hidden_size)
        self.w_o = nn.Linear(input_size, hidden_size)

        self.w_q = nn.Linear(input_size, hidden_size)
        self.w_k = nn.Linear(input_size, hidden_size)
        self.w_v = nn.Linear(input_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.w_i, self.w_f, self.w_o, self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_uniform_(w.weight)
            nn.init.zeros_(w.bias)

    def forward(self, input, hx):
        h, c, n = hx
        # input gate (Eq. 25, p.5)
        i_t = torch.exp(self.w_i(input))
        # forget gate (Eq. 26, p.5)
        f_t = torch.sigmoid(self.w_f(input))
        # output gate (Eq. 27, p.5)
        o_t = torch.sigmoid(self.w_o(input))

        # Compute query, key, value (Eq. 22, 23, 24, p.5)
        q_t = self.w_q(input)
        k_t = self.w_k(input) / math.sqrt(self.hidden_size)
        v_t = self.w_v(input)

        # update cell state (Eq. 19, p.5)
        c = f_t * c + i_t * (v_t * k_t)  # cell_state
        # normalizer state (Eq. 20, p.5)
        n = f_t * n + i_t * k_t  # normalizer_state

        # compute hidden state (Eq.21, p.5)
        h_tilde = c * q_t
        denom = torch.clamp(torch.abs(n * q_t), min=1.0)
        h = o_t * (h_tilde / denom)

        # return hidden state t, cell state c and normalizer state n
        return h, c, n


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                mLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input, hidden_state=None):
        bs, seq_len, _ = input.size()
        if hidden_state is None:
            hidden_state = [
                (
                    torch.zeros(bs, self.hidden_size, device=input.device),
                    torch.zeros(bs, self.hidden_size, device=input.device),
                    torch.zeros(bs, self.hidden_size, device=input.device),
                )
                for _ in range(self.num_layers)
            ]

        outputs = []
        for t in range(seq_len):
            x = input[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, c, n = hidden_state[layer_idx]
                h, c, n = layer(x, (h, c, n))
                hidden_state[layer_idx] = (h, c, n)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)

        return torch.stack(outputs, dim=1), hidden_state


class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = mLSTM(input_size, hidden_size, num_layers, dropout)

        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden_state=None):
        lstm_output, hidden_state = self.lstm(input, hidden_state)
        output = self.act(lstm_output)
        output = self.norm(output)
        output = self.proj(output)
        output = self.dropout_layer(output + input)
        return output, hidden_state


class xLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        block_size,
        hidden_size,
        num_layers,
        num_blocks,
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                xLSTMBlock(embed_dim, hidden_size, num_layers, dropout)
                for _ in range(self.num_blocks)
            ]
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, input, hidden_state=None):
        embed_seq = self.embedding(input)
        if hidden_state is None:
            hidden_state = [None] * self.num_blocks
        output_seq = embed_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_state[i] = block(output_seq, hidden_state[i])
        output_seq = self.output_layer(output_seq)
        return output_seq
        # return output_seq, hidden_state
