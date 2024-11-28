import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, factor=4, dropout_freq=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, factor * input_dim),
            # nn.GELU(),
            nn.ReLU(),
            nn.Linear(factor * input_dim, input_dim),
            nn.Dropout(dropout_freq),
        )

    def forward(self, x):
        return self.layers(x)
