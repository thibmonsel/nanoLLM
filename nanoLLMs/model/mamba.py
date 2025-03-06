"""Inspired from https://github.com/alxndrTL/mamba.py"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from jaxtyping import Float
from mambapy.pscan import pscan

from nanoLLMs.misc import inv_softplus

""" 
Dimension inventory : 

B : batch size
L : sequence length size
D / `d_model` : number of channels in our input x(t)
E / `expand`: expansion factor (Section 3.4)
N / `d_state`: latent state size h(t)
kernel_size : convolution filter size in Mamba block
"""


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        expand: int,
        kernel_size: int,
        conv_bias: bool,
        dt_rank: Union[int, str],
        d_state: int,
        bias: bool,
        dt_max: float,
        dt_min: float,
        dt_init_floor: float,
        dt_scale: float,
        dt_init: str,  # "constant" or "random"
        use_cuda: bool,
    ):
        super().__init__()

        self.d_model = d_model
        self.expand = expand
        self.conv_bias = conv_bias
        self.kernel_size = kernel_size
        if isinstance(dt_rank, str):
            if self.dt_rank == "auto":
                self.dt_rank: int = math.ceil(self.d_model / 16)
            else:
                raise ValueError
        else:
            self.dt_rank: int = dt_rank
        self.d_state = d_state
        self.bias = bias

        self.dt_max = dt_max
        self.dt_min = dt_min
        self.dt_init_floor = dt_init_floor
        self.dt_scale = dt_scale
        self.dt_init = dt_init

        self.use_cuda = use_cuda
        # To add as arg everywhere
        self.pscan = True
        # projects block input from D to 2*ED (two branches)
        # this correspond to the first 2 linear projection on the mamba block Figure 3
        self.in_proj = nn.Linear(
            self.d_model, 2 * self.expand * self.d_model, bias=self.conv_bias
        )

        # creating our depthwise convolution layer
        # this correspond to convolution layer in mamba block Figure 3
        # done on each channel of the input
        self.conv1d = nn.Conv1d(
            in_channels=self.expand * self.d_model,
            out_channels=self.expand * self.d_model,
            kernel_size=self.kernel_size,
            bias=self.conv_bias,
            groups=self.expand * self.d_model,
            padding=self.kernel_size - 1,
        )

        # projects x to input-dependent delta, B, C
        # one merged linear layer for s_B(x), s_C(x) and s_Δ(x)
        self.x_proj = nn.Linear(
            self.expand * self.d_model, self.dt_rank + 2 * self.d_state, bias=False
        )

        # projects delta from dt_rank to self.expand * self.d_model
        self.dt_proj = nn.Linear(self.dt_rank, self.expand * self.d_model, bias=True)
        # initialize values of
        self.initialize_dt_proj_parameters()

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1), "n -> d n", d=self.expand * self.d_model
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore

        self.D = nn.Parameter(torch.ones(self.expand * self.d_model))
        self.D._no_weight_decay = True  # type: ignore

        # projects block output from ED back to D
        self.out_proj = nn.Linear(
            self.expand * self.d_model, self.d_model, bias=self.bias
        )

        if self.use_cuda:
            try:
                # importing "official mamba scan" implemented by A.Gu
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.use_cuda = False

    def initialize_dt_proj_parameters(self):
        # initializing weight
        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        if self.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif self.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # initializing bias
        dt = torch.exp(
            torch.rand(self.expand * self.d_model)
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)

        inv_dt = inv_softplus(dt)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: Float[torch.Tensor, "B L D"]) -> Float[torch.Tensor, "B L D"]:
        _, L, _ = x.shape
        xz = self.in_proj(x)
        # splitting the projection in 2 chuncks of [B, L, ED]
        x, z = xz.chunk(2, dim=-1)
        # left branch of Figure 3
        x = x.transpose(1, 2)
        # depthwise convolution over dimension of sequence length L
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x, z)

        if self.use_cuda:
            # the rest of the operations are done in the ssm function
            # (fused with the CUDA pscan)
            output = self.out_proj(y)
            return output

        # right branch
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)

        return output

    def ssm(
        self,
        x: Float[torch.Tensor, "B L ED"],
        z: Float[torch.Tensor, "B L ED"],
    ) -> Float[torch.Tensor, "B L ED"]:
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        # projection the merge linear layer to get the transformation of B, C and Δ
        Δ, B, C = torch.split(
            self.x_proj(x),
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        Δ = torch.einsum("ij,blj->bil", self.dt_proj.weight, Δ)

        # choose which selective_scan function to use, according to config
        if self.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(
                x,
                Δ,
                A,
                B,
                C,
                D,
                z=z,
                delta_softplus=True,
                delta_bias=self.dt_proj.bias.float(),
            )
            y = y.transpose(1, 2)

        else:
            Δ = Δ.transpose(1, 2)
            Δ = F.softplus(Δ + self.dt_proj.bias)

            if self.pscan:
                y = self.selective_scan(x, Δ, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, Δ, A, B, C, D)

        return y

    def selective_scan(
        self,
        x: Float[torch.Tensor, "B L ED"],
        Δ: Float[torch.Tensor, "B L ED"],
        A: Float[torch.Tensor, "ED N"],
        B: Float[torch.Tensor, "B L ED"],
        C: Float[torch.Tensor, "B L ED"],
        D: Float[torch.Tensor, " ED"],
    ) -> Float[torch.Tensor, "B L ED"]:
        ΔA = torch.exp(Δ.unsqueeze(-1) * A)  # (B, L, ED, N)
        ΔB = Δ.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        stacked_B = ΔB * (x.unsqueeze(-1))  # (B, L, ED, N)
        # compute all hidden states
        hs = pscan(ΔA, stacked_B)
        # y_t = C h_t
        y = torch.einsum("bln, blen -> ble", C, hs)
        # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x
        return y

    def selective_scan_seq(
        self,
        x: Float[torch.Tensor, "B L ED"],
        Δ: Float[torch.Tensor, "B L ED"],
        A: Float[torch.Tensor, "ED H"],
        B: Float[torch.Tensor, "B L ED"],
        C: Float[torch.Tensor, "B L ED"],
        D: Float[torch.Tensor, " ED"],
    ) -> Float[torch.Tensor, "B L ED"]:
        _, L, _ = x.shape

        ΔA = torch.exp(Δ.unsqueeze(-1) * A)  # (B, L, ED, N)
        ΔB = Δ.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        B_stacked = ΔB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(
            x.size(0), self.expand * self.d_model, self.d_state, device=ΔA.device
        )

        hs = []
        for t in range(0, L):
            h = ΔA[:, t] * h + B_stacked[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)

        y = torch.einsum("bln, blen -> ble", C, hs).unsqueeze(-1)
        y = y + D * x

        return y

    """
    The step method and ssm_step are used for the generating 
    text with a recurrence call
    """

    def step(
        self,
        x: Float[torch.Tensor, "B D"],
        cache: Tuple[
            Float[torch.Tensor, "B ED N"], Float[torch.Tensor, "B ED d_conv-1"]
        ],
    ) -> Tuple[
        Float[torch.Tensor, "B D"],
        Tuple[Float[torch.Tensor, "B D"], Float[torch.Tensor, "B ED d_conv-1"]],
    ]:
        h, inputs = cache

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[
            :, :, self.kernel_size - 1
        ]  # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  # (B, ED, d_conv-1)

        return output, (h, inputs)

    def ssm_step(
        self, x: Float[torch.Tensor, "B ED"], h: Float[torch.Tensor, "B ED N"]
    ):
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        # projection the merge linear layer to get the transformation of B, C and Δ
        Δ, B, C = torch.split(
            self.x_proj(x),
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        Δ = self.dt_proj.weight @ Δ.transpose(1, 2)
        Δ = Δ.transpose(1, 2)
        Δ = F.softplus(Δ + self.dt_proj.bias)

        ΔA = torch.exp(Δ.unsqueeze(-1) * A)  # (B, ED, N)
        ΔB = Δ.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        if h is None:
            h = torch.zeros(
                x.size(0),
                self.expand * self.d_model,
                self.d_state,
                device=ΔA.device,
            )  # (B, ED, N)

        h = ΔA * h + ΔB * (x.unsqueeze(-1))

        y = (h @ C.unsqueeze(-1)).squeeze(2)
        y = y + D * x

        return y, h


class Mamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        expand: int,
        kernel_size: int,
        conv_bias: bool,
        dt_rank: Union[int, str],
        d_state: int,
        bias: bool,
        dt_max: float,
        dt_min: float,
        dt_init_floor: float,
        dt_scale: float,
        dt_init: str,  # "constant" or "random"
        use_cuda: bool,
        n_layers: int,
        vocab_size: int,
        block_size: int,
        pad_vocab_size_multiple: Optional[int] = None,
    ):
        super().__init__()

        if pad_vocab_size_multiple is not None and (
            vocab_size % pad_vocab_size_multiple != 0
        ):
            vocab_size += pad_vocab_size_multiple - vocab_size % pad_vocab_size_multiple

        self.d_model = d_model
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    expand,
                    kernel_size,
                    conv_bias,
                    dt_rank,
                    d_state,
                    bias,
                    dt_max,
                    dt_min,
                    dt_init_floor,
                    dt_scale,
                    dt_init,
                    use_cuda,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model, eps=1e-5, use_mup=False)

        self.norm_f = RMSNorm(d_model, 1e-5, False)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)

    def forward(
        self, tokens: Float[torch.Tensor, "B L"]
    ) -> Float[torch.Tensor, "B L vocab_size"]:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(self.norm(x)) + x
        x = self.norm_f(x)
        return self.lm_head(x)

    def step(
        self,
        x: Float[torch.Tensor, "B L D"],
        caches: list[
            Tuple[Float[torch.Tensor, "B ED N"], Float[torch.Tensor, "B ED d_conv-1"]]
        ],
    ) -> Tuple[
        Float[torch.Tensor, "B L D"],
        list[
            Tuple[Float[torch.Tensor, "B ED N"], Float[torch.Tensor, "B ED d_conv-1"]]
        ],
    ]:
        for i, layer in enumerate(self.layers):
            output, caches[i] = layer(self.norm(x), caches[i])
            x = output + x

        return x, caches


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728,
        # RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output
