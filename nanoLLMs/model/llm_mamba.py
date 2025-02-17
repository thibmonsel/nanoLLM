"""Taken and modified from https://github.com/alxndrTL/mamba.py """


import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoLLMs.model.mamba import Mamba, MambaConfig, RMSNorm


class LLMMamba(nn.Module):
    def __init__(
        self,
        model_config: MambaConfig,
        vocab_size: int,
        pad_vocab_size_multiple: int = None,
    ):
        super().__init__()

        if pad_vocab_size_multiple != None and (
            vocab_size % pad_vocab_size_multiple != 0
        ):
            vocab_size += pad_vocab_size_multiple - vocab_size % pad_vocab_size_multiple

        self.config = model_config

        self.embedding = nn.Embedding(vocab_size, self.config.d_model, padding_idx=0)

        if isinstance(self.config, MambaConfig):
            self.mamba = Mamba(self.config)
        elif isinstance(self.config, Mamba2Config):
            self.mamba = Mamba2(self.config)
        else:
            raise NotImplementedError

        self.norm_f = RMSNorm(
            self.config.d_model, self.config.rms_norm_eps, self.config.mup
        )

        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        # self.embedding.weight = self.lm_head.weight # weight-tying disabled

        # muP custom initialization
        if self.config.mup and isinstance(self.config, MambaConfig):
            for pn, p in self.named_parameters():
                if any(
                    pn.endswith(w)
                    for w in [
                        "mixer.in_proj.weight",
                        "mixer.x_proj.weight",
                        "mixer.dt_proj.weight",
                        "mixer.out_proj.weight",
                    ]
                ):  # # "hidden weights"
                    std = self.config.base_std

                    if "mixer.out_proj.weight" in pn:
                        std = (
                            std / math.sqrt(2 * self.config.n_layers)
                        )  # scale down std of layers which projects onto the residual stream (not muP related)

                    if "mixer.dt_proj.weight" in pn:
                        std = self.config.dt_rank**-0.5 * self.config.dt_scale
                    torch.nn.init.normal_(
                        p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult)
                    )
                elif "mixer.conv1d.weight" in pn:
                    torch.nn.init.zeros_(p)
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std)
                elif pn == "lm_head.weight":
                    torch.nn.init.zeros_(p)
                elif any(pn.endswith(w) for w in ["mixer.A_log", "mixer.D"]):
                    # keep Mamba default init for these params
                    pass
                else:
                    # here, we only have biases
                    assert (
                        p.dim() == 1
                    ), f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if ("in_proj.bias" in pn) or ("out_proj.bias" in pn):
                        torch.nn.init.zeros_(p)

        else:
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith("mixer.out_proj.weight"):
                    torch.nn.init.normal_(
                        p,
                        mean=0.0,
                        std=self.config.base_std / math.sqrt(2 * self.config.n_layers),
                    )

    def forward(self, tokens):
        # tokens : (B, L)
        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)
        x = self.mamba(x)
        x = self.norm_f(x)

        if self.config.mup:
            x = x / self.config.mup_width_mult

        logits = self.lm_head(x)

        return logits

    def generate(
        self,
        prompt,
        num_tokens: int,
        sample: bool = True,
        top_k: int = None,
        temperature: float = 1.0,
    ):
        # prompt : (B, L)
        # generation : (B, l)
        # L>>l

        if top_k is not None:
            top_k = min(top_k, self.vocab_size)

        input_device = prompt.device
        prompt = prompt.to(self.embedding.weight.device)

        self.eval()
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(num_tokens):
                logits = self.forward(generated)  # (B, L, vocab_size)
                next_token_logits = logits[:, -1]

                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)

                    if top_k is not None:
                        values, _ = torch.topk(
                            probs, k=top_k
                        )  # (B, k) ordered from lowest to biggest
                        probs[
                            probs < values[:, -1, None]
                        ] = 0  # zero-out all probs except the k first
                        probs = probs / probs.sum(axis=1, keepdims=True)

                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        self.train()

        return generated.to(input_device)[:, -num_tokens:]

    # non-muP init (taken from llama2.c)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)

    # adaped from llama2.c
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        if self.config.mup and isinstance(self.config, MambaConfig):
            mup_params_keys = set(
                [
                    pn
                    for pn in param_dict.keys()
                    if any(
                        pn.endswith(w)
                        for w in [
                            "mixer.in_proj.weight",
                            "mixer.x_proj.weight",
                            "mixer.dt_proj.weight",
                            "mixer.out_proj.weight",
                        ]
                    )
                ]
            )

            dim2_params_keys = set(
                [pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2]
            )
            dim2_params_keys = dim2_params_keys.difference(mup_params_keys)

            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [
                p for n, p in param_dict.items() if p.dim() < 2
            ]  # biases and D

            optim_groups = [
                {
                    "params": mup_parameters,
                    "weight_decay": weight_decay * self.config.mup_width_mult,
                    "lr": learning_rate / self.config.mup_width_mult,
                },
                {
                    "params": decay_params,
                    "weight_decay": weight_decay,
                    "lr": learning_rate,
                },
                {"params": nodecay_params, "weight_decay": 0.0, "lr": learning_rate},
            ]

        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

            optim_groups = [
                {
                    "params": decay_params,
                    "weight_decay": weight_decay,
                    "lr": learning_rate,
                },
                {"params": nodecay_params, "weight_decay": 0.0, "lr": learning_rate},
            ]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, fused=use_fused)

        return optimizer
