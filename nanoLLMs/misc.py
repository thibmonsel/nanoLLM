import os
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F


def inv_softplus(x):
    # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    return x + torch.log(-torch.expm1(-x))


def get_batch(
    data_dir, split, batch_size, device, block_size, openai_gpt2_tokenizer=False
):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    # Took the function code from https://github.com/karpathy/nanoGPT/blob/master/train.py
    if openai_gpt2_tokenizer:
        add_str = "_openai_tokenizer"
    else:
        add_str = ""

    if split == "train":
        data = np.memmap(
            os.path.join(data_dir, "train" + add_str + ".bin"),
            dtype=np.uint16,
            mode="r",
        )
    else:
        data = np.memmap(
            os.path.join(data_dir, "val" + add_str + ".bin"), dtype=np.uint16, mode="r"
        )
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device == "cuda":
        # pin arrays x,y, which allows us to move them
        #  to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokens_ids: torch.Tensor,
    generation_length: int,
    temperature: Optional[float] = 1.0,
    top_k: Optional[int] = None,
):
    for _ in range(generation_length):
        logits = model(tokens_ids[:, -model.block_size :])
        next_token_logits = logits[:, -1]
        probs = F.softmax(next_token_logits / temperature, dim=-1)

        if top_k is not None:
            values, _ = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = -float("Inf")
            probs = probs / torch.sum(probs, dim=1, keepdims=True)

        cur_tokens = torch.multinomial(probs, num_samples=1)
        tokens_ids = torch.cat([tokens_ids, cur_tokens], dim=-1)
    return tokens_ids
