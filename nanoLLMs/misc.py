import os

import numpy as np
import torch

"""
Took the function code from https://github.com/karpathy/nanoGPT/blob/master/train.py
"""


def get_batch(
    data_dir, split, batch_size, device, block_size, openai_gpt2_tokenizer=False
):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
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
