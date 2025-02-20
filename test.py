import os
import pickle

import matplotlib.pyplot as plt
import torch
from nanoLLMs.misc import get_batch
from nanoLLMs.misc import generate_text
from nanoLLMs.model import GPT2, Mamba, mLSTM
from nanoLLMs.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "data/shakespeare_char/"
meta_path = os.path.join(data_dir, "meta.pkl")
vocab_size = None

if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    itos = meta["itos"]
    stoi = meta["stoi"]
    print(f"found vocab_size = {vocab_size} (inside {meta_path})")

encode = lambda s: torch.tensor([stoi[c] for c in s])
decode = lambda l: "".join([itos[i] for i in l])

shakpeare_lines = "the king will"
x = encode(shakpeare_lines)
y = decode(x.tolist())
assert shakpeare_lines == y

print("The text '{}' is encoded and fed to the GPT as {}".format(shakpeare_lines, x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model, expand, kernel_size, conv_bias, dt_rank, d_state, bias, dt_max, dt_min, dt_init_floor, dt_scale, n_layers, dt_init, use_cuda = 32, 2, 4, False, "auto", 16,  True, 0.1, 0.001, 1e-4, 1.0, 6, "random", True
model = Mamba(d_model, expand, kernel_size, conv_bias, dt_rank, d_state, bias, dt_max, dt_min, dt_init_floor, dt_scale, dt_init, use_cuda, n_layers, vocab_size=vocab_size, block_size=256)
model = model.to(device)

max_iters, batch_size = 1000, 32
block_size = 64
get_batch_fn = lambda split, batch_size: get_batch(
    data_dir, split, batch_size, device, block_size
)

trainer = Trainer(model, lr=1e-3, checkpoint_path="../metadata/shakespeare_char/")

partial_get_batch = lambda split, batch_size: get_batch(
    data_dir, split, batch_size, device, block_size
)
trainer.train(
    partial_get_batch, max_iters=500, batch_size=2*128, patience=2000, save_every=200
)