import os
import pickle

import matplotlib.pyplot as plt
import torch
from nanoLLMs.misc import get_batch
from nanoLLMs.model.gpt2 import GPT2
from nanoLLMs.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "nanoGPT/data/shakespeare_char/"

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

block_size = 256
n_layer, n_head, n_embd, dropout = 6, 6, 384, 0.0

partial_get_batch = lambda split, batch_size: get_batch(
    data_dir, split, batch_size, device, block_size
)

model = GPT2(vocab_size, block_size, n_embd, n_head, n_layer)
model.to(device)

trainer = Trainer(
    model, 1e-3, parallel=True, checkpoint_path="metadata/shakespeare_char/", wd=0.0
)

trainer.train(get_batch, max_iters=5000, batch_size=128, patience=5000, save_every=200)

dic_load = trainer.load("last_model.pt")

x = torch.randint(0, vocab_size, (1, 1)).long().to(device)
x = encode("shall i compare thee to a summer's day?\n")
x = x.view(1, -1).to(device)
y = trainer.gpt_model.generate(x, 256)

print(x, decode(y[0].tolist()))
plt.plot(trainer.losses, label="train")
plt.plot(trainer.val_losses, label="val")
plt.legend()
plt.savefig(trainer.checkpoint_path + "loss.png")
