import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from model.gpt2 import GPT2
from train import GPTTrainer
import numpy as np
import os 
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_batch(split, batch_size):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data_dir = "nanoGPT/data/shakespeare_char/"
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y




data_dir = "nanoGPT/data/shakespeare_char/"

meta_path = os.path.join(data_dir, 'meta.pkl')
vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    itos = meta['itos']
    stoi = meta['stoi']
    print(f"found vocab_size = {vocab_size} (inside {meta_path})")


# https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py
# python train.py config/train_shakespeare_char.py --device=cuda --compile=False --eval_iters=20 --log_interval=1 
# --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
encode = lambda s: torch.tensor([stoi[c] for c in s])
decode = lambda l: ''.join([itos[i] for i in l])

block_size = 64
n_layer, n_head, n_embd, dropout = 4, 4, 128, 0.0

model = GPT2(vocab_size, block_size, n_embd, n_head, n_layer, fast=False)
model.to(device)

trainer = GPTTrainer(model, 5e-4)
# trainer.train(get_batch, max_iters=2000, batch_size=64, patience=2000, save_every=30)

dic_load = trainer.load("last_model.pt")

model.eval()
model.load_state_dict(dic_load["state"])
x = torch.randint(0, vocab_size, (1, 1)).long().to(device)
x = encode("shall i compare thee to a summer's day?\n")
x = x.view(1, -1).to(device)
print(x)
y = model.generate(x, 4*256, temperature=1.0, top_k=None)

print(decode(y[0].tolist()))
# print([i for i in y.numpy()])
# print([itos[i] for i in y.numpy()])

plt.plot(trainer.losses, label='train')
plt.plot(trainer.val_losses, label='val')
plt.legend()
plt.savefig("loss.png")
