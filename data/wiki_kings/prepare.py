import os

import numpy as np
from minbpe.minbpe.basic import BasicTokenizer

tokenizer = BasicTokenizer()
input_file_path = os.path.join(os.path.dirname(__file__), "kings.txt")
tokenizer_file_path = os.path.join(os.path.dirname(__file__), "kings_tokenizer.model")
tokenizer.load(tokenizer_file_path)

with open(input_file_path, "r") as f:
    data = f.read()

vocab_size = len(tokenizer.vocab)
data = tokenizer.encode(data)

# create the train and test splits
n = len(data)
train_ids = data[: int(n * 0.9)]
val_ids = data[int(n * 0.9) :]

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
