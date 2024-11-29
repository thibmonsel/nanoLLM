"""
Shamelessly copied from https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare_char

Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import re
import os
import pickle
import tiktoken
import requests
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--openai_gpt2_tokenizer', type=bool, default=False)

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()

if argparser.parse_args().openai_gpt2_tokenizer:
    enc = tiktoken.get_encoding("r50k_base")
    vocab_size = enc.n_vocab
    data = enc.encode(data)
    
    # create the train and test splits
    n = len(data)
    train_ids = data[:int(n*0.9)]
    val_ids = data[int(n*0.9):]

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train_openai_tokenizer.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val_openai_tokenizer.bin'))

else : 
    pattern = r"\b\w+\b"
    # Find all words
    data = re.findall(pattern, data)

    # get all the unique characters that occur in this text
    words = sorted(list(set(data)))
    vocab_size = len(set(words))
    print("the first 100 unique word tokens are:", '|'.join(words[:100]))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stow = { ch:i for i,ch in enumerate(words) }
    itow = { i:ch for i,ch in enumerate(words) }
    def encode(s):
        return [stow[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itow[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itow': itow,
        'stow': stow,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
