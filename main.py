import argparse
import os

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from nanoLLMs.misc import generate_text, get_batch, get_batch_hf_dataset
from nanoLLMs.model import GPT2
from nanoLLMs.trainer import Trainer
from transformers import GPT2TokenizerFast

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training LLM on Wikipedia scrapped data"
    )
    parser.add_argument("--data_dir_path", type=str)
    parser.add_argument("--model", choices=["mamba", "gpt", "xlstm"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")

    # Loading trained tokenizer
    # tokenizer = BasicTokenizer()
    # tokenizer_file_path = os.path.join(os.path.dirname(__file__), str(args.data_dir_path), "kings_tokenizer.model")
    # tokenizer.load(tokenizer_file_path)
    # vocab_size = len(tokenizer.vocab)

    data_dir = os.path.join(os.path.dirname(__file__), str(args.data_dir_path))
    partial_get_batch = lambda split, batch_size: get_batch_hf_dataset(
        ds, split, batch_size, device, block_size, tokenizer
    )

    if args.model == "gpt":
        block_size = 256
        n_layer, n_head, n_embd, dropout = 6, 6, 384, 0.0
        model = GPT2(vocab_size, block_size, n_embd, n_head, n_layer)
        model.to(device)
    elif args.model == "mamba":
        pass
    elif args.model == "xlstm":
        pass
    else:
        raise ValueError

    # training hyperparameters
    lr, ddp, checkpoint_path, wd = 1e-4, True, "metadata/wikimedia/", 1e-7
    max_iters, batch_size, patience, save_every = 2, 128, 10000, 500
    # generation hyperparameters
    generation_length, temperature = 256, 1.0

    trainer = Trainer(model, lr, ddp=ddp, checkpoint_path=checkpoint_path, wd=wd)
    trainer.train(
        get_batch,
        max_iters=max_iters,
        batch_size=batch_size,
        patience=patience,
        save_every=save_every,
    )

    save_filename = (
        trainer.checkpoint_path + trainer.llm_model.module.__class__.name + "_loss.png"
        if ddp
        else trainer.checkpoint_path + trainer.llm_model.__class__.name + "_loss.png"
    )
    plt.plot(trainer.losses, label="train")
    plt.plot(trainer.val_losses, label="val")
    plt.legend()
    plt.savefig(save_filename)

    # dic_load = trainer.load("last_model.pt")

    x = tokenizer.encode("The moon landing ")
    x = x.view(1, -1).to(device)
    y = generate_text(trainer.llm_model, x, generation_length, temperature=temperature)
    print("Generated sample :", tokenizer.decode(y[0].tolist()))
