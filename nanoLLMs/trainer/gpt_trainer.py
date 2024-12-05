import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# TO DO DDP
# https://jacksoncakes.com/2023/08/20/getting-started-with-distributed-data-parallel-in-pytorch-a-beginners-guide/
# https://github.com/IDRIS-CNRS/DLO-JZ/blob/main/Jour4/tp_gros_models/dlojz.py


class GPTTrainer:
    def __init__(self, gpt_model, lr, checkpoint_path="", wd=0.0):
        self.gpt_model = gpt_model
        self.opt = torch.optim.Adam(self.gpt_model.parameters(), lr=lr, weight_decay=wd)
        self.checkpoint_path = checkpoint_path

        self.losses = []
        self.val_losses = []
        self.parallel = False

        if checkpoint_path.endswith("/"):
            os.makedirs(checkpoint_path, exist_ok=True)
            print(f"Created checkpoint directory at {checkpoint_path}")

    def train(
        self,
        get_batch_fn,
        batch_size,
        max_iters=1000,
        patience=50,
        save_every=1,
    ):
        counter, best_val_step = 0, 0
        best_val_loss = float(np.inf)
        pbar = tqdm(range(max_iters))
        for i in pbar:
            inputs, targets = get_batch_fn("train", batch_size)
            # Train
            self.gpt_model.train()
            iter_loss = self.evaluate_batch(inputs, targets)

            # Valid
            self.gpt_model.eval()
            inputs, targets = get_batch_fn("val", batch_size)
            with torch.no_grad():
                val_iter_loss = self.evaluate_batch(inputs, targets, train=False)

            self.losses.append(iter_loss)
            self.val_losses.append(val_iter_loss)

            pbar.set_description(
                "Iter :{}/{} Train Loss {:.3e} / Eval Loss {:.3e}".format(
                    i, max_iters, iter_loss, val_iter_loss
                )
            )

            if val_iter_loss < best_val_loss:
                counter = 0
                best_val_step = i
                best_val_loss = val_iter_loss
                self.save(name="best_model.pt")

            # Save last model
            if i % save_every == 0:
                self.save(name="last_model.pt")

            if counter > patience:
                print("Patience exhausted (i.e early stopping). Stopping training.")
                print(
                    "Best validation loss: {:.3e} at iteration {}".format(
                        best_val_loss, best_val_step
                    )
                )
                break

            counter += 1

    def evaluate_batch(self, inputs, targets, train=True):
        self.opt.zero_grad()
        logits = self.gpt_model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if train:
            loss.backward()
            self.opt.step()
        return loss.item()

    def save(self, name):
        dic = self.get_state_dict()
        torch.save(dic, self.checkpoint_path + name)

    def get_state_dict(self):
        dic = {
            "state": (
                self.gpt_model.module.state_dict()
                if self.parallel
                else self.gpt_model.state_dict()
            ),
            "opt": self.opt.state_dict(),
            "losses": self.losses,
            "val_losses": self.val_losses,
        }
        return dic

    def load(self, name):
        if torch.cuda.is_available():
            dic = torch.load(
                self.checkpoint_path + name, map_location=torch.device("cuda")
            )
        else:
            dic = torch.load(
                self.checkpoint_path + name, map_location=torch.device("cpu")
            )

        if self.parallel:
            self.gpt_model.module.load_state_dict(dic["state"])
        else:
            self.gpt_model.load_state_dict(dic["state"])

        self.opt.load_state_dict(dic["opt"])
        self.losses, self.val_losses = dic["losses"], dic["val_losses"]
        return dic
