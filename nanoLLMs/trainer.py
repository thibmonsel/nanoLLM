import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import destroy_process_group, get_rank, init_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class Trainer:
    def __init__(
        self, llm_model, lr, checkpoint_path="", wd=0.0, ddp=False, use_zero=False
    ):
        self.ddp = ddp
        self.use_zero = use_zero
        if self.ddp:
            self.setup_ddp(llm_model)
        else:
            self.llm_model = llm_model

        if use_zero:
            self.opt = ZeroRedundancyOptimizer(
                self.llm_model.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=lr,
                weight_decay=wd,
            )
        else:
            self.opt = torch.optim.Adam(
                self.llm_model.parameters(), lr=lr, weight_decay=wd
            )
        self.checkpoint_path = checkpoint_path

        self.losses = []
        self.val_losses = []

        if checkpoint_path.endswith("/"):
            if self.ddp:
                if self.local_rank == 0:
                    os.makedirs(checkpoint_path, exist_ok=True)
                    print(
                        f"Created checkpoint directory at {checkpoint_path} "
                        f"on rank {self.local_rank}"
                    )

            else:
                os.makedirs(checkpoint_path, exist_ok=True)
                print(f"Created checkpoint directory at {checkpoint_path}")

    @staticmethod
    def cleanup():
        destroy_process_group()

    def setup_ddp(self, llm_model):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group("nccl")
        rank = get_rank()
        print(f"Start running basic DDP example on rank {rank}.")
        device_id = rank % torch.cuda.device_count()
        llm_model = llm_model.to(device_id)
        self.llm_model = DDP(llm_model, device_ids=[device_id])
        self.local_rank = device_id

    def train(
        self, get_batch_fn, batch_size, max_iters=1000, save_every=1, accum_iter=None
    ):
        best_val_loss = float(np.inf)
        pbar = tqdm(range(max_iters))
        for i in pbar:
            inputs, targets = get_batch_fn("train", batch_size)
            if self.ddp:
                assert inputs.size(0) % torch.cuda.device_count() == 0

                split_inputs = torch.chunk(inputs, torch.cuda.device_count(), dim=0)
                split_targets = torch.chunk(targets, torch.cuda.device_count(), dim=0)

                targets = split_targets[self.local_rank]
                inputs = split_inputs[self.local_rank]

            # Train
            self.llm_model.train()
            loss = self.evaluate_batch(inputs, targets)

            if self.ddp and accum_iter is not None:
                loss = loss / accum_iter
                if ((i + 1) % accum_iter == 0) or (i + 1 == max_iters):
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()
            else:
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

            self.losses.append(loss.item())

            # Valid
            self.llm_model.eval()
            inputs, targets = get_batch_fn("val", batch_size)
            with torch.no_grad():
                val_iter_loss = self.evaluate_batch(inputs, targets).item()

            self.val_losses.append(val_iter_loss)

            pbar.set_description(
                "Iter :{}/{} Train Loss {:.3e} / Eval Loss {:.3e}".format(
                    i, max_iters, loss.item(), val_iter_loss
                )
            )

            model = self.llm_model.module if self.ddp else self.llm_model
            model_name = model.__class__.__name__
            if val_iter_loss < best_val_loss and i > int(max_iters / 10):
                best_val_loss = val_iter_loss
                self.save(name=model_name + "_best_model.pt")

            # Save last model
            if i % save_every == 0 and i > 0:
                self.save(name=model_name + "_last_model.pt")

        if self.ddp:
            self.cleanup()

    def evaluate_batch(self, inputs, targets):
        with torch.autocast(device_type="cuda"):
            logits = self.llm_model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return loss

    def save(self, name):
        if self.ddp:
            if self.use_zero:
                self.opt.consolidate_state_dict(to=0)
                print("Done consolidating ZeRO optimizer")
            if self.local_rank == 0:
                print("Saving model...")
                dic = {
                    "losses": self.losses,
                    "val_losses": self.val_losses,
                    "state": self.llm_model.module.state_dict(),
                    "opt": self.opt.state_dict(),
                }
                dist.barrier()

                torch.save(dic, self.checkpoint_path + name)
                print("DONE Saving model...")
        else:
            print("Saving model...")
            dic = {
                "losses": self.losses,
                "val_losses": self.val_losses,
                "state": self.llm_model.state_dict(),
                "opt": self.opt.state_dict(),
            }
            torch.save(dic, self.checkpoint_path + name)
            print("DONE Saving model...")

    def load(self, name):
        if torch.cuda.is_available():
            dic = torch.load(
                self.checkpoint_path + name, map_location=torch.device("cuda")
            )
        else:
            dic = torch.load(
                self.checkpoint_path + name, map_location=torch.device("cpu")
            )

        if self.ddp:
            self.llm_model.module.load_state_dict(dic["state"])
        else:
            self.llm_model.load_state_dict(dic["state"])

        self.opt.load_state_dict(dic["opt"])
        self.losses, self.val_losses = dic["losses"], dic["val_losses"]
        return dic
