# trainer.py
import os
import math
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

torch.set_float32_matmul_precision("high")


class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup to base LR for `warmup_steps` optimizer updates,
    then linear decay to 0 by `total_steps`. Call scheduler.step() *after* optimizer.step().
    """
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch: int = -1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch + 1  # count optimizer steps
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * (step / self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                lr = base_lr * (1.0 - progress)
            lrs.append(lr)
        return lrs


class Trainer:
    """
    DDP-aware Trainer.

    Expects that torch.distributed has already been initialized in the main script.
    Wraps the provided model with DDP and runs training via `train_model_simple`.
    """

    def __init__(
        self,
        cfg: dict,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        rank: int,
        base_lr: float = 3e-4,
        weight_decay: float = 0.01,
    ):
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.train_loader = train_loader
        self.val_loader = val_loader

        # base (per-rank) model
        self.base_model = model.to(device)

        # optional compile before wrapping in DDP
        self.base_model = torch.compile(self.base_model)

        # DDP wrapper
        self.model = DDP(
            self.base_model,
            device_ids=[device.index],
            output_device=device.index,
        )

        self.optimizer = AdamW(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)

    def _log(self, fp, msg: str):
        if self.rank == 0:
            print(msg)
            fp.write(msg + "\n")
            fp.flush()

    def train_model_simple(
        self,
        num_epochs: int = 100,
        grad_accum_steps: int = 1,
    ) -> Dict[str, list]:
        """
        DDP-aware training loop with:
          - gradient accumulation
          - linear warmup + linear decay LR schedule
          - metric aggregation across ranks
        """
        train_losses, val_losses = [], []
        train_accs,  val_accs  = [], []
        # train_times, val_times = [], []

        K, L, M = self.cfg.get("K", None), self.cfg.get("L", None), self.cfg.get("M", None)
        out_path = f"trial_K{K}_L{L}_M{M}_VIT_rank{self.rank}.txt"

        steps_per_epoch = len(self.train_loader)                  # micro-steps per rank
        updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
        total_updates  = num_epochs * updates_per_epoch
        warmup_updates = max(1, int(0.1 * total_updates))

        scheduler = LinearWarmupLR(
            self.optimizer,
            warmup_steps=warmup_updates,
            total_steps=total_updates,
        )

        fp = open(out_path, "a", encoding="utf-8") if self.rank == 0 else None
        if self.rank == 0:
            self._log(fp, f"Epochs: {num_epochs}")
            self._log(fp, "-" * 72)

        global_update = 0

        for epoch in range(1, num_epochs + 1):
            # Let DistributedSampler reshuffle per epoch
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            if isinstance(self.val_loader.sampler, DistributedSampler):
                self.val_loader.sampler.set_epoch(epoch)

            # # === TRAIN ===
            # if "cuda" in str(self.device):
            #     torch.cuda.synchronize()
            # t0 = time.time()

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            running_loss = 0.0
            running_correct = 0.0
            running_total = 0.0
            accum_count = 0

            for images, labels in tqdm(
                self.train_loader,
                desc=f"[Rank {self.rank}] Epoch {epoch} (train)",
                disable=(self.rank != 0),
            ):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)

                # scale for accumulation
                (loss / grad_accum_steps).backward()
                accum_count += 1

                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_total   += labels.size(0)
                running_loss    += loss.item()

                if accum_count == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_update += 1

            # flush remainder
            if accum_count > 0:
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_update += 1

            # aggregate train metrics across ranks
            local_train_stats = torch.tensor(
                [running_loss, running_correct, running_total],
                dtype=torch.float64,
                device=self.device,
            )
            if dist.is_initialized():
                dist.all_reduce(local_train_stats, op=dist.ReduceOp.SUM)
            total_loss, total_correct, total_count = local_train_stats.tolist()

            # if "cuda" in str(self.device):
            #     torch.cuda.synchronize()
            # train_time = time.time() - t0
            # train_times.append(train_time)

            tr_l = total_loss / max(1, steps_per_epoch * self.world_size)
            tr_a = total_correct / max(1, total_count)
            train_losses.append(tr_l)
            train_accs.append(tr_a)

            # === VALIDATION ===
            # if "cuda" in str(self.device):
            #     torch.cuda.synchronize()
            # t1 = time.time()

            self.model.eval()
            val_loss_total = 0.0
            val_correct = 0.0
            val_total = 0.0

            with torch.no_grad():
                for images, labels in tqdm(
                    self.val_loader,
                    desc=f"[Rank {self.rank}] Epoch {epoch} (val)",
                    disable=(self.rank != 0),
                ):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    outputs = self.model(images)
                    loss = F.cross_entropy(outputs, labels)
                    val_loss_total += loss.item()
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)

            local_val_stats = torch.tensor(
                [val_loss_total, val_correct, val_total],
                dtype=torch.float64,
                device=self.device,
            )
            if dist.is_initialized():
                dist.all_reduce(local_val_stats, op=dist.ReduceOp.SUM)
            total_vloss, total_vcorrect, total_vcount = local_val_stats.tolist()

            # if "cuda" in str(self.device):
            #     torch.cuda.synchronize()
            # val_time = time.time() - t1
            # val_times.append(val_time)

            va_l = total_vloss / max(1, len(self.val_loader) * self.world_size)
            va_a = total_vcorrect / max(1, total_vcount)
            val_losses.append(va_l)
            val_accs.append(va_a)

            curr_lr = scheduler.get_last_lr()[0]

            if self.rank == 0:
                self._log(
                    fp,
                    (f"Ep{epoch:3d} | "
                     f"train_loss {tr_l:.4f}, acc {tr_a:.4f} | "
                     f"val_loss {va_l:.4f}, acc {va_a:.4f} | "
                     f"lr {curr_lr:.3e} | updates {global_update}/{total_updates}")
                )

        if self.rank == 0:
            self._log(fp, "-" * 72)
            self._log(fp, f"Log saved to: {os.path.abspath(out_path)}")
            fp.close()

        return {
            "train_loss": train_losses, "val_loss": val_losses,
            "train_acc":  train_accs,   "val_acc":  val_accs,
            # "train_time": train_times,  "val_time": val_times,
        }
