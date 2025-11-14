import time
from collections import defaultdict
from torch.distributed import init_process_group, destroy_process_group

import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
import numpy
import torch
import time
import math
from datasets import DatasetDict, concatenate_datasets
from tqdm.auto import tqdm



import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
# import tiktoken
import itertools
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler import schedule
import argparse

class Trainer:
    # @staticmethod
    # def get_default_config():
    #     C = CN()
    #     # device to train on
    #     C.device = 'auto'
    #     # dataloder parameters
    #     C.num_workers = 4
    #     # optimizer parameters
    #     C.max_iters = None
    #     C.batch_size = 64
    #     C.learning_rate = 3e-4
    #     C.betas = (0.9, 0.95)
    #     C.weight_decay = 0.1 # only applied on matmul weights
    #     C.grad_norm_clip = 1.0
    #     return C

    def collate_fn(self, batch):
        texts, labels = zip(*batch)
        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            truncation=True,
            padding='max_length',
            return_attention_mask = True,
            return_tensors='pt'
        )
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

    """
    config contains information necessary for training

    example:
        cfg = {
            "num_workers":num_workers
            "kind":kind, 
            "output_dim":N_CLASSES,
            "device":device,
            "vocab_size": VOCAB_LIMIT+1,
            "context_length": MAX_LEN,
            "emb_dim": 512,
            "n_heads": 8,
            "n_layers": 8,
            "drop_rate": 0.1,
            "qkv_bias": False,  
            "proj_dim":128    # <— add this
        }
    """
    def __init__(self, config, model, tokenizer, train_dataset, test_dataset):
        self.config = config
        self.device = config["device"]
        self.MAX_LEN = config["context_length"]
        self.rank = torch.distributed.get_rank()
        self.model = model.to(self.device)
        self.ddp_model = DDP(
            self.model,
            device_ids=[self.device.index],
            output_device=self.device.index,
        )
        self.tokenizer = tokenizer
        # self.optimizer = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def run_experiment(self, epochs=100, lr=2e-5, wd=5e-05, resume = False):

        # setup the dataloaders
        train_loader = DataLoader(
            dataset = self.train_dataset,
            sampler=DistributedSampler(self.train_dataset),
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            drop_last = True
        )

        test_loader = DataLoader(
            dataset = self.test_dataset,
            sampler=DistributedSampler(self.test_dataset),
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            drop_last = True
        )

        # train_loader = DataLoader(train_dataset, batch_size=BATCH,  num_workers = 4, pin_memory = True, collate_fn=self.collate_fn ,shuffle=True, drop_last=True)


        # for kind in attn_types:
        kind = self.config["kind"]

        # checkpoint_path = f"checkpoint_{attn_types[0]}_classification_256_yahoo_2_2_2"
        
        print(f"\n=== Training {kind.upper()} for {epochs} epochs ===")
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        # train_times, val_times = [], []

        start_epoch = 1

        # if resume:
        #   ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        #   model.load_state_dict(ckpt["model_state_dict"])
        #   opt.load_state_dict  (ckpt["optimizer_state_dict"])
        #   start_epoch = ckpt["epoch"] + 1
        #   train_losses, val_losses = ckpt["train_losses"], ckpt["val_losses"]
        #   train_accs, val_accs = ckpt["train_accs"], ckpt["val_accs"]
        # #   train_times, val_times = ckpt["train_times"], ckpt["val_times"]
        #   print(f"⟳ Resuming {kind} at epoch {start_epoch}")

        # print("compiling: ")
        opt = torch.optim.AdamW(
            self.ddp_model.parameters(), lr=lr, weight_decay=wd
        )
        self.ddp_model = torch.compile(self.ddp_model, mode="default")        # opt_mod = model
        # print("done compiling")

        for ep in range(start_epoch, epochs+1):
            train_loader.sampler.set_epoch(ep)
            test_loader.sampler.set_epoch(ep)
            # --- train timing ---
            # t0= time.time()
            self.ddp_model.train()
            tl=torch.zeros(1, device = self.config["device"])
            # tl = torch.zeros(1)
            ta = torch.zeros(1, device = self.config["device"])
            # ta = torch.zeros(1)
            # print(f"epoch {ep}")
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)

            # start.record()
            for x, mask, y in tqdm(
                        train_loader,
                        desc=f"Train epoch {ep}",
                        leave=False,
                    ):            
                # print(f"in loader: ")
                x    = x.to(self.config["device"], non_blocking=True)
                mask = mask.to(self.config["device"], non_blocking=True)
                y    = y.to(self.config["device"], non_blocking=True)
                # opt.zero_grad()
                for param in self.ddp_model.parameters():
                    param.grad = None
                logits = self.ddp_model(x,mask)
                # print("finished forward pass")
                loss   = F.cross_entropy(logits,y)
                acc    = (logits.argmax(-1)==y).float().mean()
                # print("starting backward pass")
                loss.backward(); opt.step()
                # print("finished backward pass")
                tl += loss; ta += acc

            # end.record()
            # torch.cuda.current_stream().synchronize()
            # gpu_train_time = start.elapsed_time(end)/1000

            tr_l, tr_a = tl.item()/len(train_loader), ta.item()/len(train_loader)
            # train_time = time.time() - t0
            # train_times.append(gpu_train_time)
            # train_times.append(train_time)

            # --- eval timing ---
            self.ddp_model.eval()
            # t1 = time.time()
            vl=torch.zeros(1, device = self.config["device"])
            # vl = torch.zeros(1)
            va = torch.zeros(1, device = self.config["device"])
            # va = torch.zeros(1)

            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)

            # start.record()
            with torch.no_grad():
                for x, mask, y in tqdm(
                                    test_loader,
                                    desc=f"Train epoch {ep}",
                                    leave=False,
                                ):                     
                    x    = x.to(self.config["device"], non_blocking=True)
                    mask = mask.to(self.config["device"], non_blocking=True)
                    y    = y.to(self.config["device"], non_blocking=True)
                    logits = self.ddp_model(x,mask)
                    vl += F.cross_entropy(logits,y)
                    va += (logits.argmax(-1)==y).float().mean()

            # end.record()
            # torch.cuda.current_stream().synchronize()
            # gpu_val_time = start.elapsed_time(end)/1000

            va_l, va_a = vl.item()/len(test_loader), va.item()/len(test_loader)
            # val_time = time.time() - t1
            # val_times.append(gpu_val_time)
            # val_times.append(val_time)

            train_losses.append(tr_l); train_accs.append(tr_a)
            val_losses.append(va_l); val_accs.append(va_a)

            print(
                f"Ep{ep:2d} | "
                f"train_loss {tr_l:.3f}, acc {tr_a:.3f} "
                # f"({gpu_train_time:.1f}s) | "
                f"val_loss   {va_l:.3f}, acc {va_a:.3f} "
                # f"({gpu_val_time:.1f}s)"
            )

            # torch.save({
            #     "epoch": ep,
            #     "model_state_dict": model.state_dict(),
            #     "optimizer_state_dict": opt.state_dict(),
            #     "train_losses" :  train_losses,
            #     "val_losses":  val_losses,
            #     "train_accs":train_accs,
            #     "val_accs": val_accs,
            #     # "train_times":train_times,
            #     # "val_times":val_times
            # }, checkpoint_path)
            # print(f"✓ checkpoint saved to {checkpoint_path}")


        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            # "train_time": train_times,
            # "val_time": val_times,
            "train_acc": train_accs,
            "val_acc": val_accs,
        }