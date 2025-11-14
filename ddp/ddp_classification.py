import os
import sys
import torch
from torch.utils.data import random_split
from torch.distributed import init_process_group, destroy_process_group
import psutil
# from model_ import Classifier, GPTConfig, OptimizerConfig, create_optimizer
from model import Classifier
# from trainer import Trainer, TrainerConfig
from trainer import Trainer
from dataset import YahooDataset
# from omegaconf import DictConfig
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

# torchrun --standalone --nproc_per_node=4 ddp_classification.py

def verify_min_gpu_count(min_gpus: int = 4) -> bool:
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

def ddp_setup():
    acc = torch.accelerator.current_accelerator()
    rank = int(os.environ["LOCAL_RANK"])

    physical_cpus = psutil.cpu_count(logical=False)
    os.environ["OMP_NUM_THREADS"] = str(int(physical_cpus/4))

    device: torch.device = torch.device(f"{acc}:{rank}")
    backend = torch.distributed.get_default_backend_for_device(device)
    init_process_group(backend=backend)
    torch.accelerator.set_device_index(rank)
    return rank

def get_train_objs(cfg, dataset):

    train_dataset = YahooDataset(dataset["train"]["text"], dataset["train"]["label"])
    test_dataset = YahooDataset(dataset["test"]["text"], dataset["test"]["label"])
    model = Classifier(cfg)
    
    return model, train_dataset, test_dataset

if __name__ == "__main__":
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        sys.exit()

    local_rank = ddp_setup()


    # gpt_cfg = GPTConfig(**cfg['gpt_config'])
    # opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
    # data_cfg = DataConfig(**cfg['data_config'])
    # trainer_cfg = TrainerConfig(**cfg['trainer_config'])

    df_no_ref = load_dataset("ccdv/arxiv-classification", "no_ref")
    # Combine validation and test into a single split
    test_val_ds = concatenate_datasets([df_no_ref["validation"], df_no_ref["test"]])

    # Optionally create a new DatasetDict with just train and combined test+val
    dataset = DatasetDict({
        "train": df_no_ref["train"],
        "test": test_val_ds,
    })

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ret_val = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    VOCAB_LIMIT = tokenizer.vocab_size

    device = torch.device(f"cuda:{local_rank}")
    N_CLASSES = 11
    num_workers = 4
    batch_size = 2
    # FILL IN
    MAX_LEN = 32000
    
    
    # GPT trial
    kind = "softmax"
    cfg = {
        "batch_size":batch_size,
        "num_workers":num_workers,
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

    model, train_data, test_data = get_train_objs(cfg, dataset)

    trainer = Trainer(cfg, model, tokenizer, train_data, test_data)

    trainer.run_experiment()

    destroy_process_group()
