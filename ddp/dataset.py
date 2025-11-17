import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch 
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as dataloader
import torch.nn as nn
import itertools
import math
import time
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
import os, glob, csv, random
import numpy as np
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import InterpolationMode
torch.set_float32_matmul_precision('high')
from collections import defaultdict
from torchvision.datasets import Food101


def _get_labels(ds):
    # Be robust across torchvision versions
    if hasattr(ds, "targets"): return ds.targets
    if hasattr(ds, "labels"): return ds.labels
    if hasattr(ds, "_labels"): return ds._labels
    if hasattr(ds, "samples"): return [y for _, y in ds.samples]
    raise AttributeError("Cannot find labels in dataset object.")

def _balanced_subset_fixed_total(ds, class_ids, total, seed=0):
    """
    Pick a near-balanced subset of size `total` drawn only from `class_ids`.
    Returns a list of indices into `ds`.
    """
    rng = random.Random(seed)
    labels = _get_labels(ds)

    # bucket indices per chosen class
    buckets = {c: [] for c in class_ids}
    for idx, y in enumerate(labels):
        y = int(y)
        if y in buckets:
            buckets[y].append(idx)

    num_classes = len(class_ids)
    base = total // num_classes       # floor per class
    extra = total % num_classes       # first `extra` classes get +1

    class_ids_sorted = sorted(class_ids)
    keep = []
    for i, c in enumerate(class_ids_sorted):
        idxs = buckets[c]
        rng.shuffle(idxs)
        n_this = base + (1 if i < extra else 0)
        if n_this > len(idxs):
            raise ValueError(f"Not enough examples in class {c} to sample {n_this}")
        keep.extend(idxs[:n_this])

    rng.shuffle(keep)
    return keep

def get_data_food101(
    batch_size,
    img_size=512,
    num_workers=8,
    seed=0,
    root="./data",
    distributed = True
):
    # Transforms (RGB @ img_size x img_size)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256/224), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
    ])

    # Load full Food-101 (official splits)
    try:
        ds_train_full = Food101(root=root, split="train", download=True, transform=train_tf)
        ds_test_full  = Food101(root=root, split="test",  download=True, transform=val_tf)
    except TypeError:
        ds_train_full = Food101(root=root, train=True,  download=True, transform=train_tf)
        ds_test_full  = Food101(root=root, train=False, download=True, transform=val_tf)

    # --- choose 50 classes (consistent between train and test) ---
    all_train_labels = sorted(set(int(y) for y in _get_labels(ds_train_full)))
    if len(all_train_labels) < 50:
        raise ValueError("Food101 train split has fewer than 50 classes?")
    class_ids_50 = all_train_labels[:50]   # or random.sample(all_train_labels, 50, seed)

    # --- sample 7,520 train examples across these 50 classes ---
    train_idx = _balanced_subset_fixed_total(
        ds_train_full, class_ids=class_ids_50, total=1000, seed=seed
    )

    # --- sample 2,500 test examples across the same 50 classes ---
    test_idx = _balanced_subset_fixed_total(
        ds_test_full, class_ids=class_ids_50, total=100, seed=seed
    )

    ds_train = Subset(ds_train_full, train_idx)
    ds_test  = Subset(ds_test_full,  test_idx)

    # DDP samplers
    if distributed and dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(ds_train, shuffle=True)
        val_sampler   = DistributedSampler(ds_test, shuffle=False)
        shuffle_train = False
        shuffle_val   = False
    else:
        train_sampler = None
        val_sampler   = None
        shuffle_train = True
        shuffle_val   = False

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=True,
    )

    info = {
        "num_train": len(ds_train),   # should be 5550
        "num_test": len(ds_test),     # should be 1500
        "num_classes": len(class_ids_50),  # 50
    }
    return train_loader, test_loader, info
