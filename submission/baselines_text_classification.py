#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Classification Baselines
-----------------------------------------------------------
One-file training script with multiple attention baselines and dataset helpers.
- Baselines: softmax (vanilla Transformer), angular, linear (kernelized),
             linformer (projection), RACE (ACE-based), performer (FAVOR+).
- Datasets: QNLI, QQP, SST-2, IMDB via HuggingFace `datasets`.
- Tokenizer: GPT-2 tokenizer with an added PAD token (CLS-free; uses masked mean pooling).
- Features: AMP, gradient clipping, deterministic seeding, CLI args, CSV metric dump.

Usage (examples):
  python baselines_text_classification.py --dataset qnli --attn softmax --epochs 5 --device cuda:1
  python baselines_text_classification.py --dataset qqp  --attn performer --epochs 5 --device cuda:1
  python baselines_text_classification.py --dataset sst2 --attn linformer --epochs 5 --device cuda:1
  python baselines_text_classification.py --dataset imdb --attn linear --epochs 5 --device cuda:1

Metrics CSV will be saved next to the script unless --out_dir is provided.

Requirements (pip):
  - torch, torchvision
  - transformers
  - datasets
  - numpy, matplotlib (optional for plotting)
"""

from __future__ import annotations

import os
import math
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# HuggingFace
from datasets import load_dataset
from transformers import AutoTokenizer

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# Tokenizer (GPT-2 w/ PAD)
# ------------------------------------------------------------
def build_tokenizer(model_name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # ensure right padding exists for batch padding
    if tok.pad_token is None:
        # use a unique token string to guarantee an added vocab entry
        tok.add_special_tokens({'pad_token': '<|pad|>'})
    tok.padding_side = 'right'
    return tok

# ------------------------------------------------------------
# Simple text dataset wrapper
# ------------------------------------------------------------
class TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels), "Texts/labels length mismatch"
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], int(self.labels[i])

# ------------------------------------------------------------
# Collate (tokenize on the fly)
# ------------------------------------------------------------
def make_collate_fn(tokenizer, max_len: int):
    def _collate(batch: List[Tuple[str, int]]):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        y = torch.tensor(labels, dtype=torch.long)
        return enc["input_ids"], enc["attention_mask"], y
    return _collate

# ------------------------------------------------------------
# Dataset helpers (returns train/val loaders and num_classes)
# ------------------------------------------------------------
@dataclass
class LoaderPack:
    train: DataLoader
    val: DataLoader
    num_classes: int
    tokenizer_vocab_size: int

def _split_fallback(ds, prefer_val=True):
    # Maps datasets with different split conventions into (train, val)
    if "train" in ds and "validation" in ds:
        return ds["train"], ds["validation"]
    if "train" in ds and "test" in ds:
        # Many SetFit datasets ship 'test' only; use as val
        return ds["train"], ds["test"]
    if "train" in ds and "dev" in ds:
        return ds["train"], ds["dev"]
    raise ValueError(f"Dataset splits not recognized: {list(ds.keys())}")

def _make_loaders_from_texts(
    train_texts, train_labels, val_texts, val_labels,
    tokenizer, max_len: int, batch: int, workers: int
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TextLabelDataset(train_texts, train_labels)
    val_ds   = TextLabelDataset(val_texts,   val_labels)

    collate = make_collate_fn(tokenizer, max_len)
    # Use conservative worker settings for reproducibility across systems
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, drop_last=False,
        num_workers=workers, pin_memory=True, persistent_workers=(workers > 0),
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, drop_last=False,
        num_workers=workers, pin_memory=True, persistent_workers=(workers > 0),
        collate_fn=collate
    )
    return train_loader, val_loader

def get_qnli(max_len: int, batch: int, workers: int, tokenizer) -> LoaderPack:
    ds = load_dataset("SetFit/qnli")
    tr, va = _split_fallback(ds)
    def combine(row): return {"text": f"{row['text1']} {row['text2']}"}
    tr = tr.map(combine)
    va = va.map(combine)
    train_texts, train_labels = tr["text"], tr["label"]
    val_texts,   val_labels   = va["text"], va["label"]
    train_loader, val_loader = _make_loaders_from_texts(
        train_texts, train_labels, val_texts, val_labels, tokenizer, max_len, batch, workers
    )
    return LoaderPack(train_loader, val_loader, num_classes=2, tokenizer_vocab_size=tokenizer.vocab_size)

def get_qqp(max_len: int, batch: int, workers: int, tokenizer) -> LoaderPack:
    ds = load_dataset("SetFit/qqp")
    tr, va = _split_fallback(ds)
    def combine(row): return {"text": f"{row['text1']} {row['text2']}"}
    tr = tr.map(combine)
    va = va.map(combine)
    train_texts, train_labels = tr["text"], tr["label"]
    val_texts,   val_labels   = va["text"], va["label"]
    train_loader, val_loader = _make_loaders_from_texts(
        train_texts, train_labels, val_texts, val_labels, tokenizer, max_len, batch, workers
    )
    return LoaderPack(train_loader, val_loader, num_classes=2, tokenizer_vocab_size=tokenizer.vocab_size)

def get_sst2(max_len: int, batch: int, workers: int, tokenizer) -> LoaderPack:
    ds = load_dataset("glue", "sst2")  # splits: train, validation, test
    tr, va = ds["train"], ds["validation"]
    train_texts, train_labels = tr["sentence"], tr["label"]
    val_texts,   val_labels   = va["sentence"], va["label"]
    train_loader, val_loader = _make_loaders_from_texts(
        train_texts, train_labels, val_texts, val_labels, tokenizer, max_len, batch, workers
    )
    return LoaderPack(train_loader, val_loader, num_classes=2, tokenizer_vocab_size=tokenizer.vocab_size)

def get_imdb(max_len: int, batch: int, workers: int, tokenizer) -> LoaderPack:
    ds = load_dataset("imdb")
    tr, va = ds["train"], ds["test"]
    train_texts, train_labels = tr["text"], tr["label"]
    val_texts,   val_labels   = va["text"], va["label"]
    train_loader, val_loader = _make_loaders_from_texts(
        train_texts, train_labels, val_texts, val_labels, tokenizer, max_len, batch, workers
    )
    return LoaderPack(train_loader, val_loader, num_classes=2, tokenizer_vocab_size=tokenizer.vocab_size)

DATASETS: Dict[str, Callable[..., LoaderPack]] = {
    "qnli": get_qnli,
    "qqp": get_qqp,
    "sst2": get_sst2,
    "imdb": get_imdb,
}

# ------------------------------------------------------------
# Attention primitives
# ------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Standard softmax attention (non-causal)."""
    def __init__(self, d: int, h: int, drop: float, qkv_bias: bool = False):
        super().__init__()
        assert d % h == 0, "Embedding dim must be divisible by n_heads"
        self.h = h
        self.dk = d // h
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, D = x.shape
        H, dk = self.h, self.dk
        Q = self.q(x).view(B, T, H, dk).transpose(1, 2)  # B,H,T,dk
        K = self.k(x).view(B, T, H, dk).transpose(1, 2)  # B,H,T,dk
        V = self.v(x).view(B, T, H, dk).transpose(1, 2)  # B,H,T,dk

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q, K, V = Q * keep, K * keep, V * keep

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dk)  # B,H,T,T
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, H * dk)
        return self.o(out)

class AngularAttention(nn.Module):
    """Cosine/angle-based attention variant (non-causal)."""
    def __init__(self, d: int, h: int, drop: float, qkv_bias: bool = False, pow_t: float = 8.0):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.pow_t = h, d // h, pow_t
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, D = x.shape
        H, dk = self.h, self.dk
        Q = self.q(x).view(B, T, H, dk).transpose(1, 2)  # B,H,T,dk
        K = self.k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.v(x).view(B, T, H, dk).transpose(1, 2)

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q, K, V = Q * keep, K * keep, V * keep

        Qn = F.normalize(Q, dim=-1)
        Kn = F.normalize(K, dim=-1)
        sim = (Qn @ Kn.transpose(-2, -1)).clamp(-0.999, 0.999)  # B,H,T,T
        scores = 1.0 - torch.acos(sim) / math.pi               # angle -> [0,1]
        W = scores.clamp(min=1e-6).pow(self.pow_t)
        W = W / (W.sum(-1, keepdim=True) + 1e-6)
        W = self.drop(W)
        out = (W @ V).transpose(1, 2).contiguous().view(B, T, H * dk)
        return self.o(out)

class LinearAttention(nn.Module):
    """Kernelized linear attention (ELU+1 feature map)."""
    def __init__(self, d: int, h: int, drop: float, qkv_bias: bool = False, eps: float = 1e-6):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.eps = h, d // h, eps
        self.Wq = nn.Linear(d, d, bias=qkv_bias)
        self.Wk = nn.Linear(d, d, bias=qkv_bias)
        self.Wv = nn.Linear(d, d, bias=qkv_bias)
        self.o  = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, D = x.shape
        H, dk = self.h, self.dk
        Q = self.Wq(x).view(B, T, H, dk).transpose(1, 2)  # B,H,T,dk
        K = self.Wk(x).view(B, T, H, dk).transpose(1, 2)
        V = self.Wv(x).view(B, T, H, dk).transpose(1, 2)

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q, K, V = Q * keep, K * keep, V * keep

        phi_Q = self._phi(Q)            # B,H,T,dk
        phi_K = self._phi(K)            # B,H,T,dk
        KV     = torch.einsum('bhtd,bhte->bhde', phi_K, V)  # sum over T
        K_sum  = phi_K.sum(dim=2)                          # B,H,dk
        denom  = torch.einsum('bhtd,bhd->bht', phi_Q, K_sum).unsqueeze(-1) + self.eps
        out_h  = torch.einsum('bhtd,bhde,bht1->bhte', phi_Q, KV, denom.reciprocal())
        out    = out_h.transpose(1, 2).contiguous().view(B, T, H * dk)
        out    = self.drop(out)
        return self.o(out)

class LinformerAttention(nn.Module):
    """Linformer-style projection of K,V along time axis."""
    def __init__(self, d: int, h: int, proj_dim: int, max_seq_len: int, qkv_bias: bool, drop: float):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.proj_dim = proj_dim
        self.max_seq_len = max_seq_len

        self.Wq = nn.Linear(d, d, bias=qkv_bias)
        self.Wk = nn.Linear(d, d, bias=qkv_bias)
        self.Wv = nn.Linear(d, d, bias=qkv_bias)
        self.o  = nn.Linear(d, d)

        # Learned projections T -> K along time
        self.Pk = nn.Parameter(torch.empty(max_seq_len, proj_dim))
        self.Pv = nn.Parameter(torch.empty(max_seq_len, proj_dim))
        nn.init.xavier_uniform_(self.Pk)
        nn.init.xavier_uniform_(self.Pv)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, D = x.shape
        H, dk = self.h, self.dk
        Q = self.Wq(x).view(B, T, H, dk).transpose(1, 2)  # B,H,T,dk
        K = self.Wk(x).view(B, T, H, dk).transpose(1, 2)
        V = self.Wv(x).view(B, T, H, dk).transpose(1, 2)

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q, K, V = Q * keep, K * keep, V * keep

        Pk = self.Pk[:T, :]  # T,K
        Pv = self.Pv[:T, :]

        lowK = torch.einsum("bhtd,tk->bhdk", K, Pk)  # B,H,dk,K
        lowV = torch.einsum("bhtd,tk->bhkd", V, Pv)  # B,H,K,dk
        scores = (Q @ lowK) / math.sqrt(dk)          # B,H,T,K
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        ctx = attn @ lowV                             # B,H,T,dk
        out = ctx.transpose(1, 2).contiguous().view(B, T, H * dk)
        return self.o(out)

# ------------------ RACE (ACE) --------------------
class BatchedACE(nn.Module):
    """
    ACE hashing in batches. This is a direct adaptation for classification experiments.
    """
    def __init__(self, d_k: int, K: int, L: int, M: int, device: str, share_planes: bool = False):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer('planes_T', planes.view(L * K, d_k).T)
        else:
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)
            self.register_buffer('planes_T', planes)

        corners = torch.tensor(list(__import__('itertools').product([-1., +1.], repeat=K)), device=device)
        self.register_buffer('protos_T', corners.T)

    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
        M, B, T, H, dk = Khf.shape
        S = self.L * self.R
        scale = math.sqrt(dk)
        BH = B * H

        V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

        if self.share_planes:
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M * BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M * BH, T, dk)
            projK = Kh2 @ self.planes_T
            projQ = Qh2 @ self.planes_T
        else:
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            projK = torch.einsum('mbtd,mds->mbts', Kh2, self.planes_T)
            projQ = torch.einsum('mbtd,mds->mbts', Qh2, self.planes_T)

        V2    = V2.view(M * BH, T, dk)

        projK = projK.contiguous().view(-1, T, self.L, self.K)
        projQ = projQ.contiguous().view(-1, T, self.L, self.K)
        logitsK = (projK.tanh().div(scale) @ self.protos_T)
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)
        probsK  = F.softmax(logitsK, dim=-1)
        probsQ  = F.softmax(logitsQ, dim=-1)

        probsK_S = probsK.contiguous().view(-1, T, self.L * self.R)
        probsQ_S = probsQ.contiguous().view(-1, T, self.L * self.R)

        b_sum = probsK_S.transpose(1, 2).bmm(V2)
        A = probsK_S.sum(dim=1)
        E = b_sum / (A.unsqueeze(-1) + eps)
        out2 = probsQ_S.bmm(E)
        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4)
        return out

class RACEAttention(nn.Module):
    def __init__(self, d: int, h: int, drop: float, M=1, K=2, L=2, qkv_bias=False, device="cpu"):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.M = h, d // h, M
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(self.dk, K, L, M, device, M == 1)

    def forward(self, x, mask):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk)
        K = self.k(x).view(B, T, self.h, self.dk)
        V = self.v(x).view(B, T, self.h, self.dk)

        if mask is not None:
            m = mask.unsqueeze(-1).unsqueeze(-1)
            Q, K, V = Q*m, K*m, V*m

        def pack(z): return z.unsqueeze(0).expand(self.M, -1, -1, -1, -1)
        out_m = self.ace(pack(K), pack(V), pack(Q))
        out = out_m.mean(dim=0).transpose(1, 2).reshape(B, T, -1)
        return self.drop(self.o(out))

# ------------------ Performer (FAVOR+) --------------
def favorplus_features(x, proj, eps=1e-6):
    # x: [B,H,T,D], proj: [H,M,D] (per-head Gaussian)
    xw = torch.einsum('bhtd,hmd->bhtm', x, proj)
    xw = xw - xw.max(dim=-1, keepdim=True).values
    exp_part = torch.exp(xw)                        # [B,H,T,M]
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [B,H,T,1]
    base = torch.exp(-0.5 * x_norm_sq)              # [B,H,T,1]
    return exp_part * base + eps

class FavorPlusAttention(nn.Module):
    def __init__(self, d: int, h: int, m_features: int = 256, drop: float = 0.0, qkv_bias: bool = False, seed: Optional[int] = None):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.m = h, d // h, m_features
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        if seed is not None:
            torch.manual_seed(seed)
        proj = torch.nn.init.orthogonal_(torch.randn(h, m_features, self.dk))
        self.register_buffer("proj", proj)
        self.eps = 1e-6

    def forward(self, x, mask=None):
        B, T, d = x.shape
        h, dk, m = self.h, self.dk, self.m
        Q = self.q(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        Qs = Q / math.sqrt(dk)
        Ks = K / math.sqrt(dk)
        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Ks = Ks * keep
            V  = V  * keep
        phiQ = favorplus_features(Qs, self.proj, eps=self.eps) / math.sqrt(m)
        phiK = favorplus_features(Ks, self.proj, eps=self.eps) / math.sqrt(m)
        if mask is not None:
            keep_m = mask[:, None, :, None].to(phiK.dtype)
            phiK = phiK * keep_m
        KV   = torch.einsum("bhtm,bhtd->bhmd", phiK, V)
        Ksum = phiK.sum(dim=2)  # B,H,M
        num  = torch.einsum("bhtm,bhmd->bhtd", phiQ, KV)
        den  = torch.einsum("bhtm,bhm->bht",  phiQ, Ksum).unsqueeze(-1) + self.eps
        outH = num / den
        out  = outH.transpose(1, 2).contiguous().view(B, T, h * dk)
        out  = self.drop(out)
        return self.o(out)

# ------------------------------------------------------------
# Blocks
# ------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Vanilla Transformer encoder block (softmax attention)."""
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        self.att = MultiHeadAttention(d, h, drop, qkv_bias=qkv_bias)
        self.ff  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1  = nn.LayerNorm(d)
        self.n2  = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask):
        h = x
        x = self.n1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h
        h = x
        x = self.n2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class AngularBlock(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        self.att = AngularAttention(d, h, drop, qkv_bias=qkv_bias)
        self.ff  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1  = nn.LayerNorm(d)
        self.n2  = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask):
        h = x
        x = self.n1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h
        h = x
        x = self.n2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class LinearBlock(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        self.att = LinearAttention(d, h, drop, qkv_bias=qkv_bias)
        self.ff  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1  = nn.LayerNorm(d)
        self.n2  = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask):
        h = x
        x = self.n1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h
        h = x
        x = self.n2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class LinformerBlock(nn.Module):
    def __init__(self, d, h, drop, proj_dim, max_len, qkv_bias=False):
        super().__init__()
        self.att = LinformerAttention(d, h, proj_dim, max_len, qkv_bias, drop)
        self.ff  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1  = nn.LayerNorm(d)
        self.n2  = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask):
        h = x
        x = self.n1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h
        h = x
        x = self.n2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class RACEBlock(nn.Module):
    def __init__(self, d, h, drop, device, qkv_bias=False, M=1, K=2, L=2):
        super().__init__()
        self.att = RACEAttention(d, h, drop, M=M, K=K, L=L, qkv_bias=qkv_bias, device=device)
        self.ff  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1  = nn.LayerNorm(d)
        self.n2  = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask):
        h = x
        x = self.n1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h
        h = x
        x = self.n2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class PerformerBlock(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False, m_features=256, seed=None):
        super().__init__()
        self.att = FavorPlusAttention(d, h, m_features=m_features, drop=drop, qkv_bias=qkv_bias, seed=seed)
        self.ff  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1  = nn.LayerNorm(d)
        self.n2  = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask):
        h = x
        x = self.n1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h
        h = x
        x = self.n2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

# ------------------------------------------------------------
# Classifier
# ------------------------------------------------------------
class Classifier(nn.Module):
    """
    Token + positional embedding -> N blocks -> LayerNorm -> masked mean pool -> classification head.
    """
    def __init__(self, vocab_size: int, context_len: int, d: int, h: int, n_layers: int,
                 drop: float, attn_kind: str, device: str, qkv_bias: bool = False,
                 proj_dim: int = 128, m_features: int = 256, favor_seed: Optional[int] = 42,
                 num_classes: int = 2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(context_len, d)
        self.drop    = nn.Dropout(drop)

        blocks = []
        for _ in range(n_layers):
            if attn_kind == "softmax":
                blocks.append(TransformerBlock(d, h, drop, qkv_bias=qkv_bias))
            elif attn_kind == "angular":
                blocks.append(AngularBlock(d, h, drop, qkv_bias=qkv_bias))
            elif attn_kind == "linear":
                blocks.append(LinearBlock(d, h, drop, qkv_bias=qkv_bias))
            elif attn_kind == "linformer":
                blocks.append(LinformerBlock(d, h, drop, proj_dim, context_len, qkv_bias=qkv_bias))
            elif attn_kind == "race":
                blocks.append(RACEBlock(d, h, drop, device, qkv_bias=qkv_bias, M=1, K=2, L=2))
            elif attn_kind == "performer":
                blocks.append(PerformerBlock(d, h, drop, qkv_bias=qkv_bias, m_features=m_features, seed=favor_seed))
            else:
                raise ValueError(f"Unknown attention kind: {attn_kind}")
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.norm(h)
        # CLS-free: masked mean pooling over time
        if mask is not None:
            m = mask.unsqueeze(-1).to(h.dtype)  # B,T,1
            h_sum = (h * m).sum(dim=1)
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = h_sum / denom
        else:
            pooled = h.mean(dim=1)
        return self.head(pooled)

# ------------------------------------------------------------
# Training / Evaluation
# ------------------------------------------------------------
@dataclass
class TrainConfig:
    dataset: str = "qnli"
    attn: str = "softmax"
    model_name: str = "gpt2"
    out_dir: str = "./outputs"
    context_len: int = 512
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 4
    drop: float = 0.1
    qkv_bias: bool = False
    proj_dim: int = 128
    m_features: int = 256
    favor_seed: int = 42
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-5
    wd: float = 5e-5
    max_grad_norm: float = 1.0
    workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    seed: int = 42
    compile: bool = False

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(-1) == y).float().mean().item()

def run(cfg: TrainConfig, lr=1e-5, wd=5e-05) -> Dict[str, List[float]]:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    tokenizer = build_tokenizer(cfg.model_name)

    # IMPORTANT: len(tokenizer) includes added special tokens; vocab_size does not.
    vocab_size = len(tokenizer)

    loader_pack = DATASETS[cfg.dataset](cfg.context_len, cfg.batch_size, cfg.workers, tokenizer)
    train_loader, val_loader = loader_pack.train, loader_pack.val
    num_classes = loader_pack.num_classes

    model = Classifier(
        vocab_size=vocab_size,
        context_len=cfg.context_len,
        d=cfg.d_model,
        h=cfg.n_heads,
        n_layers=cfg.n_layers,
        drop=cfg.drop,
        attn_kind=cfg.attn,
        device=cfg.device,
        qkv_bias=cfg.qkv_bias,
        proj_dim=cfg.proj_dim,
        m_features=cfg.m_features,
        favor_seed=cfg.favor_seed,
        num_classes=num_classes,
    ).to(cfg.device)

    # --- Guards: catch issues early (first batch) ---
    xb, mb, yb = next(iter(train_loader))
    mx_tok = int(xb.max())
    T = xb.shape[1]
    assert model.tok_emb.num_embeddings == vocab_size, \
        f"Embedding size {model.tok_emb.num_embeddings} != len(tokenizer) {vocab_size}"
    assert mx_tok < model.tok_emb.num_embeddings, \
        f"Max token id {mx_tok} >= embedding size {model.tok_emb.num_embeddings}"
    assert T <= model.pos_emb.num_embeddings, \
        f"Seq len {T} > context_len {model.pos_emb.num_embeddings} (increase --context_len or reduce tokenizer max_length)"
    del xb, mb, yb
    # ------------------------------------------------

    # Updated AMP API (remove deprecation warnings)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and "cuda" in cfg.device))
    module = torch.compile(model) if (cfg.compile and hasattr(torch, "compile")) else model
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_time": [], "val_time":  [],
    }

    for epoch in range(1, cfg.epochs + 1):
        # ---------------- Train ----------------
        module.train()
        t0 = time.time()
        loss_sum = 0.0
        acc_sum  = 0.0
        steps = 0

        for x, mask, y in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            mask = mask.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(cfg.amp and "cuda" in cfg.device)):
                logits = module(x, mask)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.detach().item()
            acc_sum  += (logits.argmax(-1) == y).float().mean().item()
            steps += 1

        tr_time = time.time() - t0
        tr_loss = loss_sum / max(1, steps)
        tr_acc  = acc_sum  / max(1, steps)

        # ---------------- Val ------------------
        module.eval()
        t1 = time.time()
        v_loss_sum = 0.0
        v_acc_sum  = 0.0
        v_steps = 0
        with torch.no_grad():
            for x, mask, y in val_loader:
                x = x.to(cfg.device, non_blocking=True)
                mask = mask.to(cfg.device, non_blocking=True)
                y = y.to(cfg.device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(cfg.amp and "cuda" in cfg.device)):
                    logits = module(x, mask)
                    loss = F.cross_entropy(logits, y)
                v_loss_sum += v_loss.item()
                v_acc_sum  += (logits.argmax(-1) == y).float().mean().item()
                v_steps += 1
        va_time = time.time() - t1
        va_loss = v_loss_sum / max(1, v_steps)
        va_acc  = v_acc_sum  / max(1, v_steps)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_time"].append(tr_time)
        history["val_time"].append(va_time)

        print(f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val_loss {va_loss:.4f} acc {va_acc:.4f} | t_train {tr_time:.2f}s t_val {va_time:.2f}s")

    # Dump metrics
    metrics_path = os.path.join(cfg.out_dir, f"metrics_{cfg.dataset}_{cfg.attn}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "history": history}, f, indent=2)
    print(f"[Saved] {metrics_path}")

    return history

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Anonymous Text Classification Baselines")
    p.add_argument("--dataset", type=str, default="qnli", choices=list(DATASETS.keys()))
    p.add_argument("--attn", type=str, default="softmax",
                   choices=["softmax", "angular", "linear", "linformer", "race", "performer"])
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--context_len", type=int, default=512)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--qkv_bias", action="store_true")
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--m_features", type=int, default=256)
    p.add_argument("--favor_seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=5e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")
    return p

def main():
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        dataset=args.dataset,
        attn=args.attn,
        model_name=args.model_name,
        out_dir=args.out_dir,
        context_len=args.context_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        drop=args.drop,
        qkv_bias=args.qkv_bias,
        proj_dim=args.proj_dim,
        m_features=args.m_features,
        favor_seed=args.favor_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        max_grad_norm=args.max_grad_norm,
        workers=args.workers,
        device=args.device,
        amp=args.amp,
        seed=args.seed,
        compile=args.compile,
    )
    run(cfg)

if __name__ == "__main__":
    main()

# Sample Commands:
# python baselines_text_classification.py --dataset qnli --attn softmax --epochs 5 --device cuda:1
# python baselines_text_classification.py --dataset qqp  --attn performer --epochs 5 --device cuda:1
# python baselines_text_classification.py --dataset sst2 --attn linformer --epochs 5 --device cuda:1
# python baselines_text_classification.py --dataset imdb --attn linear --epochs 5 --device cuda:1
