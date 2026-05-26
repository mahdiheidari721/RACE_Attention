import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch 
import wandb
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
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchvision.datasets import Food101


VISION_CONFIG = {
    "batch_size": 4,
    "img_size": 512,          # 512 × 512 images
    "patch_size": 4,          # 4 × 4 patches
    "num_channels": 3,
    "num_patches": 16384,     # (512 / 4)^2 = 128^2 = 16384 tokens
    "num_heads": 8,
    "embed_dim": 512,
    "mlp_dim": 2048,
    "transformer_units": 8,
    "drop_rate": 0.1,
    "num_classes": 50,        # we restrict to 50 classes
    "qkv_bias": False,
    "K": 4,
    "L": 4,
    "M": 1,
        # Hyper-LSH exact sparse branch
    "hyper_num_bits": 5,          # 32 buckets
    "hyper_block_size": 256,
    "hyper_min_seq_len": 4096,
    "hyper_neighbor_blocks": 0,   # start with 0, try 1 if needed

    # Tiny gate MLP for hyper_race
    "gate_hidden_dim": 128,
    "gate_normalize": True,
}

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

def _get_labels(ds):
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
        ds_train_full, class_ids=class_ids_50, total=5550, seed=seed
    )

    # --- sample 2,500 test examples across the same 50 classes ---
    test_idx = _balanced_subset_fixed_total(
        ds_test_full, class_ids=class_ids_50, total=1500, seed=seed
    )

    ds_train = Subset(ds_train_full, train_idx)
    ds_test  = Subset(ds_test_full,  test_idx)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    info = {
        "num_train": len(ds_train),   # should be 5550
        "num_test": len(ds_test),     # should be 1500
        "num_classes": len(class_ids_50),  # 50
    }
    return train_loader, test_loader, info


class PatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = nn.Conv2d(cfg["num_channels"], cfg["embed_dim"], kernel_size=cfg["patch_size"], stride=cfg["patch_size"])

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.dropout_p = dropout

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj= nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape   
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        Q, K, V = [t.to(dtype=torch.float16) for t in (Q, K, V)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=0.0,      # we keep dropout on the output like before
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.dropout(out)
        out = out.to(self.out_proj.weight.dtype)
        return self.out_proj(out)
def _gray_code_order(num_bits: int, device):
    """
    Gray-code order for bucket IDs.
    Adjacent IDs differ by one bit.
    """
    if num_bits == 1:
        return torch.tensor([0, 1], device=device, dtype=torch.long)

    def rec(n):
        if n == 1:
            return torch.tensor([0, 1], device=device, dtype=torch.long)
        a = rec(n - 1)
        return torch.cat([a, torch.flip(a, dims=[0]) + (1 << (n - 1))], dim=0)

    return rec(num_bits)


def _gather_tokens_3d(x: torch.Tensor, idx: torch.Tensor):
    """
    x   : [H, T, D]
    idx : [H, S]
    out : [H, S, D]
    """
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def _run_exact_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    Q, K, V : [B', H', L, D]
    returns : [B', H', L, D]

    Uses PyTorch SDPA with FlashAttention backend if available.
    """
    if Q.device.type == "cuda":
        Q16, K16, V16 = [t.to(dtype=torch.float16) for t in (Q, K, V)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                Q16, K16, V16,
                dropout_p=0.0,
                is_causal=False,
            )
        return out.to(Q.dtype)
    else:
        return F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=0.0,
            is_causal=False,
        )    
class TransformerArchitecture(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(cfg["embed_dim"])
        self.self_attention = MultiHeadAttention(d_in=cfg["embed_dim"], d_out=cfg["embed_dim"], dropout=cfg["drop_rate"], num_heads=cfg["num_heads"], qkv_bias=cfg["qkv_bias"])
        self.layer_norm_2 = nn.LayerNorm(cfg["embed_dim"])
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
        )

    def forward(self, x):
        residual_1 = x
        attention_output = self.self_attention(self.layer_norm_1(x))
        x = attention_output + residual_1
        residual_2 = x
        mlp_output = self.multi_layer_perceptron(self.layer_norm_2(x))
        x = mlp_output + residual_2
        return x
class AngularLSHGray(nn.Module):
    """
    Hard angular LSH with Gray-code bucket ordering.
    """
    def __init__(self, num_bits: int, dim: int, device="cpu"):
        super().__init__()
        self.num_bits = num_bits
        self.R = 1 << num_bits

        proj_dir = torch.randn(dim, num_bits, device=device)
        perm = _gray_code_order(num_bits, device=device)

        self.register_buffer("proj_dir", proj_dir, persistent=False)  # [D, num_bits]
        self.register_buffer("perm", perm, persistent=False)          # [2^num_bits]

    def hash(self, mat: torch.Tensor):
        """
        mat: [H, T, D] or [B, H, T, D]
        returns bucket IDs with same leading dims except D -> bucket id
        """
        proj = torch.einsum("...td,dr->...tr", mat, self.proj_dir)  # [..., T, num_bits]
        bits = (proj > 0).to(torch.long)

        enc = (2 ** torch.arange(self.num_bits, device=mat.device, dtype=torch.long)).view(
            *([1] * (bits.ndim - 1)), self.num_bits
        )
        bin_ids = (bits * enc).sum(dim=-1)
        return self.perm[bin_ids]


class HyperLSHExactAttentionVision(nn.Module):
    """
    HyperAttention-style exact sparse attention for vision tokens:

    - hard angular LSH on Q and K
    - sort Q by query buckets
    - sort K and V by key buckets
    - compute exact dense attention on aligned sorted blocks
    - inverse-permute query outputs back to original order
    """
    def __init__(
        self,
        d_in,
        d_out,
        dropout,
        num_heads,
        num_bits=7,
        block_size=256,
        min_seq_len=4096,
        neighbor_blocks=0,
        qkv_bias=False,
        device="cpu",
    ):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.block_size = block_size
        self.min_seq_len = min_seq_len
        self.neighbor_blocks = neighbor_blocks

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.head_dim, device=device)

    def _full_sdpa_fallback(self, Qh, Kh, Vh):
        # Qh,Kh,Vh: [H, T, D]
        return _run_exact_sdpa(
            Qh.unsqueeze(0),  # [1,H,T,D]
            Kh.unsqueeze(0),
            Vh.unsqueeze(0),
        )[0]

    def _same_block_exact(self, Qs, Ks, Vs, T_valid):
        """
        Exact attention only within aligned blocks after sorting.
        """
        H, T, D = Qs.shape
        bsz = self.block_size

        num_full_blocks = T_valid // bsz
        rem = T_valid % bsz
        out_sorted = torch.zeros_like(Qs)

        if num_full_blocks > 0:
            T_full = num_full_blocks * bsz

            Q_full = Qs[:, :T_full, :].reshape(H, num_full_blocks, bsz, D)
            K_full = Ks[:, :T_full, :].reshape(H, num_full_blocks, bsz, D)
            V_full = Vs[:, :T_full, :].reshape(H, num_full_blocks, bsz, D)

            # flatten (head, block) into batch dimension
            Q_flat = Q_full.reshape(H * num_full_blocks, 1, bsz, D)
            K_flat = K_full.reshape(H * num_full_blocks, 1, bsz, D)
            V_flat = V_full.reshape(H * num_full_blocks, 1, bsz, D)

            O_flat = _run_exact_sdpa(Q_flat, K_flat, V_flat)
            O_full = O_flat.reshape(H, num_full_blocks, bsz, D).reshape(H, T_full, D)
            out_sorted[:, :T_full, :] = O_full

        if rem > 0:
            q_last = Qs[:, num_full_blocks * bsz:, :]
            k_last = Ks[:, num_full_blocks * bsz:, :]
            v_last = Vs[:, num_full_blocks * bsz:, :]

            o_last = _run_exact_sdpa(
                q_last.unsqueeze(0),
                k_last.unsqueeze(0),
                v_last.unsqueeze(0),
            )[0]
            out_sorted[:, num_full_blocks * bsz:, :] = o_last

        return out_sorted

    def _neighbor_block_exact(self, Qs, Ks, Vs, T_valid):
        """
        Slightly richer mode:
        query block i attends to key blocks [i-neighbor_blocks ... i+neighbor_blocks]
        """
        H, T, D = Qs.shape
        bsz = self.block_size
        num_blocks = math.ceil(T_valid / bsz)

        out_sorted = torch.zeros_like(Qs)

        for bi in range(num_blocks):
            q0 = bi * bsz
            q1 = min((bi + 1) * bsz, T_valid)

            left = max(0, bi - self.neighbor_blocks)
            right = min(num_blocks - 1, bi + self.neighbor_blocks)

            k0 = left * bsz
            k1 = min((right + 1) * bsz, T_valid)

            q_blk = Qs[:, q0:q1, :]
            k_blk = Ks[:, k0:k1, :]
            v_blk = Vs[:, k0:k1, :]

            o_blk = _run_exact_sdpa(
                q_blk.unsqueeze(0),
                k_blk.unsqueeze(0),
                v_blk.unsqueeze(0),
            )[0]

            out_sorted[:, q0:q1, :] = o_blk

        return out_sorted

    def forward(self, x):
        """
        x: [B, T, d]
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2).contiguous()

        out = torch.zeros_like(Q)

        for b in range(B):
            Qh = Q[b]   # [H,T,D]
            Kh = K[b]
            Vh = V[b]

            if T < self.min_seq_len:
                out[b] = self._full_sdpa_fallback(Qh, Kh, Vh)
                continue

            q_bucket_ids = self.lsh.hash(Qh)   # [H,T]
            k_bucket_ids = self.lsh.hash(Kh)   # [H,T]

            q_sort_idx = torch.argsort(q_bucket_ids, dim=1, stable=True)
            k_sort_idx = torch.argsort(k_bucket_ids, dim=1, stable=True)
            q_sort_inv = torch.argsort(q_sort_idx, dim=1, stable=True)

            Qs = _gather_tokens_3d(Qh, q_sort_idx)
            Ks = _gather_tokens_3d(Kh, k_sort_idx)
            Vs = _gather_tokens_3d(Vh, k_sort_idx)

            if self.neighbor_blocks == 0:
                O_sorted = self._same_block_exact(Qs, Ks, Vs, T)
            else:
                O_sorted = self._neighbor_block_exact(Qs, Ks, Vs, T)

            O_unsorted = O_sorted.gather(
                1, q_sort_inv.unsqueeze(-1).expand(-1, -1, D)
            )
            out[b] = O_unsorted

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.dropout(out)
        return self.out_proj(out)


class HyperLSHExactBlock(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.att = HyperLSHExactAttentionVision(
            d_in=cfg["embed_dim"],
            d_out=cfg["embed_dim"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"],
            num_bits=cfg.get("hyper_num_bits", 7),
            block_size=cfg.get("hyper_block_size", 256),
            min_seq_len=cfg.get("hyper_min_seq_len", 4096),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )

        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"]),
        )
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
class BatchedACE(nn.Module):
    """
    Non-causal BatchedACE with optional shared planes.
    Inputs:
      Khf, Vhf, Qhf : [M, B, T, H, d_k]
    """
    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = False):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            # Shared planes [L, K, d_k] --> [d_k, (L*K)]
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer('planes_T', planes.view(L * K, d_k).T)   # [d_k, L*K]
        else:
            # Independent planes [M, L, K, d_k] --> [M, d_k, (L*K)]
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)           # [M, d_k, L*K]
            self.register_buffer('planes_T', planes)

        # Prototypes (corners of {-1,+1}^K): [K, R]
        corners = torch.tensor(list(itertools.product([-1., +1.], repeat=K)), device=device)
        self.register_buffer('protos_T', corners.T)                        # [K, R]

        # learnable temperature
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    
    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
        # Khf, Vhf, Qhf: [M, B, T, H, d_k]
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)

        if self.share_planes:
            # Collapse M·B·H → N
            N = M * B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)  # [N,T,dk]
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            # Projections to L*K
            projK = Kh2 @ self.planes_T                                     # [N,T,L*K]
            projQ = Qh2 @ self.planes_T                                     # [N,T,L*K]
        else:
            # Keep ensembles separate; collapse only B·H
            BH = B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)  # [M,BH,T,dk]
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            # One GEMM per ensemble
            projK = torch.einsum('mbtd,mds->mbts', Kh2, self.planes_T)        # [M,BH,T,L*K]
            projQ = torch.einsum('mbtd,mds->mbts', Qh2, self.planes_T)
            # Merge M,BH → N
            projK = projK.contiguous().view(M * BH, T, self.L * self.K)       # [N,T,L*K]
            projQ = projQ.contiguous().view(M * BH, T, self.L * self.K)
            V2    = V2.view(M * BH, T, dk)
            N     = M * BH

        # Reshape to [N,T,L,K] and soft-hash → probs over R buckets
        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)

        logitsK = (projK.tanh().div(scale) @ self.protos_T)                   # [N,T,L,R]
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)                   # [N,T,L,R]
        probsK  = F.softmax(logitsK, dim=-1)                                   # [N,T,L,R]
        probsQ  = F.softmax(logitsQ, dim=-1)                                   # [N,T,L,R]

        # -------- Non-causal bucket summaries over the full sequence --------
        # Collapse buckets L,R → S
        probsK_S = probsK.contiguous().view(N, T, S)                           # [N,T,S]
        probsQ_S = probsQ.contiguous().view(N, T, S)                           # [N,T,S]

        # Weighted sums across time:
        #   b_sum = probsK^T @ V   → [N,S,dk]
        b_sum = probsK_S.transpose(1, 2).bmm(V2)                               # [N,S,dk]
        #   A = sum_t probsK_t     → [N,S]
        A = probsK_S.sum(dim=1)                                                # [N,S]
        #   E = b_sum / (A + eps)  → [N,S,dk]$Ginger@0907&

        E = b_sum / (A.unsqueeze(-1) + eps)                                    # [N,S,dk]

        # Query lookup per time (no prefix): [N,T,S] @ [N,S,dk] → [N,T,dk]
        out2 = probsQ_S.bmm(E)                                                 # [N,T,dk]
        # Unflatten back to [M,B,T,H,dk]
        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4)                 # [M,B,T,H,dk]
        return out

    
    @torch.no_grad()
    def sync_from_soft(self, ace_soft):  # ace_soft is BatchedACE
        """
        Copy trained planes from BatchedACE (soft path) → this non-diff module.
        BatchedACE stores planes_T as [M, d_k, L*K]; we need [M, L, K, d_k].
        """
        M, L, K, d_k = self.M, self.L, self.K, self.d_k
        planes = (ace_soft.planes_T
                        .permute(0, 2, 1)          # [M, L*K, d_k]
                        .contiguous()
                        .view(M, L, K, d_k))       # [M, L, K, d_k]
        self.planes.copy_(planes.to(self.planes.dtype))
    
class RACEAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout,
                 num_heads, L, K, N_M, qkv_bias=False, device='cpu'):
        super().__init__()
        assert d_in % num_heads == 0
        self.H   = num_heads
        self.d_k = d_in // num_heads
        self.M   = N_M
        self.L = L
        self.P = K

        self.q_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.k_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.v_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.out    = nn.Linear(d_in, d_out)
        self.drop   = nn.Dropout(dropout)
        self.ace = BatchedACE(self.d_k, K, L, N_M, device=device)

    def forward(self, x):
        B, T, _ = x.shape
        H, d_k, M = self.H, self.d_k, self.M

        # 1) project & reshape for ACE
        Q = self.q_proj(x).view(B, T, H, d_k)
        K = self.k_proj(x).view(B, T, H, d_k)
        V = self.v_proj(x).view(B, T, H, d_k)

        # shape --> [M, B, T, H, d_k] by explicit unsqueeze
        def pack(Z):
            Zm = Z.unsqueeze(0).expand(M, -1, -1, -1, -1)
            return Zm

        Khf = pack(K)
        Vhf = pack(V)
        Qhf = pack(Q)

        # 2) run ACE
        out_hm = self.ace(Khf, Vhf, Qhf)  # [M,B,T,H,d_k]

        # 3) average ensembles & merge heads
        out = out_hm.mean(dim=0)          # [B,T,H,d_k]
        out = out.permute(0,2,1,3).reshape(B, T, H * d_k)

        # 4) final proj + dropout
        return self.drop(self.out(out))
    
class RACEBlock(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.att   = RACEAttention(
            d_in=cfg["embed_dim"], d_out=cfg["embed_dim"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"], qkv_bias=cfg["qkv_bias"],
            L=cfg["L"], K=cfg["K"], N_M=cfg["M"], device=device
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
class HyperRaceGatedAttentionVision(nn.Module):
    """
    Hybrid attention for vision:
      - Hyper-LSH exact sparse branch
      - RACE branch
      - 2-layer tiny MLP gate -> 2 scalar gates per token
      - weighted sum of the two outputs
    """
    def __init__(self, cfg, device='cpu'):
        super().__init__()

        d = cfg["embed_dim"]
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        # Branch 1: Hyper-LSH exact sparse branch
        self.hyper = HyperLSHExactAttentionVision(
            d_in=d,
            d_out=d,
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"],
            num_bits=cfg.get("hyper_num_bits", 7),
            block_size=cfg.get("hyper_block_size", 256),
            min_seq_len=cfg.get("hyper_min_seq_len", 4096),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )

        # Branch 2: RACE branch
        self.race = RACEAttention(
            d_in=d,
            d_out=d,
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"],
            L=cfg["L"],
            K=cfg["K"],
            N_M=cfg["M"],
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )

        # Tiny 2-layer MLP gate: [B,T,d] -> [B,T,2]
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        self.normalize_gates = cfg.get("gate_normalize", False)
        self.last_gates = None

    def forward(self, x):
        out_hyper = self.hyper(x)   # [B,T,d]
        out_race  = self.race(x)    # [B,T,d]

        gate_logits = self.gate_mlp(x)       # [B,T,2]
        gates = torch.sigmoid(gate_logits)   # [B,T,2]

        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        self.last_gates = gates.detach()

        g_hyper = gates[..., 0:1]   # [B,T,1]
        g_race  = gates[..., 1:2]   # [B,T,1]

        out = g_hyper * out_hyper + g_race * out_race
        return out


class HyperRaceGatedBlock(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.att = HyperRaceGatedAttentionVision(cfg, device=device)

        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])

        self.ff = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"]),
        )
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
def favorplus_features(x, proj, eps=1e-6):
    """
    FAVOR+ positive random features for softmax kernel.
    x:    [B,H,T,D]
    proj: [H,M,D]  (one matrix per head; rows ~ N(0, I))
    ->    [B,H,T,M]  (non-negative)
    """
    # x @ W^T  -> [B,H,T,M]
    xw = torch.einsum('bhtd,hmd->bhtm', x, proj)

    # stabilize across feature dimension
    xw = xw - xw.max(dim=-1, keepdim=True).values

    # exp( xW^T - ||x||^2/2 )
    exp_part  = torch.exp(xw)                         # [B,H,T,M]
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)    # [B,H,T,1]
    base      = torch.exp(-0.5 * x_norm_sq)           # [B,H,T,1]
    return exp_part * base + eps                      # strictly positive


class FavorPlusAttention(nn.Module):
    """
    Non-causal FAVOR+ (Performer) attention (softmax kernel via positive RF).
    - Pad-mask aware (mask: 1=keep, 0=pad).
    - Saves pre-projection context in self.last_ctx (B,T,d) and per-head in self.last_ctx_heads (B,H,T,dk).
    """
    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
        super().__init__()
        assert d % h == 0, "Embedding dim must be divisible by num_heads"
        self.h  = h
        self.dk = d // h
        self.m  = m_features

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        # Draw Gaussian projection matrices per head: [H,M,Dk]
        if seed is not None:
            torch.manual_seed(seed)
        proj = torch.nn.init.orthogonal_(torch.randn(h, m_features, self.dk))
        self.register_buffer("proj", proj)             # no grad; moves with device

        # For inspection/plots
        self.ctx = None
        self.eps = 1e-6

    def forward(self, x):
        """
        x:    (B, T, d)
        mask: (B, T) with 1/True = keep, 0/False = pad
        return: (B, T, d)
        """
        B, T, d = x.shape
        h, dk, m = self.h, self.dk, self.m

        # Projections -> (B,H,T,dk)
        Q = self.q(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        # Scale like softmax attention: exp(q·k / sqrt(dk)) ≡ use q/√dk inside features
        Qs = Q / math.sqrt(dk)
        Ks = K / math.sqrt(dk)


        # Positive random features
        phiQ = favorplus_features(Qs, self.proj, eps=self.eps)/ math.sqrt(m)   # [B,H,T,M]
        phiK = favorplus_features(Ks, self.proj, eps=self.eps) / math.sqrt(m)  # [B,H,T,M]


        # Global (non-causal) aggregation over time
        # KV   = sum_t phiK_t^T ⊗ V_t  -> (B,H,M,dk)
        # Ksum = sum_t phiK_t          -> (B,H,M)
        KV   = torch.einsum("bhtm,bhtd->bhmd", phiK, V)
        Ksum = phiK.sum(dim=2)

        # Per-query readout
        # num = phiQ @ KV   -> (B,H,T,dk)
        # den = phiQ · Ksum -> (B,H,T,1)
        num = torch.einsum("bhtm,bhmd->bhtd", phiQ, KV)
        den = torch.einsum("bhtm,bhm->bht",   phiQ, Ksum).unsqueeze(-1) + self.eps
        out_heads = num / den                          # (B,H,T,dk)

        # Save pre-projection context for visualization
        merged = out_heads.transpose(1, 2).contiguous().view(B, T, h * dk)
        self.ctx = merged

        # Standard output path
        merged = self.drop(merged)
        return self.o(merged)
    

class PerformerBlock(nn.Module):
    """
    Residual block with FAVOR+ attention + FFN, mirroring your LinearBlock.
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = FavorPlusAttention(
            d=cfg["embed_dim"],
            h=cfg["num_heads"],
            m_features=cfg.get("m_features", 256),
            drop=cfg["drop_rate"],
            qkv_bias=cfg.get("qkv_bias", False),
            seed=cfg.get("favor_seed", None),
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
                        nn.GELU(),
                        nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre-norm + attention
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        # FFN
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class LinearAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.eps = eps

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def kernel(self, x):
        # φ(x): positive-valued kernel feature map
        return F.elu(x) + 1  # [B, H, T, D]

    def forward(self, x):
        B, T, _ = x.size()

        # Linear projections
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, T, D]
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        # Apply kernel φ
        Q = self.kernel(Q)  # [B, H, T, D]
        K = self.kernel(K)  # [B, H, T, D]

        # Compute KV^T: [B, H, D, D]
        KV = torch.einsum('bhtd,bhte->bhde', K, V)  # [B, H, D, D]

        # Compute normalization factor: Z = Q * sum(K)
        K_sum = K.sum(dim=2)  # [B, H, D]
        Z = torch.einsum('bhtd,bhd->bht', Q, K_sum) + self.eps  # [B, H, T]

        # Compute output: Q @ (KV)
        context = torch.einsum('bhtd,bhde->bhte', Q, KV)  # [B, H, T, D]
        out = context / Z.unsqueeze(-1)  # [B, H, T, D]

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, H*D]
        return self.out_proj(out)

class LinearBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = LinearAttention(
            d_in=cfg["embed_dim"], d_out=cfg["embed_dim"],
            dropout=cfg["drop_rate"], num_heads=cfg["num_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"],cfg["mlp_dim"]),
                        nn.GELU(),
                        nn.Linear(cfg["mlp_dim"],cfg["embed_dim"])
                        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x); x = self.drop(x) + h
        return x
    
class AngularAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        self.h, self.dk = h, d//h
        self.q = nn.Linear(d,d, bias=qkv_bias)
        self.k = nn.Linear(d,d, bias=qkv_bias)
        self.v = nn.Linear(d,d, bias=qkv_bias)
        self.o = nn.Linear(d,d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B,T,_ = x.shape
        Q = F.normalize(self.q(x).view(B,T,self.h,self.dk).transpose(1,2), dim=-1)
        K = F.normalize(self.k(x).view(B,T,self.h,self.dk).transpose(1,2), dim=-1)
        V = self.v(x).view(B,T,self.h,self.dk).transpose(1,2)
        sim = (Q @ K.transpose(-2,-1)).clamp(-0.999,0.999)
        scores = 1 - torch.acos(sim)/math.pi
        W = scores.clamp(min=1e-6).pow(8)
        W = W / (W.sum(-1,keepdim=True)+1e-6)
        W = self.drop(W)
        out = (W @ V).transpose(1,2).contiguous().view(B,T,self.h*self.dk)
        return self.o(out)
    
class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = AngularAttention(d=cfg["embed_dim"], h=cfg["num_heads"], drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])

        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
                        nn.GELU(),
                        nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
    
class LinformerAttention(nn.Module):
    """
    Linformer-style attention: project K,V along sequence length T -> k (k << T),
    then do standard scaled dot-product attention with softmax over k.

    Shapes:
      x: (B, T, d_in)
      returns: (B, T, d_out)
    """
    def __init__(
        self,
        d: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        k_proj_dim: int,      # low-rank sequence dim
        max_seq_len: int    # allocate E up to this T, slice at runtime
    ):
        super().__init__()
        assert d % num_heads == 0, "d_out must be divisible by num_heads"
        self.h = num_heads
        self.dk = d // num_heads
        self.k_proj_dim = k_proj_dim
        self.max_seq_len = max_seq_len

        # token projections
        self.W_query = nn.Linear(d,  d, bias=qkv_bias)
        self.W_key   = nn.Linear(d,  d, bias=qkv_bias)
        self.W_value = nn.Linear(d,  d, bias=qkv_bias)

        # learnable sequence projections E_k, E_v: [T_max, k]
        self.E_k = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        self.E_v = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))

        nn.init.xavier_uniform_(self.E_k)
        nn.init.xavier_uniform_(self.E_v)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"
        h, dk, k = self.h, self.dk, self.k_proj_dim

        # Linear projections -> (B, h, T, dk)
        Q = self.W_query(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.W_key(  x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        # Sequence down-projection (T -> k) using E_k/E_v sliced to current T
        Ek = self.E_k[:T]  # (T, k)
        Ev = self.E_v[:T]  # (T, k)

        # K_proj, V_proj: (B, h, k, dk)
        # Contract over sequence axis
        K_proj = torch.einsum("bhtd,tk->bhkd", K, Ek)
        V_proj = torch.einsum("bhtd,tk->bhkd", V, Ev)

        # Scaled dot-product attention over compressed length k
        # scores: (B, h, T, k)
        scale = 1.0 / math.sqrt(dk)
        scores = torch.einsum("bhtd,bhkd->bhtk", Q, K_proj) * scale
        attn = F.softmax(scores, dim=-1)

        # Context: (B, h, T, dk)
        ctx = torch.einsum("bhtk,bhkd->bhtd", attn, V_proj)

        # Merge heads -> (B, T, d_out)
        out = ctx.transpose(1, 2).contiguous().view(B, T, h * dk)
        return self.out_proj(self.dropout(out))


class LinformerBlock(nn.Module):
    """
    Drop-in analogue of your LinearBlock but using LinformerAttention.
    Non-causal, no kernel; just K,V low-rank sequence projection.
    """
    def __init__(self, cfg):
        super().__init__()
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        k_proj_dim = 128

        self.att  = LinformerAttention(
            d=cfg["embed_dim"], dropout=drop, num_heads=cfg["num_heads"], qkv_bias=qkv_bias,
            k_proj_dim=k_proj_dim, max_seq_len=cfg["num_patches"] + 1
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
                        nn.GELU(),
                        nn.Linear(cfg["mlp_dim"], cfg["embed_dim"]),
                     )
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        # Attn sublayer
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        # FFN sublayer
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class VisionTransformer(nn.Module):
    def __init__(self, cfg, attn_type, device='cuda'):
        super().__init__()
        self.patch_embedding = PatchEmbedding(cfg)

        G = cfg["img_size"] // cfg["patch_size"]
        assert G * cfg["patch_size"] == cfg["img_size"], "img_size must be divisible by patch_size"
        num_patches = G * G
        d = cfg["embed_dim"]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # pick block
        if attn_type == "softmax":
            AttnBlock = TransformerArchitecture
        elif attn_type == "race":
            AttnBlock = lambda c: RACEBlock(c, device)
        elif attn_type == "hyper_lsh":
            AttnBlock = lambda c: HyperLSHExactBlock(c, device)
        elif attn_type == "hyper_race":
            AttnBlock = lambda c: HyperRaceGatedBlock(c, device)    
        elif attn_type == "angular":
            AttnBlock = AngularBlock
        elif attn_type == "linear":
            AttnBlock = LinearBlock
        elif attn_type == "linformer":
            AttnBlock = LinformerBlock
        elif attn_type == "performer":
            AttnBlock = PerformerBlock
        else:
            raise ValueError("Unsupported attention type")

        self.transformer_layers = nn.Sequential(
            *[AttnBlock(cfg) for _ in range(cfg["transformer_units"])]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg["mlp_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(cfg["mlp_dim"], cfg["num_classes"])
        )

    def forward(self, x):
        x = self.patch_embedding(x)                 # [B, N, d], N=G*G
        B, N, d = x.shape
        cls = self.cls_token.expand(B, -1, -1)      # [B,1,d]
        x = torch.cat([cls, x], dim=1)              # [B, N+1, d]
        x = x + self.pos_embed[:, :x.size(1), :]    # safe slice
        x = self.transformer_layers(x)
        x = x[:, 0]                                 # CLS
        return self.mlp_head(x)
    
class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup to base LR for `warmup_steps` optimizer updates,
    then linear decay to 0 by `total_steps`. Call scheduler.step() *after* optimizer.step().
    """
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # count optimizer steps
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * (step / self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (1.0 - progress)
            lrs.append(lr)
        return lrs


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    cfg,
    grad_accum_steps: int = 1
):
    """
    Classification-friendly training loop with:
      - gradient accumulation
      - linear warmup + linear decay LR schedule (per optimizer step)
    """
    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []
    train_times, val_times = [], []

    K, L, M = cfg.get("K", None), cfg.get("L", None), cfg.get("M", None)
    out_path = f"trial_K{K}_L{L}_M{M}_VIT.txt"

    steps_per_epoch = len(train_loader)                          # micro-steps
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)  # optimizer steps
    total_updates  = num_epochs * updates_per_epoch
    warmup_updates = max(1, int(0.1 * total_updates))           # 10% warmup

    scheduler = LinearWarmupLR(
        optimizer,
        warmup_steps=warmup_updates,
        total_steps=total_updates,
    )

    def _log(fp, msg):
        print(msg); fp.write(msg + "\n"); fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Epochs: {num_epochs}")
        _log(f, "-" * 72)
        global_update = 0

        for epoch in range(1, num_epochs + 1):
            # === TRAIN ===
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t0 = time.time()

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = 0.0
            running_correct = 0
            running_total = 0
            accum_count = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)                  # [B, C]
                loss = F.cross_entropy(outputs, labels)  # classification CE

                # scale for accumulation
                (loss / grad_accum_steps).backward()
                accum_count += 1

                # metrics (unscaled)
                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_total   += labels.size(0)
                running_loss    += loss.item()

                # update if we've accumulated enough micro-steps
                if accum_count == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()  # step LR *per optimizer step*
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_update += 1

            # flush any remainder
            if accum_count > 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            if "cuda" in str(device):
                torch.cuda.synchronize()
            train_time = time.time() - t0
            train_times.append(train_time)

            tr_l = running_loss / len(train_loader)
            tr_a = running_correct / max(1, running_total)
            train_losses.append(tr_l)
            train_accs.append(tr_a)

            # === VALIDATION ===
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t1 = time.time()

            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                    val_loss_total += loss.item()
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)

            if "cuda" in str(device):
                torch.cuda.synchronize()
            val_time = time.time() - t1
            val_times.append(val_time)

            va_l = val_loss_total / len(val_loader)
            va_a = val_correct / max(1, val_total)
            val_losses.append(va_l)
            val_accs.append(va_a)

            # current lr (take the first group)
            curr_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
            extra_logs = {}
         # Log branch gate values for hyper_race
            if hasattr(model, "transformer_layers"):
                for layer_idx, layer in enumerate(model.transformer_layers):
                    if hasattr(layer, "att") and hasattr(layer.att, "last_gates"):
                        gates = layer.att.last_gates
                        if gates is not None:
                            # gates shape: [B, T, 2]
                            hyper_g = gates[..., 0]
                            race_g  = gates[..., 1]

                            extra_logs[f"gates/layer{layer_idx}_hyper_mean"] = hyper_g.mean().item()
                            extra_logs[f"gates/layer{layer_idx}_race_mean"]  = race_g.mean().item()

                            extra_logs[f"gates/layer{layer_idx}_hyper_std"]  = hyper_g.std().item()
                            extra_logs[f"gates/layer{layer_idx}_race_std"]   = race_g.std().item()

                            extra_logs[f"gates/layer{layer_idx}_hyper_min"]  = hyper_g.min().item()
                            extra_logs[f"gates/layer{layer_idx}_race_min"]   = race_g.min().item()

                            extra_logs[f"gates/layer{layer_idx}_hyper_max"]  = hyper_g.max().item()
                            extra_logs[f"gates/layer{layer_idx}_race_max"]   = race_g.max().item()
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_l,
                "train/acc": tr_a,
                "val/loss": va_l,
                "val/acc": va_a,
                "lr": curr_lr,
                "time/train_sec": train_time,
                "time/val_sec": val_time,
                **extra_logs,
            }, step=epoch)
            _log(
                f,
                (f"Ep{epoch:3d} | "
                 f"train_loss {tr_l:.4f}, acc {tr_a:.4f} ({train_time:.1f}s) | "
                 f"val_loss {va_l:.4f}, acc {va_a:.4f} ({val_time:.1f}s) | "
                 f"lr {curr_lr:.3e} | updates {global_update}/{total_updates}")
            )

        _log(f, "-" * 72)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")

    return {
        "train_loss": train_losses, "val_loss": val_losses,
        "train_acc":  train_accs,   "val_acc":  val_accs,
        "train_time": train_times,  "val_time": val_times,
    }

def start_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, info = get_data_food101(batch_size=VISION_CONFIG["batch_size"])

    num_epochs = 100
    # ("hyper_lsh", 32),
    experiments = [
       
        #("race", 4),
        ("hyper_race", 8),
        
    ]

    for attn_type, grad_accum in experiments:
        run = wandb.init(
            project="RACE",
            name=f"food101_{attn_type}_{VISION_CONFIG['img_size']}",
            config={
                "dataset": "Food101-50class-subset",
                "attn_type": attn_type,
                "img_size": VISION_CONFIG["img_size"],
                "patch_size": VISION_CONFIG["patch_size"],
                "num_patches": VISION_CONFIG["num_patches"],
                "embed_dim": VISION_CONFIG["embed_dim"],
                "num_heads": VISION_CONFIG["num_heads"],
                "mlp_dim": VISION_CONFIG["mlp_dim"],
                "transformer_units": VISION_CONFIG["transformer_units"],
                "batch_size": VISION_CONFIG["batch_size"],
                "epochs": num_epochs,
                "lr": 3e-4,
                "weight_decay": 0.001,
                "grad_accum_steps": grad_accum,
                "num_classes": VISION_CONFIG["num_classes"],
                "K": VISION_CONFIG["K"],
                "L": VISION_CONFIG["L"],
                "M": VISION_CONFIG["M"],
                "hyper_num_bits": VISION_CONFIG.get("hyper_num_bits", None),
                "hyper_block_size": VISION_CONFIG.get("hyper_block_size", None),
                "hyper_min_seq_len": VISION_CONFIG.get("hyper_min_seq_len", None),
                "hyper_neighbor_blocks": VISION_CONFIG.get("hyper_neighbor_blocks", None),
                "gate_hidden_dim": VISION_CONFIG.get("gate_hidden_dim", None),
                "gate_normalize": VISION_CONFIG.get("gate_normalize", None),
                "train_subset_size": info["num_train"],
                "test_subset_size": info["num_test"],
            }
        )

        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("time/*", step_metric="epoch")
        wandb.define_metric("val/acc", summary="max")
        wandb.define_metric("val/loss", summary="min")

        print(f"\n=== Training {attn_type.upper()} ===")
        torch.manual_seed(123)

        model = VisionTransformer(VISION_CONFIG, attn_type, device=device)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.01
        )

        metrics = train_model_simple(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            cfg=VISION_CONFIG,
            grad_accum_steps=grad_accum
        )

        wandb.finish()
#def start_experiment():
#    device = "cuda:2"
#    train_loader, val_loader, info = get_data_food101(batch_size=VISION_CONFIG["batch_size"])
#    num_epochs = 100

    # print("Training Softmax model...")
    # torch.manual_seed(123)
    # model_gpt = VisionTransformer(VISION_CONFIG, "softmax")
    # model_gpt.to(device)
    # optimizer_gpt = torch.optim.AdamW(model_gpt.parameters(), lr=3e-4, weight_decay=0.01)

    # metrics_gpt = train_model_simple(
    #     model_gpt, train_loader, val_loader, optimizer_gpt, device,
    #     num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=32
    # )

    # print("Training RACE model...")
    # torch.manual_seed(123)
    # model_race = VisionTransformer(VISION_CONFIG, "race")
    # model_race.to(device)
    # optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=3e-4, weight_decay=0.01)

    # metrics_race = train_model_simple(
    #     model_race, train_loader, val_loader, optimizer_race, device,
    #     num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=32
    # )

    # print("Training Linformer model...")
    # torch.manual_seed(123)
    # model_linformer = VisionTransformer(VISION_CONFIG, "linformer")
    # model_linformer.to(device)
    # optimizer_race = torch.optim.AdamW(model_linformer.parameters(), lr=3e-4, weight_decay=0.01)

    # metrics_race = train_model_simple(
    #     model_linformer, train_loader, val_loader, optimizer_race, device,
    #     num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=16
    # )

    # print("Training LinearAttention...")
    # torch.manual_seed(123)
    # model_linear = VisionTransformer(VISION_CONFIG, "linear")
    # print(sum(p.numel() for p in model_linear.parameters() if p.requires_grad))
    # model_linear.to(device)
    # optimizer_linear = torch.optim.AdamW(model_linear.parameters(), lr=3e-4, weight_decay=0.01)

    # metrics_linear = train_model_simple(
    #     model_linear, train_loader, val_loader, optimizer_linear, device,
    #     num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=32
    # )

    # print("Training Angular Attention....")
    # torch.manual_seed(123)
    # model_angular = torch.compile(VisionTransformer(VISION_CONFIG, "angular"))
    # model_angular.to(device)
    # optimizer_angular = torch.optim.AdamW(model_angular.parameters(), lr=3e-4, weight_decay=0.01)

    # metrics_angular = train_model_simple(
    #     model_angular, train_loader, val_loader, optimizer_angular, device,
    #     num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=16
    # )

    # print("Training Performer Attention....")
    # torch.manual_seed(123)
    # model_performer = VisionTransformer(VISION_CONFIG, "performer")
    # model_performer.to(device)
    # optimizer_performer = torch.optim.AdamW(model_performer.parameters(), lr=3e-4, weight_decay=0.01)

    # metrics_performer = train_model_simple(
    #     model_performer, train_loader, val_loader, optimizer_performer, device,
    #     num_epochs=num_epochs, cfg=VISION_CONFIG, grad_accum_steps=32
    # )


start_experiment()