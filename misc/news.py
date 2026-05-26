import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import time
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader
import torchvision
from torchvision import transforms
from tqdm import tqdm
import wandb
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_float32_matmul_precision("high")

# ==================================================
# Config
# ==================================================
VISION_CONFIG = {
    # Dataset / ViT
    "dataset_name": "flowers102",       # "oxford_pet", "fashionmnist", "flowers102"
    "data_root": "./data",
    "batch_size": 2,                    # safer default for 512x512 datasets
    "img_size": 512,
    "patch_size": 4,
    "num_channels": 3,
    "num_patches": (512 // 4) ** 2,

    # Model
    "num_heads": 4,
    "embed_dim": 384,
    "transformer_units": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,

    # RACE params
    "K": 2,
    "L": 5,
    "M": 1,

    # Exact branch support
    "hyper_num_bits": 5,
    "hyper_block_size": 128,             # 128/256 is much more GPU-friendly than 32
    "hyper_neighbor_blocks": 0,          # this prototype uses same-block only
    "global_tokens": 8,
    "local_window": 16,

    # Chunk sizes for memory control
    "exact_q_chunk_size": 64,
    "lsh_block_chunk": 64,

    # Gate MLP
    "gate_hidden_dim": 128,

    # Training
    "epochs": 150,
    "lr": 6e-4,
    "weight_decay": 0.1,
    "grad_accum_steps": 2,
    "seed": 123,
    "wandb_project": "RACE",
}

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EPS = 1e-6


# ==================================================
# Dataset
# ==================================================
def set_dataset_cfg(cfg, dataset_name):
    dataset_name = dataset_name.lower()
    cfg["dataset_name"] = dataset_name

    if dataset_name == "fashionmnist":
        cfg["img_size"] = 28
        cfg["patch_size"] = 1
        cfg["num_channels"] = 1
        cfg["num_classes"] = 10
        cfg["num_patches"] = 784
        cfg["batch_size"] = max(cfg.get("batch_size", 4), 16)

    elif dataset_name == "oxford_pet":
        cfg["img_size"] = 512
        cfg["patch_size"] = 4
        cfg["num_channels"] = 3
        cfg["num_classes"] = 37
        cfg["num_patches"] = (512 // 4) ** 2

    elif dataset_name == "flowers102":
        cfg["img_size"] = 512
        cfg["patch_size"] = 4
        cfg["num_channels"] = 3
        cfg["num_classes"] = 102
        cfg["num_patches"] = (512 // 4) ** 2

    else:
        raise ValueError(dataset_name)


def get_data(cfg):
    dataset_name = cfg["dataset_name"].lower()
    root = cfg.get("data_root", "./data")
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]

    if dataset_name == "fashionmnist":
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(28, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.2860], std=[0.3530]),
        ])
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.2860], std=[0.3530]),
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=val_transform)

    elif dataset_name == "oxford_pet":
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = torchvision.datasets.OxfordIIITPet(root=root, split="trainval", target_types="category", download=True, transform=train_transform)
        val_dataset = torchvision.datasets.OxfordIIITPet(root=root, split="test", target_types="category", download=True, transform=val_transform)

    elif dataset_name == "flowers102":
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = torchvision.datasets.Flowers102(root=root, split="train", download=True, transform=train_transform)
        val_dataset = torchvision.datasets.Flowers102(root=root, split="val", download=True, transform=val_transform)

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    train_loader = dataloader.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = dataloader.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Dataset: {dataset_name}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"num_classes = {cfg['num_classes']}")
    print(f"num_channels = {cfg['num_channels']}")
    print(f"num_patches = {cfg['num_patches']}")
    print(f"batch_size = {cfg['batch_size']}")
    return train_loader, val_loader


# ==================================================
# Basic ViT utilities
# ==================================================
class PatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            cfg["num_channels"],
            cfg["embed_dim"],
            kernel_size=cfg["patch_size"],
            stride=cfg["patch_size"],
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


def _gray_code_order(num_bits: int, device):
    if num_bits == 1:
        return torch.tensor([0, 1], device=device, dtype=torch.long)

    def rec(n):
        if n == 1:
            return torch.tensor([0, 1], device=device, dtype=torch.long)
        a = rec(n - 1)
        return torch.cat([a, torch.flip(a, dims=[0]) + (1 << (n - 1))], dim=0)

    return rec(num_bits)


def gather_4d_tokens(x: torch.Tensor, idx: torch.Tensor):
    """
    x:   [B,H,T,D]
    idx: [B,H,S]
    out: [B,H,S,D]
    """
    return x.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))


class AngularLSHGray(nn.Module):
    """Hard angular LSH with Gray-code bucket ordering."""
    def __init__(self, num_bits: int, dim: int, device="cpu"):
        super().__init__()
        self.num_bits = num_bits
        self.R = 1 << num_bits
        proj_dir = torch.randn(dim, num_bits, device=device)
        perm = _gray_code_order(num_bits, device=device)
        self.register_buffer("proj_dir", proj_dir, persistent=False)  # [D,num_bits]
        self.register_buffer("perm", perm, persistent=False)          # [R]

    def hash(self, mat: torch.Tensor):
        """
        mat: [..., T, D]
        returns: [..., T]
        """
        proj = torch.einsum("...td,dr->...tr", mat, self.proj_dir)
        bits = (proj > 0).to(torch.long)
        enc = (2 ** torch.arange(self.num_bits, device=mat.device, dtype=torch.long)).view(
            *([1] * (bits.dim() - 1)), self.num_bits
        )
        bin_ids = (bits * enc).sum(dim=-1)
        return self.perm[bin_ids]


# ==================================================
# Exact full Flash/SDPA baseline
# ==================================================
def run_exact_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    Q,K,V: [B,H,T,D]
    """
    if Q.device.type == "cuda":
        Q16, K16, V16 = [t.to(torch.float16) for t in (Q, K, V)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(Q16, K16, V16, dropout_p=0.0, is_causal=False)
        return out.to(Q.dtype)
    return F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=False)


class ExactFlashAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4).contiguous()
        Q, K, V = qkv[0], qkv[1], qkv[2]
        out = run_exact_sdpa(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(self.drop(out))


# ==================================================
# Exact branch helpers: global + local window + LSH block
# ==================================================
def _masked_softmax_from_logits(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    """
    logits: [..., K]
    mask:   [..., K] bool
    Returns probs and lse. If a row has no valid keys, probs=0 and lse=-inf.
    """
    neg = torch.finfo(logits.dtype).min
    logits_m = logits.masked_fill(~mask, neg)
    valid_any = mask.any(dim=dim, keepdim=True)
    safe_logits = torch.where(valid_any, logits_m, torch.zeros_like(logits_m))
    probs = torch.softmax(safe_logits, dim=dim) * mask.to(logits.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
    probs = probs / denom
    lse = torch.logsumexp(logits_m, dim=dim)
    lse = torch.where(valid_any.squeeze(dim), lse, torch.full_like(lse, float("-inf")))
    return probs, lse


def exact_global_window_num_den(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    global_first: int,
    window: int,
    q_chunk: int,
):
    """
    Exact attention over union of local window and global-first tokens.
    Returns normalized output and log denominator.

    Q,K,V: [B,H,T,D]
    out:   [B,H,T,D]
    lse:   [B,H,T]
    """
    B, H, T, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    G = min(global_first, T)
    offsets = torch.arange(-window, window + 1, device=Q.device)
    gpos = torch.arange(G, device=Q.device)

    out = torch.zeros_like(Q)
    lse_all = torch.empty(B, H, T, device=Q.device, dtype=Q.dtype)

    K_global = K[:, :, :G, :] if G > 0 else None
    V_global = V[:, :, :G, :] if G > 0 else None

    for qs in range(0, T, q_chunk):
        qe = min(qs + q_chunk, T)
        q_idx = torch.arange(qs, qe, device=Q.device)
        q = Q[:, :, qs:qe, :]  # [B,H,q,D]
        q_len = qe - qs

        local = q_idx[:, None] + offsets[None, :]  # [q,W]
        local_valid = (local >= 0) & (local < T)
        local_clamp = local.clamp(0, T - 1)
        K_local = K[:, :, local_clamp, :]  # [B,H,q,W,D]
        V_local = V[:, :, local_clamp, :]
        logits_local = torch.einsum("bhqd,bhqwd->bhqw", q, K_local) * scale
        mask_local = local_valid.view(1, 1, q_len, -1).expand(B, H, -1, -1)

        if G > 0:
            # Add only global keys that are not already in the local window.
            l = (q_idx - window).clamp_min(0)
            r = (q_idx + window).clamp_max(T - 1)
            g_extra = ~((gpos[None, :] >= l[:, None]) & (gpos[None, :] <= r[:, None]))  # [q,G]
            logits_g = torch.einsum("bhqd,bhgd->bhqg", q, K_global) * scale
            mask_g = g_extra.view(1, 1, q_len, G).expand(B, H, -1, -1)
            V_g = V_global[:, :, None, :, :].expand(B, H, q_len, G, D)
            logits = torch.cat([logits_g, logits_local], dim=-1)
            mask = torch.cat([mask_g, mask_local], dim=-1)
            V_sel = torch.cat([V_g, V_local], dim=-2)
        else:
            logits = logits_local
            mask = mask_local
            V_sel = V_local

        probs, lse = _masked_softmax_from_logits(logits, mask, dim=-1)
        out[:, :, qs:qe, :] = torch.einsum("bhqk,bhqkd->bhqd", probs, V_sel)
        lse_all[:, :, qs:qe] = lse

    return out, lse_all


def build_gw_indices(q_idx: torch.Tensor, T: int, global_first: int, window: int):
    """
    Returns indices and mask for union(local window, global-first), without global/local duplicates.
    q_idx: [q]
    idx: [q, G+W]
    mask: [q, G+W]
    """
    device = q_idx.device
    G = min(global_first, T)
    offsets = torch.arange(-window, window + 1, device=device)
    local = q_idx[:, None] + offsets[None, :]
    local_valid = (local >= 0) & (local < T)
    local_clamp = local.clamp(0, T - 1)

    if G > 0:
        gpos = torch.arange(G, device=device)
        l = (q_idx - window).clamp_min(0)
        r = (q_idx + window).clamp_max(T - 1)
        g_extra = ~((gpos[None, :] >= l[:, None]) & (gpos[None, :] <= r[:, None]))
        g_idx = gpos[None, :].expand(q_idx.numel(), G)
        idx = torch.cat([g_idx, local_clamp], dim=-1)
        mask = torch.cat([g_extra, local_valid], dim=-1)
    else:
        idx = local_clamp
        mask = local_valid
    return idx, mask


def exact_overlap_gw_lsh(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    k_orig2sort: torch.Tensor,
    q_block_id: torch.Tensor,
    global_first: int,
    window: int,
    block_size: int,
    q_chunk: int,
):
    """
    Exact overlap between global/window support and LSH block support.
    This lets us combine exact pieces as a union:
        n_exact = n_gw + n_lsh - n_overlap.
    """
    B, H, T, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(Q)
    lse_all = torch.full((B, H, T), float("-inf"), device=Q.device, dtype=Q.dtype)

    for qs in range(0, T, q_chunk):
        qe = min(qs + q_chunk, T)
        q_idx = torch.arange(qs, qe, device=Q.device)
        q_len = qe - qs
        q = Q[:, :, qs:qe, :]

        gw_idx, gw_mask = build_gw_indices(q_idx, T, global_first, window)  # [q,U]
        U = gw_idx.size(1)
        K_sel = K[:, :, gw_idx, :]  # [B,H,q,U,D]
        V_sel = V[:, :, gw_idx, :]

        # Find which of those GW keys are also in the query's LSH key block.
        k_sorted_pos = k_orig2sort[:, :, gw_idx]                 # [B,H,q,U]
        k_block = torch.div(k_sorted_pos, block_size, rounding_mode="floor")
        qb = q_block_id[:, :, qs:qe].unsqueeze(-1)               # [B,H,q,1]
        overlap = (k_block == qb) & gw_mask.view(1, 1, q_len, U)

        logits = torch.einsum("bhqd,bhqud->bhqu", q, K_sel) * scale
        probs, lse = _masked_softmax_from_logits(logits, overlap, dim=-1)
        out[:, :, qs:qe, :] = torch.einsum("bhqu,bhqud->bhqd", probs, V_sel)
        lse_all[:, :, qs:qe] = lse

    return out, lse_all


def exact_lsh_block_num_den(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    lsh: AngularLSHGray,
    block_size: int,
    block_chunk: int,
):
    """
    HyperAttention-style sorted block exact branch.
    Same-block only, blockwise implementation.

    Returns:
      out_lsh: [B,H,T,D]
      lse_lsh: [B,H,T]
      metadata dict with q_sort_idx, k_sort_idx, q_orig2sort, k_orig2sort, q_block_id
    """
    B, H, T, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    device = Q.device

    q_ids = lsh.hash(Q)  # [B,H,T]
    k_ids = lsh.hash(K)  # [B,H,T]

    q_sort_idx = torch.argsort(q_ids, dim=2, stable=True)
    k_sort_idx = torch.argsort(k_ids, dim=2, stable=True)
    q_orig2sort = torch.argsort(q_sort_idx, dim=2, stable=True)
    k_orig2sort = torch.argsort(k_sort_idx, dim=2, stable=True)

    Qs = gather_4d_tokens(Q, q_sort_idx)
    Ks = gather_4d_tokens(K, k_sort_idx)
    Vs = gather_4d_tokens(V, k_sort_idx)

    T_pad = math.ceil(T / block_size) * block_size
    pad_len = T_pad - T
    if pad_len > 0:
        Qs = F.pad(Qs, (0, 0, 0, pad_len))
        Ks = F.pad(Ks, (0, 0, 0, pad_len))
        Vs = F.pad(Vs, (0, 0, 0, pad_len))

    num_blocks = T_pad // block_size
    valid = torch.arange(T_pad, device=device) < T
    valid_blocks = valid.view(num_blocks, block_size)  # [nb,bsz]

    Qb = Qs.view(B, H, num_blocks, block_size, D).reshape(B * H * num_blocks, block_size, D)
    Kb = Ks.view(B, H, num_blocks, block_size, D).reshape(B * H * num_blocks, block_size, D)
    Vb = Vs.view(B, H, num_blocks, block_size, D).reshape(B * H * num_blocks, block_size, D)
    vmask = valid_blocks.view(1, 1, num_blocks, block_size).expand(B, H, -1, -1).reshape(B * H * num_blocks, block_size)

    out_flat = torch.empty_like(Qb)
    lse_flat = torch.empty(B * H * num_blocks, block_size, device=device, dtype=Q.dtype)

    total_blocks = B * H * num_blocks
    for st in range(0, total_blocks, block_chunk):
        en = min(st + block_chunk, total_blocks)
        q_blk = Qb[st:en]
        k_blk = Kb[st:en]
        v_blk = Vb[st:en]
        key_mask = vmask[st:en]
        logits = torch.bmm(q_blk, k_blk.transpose(1, 2)) * scale  # [N,b,b]
        mask = key_mask[:, None, :].expand(-1, block_size, -1)
        probs, lse = _masked_softmax_from_logits(logits, mask, dim=-1)
        out_flat[st:en] = torch.bmm(probs, v_blk)
        lse_flat[st:en] = lse

    out_sorted = out_flat.view(B, H, num_blocks, block_size, D).reshape(B, H, T_pad, D)[:, :, :T, :]
    lse_sorted = lse_flat.view(B, H, num_blocks, block_size).reshape(B, H, T_pad)[:, :, :T]

    out = gather_4d_tokens(out_sorted, q_orig2sort)
    lse = lse_sorted.gather(2, q_orig2sort)

    q_block_id = torch.div(q_orig2sort, block_size, rounding_mode="floor")

    meta = {
        "q_sort_idx": q_sort_idx,
        "k_sort_idx": k_sort_idx,
        "q_orig2sort": q_orig2sort,
        "k_orig2sort": k_orig2sort,
        "q_block_id": q_block_id,
        "num_blocks": num_blocks,
        "T_pad": T_pad,
    }
    return out, lse, meta


def combine_attention_outputs(out_a, lse_a, out_b, lse_b, out_sub=None, lse_sub=None):
    """
    Stable combine of normalized attention outputs using their log denominators.
    If out_sub/lse_sub is provided, subtract that overlap contribution.
    """
    terms = [lse_a, lse_b]
    if lse_sub is not None:
        terms.append(lse_sub)
    m = torch.stack([t.nan_to_num(neginf=-1e30) for t in terms], dim=0).max(dim=0).values

    wa = torch.exp(lse_a - m).nan_to_num(0.0)
    wb = torch.exp(lse_b - m).nan_to_num(0.0)
    denom_scaled = wa + wb
    num_scaled = wa.unsqueeze(-1) * out_a + wb.unsqueeze(-1) * out_b

    if out_sub is not None and lse_sub is not None:
        ws = torch.exp(lse_sub - m).nan_to_num(0.0)
        denom_scaled = denom_scaled - ws
        num_scaled = num_scaled - ws.unsqueeze(-1) * out_sub

    denom_scaled = denom_scaled.clamp_min(EPS)
    out = num_scaled / denom_scaled.unsqueeze(-1)
    lse = m + torch.log(denom_scaled)
    # Denominator mass for m_exact; clipped to avoid inf in early training.
    d_mass = torch.exp(lse.clamp(max=20.0)).clamp_min(EPS)
    return out, lse, d_mass


# ==================================================
# RACE branch: unchanged, no exact-key elimination
# ==================================================
class RaceSoftBuckets(nn.Module):
    """RACE soft bucket probabilities using shared Q/K."""
    def __init__(self, head_dim: int, K: int, L: int, M: int, device="cpu"):
        super().__init__()
        self.d_k = head_dim
        self.K = K
        self.L = L
        self.M = M
        self.R = 1 << K
        self.S = L * self.R

        planes = torch.randn(M, head_dim, L * K, device=device)
        corners = torch.tensor(list(itertools.product([-1.0, +1.0], repeat=K)), device=device)
        self.register_buffer("planes_T", planes, persistent=False)      # [M,D,L*K]
        self.register_buffer("protos_T", corners.T, persistent=False)   # [K,R]
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0, device=device)))

    def forward(self, Q: torch.Tensor, Kt: torch.Tensor):
        """
        Q,Kt: [B,H,T,D]
        returns pQ,pK: [B,H,T,S]
        """
        B, H, T, D = Q.shape
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)
        projQ = torch.einsum("bhtd,mds->mbhts", Q, self.planes_T).view(self.M, B, H, T, self.L, self.K)
        projK = torch.einsum("bhtd,mds->mbhts", Kt, self.planes_T).view(self.M, B, H, T, self.L, self.K)
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)  # [M,B,H,T,L,R]
        logitsK = (projK.tanh().div(scale) @ self.protos_T)
        pQ = F.softmax(logitsQ, dim=-1).reshape(self.M, B, H, T, self.S).mean(dim=0)
        pK = F.softmax(logitsK, dim=-1).reshape(self.M, B, H, T, self.S).mean(dim=0)
        return pQ, pK


def race_full_readout(pQ: torch.Tensor, pK: torch.Tensor, V: torch.Tensor):
    """
    Standard/full non-causal RACE branch with no elimination.

    Returns:
      out_race: [B,H,T,D]
      d_race:   [B,H,T], proxy denominator mass used only for m_exact
    """
    A_full = pK.sum(dim=2).clamp_min(EPS)                          # [B,H,S]
    B_full = torch.einsum("bhts,bhtd->bhsd", pK, V)                 # [B,H,S,D]
    E = B_full / A_full.unsqueeze(-1)                               # [B,H,S,D]
    out = torch.einsum("bhts,bhsd->bhtd", pQ, E)                    # [B,H,T,D]
    d_race = torch.einsum("bhts,bhs->bht", pQ, A_full).clamp_min(EPS)
    return out, d_race


# ==================================================
# New attention type: no-elimination RACE + exact m-scaling
# ==================================================
class HyperGlobalLocalLSHRaceAttentionVision(nn.Module):
    """
    hyper_global_local_lsh_race

    One shared QKV projection.

    Exact branch support:
      - first global_tokens keys
      - sliding window +/- local_window
      - HyperAttention-style LSH sorted same-block keys

    RACE branch:
      - standard RACE readout, no used-key elimination

    Final:
      O = g_exact * m_exact * O_exact + g_race * O_race
      m_exact = d_exact / (d_exact + d_race + eps)
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        assert d % h == 0
        self.d = d
        self.h = h
        self.dk = d // h
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.qkv = nn.Linear(d, 3 * d, bias=cfg.get("qkv_bias", False))
        self.out_proj = nn.Linear(d, d)

        self.global_first = cfg.get("global_tokens", 8)
        self.window = cfg.get("local_window", 16)
        self.block_size = cfg.get("hyper_block_size", 128)
        self.exact_q_chunk = cfg.get("exact_q_chunk_size", 64)
        self.lsh_block_chunk = cfg.get("lsh_block_chunk", 64)

        self.lsh = AngularLSHGray(cfg.get("hyper_num_bits", 4), self.dk, device=device)
        self.race_buckets = RaceSoftBuckets(
            head_dim=self.dk,
            K=cfg["K"],
            L=cfg["L"],
            M=cfg["M"],
            device=device,
        )

        gate_hidden = cfg.get("gate_hidden_dim", 128)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )
        self.last_gates = None
        self.last_m_exact = None
        self.last_d_exact_mean = None
        self.last_d_race_mean = None

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4).contiguous()
        Q, K, V = qkv[0], qkv[1], qkv[2]  # [B,H,T,D]

        # Exact branch: global/window + LSH block + overlap correction.
        out_gw, lse_gw = exact_global_window_num_den(
            Q, K, V,
            global_first=self.global_first,
            window=self.window,
            q_chunk=self.exact_q_chunk,
        )
        out_lsh, lse_lsh, meta = exact_lsh_block_num_den(
            Q, K, V,
            lsh=self.lsh,
            block_size=self.block_size,
            block_chunk=self.lsh_block_chunk,
        )
        out_ov, lse_ov = exact_overlap_gw_lsh(
            Q, K, V,
            k_orig2sort=meta["k_orig2sort"],
            q_block_id=meta["q_block_id"],
            global_first=self.global_first,
            window=self.window,
            block_size=self.block_size,
            q_chunk=self.exact_q_chunk,
        )
        out_exact, _, d_exact = combine_attention_outputs(
            out_gw, lse_gw,
            out_lsh, lse_lsh,
            out_sub=out_ov,
            lse_sub=lse_ov,
        )  # [B,H,T,D], [B,H,T]

        # RACE branch: unchanged/full RACE, no exact-key elimination.
        pQ, pK = self.race_buckets(Q, K)  # [B,H,T,S]
        out_race, d_race = race_full_readout(pQ, pK, V)

        # Denominator-aware exact scaling with lambda = 1.
        m_exact = d_exact / (d_exact + d_race + EPS)        # [B,H,T]

        gates = torch.sigmoid(self.gate_mlp(x))             # [B,T,2]
        self.last_gates = gates.detach()
        self.last_m_exact = m_exact.detach()
        self.last_d_exact_mean = d_exact.detach().mean()
        self.last_d_race_mean = d_race.detach().mean()

        g_exact = gates[..., 0].unsqueeze(1).unsqueeze(-1)   # [B,1,T,1]
        g_race = gates[..., 1].unsqueeze(1).unsqueeze(-1)
        out_heads = g_exact * m_exact.unsqueeze(-1) * out_exact + g_race * out_race
        out = out_heads.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(self.drop(out))
        return out


# ==================================================
# Blocks
# ==================================================
class ExactFlashBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        self.att = ExactFlashAttention(d=d, h=cfg["num_heads"], drop=cfg["drop_rate"], qkv_bias=cfg.get("qkv_bias", False))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.att(self.norm1(x))
        x = self.drop(x) + h
        h = x
        x = self.ff(self.norm2(x))
        x = self.drop(x) + h
        return x


class HyperGlobalLocalLSHRaceBlock(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        self.att = HyperGlobalLocalLSHRaceAttentionVision(cfg, device=device)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.att(self.norm1(x))
        x = self.drop(x) + h
        h = x
        x = self.ff(self.norm2(x))
        x = self.drop(x) + h
        return x


class VisionTransformer(nn.Module):
    def __init__(self, cfg, attn_type, device="cpu"):
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

        if attn_type == "hyper_global_local_lsh_race":
            Block = lambda c: HyperGlobalLocalLSHRaceBlock(c, device=device)
        elif attn_type == "exact_flash":
            Block = ExactFlashBlock
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

        self.transformer_layers = nn.Sequential(*[Block(cfg) for _ in range(cfg["transformer_units"])])
        self.mlp_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg["num_classes"]))

    def forward(self, x):
        x = self.patch_embedding(x)
        B, N, d = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.transformer_layers(x)
        x = x[:, 0]
        return self.mlp_head(x)


# ==================================================
# Training utilities
# ==================================================
class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * (step / self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (1.0 - progress)
            lrs.append(lr)
        return lrs


def collect_hpr_stats(model):
    logs = {}
    gate_exact_means = []
    gate_race_means = []
    m_means = []
    d_exact_means = []
    d_race_means = []
    for li, layer in enumerate(model.transformer_layers):
        if hasattr(layer, "att"):
            att = layer.att
            if hasattr(att, "last_gates") and att.last_gates is not None:
                gates = att.last_gates.float().detach().cpu()
                ge = gates[..., 0]
                gr = gates[..., 1]
                logs[f"gates/layer{li}_exact_mean"] = ge.mean().item()
                logs[f"gates/layer{li}_race_mean"] = gr.mean().item()
                logs[f"gates/layer{li}_exact_std"] = ge.std().item()
                logs[f"gates/layer{li}_race_std"] = gr.std().item()
                gate_exact_means.append(ge.mean().item())
                gate_race_means.append(gr.mean().item())
            if hasattr(att, "last_m_exact") and att.last_m_exact is not None:
                m = att.last_m_exact.float().detach().cpu()
                logs[f"m_exact/layer{li}_mean"] = m.mean().item()
                logs[f"m_exact/layer{li}_std"] = m.std().item()
                logs[f"m_exact/layer{li}_min"] = m.min().item()
                logs[f"m_exact/layer{li}_max"] = m.max().item()
                m_means.append(m.mean().item())
            if hasattr(att, "last_d_exact_mean") and att.last_d_exact_mean is not None:
                logs[f"den/layer{li}_exact_mean"] = float(att.last_d_exact_mean.cpu())
                logs[f"den/layer{li}_race_mean"] = float(att.last_d_race_mean.cpu())
                d_exact_means.append(float(att.last_d_exact_mean.cpu()))
                d_race_means.append(float(att.last_d_race_mean.cpu()))
    if gate_exact_means:
        logs["gates/global_exact_mean"] = sum(gate_exact_means) / len(gate_exact_means)
        logs["gates/global_race_mean"] = sum(gate_race_means) / len(gate_race_means)
    if m_means:
        logs["m_exact/global_mean"] = sum(m_means) / len(m_means)
    if d_exact_means:
        logs["den/global_exact_mean"] = sum(d_exact_means) / len(d_exact_means)
        logs["den/global_race_mean"] = sum(d_race_means) / len(d_race_means)
    return logs


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, cfg, attn_type, grad_accum_steps=1):
    steps_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates = num_epochs * updates_per_epoch
    scheduler = LinearWarmupLR(optimizer, warmup_steps=max(1, int(0.01 * total_updates)), total_steps=total_updates)
    out_path = f"trial_{attn_type}_VIT.txt"

    def _log(fp, msg):
        print(msg)
        fp.write(msg + "\n")
        fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Attn: {attn_type}, Epochs: {num_epochs}")
        _log(f, "-" * 80)
        global_update = 0

        for epoch in range(1, num_epochs + 1):
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t0 = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            accum = 0

            train_iter = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
            for images, labels in train_iter:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                (loss / grad_accum_steps).backward()
                accum += 1
                preds = logits.argmax(dim=-1)
                running_correct += (preds == labels).sum().item()
                running_total += labels.numel()
                running_loss += loss.item()
                train_iter.set_postfix({
                    "loss": running_loss / max(1, len(train_iter)),
                    "acc": running_correct / max(1, running_total),
                })

                if accum == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    global_update += 1

            if accum > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            if "cuda" in str(device):
                torch.cuda.synchronize()
            train_time = time.time() - t0
            tr_l = running_loss / len(train_loader)
            tr_a = running_correct / max(1, running_total)

            if "cuda" in str(device):
                torch.cuda.synchronize()
            t1 = time.time()
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Epoch {epoch} [val]", leave=False)
                for images, labels in val_iter:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                    val_loss += loss.item()
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.numel()
                    val_iter.set_postfix({
                        "loss": val_loss / max(1, len(val_iter)),
                        "acc": val_correct / max(1, val_total),
                    })

            if "cuda" in str(device):
                torch.cuda.synchronize()
            val_time = time.time() - t1
            va_l = val_loss / len(val_loader)
            va_a = val_correct / max(1, val_total)
            lr = scheduler.get_last_lr()[0]
            extra_logs = collect_hpr_stats(model) if attn_type == "hyper_global_local_lsh_race" else {}

            wandb.log({
                "epoch": epoch,
                "train/loss": tr_l,
                "train/acc": tr_a,
                "val/loss": va_l,
                "val/acc": va_a,
                "lr": lr,
                "time/train_sec": train_time,
                "time/val_sec": val_time,
                **extra_logs,
            }, step=epoch)

            _log(
                f,
                f"Ep{epoch:3d} | train_loss {tr_l:.4f}, acc {tr_a:.4f} ({train_time:.1f}s) | "
                f"val_loss {va_l:.4f}, acc {va_a:.4f} ({val_time:.1f}s) | "
                f"lr {lr:.3e} | updates {global_update}/{total_updates}"
            )

        _log(f, "-" * 80)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")


# ==================================================
# Run
# ==================================================
def start_experiment():
    cfg = VISION_CONFIG
    set_dataset_cfg(cfg, cfg["dataset_name"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train_loader, val_loader = get_data(cfg)
    experiments = [
        ("hyper_global_local_lsh_race", cfg["grad_accum_steps"]),
        # ("exact_flash", 1),
    ]
    dataset_name = cfg["dataset_name"]

    for attn_type, grad_accum in experiments:
        run = wandb.init(
            project=cfg["wandb_project"],
            name=f"{dataset_name}_{attn_type}_N{cfg['num_patches']}",
            config={
                "dataset": dataset_name,
                "attn_type": attn_type,
                "N": cfg["num_patches"],
                "layers": cfg["transformer_units"],
                "heads": cfg["num_heads"],
                "d": cfg["embed_dim"],
                "batch_size": cfg["batch_size"],
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "epochs": cfg["epochs"],
                "grad_accum_steps": grad_accum,
                "K": cfg["K"],
                "L": cfg["L"],
                "M": cfg["M"],
                "hyper_num_bits": cfg["hyper_num_bits"],
                "hyper_block_size": cfg["hyper_block_size"],
                "global_tokens": cfg["global_tokens"],
                "local_window": cfg["local_window"],
                "gate_hidden_dim": cfg["gate_hidden_dim"],
                "race_elimination": False,
                "m_exact_formula": "d_exact / (d_exact + d_race + eps)",
            },
            settings=wandb.Settings(init_timeout=300, start_method="thread"),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("time/*", step_metric="epoch")
        wandb.define_metric("gates/*", step_metric="epoch")
        wandb.define_metric("m_exact/*", step_metric="epoch")
        wandb.define_metric("den/*", step_metric="epoch")
        wandb.define_metric("val/acc", summary="max")
        wandb.define_metric("val/loss", summary="min")

        print(f"\n=== Training {attn_type.upper()} ===")
        model = VisionTransformer(cfg, attn_type, device=DEVICE).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=cfg["weight_decay"],
        )
        train_model_simple(model, train_loader, val_loader, optimizer, DEVICE, cfg["epochs"], cfg, attn_type, grad_accum)
        wandb.finish()


if __name__ == "__main__":
    start_experiment()
