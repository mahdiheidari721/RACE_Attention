import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import math
import itertools
import random
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader
import torchvision
from torchvision import transforms
from torch.nn.attention import sdpa_kernel, SDPBackend
from tqdm import tqdm
import wandb

try:
    from flash_attn import flash_attn_func as flash_attn_func_cuda
except ImportError:
    flash_attn_func_cuda = None


torch.set_float32_matmul_precision("high")

# =========================================================
# Configuration
# =========================================================
CFG = {
    # data
    "dataset_name": "oxford_pet",   # fashionmnist | oxford_pet | flowers102
    "data_root": "./data",
    "batch_size": 16,
    "img_size": 28,
    "patch_size": 1,
    "num_channels": 1,
    "num_patches": 784,
    "num_classes": 10,

    # model
    "attn_type": "hybrid_hyper_race",   # exact_flash | race | hyper_sparse | hybrid_hyper_race
    "embed_dim": 384,
    "num_heads": 4,
    "transformer_units": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "mlp_ratio": 4.0,

    # Hyper-LSH support
    "hyper_num_bits": 4,
    "hyper_block_size": 32,
    "hyper_min_seq_len": 256,
    "hyper_neighbor_blocks": 0,

    # RACE
    "K": 2,
    "L": 5,
    "M": 1,
    "race_q_chunk_size": 256,

    # hybrid merge
    "gate_hidden_dim": 128,
    "gate_normalize": False,
    "detach_race_proxy": True,

    # denominator estimator for the exact sparse branch
    #   support_only | race_scaled | race_affine
    "denom_estimator": "race_scaled",
    "denom_eps": 1e-6,
    "race_scaled_c_init": 1.0,
    "race_affine_c_init": 1.0,
    "race_affine_b_init": 0.0,

    # training
    "epochs": 150,
    "lr": 6e-4,
    "weight_decay": 0.1,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "grad_accum_steps": 2,
    "seed": 123,
}


# =========================================================
# Data
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_dataset_cfg(cfg: Dict, dataset_name: str) -> None:
    dataset_name = dataset_name.lower()
    cfg["dataset_name"] = dataset_name

    if dataset_name == "fashionmnist":
        cfg["img_size"] = 28
        cfg["patch_size"] = 1
        cfg["num_channels"] = 1
        cfg["num_classes"] = 10
        cfg["num_patches"] = 28 * 28

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
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def get_data(cfg: Dict):
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
        train_dataset = torchvision.datasets.FashionMNIST(
            root=root, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.FashionMNIST(
            root=root, train=False, download=True, transform=val_transform
        )

    elif dataset_name == "oxford_pet":
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(
                img_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                int(img_size * 256 / 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        train_dataset = torchvision.datasets.OxfordIIITPet(
            root=root,
            split="trainval",
            target_types="category",
            download=True,
            transform=train_transform,
        )
        val_dataset = torchvision.datasets.OxfordIIITPet(
            root=root,
            split="test",
            target_types="category",
            download=True,
            transform=val_transform,
        )

    elif dataset_name == "flowers102":
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(
                img_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                int(img_size * 256 / 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        train_dataset = torchvision.datasets.Flowers102(
            root=root,
            split="train",
            download=True,
            transform=train_transform,
        )
        val_dataset = torchvision.datasets.Flowers102(
            root=root,
            split="val",
            download=True,
            transform=val_transform,
        )

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    train_data = dataloader.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_data = dataloader.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Dataset: {dataset_name}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"num_classes = {cfg['num_classes']}")
    print(f"num_channels = {cfg['num_channels']}")
    print(f"num_patches = {cfg['num_patches']}")
    return train_data, val_data


# =========================================================
# Fast exact attention helper
# =========================================================
def exact_attention_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    q, k, v: [B, H, T, D]
    returns : [B, H, T, D]
    """
    if flash_attn_func_cuda is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        out = flash_attn_func_cuda(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            causal=causal,
        )
        return out.transpose(1, 2)

    if q.is_cuda:
        qx = q.to(dtype=torch.float16) if q.dtype == torch.float32 else q
        kx = k.to(dtype=torch.float16) if k.dtype == torch.float32 else k
        vx = v.to(dtype=torch.float16) if v.dtype == torch.float32 else v
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(qx, kx, vx, dropout_p=0.0, is_causal=causal)
        return out.to(dtype=q.dtype)

    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


# =========================================================
# Indexing / LSH helpers
# =========================================================
def _gather_tokens_bhtd(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # x: [B,H,T,D], idx: [B,H,T] -> [B,H,T,D]
    return x.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))


def _gather_scores_bhts(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # x: [B,H,T,S], idx: [B,H,T] -> [B,H,T,S]
    return x.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))


def _gray_code_order(num_bits: int, device: torch.device) -> torch.Tensor:
    if num_bits == 1:
        return torch.tensor([0, 1], device=device, dtype=torch.long)

    def rec(n: int) -> torch.Tensor:
        if n == 1:
            return torch.tensor([0, 1], device=device, dtype=torch.long)
        a = rec(n - 1)
        return torch.cat([a, torch.flip(a, dims=[0]) + (1 << (n - 1))], dim=0)

    return rec(num_bits)


class AngularLSHGray(nn.Module):
    """
    Hard angular LSH with Gray-code bucket ordering.
    Input shape: [..., T, D]
    Output shape: [..., T]
    """
    def __init__(self, num_bits: int, dim: int, device: str = "cpu"):
        super().__init__()
        self.num_bits = num_bits
        self.R = 1 << num_bits
        proj_dir = torch.randn(dim, num_bits, device=device)
        perm = _gray_code_order(num_bits, torch.device(device))
        self.register_buffer("proj_dir", proj_dir, persistent=False)
        self.register_buffer("perm", perm, persistent=False)

    def hash(self, mat: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum("...td,dr->...tr", mat, self.proj_dir)
        bits = (proj > 0).to(torch.long)
        enc = (2 ** torch.arange(self.num_bits, device=mat.device, dtype=torch.long)).view(
            *([1] * (bits.ndim - 1)), self.num_bits
        )
        bin_ids = (bits * enc).sum(dim=-1)
        return self.perm[bin_ids]


# =========================================================
# Model primitives
# =========================================================
class PatchEmbedding(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            cfg["num_channels"],
            cfg["embed_dim"],
            kernel_size=cfg["patch_size"],
            stride=cfg["patch_size"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FeedForward(nn.Module):
    def __init__(self, d: int, mlp_ratio: float, drop: float):
        super().__init__()
        hidden = int(mlp_ratio * d)
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================
# Baseline exact attention
# =========================================================
class ExactFlashAttention(nn.Module):
    def __init__(self, d: int, num_heads: int, drop: float, qkv_bias: bool = False):
        super().__init__()
        assert d % num_heads == 0
        self.h = num_heads
        self.dk = d // num_heads
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.h, self.dk
        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()
        out = exact_attention_output(Q, K, V, causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.drop(out)
        return self.o(out)


# =========================================================
# Hyper preprocessing: compute ONCE and reuse
# =========================================================
class HyperSupportPrep(nn.Module):
    def __init__(self, d: int, num_heads: int, num_bits: int, qkv_bias: bool = False, device: str = "cpu"):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.W_query = nn.Linear(d, d, bias=qkv_bias)
        self.W_key   = nn.Linear(d, d, bias=qkv_bias)
        self.W_value = nn.Linear(d, d, bias=qkv_bias)
        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.head_dim, device=device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2).contiguous()

        q_ids = self.lsh.hash(Q)
        k_ids = self.lsh.hash(K)

        q_sort_idx = torch.argsort(q_ids, dim=2, stable=True)
        k_sort_idx = torch.argsort(k_ids, dim=2, stable=True)
        q_sort_inv = torch.argsort(q_sort_idx, dim=2, stable=True)

        Qs = _gather_tokens_bhtd(Q, q_sort_idx)
        Ks = _gather_tokens_bhtd(K, k_sort_idx)
        Vs = _gather_tokens_bhtd(V, k_sort_idx)

        return {
            "Q": Q,
            "K": K,
            "V": V,
            "q_ids": q_ids,
            "k_ids": k_ids,
            "q_sort_idx": q_sort_idx,
            "k_sort_idx": k_sort_idx,
            "q_sort_inv": q_sort_inv,
            "Qs": Qs,
            "Ks": Ks,
            "Vs": Vs,
        }


# =========================================================
# RACE core: compute branch output + proxy probs ONCE
# =========================================================
class BatchedACE(nn.Module):
    def __init__(self, d_k: int, K: int, L: int, M: int, device: str = "cpu", share_planes: bool = False):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer("planes_T", planes.view(L * K, d_k).T)
        else:
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)   # [M,d_k,L*K]
            self.register_buffer("planes_T", planes)

        corners = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=K)), device=device)
        self.register_buffer("protos_T", corners.T)               # [K,R]
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def probs_and_values(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """
        Q, K, V: [B,T,H,d_k]

        Returns
        -------
        probsQ_S : [M,B,H,T,S]
        probsK_S : [M,B,H,T,S]
        V_flat   : [M*B*H,T,d_k]
        """
        M = self.M
        B, T, H, dk = Q.shape
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)

        def pack(z: torch.Tensor) -> torch.Tensor:
            return z.unsqueeze(0).expand(M, -1, -1, -1, -1)  # [M,B,T,H,dk]

        Qhf = pack(Q)
        Khf = pack(K)
        Vhf = pack(V)

        if self.share_planes:
            N = M * B * H
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            projQ = (Qh2 @ self.planes_T).view(M, B, H, T, self.L, self.K)
            projK = (Kh2 @ self.planes_T).view(M, B, H, T, self.L, self.K)
        else:
            BH = B * H
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            projQ = torch.einsum("mbtd,mds->mbts", Qh2, self.planes_T)
            projK = torch.einsum("mbtd,mds->mbts", Kh2, self.planes_T)
            projQ = projQ.contiguous().view(M, B, H, T, self.L, self.K)
            projK = projK.contiguous().view(M, B, H, T, self.L, self.K)
            V2 = V2.contiguous().view(M * BH, T, dk)

        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)
        logitsK = (projK.tanh().div(scale) @ self.protos_T)

        probsQ = F.softmax(logitsQ, dim=-1)
        probsK = F.softmax(logitsK, dim=-1)
        probsQ_S = probsQ.contiguous().view(M, B, H, T, S)
        probsK_S = probsK.contiguous().view(M, B, H, T, S)
        return probsQ_S, probsK_S, V2


class BucketExcludedRACECore(nn.Module):
    def __init__(
        self,
        d: int,
        num_heads: int,
        L: int,
        K: int,
        M: int,
        hard_num_bits: int,
        drop: float,
        q_chunk_size: int = 256,
        qkv_bias: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        assert d % num_heads == 0
        self.H = num_heads
        self.d_k = d // num_heads
        self.M = M
        self.L = L
        self.K = K
        self.S = L * (1 << K)
        self.hard_R = 1 << hard_num_bits
        self.q_chunk_size = q_chunk_size

        self.q_proj = nn.Linear(d, d, bias=qkv_bias)
        self.k_proj = nn.Linear(d, d, bias=qkv_bias)
        self.v_proj = nn.Linear(d, d, bias=qkv_bias)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(self.d_k, K, L, M, device=device)

    def forward(
        self,
        x: torch.Tensor,
        q_hard_bucket_ids: Optional[torch.Tensor] = None,
        k_hard_bucket_ids: Optional[torch.Tensor] = None,
        return_proxy: bool = False,
    ):
        B, T, _ = x.shape
        H, dk, M, S = self.H, self.d_k, self.M, self.S

        Q = self.q_proj(x).view(B, T, H, dk)
        K = self.k_proj(x).view(B, T, H, dk)
        V = self.v_proj(x).view(B, T, H, dk)

        probsQ_S, probsK_S, V2 = self.ace.probs_and_values(Q, K, V)          # [M,B,H,T,S], [M,B,H,T,S], [N,T,dk]
        probsQ_flat = probsQ_S.contiguous().view(M * B * H, T, S)
        probsK_flat = probsK_S.contiguous().view(M * B * H, T, S)

        total_num = probsK_flat.transpose(1, 2).bmm(V2)                      # [N,S,dk]
        total_den = probsK_flat.sum(dim=1)                                   # [N,S]
        denom = total_den.unsqueeze(-1) + 1e-6
        E_all = total_num / denom
        out2 = probsQ_flat.bmm(E_all)                                        # [N,T,dk]

        # modified RACE: exclude same hard Hyper bucket from numerator only
        if q_hard_bucket_ids is not None and k_hard_bucket_ids is not None:
            BH = B * H
            q_ids = (
                q_hard_bucket_ids.contiguous().view(BH, T).unsqueeze(0).expand(M, -1, -1).contiguous().view(M * BH, T)
            )
            k_ids = (
                k_hard_bucket_ids.contiguous().view(BH, T).unsqueeze(0).expand(M, -1, -1).contiguous().view(M * BH, T)
            )

            for bucket_id in range(self.hard_R):
                qmask_b = (q_ids == bucket_id)
                if not bool(qmask_b.any().item()):
                    continue

                kmask_b = (k_ids == bucket_id).to(probsK_flat.dtype)
                same_num_b = torch.einsum("nts,nt,ntd->nsd", probsK_flat, kmask_b, V2)
                E_same_b = same_num_b / denom

                for qs in range(0, T, self.q_chunk_size):
                    qe = min(qs + self.q_chunk_size, T)
                    qmask_chunk = qmask_b[:, qs:qe]
                    if not bool(qmask_chunk.any().item()):
                        continue
                    remove_chunk = probsQ_flat[:, qs:qe, :].bmm(E_same_b)
                    out2[:, qs:qe, :] = out2[:, qs:qe, :] - (
                        qmask_chunk.unsqueeze(-1).to(out2.dtype) * remove_chunk
                    )

        out = out2.view(M, B, H, T, dk).mean(dim=0)                          # [B,H,T,dk]
        out = out.transpose(1, 2).contiguous().view(B, T, H * dk)
        out = self.drop(self.out(out))

        if not return_proxy:
            return out, {}

        probsQ_avg = probsQ_S.mean(dim=0)                                    # [B,H,T,S]
        probsK_avg = probsK_S.mean(dim=0)                                    # [B,H,T,S]
        return out, {"probsQ_avg": probsQ_avg, "probsK_avg": probsK_avg}


# =========================================================
# Residual denominator estimators (plug-in)
# =========================================================
class ResidualDenomEstimator(nn.Module):
    requires_race: bool = True

    def block_d2(
        self,
        *,
        q_prob_blk: torch.Tensor,
        A_full: torch.Tensor,
        A_H: torch.Tensor,
        row_shift: torch.Tensor,
        d1: torch.Tensor,
        q_blk: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def log_stats(self, layer_idx: int) -> Dict[str, float]:
        return {}


class SupportOnlyEstimator(ResidualDenomEstimator):
    requires_race = False

    def block_d2(self, **kwargs) -> torch.Tensor:
        d1 = kwargs["d1"]
        return torch.zeros_like(d1)


class RaceScaledEstimator(ResidualDenomEstimator):
    requires_race = True

    def __init__(self, num_heads: int, c_init: float = 1.0):
        super().__init__()
        init_log_c = math.log(max(c_init, 1e-6))
        self.log_c = nn.Parameter(torch.full((num_heads,), init_log_c))

    def block_d2(self, *, q_prob_blk, A_full, A_H, row_shift, d1, q_blk=None) -> torch.Tensor:
        # q_prob_blk: [H,q,S], A_full/A_H: [H,S], row_shift: [H,q]
        X = torch.einsum("hqs,hs->hq", q_prob_blk, A_full - A_H)
        X_shifted = torch.exp(-row_shift) * X
        c = F.softplus(self.log_c).unsqueeze(-1)                             # [H,1]
        return c * X_shifted

    def log_stats(self, layer_idx: int) -> Dict[str, float]:
        c_vals = F.softplus(self.log_c.detach()).cpu()
        logs: Dict[str, float] = {}
        for h, val in enumerate(c_vals.tolist()):
            logs[f"denom/layer{layer_idx}_c_head{h}"] = val
        logs[f"denom/layer{layer_idx}_c_mean"] = c_vals.mean().item()
        logs[f"denom/layer{layer_idx}_c_std"] = c_vals.std().item()
        return logs


class RaceAffineEstimator(ResidualDenomEstimator):
    requires_race = True

    def __init__(self, num_heads: int, c_init: float = 1.0, b_init: float = 0.0):
        super().__init__()
        init_log_c = math.log(max(c_init, 1e-6))
        init_b = math.log(math.expm1(max(b_init, 1e-6))) if b_init > 0 else -8.0
        self.log_c = nn.Parameter(torch.full((num_heads,), init_log_c))
        self.raw_b = nn.Parameter(torch.full((num_heads,), init_b))

    def block_d2(self, *, q_prob_blk, A_full, A_H, row_shift, d1, q_blk=None) -> torch.Tensor:
        X = torch.einsum("hqs,hs->hq", q_prob_blk, A_full - A_H)
        X_shifted = torch.exp(-row_shift) * X
        c = F.softplus(self.log_c).unsqueeze(-1)
        b = F.softplus(self.raw_b).unsqueeze(-1)
        return c * X_shifted + b

    def log_stats(self, layer_idx: int) -> Dict[str, float]:
        c_vals = F.softplus(self.log_c.detach()).cpu()
        b_vals = F.softplus(self.raw_b.detach()).cpu()
        logs: Dict[str, float] = {}
        for h, (cv, bv) in enumerate(zip(c_vals.tolist(), b_vals.tolist())):
            logs[f"denom/layer{layer_idx}_c_head{h}"] = cv
            logs[f"denom/layer{layer_idx}_b_head{h}"] = bv
        logs[f"denom/layer{layer_idx}_c_mean"] = c_vals.mean().item()
        logs[f"denom/layer{layer_idx}_b_mean"] = b_vals.mean().item()
        return logs


def build_denom_estimator(cfg: Dict, num_heads: int) -> ResidualDenomEstimator:
    name = cfg.get("denom_estimator", "race_scaled")
    if name == "support_only":
        return SupportOnlyEstimator()
    if name == "race_scaled":
        return RaceScaledEstimator(num_heads=num_heads, c_init=cfg.get("race_scaled_c_init", 1.0))
    if name == "race_affine":
        return RaceAffineEstimator(
            num_heads=num_heads,
            c_init=cfg.get("race_affine_c_init", 1.0),
            b_init=cfg.get("race_affine_b_init", 0.0),
        )
    raise ValueError(f"Unsupported denom_estimator: {name}")


# =========================================================
# Exact sparse branch that consumes cached Hyper / RACE stats
# =========================================================
class HyperSparseBranch(nn.Module):
    def __init__(
        self,
        d: int,
        num_heads: int,
        block_size: int,
        min_seq_len: int,
        neighbor_blocks: int,
        drop: float,
        qkv_bias: bool,
        denom_estimator: ResidualDenomEstimator,
        denom_eps: float = 1e-6,
    ):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.block_size = block_size
        self.min_seq_len = min_seq_len
        self.neighbor_blocks = neighbor_blocks
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.denom_eps = denom_eps
        self.denom_estimator = denom_estimator
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, hyper_cache: Dict[str, torch.Tensor], race_cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = hyper_cache["Q"]
        K = hyper_cache["K"]
        V = hyper_cache["V"]
        Qs = hyper_cache["Qs"]
        Ks = hyper_cache["Ks"]
        Vs = hyper_cache["Vs"]
        q_sort_idx = hyper_cache["q_sort_idx"]
        k_sort_idx = hyper_cache["k_sort_idx"]
        q_sort_inv = hyper_cache["q_sort_inv"]

        # Short sequence fallback: exact full attention using the same Hyper branch projections
        if T < self.min_seq_len:
            out = exact_attention_output(Q, K, V, causal=False)
            out = out.transpose(1, 2).contiguous().view(B, T, H * D)
            out = self.dropout(out)
            return self.out_proj(out)

        q_probs_sorted = None
        k_probs_sorted = None
        if self.denom_estimator.requires_race:
            assert race_cache is not None, "This denominator estimator needs RACE proxy statistics."
            q_probs_sorted = _gather_scores_bhts(race_cache["probsQ_avg"], q_sort_idx)   # [B,H,T,S]
            k_probs_sorted = _gather_scores_bhts(race_cache["probsK_avg"], k_sort_idx)   # [B,H,T,S]

        out_sorted = torch.zeros_like(Qs)  # [B,H,T,D]
        num_blocks = math.ceil(T / self.block_size)

        for b in range(B):
            A_full = None
            if self.denom_estimator.requires_race:
                A_full = k_probs_sorted[b].sum(dim=1)    # [H,S]

            for bi in range(num_blocks):
                q0 = bi * self.block_size
                q1 = min((bi + 1) * self.block_size, T)

                left = max(0, bi - self.neighbor_blocks)
                right = min(num_blocks - 1, bi + self.neighbor_blocks)
                k0 = left * self.block_size
                k1 = min((right + 1) * self.block_size, T)

                q_blk = Qs[b, :, q0:q1, :]        # [H,q,D]
                k_blk = Ks[b, :, k0:k1, :]        # [H,k,D]
                v_blk = Vs[b, :, k0:k1, :]        # [H,k,D]

                logits = torch.einsum("hqd,hkd->hqk", q_blk, k_blk) * self.scale
                row_shift = logits.max(dim=-1).values               # [H,q]
                y = torch.exp(logits - row_shift.unsqueeze(-1))     # [H,q,k]

                n1_blk = torch.einsum("hqk,hkd->hqd", y, v_blk)    # [H,q,D]
                d1_blk = y.sum(dim=-1)                              # [H,q]

                if self.denom_estimator.requires_race:
                    q_prob_blk = q_probs_sorted[b, :, q0:q1, :]         # [H,q,S]
                    A_H = k_probs_sorted[b, :, k0:k1, :].sum(dim=1)     # [H,S]
                    d2_blk = self.denom_estimator.block_d2(
                        q_prob_blk=q_prob_blk,
                        A_full=A_full,
                        A_H=A_H,
                        row_shift=row_shift,
                        d1=d1_blk,
                        q_blk=q_blk,
                    )                                                   # [H,q]
                else:
                    d2_blk = torch.zeros_like(d1_blk)

                denom_blk = (d1_blk + d2_blk).clamp_min(self.denom_eps)
                out_sorted[b, :, q0:q1, :] = n1_blk / denom_blk.unsqueeze(-1)

        out = out_sorted.gather(2, q_sort_inv.unsqueeze(-1).expand(-1, -1, -1, D))
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.dropout(out)
        return self.out_proj(out)


# =========================================================
# Attention modules
# =========================================================
class RACEAttentionOnly(nn.Module):
    def __init__(self, cfg: Dict, device: str = "cpu"):
        super().__init__()
        self.core = BucketExcludedRACECore(
            d=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            L=cfg["L"],
            K=cfg["K"],
            M=cfg["M"],
            hard_num_bits=cfg["hyper_num_bits"],
            drop=cfg["drop_rate"],
            q_chunk_size=cfg.get("race_q_chunk_size", 256),
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.core(x, q_hard_bucket_ids=None, k_hard_bucket_ids=None, return_proxy=False)
        return out


class HyperSparseOnlyAttention(nn.Module):
    def __init__(self, cfg: Dict, device: str = "cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        self.hyper_prep = HyperSupportPrep(
            d=d,
            num_heads=cfg["num_heads"],
            num_bits=cfg["hyper_num_bits"],
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )
        estimator = SupportOnlyEstimator()
        self.sparse_branch = HyperSparseBranch(
            d=d,
            num_heads=cfg["num_heads"],
            block_size=cfg["hyper_block_size"],
            min_seq_len=cfg["hyper_min_seq_len"],
            neighbor_blocks=cfg["hyper_neighbor_blocks"],
            drop=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            denom_estimator=estimator,
            denom_eps=cfg.get("denom_eps", 1e-6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hyper_cache = self.hyper_prep(x)
        return self.sparse_branch(x, hyper_cache=hyper_cache, race_cache=None)


class HybridHyperRaceAttention(nn.Module):
    """
    Shared-compute hybrid attention.

    Expensive parts computed ONCE:
      - Hyper support prep (Q/K/V, hashes, sorts)
      - RACE q/k/v, probs, summaries, branch-2 output

    Then branch-1 exact sparse output is formed using the cached Hyper support
    and cached RACE proxy probabilities.
    """
    def __init__(self, cfg: Dict, device: str = "cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        self.detach_race_proxy = cfg.get("detach_race_proxy", True)
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        self.hyper_prep = HyperSupportPrep(
            d=d,
            num_heads=cfg["num_heads"],
            num_bits=cfg["hyper_num_bits"],
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )

        self.race_core = BucketExcludedRACECore(
            d=d,
            num_heads=cfg["num_heads"],
            L=cfg["L"],
            K=cfg["K"],
            M=cfg["M"],
            hard_num_bits=cfg["hyper_num_bits"],
            drop=cfg["drop_rate"],
            q_chunk_size=cfg.get("race_q_chunk_size", 256),
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )

        self.denom_estimator = build_denom_estimator(cfg, cfg["num_heads"])
        self.hyper_branch = HyperSparseBranch(
            d=d,
            num_heads=cfg["num_heads"],
            block_size=cfg["hyper_block_size"],
            min_seq_len=cfg["hyper_min_seq_len"],
            neighbor_blocks=cfg["hyper_neighbor_blocks"],
            drop=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            denom_estimator=self.denom_estimator,
            denom_eps=cfg.get("denom_eps", 1e-6),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )
        self.normalize_gates = cfg.get("gate_normalize", False)
        self.last_gates: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Hyper support once
        hyper_cache = self.hyper_prep(x)

        # 2) RACE once (branch output + proxy probs)
        race_out, race_cache = self.race_core(
            x,
            q_hard_bucket_ids=hyper_cache["q_ids"],
            k_hard_bucket_ids=hyper_cache["k_ids"],
            return_proxy=self.denom_estimator.requires_race,
        )

        if self.detach_race_proxy and race_cache:
            race_cache = {k: v.detach() for k, v in race_cache.items()}

        # 3) Hyper exact sparse branch using cached stats
        hyper_out = self.hyper_branch(x, hyper_cache=hyper_cache, race_cache=race_cache)

        # 4) Same gate style as your previous hybrid blocks
        gate_logits = self.gate_mlp(x)                    # [B,T,2]
        gates = torch.sigmoid(gate_logits)               # [B,T,2]
        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        self.last_gates = gates.detach()

        g_hyper = gates[..., 0:1]
        g_race = gates[..., 1:2]
        out = g_hyper * hyper_out + g_race * race_out
        return out

    def extra_logs(self, layer_idx: int) -> Dict[str, float]:
        return self.denom_estimator.log_stats(layer_idx)


# =========================================================
# Transformer blocks
# =========================================================
class GenericBlock(nn.Module):
    def __init__(self, attn: nn.Module, d: int, mlp_ratio: float, drop: float):
        super().__init__()
        self.att = attn
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = FeedForward(d, mlp_ratio, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


# =========================================================
# Vision Transformer
# =========================================================
class VisionTransformer(nn.Module):
    def __init__(self, cfg: Dict, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = PatchEmbedding(cfg)

        G = cfg["img_size"] // cfg["patch_size"]
        assert G * cfg["patch_size"] == cfg["img_size"], "img_size must be divisible by patch_size"
        num_patches = G * G
        d = cfg["embed_dim"]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        attn_type = cfg["attn_type"]
        blocks = []
        for _ in range(cfg["transformer_units"]):
            if attn_type == "exact_flash":
                attn = ExactFlashAttention(d=d, num_heads=cfg["num_heads"], drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
            elif attn_type == "race":
                attn = RACEAttentionOnly(cfg, device=device)
            elif attn_type == "hyper_sparse":
                attn = HyperSparseOnlyAttention(cfg, device=device)
            elif attn_type == "hybrid_hyper_race":
                attn = HybridHyperRaceAttention(cfg, device=device)
            else:
                raise ValueError(f"Unsupported attn_type: {attn_type}")

            blocks.append(GenericBlock(attn, d=d, mlp_ratio=cfg["mlp_ratio"], drop=cfg["drop_rate"]))

        self.transformer_layers = nn.ModuleList(blocks)
        self.mlp_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg["num_classes"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)                # [B,N,d]
        B, N, d = x.shape
        cls = self.cls_token.expand(B, -1, -1)    # [B,1,d]
        x = torch.cat([cls, x], dim=1)            # [B,N+1,d]
        x = x + self.pos_embed[:, :x.size(1), :]
        for blk in self.transformer_layers:
            x = blk(x)
        x = x[:, 0]
        return self.mlp_head(x)


# =========================================================
# Scheduler and training
# =========================================================
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


def collect_hybrid_logs(model: VisionTransformer, attn_type: str, epoch: int) -> Dict[str, float]:
    extra_logs: Dict[str, float] = {}
    if attn_type != "hybrid_hyper_race":
        return extra_logs

    gate_stats = {}
    for layer_idx, layer in enumerate(model.transformer_layers):
        att = getattr(layer, "att", None)
        if att is None:
            continue

        # gate logs
        if hasattr(att, "last_gates") and att.last_gates is not None:
            gates = att.last_gates
            hyper_g = gates[..., 0].reshape(-1).detach().cpu().float()
            race_g = gates[..., 1].reshape(-1).detach().cpu().float()

            gate_stats[layer_idx] = {
                "hyper_mean": hyper_g.mean().item(),
                "race_mean": race_g.mean().item(),
                "hyper_std": hyper_g.std().item(),
                "race_std": race_g.std().item(),
                "hyper_min": hyper_g.min().item(),
                "race_min": race_g.min().item(),
                "hyper_max": hyper_g.max().item(),
                "race_max": race_g.max().item(),
                "hyper_hist": hyper_g[: min(20000, hyper_g.numel())].numpy(),
                "race_hist": race_g[: min(20000, race_g.numel())].numpy(),
            }

        # denominator estimator logs
        if hasattr(att, "extra_logs"):
            extra_logs.update(att.extra_logs(layer_idx))

    if gate_stats:
        hyper_means = []
        race_means = []
        for layer_idx, st in gate_stats.items():
            extra_logs[f"gates/layer{layer_idx}_hyper_mean"] = st["hyper_mean"]
            extra_logs[f"gates/layer{layer_idx}_race_mean"] = st["race_mean"]
            extra_logs[f"gates/layer{layer_idx}_hyper_std"] = st["hyper_std"]
            extra_logs[f"gates/layer{layer_idx}_race_std"] = st["race_std"]
            extra_logs[f"gates/layer{layer_idx}_hyper_min"] = st["hyper_min"]
            extra_logs[f"gates/layer{layer_idx}_race_min"] = st["race_min"]
            extra_logs[f"gates/layer{layer_idx}_hyper_max"] = st["hyper_max"]
            extra_logs[f"gates/layer{layer_idx}_race_max"] = st["race_max"]
            hyper_means.append(st["hyper_mean"])
            race_means.append(st["race_mean"])
            if epoch % 5 == 0:
                extra_logs[f"gates_hist/layer{layer_idx}_hyper"] = wandb.Histogram(st["hyper_hist"])
                extra_logs[f"gates_hist/layer{layer_idx}_race"] = wandb.Histogram(st["race_hist"])

        extra_logs["gates/global_hyper_mean"] = sum(hyper_means) / len(hyper_means)
        extra_logs["gates/global_race_mean"] = sum(race_means) / len(race_means)

    return extra_logs


def train_model(
    model: VisionTransformer,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs: int,
    cfg: Dict,
    attn_type: str,
    grad_accum_steps: int = 1,
):
    steps_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates = num_epochs * updates_per_epoch
    warmup_updates = max(1, int(0.01 * total_updates))

    scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_updates, total_steps=total_updates)
    out_path = f"trial_{attn_type}_{cfg['dataset_name']}.txt"

    def _log(fp, msg: str) -> None:
        print(msg)
        fp.write(msg + "\n")
        fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Attn: {attn_type}, epochs={num_epochs}")
        _log(f, "-" * 88)
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
            accum_count = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                (loss / grad_accum_steps).backward()
                accum_count += 1

                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_total += labels.size(0)
                running_loss += loss.item()

                if accum_count == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_update += 1

            if accum_count > 0:
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
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                    val_loss_total += loss.item()
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            if "cuda" in str(device):
                torch.cuda.synchronize()
            val_time = time.time() - t1

            va_l = val_loss_total / len(val_loader)
            va_a = val_correct / max(1, val_total)
            curr_lr = scheduler.get_last_lr()[0]

            extra_logs = collect_hybrid_logs(model, attn_type, epoch)
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": tr_l,
                    "train/acc": tr_a,
                    "val/loss": va_l,
                    "val/acc": va_a,
                    "lr": curr_lr,
                    "time/train_sec": train_time,
                    "time/val_sec": val_time,
                    **extra_logs,
                },
                step=epoch,
            )

            _log(
                f,
                (
                    f"Ep{epoch:3d} | train_loss {tr_l:.4f}, acc {tr_a:.4f} ({train_time:.1f}s) | "
                    f"val_loss {va_l:.4f}, acc {va_a:.4f} ({val_time:.1f}s) | "
                    f"lr {curr_lr:.3e} | updates {global_update}/{total_updates}"
                ),
            )

        _log(f, "-" * 88)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")


# =========================================================
# Main
# =========================================================
def start_experiment() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(CFG["seed"])
    set_dataset_cfg(CFG, CFG["dataset_name"])
    train_loader, val_loader = get_data(CFG)

    run = wandb.init(
        project="RACE",
        name=f"{CFG['dataset_name']}_{CFG['attn_type']}_{CFG.get('denom_estimator', 'na')}",
        config=CFG,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    wandb.define_metric("time/*", step_metric="epoch")
    wandb.define_metric("gates/*", step_metric="epoch")
    wandb.define_metric("denom/*", step_metric="epoch")
    wandb.define_metric("val/acc", summary="max")
    wandb.define_metric("val/loss", summary="min")

    print(f"\n=== Training {CFG['attn_type'].upper()} with denom_estimator={CFG.get('denom_estimator')} ===")
    model = VisionTransformer(CFG, device=device).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        betas=CFG["betas"],
        eps=CFG["eps"],
        weight_decay=CFG["weight_decay"],
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=CFG["epochs"],
        cfg=CFG,
        attn_type=CFG["attn_type"],
        grad_accum_steps=CFG["grad_accum_steps"],
    )
    wandb.finish()


if __name__ == "__main__":
    start_experiment()
