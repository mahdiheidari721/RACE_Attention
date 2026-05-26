import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch 
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as dataloader
import torch.nn as nn
import itertools
import math
import wandb
from torch.nn.attention import sdpa_kernel, SDPBackend
import time
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
torch.set_float32_matmul_precision('high')

VISION_CONFIG = {
    # Optional learnable lambda for:
    # m_exact = d_exact / (d_exact + lambda * d_race + eps)
    "mexact_lambda_learnable": True,   # True -> learn lambda, False -> lambda = 1
    "mexact_lambda_init": 1.0,         # lambda is initialized to 1, so it starts as old behavior
    "batch_size": 32,        # changed from 28 -> 32
    "img_size": 28,
    "patch_size": 1,
    "num_channels": 1,
    "num_patches": 784,
    "num_heads": 4,
    "embed_dim": 384,
    "transformer_units": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "hyper_plus_race_q_chunk_size": 256,    
    # RACE params (keep your current setting)
    "K": 2,
    "L": 5,
    "M": 1,
    "dataset_name": "fashionmnist",  # "oxford_pet", "fashionmnist", "flowers102"
    "data_root": "./data",
    # Hyper-LSH sparse branch
    "hyper_num_bits": 4,
    "hyper_block_size": 32,
    "hyper_min_seq_len": 256,
    "hyper_neighbor_blocks": 0,   # if accuracy is weak, first try 1
    # Extra exact keys for new hyper_race_gl_mexact
    "hyper_global_tokens": 8,
    "hyper_local_window": 16,
    "hyper_exact_q_chunk_size": 32,
    "mexact_eps": 1e-6,
    # Tiny 2-layer gate MLP
    "gate_hidden_dim": 128,
    "gate_normalize": False,      # set True if you want normalized branch weights

    # --- true_hyper_plus_race: HyperAttention ApproxD-style denominator ---
    "true_hpr_num_samples": 64,     # m in Algorithm 2
    "true_hpr_eps": 1.0,            # epsilon in Algorithm 2
    "true_hpr_kappa": 4.0,          # kappa in Algorithm 2
    "true_hpr_clip_scale": 1.0,     # concrete constant for the Theta(.) in line 6
    # --- hyper_plus_race_estimate ---
    "hyper_plus_race_estimate_num_samples": 8,   # m: exact residual samples per query
    "hyper_plus_race_estimate_c_init": 1.0,      # initial value for control-variate scale c
    "hyper_plus_race_estimate_eps": 1e-6,        # denominator floor
        # --- hyper_plus_race_nosample ---
    "hyper_plus_race_nosample_c_init": 1.0,
    "hyper_plus_race_nosample_eps": 1e-6,
    # --------------------------------------------------
# Query-dependent lambda for hyper_race_dependent_lambda:
#
# lambda_i = offset + sigmoid(w^T q_i + b)
#
# If use_bias=True, b is initialized so lambda starts near init_target.
# If use_bias=False, the formula is exactly:
#     lambda_i = offset + sigmoid(w^T q_i)
# --------------------------------------------------
    "mexact_dependent_lambda_offset": 0.3, 
    "mexact_dependent_lambda_offset_learnable": True,      # True: learn c, False: fixed c
    "mexact_dependent_lambda_offset_positive": True,       # keep c >= 0 using clamp
    "mexact_dependent_lambda_init_target": 0.8,            # initial average lambda target
    "mexact_dependent_lambda_use_bias": True,
    "mexact_dependent_lambda_detach_q": True,
    "mexact_dependent_lambda_min": 1e-6,
    "mexact_dependent_lambda_w_init_std": 1e-3,
}

def set_dataset_cfg(cfg, dataset_name):
    dataset_name = dataset_name.lower()
    cfg["dataset_name"] = dataset_name

    if dataset_name == "fashionmnist":
        cfg["img_size"] = 28
        cfg["patch_size"] = 1
        cfg["num_channels"] = 1
        cfg["num_classes"] = 10
        cfg["num_patches"] = 784

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
                interpolation=transforms.InterpolationMode.BILINEAR
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
                interpolation=transforms.InterpolationMode.BILINEAR
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
                interpolation=transforms.InterpolationMode.BILINEAR
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
                interpolation=transforms.InterpolationMode.BILINEAR
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
    print(f"Val size: {len(val_dataset)}")
    print(f"num_classes = {cfg['num_classes']}")
    print(f"num_channels = {cfg['num_channels']}")
    print(f"num_patches = {cfg['num_patches']}")

    return train_data, val_data

class PatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = nn.Conv2d(cfg["num_channels"], cfg["embed_dim"], kernel_size=cfg["patch_size"], stride=cfg["patch_size"])

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
    
def _gray_code_order(num_bits: int, device):
    """
    Gray-code order so adjacent bucket IDs differ by one bit.
    Returns a LongTensor of length 2^num_bits.
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
    returns gathered x along token dim -> [H, S, D]
    """
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def _run_exact_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    Q, K, V : [B', H', L, D]
    Returns : [B', H', L, D]

    Uses FlashAttention-backed SDPA on CUDA if possible.
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
class ExactFlashAttention(nn.Module):
    """
    Standard exact full attention using PyTorch SDPA with FlashAttention backend.
    """
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.h, self.dk

        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()

        out = _run_exact_sdpa(Q, K, V)  # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.drop(out)
        return self.o(out)
class AngularLSHGray(nn.Module):
    """
    HyperAttention-style hard angular LSH with Gray-code bucket ordering.
    Input expected in shape [..., T, D].
    Output is integer bucket IDs in shape [..., T].
    """
    def __init__(self, num_bits: int, dim: int, device="cpu"):
        super().__init__()
        self.num_bits = num_bits
        self.R = 1 << num_bits

        proj_dir = torch.randn(dim, num_bits, device=device)
        perm = _gray_code_order(num_bits, device=device)
        enc_vec = (2 ** torch.arange(num_bits, device=device, dtype=torch.long)).view(
            *([1] * 2), num_bits
        )

        self.register_buffer("proj_dir", proj_dir, persistent=False)   # [D, num_bits]
        self.register_buffer("perm", perm, persistent=False)           # [R]
        self.register_buffer("enc_vec", enc_vec, persistent=False)     # [1,1,num_bits]

    def hash(self, mat: torch.Tensor):
        """
        mat: [H, T, D] or [B, H, T, D]
        return: [H, T] or [B, H, T]
        """
        proj = torch.einsum("...td,dr->...tr", mat, self.proj_dir)  # [..., T, num_bits]
        bits = (proj > 0).to(torch.long)
        bin_ids = (bits * self.enc_vec).sum(dim=-1)
        return self.perm[bin_ids]
class HyperLSHExactAttention(nn.Module):
    """
    HyperAttention-style exact sparse attention:

    - hard angular LSH
    - sort queries and keys by bucket ID
    - reorder values with keys
    - compute dense exact attention on aligned blocks
    - inverse-permute query outputs back to original order
    """

    def __init__(
        self,
        d,
        h,
        drop,
        num_bits=5,
        block_size=256,
        min_seq_len=4096,
        neighbor_blocks=0,
        qkv_bias=False,
        device="cpu",
    ):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.num_bits = num_bits
        self.block_size = block_size
        self.min_seq_len = min_seq_len
        self.neighbor_blocks = neighbor_blocks

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.dk, device=device)

    def _full_sdpa_fallback(self, Qh, Kh, Vh):
        out = _run_exact_sdpa(
            Qh.unsqueeze(0),  # [1,H,T,D]
            Kh.unsqueeze(0),
            Vh.unsqueeze(0),
        )[0]
        return out

    def _same_block_exact(self, Qs, Ks, Vs, valid_T):
        H, T_valid, D = Qs.shape
        bsz = self.block_size

        num_full_blocks = T_valid // bsz
        rem = T_valid % bsz

        out_sorted = torch.zeros_like(Qs)

        if num_full_blocks > 0:
            T_full = num_full_blocks * bsz

            Q_full = Qs[:, :T_full, :].view(H, num_full_blocks, bsz, D)
            K_full = Ks[:, :T_full, :].view(H, num_full_blocks, bsz, D)
            V_full = Vs[:, :T_full, :].view(H, num_full_blocks, bsz, D)

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

    def _neighbor_block_exact(self, Qs, Ks, Vs, valid_T):
        H, T_valid, D = Qs.shape
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
        H, D = self.h, self.dk

        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()

        out = torch.zeros_like(Q)

        for b in range(B):
            valid_T = T

            Qh = Q[b, :, :valid_T, :]
            Kh = K[b, :, :valid_T, :]
            Vh = V[b, :, :valid_T, :]

            if valid_T < self.min_seq_len:
                out[b, :, :valid_T, :] = self._full_sdpa_fallback(Qh, Kh, Vh)
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
                O_sorted = self._same_block_exact(Qs, Ks, Vs, valid_T)
            else:
                O_sorted = self._neighbor_block_exact(Qs, Ks, Vs, valid_T)

            O_unsorted = O_sorted.gather(
                1, q_sort_inv.unsqueeze(-1).expand(-1, -1, D)
            )
            out[b, :, :valid_T, :] = O_unsorted

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.drop(out)
        return self.o(out)
class ExactFlashBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = ExactFlashAttention(
            d=d,
            h=cfg["num_heads"],
            drop=drop,
            qkv_bias=cfg["qkv_bias"],
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

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


class HyperLSHExactBlock(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperLSHExactAttention(
            d=d,
            h=cfg["num_heads"],
            drop=drop,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 256),
            min_seq_len=cfg.get("hyper_min_seq_len", 4096),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

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

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=False, dropout_p=self.dropout_p)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)
    
class TransformerArchitecture(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(cfg["embed_dim"])
        self.self_attention = MultiHeadAttention(d_in=cfg["embed_dim"], d_out=cfg["embed_dim"], dropout=cfg["drop_rate"], num_heads=cfg["num_heads"], qkv_bias=cfg["qkv_bias"])
        self.layer_norm_2 = nn.LayerNorm(cfg["embed_dim"])
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4*cfg["embed_dim"]),
            nn.GELU(),
            nn.Linear(4*cfg["embed_dim"], cfg["embed_dim"])
        )

    def forward(self, x):
        residual_1 = x
        attention_output = self.self_attention(self.layer_norm_1(x))
        x = attention_output + residual_1
        residual_2 = x
        mlp_output = self.multi_layer_perceptron(self.layer_norm_2(x))
        x = mlp_output + residual_2
        return x
# ==================================================
# Hyper-LSH exact sparse attention helpers
# ==================================================
def _gray_code_order(num_bits: int, device):
    """
    Gray-code ordering for bucket IDs.
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

    Uses FlashAttention-backed SDPA on CUDA if possible.
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
        mat: [H, T, D]
        returns bucket IDs: [H, T]
        """
        proj = torch.einsum("htd,dr->htr", mat, self.proj_dir)  # [H, T, num_bits]
        bits = (proj > 0).to(torch.long)

        enc = (2 ** torch.arange(self.num_bits, device=mat.device, dtype=torch.long)).view(
            1, 1, self.num_bits
        )
        bin_ids = (bits * enc).sum(dim=-1)
        return self.perm[bin_ids]


class HyperLSHExactAttentionVision(nn.Module):
    """
    HyperAttention-style exact sparse attention for vision tokens.

    Steps:
      1) project Q/K/V
      2) hard-hash Q and K
      3) sort Q separately from K/V
      4) compute exact dense attention on aligned sorted blocks
      5) inverse-permute query outputs back
    """
    def __init__(
        self,
        d_in,
        d_out,
        dropout,
        num_heads,
        num_bits=5,
        block_size=32,
        min_seq_len=256,
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
        return _run_exact_sdpa(
            Qh.unsqueeze(0),  # [1,H,T,D]
            Kh.unsqueeze(0),
            Vh.unsqueeze(0),
        )[0]

    def _same_block_exact(self, Qs, Ks, Vs, T_valid):
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

            q_bucket_ids = self.lsh.hash(Qh)
            k_bucket_ids = self.lsh.hash(Kh)

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
        scale = self.logit_temp.exp().clamp(1e-2, 20.0) # uncomment when you make temp learnable

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
            nn.Linear(cfg["embed_dim"], 4*cfg["embed_dim"]),
            nn.GELU(),
            nn.Linear(4*cfg["embed_dim"], cfg["embed_dim"])
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
    Hybrid attention:
      - Hyper-LSH exact sparse branch
      - RACE branch
      - tiny 2-layer MLP gate -> 2 scalar gates per token
      - weighted sum of branch outputs
    """
    def __init__(self, cfg, device='cpu'):
        super().__init__()

        d = cfg["embed_dim"]
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        self.hyper = HyperLSHExactAttentionVision(
            d_in=d,
            d_out=d,
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"],
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )

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
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"])
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
# ==================================================
# NEW ATTENTION TYPE: hyper_race_gl_mexact
# Hyper-LSH + global/local exact branch + normal RACE branch
# Final: g_hyper * m_exact * out_hyper + g_race * out_race
# RACE branch is NOT eliminated / NOT modified except returning d_race proxy.
# ==================================================
# ==================================================
# NEW ATTENTION TYPE: hyper_race_mexact
# Same as hyper_race, but multiply exact/Hyper-LSH branch by:
#     m_exact = d_exact / (d_exact + d_race + eps)
#
# No global tokens.
# No local window.
# No RACE elimination.
# ==================================================

def _gather_tokens_4d(x: torch.Tensor, idx: torch.Tensor):
    """
    x   : [B, H, T, D]
    idx : [B, H, S]
    out : [B, H, S, D]
    """
    return x.gather(
        2,
        idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
    )


class HyperLSHExactWithLogDenomAttentionVision(nn.Module):
    """
    Hyper-LSH exact sparse attention, same support as your HyperLSHExactAttentionVision,
    but it also returns a denominator proxy for each token.

    It computes:
        out_hyper = exact softmax attention over LSH sorted same-block keys

    and:
        log_d_exact(i,h) = log sum_{j in LSH-support(i,h)}
                           exp(q_i^T k_j / sqrt(d_head))

    Then it returns:
        out_hyper        : [B,T,d]
        log_d_exact_token: [B,T,1]

    Important:
    - The output path still uses _run_exact_sdpa, so the branch output uses
      FlashAttention-backed SDPA when available.
    - The log denominator is computed under no_grad, because it is only used
      for the m_exact correction signal.
    """

    def __init__(
        self,
        d_in,
        d_out,
        dropout,
        num_heads,
        num_bits=5,
        block_size=32,
        min_seq_len=256,
        neighbor_blocks=0,
        qkv_bias=False,
        device="cpu",
        mexact_eps=1e-6,
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.block_size = block_size
        self.min_seq_len = min_seq_len
        self.neighbor_blocks = neighbor_blocks
        self.mexact_eps = mexact_eps
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.head_dim, device=device)

    def _full_sdpa_fallback_with_lse(self, Qh, Kh, Vh):
        """
        Qh,Kh,Vh: [H,T,D]

        Returns:
            out_h : [H,T,D]
            lse_h : [H,T]
        """
        out_h = _run_exact_sdpa(
            Qh.unsqueeze(0),
            Kh.unsqueeze(0),
            Vh.unsqueeze(0),
        )[0]

        with torch.no_grad():
            logits = torch.einsum("hqd,hkd->hqk", Qh, Kh) * self.scale
            lse_h = torch.logsumexp(logits.float(), dim=-1).to(Qh.dtype)

        return out_h, lse_h

    def _same_block_exact_with_lse(self, Qs, Ks, Vs, valid_T):
        """
        Same-block Hyper-LSH exact attention.

        Qs,Ks,Vs: [H,T,D]

        Returns:
            out_sorted : [H,T,D]
            lse_sorted : [H,T]
        """
        H, T, D = Qs.shape
        bsz = self.block_size

        num_full_blocks = valid_T // bsz
        rem = valid_T % bsz

        out_sorted = torch.zeros_like(Qs)
        lse_sorted = torch.empty(H, valid_T, device=Qs.device, dtype=Qs.dtype)

        if num_full_blocks > 0:
            T_full = num_full_blocks * bsz

            Q_full = Qs[:, :T_full, :].reshape(H, num_full_blocks, bsz, D)
            K_full = Ks[:, :T_full, :].reshape(H, num_full_blocks, bsz, D)
            V_full = Vs[:, :T_full, :].reshape(H, num_full_blocks, bsz, D)

            Q_flat = Q_full.reshape(H * num_full_blocks, 1, bsz, D)
            K_flat = K_full.reshape(H * num_full_blocks, 1, bsz, D)
            V_flat = V_full.reshape(H * num_full_blocks, 1, bsz, D)

            # FlashAttention-backed output.
            O_flat = _run_exact_sdpa(Q_flat, K_flat, V_flat)
            O_full = O_flat.reshape(H, num_full_blocks, bsz, D).reshape(H, T_full, D)
            out_sorted[:, :T_full, :] = O_full

            # Denominator for m_exact only.
            with torch.no_grad():
                logits = torch.einsum(
                    "hnqd,hnkd->hnqk",
                    Q_full,
                    K_full,
                ) * self.scale
                lse_full = torch.logsumexp(logits.float(), dim=-1).to(Qs.dtype)
                lse_sorted[:, :T_full] = lse_full.reshape(H, T_full)

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

            with torch.no_grad():
                logits = torch.einsum("hqd,hkd->hqk", q_last, k_last) * self.scale
                lse_last = torch.logsumexp(logits.float(), dim=-1).to(Qs.dtype)
                lse_sorted[:, num_full_blocks * bsz:] = lse_last

        return out_sorted, lse_sorted

    def _neighbor_block_exact_with_lse(self, Qs, Ks, Vs, valid_T):
        """
        Neighbor-block Hyper-LSH exact attention.

        This is less efficient than same-block because it loops over blocks,
        but it preserves the behavior of your existing neighbor_blocks option.
        """
        H, T, D = Qs.shape
        bsz = self.block_size
        num_blocks = math.ceil(valid_T / bsz)

        out_sorted = torch.zeros_like(Qs)
        lse_sorted = torch.empty(H, valid_T, device=Qs.device, dtype=Qs.dtype)

        for bi in range(num_blocks):
            q0 = bi * bsz
            q1 = min((bi + 1) * bsz, valid_T)

            left = max(0, bi - self.neighbor_blocks)
            right = min(num_blocks - 1, bi + self.neighbor_blocks)

            k0 = left * bsz
            k1 = min((right + 1) * bsz, valid_T)

            q_blk = Qs[:, q0:q1, :]
            k_blk = Ks[:, k0:k1, :]
            v_blk = Vs[:, k0:k1, :]

            o_blk = _run_exact_sdpa(
                q_blk.unsqueeze(0),
                k_blk.unsqueeze(0),
                v_blk.unsqueeze(0),
            )[0]
            out_sorted[:, q0:q1, :] = o_blk

            with torch.no_grad():
                logits = torch.einsum("hqd,hkd->hqk", q_blk, k_blk) * self.scale
                lse_blk = torch.logsumexp(logits.float(), dim=-1).to(Qs.dtype)
                lse_sorted[:, q0:q1] = lse_blk

        return out_sorted, lse_sorted

    def forward(self, x):
        """
        x: [B,T,d]

        Returns:
            out_hyper          : [B,T,d]
            log_d_exact_token  : [B,T,1]
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2).contiguous()

        out_heads = torch.zeros_like(Q)
        lse_heads = torch.empty(B, H, T, device=x.device, dtype=Q.dtype)

        for b in range(B):
            Qh = Q[b]   # [H,T,D]
            Kh = K[b]
            Vh = V[b]

            if T < self.min_seq_len:
                out_b, lse_b = self._full_sdpa_fallback_with_lse(Qh, Kh, Vh)
                out_heads[b] = out_b
                lse_heads[b] = lse_b
                continue

            q_bucket_ids = self.lsh.hash(Qh)
            k_bucket_ids = self.lsh.hash(Kh)

            q_sort_idx = torch.argsort(q_bucket_ids, dim=1, stable=True)
            k_sort_idx = torch.argsort(k_bucket_ids, dim=1, stable=True)
            q_sort_inv = torch.argsort(q_sort_idx, dim=1, stable=True)

            Qs = _gather_tokens_3d(Qh, q_sort_idx)
            Ks = _gather_tokens_3d(Kh, k_sort_idx)
            Vs = _gather_tokens_3d(Vh, k_sort_idx)

            if self.neighbor_blocks == 0:
                O_sorted, LSE_sorted = self._same_block_exact_with_lse(Qs, Ks, Vs, T)
            else:
                O_sorted, LSE_sorted = self._neighbor_block_exact_with_lse(Qs, Ks, Vs, T)

            O_unsorted = O_sorted.gather(
                1,
                q_sort_inv.unsqueeze(-1).expand(-1, -1, D),
            )
            LSE_unsorted = LSE_sorted.gather(1, q_sort_inv)

            out_heads[b] = O_unsorted
            lse_heads[b] = LSE_unsorted

        # Convert per-head log denominators into one token-level log denominator:
        # d_exact_token = mean_h exp(lse_h)
        # log d_exact_token = logsumexp_h(lse_h) - log(H)
        with torch.no_grad():
            log_d_exact_token = (
                torch.logsumexp(lse_heads.float(), dim=1, keepdim=False)
                - math.log(H)
            ).to(Q.dtype).unsqueeze(-1)  # [B,T,1]

        out = out_heads.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.dropout(out)
        out = self.out_proj(out)

        return out, log_d_exact_token


class RACEAttentionWithDenom(RACEAttention):
    """
    Same RACE branch as your previous RACEAttention, but it also returns
    a denominator proxy:

        d_race(i) = mean_h mean_m sum_s pQ(i,s) A_full(s)

    No used-key elimination.
    No subtraction.
    """

    def _ace_with_denom(self, Khf, Vhf, Qhf, eps: float = 1e-6):
        ace = self.ace

        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k

        S = ace.L * ace.R
        scale = ace.logit_temp.exp().clamp(1e-2, 20.0)

        if ace.share_planes:
            N = M * B * H

            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            projK = Kh2 @ ace.planes_T
            projQ = Qh2 @ ace.planes_T

        else:
            BH = B * H

            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            projK = torch.einsum("mbtd,mds->mbts", Kh2, ace.planes_T)
            projQ = torch.einsum("mbtd,mds->mbts", Qh2, ace.planes_T)

            projK = projK.contiguous().view(M * BH, T, ace.L * ace.K)
            projQ = projQ.contiguous().view(M * BH, T, ace.L * ace.K)
            V2    = V2.contiguous().view(M * BH, T, dk)
            N     = M * BH

        projK = projK.view(N, T, ace.L, ace.K)
        projQ = projQ.view(N, T, ace.L, ace.K)

        logitsK = (projK.tanh().div(scale) @ ace.protos_T)
        logitsQ = (projQ.tanh().div(scale) @ ace.protos_T)

        probsK = F.softmax(logitsK, dim=-1)
        probsQ = F.softmax(logitsQ, dim=-1)

        probsK_S = probsK.contiguous().view(N, T, S)
        probsQ_S = probsQ.contiguous().view(N, T, S)

        total_num = probsK_S.transpose(1, 2).bmm(V2)  # [N,S,dk]
        total_den = probsK_S.sum(dim=1)               # [N,S]

        E = total_num / (total_den.unsqueeze(-1) + eps)
        out2 = probsQ_S.bmm(E)                        # [N,T,dk]

        # RACE denominator proxy.
        d2 = torch.einsum("nts,ns->nt", probsQ_S, total_den).clamp_min(eps)  # [N,T]

        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4).contiguous()
        den = d2.view(M, B, H, T).permute(0, 1, 3, 2).contiguous()           # [M,B,T,H]

        return out, den

    def forward(self, x):
        """
        Returns:
            out_race     : [B,T,d]
            d_race_token : [B,T,1]
        """
        B, T, _ = x.shape
        H, dk, M = self.H, self.d_k, self.M

        Q = self.q_proj(x).view(B, T, H, dk)
        K = self.k_proj(x).view(B, T, H, dk)
        V = self.v_proj(x).view(B, T, H, dk)

        def pack(Z):
            return Z.unsqueeze(0).expand(M, -1, -1, -1, -1)

        out_m, den_m = self._ace_with_denom(pack(K), pack(V), pack(Q))

        out_heads = out_m.mean(dim=0)  # [B,T,H,dk]

        # This preserves your existing RACEAttention output layout.
        # If you later want the mathematically standard head merge, replace this line by:
        #     out = out_heads.contiguous().view(B, T, H * dk)
        out = out_heads.permute(0, 2, 1, 3).reshape(B, T, H * dk)

        d_race_heads = den_m.mean(dim=0)  # [B,T,H]
        d_race_token = d_race_heads.mean(dim=-1, keepdim=True).clamp_min(1e-6)

        out = self.drop(self.out(out))
        return out, d_race_token


class HyperRaceMExactAttentionVision(nn.Module):
    """
    hyper_race_mexact

    Same as your original hyper_race, except the Hyper-LSH exact branch is
    multiplied by:

        m_exact = d_exact / (d_exact + d_race + eps)

    Computed in log-space as:

        m_exact = sigmoid(log_d_exact - log_d_race)

    RACE branch is unchanged. No global/local. No elimination.
    """

    def __init__(self, cfg, device="cpu"):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        self.mexact_eps = cfg.get("mexact_eps", 1e-6)
        # ------------------------------------------------------------
        # Optional learnable lambda for:
        #   m_exact = d_exact / (d_exact + lambda * d_race + eps)
        #
        # We parameterize lambda as exp(log_lambda), so lambda is always positive.
        # If mexact_lambda_learnable=False, lambda is fixed to 1.
        # ------------------------------------------------------------
        self.mexact_lambda_learnable = bool(cfg.get("mexact_lambda_learnable", False))

        lambda_init = float(cfg.get("mexact_lambda_init", 1.0))
        if lambda_init <= 0:
            raise ValueError("mexact_lambda_init must be > 0")

        if self.mexact_lambda_learnable:
            self.log_mexact_lambda = nn.Parameter(
                torch.tensor(math.log(lambda_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_mexact_lambda",
                torch.tensor(0.0, dtype=torch.float32),  # log(1)=0
                persistent=False,
            )
        self.hyper = HyperLSHExactWithLogDenomAttentionVision(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
            mexact_eps=self.mexact_eps,
        )

        self.race = RACEAttentionWithDenom(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            L=cfg["L"],
            K=cfg["K"],
            N_M=cfg["M"],
            qkv_bias=qkv_bias,
            device=device,
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        self.normalize_gates = cfg.get("gate_normalize", False)

        self.last_gates = None
        self.last_m_exact = None
        self.last_d_exact_mean = None
        self.last_d_race_mean = None
        self.last_mexact_lambda = None
    def forward(self, x):
        out_hyper, log_d_exact = self.hyper(x)  # [B,T,d], [B,T,1]
        out_race, d_race = self.race(x)         # [B,T,d], [B,T,1]

        # log-space equivalent of:
        #     d_exact / (d_exact + d_race + eps)
        #
        # Detach denominator signal to keep this a stable correction multiplier,
        # not a heavy second optimization path.
        #log_d_race = torch.log(d_race.clamp_min(self.mexact_eps))
        #m_exact = torch.sigmoid((log_d_exact - log_d_race).detach())  # [B,T,1]
        # ------------------------------------------------------------
        # m_exact correction
        #
        # If mexact_lambda_learnable=False:
        #     m_exact = d_exact / (d_exact + d_race + eps)
        #
        # If mexact_lambda_learnable=True:
        #     m_exact = d_exact / (d_exact + lambda * d_race + eps)
        #
        # We compute this in log-space for stability:
        #     log_den = logsumexp(log_d_exact, log_lambda + log_d_race, log_eps)
        #     m_exact = exp(log_d_exact - log_den)
        #
        # d_exact and d_race are detached, but lambda is NOT detached.
        # Therefore lambda can learn, while denominator estimation does not become
        # a heavy extra gradient path.
        # ------------------------------------------------------------
        log_d_exact_det = log_d_exact.detach().float()  # [B,T,1]
        log_d_race_det = torch.log(
            d_race.detach().float().clamp_min(self.mexact_eps)
        )  # [B,T,1]

        log_lambda = self.log_mexact_lambda.float()  # scalar, learnable if enabled
        log_eps = torch.full_like(log_d_exact_det, math.log(self.mexact_eps))

        log_den = torch.logsumexp(
            torch.stack(
                [
                    log_d_exact_det,
                    log_lambda + log_d_race_det,
                    log_eps,
                ],
                dim=0,
            ),
            dim=0,
        )  # [B,T,1]

        m_exact = torch.exp(log_d_exact_det - log_den).to(out_hyper.dtype)  # [B,T,1]

        lambda_value = self.log_mexact_lambda.detach().exp()
        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)

        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        self.last_gates = gates.detach()
        self.last_m_exact = m_exact.detach()
        self.last_d_exact_mean = torch.exp(log_d_exact.detach().clamp(max=20.0)).mean()
        self.last_d_race_mean = d_race.detach().mean()
        self.last_mexact_lambda = lambda_value
        g_hyper = gates[..., 0:1]
        g_race = gates[..., 1:2]

        out = g_hyper * m_exact * out_hyper + g_race * out_race
        return out


class HyperRaceMExactBlock(nn.Module):
    """
    Standard transformer block wrapper for hyper_race_mexact.
    """

    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperRaceMExactAttentionVision(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

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
def _gather_tokens_4d(x: torch.Tensor, idx: torch.Tensor):
    """
    x   : [B, H, T, D]
    idx : [B, H, S]
    out : [B, H, S, D]
    """
    return x.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))


def _masked_softmax_lse(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    """
    logits: [..., S]
    mask  : [..., S] bool

    Returns:
        probs: [..., S]
        lse  : [...]
    """
    neg = torch.finfo(logits.dtype).min
    logits_m = logits.masked_fill(~mask, neg)

    valid_any = mask.any(dim=dim, keepdim=True)
    safe_logits = torch.where(valid_any, logits_m, torch.zeros_like(logits_m))

    probs = torch.softmax(safe_logits, dim=dim) * mask.to(logits.dtype)
    probs = probs / (probs.sum(dim=dim, keepdim=True) + 1e-6)

    lse = torch.logsumexp(logits_m, dim=dim)
    lse = torch.where(
        valid_any.squeeze(dim),
        lse,
        torch.full_like(lse, float("-inf")),
    )
    return probs, lse


def _combine_attentions_from_lse(outputs, lses, eps: float = 1e-6):
    """
    Efficiently combine several normalized attention outputs using their log-denominators.

    outputs: list of tensors [B,H,T,D]
    lses   : list of tensors [B,H,T]

    Returns:
        out_combined  : [B,H,T,D]
        d_token_proxy : [B,T,1]
    """
    assert len(outputs) == len(lses)
    with torch.no_grad():
        lse_stack = torch.stack([x.nan_to_num(neginf=-1e30) for x in lses], dim=0)  # [R,B,H,T]
        m = lse_stack.max(dim=0).values                                             # [B,H,T]
        w = torch.exp(lse_stack - m.unsqueeze(0))                                   # [R,B,H,T]
        w = w.nan_to_num(0.0)
        w_sum = w.sum(dim=0).clamp_min(eps)                                         # [B,H,T]
        alpha = w / w_sum.unsqueeze(0)                                               # [R,B,H,T]

        lse_total = m + torch.log(w_sum)
        d_heads = torch.exp(lse_total.clamp(max=20.0)).clamp_min(eps)                # [B,H,T]
        d_token = d_heads.mean(dim=1).unsqueeze(-1)                                  # [B,T,1]

    out = 0.0
    for a, o in zip(alpha, outputs):
        out = out + a.detach().unsqueeze(-1) * o

    return out, d_token


class HyperLSHGlobalLocalExactAttentionVision(nn.Module):
    """
    Faster exact branch for hyper_race_gl_mexact.

    It avoids the previous huge irregular support tensor and avoids the large
    masked softmax/logsumexp that crashed.

    Exact branch is computed as three structured pieces:

      1) global exact attention:
            all queries attend to first G keys
            implemented with _run_exact_sdpa -> FlashAttention-backed SDPA

      2) local exact attention:
            all queries attend to [i-w, ..., i+w]
            implemented with a small explicit softmax over only 2w+1 keys

      3) LSH exact attention:
            HyperAttention-style sortLSH same-block attention
            implemented with _run_exact_sdpa on sorted blocks

    Then the three branch outputs are denominator-combined:

        O_exact = sum_r exp(lse_r) O_r / sum_r exp(lse_r)

    It returns:
        out_exact      : [B,T,d]
        d_exact_token  : [B,T,1]
    """

    def __init__(
        self,
        d_in,
        d_out,
        dropout,
        num_heads,
        num_bits=5,
        block_size=32,
        neighbor_blocks=0,
        global_tokens=8,
        local_window=16,
        q_chunk_size=32,
        qkv_bias=False,
        device="cpu",
        mexact_eps=1e-6,
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.block_size = block_size
        self.neighbor_blocks = neighbor_blocks
        self.global_tokens = global_tokens
        self.local_window = local_window
        self.q_chunk_size = q_chunk_size
        self.mexact_eps = mexact_eps

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.head_dim, device=device)

    @torch.no_grad()
    def _bucket_sort_metadata(self, Q, K):
        """
        Q,K: [B,H,T,D]

        Returns:
            q_sort_idx   : [B,H,T]
            k_sort_idx   : [B,H,T]
            q_orig2sort  : [B,H,T]
        """
        B, H, T, D = Q.shape

        q_ids = []
        k_ids = []

        # Your AngularLSHGray.hash expects [H,T,D], so keep batch loop.
        # This is cheap compared with attention.
        for b in range(B):
            q_ids.append(self.lsh.hash(Q[b]))  # [H,T]
            k_ids.append(self.lsh.hash(K[b]))  # [H,T]

        q_ids = torch.stack(q_ids, dim=0)      # [B,H,T]
        k_ids = torch.stack(k_ids, dim=0)      # [B,H,T]

        q_sort_idx = torch.argsort(q_ids, dim=2, stable=True)
        k_sort_idx = torch.argsort(k_ids, dim=2, stable=True)
        q_orig2sort = torch.argsort(q_sort_idx, dim=2, stable=True)

        return q_sort_idx, k_sort_idx, q_orig2sort

    def _global_attention(self, Q, K, V):
        """
        Global exact branch with FlashAttention-backed SDPA.

        Q,K,V: [B,H,T,D]

        Returns:
            out_g : [B,H,T,D]
            lse_g : [B,H,T]
        """
        B, H, T, D = Q.shape
        G = min(self.global_tokens, T)

        if G <= 0:
            out = torch.zeros_like(Q)
            lse = torch.full((B, H, T), float("-inf"), device=Q.device, dtype=Q.dtype)
            return out, lse

        Kg = K[:, :, :G, :]
        Vg = V[:, :, :G, :]

        # FlashAttention-backed SDPA for output.
        out = _run_exact_sdpa(Q, Kg, Vg)  # [B,H,T,D]

        # Small denominator computation over G keys.
        # Detached because m_exact is a scaling diagnostic/control signal.
        with torch.no_grad():
            scale = 1.0 / math.sqrt(D)
            logits = torch.einsum("bhtd,bhgd->bhtg", Q, Kg) * scale
            lse = torch.logsumexp(logits.float(), dim=-1).to(Q.dtype)  # [B,H,T]

        return out, lse

    def _local_attention(self, Q, K, V):
        """
        Local exact branch over [i-w, ..., i+w].

        This uses an explicit softmax, but only over 2w+1 keys.
        For w=16, that is 33 keys, so this is lightweight.

        To avoid double-counting global keys too badly, local branch masks out
        keys with index < global_tokens, because global branch already covers them.
        """
        B, H, T, D = Q.shape
        device = Q.device
        dtype = Q.dtype
        scale = 1.0 / math.sqrt(D)

        w = self.local_window
        G = min(self.global_tokens, T)

        offsets = torch.arange(-w, w + 1, device=device)  # [W]
        W = offsets.numel()

        out = torch.zeros_like(Q)
        lse_all = torch.full((B, H, T), float("-inf"), device=device, dtype=dtype)

        for qs in range(0, T, self.q_chunk_size):
            qe = min(qs + self.q_chunk_size, T)
            q_len = qe - qs

            q_idx = torch.arange(qs, qe, device=device)        # [q]
            local_idx = q_idx[:, None] + offsets[None, :]      # [q,W]

            valid = (local_idx >= 0) & (local_idx < T)

            # Remove global tokens from the local branch to avoid duplicate mass
            # between global and local branches.
            if G > 0:
                valid = valid & (local_idx >= G)

            local_idx = local_idx.clamp(0, T - 1)

            q = Q[:, :, qs:qe, :]                              # [B,H,q,D]
            K_sel = K[:, :, local_idx, :]                      # [B,H,q,W,D]
            V_sel = V[:, :, local_idx, :]                      # [B,H,q,W,D]

            logits = torch.einsum("bhqd,bhqwd->bhqw", q, K_sel) * scale
            mask = valid.view(1, 1, q_len, W).expand(B, H, -1, -1)

            neg = torch.finfo(logits.dtype).min
            logits_m = logits.masked_fill(~mask, neg)
            valid_any = mask.any(dim=-1, keepdim=True)

            safe_logits = torch.where(valid_any, logits_m, torch.zeros_like(logits_m))
            probs = torch.softmax(safe_logits, dim=-1) * mask.to(dtype)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)

            out[:, :, qs:qe, :] = torch.einsum("bhqw,bhqwd->bhqd", probs, V_sel)

            with torch.no_grad():
                lse = torch.logsumexp(logits_m.float(), dim=-1).to(dtype)
                lse = torch.where(
                    valid_any.squeeze(-1),
                    lse,
                    torch.full_like(lse, float("-inf")),
                )
                lse_all[:, :, qs:qe] = lse

        return out, lse_all

    def _lsh_block_attention(self, Q, K, V):
        """
        HyperAttention-style sorted same-block exact branch.

        Uses FlashAttention-backed SDPA for block outputs.
        Computes LSE separately in no_grad for m_exact.
        """
        B, H, T, D = Q.shape
        device = Q.device
        dtype = Q.dtype
        scale = 1.0 / math.sqrt(D)
        bsz = self.block_size

        q_sort_idx, k_sort_idx, q_orig2sort = self._bucket_sort_metadata(Q, K)

        Qs = _gather_tokens_4d(Q, q_sort_idx)  # [B,H,T,D]
        Ks = _gather_tokens_4d(K, k_sort_idx)
        Vs = _gather_tokens_4d(V, k_sort_idx)

        out_sorted = torch.zeros_like(Qs)
        lse_sorted = torch.empty(B, H, T, device=device, dtype=dtype)

        num_full_blocks = T // bsz
        rem = T % bsz

        # --------------------------------------------------
        # Full blocks: one Flash/SDPA call for all blocks.
        # --------------------------------------------------
        if num_full_blocks > 0:
            T_full = num_full_blocks * bsz

            Q_full = Qs[:, :, :T_full, :].reshape(B, H, num_full_blocks, bsz, D)
            K_full = Ks[:, :, :T_full, :].reshape(B, H, num_full_blocks, bsz, D)
            V_full = Vs[:, :, :T_full, :].reshape(B, H, num_full_blocks, bsz, D)

            Q_flat = Q_full.reshape(B * H * num_full_blocks, 1, bsz, D)
            K_flat = K_full.reshape(B * H * num_full_blocks, 1, bsz, D)
            V_flat = V_full.reshape(B * H * num_full_blocks, 1, bsz, D)

            O_flat = _run_exact_sdpa(Q_flat, K_flat, V_flat)  # [B*H*NB,1,bsz,D]
            O_full = O_flat.reshape(B, H, num_full_blocks, bsz, D).reshape(B, H, T_full, D)
            out_sorted[:, :, :T_full, :] = O_full

            # LSE for m_exact only. No gradient, chunked over blocks to avoid memory spikes.
            with torch.no_grad():
                lse_chunks = []
                block_chunk = 64
                for bs in range(0, num_full_blocks, block_chunk):
                    be = min(bs + block_chunk, num_full_blocks)
                    logits = torch.einsum(
                        "bhnqd,bhnkd->bhnqk",
                        Q_full[:, :, bs:be, :, :],
                        K_full[:, :, bs:be, :, :],
                    ) * scale
                    lse_blk = torch.logsumexp(logits.float(), dim=-1).to(dtype)  # [B,H,chunk,bsz]
                    lse_chunks.append(lse_blk)
                lse_full = torch.cat(lse_chunks, dim=2).reshape(B, H, T_full)
                lse_sorted[:, :, :T_full] = lse_full

        # --------------------------------------------------
        # Remainder block.
        # --------------------------------------------------
        if rem > 0:
            q_rem = Qs[:, :, num_full_blocks * bsz:, :]
            k_rem = Ks[:, :, num_full_blocks * bsz:, :]
            v_rem = Vs[:, :, num_full_blocks * bsz:, :]

            o_rem = _run_exact_sdpa(q_rem, k_rem, v_rem)
            out_sorted[:, :, num_full_blocks * bsz:, :] = o_rem

            with torch.no_grad():
                logits = torch.einsum("bhrd,bhsd->bhrs", q_rem, k_rem) * scale
                lse_rem = torch.logsumexp(logits.float(), dim=-1).to(dtype)
                lse_sorted[:, :, num_full_blocks * bsz:] = lse_rem

        # Unsort queries back to original order.
        out = _gather_tokens_4d(out_sorted, q_orig2sort)
        lse = lse_sorted.gather(2, q_orig2sort)

        return out, lse

    def forward(self, x):
        """
        Returns:
            out_exact     : [B,T,d]
            d_exact_token : [B,T,1]
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2).contiguous()

        # 1) Global: Flash/SDPA
        out_g, lse_g = self._global_attention(Q, K, V)

        # 2) Local: tiny explicit softmax over 2w+1 keys
        out_l, lse_l = self._local_attention(Q, K, V)

        # 3) LSH: HyperAttention-style sorted block + Flash/SDPA
        out_h, lse_h = self._lsh_block_attention(Q, K, V)

        # Combine the three exact components by denominator mass.
        out_heads, d_exact_token = _combine_attentions_from_lse(
            outputs=[out_g, out_l, out_h],
            lses=[lse_g, lse_l, lse_h],
            eps=self.mexact_eps,
        )

        out = out_heads.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.dropout(out)
        out = self.out_proj(out)

        return out, d_exact_token

class RACEAttentionWithDenom(RACEAttention):
    """
    Same RACE branch behavior as your normal RACEAttention, but also returns
    a denominator proxy:

        d_race(i) = pQ(i) dot A_full

    No used-key elimination.
    """

    def _ace_with_denom(self, Khf, Vhf, Qhf, eps: float = 1e-6):
        ace = self.ace

        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k

        S = ace.L * ace.R
        scale = ace.logit_temp.exp().clamp(1e-2, 20.0)

        if ace.share_planes:
            N = M * B * H

            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            projK = Kh2 @ ace.planes_T
            projQ = Qh2 @ ace.planes_T

        else:
            BH = B * H

            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            projK = torch.einsum("mbtd,mds->mbts", Kh2, ace.planes_T)
            projQ = torch.einsum("mbtd,mds->mbts", Qh2, ace.planes_T)

            projK = projK.contiguous().view(M * BH, T, ace.L * ace.K)
            projQ = projQ.contiguous().view(M * BH, T, ace.L * ace.K)
            V2    = V2.contiguous().view(M * BH, T, dk)
            N     = M * BH

        projK = projK.view(N, T, ace.L, ace.K)
        projQ = projQ.view(N, T, ace.L, ace.K)

        logitsK = (projK.tanh().div(scale) @ ace.protos_T)  # [N,T,L,R]
        logitsQ = (projQ.tanh().div(scale) @ ace.protos_T)

        probsK = F.softmax(logitsK, dim=-1)
        probsQ = F.softmax(logitsQ, dim=-1)

        probsK_S = probsK.contiguous().view(N, T, S)  # [N,T,S]
        probsQ_S = probsQ.contiguous().view(N, T, S)  # [N,T,S]

        total_num = probsK_S.transpose(1, 2).bmm(V2)  # [N,S,dk]
        total_den = probsK_S.sum(dim=1)               # [N,S]

        E = total_num / (total_den.unsqueeze(-1) + eps)  # [N,S,dk]
        out2 = probsQ_S.bmm(E)                           # [N,T,dk]

        d2 = torch.einsum("nts,ns->nt", probsQ_S, total_den).clamp_min(eps)  # [N,T]

        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4).contiguous()  # [M,B,T,H,dk]
        den = d2.view(M, B, H, T).permute(0, 1, 3, 2).contiguous()           # [M,B,T,H]

        return out, den

    def forward(self, x):
        """
        Returns:
            out_race      : [B,T,d]
            d_race_token  : [B,T,1]
        """
        B, T, _ = x.shape
        H, dk, M = self.H, self.d_k, self.M

        Q = self.q_proj(x).view(B, T, H, dk)
        K = self.k_proj(x).view(B, T, H, dk)
        V = self.v_proj(x).view(B, T, H, dk)

        def pack(Z):
            return Z.unsqueeze(0).expand(M, -1, -1, -1, -1)

        out_m, den_m = self._ace_with_denom(pack(K), pack(V), pack(Q))  # [M,B,T,H,dk], [M,B,T,H]

        out_heads = out_m.mean(dim=0)  # [B,T,H,dk]

        # Correct merge: [B,T,H,dk] -> [B,T,H*dk]
        out = out_heads.contiguous().view(B, T, H * dk)

        d_race_heads = den_m.mean(dim=0)                 # [B,T,H]
        d_race_token = d_race_heads.mean(dim=-1, keepdim=True).clamp_min(1e-6)  # [B,T,1]

        out = self.drop(self.out(out))
        return out, d_race_token


class HyperRaceGlobalLocalMExactAttentionVision(nn.Module):
    """
    New lightweight version based on hyper_race:

      - exact branch:
          Hyper-LSH same-block keys
          + first G global keys
          + local sliding window keys

      - race branch:
          unchanged full RACE branch
          no elimination

      - final:
          out = g_hyper * m_exact * out_exact + g_race * out_race

        where:
          m_exact = d_exact / (d_exact + d_race + eps)
    """

    def __init__(self, cfg, device="cpu"):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        self.mexact_eps = cfg.get("mexact_eps", 1e-6)

        self.hyper = HyperLSHGlobalLocalExactAttentionVision(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            global_tokens=cfg.get("hyper_global_tokens", 8),
            local_window=cfg.get("hyper_local_window", 16),
            q_chunk_size=cfg.get("hyper_exact_q_chunk_size", 32),
            qkv_bias=qkv_bias,
            device=device,
            mexact_eps=self.mexact_eps,
        )

        self.race = RACEAttentionWithDenom(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            L=cfg["L"],
            K=cfg["K"],
            N_M=cfg["M"],
            qkv_bias=qkv_bias,
            device=device,
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        self.normalize_gates = cfg.get("gate_normalize", False)

        self.last_gates = None
        self.last_m_exact = None
        self.last_d_exact_mean = None
        self.last_d_race_mean = None

    def forward(self, x):
        out_hyper, d_exact = self.hyper(x)  # [B,T,d], [B,T,1]
        out_race, d_race = self.race(x)     # [B,T,d], [B,T,1]

        m_exact = d_exact / (d_exact + d_race + self.mexact_eps)  # [B,T,1]

        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)

        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        self.last_gates = gates.detach()
        self.last_m_exact = m_exact.detach()
        self.last_d_exact_mean = d_exact.detach().mean()
        self.last_d_race_mean = d_race.detach().mean()

        g_hyper = gates[..., 0:1]
        g_race = gates[..., 1:2]

        out = g_hyper * m_exact * out_hyper + g_race * out_race
        return out


class HyperRaceGlobalLocalMExactBlock(nn.Module):
    """
    Standard transformer block wrapper for hyper_race_gl_mexact.
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperRaceGlobalLocalMExactAttentionVision(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

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
                        nn.Linear(cfg["embed_dim"], 4*cfg["embed_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["embed_dim"], cfg["embed_dim"])
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
                        nn.Linear(cfg["embed_dim"],4*cfg["embed_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["embed_dim"],cfg["embed_dim"])
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
                        nn.Linear(cfg["embed_dim"], 4*cfg["embed_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["embed_dim"], cfg["embed_dim"])
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
                        nn.Linear(cfg["embed_dim"], 4*cfg['embed_dim']),
                        nn.GELU(),
                        nn.Linear(4*cfg['embed_dim'], cfg["embed_dim"]),
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
# ==================================================
# NEW ATTENTION TYPE: hyper_plus_race
# Paste this block RIGHT BEFORE the final VisionTransformer class
# ==================================================

class BucketExcludedRACEAttention(RACEAttention):
    def __init__(
        self,
        d_in,
        d_out,
        dropout,
        num_heads,
        L,
        K,
        N_M,
        hard_num_bits,
        q_chunk_size=256,   # NEW: query chunk size for memory-safe correction
        qkv_bias=False,
        device="cpu",
    ):
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            num_heads=num_heads,
            L=L,
            K=K,
            N_M=N_M,
            qkv_bias=qkv_bias,
            device=device,
        )
        self.hard_num_bits = hard_num_bits
        self.hard_R = 1 << hard_num_bits
        self.q_chunk_size = q_chunk_size

    def _excluded_ace(
        self,
        Khf,                    # [M, B, T, H, dk]
        Vhf,                    # [M, B, T, H, dk]
        Qhf,                    # [M, B, T, H, dk]
        q_hard_bucket_ids,      # [B, H, T]
        k_hard_bucket_ids,      # [B, H, T]
        eps: float = 1e-6,
    ):
        """
        Memory-safe version of the modified RACE path.

        Same math as before, but avoids materializing:
            [N, T, S, dk]
        """
        ace = self.ace

        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k

        S = ace.L * ace.R
        scale = ace.logit_temp.exp().clamp(1e-2, 20.0)

        # ------------------------------------------------------------
        # 1) Compute the standard soft RACE probabilities
        # ------------------------------------------------------------
        if ace.share_planes:
            N = M * B * H

            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            projK = Kh2 @ ace.planes_T
            projQ = Qh2 @ ace.planes_T
        else:
            BH = B * H

            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            projK = torch.einsum("mbtd,mds->mbts", Kh2, ace.planes_T)
            projQ = torch.einsum("mbtd,mds->mbts", Qh2, ace.planes_T)

            projK = projK.contiguous().view(M * BH, T, ace.L * ace.K)
            projQ = projQ.contiguous().view(M * BH, T, ace.L * ace.K)
            V2    = V2.contiguous().view(M * BH, T, dk)
            N     = M * BH

        projK = projK.view(N, T, ace.L, ace.K)
        projQ = projQ.view(N, T, ace.L, ace.K)

        logitsK = (projK.tanh().div(scale) @ ace.protos_T)   # [N,T,L,R]
        logitsQ = (projQ.tanh().div(scale) @ ace.protos_T)   # [N,T,L,R]

        probsK = F.softmax(logitsK, dim=-1)                  # [N,T,L,R]
        probsQ = F.softmax(logitsQ, dim=-1)                  # [N,T,L,R]

        probsK_S = probsK.contiguous().view(N, T, S)         # [N,T,S]
        probsQ_S = probsQ.contiguous().view(N, T, S)         # [N,T,S]

        # ------------------------------------------------------------
        # 2) Standard full RACE summaries
        # ------------------------------------------------------------
        total_num = probsK_S.transpose(1, 2).bmm(V2)         # [N,S,dk]
        total_den = probsK_S.sum(dim=1)                      # [N,S]
        denom = total_den.unsqueeze(-1) + eps               # [N,S,1]

        # ------------------------------------------------------------
        # 3) Flatten shared hard bucket ids to [N,T]
        # ------------------------------------------------------------
        BH = B * H

        q_ids = (
            q_hard_bucket_ids.contiguous()
            .view(BH, T)
            .unsqueeze(0)
            .expand(M, -1, -1)
            .contiguous()
            .view(N, T)
        )

        k_ids = (
            k_hard_bucket_ids.contiguous()
            .view(BH, T)
            .unsqueeze(0)
            .expand(M, -1, -1)
            .contiguous()
            .view(N, T)
        )

        # ------------------------------------------------------------
        # 4) Base RACE output (same size as your original RACE output)
        # ------------------------------------------------------------
        E_all = total_num / denom                            # [N,S,dk]
        out2 = probsQ_S.bmm(E_all)                           # [N,T,dk]

        # ------------------------------------------------------------
        # 5) Subtract same-hard-bucket contribution
        #    BUT do it bucket-by-bucket and query-chunk-by-query-chunk
        #    so we never create [N,T,S,dk]
        # ------------------------------------------------------------
        q_chunk = self.q_chunk_size

        for b in range(self.hard_R):
            qmask_b = (q_ids == b)                           # [N,T] bool
            if not bool(qmask_b.any().item()):
                continue

            # Keys in hard bucket b
            kmask_b = (k_ids == b).to(probsK_S.dtype)       # [N,T]

            # Same-bucket numerator only:
            #   same_num_b[n,s,d] = sum_t probsK_S[n,t,s] * 1{k_ids[n,t]=b} * V2[n,t,d]
            same_num_b = torch.einsum(
                "nts,nt,ntd->nsd",
                probsK_S,
                kmask_b,
                V2,
            )                                               # [N,S,dk]

            # Divide by FULL denominator (same-bucket keys stay in denominator)
            E_same_b = same_num_b / denom                   # [N,S,dk]

            # Only compute correction in query chunks
            for qs in range(0, T, q_chunk):
                qe = min(qs + q_chunk, T)
                qmask_chunk = qmask_b[:, qs:qe]             # [N,q]
                if not bool(qmask_chunk.any().item()):
                    continue

                remove_chunk = probsQ_S[:, qs:qe, :].bmm(E_same_b)   # [N,q,dk]

                out2[:, qs:qe, :] = out2[:, qs:qe, :] - (
                    qmask_chunk.unsqueeze(-1).to(out2.dtype) * remove_chunk
                )

        # ------------------------------------------------------------
        # 6) Restore [M,B,T,H,dk]
        # ------------------------------------------------------------
        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4).contiguous()
        return out

    def forward(self, x, q_hard_bucket_ids, k_hard_bucket_ids):
        """
        x:                [B,T,d]
        q_hard_bucket_ids [B,H,T]
        k_hard_bucket_ids [B,H,T]
        """
        B, T, _ = x.shape
        H, dk, M = self.H, self.d_k, self.M

        # Standard RACE projections
        Q = self.q_proj(x).view(B, T, H, dk)   # [B,T,H,dk]
        K = self.k_proj(x).view(B, T, H, dk)   # [B,T,H,dk]
        V = self.v_proj(x).view(B, T, H, dk)   # [B,T,H,dk]

        # Pack across M ensembles, same as your current RACEAttention
        def pack(Z):
            return Z.unsqueeze(0).expand(M, -1, -1, -1, -1)   # [M,B,T,H,dk]

        out_m = self._excluded_ace(
            pack(K),
            pack(V),
            pack(Q),
            q_hard_bucket_ids,
            k_hard_bucket_ids,
        )                                                     # [M,B,T,H,dk]

        # Average ensembles
        out = out_m.mean(dim=0)                               # [B,T,H,dk]

        # Merge heads CORRECTLY into [B,T,d]
        out = out.contiguous().view(B, T, H * dk)

        return self.drop(self.out(out))


class FixedHyperPlusRaceAttentionVision(nn.Module):
    """
    New hybrid attention:
      1) Hyper-LSH exact sparse branch (your current hyper_lsh implementation)
      2) RACE branch whose numerator excludes keys in the same HARD Hyper-LSH bucket
      3) Two global learnable scalars a and b, both initialized to 1

    Final output:
      out = a * out_race + b * out_hyper
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]

        # ------------------------------------------------------------
        # Hyper-LSH exact sparse branch: kept exactly in your current style
        # ------------------------------------------------------------
        self.hyper = HyperLSHExactAttentionVision(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
        )

        # IMPORTANT:
        # This is the COMMON hard-bucket mapper shared by both directions.
        # We reuse the exact same LSH object already inside the hyper branch.
        self.shared_lsh = self.hyper.lsh

        # ------------------------------------------------------------
        # Modified RACE branch:
        # same hard Hyper-LSH buckets are used only to decide which keys
        # should be removed from the RACE numerator for each query.
        # ------------------------------------------------------------
        self.race = BucketExcludedRACEAttention(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            L=cfg["L"],
            K=cfg["K"],
            N_M=cfg["M"],
            hard_num_bits=cfg.get("hyper_num_bits", 5),
            q_chunk_size=cfg.get("hyper_plus_race_q_chunk_size", 256),
            qkv_bias=qkv_bias,
            device=device,
        )

        # Global learnable scalars, initialized to 1 exactly as requested
        self.a = nn.Parameter(torch.tensor(1.0))   # multiplies O_race
        self.b = nn.Parameter(torch.tensor(1.0))   # multiplies O_hyper
    @torch.no_grad()
    def _shared_hard_bucket_ids(self, x):
        """
        Compute HARD Hyper-LSH bucket ids using the HYPER branch's own
        Q/K projections and the shared_lsh object.

        Returned shapes:
            q_ids: [B,H,T]
            k_ids: [B,H,T]
        """
        B, T, _ = x.shape
        H = self.hyper.num_heads
        D = self.hyper.head_dim

        # Use the Hyper-LSH branch projections to define the shared partition
        Qh = self.hyper.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        Kh = self.hyper.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()    # [B,H,T,D]

        q_ids = []
        k_ids = []

        # Your active AngularLSHGray.hash expects [H,T,D], so loop over batch items
        for b in range(B):
            q_ids.append(self.shared_lsh.hash(Qh[b]))   # [H,T]
            k_ids.append(self.shared_lsh.hash(Kh[b]))   # [H,T]

        q_ids = torch.stack(q_ids, dim=0)               # [B,H,T]
        k_ids = torch.stack(k_ids, dim=0)               # [B,H,T]
        return q_ids, k_ids

    def forward(self, x):
        # 1) Shared hard Hyper-LSH bucket assignment
        q_hard_bucket_ids, k_hard_bucket_ids = self._shared_hard_bucket_ids(x)

        # 2) Hyper-LSH exact sparse output (unchanged branch)
        out_hyper = self.hyper(x)                       # [B,T,d]

        # 3) Modified RACE output using the SAME hard bucket partition
        out_race = self.race(
            x,
            q_hard_bucket_ids=q_hard_bucket_ids,
            k_hard_bucket_ids=k_hard_bucket_ids,
        )                                              # [B,T,d]

        # 4) Two global scalar gates only
        out = self.a * out_race + self.b * out_hyper
        return out

class HyperPlusRaceAttentionVision(nn.Module):
    """
    Hyper+RACE with input-dependent gates, in the SAME style as hyper_race:
      - Hyper-LSH exact sparse branch
      - modified RACE branch (bucket-excluded RACE)
      - tiny 2-layer MLP gate -> 2 scalar gates per token
      - pure sigmoid gating
      - weighted sum of the two branch outputs

    Gate convention:
      gates[..., 0] -> hyper branch
      gates[..., 1] -> modified race branch
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        # Hyper-LSH branch
        self.hyper = HyperLSHExactAttentionVision(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
        )

        # Shared hard-bucket mapper
        self.shared_lsh = self.hyper.lsh

        # Modified RACE branch
        self.race = BucketExcludedRACEAttention(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            L=cfg["L"],
            K=cfg["K"],
            N_M=cfg["M"],
            hard_num_bits=cfg.get("hyper_num_bits", 5),
            q_chunk_size=cfg.get("hyper_plus_race_q_chunk_size", 256),
            qkv_bias=qkv_bias,
            device=device,
        )

        # SAME gate structure as hyper_race
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        # for WandB logging, same pattern as hyper_race
        self.last_gates = None

    @torch.no_grad()
    def _shared_hard_bucket_ids(self, x):
        """
        Compute shared HARD Hyper-LSH bucket ids from the hyper branch projections.
        Returns:
            q_ids: [B,H,T]
            k_ids: [B,H,T]
        """
        B, T, _ = x.shape
        H = self.hyper.num_heads
        D = self.hyper.head_dim

        Qh = self.hyper.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        Kh = self.hyper.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()    # [B,H,T,D]

        q_ids = []
        k_ids = []
        for b in range(B):
            q_ids.append(self.shared_lsh.hash(Qh[b]))   # [H,T]
            k_ids.append(self.shared_lsh.hash(Kh[b]))   # [H,T]

        q_ids = torch.stack(q_ids, dim=0)               # [B,H,T]
        k_ids = torch.stack(k_ids, dim=0)               # [B,H,T]
        return q_ids, k_ids

    def forward(self, x):
        # 1) Shared hard bucket ids
        q_hard_bucket_ids, k_hard_bucket_ids = self._shared_hard_bucket_ids(x)

        # 2) Branch outputs
        out_hyper = self.hyper(x)   # [B,T,d]
        out_race  = self.race(
            x,
            q_hard_bucket_ids=q_hard_bucket_ids,
            k_hard_bucket_ids=k_hard_bucket_ids,
        )                           # [B,T,d]

        # 3) SAME gate style as hyper_race: pure sigmoid
        gate_logits = self.gate_mlp(x)      # [B,T,2]
        gates = torch.sigmoid(gate_logits)  # [B,T,2]

        # Save for logging
        self.last_gates = gates.detach()

        # 4) Split gates
        g_hyper = gates[..., 0:1]   # [B,T,1]
        g_race  = gates[..., 1:2]   # [B,T,1]

        # 5) Weighted sum
        out = g_hyper * out_hyper + g_race * out_race
        return out
class HyperPlusRaceBlock(nn.Module):
    """
    Standard transformer block wrapper for hyper_plus_race.
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperPlusRaceAttentionVision(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

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

# ==================================================
# NEW ATTENTION TYPE: hyper_plus_race_nosample
# Paste this block RIGHT AFTER HyperPlusRaceBlock
# and RIGHT BEFORE VisionTransformer
# ==================================================

def _gather_scores_3d(x: torch.Tensor, idx: torch.Tensor):
    """
    Gather along token dimension for score tensors.

    x   : [H, T, S]
    idx : [H, T]
    out : [H, T, S]
    """
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def race_avg_bucket_probs_from_x(race_attn: "RACEAttention", x: torch.Tensor):
    """
    Compute RACE soft bucket probabilities from the CURRENT RACE branch.

    Returns
    -------
    probsQ_S_avg : [B, H, T, S]
    probsK_S_avg : [B, H, T, S]

    where
        S = L * (2^K)
    and the probabilities are averaged across M ensembles.

    IMPORTANT
    ---------
    This uses the same q_proj / k_proj / ACE parameters as the RACE branch,
    so the estimator branch and the RACE branch are aligned.
    """
    B, T, _ = x.shape
    H, dk, M = race_attn.H, race_attn.d_k, race_attn.M
    ace = race_attn.ace
    S = ace.L * ace.R

    scale = ace.logit_temp.exp().clamp(1e-2, 20.0)

    Q = race_attn.q_proj(x).view(B, T, H, dk)   # [B,T,H,dk]
    K = race_attn.k_proj(x).view(B, T, H, dk)   # [B,T,H,dk]

    def pack(Z):
        return Z.unsqueeze(0).expand(M, -1, -1, -1, -1)   # [M,B,T,H,dk]

    Qhf = pack(Q)
    Khf = pack(K)

    if ace.share_planes:
        N = M * B * H

        Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)  # [N,T,dk]
        Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

        projQ = (Qh2 @ ace.planes_T).view(M, B, H, T, ace.L, ace.K)
        projK = (Kh2 @ ace.planes_T).view(M, B, H, T, ace.L, ace.K)
    else:
        BH = B * H

        Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)  # [M,BH,T,dk]
        Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

        projQ = torch.einsum("mbtd,mds->mbts", Qh2, ace.planes_T)
        projK = torch.einsum("mbtd,mds->mbts", Kh2, ace.planes_T)

        projQ = projQ.contiguous().view(M, B, H, T, ace.L, ace.K)
        projK = projK.contiguous().view(M, B, H, T, ace.L, ace.K)

    logitsQ = (projQ.tanh().div(scale) @ ace.protos_T)      # [M,B,H,T,L,R]
    logitsK = (projK.tanh().div(scale) @ ace.protos_T)      # [M,B,H,T,L,R]

    probsQ = F.softmax(logitsQ, dim=-1)                     # [M,B,H,T,L,R]
    probsK = F.softmax(logitsK, dim=-1)

    probsQ_S = probsQ.contiguous().view(M, B, H, T, S)     # [M,B,H,T,S]
    probsK_S = probsK.contiguous().view(M, B, H, T, S)

    # Average across ensembles to get one proxy distribution
    probsQ_S_avg = probsQ_S.mean(dim=0)                     # [B,H,T,S]
    probsK_S_avg = probsK_S.mean(dim=0)                     # [B,H,T,S]

    return probsQ_S_avg, probsK_S_avg


class HyperLSHExactNoSampleEstimateAttentionVision(nn.Module):
    """
    Hyper-LSH exact sparse branch with a deterministic RACE-based denominator estimate.

    For each query i:
      1) exact sparse support H_i from Hyper-LSH
      2) exact numerator:
            n1(i) = sum_{j in H_i} y_ij v_j
      3) exact sparse denom part:
            d1(i) = sum_{j in H_i} y_ij
      4) residual RACE proxy mass:
            X_i = sum_{j in R_i} x_ij
      5) scale the proxy using the query-wise support max shift m_i:
            X_i' = exp(-m_i) * X_i
      6) denominator estimate:
            D_hat(i) = d1(i) + c_h * X_i'
      7) output:
            out_i = n1(i) / D_hat(i)

    where:
      y_ij = exp(q_i^T k_j / sqrt(d))
      x_ij = < race_probs_Q(i), race_probs_K(j) >

    IMPORTANT
    ---------
    - No sampling.
    - Much lighter than HyperAttention Algorithm 2.
    - Uses the RACE proxy only as a deterministic residual denominator estimate.
    """
    def __init__(
        self,
        d_in,
        d_out,
        dropout,
        num_heads,
        num_bits=5,
        block_size=32,
        min_seq_len=256,
        neighbor_blocks=0,
        qkv_bias=False,
        device="cpu",
        c_init=1.0,
        estimator_eps=1e-6,
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.block_size = block_size
        self.min_seq_len = min_seq_len
        self.neighbor_blocks = neighbor_blocks
        self.estimator_eps = estimator_eps
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.head_dim, device=device)

        # positive calibration c_h = exp(log_c_h), one scalar per head
        init_log_c = math.log(max(c_init, 1e-6))
        self.log_c = nn.Parameter(torch.full((num_heads,), init_log_c))

    def _full_sdpa_fallback(self, Qh, Kh, Vh):
        return _run_exact_sdpa(
            Qh.unsqueeze(0),   # [1,H,T,D]
            Kh.unsqueeze(0),
            Vh.unsqueeze(0),
        )[0]

    def forward(self, x, probsQ_race: torch.Tensor, probsK_race: torch.Tensor):
        """
        x          : [B,T,d]
        probsQ_race: [B,H,T,S]
        probsK_race: [B,H,T,S]
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2).contiguous()

        out = torch.zeros_like(Q)                                           # [B,H,T,D]

        for b in range(B):
            Qh = Q[b]   # [H,T,D]
            Kh = K[b]
            Vh = V[b]

            # Short-sequence fallback: exact full attention
            if T < self.min_seq_len:
                out[b] = self._full_sdpa_fallback(Qh, Kh, Vh)
                continue

            # --------------------------------------------------------
            # 1) Hyper-LSH sorting / support
            # --------------------------------------------------------
            q_bucket_ids = self.lsh.hash(Qh)     # [H,T]
            k_bucket_ids = self.lsh.hash(Kh)     # [H,T]

            q_sort_idx = torch.argsort(q_bucket_ids, dim=1, stable=True)
            k_sort_idx = torch.argsort(k_bucket_ids, dim=1, stable=True)
            q_sort_inv = torch.argsort(q_sort_idx, dim=1, stable=True)

            Qs = _gather_tokens_3d(Qh, q_sort_idx)                  # [H,T,D]
            Ks = _gather_tokens_3d(Kh, k_sort_idx)
            Vs = _gather_tokens_3d(Vh, k_sort_idx)

            # Sort RACE probabilities into the SAME coordinates
            q_probs_sorted = _gather_scores_3d(probsQ_race[b], q_sort_idx)   # [H,T,S]
            k_probs_sorted = _gather_scores_3d(probsK_race[b], k_sort_idx)   # [H,T,S]

            out_sorted = torch.zeros_like(Qs)                                # [H,T,D]
            num_blocks = math.ceil(T / self.block_size)

            for h in range(H):
                c_h = self.log_c[h].exp()                                    # positive scalar
                A_full = k_probs_sorted[h].sum(dim=0)                        # [S]

                for bi in range(num_blocks):
                    q0 = bi * self.block_size
                    q1 = min((bi + 1) * self.block_size, T)

                    left  = max(0, bi - self.neighbor_blocks)
                    right = min(num_blocks - 1, bi + self.neighbor_blocks)

                    k0 = left * self.block_size
                    k1 = min((right + 1) * self.block_size, T)

                    q_blk = Qs[h, q0:q1, :]                                   # [q,D]
                    k_blk = Ks[h, k0:k1, :]                                   # [k,D]
                    v_blk = Vs[h, k0:k1, :]                                   # [k,D]
                    q_prob_blk = q_probs_sorted[h, q0:q1, :]                  # [q,S]

                    # --------------------------------------------------
                    # Exact sparse numerator / exact sparse denominator
                    # --------------------------------------------------
                    logits_support = torch.einsum("qd,kd->qk", q_blk, k_blk) * self.scale   # [q,k]

                    # query-wise stability shift from exact support
                    row_shift = logits_support.max(dim=-1, keepdim=True).values              # [q,1]

                    y_support = torch.exp(logits_support - row_shift)                        # [q,k]

                    n1 = y_support @ v_blk                                                   # [q,D]
                    d1 = y_support.sum(dim=-1)                                               # [q]

                    # --------------------------------------------------
                    # Deterministic residual proxy from RACE
                    #   X_i = sum_{j in R_i} x_ij
                    # Efficiently:
                    #   X_i = q_prob_i · (A_full - A_H)
                    # --------------------------------------------------
                    A_H = k_probs_sorted[h, k0:k1, :].sum(dim=0)                             # [S]
                    A_R = A_full - A_H                                                       # [S]

                    X = q_prob_blk @ A_R                                                     # [q]

                    # Put proxy on the SAME shifted scale as y_support
                    X_shifted = torch.exp(-row_shift.squeeze(-1)) * X                        # [q]

                    denom = d1 + c_h * X_shifted                                             # [q]
                    denom = denom.clamp_min(self.estimator_eps)

                    out_sorted[h, q0:q1, :] = n1 / denom.unsqueeze(-1)

            # --------------------------------------------------------
            # 2) Unsort back to original query order
            # --------------------------------------------------------
            out[b] = out_sorted.gather(
                1, q_sort_inv.unsqueeze(-1).expand(-1, -1, D)
            )

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)             # [B,T,d]
        out = self.dropout(out)
        return self.out_proj(out)


class HyperPlusRaceNoSampleAttentionVision(nn.Module):
    """
    hyper_plus_race_nosample:
      - Branch 1: exact Hyper-LSH sparse numerator + deterministic RACE denominator estimate
      - Branch 2: same modified RACE branch as current hyper_plus_race
      - Final merge: same input-dependent sigmoid gates as hyper_plus_race
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        # Branch 1: exact sparse + no-sampling denominator estimate
        self.hyper = HyperLSHExactNoSampleEstimateAttentionVision(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
            c_init=cfg.get("hyper_plus_race_nosample_c_init", 1.0),
            estimator_eps=cfg.get("hyper_plus_race_nosample_eps", 1e-6),
        )

        # Shared hard-bucket mapper for the modified RACE branch
        self.shared_lsh = self.hyper.lsh

        # Branch 2: keep your current modified RACE branch
        self.race = BucketExcludedRACEAttention(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            L=cfg["L"],
            K=cfg["K"],
            N_M=cfg["M"],
            hard_num_bits=cfg.get("hyper_num_bits", 5),
            q_chunk_size=cfg.get("hyper_plus_race_q_chunk_size", 256),
            qkv_bias=qkv_bias,
            device=device,
        )

        # Same gate style as hyper_plus_race
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        self.last_gates = None

    @torch.no_grad()
    def _shared_hard_bucket_ids(self, x):
        """
        Use the Hyper branch projections / LSH to define the same hard partition
        for the modified RACE branch.
        """
        B, T, _ = x.shape
        H = self.hyper.num_heads
        D = self.hyper.head_dim

        Qh = self.hyper.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        Kh = self.hyper.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()    # [B,H,T,D]

        q_ids = []
        k_ids = []
        for b in range(B):
            q_ids.append(self.shared_lsh.hash(Qh[b]))   # [H,T]
            k_ids.append(self.shared_lsh.hash(Kh[b]))   # [H,T]

        q_ids = torch.stack(q_ids, dim=0)               # [B,H,T]
        k_ids = torch.stack(k_ids, dim=0)               # [B,H,T]
        return q_ids, k_ids

    def forward(self, x):
        # Shared hard partition for both directions
        q_hard_bucket_ids, k_hard_bucket_ids = self._shared_hard_bucket_ids(x)

        # ------------------------------------------------------------
        # Compute RACE soft bucket probabilities WITHOUT autograd
        # to reduce memory usage. The RACE branch itself still gets
        # gradients through its own forward pass below.
        # ------------------------------------------------------------
        with torch.no_grad():
            probsQ_race, probsK_race = race_avg_bucket_probs_from_x(self.race, x)

        # Branch 1: exact sparse numerator + deterministic denominator estimate
        out_hyper = self.hyper(
            x,
            probsQ_race=probsQ_race,
            probsK_race=probsK_race,
        )                                                                       # [B,T,d]

        # Branch 2: existing modified RACE branch
        out_race = self.race(
            x,
            q_hard_bucket_ids=q_hard_bucket_ids,
            k_hard_bucket_ids=k_hard_bucket_ids,
        )                                                                       # [B,T,d]

        # Same sigmoid gating style as hyper_plus_race
        gate_logits = self.gate_mlp(x)      # [B,T,2]
        gates = torch.sigmoid(gate_logits)  # [B,T,2]
        self.last_gates = gates.detach()

        g_hyper = gates[..., 0:1]           # [B,T,1]
        g_race  = gates[..., 1:2]           # [B,T,1]

        out = g_hyper * out_hyper + g_race * out_race
        return out


class HyperPlusRaceNoSampleBlock(nn.Module):
    """
    Standard transformer block wrapper for hyper_plus_race_nosample.
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperPlusRaceNoSampleAttentionVision(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

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
# ==================================================
# NEW ATTENTION TYPE: true_hyper_plus_race
# Paste this block RIGHT AFTER HyperPlusRaceBlock
# and RIGHT BEFORE VisionTransformer
# ==================================================
# ==================================================
# NEW ATTENTION TYPE: hyper_plus_race_estimate
# Paste this block RIGHT AFTER HyperPlusRaceBlock
# and RIGHT BEFORE VisionTransformer
# ==================================================

def _gather_scores_3d(x: torch.Tensor, idx: torch.Tensor):
    """
    Gather along the token dimension for score tensors.

    x   : [H, T, S]
    idx : [H, T]
    out : [H, T, S]
    """
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def race_avg_bucket_probs_from_x(race_attn: "RACEAttention", x: torch.Tensor):
    """
    Compute RACE soft bucket probabilities from the CURRENT RACE branch.

    Returns
    -------
    probsQ_S_avg : [B, H, T, S]
    probsK_S_avg : [B, H, T, S]

    where
        S = L * (2^K)
    and the probabilities are averaged across M ensembles.

    IMPORTANT
    ---------
    This uses the same q_proj / k_proj / ACE parameters as the RACE branch,
    so the estimator branch and the RACE branch are aligned.
    """
    B, T, _ = x.shape
    H, dk, M = race_attn.H, race_attn.d_k, race_attn.M
    ace = race_attn.ace
    S = ace.L * ace.R

    scale = ace.logit_temp.exp().clamp(1e-2, 20.0)

    Q = race_attn.q_proj(x).view(B, T, H, dk)   # [B,T,H,dk]
    K = race_attn.k_proj(x).view(B, T, H, dk)   # [B,T,H,dk]

    def pack(Z):
        return Z.unsqueeze(0).expand(M, -1, -1, -1, -1)   # [M,B,T,H,dk]

    Qhf = pack(Q)
    Khf = pack(K)

    if ace.share_planes:
        N = M * B * H

        Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)  # [N,T,dk]
        Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

        projQ = (Qh2 @ ace.planes_T).view(M, B, H, T, ace.L, ace.K)
        projK = (Kh2 @ ace.planes_T).view(M, B, H, T, ace.L, ace.K)
    else:
        BH = B * H

        Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)  # [M,BH,T,dk]
        Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

        projQ = torch.einsum("mbtd,mds->mbts", Qh2, ace.planes_T)
        projK = torch.einsum("mbtd,mds->mbts", Kh2, ace.planes_T)

        projQ = projQ.contiguous().view(M, B, H, T, ace.L, ace.K)
        projK = projK.contiguous().view(M, B, H, T, ace.L, ace.K)

    logitsQ = (projQ.tanh().div(scale) @ ace.protos_T)      # [M,B,H,T,L,R]
    logitsK = (projK.tanh().div(scale) @ ace.protos_T)      # [M,B,H,T,L,R]

    probsQ = F.softmax(logitsQ, dim=-1)                     # [M,B,H,T,L,R]
    probsK = F.softmax(logitsK, dim=-1)

    probsQ_S = probsQ.contiguous().view(M, B, H, T, S)     # [M,B,H,T,S]
    probsK_S = probsK.contiguous().view(M, B, H, T, S)

    # Average across ensembles to get one proxy distribution
    probsQ_S_avg = probsQ_S.mean(dim=0)                     # [B,H,T,S]
    probsK_S_avg = probsK_S.mean(dim=0)                     # [B,H,T,S]

    return probsQ_S_avg, probsK_S_avg


class HyperLSHExactRACEEstimateAttentionVision(nn.Module):
    """
    Hyper-LSH exact sparse branch with a RACE control-variate denominator estimator.

    For each query i:
      - exact sparse support H_i from Hyper-LSH
      - exact sparse numerator   n1(i) = sum_{j in H_i} y_ij v_j
      - exact sparse denom part  d1(i) = sum_{j in H_i} y_ij
      - residual proxy from RACE:
            X_i = sum_{j in R_i} x_ij
      - residual correction by exact uniform samples from R_i:
            d2_hat(i) = c * X_i + |R_i| * mean_j [ y_ij - c * x_ij ]
      - final sparse-branch output:
            out_i = n1(i) / (d1(i) + d2_hat(i))

    where:
      y_ij = exp(q_i^T k_j / sqrt(d))
      x_ij = < race_probs_Q(i), race_probs_K(j) >

    IMPORTANT
    ---------
    - This is intentionally simpler and lighter than HyperAttention Algorithm 2.
    - It uses a control variate based on the full residual RACE proxy mass.
    - It uses uniform residual sampling because it is much cheaper to implement.
    """
    def __init__(
        self,
        d_in,
        d_out,
        dropout,
        num_heads,
        num_bits=5,
        block_size=32,
        min_seq_len=256,
        neighbor_blocks=0,
        qkv_bias=False,
        device="cpu",
        num_samples=8,
        c_init=1.0,
        estimator_eps=1e-6,
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.block_size = block_size
        self.min_seq_len = min_seq_len
        self.neighbor_blocks = neighbor_blocks

        self.num_samples = num_samples
        self.estimator_eps = estimator_eps
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.head_dim, device=device)

        # positive calibration c_h = exp(log_c_h), one scalar per head
        init_log_c = math.log(max(c_init, 1e-6))
        self.log_c = nn.Parameter(torch.full((num_heads,), init_log_c))

    def _full_sdpa_fallback(self, Qh, Kh, Vh):
        return _run_exact_sdpa(
            Qh.unsqueeze(0),  # [1,H,T,D]
            Kh.unsqueeze(0),
            Vh.unsqueeze(0),
        )[0]

    def forward(self, x, probsQ_race: torch.Tensor, probsK_race: torch.Tensor):
        """
        x          : [B,T,d]
        probsQ_race: [B,H,T,S]
        probsK_race: [B,H,T,S]
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2).contiguous()

        out = torch.zeros_like(Q)                                           # [B,H,T,D]

        for b in range(B):
            Qh = Q[b]   # [H,T,D]
            Kh = K[b]
            Vh = V[b]

            # Keep exact full-attention fallback for short sequences
            if T < self.min_seq_len:
                out[b] = self._full_sdpa_fallback(Qh, Kh, Vh)
                continue

            # --------------------------------------------------------
            # 1) Hyper-LSH sorting and support (same philosophy as hyper_lsh)
            # --------------------------------------------------------
            q_bucket_ids = self.lsh.hash(Qh)     # [H,T]
            k_bucket_ids = self.lsh.hash(Kh)     # [H,T]

            q_sort_idx = torch.argsort(q_bucket_ids, dim=1, stable=True)
            k_sort_idx = torch.argsort(k_bucket_ids, dim=1, stable=True)
            q_sort_inv = torch.argsort(q_sort_idx, dim=1, stable=True)

            Qs = _gather_tokens_3d(Qh, q_sort_idx)                  # [H,T,D]
            Ks = _gather_tokens_3d(Kh, k_sort_idx)
            Vs = _gather_tokens_3d(Vh, k_sort_idx)

            # Sort RACE probabilities into the SAME coordinates
            q_probs_sorted = _gather_scores_3d(probsQ_race[b], q_sort_idx)   # [H,T,S]
            k_probs_sorted = _gather_scores_3d(probsK_race[b], k_sort_idx)   # [H,T,S]

            out_sorted = torch.zeros_like(Qs)                                # [H,T,D]
            num_blocks = math.ceil(T / self.block_size)

            # --------------------------------------------------------
            # 2) Head-by-head exact sparse numerator + RACE estimate denominator
            # --------------------------------------------------------
            for h in range(H):
                c_h = self.log_c[h].exp()                                    # positive scale
                A_full = k_probs_sorted[h].sum(dim=0)                        # [S]

                for bi in range(num_blocks):
                    q0 = bi * self.block_size
                    q1 = min((bi + 1) * self.block_size, T)

                    left  = max(0, bi - self.neighbor_blocks)
                    right = min(num_blocks - 1, bi + self.neighbor_blocks)

                    k0 = left * self.block_size
                    k1 = min((right + 1) * self.block_size, T)

                    q_blk = Qs[h, q0:q1, :]                                   # [q,D]
                    k_blk = Ks[h, k0:k1, :]                                   # [k,D]
                    v_blk = Vs[h, k0:k1, :]                                   # [k,D]
                    q_prob_blk = q_probs_sorted[h, q0:q1, :]                  # [q,S]

                    # --------------------------------------------------
                    # Exact sparse numerator / denominator on support H_i
                    # --------------------------------------------------
                    logits_support = torch.einsum("qd,kd->qk", q_blk, k_blk) * self.scale
                    y_support = torch.exp(logits_support)                     # [q,k]

                    n1 = y_support @ v_blk                                    # [q,D]
                    d1 = y_support.sum(dim=-1)                                # [q]

                    # --------------------------------------------------
                    # Full residual proxy mass from RACE:
                    #   X_i = sum_{j in R_i} x_ij
                    # where x_ij = <q_prob_i, k_prob_j>
                    # Efficiently:
                    #   X_i = q_prob_i · (A_full - A_H)
                    # --------------------------------------------------
                    A_H = k_probs_sorted[h, k0:k1, :].sum(dim=0)              # [S]
                    A_R = A_full - A_H                                        # [S]
                    X = q_prob_blk @ A_R                                      # [q]

                    # --------------------------------------------------
                    # Uniform residual sampling for correction term
                    # --------------------------------------------------
                    support_size = k1 - k0
                    residual_count = T - support_size

                    if residual_count > 0 and self.num_samples > 0:
                        q_len = q1 - q0
                        m_eff = min(self.num_samples, residual_count)

                        # Sample uniformly from the complement of [k0, k1)
                        # by sampling in [0, residual_count) and shifting.
                        u = torch.randint(
                            low=0,
                            high=residual_count,
                            size=(q_len, m_eff),
                            device=x.device,
                        )                                                     # [q,m]

                        sample_idx = u + (u >= k0).long() * support_size      # [q,m]

                        # exact sampled y_ij
                        Ks_sample = Ks[h].index_select(0, sample_idx.reshape(-1))
                        Ks_sample = Ks_sample.view(q_len, m_eff, D)            # [q,m,D]

                        logits_sample = (q_blk.unsqueeze(1) * Ks_sample).sum(dim=-1) * self.scale
                        y_sample = torch.exp(logits_sample)                    # [q,m]

                        # sampled RACE proxy x_ij
                        kprob_sample = k_probs_sorted[h].index_select(0, sample_idx.reshape(-1))
                        kprob_sample = kprob_sample.view(q_len, m_eff, -1)     # [q,m,S]

                        x_sample = (q_prob_blk.unsqueeze(1) * kprob_sample).sum(dim=-1)  # [q,m]

                        # Since p_i(j) = 1 / |R_i|,
                        #   (1/m) sum (y-cx)/p = |R_i| * mean(y-cx)
                        correction = residual_count * (y_sample - c_h * x_sample).mean(dim=-1)  # [q]

                        d2_hat = c_h * X + correction                          # [q]
                    else:
                        d2_hat = c_h * X

                    denom = (d1 + d2_hat).clamp_min(self.estimator_eps)       # [q]
                    out_sorted[h, q0:q1, :] = n1 / denom.unsqueeze(-1)

            # --------------------------------------------------------
            # 3) Unsort back to original query order
            # --------------------------------------------------------
            out[b] = out_sorted.gather(
                1, q_sort_inv.unsqueeze(-1).expand(-1, -1, D)
            )

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)             # [B,T,d]
        out = self.dropout(out)
        return self.out_proj(out)


class HyperPlusRaceEstimateAttentionVision(nn.Module):
    """
    hyper_plus_race_estimate:
      - Branch 1: exact Hyper-LSH sparse numerator + RACE control-variate denominator estimate
      - Branch 2: same modified RACE branch as current hyper_plus_race
      - Final merge: same input-dependent sigmoid gates as hyper_plus_race
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        # Branch 1: Hyper exact sparse + estimated denominator
        self.hyper = HyperLSHExactRACEEstimateAttentionVision(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
            num_samples=cfg.get("hyper_plus_race_estimate_num_samples", 8),
            c_init=cfg.get("hyper_plus_race_estimate_c_init", 1.0),
            estimator_eps=cfg.get("hyper_plus_race_estimate_eps", 1e-6),
        )

        # Shared hard-bucket mapper for the modified RACE branch
        self.shared_lsh = self.hyper.lsh

        # Branch 2: keep the current modified RACE branch
        self.race = BucketExcludedRACEAttention(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            L=cfg["L"],
            K=cfg["K"],
            N_M=cfg["M"],
            hard_num_bits=cfg.get("hyper_num_bits", 5),
            q_chunk_size=cfg.get("hyper_plus_race_q_chunk_size", 256),
            qkv_bias=qkv_bias,
            device=device,
        )

        # Same gate style as hyper_plus_race
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        self.last_gates = None

    @torch.no_grad()
    def _shared_hard_bucket_ids(self, x):
        """
        Use the Hyper branch projections / LSH to define the same hard partition
        for the modified RACE branch.
        """
        B, T, _ = x.shape
        H = self.hyper.num_heads
        D = self.hyper.head_dim

        Qh = self.hyper.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        Kh = self.hyper.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()    # [B,H,T,D]

        q_ids = []
        k_ids = []
        for b in range(B):
            q_ids.append(self.shared_lsh.hash(Qh[b]))   # [H,T]
            k_ids.append(self.shared_lsh.hash(Kh[b]))   # [H,T]

        q_ids = torch.stack(q_ids, dim=0)               # [B,H,T]
        k_ids = torch.stack(k_ids, dim=0)               # [B,H,T]
        return q_ids, k_ids

    def forward(self, x):
        # Shared hard partition for both directions
        q_hard_bucket_ids, k_hard_bucket_ids = self._shared_hard_bucket_ids(x)

        # RACE soft bucket probabilities for the estimator branch
        probsQ_race, probsK_race = race_avg_bucket_probs_from_x(self.race, x)   # [B,H,T,S], [B,H,T,S]

        # Branch 1: exact sparse numerator + estimated denominator
        out_hyper = self.hyper(
            x,
            probsQ_race=probsQ_race,
            probsK_race=probsK_race,
        )                                                                       # [B,T,d]

        # Branch 2: existing modified RACE branch
        out_race = self.race(
            x,
            q_hard_bucket_ids=q_hard_bucket_ids,
            k_hard_bucket_ids=k_hard_bucket_ids,
        )                                                                       # [B,T,d]

        # Same sigmoid gating style as hyper_plus_race
        gate_logits = self.gate_mlp(x)      # [B,T,2]
        gates = torch.sigmoid(gate_logits)  # [B,T,2]
        self.last_gates = gates.detach()

        g_hyper = gates[..., 0:1]           # [B,T,1]
        g_race  = gates[..., 1:2]           # [B,T,1]

        out = g_hyper * out_hyper + g_race * out_race
        return out


class HyperPlusRaceEstimateBlock(nn.Module):
    """
    Standard transformer block wrapper for hyper_plus_race_estimate.
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperPlusRaceEstimateAttentionVision(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

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
class HyperLSHExactApproxDAttentionVision(nn.Module):
    """
    Hyper-LSH exact sparse branch with HyperAttention Algorithm-2 style denominator.

    This module:
      1) uses the SAME hard Hyper-LSH sorting/block support idea as hyper_lsh
      2) computes the sparse exact numerator on the selected support
      3) replaces the old sparse-branch denominator by an ApproxD-style estimate
         of the FULL row-sum:
             d_tilde_i = masked_sum_i + max(d_i, tau / kappa)

    IMPORTANT:
    - This is a practical implementation of ApproxD.
    - Algorithm 2 line 6 has a Theta(.) constant; here it is exposed by
      `clip_scale`.
    - To stay closest to HyperAttention Algorithm 1 + 2, use:
          hyper_neighbor_blocks = 0
    """
    def __init__(
        self,
        d_in,
        d_out,
        dropout,
        num_heads,
        num_bits=5,
        block_size=32,
        min_seq_len=256,
        neighbor_blocks=0,
        qkv_bias=False,
        device="cpu",
        num_samples=64,
        approx_eps=1.0,
        approx_kappa=4.0,
        clip_scale=1.0,
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.block_size = block_size
        self.min_seq_len = min_seq_len
        self.neighbor_blocks = neighbor_blocks

        self.num_samples = num_samples
        self.approx_eps = approx_eps
        self.approx_kappa = approx_kappa
        self.clip_scale = clip_scale

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.head_dim, device=device)

    def _full_sdpa_fallback(self, Qh, Kh, Vh):
        # Qh,Kh,Vh: [H, T, D]
        return _run_exact_sdpa(
            Qh.unsqueeze(0),   # [1,H,T,D]
            Kh.unsqueeze(0),
            Vh.unsqueeze(0),
        )[0]

    def _support_max(self, Qs, Ks, valid_T):
        """
        Compute the maximum support logit over all exact sparse blocks.
        This is used as part of a global stability shift before exponentiation.
        """
        device = Qs.device
        dtype = Qs.dtype
        H = Qs.size(0)
        _ = H  # kept for readability

        num_blocks = math.ceil(valid_T / self.block_size)
        max_logit = torch.tensor(float("-inf"), device=device, dtype=dtype)

        for bi in range(num_blocks):
            q0 = bi * self.block_size
            q1 = min((bi + 1) * self.block_size, valid_T)

            left = max(0, bi - self.neighbor_blocks)
            right = min(num_blocks - 1, bi + self.neighbor_blocks)

            k0 = left * self.block_size
            k1 = min((right + 1) * self.block_size, valid_T)

            q_blk = Qs[:, q0:q1, :]   # [H,q,D]
            k_blk = Ks[:, k0:k1, :]   # [H,k,D]

            logits = torch.einsum("hqd,hkd->hqk", q_blk, k_blk) * self.scale
            max_logit = torch.maximum(max_logit, logits.max())

        return max_logit

    def _exact_sparse_num_and_masked_sum(self, Qs, Ks, Vs, valid_T, global_shift):
        """
        Compute the exact sparse numerator and exact masked row-sum
        on the Hyper-LSH block support.

        Returns
        -------
        num_sorted   : [H, T, D]
        masked_sum   : [H, T]
        """
        H, _, D = Qs.shape
        num_sorted = torch.zeros_like(Qs)                              # [H,T,D]
        masked_sum = torch.zeros(H, valid_T, device=Qs.device, dtype=Qs.dtype)

        num_blocks = math.ceil(valid_T / self.block_size)

        for bi in range(num_blocks):
            q0 = bi * self.block_size
            q1 = min((bi + 1) * self.block_size, valid_T)

            left = max(0, bi - self.neighbor_blocks)
            right = min(num_blocks - 1, bi + self.neighbor_blocks)

            k0 = left * self.block_size
            k1 = min((right + 1) * self.block_size, valid_T)

            q_blk = Qs[:, q0:q1, :]   # [H,q,D]
            k_blk = Ks[:, k0:k1, :]   # [H,k,D]
            v_blk = Vs[:, k0:k1, :]   # [H,k,D]

            logits = torch.einsum("hqd,hkd->hqk", q_blk, k_blk) * self.scale
            weights = torch.exp(logits - global_shift)                 # [H,q,k]

            masked_sum[:, q0:q1] = weights.sum(dim=-1)                # [H,q]
            num_sorted[:, q0:q1, :] = torch.einsum("hqk,hkd->hqd", weights, v_blk)

        return num_sorted, masked_sum

    def _approxd_denominator(
        self,
        Qs,
        Ks,
        masked_sum,
        valid_T,
        global_shift,
        sample_rows,
        sample_cols,
        full_logits_rows,
        sampled_col_logits,
    ):
        """
        HyperAttention Algorithm-2 style denominator:
            d_tilde_i = masked_sum_i + max(d_i, tau / kappa)

        Here:
          - masked_sum_i is exact over the sparse support
          - d_i is the clipped sampled estimate of the residual row-sum
        """
        H = Qs.size(0)
        device = Qs.device
        dtype = Qs.dtype
        m = sample_rows.numel()

        if m == 0:
            return masked_sum

        # block ids in SORTED coordinates
        block_ids = torch.div(
            torch.arange(valid_T, device=device),
            self.block_size,
            rounding_mode="floor",
        )  # [T]

        # ------------------------------------------------------------
        # Step 1: tau = max residual row-sum over sampled rows
        # ------------------------------------------------------------
        row_block_ids = block_ids[sample_rows]                         # [m]
        support_mask_rows = (
            (row_block_ids[:, None] - block_ids[None, :]).abs() <= self.neighbor_blocks
        )                                                              # [m,T]

        exp_rows = torch.exp(full_logits_rows - global_shift)          # [H,m,T]
        residual_sample_rows = (
            (~support_mask_rows).unsqueeze(0).to(dtype) * exp_rows
        ).sum(dim=-1)                                                  # [H,m]

        tau = residual_sample_rows.max(dim=1).values                   # [H]

        # ------------------------------------------------------------
        # Step 2: C_i from Algorithm 2 line 6
        # ------------------------------------------------------------
        log_term = max(math.log(max(valid_T, 2)), 1.0)

        Ci = self.clip_scale * (
            (self.approx_eps ** 2) * float(m) / (float(valid_T) * log_term)
        ) * (
            masked_sum + tau.unsqueeze(-1) / self.approx_kappa
        )                                                              # [H,T]

        # ------------------------------------------------------------
        # Step 3: d_i from Algorithm 2 line 7
        # ------------------------------------------------------------
        sampled_col_block_ids = block_ids[sample_cols]                 # [m]
        support_mask_cols = (
            (block_ids[:, None] - sampled_col_block_ids[None, :]).abs() <= self.neighbor_blocks
        )                                                              # [T,m]

        exp_cols = torch.exp(sampled_col_logits - global_shift)        # [H,T,m]
        clipped_cols = torch.minimum(exp_cols, Ci.unsqueeze(-1))       # [H,T,m]

        d_i = (float(valid_T) / float(m)) * (
            (~support_mask_cols).unsqueeze(0).to(dtype) * clipped_cols
        ).sum(dim=-1)                                                  # [H,T]

        # ------------------------------------------------------------
        # Step 4: d_tilde_i = masked_sum_i + max(d_i, tau/kappa)
        # ------------------------------------------------------------
        tau_floor = (tau / self.approx_kappa).unsqueeze(-1).expand_as(d_i)  # [H,T]
        d_tilde = masked_sum + torch.maximum(d_i, tau_floor)                # [H,T]

        return d_tilde

    def forward(self, x):
        """
        x: [B, T, d]
        returns: [B, T, d]
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()   # [B,H,T,D]
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2).contiguous()

        out = torch.zeros_like(Q)                                            # [B,H,T,D]

        for b in range(B):
            valid_T = T

            Qh = Q[b]   # [H,T,D]
            Kh = K[b]
            Vh = V[b]

            # Keep the same short-sequence fallback behavior as your current hyper_lsh
            if valid_T < self.min_seq_len:
                out[b] = self._full_sdpa_fallback(Qh, Kh, Vh)
                continue

            # --------------------------------------------------------
            # 1) Hyper-LSH sorting (same as your current hyper_lsh)
            # --------------------------------------------------------
            q_bucket_ids = self.lsh.hash(Qh)   # [H,T]
            k_bucket_ids = self.lsh.hash(Kh)   # [H,T]

            q_sort_idx = torch.argsort(q_bucket_ids, dim=1, stable=True)
            k_sort_idx = torch.argsort(k_bucket_ids, dim=1, stable=True)
            q_sort_inv = torch.argsort(q_sort_idx, dim=1, stable=True)

            Qs = _gather_tokens_3d(Qh, q_sort_idx)   # [H,T,D]
            Ks = _gather_tokens_3d(Kh, k_sort_idx)
            Vs = _gather_tokens_3d(Vh, k_sort_idx)

            # --------------------------------------------------------
            # 2) Pre-sample rows / cols for ApproxD
            # --------------------------------------------------------
            m = min(self.num_samples, valid_T)
            if m > 0:
                sample_rows = torch.randperm(valid_T, device=x.device)[:m]      # subset T
                sample_cols = torch.randint(valid_T, (m,), device=x.device)      # i.i.d. columns
            else:
                sample_rows = torch.empty(0, dtype=torch.long, device=x.device)
                sample_cols = torch.empty(0, dtype=torch.long, device=x.device)

            # --------------------------------------------------------
            # 3) Build a GLOBAL stability shift before exponentiation
            #    (keeps the final ratio unchanged)
            # --------------------------------------------------------
            support_max = self._support_max(Qs, Ks, valid_T)

            if m > 0:
                full_logits_rows = torch.einsum(
                    "hmd,htd->hmt",
                    Qs[:, sample_rows, :],
                    Ks,
                ) * self.scale                                                # [H,m,T]

                sampled_col_logits = torch.einsum(
                    "htd,hmd->htm",
                    Qs,
                    Ks[:, sample_cols, :],
                ) * self.scale                                                # [H,T,m]

                global_shift = torch.max(
                    torch.stack([
                        support_max,
                        full_logits_rows.max(),
                        sampled_col_logits.max(),
                    ])
                )
            else:
                full_logits_rows = None
                sampled_col_logits = None
                global_shift = support_max

            # --------------------------------------------------------
            # 4) Exact sparse numerator + exact masked row-sum
            # --------------------------------------------------------
            num_sorted, masked_sum = self._exact_sparse_num_and_masked_sum(
                Qs, Ks, Vs, valid_T, global_shift
            )                                                                # [H,T,D], [H,T]

            # --------------------------------------------------------
            # 5) ApproxD denominator estimate
            # --------------------------------------------------------
            d_tilde = self._approxd_denominator(
                Qs=Qs,
                Ks=Ks,
                masked_sum=masked_sum,
                valid_T=valid_T,
                global_shift=global_shift,
                sample_rows=sample_rows,
                sample_cols=sample_cols,
                full_logits_rows=full_logits_rows,
                sampled_col_logits=sampled_col_logits,
            )                                                                # [H,T]

            # --------------------------------------------------------
            # 6) Sparse exact branch output:
            #       n1 / d_tilde
            # --------------------------------------------------------
            out_sorted = num_sorted / (d_tilde.unsqueeze(-1) + 1e-6)         # [H,T,D]

            # unsort query order back to original
            out[b] = out_sorted.gather(
                1, q_sort_inv.unsqueeze(-1).expand(-1, -1, D)
            )

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.dropout(out)
        return self.out_proj(out)


class TrueHyperPlusRaceAttentionVision(nn.Module):
    """
    true_hyper_plus_race:
      - Branch 1: Hyper-LSH exact sparse numerator with ApproxD-style full denominator
      - Branch 2: existing modified RACE branch (same as current hyper_plus_race)
      - Final merge: same input-dependent sigmoid gates as hyper_plus_race
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        # ------------------------------------------------------------
        # Branch 1: Hyper-LSH + ApproxD denominator
        # ------------------------------------------------------------
        self.hyper = HyperLSHExactApproxDAttentionVision(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
            num_samples=cfg.get("true_hpr_num_samples", 64),
            approx_eps=cfg.get("true_hpr_eps", 1.0),
            approx_kappa=cfg.get("true_hpr_kappa", 4.0),
            clip_scale=cfg.get("true_hpr_clip_scale", 1.0),
        )

        # shared hard-bucket mapper for the modified RACE branch
        self.shared_lsh = self.hyper.lsh

        # ------------------------------------------------------------
        # Branch 2: keep your current modified RACE branch exactly
        # ------------------------------------------------------------
        self.race = BucketExcludedRACEAttention(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            L=cfg["L"],
            K=cfg["K"],
            N_M=cfg["M"],
            hard_num_bits=cfg.get("hyper_num_bits", 5),
            q_chunk_size=cfg.get("hyper_plus_race_q_chunk_size", 256),
            qkv_bias=qkv_bias,
            device=device,
        )

        # same gate style as hyper_plus_race
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        self.last_gates = None

    @torch.no_grad()
    def _shared_hard_bucket_ids(self, x):
        """
        Use the SAME Hyper branch projections / LSH object to define
        the hard partition shared by the modified RACE branch.
        """
        B, T, _ = x.shape
        H = self.hyper.num_heads
        D = self.hyper.head_dim

        Qh = self.hyper.W_query(x).view(B, T, H, D).transpose(1, 2).contiguous()  # [B,H,T,D]
        Kh = self.hyper.W_key(x).view(B, T, H, D).transpose(1, 2).contiguous()    # [B,H,T,D]

        q_ids = []
        k_ids = []
        for b in range(B):
            q_ids.append(self.shared_lsh.hash(Qh[b]))   # [H,T]
            k_ids.append(self.shared_lsh.hash(Kh[b]))   # [H,T]

        q_ids = torch.stack(q_ids, dim=0)               # [B,H,T]
        k_ids = torch.stack(k_ids, dim=0)               # [B,H,T]
        return q_ids, k_ids

    def forward(self, x):
        # same hard partition for both directions
        q_hard_bucket_ids, k_hard_bucket_ids = self._shared_hard_bucket_ids(x)

        # Branch 1: true hyper + ApproxD denominator
        out_hyper = self.hyper(x)   # [B,T,d]

        # Branch 2: current modified RACE branch
        out_race = self.race(
            x,
            q_hard_bucket_ids=q_hard_bucket_ids,
            k_hard_bucket_ids=k_hard_bucket_ids,
        )                           # [B,T,d]

        # same gate style as hyper_plus_race
        gate_logits = self.gate_mlp(x)      # [B,T,2]
        gates = torch.sigmoid(gate_logits)  # [B,T,2]

        self.last_gates = gates.detach()

        g_hyper = gates[..., 0:1]   # [B,T,1]
        g_race  = gates[..., 1:2]   # [B,T,1]

        out = g_hyper * out_hyper + g_race * out_race
        return out


class TrueHyperPlusRaceBlock(nn.Module):
    """
    Standard transformer block wrapper for true_hyper_plus_race.
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = TrueHyperPlusRaceAttentionVision(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

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
# ==================================================
# NEW ATTENTION TYPE: hyper_race_dependent_lambda
#
# Same as hyper_race_mexact, but lambda is token/query dependent:
#
#   lambda_i = offset + sigmoid(w^T q_i + b)
#
# and:
#
#   m_exact_i = d_exact_i / (d_exact_i + lambda_i * d_race_i + eps)
#
# No global/local.
# No RACE elimination.
# ==================================================


class HyperRaceDependentLambdaAttentionVision(nn.Module):
    """
    hyper_race_dependent_lambda

    Same base structure as hyper_race_mexact:

        out = g_hyper * m_exact * out_hyper + g_race * out_race

    but instead of a scalar lambda, each token gets its own lambda:

        lambda_i = offset + sigmoid(w^T q_i + b)

    where q_i is the projected query vector from the Hyper-LSH exact branch.

    Then:

        m_exact_i =
            d_exact_i / (d_exact_i + lambda_i * d_race_i + eps)

    Implementation notes
    --------------------
    - The Hyper-LSH exact branch is unchanged.
    - The RACE branch is unchanged.
    - Only the m_exact correction changes.
    - d_exact and d_race are detached from the m_exact path.
    - lambda remains learnable.
    - By default q_i is detached, so the lambda path learns w and b without
      sending extra gradients into the query projection.
    """

    def __init__(self, cfg, device="cpu"):
        super().__init__()

        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 128)

        self.mexact_eps = cfg.get("mexact_eps", 1e-6)

# ------------------------------------------------------------
# Lambda(q) configuration
#
#   lambda_i = c + sigmoid(w^T q_i + b)
#
# c can be fixed or learnable:
#   mexact_dependent_lambda_offset_learnable = True / False
#
# If mexact_dependent_lambda_offset_positive=True, the forward value of c is
# constrained to be nonnegative. I use a straight-through clamp so that c can
# still learn even if initialized at exactly 0.
# ------------------------------------------------------------
        self.lambda_min = float(cfg.get("mexact_dependent_lambda_min", 1e-6))

        self.lambda_offset_learnable = bool(
            cfg.get("mexact_dependent_lambda_offset_learnable", False)
        )

        self.lambda_offset_positive = bool(
            cfg.get("mexact_dependent_lambda_offset_positive", True)
        )

        offset_init = float(cfg.get("mexact_dependent_lambda_offset", 0.3))

        if self.lambda_offset_learnable:
            self.lambda_offset_raw = nn.Parameter(
                torch.tensor(offset_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "lambda_offset_raw",
                torch.tensor(offset_init, dtype=torch.float32),
                persistent=False,
            )

        self.lambda_use_bias = bool(cfg.get("mexact_dependent_lambda_use_bias", True))
        self.lambda_detach_q = bool(cfg.get("mexact_dependent_lambda_detach_q", True))

        lambda_w_init_std = float(cfg.get("mexact_dependent_lambda_w_init_std", 1e-3))
        if lambda_w_init_std < 0:
            raise ValueError("mexact_dependent_lambda_w_init_std must be >= 0.")

        # w in lambda_i = c + sigmoid(w^T q_i + b)
        self.lambda_w = nn.Parameter(torch.empty(d, dtype=torch.float32))
        nn.init.normal_(self.lambda_w, mean=0.0, std=lambda_w_init_std)

        if self.lambda_use_bias:
            # Initialize b so that:
            #
            #   c_init + sigmoid(b) ≈ init_target
            #
            # If c_init=0.3 and init_target=0.8:
            #   sigmoid(b)=0.5
            #   b=0
            #
            # If c_init=0.0 and init_target=0.8:
            #   sigmoid(b)=0.8
            #   b=logit(0.8)
            effective_offset_init = max(offset_init, 0.0) if self.lambda_offset_positive else offset_init

            init_target = float(
                cfg.get("mexact_dependent_lambda_init_target", effective_offset_init + 0.5)
            )

            init_prob = init_target - effective_offset_init

            # Avoid crashing if user sets an aggressive target. This just initializes
            # the sigmoid part close to the boundary instead of throwing an error.
            init_prob = min(max(init_prob, 1e-4), 1.0 - 1e-4)

            init_bias = math.log(init_prob / (1.0 - init_prob))
            self.lambda_bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
        else:
            self.register_buffer(
                "lambda_bias",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=False,
            )

        # ------------------------------------------------------------
        # Hyper-LSH exact branch with denominator
        # ------------------------------------------------------------
        self.hyper = HyperLSHExactWithLogDenomAttentionVision(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
            mexact_eps=self.mexact_eps,
        )

        # ------------------------------------------------------------
        # Normal RACE branch with denominator proxy
        # ------------------------------------------------------------
        self.race = RACEAttentionWithDenom(
            d_in=d,
            d_out=d,
            dropout=drop,
            num_heads=h,
            L=cfg["L"],
            K=cfg["K"],
            N_M=cfg["M"],
            qkv_bias=qkv_bias,
            device=device,
        )

        # ------------------------------------------------------------
        # Same gate MLP style as hyper_race / hyper_race_mexact
        # ------------------------------------------------------------
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        self.normalize_gates = cfg.get("gate_normalize", False)

        # ------------------------------------------------------------
        # Saved tensors for logging
        # ------------------------------------------------------------
        self.last_gates = None
        self.last_m_exact = None
        self.last_d_exact_mean = None
        self.last_d_race_mean = None
        self.last_mexact_lambda = None
        self.last_mexact_lambda_logits = None
        self.last_mexact_lambda_offset = None
        self.last_mexact_lambda_sigmoid = None
    def _current_lambda_offset(self, dtype, device):
        """
        Returns the current scalar offset c.

        If lambda_offset_positive=True, the forward value is clamped to c >= 0.

        I use a straight-through clamp:

            c_forward = clamp(c_raw, min=0)
            c = c_raw + stop_grad(c_forward - c_raw)

        So the forward value is nonnegative, but gradients can still move c_raw.
        This is useful when you initialize c exactly at 0.
        """
        c_raw = self.lambda_offset_raw.to(device=device, dtype=dtype)

        if self.lambda_offset_positive:
            c_forward = c_raw.clamp_min(0.0)
            c = c_raw + (c_forward - c_raw).detach()
        else:
            c = c_raw

        return c    

    def _compute_query_dependent_lambda(self, x):
        """
        Computes token-wise lambda.

        Uses the Hyper branch's W_query projection, so q_i is aligned with
        the exact sparse branch.

        Returns:
            lambda_q       : [B,T,1]
            lambda_logits  : [B,T]
            lambda_offset  : scalar tensor
            lambda_sigmoid : [B,T]
        """
        if self.lambda_detach_q:
            with torch.no_grad():
                q_for_lambda = self.hyper.W_query(x)  # [B,T,d]
        else:
            q_for_lambda = self.hyper.W_query(x)      # [B,T,d]

        q_for_lambda = q_for_lambda.float()

        lambda_logits = torch.matmul(
            q_for_lambda,
            self.lambda_w.float(),
        )  # [B,T]

        if self.lambda_use_bias:
            lambda_logits = lambda_logits + self.lambda_bias.float()

        lambda_sigmoid = torch.sigmoid(lambda_logits)  # [B,T]

        lambda_offset = self._current_lambda_offset(
            dtype=lambda_sigmoid.dtype,
            device=lambda_sigmoid.device,
        )  # scalar

        lambda_q = lambda_offset + lambda_sigmoid      # [B,T]
        lambda_q = lambda_q.clamp_min(self.lambda_min)
        lambda_q = lambda_q.unsqueeze(-1)              # [B,T,1]

        return lambda_q, lambda_logits, lambda_offset, lambda_sigmoid

    def forward(self, x):
        # ------------------------------------------------------------
        # 1) Branch outputs and denominator proxies
        # ------------------------------------------------------------
        out_hyper, log_d_exact = self.hyper(x)  # [B,T,d], [B,T,1]
        out_race, d_race = self.race(x)         # [B,T,d], [B,T,1]

        # ------------------------------------------------------------
        # 2) Token/query-dependent lambda
        # ------------------------------------------------------------
        lambda_q, lambda_logits, lambda_offset, lambda_sigmoid = self._compute_query_dependent_lambda(x)

        # ------------------------------------------------------------
        # 3) m_exact in log-space:
        #
        #   m_exact =
        #       d_exact / (d_exact + lambda_q * d_race + eps)
        #
        #   log_den =
        #       logsumexp(log_d_exact,
        #                 log(lambda_q) + log_d_race,
        #                 log_eps)
        #
        # d_exact and d_race are detached.
        # lambda_q is NOT detached, so lambda_w / lambda_bias can learn.
        # ------------------------------------------------------------
        log_d_exact_det = log_d_exact.detach().float()  # [B,T,1]

        log_d_race_det = torch.log(
            d_race.detach().float().clamp_min(self.mexact_eps)
        )  # [B,T,1]

        log_lambda_q = torch.log(
            lambda_q.float().clamp_min(self.mexact_eps)
        )  # [B,T,1]

        log_eps = torch.full_like(
            log_d_exact_det,
            math.log(self.mexact_eps),
        )

        log_den = torch.logsumexp(
            torch.stack(
                [
                    log_d_exact_det,
                    log_lambda_q + log_d_race_det,
                    log_eps,
                ],
                dim=0,
            ),
            dim=0,
        )  # [B,T,1]

        m_exact = torch.exp(log_d_exact_det - log_den).to(out_hyper.dtype)  # [B,T,1]

        # ------------------------------------------------------------
        # 4) Normal branch gates
        # ------------------------------------------------------------
        gate_logits = self.gate_mlp(x)      # [B,T,2]
        gates = torch.sigmoid(gate_logits)  # [B,T,2]

        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        g_hyper = gates[..., 0:1]  # [B,T,1]
        g_race = gates[..., 1:2]   # [B,T,1]

        # ------------------------------------------------------------
        # 5) Save logging stats
        # ------------------------------------------------------------
        self.last_gates = gates.detach()
        self.last_m_exact = m_exact.detach()
        self.last_d_exact_mean = torch.exp(
            log_d_exact.detach().clamp(max=20.0)
        ).mean()
        self.last_d_race_mean = d_race.detach().mean()
        self.last_mexact_lambda = lambda_q.detach()
        self.last_mexact_lambda_logits = lambda_logits.detach()
        self.last_mexact_lambda_offset = lambda_offset.detach()
        self.last_mexact_lambda_sigmoid = lambda_sigmoid.detach()

        # ------------------------------------------------------------
        # 6) Final hybrid output
        # ------------------------------------------------------------
        out = g_hyper * m_exact * out_hyper + g_race * out_race
        return out


class HyperRaceDependentLambdaBlock(nn.Module):
    """
    Transformer block wrapper for hyper_race_dependent_lambda.
    """

    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]

        self.att = HyperRaceDependentLambdaAttentionVision(cfg, device=device)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

        self.drop = nn.Dropout(drop)

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
class VisionTransformer(nn.Module):
    def __init__(self, cfg, attn_type, device='cpu'):
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
        elif attn_type == "true_hyper_plus_race":
            AttnBlock = lambda c: TrueHyperPlusRaceBlock(c, device)    
        elif attn_type == "hyper_race":
            AttnBlock = lambda c: HyperRaceGatedBlock(c, device)  
        elif attn_type == "hyper_plus_race":
            AttnBlock = lambda c: HyperPlusRaceBlock(c, device)  
        elif attn_type == "hyper_race_mexact":
            AttnBlock = lambda c: HyperRaceMExactBlock(c, device) 
        elif attn_type == "hyper_race_dependent_lambda":
            AttnBlock = lambda c: HyperRaceDependentLambdaBlock(c, device)        
        elif attn_type == "hyper_race_gl_mexact":
            AttnBlock = lambda c: HyperRaceGlobalLocalMExactBlock(c, device)            
        elif attn_type == "hyper_plus_race_estimate":
            AttnBlock = lambda c: HyperPlusRaceEstimateBlock(c, device)       
        elif attn_type == "angular":
            AttnBlock = AngularBlock
        elif attn_type == "hyper_plus_race_nosample":
            AttnBlock = lambda c: HyperPlusRaceNoSampleBlock(c, device)    
        elif attn_type == "linear":
            AttnBlock = LinearBlock
        elif attn_type == "linformer":
            AttnBlock = LinformerBlock
        elif attn_type == "performer":
            AttnBlock = PerformerBlock
        elif attn_type == "exact_flash":
            AttnBlock = ExactFlashBlock
        elif attn_type == "hyper_lsh":
            AttnBlock = lambda c: HyperLSHExactBlock(c, device)    
        else:
            raise ValueError("Unsupported attention type")

        self.transformer_layers = nn.Sequential(
            *[AttnBlock(cfg) for _ in range(cfg["transformer_units"])]
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg["num_classes"]))

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
    attn_type,
    grad_accum_steps: int = 1
):
    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []
    train_times, val_times = [], []

    #K, L, M = cfg.get("K", None), cfg.get("L", None), cfg.get("M", None)
    #out_path = f"trial_{attn_type}_K{K}_L{L}_M{M}_VIT.txt"
    K, L, M = cfg.get("K", None), cfg.get("L", None), cfg.get("M", None)

    log_attn_name = (
        "hyper_race_mexact_lambda"
        if attn_type == "hyper_race_mexact" and cfg.get("mexact_lambda_learnable", False)
        else attn_type
    )

    out_path = f"trial_{log_attn_name}_K{K}_L{L}_M{M}_VIT.txt"
    steps_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates  = num_epochs * updates_per_epoch
    warmup_updates = max(1, int(0.01 * total_updates))

    scheduler = LinearWarmupLR(
        optimizer,
        warmup_steps=warmup_updates,
        total_steps=total_updates,
    )

    def _log(fp, msg):
        print(msg)
        fp.write(msg + "\n")
        fp.flush()

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

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

                (loss / grad_accum_steps).backward()
                accum_count += 1

                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_total   += labels.size(0)
                running_loss    += loss.item()

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

            gate_stats = {}

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                    val_loss_total += loss.item()

                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)

                    if attn_type in { "hyper_race_dependent_lambda","hyper_race_mexact","hyper_race", "hyper_race_gl_mexact","hyper_plus_race","true_hyper_plus_race","hyper_plus_race_estimate","hyper_plus_race_nosample"}:
                        for layer_idx, layer in enumerate(model.transformer_layers):
                            if hasattr(layer, "att") and hasattr(layer.att, "last_gates"):
                                gates = layer.att.last_gates
                                if gates is None:
                                    continue

                                hyper_g = gates[..., 0].reshape(-1).detach().cpu().float()
                                race_g  = gates[..., 1].reshape(-1).detach().cpu().float()

                                if layer_idx not in gate_stats:
                                    gate_stats[layer_idx] = {
                                        "hyper_sum": 0.0,
                                        "hyper_sumsq": 0.0,
                                        "hyper_min": float("inf"),
                                        "hyper_max": float("-inf"),
                                        "race_sum": 0.0,
                                        "race_sumsq": 0.0,
                                        "race_min": float("inf"),
                                        "race_max": float("-inf"),
                                        "count": 0,
                                        "hyper_hist_sample": None,
                                        "race_hist_sample": None,
                                    }

                                st = gate_stats[layer_idx]

                                st["hyper_sum"] += hyper_g.sum().item()
                                st["hyper_sumsq"] += (hyper_g ** 2).sum().item()
                                st["hyper_min"] = min(st["hyper_min"], hyper_g.min().item())
                                st["hyper_max"] = max(st["hyper_max"], hyper_g.max().item())

                                st["race_sum"] += race_g.sum().item()
                                st["race_sumsq"] += (race_g ** 2).sum().item()
                                st["race_min"] = min(st["race_min"], race_g.min().item())
                                st["race_max"] = max(st["race_max"], race_g.max().item())

                                st["count"] += hyper_g.numel()

                                if st["hyper_hist_sample"] is None:
                                    st["hyper_hist_sample"] = hyper_g[:min(20000, hyper_g.numel())].numpy()
                                if st["race_hist_sample"] is None:
                                    st["race_hist_sample"] = race_g[:min(20000, race_g.numel())].numpy()

            if "cuda" in str(device):
                torch.cuda.synchronize()
            val_time = time.time() - t1
            val_times.append(val_time)

            va_l = val_loss_total / len(val_loader)
            va_a = val_correct / max(1, val_total)
            val_losses.append(va_l)
            val_accs.append(va_a)

            curr_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]

            extra_logs = {}
                        # --------------------------------------------------
            # Log learnable c parameters for hyper_plus_race_nosample
            # --------------------------------------------------
            if attn_type == "hyper_plus_race_nosample":
                c_all = []

                for layer_idx, layer in enumerate(model.transformer_layers):
                    if hasattr(layer, "att"):
                        att = layer.att
                        if hasattr(att, "hyper") and hasattr(att.hyper, "log_c"):
                            c_vals = att.hyper.log_c.detach().exp().cpu()   # [num_heads]

                            for head_idx, c_val in enumerate(c_vals.tolist()):
                                extra_logs[f"c/layer{layer_idx}_head{head_idx}"] = c_val
                                c_all.append(c_val)

                            extra_logs[f"c/layer{layer_idx}_mean"] = c_vals.mean().item()
                            extra_logs[f"c/layer{layer_idx}_std"]  = c_vals.std().item()
                            extra_logs[f"c/layer{layer_idx}_min"]  = c_vals.min().item()
                            extra_logs[f"c/layer{layer_idx}_max"]  = c_vals.max().item()

                if len(c_all) > 0:
                    c_tensor = torch.tensor(c_all, dtype=torch.float32)
                    extra_logs["c/global_mean"] = c_tensor.mean().item()
                    extra_logs["c/global_std"]  = c_tensor.std().item()
                    extra_logs["c/global_min"]  = c_tensor.min().item()
                    extra_logs["c/global_max"]  = c_tensor.max().item()            # --------------------------------------------------
            # Log learnable c parameters for hyper_plus_race_estimate
            # --------------------------------------------------
            if attn_type == "hyper_plus_race_estimate":
                c_all = []

                for layer_idx, layer in enumerate(model.transformer_layers):
                    if hasattr(layer, "att"):
                        att = layer.att

                        # att is HyperPlusRaceEstimateAttentionVision
                        if hasattr(att, "hyper") and hasattr(att.hyper, "log_c"):
                            c_vals = att.hyper.log_c.detach().exp().cpu()   # [num_heads]

                            for head_idx, c_val in enumerate(c_vals.tolist()):
                                extra_logs[f"c/layer{layer_idx}_head{head_idx}"] = c_val
                                c_all.append(c_val)

                            # per-layer summary
                            extra_logs[f"c/layer{layer_idx}_mean"] = c_vals.mean().item()
                            extra_logs[f"c/layer{layer_idx}_std"]  = c_vals.std().item()
                            extra_logs[f"c/layer{layer_idx}_min"]  = c_vals.min().item()
                            extra_logs[f"c/layer{layer_idx}_max"]  = c_vals.max().item()

                if len(c_all) > 0:
                    c_tensor = torch.tensor(c_all, dtype=torch.float32)
                    extra_logs["c/global_mean"] = c_tensor.mean().item()
                    extra_logs["c/global_std"]  = c_tensor.std().item()
                    extra_logs["c/global_min"]  = c_tensor.min().item()
                    extra_logs["c/global_max"]  = c_tensor.max().item()
            # --------------------------------------------------
            # Existing hyper_race gate logging
            # --------------------------------------------------
            if attn_type in { "hyper_race_dependent_lambda","hyper_race_mexact","hyper_race_gl_mexact","hyper_plus_race_nosample","hyper_race", "hyper_plus_race","true_hyper_plus_race","hyper_plus_race_estimate"} and len(gate_stats) > 0:
                hyper_means = []
                race_means = []

                for layer_idx, st in gate_stats.items():
                    count = max(st["count"], 1)

                    hyper_mean = st["hyper_sum"] / count
                    race_mean  = st["race_sum"] / count

                    hyper_var = max(st["hyper_sumsq"] / count - hyper_mean ** 2, 0.0)
                    race_var  = max(st["race_sumsq"] / count - race_mean ** 2, 0.0)

                    hyper_std = math.sqrt(hyper_var)
                    race_std  = math.sqrt(race_var)

                    extra_logs[f"gates/layer{layer_idx}_hyper_mean"] = hyper_mean
                    extra_logs[f"gates/layer{layer_idx}_race_mean"] = race_mean
                    extra_logs[f"gates/layer{layer_idx}_hyper_std"] = hyper_std
                    extra_logs[f"gates/layer{layer_idx}_race_std"] = race_std
                    extra_logs[f"gates/layer{layer_idx}_hyper_min"] = st["hyper_min"]
                    extra_logs[f"gates/layer{layer_idx}_race_min"] = st["race_min"]
                    extra_logs[f"gates/layer{layer_idx}_hyper_max"] = st["hyper_max"]
                    extra_logs[f"gates/layer{layer_idx}_race_max"] = st["race_max"]

                    hyper_means.append(hyper_mean)
                    race_means.append(race_mean)

                    if epoch % 5 == 0:
                        extra_logs[f"gates_hist/layer{layer_idx}_hyper"] = wandb.Histogram(st["hyper_hist_sample"])
                        extra_logs[f"gates_hist/layer{layer_idx}_race"] = wandb.Histogram(st["race_hist_sample"])

                extra_logs["gates/global_hyper_mean"] = sum(hyper_means) / len(hyper_means)
                extra_logs["gates/global_race_mean"] = sum(race_means) / len(race_means)
                        # --------------------------------------------------
            # Log m_exact and denominator stats for hyper_race_mexact
            # --------------------------------------------------
            # --------------------------------------------------
            # Log m_exact, denominator stats, and optional lambda
            # for hyper_race_mexact
            # --------------------------------------------------
            # --------------------------------------------------
# Log m_exact, denominator stats, and lambda stats
# for hyper_race_mexact and hyper_race_dependent_lambda
# --------------------------------------------------
            if attn_type in {"hyper_race_mexact", "hyper_race_dependent_lambda"}:
                m_vals = []
                d_exact_vals = []
                d_race_vals = []
                lambda_tensors = []
                lambda_logit_tensors = []
                lambda_offset_vals = []
                lambda_sigmoid_tensors = []
                for layer_idx, layer in enumerate(model.transformer_layers):
                    if not hasattr(layer, "att"):
                        continue

                    att = layer.att

                    # ------------------------------
                    # m_exact stats
                    # ------------------------------
                    if hasattr(att, "last_m_exact") and att.last_m_exact is not None:
                        m = att.last_m_exact.detach().cpu().float().reshape(-1)

                        extra_logs[f"m_exact/layer{layer_idx}_mean"] = m.mean().item()
                        extra_logs[f"m_exact/layer{layer_idx}_std"] = m.std(unbiased=False).item()
                        extra_logs[f"m_exact/layer{layer_idx}_min"] = m.min().item()
                        extra_logs[f"m_exact/layer{layer_idx}_max"] = m.max().item()

                        m_vals.append(m.mean().item())

                    # ------------------------------
                    # denominator stats
                    # ------------------------------
                    if hasattr(att, "last_d_exact_mean") and att.last_d_exact_mean is not None:
                        d_exact = float(att.last_d_exact_mean.detach().cpu())
                        d_race = float(att.last_d_race_mean.detach().cpu())

                        extra_logs[f"den/layer{layer_idx}_exact_mean"] = d_exact
                        extra_logs[f"den/layer{layer_idx}_race_mean"] = d_race

                        d_exact_vals.append(d_exact)
                        d_race_vals.append(d_race)
                    # ------------------------------
                    # lambda offset c
                    # Only exists for hyper_race_dependent_lambda.
                    # ------------------------------
                    if hasattr(att, "last_mexact_lambda_offset") and att.last_mexact_lambda_offset is not None:
                        c_val = float(att.last_mexact_lambda_offset.detach().cpu())

                        extra_logs[f"lambda_offset/layer{layer_idx}"] = c_val
                        lambda_offset_vals.append(c_val)

                    # ------------------------------
                    # sigmoid(w^T q + b) part
                    # Only exists for hyper_race_dependent_lambda.
                    # ------------------------------
                    if hasattr(att, "last_mexact_lambda_sigmoid") and att.last_mexact_lambda_sigmoid is not None:
                        sig = att.last_mexact_lambda_sigmoid.detach().cpu().float().reshape(-1)

                        extra_logs[f"lambda_sigmoid/layer{layer_idx}_mean"] = sig.mean().item()
                        extra_logs[f"lambda_sigmoid/layer{layer_idx}_std"] = sig.std(unbiased=False).item()
                        extra_logs[f"lambda_sigmoid/layer{layer_idx}_min"] = sig.min().item()
                        extra_logs[f"lambda_sigmoid/layer{layer_idx}_max"] = sig.max().item()

                        lambda_sigmoid_tensors.append(sig)

                        if epoch % 5 == 0:
                            extra_logs[f"lambda_sigmoid_hist/layer{layer_idx}"] = wandb.Histogram(
                                sig[:min(20000, sig.numel())].numpy()
                            )
                    # ------------------------------
                    # lambda stats
                    #
                    # For hyper_race_mexact:
                    #   last_mexact_lambda is scalar if scalar lambda is enabled.
                    #
                    # For hyper_race_dependent_lambda:
                    #   last_mexact_lambda is [B,T,1].
                    # ------------------------------
                    if hasattr(att, "last_mexact_lambda") and att.last_mexact_lambda is not None:
                        lam = att.last_mexact_lambda.detach().cpu().float().reshape(-1)

                        extra_logs[f"lambda/layer{layer_idx}_mean"] = lam.mean().item()
                        extra_logs[f"lambda/layer{layer_idx}_std"] = lam.std(unbiased=False).item()
                        extra_logs[f"lambda/layer{layer_idx}_min"] = lam.min().item()
                        extra_logs[f"lambda/layer{layer_idx}_max"] = lam.max().item()

                        lambda_tensors.append(lam)

                        if epoch % 5 == 0:
                            extra_logs[f"lambda_hist/layer{layer_idx}"] = wandb.Histogram(
                                lam[:min(20000, lam.numel())].numpy()
                            )

                    # ------------------------------
                    # lambda logit stats
                    # Only exists for hyper_race_dependent_lambda.
                    # ------------------------------
                    if hasattr(att, "last_mexact_lambda_logits") and att.last_mexact_lambda_logits is not None:
                        lam_logits = att.last_mexact_lambda_logits.detach().cpu().float().reshape(-1)

                        extra_logs[f"lambda_logits/layer{layer_idx}_mean"] = lam_logits.mean().item()
                        extra_logs[f"lambda_logits/layer{layer_idx}_std"] = lam_logits.std(unbiased=False).item()
                        extra_logs[f"lambda_logits/layer{layer_idx}_min"] = lam_logits.min().item()
                        extra_logs[f"lambda_logits/layer{layer_idx}_max"] = lam_logits.max().item()

                        lambda_logit_tensors.append(lam_logits)

                        if epoch % 5 == 0:
                            extra_logs[f"lambda_logits_hist/layer{layer_idx}"] = wandb.Histogram(
                                lam_logits[:min(20000, lam_logits.numel())].numpy()
                            )

                # ------------------------------
                # global m_exact stats
                # ------------------------------
                if len(m_vals) > 0:
                    extra_logs["m_exact/global_mean"] = sum(m_vals) / len(m_vals)

                # ------------------------------
                # global denominator stats
                # ------------------------------
                if len(d_exact_vals) > 0:
                    extra_logs["den/global_exact_mean"] = sum(d_exact_vals) / len(d_exact_vals)
                    extra_logs["den/global_race_mean"] = sum(d_race_vals) / len(d_race_vals)

                # ------------------------------
                # global lambda stats
                # ------------------------------
                if len(lambda_tensors) > 0:
                    lambda_all = torch.cat(lambda_tensors, dim=0)

                    extra_logs["lambda/global_mean"] = lambda_all.mean().item()
                    extra_logs["lambda/global_std"] = lambda_all.std(unbiased=False).item()
                    extra_logs["lambda/global_min"] = lambda_all.min().item()
                    extra_logs["lambda/global_max"] = lambda_all.max().item()

                # ------------------------------
                # global lambda logit stats
                # ------------------------------
                if len(lambda_logit_tensors) > 0:
                    lambda_logits_all = torch.cat(lambda_logit_tensors, dim=0)

                    extra_logs["lambda_logits/global_mean"] = lambda_logits_all.mean().item()
                    extra_logs["lambda_logits/global_std"] = lambda_logits_all.std(unbiased=False).item()
                    extra_logs["lambda_logits/global_min"] = lambda_logits_all.min().item()
                    extra_logs["lambda_logits/global_max"] = lambda_logits_all.max().item()


                # ------------------------------
                # global lambda offset stats
                # ------------------------------
                if len(lambda_offset_vals) > 0:
                    c_tensor = torch.tensor(lambda_offset_vals, dtype=torch.float32)

                    extra_logs["lambda_offset/global_mean"] = c_tensor.mean().item()
                    extra_logs["lambda_offset/global_std"] = c_tensor.std(unbiased=False).item()
                    extra_logs["lambda_offset/global_min"] = c_tensor.min().item()
                    extra_logs["lambda_offset/global_max"] = c_tensor.max().item()

                # ------------------------------
                # global sigmoid(wq+b) stats
                # ------------------------------
                if len(lambda_sigmoid_tensors) > 0:
                    sigmoid_all = torch.cat(lambda_sigmoid_tensors, dim=0)

                    extra_logs["lambda_sigmoid/global_mean"] = sigmoid_all.mean().item()
                    extra_logs["lambda_sigmoid/global_std"] = sigmoid_all.std(unbiased=False).item()
                    extra_logs["lambda_sigmoid/global_min"] = sigmoid_all.min().item()
                    extra_logs["lambda_sigmoid/global_max"] = sigmoid_all.max().item()    
                                    
            # Log m_exact and denominator stats for hyper_race_gl_mexact
            # --------------------------------------------------
            if attn_type == "hyper_race_gl_mexact":
                m_vals = []
                d_exact_vals = []
                d_race_vals = []

                for layer_idx, layer in enumerate(model.transformer_layers):
                    if hasattr(layer, "att"):
                        att = layer.att

                        if hasattr(att, "last_m_exact") and att.last_m_exact is not None:
                            m = att.last_m_exact.detach().cpu().float().reshape(-1)

                            extra_logs[f"m_exact/layer{layer_idx}_mean"] = m.mean().item()
                            extra_logs[f"m_exact/layer{layer_idx}_std"] = m.std().item()
                            extra_logs[f"m_exact/layer{layer_idx}_min"] = m.min().item()
                            extra_logs[f"m_exact/layer{layer_idx}_max"] = m.max().item()

                            m_vals.append(m.mean().item())

                        if hasattr(att, "last_d_exact_mean") and att.last_d_exact_mean is not None:
                            d_exact = float(att.last_d_exact_mean.detach().cpu())
                            d_race = float(att.last_d_race_mean.detach().cpu())

                            extra_logs[f"den/layer{layer_idx}_exact_mean"] = d_exact
                            extra_logs[f"den/layer{layer_idx}_race_mean"] = d_race

                            d_exact_vals.append(d_exact)
                            d_race_vals.append(d_race)

                if len(m_vals) > 0:
                    extra_logs["m_exact/global_mean"] = sum(m_vals) / len(m_vals)

                if len(d_exact_vals) > 0:
                    extra_logs["den/global_exact_mean"] = sum(d_exact_vals) / len(d_exact_vals)
                    extra_logs["den/global_race_mean"] = sum(d_race_vals) / len(d_race_vals)
            # --------------------------------------------------
            # New hyper_plus_race scalar-mixing logging
            # --------------------------------------------------
            if attn_type == "fixed_hyper_plus_race":
                a_vals = []
                b_vals = []

                for layer_idx, layer in enumerate(model.transformer_layers):
                    if hasattr(layer, "att"):
                        att = layer.att
                        if hasattr(att, "a") and hasattr(att, "b"):
                            a_val = att.a.detach().item()
                            b_val = att.b.detach().item()

                            extra_logs[f"mix/layer{layer_idx}_a"] = a_val
                            extra_logs[f"mix/layer{layer_idx}_b"] = b_val

                            # useful normalized view too
                            denom = abs(a_val) + abs(b_val) + 1e-8
                            extra_logs[f"mix/layer{layer_idx}_a_frac"] = abs(a_val) / denom
                            extra_logs[f"mix/layer{layer_idx}_b_frac"] = abs(b_val) / denom

                            a_vals.append(a_val)
                            b_vals.append(b_val)

                if len(a_vals) > 0:
                    extra_logs["mix/global_a_mean"] = sum(a_vals) / len(a_vals)
                    extra_logs["mix/global_b_mean"] = sum(b_vals) / len(b_vals)

                    denom = sum(abs(v) for v in a_vals) + sum(abs(v) for v in b_vals) + 1e-8
                    extra_logs["mix/global_a_frac"] = sum(abs(v) for v in a_vals) / denom
                    extra_logs["mix/global_b_frac"] = sum(abs(v) for v in b_vals) / denom

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_dataset_cfg(VISION_CONFIG, VISION_CONFIG["dataset_name"])  #oxford_pet , fashionmnist , flowers102
    train_loader, val_loader = get_data(VISION_CONFIG)
    num_epochs = 150

    experiments = [
        #("hyper_plus_race_nosample", 2)
       # ("hyper_plus_race_estimate", 2)
        #("true_hyper_plus_race",2)
#       ("hyper_plus_race", 2)
 #      ("hyper_race_gl_mexact", 2)
       #("hyper_race_mexact", 2),
       #("hyper_race_dependent_lambda", 2)
       #("hyper_lsh", 2),
       ("race", 2),
       ("hyper_race", 2),
       
        #("exact_flash", 1),
    ]
    dataset_name =VISION_CONFIG["dataset_name"]
    for attn_type, grad_accum in experiments:
        wandb_attn_name = (
            "hyper_race_mexact_lambda"
            if attn_type == "hyper_race_mexact" and VISION_CONFIG.get("mexact_lambda_learnable", False)
            else attn_type
        )
        run = wandb.init(
            project="RACE",
            name=f"{dataset_name}_{wandb_attn_name}_N{VISION_CONFIG['num_patches']}",
            config={
                "dataset": dataset_name,
                "attn_type": wandb_attn_name,
                "base_attn_type": attn_type,
                "N": VISION_CONFIG["num_patches"],
                "layers": VISION_CONFIG["transformer_units"],
                "heads": VISION_CONFIG["num_heads"],
                "d": VISION_CONFIG["embed_dim"],
                "batch_size": VISION_CONFIG["batch_size"],
                "lr": 6e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.1,
                #"mix_type": "global_scalars",
                #"hyper_plus_race_init_a": 1.0,
                #"hyper_plus_race_init_b": 1.0,
                "dropout": VISION_CONFIG["drop_rate"],
                "epochs": num_epochs,
                "grad_accum_steps": grad_accum,
                "K": VISION_CONFIG["K"],
                "L": VISION_CONFIG["L"],
                "M": VISION_CONFIG["M"],
                "hyper_num_bits": VISION_CONFIG["hyper_num_bits"],
                "hyper_block_size": VISION_CONFIG["hyper_block_size"],
                "hyper_min_seq_len": VISION_CONFIG["hyper_min_seq_len"],
                "hyper_neighbor_blocks": VISION_CONFIG["hyper_neighbor_blocks"],
                "gate_hidden_dim": VISION_CONFIG["gate_hidden_dim"],
                "gate_normalize": VISION_CONFIG["gate_normalize"],
                "hyper_global_tokens": VISION_CONFIG["hyper_global_tokens"],
                "hyper_local_window": VISION_CONFIG["hyper_local_window"],
                "hyper_exact_q_chunk_size": VISION_CONFIG["hyper_exact_q_chunk_size"],
                "mexact_eps": VISION_CONFIG["mexact_eps"],
                "mexact_lambda_learnable": VISION_CONFIG["mexact_lambda_learnable"],
                "mexact_lambda_init": VISION_CONFIG["mexact_lambda_init"],
                "mexact_dependent_lambda_offset": VISION_CONFIG["mexact_dependent_lambda_offset"],
                "mexact_dependent_lambda_init_target": VISION_CONFIG["mexact_dependent_lambda_init_target"],
                "mexact_dependent_lambda_use_bias": VISION_CONFIG["mexact_dependent_lambda_use_bias"],
                "mexact_dependent_lambda_w_init_std": VISION_CONFIG["mexact_dependent_lambda_w_init_std"],
                "mexact_dependent_lambda_detach_q": VISION_CONFIG["mexact_dependent_lambda_detach_q"],
                "mexact_dependent_lambda_offset_learnable": VISION_CONFIG["mexact_dependent_lambda_offset_learnable"],
                "mexact_dependent_lambda_offset_positive": VISION_CONFIG["mexact_dependent_lambda_offset_positive"],
                "mexact_dependent_lambda_min": VISION_CONFIG["mexact_dependent_lambda_min"],
                
            }
        )
        #wandb.define_metric("mix/*", step_metric="epoch")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("time/*", step_metric="epoch")
        wandb.define_metric("gates/*", step_metric="epoch")
        wandb.define_metric("val/acc", summary="max")
        wandb.define_metric("val/loss", summary="min")
        wandb.define_metric("c/*", step_metric="epoch")
        wandb.define_metric("m_exact/*", step_metric="epoch")
        wandb.define_metric("den/*", step_metric="epoch")  
        wandb.define_metric("lambda/*", step_metric="epoch")
        wandb.define_metric("lambda_hist/*", step_metric="epoch")
        wandb.define_metric("lambda_logits/*", step_metric="epoch")
        wandb.define_metric("lambda_logits_hist/*", step_metric="epoch")
        wandb.define_metric("lambda_offset/*", step_metric="epoch")
        wandb.define_metric("lambda_sigmoid/*", step_metric="epoch")
        wandb.define_metric("lambda_sigmoid_hist/*", step_metric="epoch")
    
        #print(f"\n=== Training {attn_type.upper()} ===")
        print(f"\n=== Training {wandb_attn_name.upper()} ===")
        torch.manual_seed(123)

        model = VisionTransformer(VISION_CONFIG, attn_type, device=device)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=6e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.1,
        )

        metrics = train_model_simple(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            cfg=VISION_CONFIG,
            attn_type=attn_type,
            grad_accum_steps=grad_accum
        )

        wandb.finish()

start_experiment()
