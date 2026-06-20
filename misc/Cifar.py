"""
============================================================================
ELSAA CAUSAL VISION — CIFAR-10 and CIFAR-100 Classification
============================================================================

Single-file implementation for CIFAR-10 and CIFAR-100 with causal attention.

Task:
    CIFAR-10:  10 classes, 32×32 RGB, 50K train / 10K test
    CIFAR-100: 100 classes, 32×32 RGB, 50K train / 10K test
    
    Images are patchified in raster order with causal masking.
    Classification uses the LAST patch's representation.

Attention types supported:
    - elsaa            : Causal sortLSH sparse + Causal RACE + m_sparse fusion
    - elsaa_lambda     : ELSAA with query-dependent lambda
    - causal_sparse    : Sparse branch only
    - causal_race      : RACE branch only
    - exact            : SDPA causal exact attention
    - linear           : Causal linear attention (ELU+1)
    - performer        : Causal Performer (FAVOR+ random Fourier features)

Patch configurations:
    - patch_size=1  → 1024 tokens (32×32) [pixel-level, very long]
    - patch_size=2  → 256 tokens  (16×16) [medium length]
    - patch_size=4  → 64 tokens   (8×8)   [short length]

Dataset download:
    Datasets auto-download via torchvision.datasets

============================================================================
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import time
import random
import itertools
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[warn] wandb not installed; logging disabled")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")


# ============================================================================
# SECTION 1: LSH UTILITIES
# ============================================================================

def _gray_code_order(num_bits: int, device):
    if num_bits == 1:
        return torch.tensor([0, 1], device=device, dtype=torch.long)

    def rec(n):
        if n == 1:
            return torch.tensor([0, 1], device=device, dtype=torch.long)
        a = rec(n - 1)
        return torch.cat([a, torch.flip(a, dims=[0]) + (1 << (n - 1))], dim=0)

    return rec(num_bits)


class AngularLSHGray(nn.Module):
    """Hard angular LSH with Gray-code bucket ordering."""

    def __init__(self, num_bits: int, dim: int, device="cpu"):
        super().__init__()
        self.num_bits = num_bits
        self.R = 1 << num_bits

        proj_dir = torch.randn(dim, num_bits, device=device)
        perm = _gray_code_order(num_bits, device=device)
        enc_vec = (2 ** torch.arange(num_bits, device=device, dtype=torch.long)).view(
            *([1] * 2), num_bits
        )

        self.register_buffer("proj_dir", proj_dir, persistent=False)
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("enc_vec", enc_vec, persistent=False)

    def hash(self, mat: torch.Tensor):
        proj = torch.einsum("...td,dr->...tr", mat, self.proj_dir)
        bits = (proj > 0).to(torch.long)
        bin_ids = (bits * self.enc_vec).sum(dim=-1)
        return self.perm[bin_ids]


def indexing(x, indices, chunk_size=-1):
    """Pad indices to multiple of chunk_size and gather. x:[B,H,T,D]."""
    if chunk_size > 0:
        n_new = math.ceil(indices.shape[2] / chunk_size) * chunk_size
        if n_new != indices.shape[2]:
            pad_len = n_new - indices.shape[2]
            pad = indices[:, :, :1].expand(-1, -1, pad_len)
            indices = torch.cat([indices, pad], dim=2)
    return x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))


# ============================================================================
# SECTION 2: EXACT ATTENTION HELPERS
# ============================================================================

def exact_attention_sdpa(query, key, value, scale=None, causal=False):
    """Exact attention via SDPA. Returns (out, lse)."""
    B, H, Tq, D = query.shape
    Tk = key.shape[2]
    if scale is None:
        scale = D ** -0.5

    if query.device.type == "cuda":
        q16, k16, v16 = [t.to(torch.float16) for t in (query, key, value)]
        import warnings
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        for backend in backends:
            try:
                with warnings.catch_warnings(), sdpa_kernel(backend):
                    warnings.simplefilter("ignore")
                    out = F.scaled_dot_product_attention(
                        q16, k16, v16, dropout_p=0.0, is_causal=causal, scale=scale)
                break
            except RuntimeError:
                continue
        out = out.to(query.dtype)
    else:
        out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=causal, scale=scale)

    with torch.no_grad():
        logits = torch.einsum("bhqd,bhkd->bhqk", query.float(), key.float()) * scale
        if causal:
            mask = torch.ones(Tq, Tk, device=query.device, dtype=torch.bool).tril()
            logits = logits.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(logits, dim=-1).to(query.dtype)
    return out, lse


def add_self_attentions_lse(attn1, lse1, attn2, lse2):
    """Combine two attention outputs with their LSEs."""
    if lse1.dim() == 4:
        lse1 = lse1.squeeze(-1)
    if lse2.dim() == 4:
        lse2 = lse2.squeeze(-1)

    m = torch.maximum(lse1, lse2)
    w1 = torch.exp(lse1 - m)
    w2 = torch.exp(lse2 - m)
    denom = w1 + w2

    out = (w1.unsqueeze(-1) * attn1 + w2.unsqueeze(-1) * attn2) / denom.unsqueeze(-1).clamp_min(1e-12)
    new_lse = m + torch.log(denom.clamp_min(1e-12))
    return out, new_lse


# ============================================================================
# SECTION 3: CAUSAL EXACT ATTENTION
# ============================================================================

class CausalExactAttention(nn.Module):
    """Standard causal exact attention via SDPA/FlashAttention."""

    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()

        out, _ = exact_attention_sdpa(Q, K, V, causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.drop(out)
        return self.o(out)


# ============================================================================
# SECTION 4: CAUSAL LINEAR ATTENTION
# ============================================================================

class CausalLinearAttention(nn.Module):
    """ELU+1 kernel causal linear attention with cumulative state."""

    def __init__(self, d, h, drop, qkv_bias=False, chunk_size=128):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.chunk_size = chunk_size
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.eps = 1e-6

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1.0

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        K = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)

        phiQ = self._phi(Q)
        phiK = self._phi(K)

        D = self.dk
        state_kv = torch.zeros(B, self.h, D, D, device=x.device, dtype=Q.dtype)
        state_k = torch.zeros(B, self.h, D, device=x.device, dtype=Q.dtype)

        out_chunks = []
        for cs in range(0, T, self.chunk_size):
            ce = min(cs + self.chunk_size, T)
            pK = phiK[:, :, cs:ce, :]
            pQ = phiQ[:, :, cs:ce, :]
            vC = V[:, :, cs:ce, :]

            kv_outer = torch.einsum("bhtd,bhte->bhtde", pK, vC)
            kv_local = torch.cumsum(kv_outer, dim=2)
            k_local = torch.cumsum(pK, dim=2)

            kv_at_t = state_kv.unsqueeze(2) + kv_local
            k_at_t = state_k.unsqueeze(2) + k_local

            num = torch.einsum("bhtd,bhtde->bhte", pQ, kv_at_t)
            den = torch.einsum("bhtd,bhtd->bht", pQ, k_at_t).unsqueeze(-1) + self.eps
            out_chunks.append(num / den)

            state_kv = state_kv + kv_local[:, :, -1, :, :]
            state_k = state_k + k_local[:, :, -1, :]

        out = torch.cat(out_chunks, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.drop(out)
        return self.o(out)


# ============================================================================
# SECTION 4b: CAUSAL PERFORMER ATTENTION
# ============================================================================

class CausalPerformerAttention(nn.Module):
    """Performer attention using FAVOR+ (random Fourier features)."""

    def __init__(self, d, h, drop, num_features=None, qkv_bias=False, 
                 chunk_size=128, ortho_scaling=False, device="cpu"):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.num_features = num_features if num_features is not None else self.dk
        self.chunk_size = chunk_size
        self.ortho_scaling = ortho_scaling
        
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.eps = 1e-6
        
        self._create_projection_matrix(device)
    
    def _create_projection_matrix(self, device):
        """Create random orthogonal projection matrix for FAVOR+."""
        num_blocks = math.ceil(self.num_features / self.dk)
        
        blocks = []
        for _ in range(num_blocks):
            unstructured = torch.randn(self.dk, self.dk, device=device)
            q, _ = torch.linalg.qr(unstructured)
            blocks.append(q)
        
        projection = torch.cat(blocks, dim=1)[:, :self.num_features]
        
        if self.ortho_scaling:
            multiplier = torch.randn(self.num_features, device=device).norm()
            projection = projection * (multiplier / math.sqrt(self.dk))
        
        self.register_buffer("projection_matrix", projection, persistent=False)
    
    def _phi(self, x):
        """FAVOR+ kernel feature map."""
        projection = torch.einsum("bhtd,dm->bhtm", x, self.projection_matrix)
        x_squared = (x ** 2).sum(dim=-1, keepdim=True) / 2.0
        features = torch.exp(projection - x_squared - 0.5 * math.log(self.num_features))
        return features
    
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        
        Q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        K = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        
        phiQ = self._phi(Q)
        phiK = self._phi(K)
        
        m = self.num_features
        state_kv = torch.zeros(B, self.h, m, self.dk, device=x.device, dtype=Q.dtype)
        state_k = torch.zeros(B, self.h, m, device=x.device, dtype=Q.dtype)
        
        out_chunks = []
        for cs in range(0, T, self.chunk_size):
            ce = min(cs + self.chunk_size, T)
            pK = phiK[:, :, cs:ce, :]
            pQ = phiQ[:, :, cs:ce, :]
            vC = V[:, :, cs:ce, :]
            
            kv_outer = torch.einsum("bhtm,bhtd->bhtmd", pK, vC)
            kv_local = torch.cumsum(kv_outer, dim=2)
            k_local = torch.cumsum(pK, dim=2)
            
            kv_at_t = state_kv.unsqueeze(2) + kv_local
            k_at_t = state_k.unsqueeze(2) + k_local
            
            num = torch.einsum("bhtm,bhtmd->bhtd", pQ, kv_at_t)
            den = torch.einsum("bhtm,bhtm->bht", pQ, k_at_t).unsqueeze(-1) + self.eps
            out_chunks.append(num / den)
            
            state_kv = state_kv + kv_local[:, :, -1, :, :]
            state_k = state_k + k_local[:, :, -1, :]
        
        out = torch.cat(out_chunks, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.drop(out)
        return self.o(out)


# ============================================================================
# SECTION 5: CAUSAL RACE ATTENTION
# ============================================================================

class CausalRACEAttention(nn.Module):
    """Causal RACE Attention with chunked cumsum."""

    def __init__(self, d, h, drop, num_bits=4, num_tables=4, beta_init=1.0,
                 chunk_size=64, qkv_bias=False, device="cpu",
                 chunk_group_size=None):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.R = 1 << num_bits
        self.C = chunk_size
        self.chunk_group_size = chunk_group_size

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        planes = torch.randn(num_tables, num_bits, self.dk, device=device)
        self.register_buffer("planes", planes, persistent=False)
        corners = torch.tensor(
            list(itertools.product([-1.0, 1.0], repeat=num_bits)),
            device=device, dtype=torch.float32,
        )
        self.register_buffer("corners", corners, persistent=False)
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta_init, dtype=torch.float32)))

    def _phi(self, x):
        """x:[B,H,T,dk] -> [B,H,L,T,R]"""
        proj = torch.einsum("bhtd,lpd->bhtlp", x, self.planes)
        tan = torch.tanh(proj)
        beta = self.log_beta.exp().clamp(1e-2, 20.0)
        logits = beta * torch.einsum("bhtlp,rp->bhtlr", tan, self.corners)
        probs = F.softmax(logits, dim=-1)
        return probs.permute(0, 1, 3, 2, 4).contiguous()

    def forward_core(self, Q, K, V, return_den=False):
        B, H, T, dk = Q.shape
        L, R, C = self.num_tables, self.R, self.C

        phiQ = self._phi(Q)
        phiK = self._phi(K)

        pad = (-T) % C
        if pad:
            phiQ = F.pad(phiQ, (0, 0, 0, pad))
            phiK = F.pad(phiK, (0, 0, 0, pad))
            V_p = F.pad(V, (0, 0, 0, pad))
        else:
            V_p = V
        Tp = T + pad
        n_chunks = Tp // C

        phiQ_c = phiQ.view(B, H, L, n_chunks, C, R)
        phiK_c = phiK.view(B, H, L, n_chunks, C, R)
        V_c = V_p.view(B, H, n_chunks, C, dk)

        chunk_A = phiK_c.sum(dim=4)
        chunk_B = torch.einsum("bhlncr,bhncd->bhlnrd", phiK_c, V_c)

        state_A = F.pad(torch.cumsum(chunk_A, dim=3)[:, :, :, :-1, :], (0, 0, 1, 0))
        state_B = F.pad(torch.cumsum(chunk_B, dim=3)[:, :, :, :-1, :, :], (0, 0, 0, 0, 1, 0))

        inter_num = torch.einsum("bhlncr,bhlnrd->bhlncd", phiQ_c, state_B)
        inter_den = torch.einsum("bhlncr,bhlnr->bhlnc", phiQ_c, state_A)

        if self.chunk_group_size is None or self.chunk_group_size >= n_chunks:
            M_avg = torch.einsum("bhlncr,bhlnsr->bhncs", phiQ_c, phiK_c) / L
            tri = torch.tril(torch.ones(C, C, device=Q.device, dtype=M_avg.dtype))
            M_avg = M_avg * tri
            intra_num = torch.einsum("bhncs,bhnsd->bhncd", M_avg, V_c)
            intra_den = M_avg.sum(dim=-1)
        else:
            g = self.chunk_group_size
            intra_num_parts, intra_den_parts = [], []
            tri = torch.tril(torch.ones(C, C, device=Q.device, dtype=phiQ.dtype))
            for gs in range(0, n_chunks, g):
                ge = min(gs + g, n_chunks)
                pQ_g = phiQ_c[:, :, :, gs:ge, :, :]
                pK_g = phiK_c[:, :, :, gs:ge, :, :]
                V_g = V_c[:, :, gs:ge, :, :]
                M_g = torch.einsum("bhlncr,bhlnsr->bhncs", pQ_g, pK_g) / L
                M_g = M_g * tri
                intra_num_parts.append(torch.einsum("bhncs,bhnsd->bhncd", M_g, V_g))
                intra_den_parts.append(M_g.sum(dim=-1))
            intra_num = torch.cat(intra_num_parts, dim=2)
            intra_den = torch.cat(intra_den_parts, dim=2)

        num = inter_num.mean(dim=2) + intra_num
        den = inter_den.mean(dim=2) + intra_den

        head_out = (num / den.unsqueeze(-1).clamp_min(1e-6)).view(B, H, Tp, dk)
        head_out = head_out[:, :, :T, :]

        d_token = den.view(B, H, Tp)[:, :, :T].mean(dim=1).unsqueeze(-1).clamp_min(1e-6)

        return (head_out, d_token) if return_den else head_out

    def forward(self, x, mask=None, return_den=False):
        B, T, _ = x.shape
        H, dk = self.h, self.dk
        Q = self.q(x).view(B, T, H, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, H, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, dk).transpose(1, 2).contiguous()

        if return_den:
            head_out, d_token = self.forward_core(Q, K, V, return_den=True)
        else:
            head_out = self.forward_core(Q, K, V, return_den=False)

        out = head_out.transpose(1, 2).contiguous().view(B, T, H * dk)
        out = self.drop(self.o(out))

        return (out, d_token) if return_den else out


# ============================================================================
# SECTION 6: CAUSAL SPARSE ATTENTION
# ============================================================================

class CausalHyperSparseAttention(nn.Module):
    """Causal sortLSH sparse attention via HyperAttention's recursive split."""

    def __init__(self, d, h, drop, num_bits=5, block_size=64, min_seq_len=256,
                 qkv_bias=False, device="cpu"):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.scale = self.dk ** -0.5
        self.num_bits = num_bits
        self.block_size = block_size
        self.min_seq_len = min_seq_len

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.dk, device=device)

    def _exact_attention(self, q, k, v, causal=False):
        return exact_attention_sdpa(q, k, v, scale=self.scale, causal=causal)

    def _noncausal_sortlsh(self, q, k, v):
        B, H, Tq, D = q.shape
        Tk = k.shape[2]

        q_hash = self.lsh.hash(q)
        k_hash = self.lsh.hash(k)
        _, q_sort_idx = torch.sort(q_hash, dim=2, stable=True)
        _, k_sort_idx = torch.sort(k_hash, dim=2, stable=True)
        q_sort_inv = torch.argsort(q_sort_idx, dim=2, stable=True)

        bs = self.block_size
        q_sorted = indexing(q, q_sort_idx, bs)
        k_sorted = indexing(k, k_sort_idx, bs)
        v_sorted = indexing(v, k_sort_idx, bs)

        num_blocks = k_sorted.shape[2] // bs
        if num_blocks == 0:
            return self._exact_attention(q, k, v, causal=False)

        q_block_size = q_sorted.shape[2] // num_blocks

        q_b = q_sorted.reshape(B * H * num_blocks, 1, q_block_size, D)
        k_b = k_sorted.reshape(B * H * num_blocks, 1, bs, D)
        v_b = v_sorted.reshape(B * H * num_blocks, 1, bs, D)

        out_b, lse_b = self._exact_attention(q_b, k_b, v_b, causal=False)
        out_blocked = out_b.reshape(B, H, num_blocks * q_block_size, D)
        lse_blocked = lse_b.reshape(B, H, num_blocks * q_block_size)

        out_blocked = out_blocked[:, :, :Tq, :]
        lse_blocked = lse_blocked[:, :, :Tq]

        idx = q_sort_inv.unsqueeze(-1).expand(-1, -1, -1, D)
        out_unsorted = out_blocked.gather(2, idx)
        lse_unsorted = lse_blocked.gather(2, q_sort_inv)
        return out_unsorted, lse_unsorted

    def _causal_forward(self, q, k, v):
        B, H, N, D = q.shape

        if N <= self.min_seq_len:
            return self._exact_attention(q, k, v, causal=True)

        n_orig = N
        if N % 2:
            q = F.pad(q, (0, 0, 0, 1))
            k = F.pad(k, (0, 0, 0, 1))
            v = F.pad(v, (0, 0, 0, 1))
            N = N + 1

        half = N // 2

        q_past, q_future = q[:, :, :half, :], q[:, :, half:, :]
        k_past, k_future = k[:, :, :half, :], k[:, :, half:, :]
        v_past, v_future = v[:, :, :half, :], v[:, :, half:, :]

        out_top, lse_top = self._causal_forward(q_past, k_past, v_past)
        out_bot_diag, lse_bot_diag = self._causal_forward(q_future, k_future, v_future)

        out_off, lse_off = self._noncausal_sortlsh(q_future, k_past, v_past)

        out_bot, lse_bot = add_self_attentions_lse(
            out_bot_diag, lse_bot_diag, out_off, lse_off
        )

        out = torch.cat([out_top, out_bot], dim=2)
        lse = torch.cat([lse_top, lse_bot], dim=2)

        if n_orig != N:
            out = out[:, :, :n_orig, :]
            lse = lse[:, :, :n_orig]

        return out, lse

    def forward_core(self, Q, K, V, return_lse=False):
        head_out, lse_heads = self._causal_forward(Q, K, V)

        if return_lse:
            with torch.no_grad():
                log_d_token = (torch.logsumexp(lse_heads.float(), dim=1)
                               - math.log(self.h)).to(head_out.dtype).unsqueeze(-1)
            return head_out, log_d_token
        return head_out

    def forward(self, x, mask=None, return_lse=False):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()

        if return_lse:
            head_out, log_d_token = self.forward_core(Q, K, V, return_lse=True)
        else:
            head_out = self.forward_core(Q, K, V, return_lse=False)

        out = head_out.transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.drop(out)
        out = self.o(out)

        return (out, log_d_token) if return_lse else out


# ============================================================================
# SECTION 7: CAUSAL ELSAA
# ============================================================================

class CausalELSAAAttention(nn.Module):
    """Causal ELSAA with shared Q/K/V/O."""

    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg.get("qkv_bias", False)
        gate_hidden = cfg.get("gate_hidden_dim", 64)

        assert d % h == 0
        self.d = d
        self.h = h
        self.dk = d // h

        self.mexact_eps = cfg.get("mexact_eps", 1e-6)
        self.lambda_dep = bool(cfg.get("lambda_dependent", False))

        # SHARED Q/K/V/O
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.out_drop = nn.Dropout(drop)

        # Branches
        self.sparse = CausalHyperSparseAttention(
            d=d, h=h, drop=drop,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 64),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            qkv_bias=qkv_bias, device=device,
        )
        self.race = CausalRACEAttention(
            d=d, h=h, drop=drop,
            num_bits=cfg.get("race_num_bits", 4),
            num_tables=cfg.get("race_num_tables", 4),
            chunk_size=cfg.get("race_chunk_size", 64),
            chunk_group_size=cfg.get("race_chunk_group_size", None),
            qkv_bias=qkv_bias, device=device,
        )

        # Lambda
        if self.lambda_dep:
            offset_init = float(cfg.get("lambda_offset_init", 0.3))
            self.lambda_offset_raw = nn.Parameter(torch.tensor(offset_init, dtype=torch.float32))
            self.lambda_w = nn.Parameter(torch.empty(d, dtype=torch.float32))
            nn.init.normal_(self.lambda_w, mean=0.0, std=1e-3)

            init_target = float(cfg.get("lambda_init_target", 0.8))
            init_prob = init_target - max(offset_init, 0.0)
            init_prob = min(max(init_prob, 1e-4), 1.0 - 1e-4)
            init_bias = math.log(init_prob / (1.0 - init_prob))
            self.lambda_bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
        else:
            init = float(cfg.get("mexact_lambda_init", 1.0))
            self.log_lambda = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))

        # Gate MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        # Logging cache
        self.last_gates = None
        self.last_m_sparse = None
        self.last_lambda = None
        self.last_d_sparse_mean = None
        self.last_d_race_mean = None

    def _compute_lambda(self, x):
        if self.lambda_dep:
            with torch.no_grad():
                q_for_lambda = self.q(x).float()
            lambda_logits = q_for_lambda @ self.lambda_w.float() + self.lambda_bias.float()
            lambda_sigmoid = torch.sigmoid(lambda_logits)
            c_raw = self.lambda_offset_raw.float()
            c_forward = c_raw.clamp_min(0.0)
            c = c_raw + (c_forward - c_raw).detach()
            lam = (c + lambda_sigmoid).clamp_min(self.mexact_eps).unsqueeze(-1)
            return lam
        else:
            return self.log_lambda.exp().clamp_min(self.mexact_eps)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        H, Dk = self.h, self.dk

        # SHARED Q/K/V
        Q = self.q(x).view(B, T, H, Dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, H, Dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, Dk).transpose(1, 2).contiguous()

        # Branch forwards
        head_sparse, log_d_sparse = self.sparse.forward_core(Q, K, V, return_lse=True)
        head_race, d_race = self.race.forward_core(Q, K, V, return_den=True)

        # Lambda
        lam = self._compute_lambda(x)

        # m_sparse in log-space
        log_d_sparse_det = log_d_sparse.detach().float()
        log_d_race_det = torch.log(d_race.detach().float().clamp_min(self.mexact_eps))
        log_lambda = torch.log(lam.float().clamp_min(self.mexact_eps)) if isinstance(lam, torch.Tensor) \
                     else torch.log(lam.float().clamp_min(self.mexact_eps))
        log_eps = torch.full_like(log_d_sparse_det, math.log(self.mexact_eps))

        log_den = torch.logsumexp(
            torch.stack([log_d_sparse_det, log_lambda + log_d_race_det, log_eps], dim=0),
            dim=0,
        )
        m_sparse = torch.exp(log_d_sparse_det - log_den).to(head_sparse.dtype)

        # Gates
        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)
        g_sparse = gates[..., 0:1]
        g_race = gates[..., 1:2]

        # Broadcast [B,T,1] -> [B,1,T,1]
        m_sparse_h = m_sparse.unsqueeze(1)
        g_sparse_h = g_sparse.unsqueeze(1).to(head_sparse.dtype)
        g_race_h = g_race.unsqueeze(1).to(head_race.dtype)

        # Fuse per-head BEFORE shared W_O
        head_fused = g_sparse_h * m_sparse_h * head_sparse + g_race_h * (1 - m_sparse_h) * head_race

        # Shared output projection
        fused = head_fused.transpose(1, 2).contiguous().view(B, T, H * Dk)
        out = self.o(self.out_drop(fused))

        # Logging cache
        self.last_gates = gates.detach()
        self.last_m_sparse = m_sparse.detach()
        self.last_lambda = lam.detach() if isinstance(lam, torch.Tensor) else lam
        self.last_d_sparse_mean = torch.exp(log_d_sparse.detach().clamp(max=20.0)).mean()
        self.last_d_race_mean = d_race.detach().mean()

        return out


# ============================================================================
# SECTION 8: TRANSFORMER BLOCK
# ============================================================================

class CausalTransformerBlock(nn.Module):
    def __init__(self, cfg, attn_type, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )

        if attn_type == "exact":
            self.att = CausalExactAttention(d, cfg["num_heads"], drop, cfg.get("qkv_bias", False))
        elif attn_type == "linear":
            self.att = CausalLinearAttention(
                d, cfg["num_heads"], drop, cfg.get("qkv_bias", False),
                chunk_size=cfg.get("linear_chunk_size", 128))
        elif attn_type == "performer":
            self.att = CausalPerformerAttention(
                d, cfg["num_heads"], drop,
                num_features=cfg.get("performer_num_features", None),
                qkv_bias=cfg.get("qkv_bias", False),
                chunk_size=cfg.get("performer_chunk_size", 128),
                ortho_scaling=cfg.get("performer_ortho_scaling", False),
                device=device)
        elif attn_type == "causal_race":
            self.att = CausalRACEAttention(
                d, cfg["num_heads"], drop,
                num_bits=cfg.get("race_num_bits", 4),
                num_tables=cfg.get("race_num_tables", 4),
                chunk_size=cfg.get("race_chunk_size", 64),
                qkv_bias=cfg.get("qkv_bias", False), device=device)
        elif attn_type == "causal_sparse":
            self.att = CausalHyperSparseAttention(
                d, cfg["num_heads"], drop,
                num_bits=cfg.get("hyper_num_bits", 5),
                block_size=cfg.get("hyper_block_size", 64),
                min_seq_len=cfg.get("hyper_min_seq_len", 256),
                qkv_bias=cfg.get("qkv_bias", False), device=device)
        elif attn_type in ("elsaa", "elsaa_lambda"):
            cfg_local = dict(cfg)
            cfg_local["lambda_dependent"] = (attn_type == "elsaa_lambda")
            self.att = CausalELSAAAttention(cfg_local, device=device)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

    def forward(self, x, mask=None):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask=mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


# ============================================================================
# SECTION 9: PATCH EMBEDDING + CAUSAL VISION TRANSFORMER
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert image to a sequence of patch embeddings in raster order."""

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


class CausalVisionTransformer(nn.Module):
    """Causal Vision Transformer for classification."""

    def __init__(self, cfg, attn_type, device="cpu"):
        super().__init__()
        self.cfg = cfg
        d = cfg["embed_dim"]

        self.patch_embed = PatchEmbedding(cfg)

        num_patches = cfg["num_patches"]
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.drop = nn.Dropout(cfg["drop_rate"])

        self.layers = nn.ModuleList([
            CausalTransformerBlock(cfg, attn_type, device=device)
            for _ in range(cfg["num_layers"])
        ])

        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["num_classes"])

    def forward(self, x):
        h = self.patch_embed(x)
        h = h + self.pos_emb
        h = self.drop(h)

        for blk in self.layers:
            h = blk(h, mask=None)

        h = self.norm(h)
        last = h[:, -1, :]
        return self.head(last)


# ============================================================================
# SECTION 10: CIFAR DATA
# ============================================================================

def build_cifar_data(cfg, dataset_name="cifar10", data_root="./data"):
    """Build CIFAR-10 or CIFAR-100 dataloaders."""
    
    batch_size = cfg["batch_size"]
    
    # Standard CIFAR normalization
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    # Training augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Select dataset
    if dataset_name == "cifar10":
        train_ds = CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        test_ds = CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        train_ds = CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
        test_ds = CIFAR100(root=data_root, train=False, download=True, transform=test_transform)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=(DEVICE == "cuda"),
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(DEVICE == "cuda"),
    )
    
    print(f"[{dataset_name}] train: {len(train_ds)}, test: {len(test_ds)}, classes: {num_classes}")
    return train_dl, test_dl, num_classes


# ============================================================================
# SECTION 11: TRAINING
# ============================================================================

class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        out = []
        for base in self.base_lrs:
            if step <= self.warmup_steps:
                out.append(base * (step / self.warmup_steps))
            else:
                p = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                out.append(base * (1.0 - p))
        return out


def collect_attn_stats(model, attn_type):
    """Collect ELSAA-specific stats for wandb logging."""
    stats = {}
    if attn_type not in ("elsaa", "elsaa_lambda"):
        return stats
    for li, blk in enumerate(model.layers):
        att = getattr(blk, "att", None)
        if att is None:
            continue
        if hasattr(att, "last_gates") and att.last_gates is not None:
            g = att.last_gates
            stats[f"gates/layer{li}_sparse"] = g[..., 0].mean().item()
            stats[f"gates/layer{li}_race"] = g[..., 1].mean().item()
        if hasattr(att, "last_m_sparse") and att.last_m_sparse is not None:
            stats[f"m_sparse/layer{li}_mean"] = att.last_m_sparse.mean().item()
        if hasattr(att, "last_lambda") and att.last_lambda is not None:
            if isinstance(att.last_lambda, torch.Tensor):
                stats[f"lambda/layer{li}_mean"] = att.last_lambda.float().mean().item()
        if hasattr(att, "log_lambda"):
            stats[f"lambda_param/layer{li}"] = float(att.log_lambda.detach().exp().item())
        if hasattr(att, "last_d_sparse_mean") and att.last_d_sparse_mean is not None:
            stats[f"denominators/layer{li}_sparse"] = float(att.last_d_sparse_mean.item())
        if hasattr(att, "last_d_race_mean") and att.last_d_race_mean is not None:
            stats[f"denominators/layer{li}_race"] = float(att.last_d_race_mean.item())
    return stats


def train_one_run(model, train_dl, test_dl, optimizer, cfg, attn_type,
                  num_epochs, grad_accum_steps, log_to_wandb=True):
    steps_per_epoch = len(train_dl)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates = num_epochs * updates_per_epoch
    warmup_updates = max(1, int(0.05 * total_updates))
    scheduler = LinearWarmupLR(optimizer, warmup_updates, total_updates)

    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # TRAIN
        model.train()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        pbar = tqdm(train_dl, desc=f"Ep{epoch} train", leave=False)
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            (loss / grad_accum_steps).backward()
            accum += 1

            preds = logits.argmax(dim=-1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            if accum == grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

            pbar.set_postfix({
                "loss": running_loss / max(1, len(pbar)),
                "acc": running_correct / max(1, running_total),
            })

        if accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        tr_loss = running_loss / max(1, len(train_dl))
        tr_acc = running_correct / max(1, running_total)

        # TEST
        model.eval()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        attn_stats_acc = defaultdict(list)
        
        with torch.no_grad():
            for images, labels in tqdm(test_dl, desc=f"Ep{epoch} test", leave=False):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                test_loss += loss.item()
                preds = logits.argmax(dim=-1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
                
                # Collect ELSAA stats from test batches
                s = collect_attn_stats(model, attn_type)
                for k, v in s.items():
                    attn_stats_acc[k].append(v)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        test_time = time.time() - t1
        te_loss = test_loss / max(1, len(test_dl))
        te_acc = test_correct / max(1, test_total)
        best_test_acc = max(best_test_acc, te_acc)
        cur_lr = scheduler.get_last_lr()[0]

        log = {
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "test/loss": te_loss, "test/acc": te_acc,
            "test/best_acc": best_test_acc,
            "lr": cur_lr,
            "time/train_sec": train_time, "time/test_sec": test_time,
        }
        
        # Add ELSAA stats (averaged over test batches)
        for k, vlist in attn_stats_acc.items():
            log[k] = float(np.mean(vlist))

        if log_to_wandb and HAS_WANDB:
            wandb.log(log, step=epoch)

        print(f"Ep{epoch:3d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} ({train_time:.1f}s) "
              f"| te_loss {te_loss:.4f} te_acc {te_acc:.4f} ({test_time:.1f}s) "
              f"| best {best_test_acc:.4f} | lr {cur_lr:.2e}")

    return best_test_acc


# ============================================================================
# SECTION 12: EXPERIMENT RUNNER
# ============================================================================

DEFAULT_CFG = {
    # Image config
    "img_size": 32,
    "patch_size": 1,        # 2 → 256 tokens, 4 → 64 tokens, 1 → 1024 tokens
    "num_channels": 3,
    "num_patches": 1024,    # Will be recomputed from img_size / patch_size
    "num_classes": 10,      # Will be set based on dataset
    # Transformer
    "embed_dim": 256,
    "num_heads": 4,
    "mlp_dim": 1024,
    "num_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
    # Sparse branch
    "hyper_num_bits": 5,
    "hyper_block_size": 32,
    "hyper_min_seq_len": 64,
    # RACE branch
    "race_num_bits": 3,
    "race_num_tables": 4,
    "race_chunk_size": 32,
    # Linear baseline
    "linear_chunk_size": 64,
    # Performer baseline
    "performer_num_features": None,
    "performer_chunk_size": 64,
    "performer_ortho_scaling": False,
    # ELSAA
    "mexact_eps": 1e-6,
    "mexact_lambda_init": 1.0,
    "lambda_offset_init": 0.3,
    "lambda_init_target": 0.8,
    "gate_hidden_dim": 128,
}


def run_experiment(attn_type, dataset_name="cifar10", num_epochs=100, 
                   batch_size=64, grad_accum_steps=2, lr=3e-4, weight_decay=0.05,
                   wandb_project="ELSAA_CIFAR", **overrides):
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    cfg["batch_size"] = batch_size

    # Recompute num_patches
    G = cfg["img_size"] // cfg["patch_size"]
    assert G * cfg["patch_size"] == cfg["img_size"], "img_size must divide patch_size"
    cfg["num_patches"] = G * G

    train_dl, test_dl, num_classes = build_cifar_data(cfg, dataset_name=dataset_name)
    cfg["num_classes"] = num_classes

    print(f"\n=== {dataset_name.upper()} | Method: {attn_type} | "
          f"patch={cfg['patch_size']} tokens={cfg['num_patches']} ===")
    model = CausalVisionTransformer(cfg, attn_type, device=DEVICE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params/1e6:.1f}M params, layers={cfg['num_layers']}, "
          f"d={cfg['embed_dim']}, h={cfg['num_heads']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_wandb = HAS_WANDB
    if log_wandb:
        run_name = f"{dataset_name}_{attn_type}_p{cfg['patch_size']}_T{cfg['num_patches']}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={**cfg, "attn_type": attn_type, "dataset": dataset_name,
                    "lr": lr, "weight_decay": weight_decay,
                    "epochs": num_epochs, "grad_accum_steps": grad_accum_steps,
                    "n_params": n_params},
            reinit=True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("time/*", step_metric="epoch")
        wandb.define_metric("gates/*", step_metric="epoch")
        wandb.define_metric("m_sparse/*", step_metric="epoch")
        wandb.define_metric("lambda/*", step_metric="epoch")
        wandb.define_metric("lambda_param/*", step_metric="epoch")
        wandb.define_metric("denominators/*", step_metric="epoch")

    try:
        best_acc = train_one_run(
            model, train_dl, test_dl, optimizer, cfg, attn_type,
            num_epochs=num_epochs, grad_accum_steps=grad_accum_steps,
            log_to_wandb=log_wandb,
        )
        print(f"\n[done] {attn_type} on {dataset_name}: best test acc = {best_acc:.4f}")
        return best_acc
    finally:
        if log_wandb:
            wandb.finish()


# ============================================================================
# SECTION 13: EXPERIMENT CONFIGURATIONS
# ============================================================================

# Run both CIFAR-10 and CIFAR-100 with the same attention method
EXPERIMENTS = [
    # CIFAR-10 experiments
    #dict(dataset="cifar10", method="exact",         batch_size=128, epochs=100),
    #dict(dataset="cifar10", method="linear",        batch_size=32, epochs=100),
    dict(dataset="cifar10", method="performer",     batch_size=32, epochs=100),
    #dict(dataset="cifar10", method="causal_race",   batch_size=128, epochs=100),
    #dict(dataset="cifar10", method="causal_sparse", batch_size=128, epochs=100),
    #dict(dataset="cifar10", method="elsaa",         batch_size=64, epochs=100),
    
    # CIFAR-100 experiments (harder, 100 classes)
    
    #dict(dataset="cifar100", method="exact",         batch_size=128, epochs=100),
    #dict(dataset="cifar100", method="linear",        batch_size=32, epochs=100),
    dict(dataset="cifar100", method="performer",     batch_size=32, epochs=100),
    #dict(dataset="cifar100", method="causal_race",   batch_size=128, epochs=100),
    #dict(dataset="cifar100", method="causal_sparse", batch_size=128, epochs=100),
    #dict(dataset="cifar100", method="elsaa",         batch_size=64, epochs=100),
]


def main():
    """Run all experiments."""
    
    results = {}
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        dataset = exp.pop("dataset")
        method = exp.pop("method")
        epochs = exp.pop("epochs")
        batch_size = exp.pop("batch_size")
        grad_accum_steps = exp.get("grad_accum_steps", 2)
        
        print(f"\n{'='*70}")
        print(f"  Experiment {i}/{len(EXPERIMENTS)}: {dataset.upper()} + {method}")
        print(f"{'='*70}\n")
        
        try:
            best_acc = run_experiment(
                attn_type=method,
                dataset_name=dataset,
                num_epochs=epochs,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                lr=3e-4,
                weight_decay=0.05,
                **exp,
            )
            
            key = f"{dataset}_{method}"
            results[key] = best_acc
            
        except Exception as e:
            print(f"\n[ERROR] {dataset}_{method} failed: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for dataset in ["cifar10", "cifar100"]:
        print(f"\n{dataset.upper()}:")
        for method in ["exact", "linear", "performer", "causal_race", "causal_sparse", "elsaa"]:
            key = f"{dataset}_{method}"
            if key in results:
                print(f"  {method:15s}: {results[key]:.4f}")
            else:
                print(f"  {method:15s}: FAILED")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()