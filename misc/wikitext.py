"""
============================================================================
ELSAA CAUSAL — WikiText-103 Language Modeling (Perplexity)
============================================================================

Single-file paste-ready implementation. Trains a causal transformer on
WikiText-103, evaluates perplexity at the training length AND at longer
lengths to test length extrapolation.

Hyperparameters (from RACE paper):
    context length: 2048 (you asked to increase from 1024)
    layers: 8
    heads:  8
    d:      512
    batch:  16
    lr:     6e-4
    betas:  (0.9, 0.999)
    eps:    1e-8
    wd:     0.1
    dropout: 0.1
    epochs: 100

Attention types tested:
    - elsaa            : Causal sortLSH sparse + Causal RACE + m_sparse fusion
    - causal_race      : RACE branch only
    - exact            : SDPA causal exact attention
    - linear           : Causal linear attention (ELU+1 kernel)

Evaluation:
    - Train + validate at 2048
    - Test perplexity at: 2048, 4096, 8192, 16384

============================================================================
DATASET DOWNLOAD
============================================================================

WikiText-103 raw character version, ~180MB compressed.

```bash
cd RACE_Attention/data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
# Creates: wikitext-103/wiki.train.tokens, wiki.valid.tokens, wiki.test.tokens
```

Alternative (HuggingFace datasets, simpler):

```bash
pip install datasets
# The script will auto-download via datasets library if the local files are absent
```

You also need a tokenizer. We use a simple GPT-2 BPE tokenizer:

```bash
pip install tokenizers transformers
```

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
from torch.utils.data import Dataset, DataLoader
from torch.nn.attention import sdpa_kernel, SDPBackend
from tqdm import tqdm

# Optional: Triton FlashAttention for non-causal blocks
try:
    from flash_attn_triton import flash_attn_func as _flash_attn_triton
    _HAS_TRITON_FLASH = True
except ImportError:
    _flash_attn_triton = None
    _HAS_TRITON_FLASH = False

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
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q16, k16, v16, dropout_p=0.0, is_causal=causal, scale=scale)
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


def exact_attention_flash(query, key, value, scale=None, causal=False):
    """Triton FlashAttention path. Used only for non-causal."""
    B, H, Tq, D = query.shape
    Tk = key.shape[2]
    if scale is None:
        scale = D ** -0.5

    if (not _HAS_TRITON_FLASH) or query.device.type != "cuda":
        return exact_attention_sdpa(query, key, value, scale=scale, causal=causal)

    in_dtype = query.dtype
    q = query.permute(0, 2, 1, 3).to(torch.float16).contiguous()
    k = key.permute(0, 2, 1, 3).to(torch.float16).contiguous()
    v = value.permute(0, 2, 1, 3).to(torch.float16).contiguous()

    out_t, lse_padded = _flash_attn_triton(q, k, v, None, causal, scale)

    out = out_t.permute(0, 2, 1, 3).to(in_dtype).contiguous()
    lse = lse_padded[:, :, :Tq]
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
# SECTION 3: CAUSAL EXACT (baseline)
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
# SECTION 4: CAUSAL LINEAR ATTENTION (baseline)
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
# SECTION 5: CAUSAL RACE ATTENTION
# ============================================================================

class CausalRACEAttention(nn.Module):
    """Causal RACE Attention with chunked cumsum.

    Two call modes:
      A. forward(x, ...)                    — standalone (has own W_O)
      B. forward_core(Q, K, V, ...)         — ELSAA branch, no W_O
    """

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
# SECTION 6: CAUSAL SPARSE ATTENTION (HyperAttention recursive split)
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
        # Always use SDPA (Triton causal kernel has a known bug, and we keep it simple here)
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
# SECTION 7: CAUSAL ELSAA (FUSION) — shared QKV + separate W_O per branch
# ============================================================================

class CausalELSAAAttention(nn.Module):
    """Causal ELSAA.

    Shared Q/K/V across branches (ensures both branches approximate the SAME
    attention operation), but SEPARATE W_O per branch and POST-PROJECTION
    fusion. This is the cleaner architecture: each branch produces a complete
    attention output, then the fusion gates combine them.
    """

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

        # SHARED Q/K/V (one set of input projections for both branches)
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)

        # SEPARATE output projections per branch
        self.o_sparse = nn.Linear(d, d)
        self.o_race = nn.Linear(d, d)
        self.out_drop = nn.Dropout(drop)

        # Branches (their own QKV exists for standalone use; bypassed via forward_core)
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

        # Lambda (scalar or query-dependent)
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

        # Branch forwards (per-head outputs + denominator proxies)
        head_sparse, log_d_sparse = self.sparse.forward_core(Q, K, V, return_lse=True)
        head_race, d_race = self.race.forward_core(Q, K, V, return_den=True)

        # SEPARATE W_O per branch — each branch now produces a full output
        merged_sparse = head_sparse.transpose(1, 2).contiguous().view(B, T, H * Dk)
        merged_race = head_race.transpose(1, 2).contiguous().view(B, T, H * Dk)
        out_sparse = self.o_sparse(self.out_drop(merged_sparse))
        out_race = self.o_race(self.out_drop(merged_race))

        # Lambda (scalar or [B,T,1])
        lam = self._compute_lambda(x)

        # m_sparse in log-space
        log_d_sparse_det = log_d_sparse.detach().float()
        log_d_race_det = torch.log(d_race.detach().float().clamp_min(self.mexact_eps))
        if isinstance(lam, torch.Tensor) and lam.dim() > 0:
            log_lambda = torch.log(lam.float().clamp_min(self.mexact_eps))
        else:
            log_lambda = torch.log(lam.float().clamp_min(self.mexact_eps))
        log_eps = torch.full_like(log_d_sparse_det, math.log(self.mexact_eps))

        log_den = torch.logsumexp(
            torch.stack([log_d_sparse_det, log_lambda + log_d_race_det, log_eps], dim=0),
            dim=0,
        )
        m_sparse = torch.exp(log_d_sparse_det - log_den).to(out_sparse.dtype)  # [B,T,1]

        # Gates
        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)
        g_sparse = gates[..., 0:1]
        g_race = gates[..., 1:2]

        # Post-projection fusion
        out = g_sparse * m_sparse * out_sparse + g_race * out_race

        # Logging
        self.last_gates = gates.detach()
        self.last_m_sparse = m_sparse.detach()
        self.last_lambda = lam.detach() if isinstance(lam, torch.Tensor) else lam.detach()
        self.last_d_sparse_mean = torch.exp(log_d_sparse.detach().clamp(max=20.0)).mean()
        self.last_d_race_mean = d_race.detach().mean()

        return out


# ============================================================================
# SECTION 8: TRANSFORMER BLOCK (pre-norm, causal)
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
# SECTION 9: CAUSAL LANGUAGE MODEL (with positional embeddings)
# ============================================================================

class CausalLM(nn.Module):
    """Causal language model.

    Token embedding + learned positional embedding + causal transformer blocks +
    LM head tied to token embedding (standard GPT-2 style).

    Supports `extended_pos_size` for evaluation at longer lengths than training:
    when validating at lengths > train length, learned position embeddings are
    interpolated rather than truncated.
    """

    def __init__(self, cfg, attn_type, device="cpu"):
        super().__init__()
        self.cfg = cfg
        d = cfg["embed_dim"]
        self.vocab_size = cfg["vocab_size"]
        self.max_seq_len = cfg["max_seq_len"]

        self.tok_emb = nn.Embedding(self.vocab_size, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_seq_len, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        self.drop = nn.Dropout(cfg["drop_rate"])

        self.layers = nn.ModuleList([
            CausalTransformerBlock(cfg, attn_type, device=device)
            for _ in range(cfg["num_layers"])
        ])

        self.norm = nn.LayerNorm(d)
        # Tied output projection: weights shared with input embedding
        self.head = nn.Linear(d, self.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def _get_pos_emb(self, T):
        """Return position embeddings for length T.

        If T <= max_seq_len, return the first T positions.
        If T > max_seq_len, interpolate the position embedding to length T.
        """
        if T <= self.max_seq_len:
            return self.pos_emb[:, :T, :]
        # Linear interpolation of position embeddings for length extrapolation.
        pe = self.pos_emb.transpose(1, 2)  # [1, d, max_seq_len]
        pe = F.interpolate(pe, size=T, mode="linear", align_corners=False)
        return pe.transpose(1, 2)  # [1, T, d]

    def forward(self, x):
        # x: [B, T] token ids
        B, T = x.shape
        h = self.tok_emb(x)                    # [B, T, d]
        h = h + self._get_pos_emb(T)           # add positional embedding
        h = self.drop(h)

        for blk in self.layers:
            h = blk(h, mask=None)

        h = self.norm(h)
        return self.head(h)                    # [B, T, vocab_size]


# ============================================================================
# SECTION 10: WIKITEXT-103 DATA
# ============================================================================

def _load_or_download_wikitext(data_root):
    """Load WikiText-103. Uses HF datasets streaming to avoid RAM spike."""
    candidate_dir = os.path.join(data_root, "wikitext-103")
    train_path = os.path.join(candidate_dir, "wiki.train.tokens")
    valid_path = os.path.join(candidate_dir, "wiki.valid.tokens")
    test_path  = os.path.join(candidate_dir, "wiki.test.tokens")

    if all(os.path.exists(p) for p in (train_path, valid_path, test_path)):
        print(f"[wikitext] using local files at {candidate_dir}")
        with open(train_path, "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(valid_path, "r", encoding="utf-8") as f:
            valid_text = f.read()
        with open(test_path, "r", encoding="utf-8") as f:
            test_text = f.read()
        return train_text, valid_text, test_text

    print("[wikitext] loading via HuggingFace datasets ...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Join with newline but DON'T load all into one giant string
    # Return as list of strings instead — tokenizer will handle chunks
    train_text = "\n".join(x for x in ds["train"]["text"] if x.strip())
    valid_text = "\n".join(x for x in ds["validation"]["text"] if x.strip())
    test_text  = "\n".join(x for x in ds["test"]["text"] if x.strip())

    return train_text, valid_text, test_text

def _get_tokenizer():
    """Load a GPT-2 BPE tokenizer."""
    try:
        from transformers import GPT2TokenizerFast
        return GPT2TokenizerFast.from_pretrained("gpt2")
    except ImportError:
        raise ImportError(
            "Need transformers for GPT-2 tokenizer. Install with: pip install transformers"
        )


def _tokenize_and_cache(text, tokenizer, cache_path):
    """Tokenize in chunks to avoid RAM explosion."""
    if os.path.exists(cache_path):
        print(f"[wikitext] loading cached tokens from {cache_path}")
        return torch.load(cache_path)

    print(f"[wikitext] tokenizing ({len(text)/1e6:.1f}M chars) in chunks ...")

    # Split into chunks of ~1M chars to avoid RAM spike
    chunk_size = 1_000_000
    all_ids = []

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    for i, chunk in enumerate(tqdm(chunks, desc="  tokenizing")):
        ids = tokenizer.encode(chunk)
        all_ids.extend(ids)
        if i % 50 == 0:
            # Periodic progress print
            print(f"  chunk {i}/{len(chunks)}, total tokens so far: {len(all_ids)/1e6:.1f}M")

    ids = torch.tensor(all_ids, dtype=torch.long)
    torch.save(ids, cache_path)
    print(f"[wikitext] cached {len(ids)/1e6:.1f}M tokens to {cache_path}")
    return ids


class TokenChunkDataset(Dataset):
    """Random-offset chunks of length seq_len+1 (input + shifted target)."""

    def __init__(self, tokens, seq_len, num_chunks=None, deterministic=False):
        super().__init__()
        self.tokens = tokens
        self.seq_len = seq_len
        self.deterministic = deterministic
        if num_chunks is None:
            self.num_chunks = (len(tokens) - 1) // seq_len
        else:
            self.num_chunks = num_chunks

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        if self.deterministic:
            start = idx * self.seq_len
            start = min(start, len(self.tokens) - self.seq_len - 1)
        else:
            start = random.randint(0, len(self.tokens) - self.seq_len - 2)
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def _tokenize_hf_dataset_split(split_data, tokenizer, cache_path):
    """Stream-tokenize a HF dataset split without building a giant string."""
    if os.path.exists(cache_path):
        print(f"[wikitext] loading cached tokens from {cache_path}")
        return torch.load(cache_path)

    print(f"[wikitext] stream-tokenizing {len(split_data)} articles ...")
    all_ids = []

    for i, row in enumerate(tqdm(split_data, desc="  tokenizing")):
        text = row["text"]
        if not text.strip():
            continue
        ids = tokenizer.encode(text)
        all_ids.extend(ids)

    ids = torch.tensor(all_ids, dtype=torch.long)
    torch.save(ids, cache_path)
    print(f"[wikitext] cached {len(ids)/1e6:.1f}M tokens to {cache_path}")
    return ids


def build_wikitext_data(cfg, data_root="./data"):
    """Build WikiText-103 dataloaders — memory safe."""
    from datasets import load_dataset

    cache_dir = os.path.join(data_root, "wikitext-103", "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = _get_tokenizer()

    train_cache = os.path.join(cache_dir, "train.pt")
    valid_cache = os.path.join(cache_dir, "valid.pt")
    test_cache  = os.path.join(cache_dir, "test.pt")

    # Check if all caches exist — skip download entirely
    if not all(os.path.exists(p) for p in (train_cache, valid_cache, test_cache)):
        print("[wikitext] downloading dataset ...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")

        train_tokens = _tokenize_hf_dataset_split(ds["train"],      tokenizer, train_cache)
        valid_tokens = _tokenize_hf_dataset_split(ds["validation"], tokenizer, valid_cache)
        test_tokens  = _tokenize_hf_dataset_split(ds["test"],       tokenizer, test_cache)
    else:
        print("[wikitext] all caches found, loading ...")
        train_tokens = torch.load(train_cache)
        valid_tokens = torch.load(valid_cache)
        test_tokens  = torch.load(test_cache)

    cfg["vocab_size"] = tokenizer.vocab_size

    seq_len    = cfg["train_seq_len"]
    batch_size = cfg["batch_size"]

    train_ds = TokenChunkDataset(train_tokens, seq_len, deterministic=False)
    valid_ds = TokenChunkDataset(valid_tokens, seq_len, deterministic=True)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=(DEVICE == "cuda"),
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(DEVICE == "cuda"),
    )

    print(f"[wikitext] train: {len(train_tokens)/1e6:.1f}M tokens  "
          f"valid: {len(valid_tokens)/1e6:.1f}M tokens  "
          f"test: {len(test_tokens)/1e6:.1f}M tokens")
    print(f"[wikitext] train chunks: {len(train_ds)}  "
          f"valid chunks: {len(valid_ds)}  "
          f"seq_len: {seq_len}  batch: {batch_size}")

    return train_dl, valid_dl, test_tokens
# ============================================================================
# SECTION 11: PERPLEXITY EVALUATION (at multiple sequence lengths)
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(model, tokens, seq_len, batch_size=4, max_batches=None):
    """Evaluate causal LM perplexity on a token stream at the given seq_len.

    Uses non-overlapping chunks of length seq_len. Loss is averaged across all
    predicted positions in all chunks.
    """
    model.eval()
    # Build non-overlapping chunks
    n_chunks = (len(tokens) - 1) // seq_len
    if n_chunks <= 0:
        return float("nan")

    total_loss = 0.0
    total_tokens = 0

    n_batches = math.ceil(n_chunks / batch_size)
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    pbar = tqdm(range(n_batches), desc=f"  eval@{seq_len}", leave=False)
    for batch_idx in pbar:
        chunks_x, chunks_y = [], []
        for k in range(batch_size):
            i = batch_idx * batch_size + k
            if i >= n_chunks:
                break
            start = i * seq_len
            end = start + seq_len + 1
            chunk = tokens[start:end]
            chunks_x.append(chunk[:-1])
            chunks_y.append(chunk[1:])
        if not chunks_x:
            break
        x = torch.stack(chunks_x, 0).to(DEVICE)
        y = torch.stack(chunks_y, 0).to(DEVICE)

        logits = model(x)              # [B, T, V]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    return math.exp(min(avg_loss, 50))  # cap to avoid overflow


# ============================================================================
# SECTION 12: TRAINING
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
    """ELSAA-specific stats for wandb logging."""
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
    return stats


def train_one_run(model, train_dl, valid_dl, test_tokens, optimizer, cfg, attn_type,
                  num_epochs, grad_accum_steps, eval_lengths, log_to_wandb=True):
    steps_per_epoch = len(train_dl)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates = num_epochs * updates_per_epoch
    warmup_updates = max(1, int(0.02 * total_updates))
    scheduler = LinearWarmupLR(optimizer, warmup_updates, total_updates)

    global_update = 0
    best_val_ppl = float("inf")

    for epoch in range(1, num_epochs + 1):
        # --- TRAIN ---
        model.train()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        pbar = tqdm(train_dl, desc=f"Ep{epoch} train", leave=False)
        for x, y in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            (loss / grad_accum_steps).backward()
            accum += 1
            running_loss += loss.item()
            n_batches += 1

            if accum == grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_update += 1

            avg_loss = running_loss / max(1, n_batches)
            pbar.set_postfix({"loss": avg_loss, "ppl": math.exp(min(avg_loss, 20))})

        if accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        tr_loss = running_loss / max(1, n_batches)
        tr_ppl = math.exp(min(tr_loss, 20))

        # --- VALIDATION at training length ---
        model.eval()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        val_loss_sum = 0.0
        val_token_count = 0
        attn_stats_acc = defaultdict(list)
        with torch.no_grad():
            for x, y in tqdm(valid_dl, desc=f"Ep{epoch} val", leave=False):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    reduction="sum",
                )
                val_loss_sum += loss.item()
                val_token_count += y.numel()
                s = collect_attn_stats(model, attn_type)
                for k, v in s.items():
                    attn_stats_acc[k].append(v)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        val_time = time.time() - t1
        va_loss = val_loss_sum / max(1, val_token_count)
        va_ppl = math.exp(min(va_loss, 20))
        best_val_ppl = min(best_val_ppl, va_ppl)
        cur_lr = scheduler.get_last_lr()[0]

        log = {
            "epoch": epoch,
            "train/loss": tr_loss, "train/ppl": tr_ppl,
            "val/loss": va_loss, "val/ppl": va_ppl,
            "val/best_ppl": best_val_ppl,
            "lr": cur_lr,
            "time/train_sec": train_time, "time/val_sec": val_time,
        }
        for k, vlist in attn_stats_acc.items():
            log[k] = float(np.mean(vlist))

        if log_to_wandb and HAS_WANDB:
            wandb.log(log, step=epoch)

        print(f"Ep{epoch:3d} | tr_loss {tr_loss:.4f} ppl {tr_ppl:.2f} ({train_time:.1f}s) "
              f"| va_loss {va_loss:.4f} ppl {va_ppl:.2f} ({val_time:.1f}s) "
              f"| best {best_val_ppl:.2f} | lr {cur_lr:.2e}")

    # --- FINAL LENGTH-EXTRAPOLATION EVALUATION ---
    print(f"\n[{attn_type}] Final length-extrapolation evaluation on TEST set ...")
    extrapolation_results = {}
    for L in eval_lengths:
        eval_bs = max(1, cfg.get("eval_batch_size", 4))
        # Reduce batch size for very long sequences to fit in memory
        if L >= 8192:
            eval_bs = max(1, eval_bs // 2)
        if L >= 16384:
            eval_bs = 1
        ppl = evaluate_perplexity(model, test_tokens, seq_len=L, batch_size=eval_bs)
        extrapolation_results[L] = ppl
        print(f"  test ppl @ {L:>6d} tokens: {ppl:.3f}")
        if log_to_wandb and HAS_WANDB:
            wandb.log({f"test_ppl/len_{L}": ppl}, step=num_epochs)

    return best_val_ppl, extrapolation_results


# ============================================================================
# SECTION 13: EXPERIMENT RUNNER
# ============================================================================

DEFAULT_CFG = {
    # Transformer (RACE paper hyperparameters)
    "embed_dim": 512,
    "num_heads": 8,
    "mlp_dim": 2048,           # 4 * embed_dim
    "num_layers": 8,
    "drop_rate": 0.1,
    "qkv_bias": False,
    # Sequence
    "train_seq_len": 2048,     # train at 2048 (you asked to increase from 1024)
    "max_seq_len": 2048,       # position embedding table size — interpolated for longer
    "eval_batch_size": 4,
    # Sparse branch (sortLSH recursive)
    "hyper_num_bits": 5,
    "hyper_block_size": 64,
    "hyper_min_seq_len": 256,
    # RACE branch
    "race_num_bits": 4,
    "race_num_tables": 4,
    "race_chunk_size": 64,
    # Linear baseline
    "linear_chunk_size": 128,
    # ELSAA
    "mexact_eps": 1e-6,
    "mexact_lambda_init": 1.0,
    "lambda_offset_init": 0.3,
    "lambda_init_target": 0.8,
    "gate_hidden_dim": 128,
}


def run_experiment(attn_type, num_epochs=100, batch_size=16,
                   grad_accum_steps=1, lr=6e-4, weight_decay=0.1,
                   wandb_project="ELSAA_WikiText103",
                   eval_lengths=(2048, 4096, 8192, 16384),
                   data_root="./data", **overrides):
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    cfg["batch_size"] = batch_size

    train_dl, valid_dl, test_tokens = build_wikitext_data(cfg, data_root=data_root)

    print(f"\n=== WikiText-103 | Method: {attn_type} | "
          f"train_len={cfg['train_seq_len']} ===")
    model = CausalLM(cfg, attn_type, device=DEVICE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params/1e6:.1f}M params, layers={cfg['num_layers']}, "
          f"d={cfg['embed_dim']}, h={cfg['num_heads']}, vocab={cfg['vocab_size']}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    log_wandb = HAS_WANDB
    if log_wandb:
        run_name = f"wikitext103_{attn_type}_L{cfg['train_seq_len']}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={**cfg, "attn_type": attn_type,
                    "lr": lr, "weight_decay": weight_decay,
                    "epochs": num_epochs, "grad_accum_steps": grad_accum_steps,
                    "n_params": n_params,
                    "eval_lengths": list(eval_lengths)},
            reinit=True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test_ppl/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("time/*", step_metric="epoch")
        wandb.define_metric("gates/*", step_metric="epoch")
        wandb.define_metric("m_sparse/*", step_metric="epoch")
        wandb.define_metric("lambda/*", step_metric="epoch")
        wandb.define_metric("val/ppl", summary="min")
        wandb.define_metric("val/loss", summary="min")

    try:
        best_ppl, extrap = train_one_run(
            model, train_dl, valid_dl, test_tokens, optimizer, cfg, attn_type,
            num_epochs=num_epochs, grad_accum_steps=grad_accum_steps,
            eval_lengths=eval_lengths, log_to_wandb=log_wandb,
        )
        print(f"\n[done] {attn_type}: best valid ppl = {best_ppl:.3f}")
        print(f"[done] {attn_type}: extrapolation results:")
        for L, ppl in extrap.items():
            print(f"   {L:>6d} tokens -> ppl = {ppl:.3f}")
    finally:
        if log_wandb:
            wandb.finish()


# ============================================================================
# SECTION 14: EXPERIMENT LIST
# ============================================================================

SHARED_CFG = dict(
    epochs        = 40,                # RACE paper setting
    lr            = 6e-4,
    weight_decay  = 0.1,
    batch_size    = 16,                 # RACE paper setting
    embed_dim     = 512,                # RACE paper setting
    num_layers    = 8,                  # RACE paper setting
    num_heads     = 8,                  # RACE paper setting
    mlp_dim       = 2048,
    train_seq_len = 1024,               # increased from RACE paper's 1024
    max_seq_len   = 1024,
    eval_lengths  = (1024,2048, 4096, 8192, 16384),
    hyper_min_seq_len = 256,
    data_root     = "./data",
    wandb_project = "ELSAA_WikiText103",
)

EXPERIMENTS = [
    #dict(method="exact",        grad_accum_steps=1),
    #dict(method="elsaa",        grad_accum_steps=1),
    dict(method="causal_race",  grad_accum_steps=1),
    #dict(method="linear",       grad_accum_steps=1),
    # dict(method="causal_sparse",grad_accum_steps=1),
    # dict(method="elsaa_lambda", grad_accum_steps=1),
]


def main():
    for i, exp in enumerate(EXPERIMENTS, 1):
        cfg = dict(SHARED_CFG)
        cfg.update(exp)

        method            = cfg.pop("method")
        epochs            = cfg.pop("epochs")
        batch_size        = cfg.pop("batch_size")
        grad_accum_steps  = cfg.pop("grad_accum_steps")
        lr                = cfg.pop("lr")
        weight_decay      = cfg.pop("weight_decay")
        wandb_project     = cfg.pop("wandb_project")
        eval_lengths      = cfg.pop("eval_lengths")
        data_root         = cfg.pop("data_root")

        print(f"\n{'='*70}")
        print(f"  Experiment {i}/{len(EXPERIMENTS)}: method={method} "
              f"bs={batch_size} accum={grad_accum_steps} epochs={epochs}")
        print(f"{'='*70}\n")

        run_experiment(
            attn_type=method,
            num_epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            weight_decay=weight_decay,
            wandb_project=wandb_project,
            eval_lengths=eval_lengths,
            data_root=data_root,
            **cfg,
        )


if __name__ == "__main__":
    main()