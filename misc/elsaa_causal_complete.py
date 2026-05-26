"""
============================================================================
ELSAA CAUSAL — Complete Single-File Implementation
============================================================================

Three experiments in one runnable script:
    1. Causal arXiv classification @ 16K / 32K / 64K
    2. MQAR (Multi-Query Associative Recall) diagnostic @ 1K -> 32K
    3. LRA-style causal Text Retrieval pair classification

Attention types supported:
    - elsaa            : Causal sortLSH sparse + Causal RACE + m_sparse fusion
    - elsaa_lambda     : ELSAA with query-dependent lambda
    - causal_sparse    : Sparse branch only (HyperAttention recursive split)
    - causal_race      : RACE branch only (Algorithm 2 chunked cumsum)
    - exact            : SDPA causal exact attention
    - linear           : Causal linear attention (ELU+1)

Inspired by and built on:
    - HyperAttention (Han et al., ICLR 2024)
    - RACE Attention (Joshi/Chowdhury et al., arXiv 2025)
    - User's existing ELSAA non-causal implementation

Usage:
    python elsaa_causal_complete.py --task arxiv --method elsaa --length 16384
    python elsaa_causal_complete.py --task mqar  --method elsaa --length 8192
    python elsaa_causal_complete.py --task lra   --method elsaa --length 16384

============================================================================
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import time
import random
import argparse
import itertools
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.attention import sdpa_kernel, SDPBackend
from tqdm import tqdm
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

torch.set_float32_matmul_precision('high')




# ============================================================================
# SECTION 1: LSH UTILITIES (from HyperAttention)
# ============================================================================

class AngularLSH(nn.Module):
    """Hard angular LSH with Gray-code bucket ordering (HyperAttention style).
    Hashes vectors of shape [..., T, D] into bucket IDs of shape [..., T].
    """
    def __init__(self, num_projs: int, dim, rng=None):
        super().__init__()
        self.num_projs = num_projs
        if num_projs > 0:
            proj_dir = torch.randn(dim + (num_projs,), generator=rng)
            perm = self._gray_code(num_projs)
            enc_vec = (2 ** torch.arange(num_projs)).view(1, 1, 1, -1)
            self.register_buffer('proj_dir', proj_dir, persistent=False)
            self.register_buffer('perm', perm, persistent=False)
            self.register_buffer('enc_vec', enc_vec, persistent=False)

    @staticmethod
    def _gray_code(n: int):
        if n == 1:
            return torch.tensor([0, 1], dtype=torch.long)
        a = AngularLSH._gray_code(n - 1)
        return torch.cat([a, torch.flip(a, dims=[0]) + (1 << (n - 1))], 0)

    def hash(self, mat):
        if self.num_projs <= 0:
            return torch.zeros(mat.shape[:-1], device=mat.device, dtype=torch.int32)
        mask = torch.einsum('...nd,...dr->...nr', mat, self.proj_dir)
        bits = (mask > 0).long()
        bin_ids = (bits * self.enc_vec).sum(-1)
        return self.perm[bin_ids]


def indexing(x, indices, chunk_size=-1):
    """Gather along token dim. x:[B,H,T,D], indices:[B,H,S] -> [B,H,S,D]."""
    if chunk_size > 0:
        # round up to multiple of chunk_size with pad
        n_new = math.ceil(indices.shape[2] / chunk_size) * chunk_size
        if n_new != indices.shape[2]:
            pad_len = n_new - indices.shape[2]
            pad = indices[:, :, :1].expand(-1, -1, pad_len)
            indices = torch.cat([indices, pad], dim=2)
    return x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))


# ============================================================================
# SECTION 2: EXACT ATTENTION HELPERS
# ============================================================================

#def exact_attention_sdpa(query, key, value, scale=None, causal=False):
#    """Exact attention via PyTorch SDPA. Returns (out, lse).
#    query/key/value: [B, H, T, D]
#    """
#    B, H, Tq, D = query.shape
#    Tk = key.shape[2]
#    if scale is None:
#        scale = D ** -0.5

#    if query.device.type == "cuda":
#        q16, k16, v16 = [t.to(torch.float16) for t in (query, key, value)]
#        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
#            out = F.scaled_dot_product_attention(
#                q16, k16, v16, dropout_p=0.0, is_causal=causal, scale=scale)
#        out = out.to(query.dtype)
#    else:
#        out = F.scaled_dot_product_attention(
#            query, key, value, dropout_p=0.0, is_causal=causal, scale=scale)

#    # Compute LSE separately (need it for fusion)
#    with torch.no_grad():
#       logits = torch.einsum('bhqd,bhkd->bhqk', query.float(), key.float()) * scale
#        if causal:
#            mask = torch.ones(Tq, Tk, device=query.device, dtype=torch.bool).tril()
#            logits = logits.masked_fill(~mask, float('-inf'))
#        lse = torch.logsumexp(logits, dim=-1).to(query.dtype)  # [B,H,T]
#    return out, lse
def exact_attention_sdpa(query, key, value, scale=None, causal=False):
    """CPU/fallback exact attention. Single QK matmul, derive out and lse together."""
    B, H, Tq, D = query.shape
    Tk = key.shape[2]
    if scale is None:
        scale = D ** -0.5

    if query.device.type == "cuda":
        # GPU path: use SDPA for output, single fp32 matmul for LSE
        q16, k16, v16 = [t.to(torch.float16) for t in (query, key, value)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q16, k16, v16, dropout_p=0.0, is_causal=causal, scale=scale)
        out = out.to(query.dtype)
    else:
        out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=causal, scale=scale)

    with torch.no_grad():
        logits = torch.einsum('bhqd,bhkd->bhqk', query.float(), key.float()) * scale
        if causal:
            mask = torch.ones(Tq, Tk, device=query.device, dtype=torch.bool).tril()
            logits = logits.masked_fill(~mask, float('-inf'))
        lse = torch.logsumexp(logits, dim=-1).to(query.dtype)
    return out, lse

#def exact_attention_flash(query, key, value, scale=None, causal=False):
    """
    Triton FlashAttention drop-in for exact_attention_sdpa.
    LSE comes directly from the kernel — no redundant fp32 QK^T pass.

    query, key, value: [B, H, T, D]  (this script's layout)
    Returns:
        out: [B, H, Tq, D]   (same dtype as input)
        lse: [B, H, Tq]      (fp32; = log sum_j exp(scale * <q_i, k_j>),
                              already causal-masked when causal=True)
    """
#    B, H, Tq, D = query.shape
#    Tk = key.shape[2]
#    if scale is None:
#        scale = D ** -0.5

    # Fall back if Triton not available or we're on CPU
#    if (not _HAS_TRITON_FLASH) or query.device.type != "cuda":
#        return exact_attention_sdpa(query, key, value, scale=scale, causal=causal)

    # Triton kernel expects (B, T, H, D), fp16/bf16, last dim contiguous
#    in_dtype = query.dtype
#    q = query.transpose(1, 2).contiguous().to(torch.float16)   # (B, Tq, H, D)
#    k = key.transpose(1, 2).contiguous().to(torch.float16)     # (B, Tk, H, D)
#    v = value.transpose(1, 2).contiguous().to(torch.float16)   # (B, Tk, H, D)

    # Call autograd-aware Triton function. Returns (out, lse).
#    out_t, lse_padded = _flash_attn_triton(q, k, v, None, causal, scale)
    # out_t:      (B, Tq, H, D) fp16
    # lse_padded: (B, H, ceil(Tq/128)*128) fp32

#    out = out_t.transpose(1, 2).contiguous().to(in_dtype)      # (B, H, Tq, D)
#    lse = lse_padded[:, :, :Tq]                                # (B, H, Tq) fp32
#    return out, lse
def exact_attention_flash(query, key, value, scale=None, causal=False):
    B, H, Tq, D = query.shape
    Tk = key.shape[2]
    if scale is None:
        scale = D ** -0.5

    if (not _HAS_TRITON_FLASH) or query.device.type != "cuda":
        return exact_attention_sdpa(query, key, value, scale=scale, causal=causal)

    in_dtype = query.dtype

    # One allocation each: permute+cast+copy fused into copy_
    q = torch.empty(B, Tq, H, D, dtype=torch.float16, device=query.device)
    k = torch.empty(B, Tk, H, D, dtype=torch.float16, device=key.device)
    v = torch.empty(B, Tk, H, D, dtype=torch.float16, device=value.device)
    q.copy_(query.permute(0, 2, 1, 3))
    k.copy_(key.permute(0, 2, 1, 3))
    v.copy_(value.permute(0, 2, 1, 3))

    out_t, lse_padded = _flash_attn_triton(q, k, v, None, causal, scale)

    out = torch.empty(B, H, Tq, D, dtype=in_dtype, device=query.device)
    out.copy_(out_t.permute(0, 2, 1, 3))
    lse = lse_padded[:, :, :Tq]
    return out, lse
def add_self_attentions_lse(attn1, lse1, attn2, lse2):
    """Combine two attention outputs with their LSEs (HyperAttention helper).
    attn:  [B, H, T, D]
    lse:   [B, H, T] or [B, H, T, 1]
    Returns combined (attn, lse).
    """
    if lse1.dim() == 4:
        lse1 = lse1.squeeze(-1)
    if lse2.dim() == 4:
        lse2 = lse2.squeeze(-1)

    # Numerically stable combination
    m = torch.maximum(lse1, lse2)
    w1 = torch.exp(lse1 - m)
    w2 = torch.exp(lse2 - m)
    denom = w1 + w2  # [B,H,T]

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
        self.scale = self.dk ** -0.5
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

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q = Q * keep; K = K * keep; V = V * keep

        if Q.device.type == "cuda":
            q16, k16, v16 = [t.to(torch.float16) for t in (Q, K, V)]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    q16, k16, v16, dropout_p=0.0, is_causal=True)
            out = out.to(Q.dtype)
        else:
            out = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=0.0, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.drop(out)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


# ============================================================================
# SECTION 4: CAUSAL LINEAR ATTENTION (baseline)
# ============================================================================

class CausalLinearAttention(nn.Module):
    """ELU+1 kernel causal linear attention with cumulative state."""
    def __init__(self, d, h, drop, qkv_bias=False, chunk_size=256):
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

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            K = K * keep; V = V * keep

        phiQ = self._phi(Q)  # [B,H,T,D]
        phiK = self._phi(K)

        # Chunked causal cumulative state
        D = self.dk
        state_kv = torch.zeros(B, self.h, D, D, device=x.device, dtype=Q.dtype)
        state_k = torch.zeros(B, self.h, D, device=x.device, dtype=Q.dtype)

        out_chunks = []
        for cs in range(0, T, self.chunk_size):
            ce = min(cs + self.chunk_size, T)
            pK = phiK[:, :, cs:ce, :]
            pQ = phiQ[:, :, cs:ce, :]
            vC = V[:, :, cs:ce, :]

            # Local cumulative within chunk
            # kv_local[t] = sum_{s<=t in chunk} pK[s] outer vC[s]
            kv_outer = torch.einsum('bhtd,bhte->bhtde', pK, vC)  # [B,H,c,D,D]
            kv_local = torch.cumsum(kv_outer, dim=2)
            k_local = torch.cumsum(pK, dim=2)

            # Full state at position t = state + local
            kv_at_t = state_kv.unsqueeze(2) + kv_local
            k_at_t = state_k.unsqueeze(2) + k_local

            num = torch.einsum('bhtd,bhtde->bhte', pQ, kv_at_t)
            den = torch.einsum('bhtd,bhtd->bht', pQ, k_at_t).unsqueeze(-1) + self.eps
            out_chunks.append(num / den)

            state_kv = state_kv + kv_local[:, :, -1, :, :]
            state_k = state_k + k_local[:, :, -1, :]

        out = torch.cat(out_chunks, dim=2)  # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.drop(out)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


# ============================================================================
# SECTION 5: CAUSAL RACE ATTENTION (Algorithm 2 — chunked cumsum)
# ============================================================================
try:
    from race_kernel import race_chunkwise_triton_forward
    _HAS_TRITON_RACE = True
except ImportError:
    _HAS_TRITON_RACE = False


class _RaceChunkwiseTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phiQ, phiK, V, state_A, state_B, C):
        ctx.save_for_backward(phiQ, phiK, V, state_A, state_B)
        ctx.C = C
        with torch.no_grad():
            num, den = race_chunkwise_triton_forward(phiQ, phiK, V, state_A, state_B, C)
        return num, den

    @staticmethod
    def backward(ctx, dnum, dden):
        phiQ, phiK, V, state_A, state_B = ctx.saved_tensors
        C = ctx.C

        with torch.enable_grad():   # ← re-enable grad so ops build a graph
            phiQ_g = phiQ.detach().requires_grad_(True)
            phiK_g = phiK.detach().requires_grad_(True)
            V_g    = V.detach().requires_grad_(True)
            sA_g   = state_A.detach().requires_grad_(True)
            sB_g   = state_B.detach().requires_grad_(True)

            B, H, L, T, R = phiQ_g.shape
            D = V_g.shape[-1]
            n = T // C

            phiQ_c = phiQ_g.view(B, H, L, n, C, R)
            phiK_c = phiK_g.view(B, H, L, n, C, R)
            V_c    = V_g.view(B, H, n, C, D)

            inter_num = torch.einsum('bhlncr,bhlnrd->bhlncd', phiQ_c, sB_g)
            inter_den = torch.einsum('bhlncr,bhlnr->bhlnc',   phiQ_c, sA_g)
            M = torch.einsum('bhlncr,bhlnsr->bhlncs', phiQ_c, phiK_c)
            tri = torch.tril(torch.ones(C, C, device=phiQ.device, dtype=M.dtype))
            M = M * tri
            intra_num = torch.einsum('bhlncs,bhnsd->bhlncd', M, V_c)
            intra_den = M.sum(dim=-1)

            num = (inter_num + intra_num).sum(dim=2).view(B, H, T, D)
            den = (inter_den + intra_den).sum(dim=2).view(B, H, T)

            grads = torch.autograd.grad(   # ← use .grad, not .backward
                [num, den],
                [phiQ_g, phiK_g, V_g, sA_g, sB_g],
                grad_outputs=[dnum, dden],
                allow_unused=True,
            )

        # Replace None (unused grads) with zero tensors
        fixed = [
            g if g is not None else torch.zeros_like(t)
            for g, t in zip(grads, [phiQ, phiK, V, state_A, state_B])
        ]
        return (*fixed, None)   # last None is for the C argument
    
class CausalRACEAttention(nn.Module):
    """
    Causal RACE Attention — chunkwise reformulation.

    Mathematically identical to Algorithm 2 of the RACE paper, but the
    per-chunk outer-product cumsum [B,H,L,C,R,dk] is replaced by:
        - inter-chunk: exclusive cumsum over per-chunk reductions
                       (state_A: [B,H,L,n,R], state_B: [B,H,L,n,R,dk])
        - intra-chunk: causal-shaped matmul on R-dim features
                       (M = phiQ @ phiK^T, masked tril)
    Result: no Python loop, no [B,H,L,C,R,dk] tensor.

    For very long contexts where M = [B,H,n,C,C] gets too big, set
    chunk_group_size to process chunks in groups (still vectorized within
    each group). None = process all chunks at once.
    """
    def __init__(self, d, h, drop, num_bits=4, num_tables=4, beta_init=1.0,
                 chunk_size=256, qkv_bias=False, device="cpu",
                 chunk_group_size=64):
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
        self.register_buffer('planes', planes, persistent=False)
        corners = torch.tensor(
            list(itertools.product([-1.0, 1.0], repeat=num_bits)),
            device=device, dtype=torch.float32,
        )
        self.register_buffer('corners', corners, persistent=False)
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta_init, dtype=torch.float32)))
        self.last_den = None

    def _phi(self, x):
        """x:[B,H,T,dk] -> [B,H,L,T,R] (softmax over R buckets per table)."""
        proj = torch.einsum('bhtd,lpd->bhtlp', x, self.planes)
        tan = torch.tanh(proj)
        beta = self.log_beta.exp().clamp(1e-2, 20.0)
        logits = beta * torch.einsum('bhtlp,rp->bhtlr', tan, self.corners)
        probs = F.softmax(logits, dim=-1)
        return probs.permute(0, 1, 3, 2, 4).contiguous()   # [B,H,L,T,R]

    def forward(self, x, mask=None, return_den=False):
        B, T, _ = x.shape
        H, dk, L, R, C = self.h, self.dk, self.num_tables, self.R, self.C

        Q = self.q(x).view(B, T, H, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, H, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, dk).transpose(1, 2).contiguous()

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            K = K * keep
            V = V * keep

        phiQ = self._phi(Q)    # [B,H,L,T,R]
        phiK = self._phi(K)

        pad = (-T) % C
        if pad:
            phiQ = F.pad(phiQ, (0, 0, 0, pad))
            phiK = F.pad(phiK, (0, 0, 0, pad))
            V_p  = F.pad(V,   (0, 0, 0, pad))
        else:
            V_p = V
        Tp = T + pad
        n_chunks = Tp // C

        phiQ_c = phiQ.view(B, H, L, n_chunks, C, R)
        phiK_c = phiK.view(B, H, L, n_chunks, C, R)
        V_c    = V_p.view(B, H, n_chunks, C, dk)

        # ----- STATE (cheap; needed by both Triton and PyTorch paths) -----
        chunk_A = phiK_c.sum(dim=4)                               # [B,H,L,n,R]
        chunk_B = torch.einsum('bhlncr,bhncd->bhlnrd',
                            phiK_c, V_c)                        # [B,H,L,n,R,dk]
        state_A = F.pad(torch.cumsum(chunk_A, dim=3)[:, :, :, :-1, :],
                        (0, 0, 1, 0))                              # [B,H,L,n,R]
        state_B = F.pad(torch.cumsum(chunk_B, dim=3)[:, :, :, :-1, :, :],
                        (0, 0, 0, 0, 1, 0))                        # [B,H,L,n,R,dk]

        # ----- TRITON PATH (inter + intra fused, no [B,H,n,C,C] tensor) -----
        if _HAS_TRITON_RACE and phiQ.device.type == "cuda" and pad == 0:
            num_sumL, den_sumL = _RaceChunkwiseTritonFn.apply(
                phiQ.contiguous(), phiK.contiguous(), V_p.contiguous(),
                state_A.contiguous(), state_B.contiguous(), C,
            )
            out = (num_sumL / den_sumL.unsqueeze(-1).clamp_min(1e-6))[:, :, :T, :]
            out = out.transpose(1, 2).contiguous().view(B, T, H * dk)
            out = self.drop(self.o(out))
            if mask is not None:
                out = out * mask.unsqueeze(-1).to(out.dtype)
            d_token = (den_sumL[:, :, :T].mean(dim=1) / L).unsqueeze(-1).clamp_min(1e-6)
            self.last_den = d_token.detach()
            return (out, d_token) if return_den else out

        # ----- PYTORCH FALLBACK -----
        # Inter-chunk
        inter_num = torch.einsum('bhlncr,bhlnrd->bhlncd', phiQ_c, state_B)  # [B,H,L,n,C,dk]
        inter_den = torch.einsum('bhlncr,bhlnr->bhlnc',   phiQ_c, state_A)  # [B,H,L,n,C]

        # Intra-chunk (causal, L pre-averaged to save memory)
        if self.chunk_group_size is None or self.chunk_group_size >= n_chunks:
            M_avg = torch.einsum('bhlncr,bhlnsr->bhncs', phiQ_c, phiK_c) / L  # [B,H,n,C,C]
            tri = torch.tril(torch.ones(C, C, device=x.device, dtype=M_avg.dtype))
            M_avg = M_avg * tri
            intra_num = torch.einsum('bhncs,bhnsd->bhncd', M_avg, V_c)        # [B,H,n,C,dk]
            intra_den = M_avg.sum(dim=-1)                                       # [B,H,n,C]
        else:
            g = self.chunk_group_size
            intra_num_parts, intra_den_parts = [], []
            tri = torch.tril(torch.ones(C, C, device=x.device, dtype=phiQ.dtype))
            for gs in range(0, n_chunks, g):
                ge = min(gs + g, n_chunks)
                M_g = torch.einsum('bhlncr,bhlnsr->bhncs',
                                phiQ_c[:, :, :, gs:ge], phiK_c[:, :, :, gs:ge]) / L
                M_g = M_g * tri
                intra_num_parts.append(torch.einsum('bhncs,bhnsd->bhncd',
                                                    M_g, V_c[:, :, gs:ge]))
                intra_den_parts.append(M_g.sum(dim=-1))
            intra_num = torch.cat(intra_num_parts, dim=2)
            intra_den = torch.cat(intra_den_parts, dim=2)

        # Combine
        num = inter_num.mean(dim=2) + intra_num   # [B,H,n,C,dk]
        den = inter_den.mean(dim=2) + intra_den   # [B,H,n,C]

        out = (num / den.unsqueeze(-1).clamp_min(1e-6)).view(B, H, Tp, dk)
        out = out[:, :, :T, :]
        out = out.transpose(1, 2).contiguous().view(B, T, H * dk)
        out = self.drop(self.o(out))
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        d_token = den.view(B, H, Tp)[:, :, :T].mean(dim=1).unsqueeze(-1).clamp_min(1e-6)
        self.last_den = d_token.detach()
        return (out, d_token) if return_den else out

# ============================================================================
# SECTION 6: CAUSAL SPARSE ATTENTION (HyperAttention recursive split)
# ============================================================================

class CausalHyperSparseAttention(nn.Module):
    """
    Causal sortLSH sparse attention via HyperAttention's recursive split
    (adapted from the user-pasted hyper_attn.py).

    Algorithm (causal):
        if N <= min_seq_len:
            return exact causal attention via SDPA
        else:
            split into past half and future half
            recurse on (Q_past, K_past) and (Q_future, K_future) — both causal
            run non-causal sortLSH on (Q_future, K_past) — off-diagonal block
            combine via LSE-weighted addition

    Returns (out, lse) where exp(lse) is the prefix-restricted denominator.
    """
    def __init__(self, d, h, drop, num_bits=7, block_size=256, min_seq_len=2048,
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

        self.lsh = AngularLSH(num_projs=num_bits, dim=(1, 1, self.dk))

    #def _exact_attention(self, q, k, v, causal=False):
    #    """q,k,v: [B,H,T,D] -> (out, lse)."""
    #    return exact_attention_sdpa(q, k, v, scale=self.scale, causal=causal)

    def _exact_attention(self, q, k, v, causal=False):
        """q,k,v: [B,H,T,D] -> (out, lse). Uses Triton on CUDA, SDPA fallback on CPU."""
        if q.device.type == "cuda" and _HAS_TRITON_FLASH:
            return exact_attention_flash(q, k, v, scale=self.scale, causal=causal)
        return exact_attention_sdpa(q, k, v, scale=self.scale, causal=causal)
    
    def _noncausal_sortlsh(self, q, k, v):
        """Non-causal sortLSH same-block exact attention.
        q,k,v: [B,H,T,D] -> (out, lse).
        """
        B, H, Tq, D = q.shape
        Tk = k.shape[2]

        # Hash and sort
        q_hash = self.lsh.hash(q)  # [B,H,Tq]
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
            # Tiny case - just do exact attention
            return self._exact_attention(q, k, v, causal=False)

        q_block_size = q_sorted.shape[2] // num_blocks

        # Reshape into blocks and run SDPA per-block in batched form
        q_b = q_sorted.reshape(B * H * num_blocks, 1, q_block_size, D)
        k_b = k_sorted.reshape(B * H * num_blocks, 1, bs, D)
        v_b = v_sorted.reshape(B * H * num_blocks, 1, bs, D)

        out_b, lse_b = self._exact_attention(q_b, k_b, v_b, causal=False)
        # out_b: [B*H*nb, 1, qbs, D]; lse_b: [B*H*nb, 1, qbs]
        out_blocked = out_b.reshape(B, H, num_blocks * q_block_size, D)
        lse_blocked = lse_b.reshape(B, H, num_blocks * q_block_size)

        # Trim padding if any
        out_blocked = out_blocked[:, :, :Tq, :]
        lse_blocked = lse_blocked[:, :, :Tq]

        # Unsort
        idx = q_sort_inv.unsqueeze(-1).expand(-1, -1, -1, D)
        out_unsorted = torch.gather(out_blocked, 2, idx)
        lse_unsorted = torch.gather(lse_blocked, 2, q_sort_inv)
        return out_unsorted, lse_unsorted

    def _causal_forward(self, q, k, v):
        """Recursive causal attention via HyperAttention split.
        q,k,v: [B,H,N,D] -> (out, lse).
        """
        B, H, N, D = q.shape

        if N <= self.min_seq_len:
            return self._exact_attention(q, k, v, causal=True)

        # Pad odd N
        n_orig = N
        if N % 2:
            q = F.pad(q, (0, 0, 0, 1))
            k = F.pad(k, (0, 0, 0, 1))
            v = F.pad(v, (0, 0, 0, 1))
            N = N + 1

        half = N // 2

        # Recurse on diagonal blocks
        q_past, q_future = q[:, :, :half, :], q[:, :, half:, :]
        k_past, k_future = k[:, :, :half, :], k[:, :, half:, :]
        v_past, v_future = v[:, :, :half, :], v[:, :, half:, :]

        out_top, lse_top = self._causal_forward(q_past, k_past, v_past)
        out_bot_diag, lse_bot_diag = self._causal_forward(q_future, k_future, v_future)

        # Off-diagonal: Q_future attends to K_past, no causal mask needed
        out_off, lse_off = self._noncausal_sortlsh(q_future, k_past, v_past)

        # Bottom half: combine bot_diag with off
        out_bot, lse_bot = add_self_attentions_lse(
            out_bot_diag, lse_bot_diag, out_off, lse_off
        )

        out = torch.cat([out_top, out_bot], dim=2)
        lse = torch.cat([lse_top, lse_bot], dim=2)

        if n_orig != N:
            out = out[:, :, :n_orig, :]
            lse = lse[:, :, :n_orig]

        return out, lse

    def forward(self, x, mask=None, return_lse=False):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q = Q * keep; K = K * keep; V = V * keep

        out_heads, lse_heads = self._causal_forward(Q, K, V)  # [B,H,T,D], [B,H,T]

        out = out_heads.transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.drop(out)
        out = self.o(out)

        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        if return_lse:
            # Token-level log-denominator: log mean_h exp(lse_h) = logsumexp_h(lse_h) - log H
            with torch.no_grad():
                log_d_token = (torch.logsumexp(lse_heads.float(), dim=1) - math.log(self.h)).to(out.dtype)
            log_d_token = log_d_token.unsqueeze(-1)  # [B,T,1]
            return out, log_d_token
        return out


# ============================================================================
# SECTION 7: CAUSAL ELSAA (FUSION)
# ============================================================================

class CausalELSAAAttention(nn.Module):
    """
    Causal ELSAA = sparse + RACE branches fused with denominator-aware m_sparse:

        m_sparse(t) = d_sparse(t) / (d_sparse(t) + lambda(t) * d_race(t) + eps)
        out(t) = g_sparse(t) * m_sparse(t) * out_sparse(t) + g_race(t) * out_race(t)

    Lambda can be either a learned scalar or a query-dependent quantity:
        lambda(t) = c + sigmoid(w^T q_t + b)
    """
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg.get("qkv_bias", False)
        self.mexact_eps = cfg.get("mexact_eps", 1e-6)
        self.lambda_dep = bool(cfg.get("lambda_dependent", False))

        # Sparse branch (HyperAttention recursive)
        self.sparse = CausalHyperSparseAttention(
            d=d, h=h, drop=drop,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 256),
            min_seq_len=cfg.get("hyper_min_seq_len", 8192),
            qkv_bias=qkv_bias, device=device,
        )

        # RACE branch (Algorithm 2 cumsum)
        self.race = CausalRACEAttention(
            d=d, h=h, drop=drop,
            num_bits=cfg.get("race_num_bits", 4),
            num_tables=cfg.get("race_num_tables", 4),
            chunk_size=cfg.get("race_chunk_size", 256),
            chunk_group_size=cfg.get("race_chunk_group_size", None),  # add this line
            qkv_bias=qkv_bias, device=device,
        )

        # Lambda
        if self.lambda_dep:
            # lambda(t) = c + sigmoid(w^T q + b)
            self.lambda_offset = nn.Parameter(torch.tensor(
                cfg.get("lambda_offset_init", 0.3), dtype=torch.float32))
            self.lambda_w = nn.Parameter(torch.empty(d, dtype=torch.float32))
            nn.init.normal_(self.lambda_w, std=1e-3)
            init_target = cfg.get("lambda_init_target", 0.8)
            init_prob = max(min(init_target - cfg.get("lambda_offset_init", 0.3), 0.999), 0.001)
            init_b = math.log(init_prob / (1.0 - init_prob))
            self.lambda_bias = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        else:
            # Scalar learnable lambda via exp(log_lambda)
            init = cfg.get("mexact_lambda_init", 1.0)
            self.log_lambda = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))

        # Gate MLP
        gate_hidden = cfg.get("gate_hidden_dim", 64)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )

        # Logging cache
        self.last_gates = None
        self.last_m_sparse = None
        self.last_lambda = None

    def _compute_lambda(self, x):
        if self.lambda_dep:
            with torch.no_grad():
                # Use sparse-branch q projection for q_t
                q_for_lambda = self.sparse.q(x).float()
            logits = q_for_lambda @ self.lambda_w.float() + self.lambda_bias.float()
            offset = self.lambda_offset.float().clamp_min(0.0)
            lam = offset + torch.sigmoid(logits)
            return lam.unsqueeze(-1).clamp_min(self.mexact_eps)  # [B,T,1]
        else:
            return self.log_lambda.exp().clamp_min(self.mexact_eps)

    def forward(self, x, mask=None):
        # Sparse branch
        out_sparse, log_d_sparse = self.sparse(x, mask=mask, return_lse=True)  # [B,T,d], [B,T,1]
        # Race branch
        out_race, d_race = self.race(x, mask=mask, return_den=True)  # [B,T,d], [B,T,1]

        # Lambda
        lam = self._compute_lambda(x)

        # m_sparse in log-space for stability
        log_d_sparse_det = log_d_sparse.detach().float()
        log_d_race_det = torch.log(d_race.detach().float().clamp_min(self.mexact_eps))
        if isinstance(lam, torch.Tensor) and lam.dim() == 3:
            log_lam = torch.log(lam.float().clamp_min(self.mexact_eps))
        else:
            log_lam = torch.log(lam.float().clamp_min(self.mexact_eps))
        log_eps = torch.full_like(log_d_sparse_det, math.log(self.mexact_eps))

        log_den = torch.logsumexp(torch.stack([
            log_d_sparse_det,
            log_lam + log_d_race_det,
            log_eps,
        ], dim=0), dim=0)
        m_sparse = torch.exp(log_d_sparse_det - log_den).to(out_sparse.dtype)  # [B,T,1]

        # Gates
        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)  # [B,T,2]
        g_sparse = gates[..., 0:1]
        g_race = gates[..., 1:2]

        out = g_sparse * m_sparse * out_sparse + g_race * out_race
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        # Logging
        self.last_gates = gates.detach()
        self.last_m_sparse = m_sparse.detach()
        if isinstance(lam, torch.Tensor):
            self.last_lambda = lam.detach()
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
            self.att = CausalExactAttention(
                d, cfg["num_heads"], drop, cfg.get("qkv_bias", False))
        elif attn_type == "linear":
            self.att = CausalLinearAttention(
                d, cfg["num_heads"], drop, cfg.get("qkv_bias", False),
                chunk_size=cfg.get("linear_chunk_size", 256))
        elif attn_type == "causal_race":
            self.att = CausalRACEAttention(
                d, cfg["num_heads"], drop,
                num_bits=cfg.get("race_num_bits", 4),
                num_tables=cfg.get("race_num_tables", 4),
                chunk_size=cfg.get("race_chunk_size", 256),
                qkv_bias=cfg.get("qkv_bias", False), device=device)
        elif attn_type == "causal_sparse":
            self.att = CausalHyperSparseAttention(
                d, cfg["num_heads"], drop,
                num_bits=cfg.get("hyper_num_bits", 7),
                block_size=cfg.get("hyper_block_size", 256),
                min_seq_len=cfg.get("hyper_min_seq_len", 8192),
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
# SECTION 9: CAUSAL CLASSIFIER MODEL
# ============================================================================

class CausalTransformerClassifier(nn.Module):
    """Causal transformer with classification head.
    Classification uses the LAST valid token's representation.
    """
    def __init__(self, cfg, attn_type, device="cpu"):
        super().__init__()
        self.cfg = cfg
        d = cfg["embed_dim"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.pos_emb = nn.Embedding(cfg["max_len"], d)
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.layers = nn.ModuleList([
            CausalTransformerBlock(cfg, attn_type, device=device)
            for _ in range(cfg["num_layers"])
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["num_classes"])

    def forward(self, x, mask):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.layers:
            h = blk(h, mask=mask)
        h = self.norm(h)
        # Take last valid token per row
        # last valid position = sum(mask) - 1
        last_idx = (mask.sum(dim=1) - 1).clamp_min(0)  # [B]
        gather_idx = last_idx.view(B, 1, 1).expand(-1, 1, h.size(-1))
        last_h = h.gather(1, gather_idx).squeeze(1)  # [B, d]
        return self.head(last_h)


class CausalTransformerLM(nn.Module):
    """Causal LM head for MQAR-style next-token prediction at specific positions."""
    def __init__(self, cfg, attn_type, device="cpu"):
        super().__init__()
        self.cfg = cfg
        d = cfg["embed_dim"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.pos_emb = nn.Embedding(cfg["max_len"], d)
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.layers = nn.ModuleList([
            CausalTransformerBlock(cfg, attn_type, device=device)
            for _ in range(cfg["num_layers"])
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["vocab_size"], bias=False)

    def forward(self, x, mask=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.layers:
            h = blk(h, mask=mask)
        h = self.norm(h)
        return self.head(h)  # [B, T, V]


# ============================================================================
# SECTION 10: ARXIV CAUSAL CLASSIFICATION DATA
# ============================================================================

import re
_BASIC_TOK_RE = re.compile(
    r"""([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])|(\d+[%]?)|([A-Za-z]+(?:'[A-Za-z]+)?)""",
    re.VERBOSE,
)

def basic_english_tokenizer(text):
    text = text.lower()
    out = []
    for p, n, w in _BASIC_TOK_RE.findall(text):
        if p: out.append(p)
        elif n: out.append(n)
        elif w: out.append(w)
    return out


class ArxivCausalDataset(Dataset):
    """Reuses the user's existing arxiv pack-and-tokenize pipeline."""
    def __init__(self, examples, max_len, stoi, pad_idx=0, unk_idx=1):
        self.examples = examples
        self.max_len = max_len
        self.stoi = stoi
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        lbl, txt = self.examples[idx]
        toks = basic_english_tokenizer(str(txt))[: self.max_len]
        ids = [self.stoi.get(t, self.unk_idx) for t in toks]
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        return int(lbl), torch.tensor(ids, dtype=torch.long)

    def collate(self, batch):
        labels, tokens = zip(*batch)
        tokens = torch.stack(tokens, dim=0)
        masks = (tokens != self.pad_idx).long()
        return tokens, masks, torch.tensor(labels, dtype=torch.long)


def build_arxiv_data(max_len, batch_size, desired_train=8000, desired_test=2000,
                     min_doc_len=1000, pack_min_frac=0.8):
    """Build the arxiv causal classification data using HuggingFace datasets."""
    from datasets import load_dataset

    print(f"[arxiv] Loading ccdv/arxiv-classification...")
    raw = load_dataset("ccdv/arxiv-classification")
    if "validation" in raw:
        train_split, test_split = raw["train"], raw["validation"]
    elif "test" in raw:
        train_split, test_split = raw["train"], raw["test"]
    else:
        sp = raw["train"].train_test_split(test_size=0.2, seed=SEED)
        train_split, test_split = sp["train"], sp["test"]

    def make_balanced_long(split, desired, name):
        labels = list(split["label"])
        texts = list(split["text"])
        print(f"[arxiv] Tokenizing {name} for length filter...")
        lengths = np.array([len(basic_english_tokenizer(str(t))) for t in texts])
        buckets = defaultdict(list)
        for i, (y, L) in enumerate(zip(labels, lengths)):
            if L >= min_doc_len:
                buckets[int(y)].append(i)
        nc = len(buckets)
        max_pc = min(len(v) for v in buckets.values())
        per_class = min(max_pc, desired // nc)
        rng = random.Random(SEED)
        chosen = []
        for y, idxs in buckets.items():
            rng.shuffle(idxs)
            chosen.extend(idxs[:per_class])
        rng.shuffle(chosen)
        examples = [(int(labels[i]), texts[i]) for i in chosen]
        print(f"[arxiv]   {name}: {len(examples)} docs, {nc} classes")
        return examples, nc

    train_docs, nc1 = make_balanced_long(train_split, desired_train, "train")
    test_docs, nc2 = make_balanced_long(test_split, desired_test, "test")
    assert nc1 == nc2

    def pack(examples, target_len, min_frac, seed):
        rng = random.Random(seed)
        per_class = defaultdict(list)
        for y, t in examples:
            per_class[int(y)].append(str(t))
        new = []
        for y, docs in per_class.items():
            rng.shuffle(docs)
            buf = []
            for txt in docs:
                toks = basic_english_tokenizer(txt)
                j = 0
                while j < len(toks):
                    rem = target_len - len(buf)
                    if rem <= 0:
                        if len(buf) >= int(min_frac * target_len):
                            new.append((y, " ".join(buf)))
                        buf = []
                        rem = target_len
                    take = min(rem, len(toks) - j)
                    buf.extend(toks[j:j+take])
                    j += take
                    if len(buf) == target_len:
                        new.append((y, " ".join(buf)))
                        buf = []
            if len(buf) >= int(min_frac * target_len):
                new.append((y, " ".join(buf)))
        return new

    print(f"[arxiv] Packing to {max_len}...")
    train_packed = pack(train_docs, max_len, pack_min_frac, SEED)
    test_packed = pack(test_docs, max_len, pack_min_frac, SEED+1)

    # Vocab
    print("[arxiv] Building vocab...")
    cnt = Counter()
    for _, txt in train_packed:
        cnt.update(basic_english_tokenizer(str(txt)))
    most_common = [w for w, _ in cnt.most_common(50000)]
    stoi = {"<pad>": 0, "<unk>": 1}
    for i, w in enumerate(most_common):
        stoi[w] = i + 2
    if "<cls>" not in stoi: stoi["<cls>"] = len(stoi)
    if "<sep>" not in stoi: stoi["<sep>"] = len(stoi)

    train_ds = ArxivCausalDataset(train_packed, max_len, stoi)
    test_ds = ArxivCausalDataset(test_packed, max_len, stoi)

    print(f"[arxiv] Train: {len(train_packed)} packed seqs, Test: {len(test_packed)}, "
          f"vocab: {len(stoi)}, classes: {nc1}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                         pin_memory=(DEVICE=="cuda"), num_workers=2, collate_fn=train_ds.collate)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         pin_memory=(DEVICE=="cuda"), num_workers=2, collate_fn=test_ds.collate)

    return train_dl, test_dl, stoi, nc1


# ============================================================================
# SECTION 11: MQAR DATA
# ============================================================================

class MQARDataset(Dataset):
    """
    Multi-Query Associative Recall (Zoology-style).
    Each sequence: [k1, v1, k2, v2, ..., kN, vN, q1, q2, ..., qK]
    Labels: at each query position, predict the value associated with that key.
    For simplicity here: predict at the LAST position only (single-query MQAR).

    Vocabulary:
        token 0: PAD
        token 1: <q-marker> (separates KV section from queries)
        tokens 2 .. (2+kv_vocab): keys
        tokens (2+kv_vocab) .. (2+2*kv_vocab): values
    """
    def __init__(self, num_examples, seq_len, kv_vocab=512, n_pairs=None, seed=0):
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.kv_vocab = kv_vocab
        if n_pairs is None:
            # default: half sequence is KV pairs, last token is a query
            self.n_pairs = (seq_len - 2) // 2
        else:
            self.n_pairs = n_pairs
        # Cannot sample more unique keys than the vocabulary allows
        self.n_pairs = min(self.n_pairs, kv_vocab)

        self.pad_idx = 0
        self.qmark = 1
        self.key_offset = 2
        self.val_offset = 2 + kv_vocab
        self.vocab_size = 2 + 2 * kv_vocab

        # Pre-generate examples
        self.rng = np.random.RandomState(seed)
        self.data = []
        for _ in range(num_examples):
            keys = self.rng.choice(kv_vocab, size=self.n_pairs, replace=False)
            vals = self.rng.choice(kv_vocab, size=self.n_pairs, replace=True)
            # pick one key to query
            q_idx = self.rng.randint(self.n_pairs)
            seq = []
            for k, v in zip(keys, vals):
                seq.append(self.key_offset + int(k))
                seq.append(self.val_offset + int(v))
            # query: <q-marker> <key>
            seq.append(self.qmark)
            seq.append(self.key_offset + int(keys[q_idx]))
            target = self.val_offset + int(vals[q_idx])

            # Pad to seq_len
            if len(seq) > seq_len:
                seq = seq[:seq_len]
            # We want target at position seq_len - 1, so pad in front if shorter
            while len(seq) < seq_len:
                seq.append(self.pad_idx)
            # Ensure last position has the query key (and target is what we predict next)
            # Actually: we'll make the model predict at the position of the query KEY token.
            self.data.append((torch.tensor(seq, dtype=torch.long), int(target)))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, batch):
        tokens, targets = zip(*batch)
        tokens = torch.stack(tokens, 0)
        masks = (tokens != self.pad_idx).long()
        targets = torch.tensor(targets, dtype=torch.long)
        return tokens, masks, targets


def build_mqar_data(seq_len, batch_size, num_train=20000, num_test=2000,
                    kv_vocab=512, seed=0):
    train_ds = MQARDataset(num_train, seq_len, kv_vocab=kv_vocab, seed=seed)
    test_ds = MQARDataset(num_test, seq_len, kv_vocab=kv_vocab, seed=seed+1)
    vocab_size = train_ds.vocab_size
    print(f"[mqar] seq_len={seq_len}, n_pairs={train_ds.n_pairs}, "
          f"vocab={vocab_size}, train={num_train}, test={num_test}")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                         pin_memory=(DEVICE=="cuda"), num_workers=2, collate_fn=train_ds.collate)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         pin_memory=(DEVICE=="cuda"), num_workers=2, collate_fn=test_ds.collate)
    return train_dl, test_dl, vocab_size


# ============================================================================
# SECTION 12: LRA RETRIEVAL PAIR DATA (causal version, built on arxiv packed)
# ============================================================================

class LRARetrievalPairDataset(Dataset):
    """LRA-style binary retrieval pair: [CLS] doc_a [SEP] doc_b
    Label: 1 if same arxiv class, 0 otherwise.
    Pool the LAST token (causal).
    """
    def __init__(self, pairs, max_len, stoi, pad_idx, unk_idx, cls_idx, sep_idx):
        self.pairs = pairs
        self.max_len = max_len
        self.stoi = stoi
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx
        usable = max_len - 2
        self.left_budget = usable // 2
        self.right_budget = usable - self.left_budget

    def __len__(self):
        return len(self.pairs)

    def _enc(self, toks, budget):
        toks = toks[:budget]
        return [self.stoi.get(t, self.unk_idx) for t in toks]

    def __getitem__(self, idx):
        label, ta, tb = self.pairs[idx]
        toks_a = basic_english_tokenizer(str(ta))
        toks_b = basic_english_tokenizer(str(tb))
        ids = [self.cls_idx] + self._enc(toks_a, self.left_budget) \
              + [self.sep_idx] + self._enc(toks_b, self.right_budget)
        ids = ids[:self.max_len]
        while len(ids) < self.max_len:
            ids.append(self.pad_idx)
        return int(label), torch.tensor(ids, dtype=torch.long)

    def collate(self, batch):
        labels, tokens = zip(*batch)
        tokens = torch.stack(tokens, 0)
        masks = (tokens != self.pad_idx).long()
        return tokens, masks, torch.tensor(labels, dtype=torch.long)


def build_lra_pair_data(max_len, batch_size, n_train_pairs=4000, n_test_pairs=1000):
    """Build LRA-style binary retrieval pairs from arxiv classes."""
    # First build arxiv packed sequences
    print("[lra] Building underlying arxiv packed sequences...")
    train_dl_arx, test_dl_arx, stoi, num_classes = build_arxiv_data(
        max_len=max_len, batch_size=batch_size,
        desired_train=8000, desired_test=2000,
        min_doc_len=1000, pack_min_frac=0.8,
    )

    # Pull packed examples back out
    train_packed = train_dl_arx.dataset.examples
    test_packed = test_dl_arx.dataset.examples

    def make_pairs(examples, n_pairs, seed):
        rng = random.Random(seed)
        by_class = defaultdict(list)
        for y, t in examples:
            by_class[int(y)].append(str(t))
        labels = list(by_class.keys())
        pos_labels = [y for y, v in by_class.items() if len(v) >= 2]
        n_pos = n_pairs // 2
        n_neg = n_pairs - n_pos
        pairs = []
        for _ in range(n_pos):
            y = rng.choice(pos_labels)
            a, b = rng.sample(by_class[y], 2)
            pairs.append((1, a, b))
        for _ in range(n_neg):
            y1, y2 = rng.sample(labels, 2)
            a = rng.choice(by_class[y1])
            b = rng.choice(by_class[y2])
            pairs.append((0, a, b))
        rng.shuffle(pairs)
        return pairs

    train_pairs = make_pairs(train_packed, n_train_pairs, SEED+100)
    test_pairs = make_pairs(test_packed, n_test_pairs, SEED+101)

    cls_idx = stoi["<cls>"]
    sep_idx = stoi["<sep>"]

    train_ds = LRARetrievalPairDataset(train_pairs, max_len, stoi, 0, 1, cls_idx, sep_idx)
    test_ds = LRARetrievalPairDataset(test_pairs, max_len, stoi, 0, 1, cls_idx, sep_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                         pin_memory=(DEVICE=="cuda"), num_workers=2, collate_fn=train_ds.collate)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         pin_memory=(DEVICE=="cuda"), num_workers=2, collate_fn=test_ds.collate)

    print(f"[lra] train_pairs={len(train_pairs)}, test_pairs={len(test_pairs)}, vocab={len(stoi)}")
    return train_dl, test_dl, len(stoi), 2


# ============================================================================
# SECTION 13: TRAINING UTILITIES
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
    """Collect ELSAA gate / m_sparse stats for wandb logging."""
    stats = {}
    if attn_type not in ("elsaa", "elsaa_lambda"):
        return stats
    for li, blk in enumerate(model.layers):
        att = getattr(blk, "att", None)
        if att is None: continue
        if hasattr(att, "last_gates") and att.last_gates is not None:
            g = att.last_gates
            stats[f"gates/layer{li}_sparse"] = g[..., 0].mean().item()
            stats[f"gates/layer{li}_race"] = g[..., 1].mean().item()
        if hasattr(att, "last_m_sparse") and att.last_m_sparse is not None:
            stats[f"m_sparse/layer{li}_mean"] = att.last_m_sparse.mean().item()
        if hasattr(att, "last_lambda") and att.last_lambda is not None:
            stats[f"lambda/layer{li}_mean"] = att.last_lambda.float().mean().item()
    return stats


def train_one_run(model, train_dl, test_dl, optimizer, cfg, attn_type,
                  task, num_epochs, grad_accum_steps, log_to_wandb=True,
                  mqar_target_position=None):
    """Single training run. Logs to wandb and console."""
    steps_per_epoch = len(train_dl)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates = num_epochs * updates_per_epoch
    warmup_updates = max(1, int(0.05 * total_updates))
    scheduler = LinearWarmupLR(optimizer, warmup_updates, total_updates)

    global_update = 0
    for epoch in range(1, num_epochs + 1):
        # --- TRAIN ---
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
        for batch in pbar:
            tokens, masks, targets = batch
            tokens = tokens.to(DEVICE)
            masks = masks.to(DEVICE)
            targets = targets.to(DEVICE)

            if task == "mqar":
                logits_all = model(tokens, masks)  # [B,T,V]
                # Predict at the position of the last non-pad token
                B = tokens.size(0)
                last_idx = (masks.sum(dim=1) - 1).clamp_min(0)
                gather_idx = last_idx.view(B, 1, 1).expand(-1, 1, logits_all.size(-1))
                logits = logits_all.gather(1, gather_idx).squeeze(1)  # [B,V]
            else:
                logits = model(tokens, masks)  # [B,C]

            loss = F.cross_entropy(logits, targets)
            (loss / grad_accum_steps).backward()
            accum += 1

            preds = logits.argmax(dim=-1)
            running_correct += (preds == targets).sum().item()
            running_total += targets.size(0)
            running_loss += loss.item()

            if accum == grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_update += 1

            pbar.set_postfix({
                "loss": running_loss / max(1, len(pbar)),
                "acc": running_correct / max(1, running_total),
            })
        if accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        tr_loss = running_loss / max(1, len(train_dl))
        tr_acc = running_correct / max(1, running_total)

        # --- EVAL ---
        model.eval()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        val_loss = 0.0; val_correct = 0; val_total = 0
        attn_stats_acc = defaultdict(list)
        with torch.no_grad():
            pbar = tqdm(test_dl, desc=f"Ep{epoch} val", leave=False)
            for batch in pbar:
                tokens, masks, targets = batch
                tokens = tokens.to(DEVICE)
                masks = masks.to(DEVICE)
                targets = targets.to(DEVICE)
                if task == "mqar":
                    logits_all = model(tokens, masks)
                    B = tokens.size(0)
                    last_idx = (masks.sum(dim=1) - 1).clamp_min(0)
                    gather_idx = last_idx.view(B, 1, 1).expand(-1, 1, logits_all.size(-1))
                    logits = logits_all.gather(1, gather_idx).squeeze(1)
                else:
                    logits = model(tokens, masks)
                loss = F.cross_entropy(logits, targets)
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                # collect ELSAA stats once per epoch
                s = collect_attn_stats(model, attn_type)
                for k, v in s.items():
                    attn_stats_acc[k].append(v)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        val_time = time.time() - t1
        va_loss = val_loss / max(1, len(test_dl))
        va_acc = val_correct / max(1, val_total)
        cur_lr = scheduler.get_last_lr()[0]

        log = {
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss, "val/acc": va_acc,
            "lr": cur_lr,
            "time/train_sec": train_time, "time/val_sec": val_time,
        }
        for k, vlist in attn_stats_acc.items():
            log[k] = float(np.mean(vlist))

        if log_to_wandb and HAS_WANDB:
            wandb.log(log, step=epoch)

        print(f"Ep{epoch:3d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} ({train_time:.1f}s) "
              f"| va_loss {va_loss:.4f} va_acc {va_acc:.4f} ({val_time:.1f}s) | lr {cur_lr:.2e}")


# ============================================================================
# SECTION 14: EXPERIMENT RUNNER
# ============================================================================

DEFAULT_CFG = {
    "embed_dim": 256,
    "num_heads": 4,
    "mlp_dim": 1024,
    "num_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    # Sparse (HyperAttention recursive)
    "hyper_num_bits": 5,
    "hyper_block_size": 256,
    "hyper_min_seq_len": 8192,
    # RACE (Causal Algorithm 2)
    "race_num_bits": 4,        # R = 2^4 = 16 buckets per table
    "race_num_tables": 5,      # L = 5
    "race_chunk_size": 256,
    # Linear baseline
    "linear_chunk_size": 256,
    # ELSAA
    "mexact_eps": 1e-6,
    "mexact_lambda_init": 1.0,
    "lambda_offset_init": 0.3,
    "lambda_init_target": 0.8,
    "gate_hidden_dim": 128,
}


def run_experiment(task, attn_type, max_len, num_epochs=50, batch_size=2,
                   grad_accum_steps=8, lr=3e-4, weight_decay=0.01,
                   wandb_project="ELSAA_Causal", **overrides):
    """Run a single (task, attn_type, length) experiment."""
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    cfg["max_len"] = max_len
    cfg["batch_size"] = batch_size

    # Build data
    if task == "arxiv":
        train_dl, test_dl, stoi, num_classes = build_arxiv_data(
            max_len=max_len, batch_size=batch_size,
        )
        cfg["vocab_size"] = len(stoi)
        cfg["num_classes"] = num_classes
        ModelCls = CausalTransformerClassifier
        mqar_target_position = None
    elif task == "mqar":
        train_dl, test_dl, vocab_size = build_mqar_data(
            seq_len=max_len, batch_size=batch_size,
            num_train=20000, num_test=2000,
            kv_vocab=cfg.get("mqar_kv_vocab", 512),
        )
        cfg["vocab_size"] = vocab_size
        cfg["num_classes"] = vocab_size  # not used for MQAR (LM head outputs full vocab)
        ModelCls = CausalTransformerLM
        mqar_target_position = -1
    elif task == "lra":
        train_dl, test_dl, vocab_size, num_classes = build_lra_pair_data(
            max_len=max_len, batch_size=batch_size,
            n_train_pairs=4000, n_test_pairs=1000,
        )
        cfg["vocab_size"] = vocab_size
        cfg["num_classes"] = num_classes
        ModelCls = CausalTransformerClassifier
        mqar_target_position = None
    else:
        raise ValueError(f"Unknown task: {task}")

    # Build model
    print(f"\n=== Task: {task} | Method: {attn_type} | Length: {max_len} ===")
    model = ModelCls(cfg, attn_type, device=DEVICE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params/1e6:.1f}M params, "
          f"layers={cfg['num_layers']}, d={cfg['embed_dim']}, h={cfg['num_heads']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # WandB init
    log_wandb = HAS_WANDB
    if log_wandb:
        run_name = f"{task}_{attn_type}_L{max_len}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={**cfg, "task": task, "attn_type": attn_type,
                    "lr": lr, "weight_decay": weight_decay,
                    "epochs": num_epochs, "grad_accum_steps": grad_accum_steps,
                    "n_params": n_params},
            reinit=True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("time/*", step_metric="epoch")
        wandb.define_metric("gates/*", step_metric="epoch")
        wandb.define_metric("m_sparse/*", step_metric="epoch")
        wandb.define_metric("lambda/*", step_metric="epoch")
        wandb.define_metric("val/acc", summary="max")
        wandb.define_metric("val/loss", summary="min")

    try:
        train_one_run(
            model, train_dl, test_dl, optimizer, cfg, attn_type, task,
            num_epochs=num_epochs, grad_accum_steps=grad_accum_steps,
            log_to_wandb=log_wandb, mqar_target_position=mqar_target_position,
        )
    finally:
        if log_wandb:
            wandb.finish()


# ============================================================================
# SECTION 15: EXPERIMENT LIST + SHARED CONFIG
# ============================================================================
# Edit SHARED_CFG for hyperparameters that are the same across all experiments.
# Add/remove entries in EXPERIMENTS to control what runs and in what order.
# Each experiment only needs to specify what differs from SHARED_CFG.
# ============================================================================

SHARED_CFG = dict(
    task          = "arxiv",          # "mqar" | "arxiv" | "lra"
    length        = 64000,            # sequence length
    epochs        = 1,
    lr            = 3e-4,
    weight_decay  = 0.01,
    embed_dim     = 256,
    num_layers    = 4,
    num_heads     = 4,
    mlp_dim       = 1024,
    mqar_kv_vocab = 512,
    hyper_min_seq_len = 8192,
    wandb_project = "ELSAA_Causal",
)

# ----------------------------------------------------------------------------
# Add / remove / reorder experiments here.
# Required keys per entry: "method", "batch_size", "grad_accum_steps"
# Any other key overrides the matching SHARED_CFG value for that run only.
# ----------------------------------------------------------------------------
EXPERIMENTS = [
    # 1. ELSAA  — large effective batch via accumulation
    #dict(method="elsaa",        batch_size=1,  grad_accum_steps=8),
    # 2. ELSAA with query-dependent lambda
    #dict(method="elsaa_lambda", batch_size=2,  grad_accum_steps=8),
    # 3. Sparse branch only (HyperAttention recursive split)
    #dict(method="causal_sparse",batch_size=4,  grad_accum_steps=4),
    # 4. RACE branch only (Algorithm 2 chunked cumsum)
    dict(method="causal_race",  batch_size=1,  grad_accum_steps=8),
    # 5. Exact causal attention (FlashAttention / SDPA)
    #dict(method="exact",        batch_size=8,  grad_accum_steps=1),
    # 6. Linear causal attention (ELU+1)
    #dict(method="linear",       batch_size=8,  grad_accum_steps=2),
]
def _sanity_check_flash_vs_sdpa():
    if not _HAS_TRITON_FLASH or not torch.cuda.is_available():
        print("Triton or CUDA missing — skipping check.")
        return
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 4096, 64
    q = torch.randn(B, H, T, D, device="cuda")
    k = torch.randn(B, H, T, D, device="cuda")
    v = torch.randn(B, H, T, D, device="cuda")
    for causal in (False, True):
        o_s, lse_s = exact_attention_sdpa(q, k, v, scale=D**-0.5, causal=causal)
        o_f, lse_f = exact_attention_flash(q, k, v, scale=D**-0.5, causal=causal)
        out_diff = (o_s - o_f).abs().max().item()
        lse_diff = (lse_s - lse_f).abs().max().item()
        print(f"causal={causal}  max|Δout|={out_diff:.4g}  max|Δlse|={lse_diff:.4g}")

#_sanity_check_flash_vs_sdpa()

def main():
    for i, exp in enumerate(EXPERIMENTS, 1):
        cfg = dict(SHARED_CFG)
        cfg.update(exp)                      # per-experiment overrides

        method            = cfg.pop("method")
        task              = cfg.pop("task")
        length            = cfg.pop("length")
        epochs            = cfg.pop("epochs")
        batch_size        = cfg.pop("batch_size")
        grad_accum_steps  = cfg.pop("grad_accum_steps")
        lr                = cfg.pop("lr")
        weight_decay      = cfg.pop("weight_decay")
        wandb_project     = cfg.pop("wandb_project")

        print(f"\n{'='*70}")
        print(f"  Experiment {i}/{len(EXPERIMENTS)}: method={method}  "
              f"task={task}  length={length}  "
              f"bs={batch_size}  accum={grad_accum_steps}")
        print(f"{'='*70}\n")

        run_experiment(
            task=task,
            attn_type=method,
            max_len=length,
            num_epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            weight_decay=weight_decay,
            wandb_project=wandb_project,
            **cfg,                           # remaining per-exp overrides
        )


if __name__ == "__main__":
    main()
