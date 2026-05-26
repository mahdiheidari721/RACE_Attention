"""
============================================================================
ELSAA CAUSAL — Copy Task (Perplexity) with Length Extrapolation
============================================================================

Synthetic Copy task from Scatterbrain (Chen et al., 2021).

Task definition:
    A sequence consists of TWO halves:
      1. A "source" segment of K random tokens
      2. A "copy" segment that EXACTLY repeats the source

    The model is causal. It sees the source freely, then a separator token,
    then must predict the copy segment token-by-token.

    Perfect copying => perplexity = 1.0
    Random guessing => perplexity = vocab_size

Sequence structure:
    [t1, t2, ..., tK, SEP, t1, t2, ..., tK]
    Length = 2*K + 1

The model learns to retrieve from the source positions. This DIRECTLY tests
in-context recall:
    - Exact attention can do it trivially
    - Sparse attention should do it (the sparse branch should locate the
      matching token via LSH/proximity in feature space)
    - Linear / kernel attention typically fails: they can't represent the
      identity-like retrieval map needed here

LENGTH EXTRAPOLATION:
    - Train at one K (e.g., K=512, seq_len=1025)
    - Evaluate at multiple K values (256, 512, 1024, 2048, 4096)
    - Use LARGE max_seq_len to support all eval lengths
    - Positional embeddings are fixed-size, so interpolation happens for
      lengths beyond training length

Attention types tested:
    - elsaa            : Causal sortLSH sparse + Causal RACE + m_sparse fusion
    - causal_race      : RACE branch only
    - exact            : SDPA causal exact attention
    - linear           : Causal linear attention (ELU+1 kernel)
    - performer        : Causal Performer (FAVOR+ random Fourier features)
    - causal_sparse    : Causal sparse (sortLSH)

============================================================================
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import time
import random
import itertools
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.attention import sdpa_kernel, SDPBackend
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
# SECTION 4b: CAUSAL PERFORMER ATTENTION (FAVOR+ RFF baseline)
# ============================================================================

class CausalPerformerAttention(nn.Module):
    """Performer attention using FAVOR+ (random Fourier features).
    
    Uses positive random features to approximate softmax kernel:
    exp(q^T k / sqrt(d)) ≈ φ(q)^T φ(k)
    
    where φ(x) uses random orthogonal features for better approximation.
    This is a better low-rank baseline than Linear (ELU+1 kernel).
    """

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
        
        # Create random features (fixed, not learned)
        # Use orthogonal random features for better approximation
        self._create_projection_matrix(device)
    
    def _create_projection_matrix(self, device):
        """Create random orthogonal projection matrix for FAVOR+."""
        # Number of blocks for orthogonal features
        num_blocks = math.ceil(self.num_features / self.dk)
        
        # Create orthogonal blocks
        blocks = []
        for _ in range(num_blocks):
            # Generate random matrix and orthogonalize via QR
            unstructured = torch.randn(self.dk, self.dk, device=device)
            q, _ = torch.linalg.qr(unstructured)
            blocks.append(q)
        
        # Concatenate and take first num_features columns
        projection = torch.cat(blocks, dim=1)[:, :self.num_features]
        
        # Optional: scaling for orthogonal features
        if self.ortho_scaling:
            # Scaling factor from Performer paper for better approximation
            multiplier = torch.randn(self.num_features, device=device).norm()
            projection = projection * (multiplier / math.sqrt(self.dk))
        
        # Register as buffer (not a parameter - stays fixed)
        self.register_buffer("projection_matrix", projection, persistent=False)
    
    def _phi(self, x):
        """FAVOR+ kernel feature map.
        
        φ(x) = exp(w^T x - ||x||²/2 - log(m)/2) / sqrt(m)
        
        where w are random features and m is num_features.
        """
        # x: [B, H, T, d_k]
        
        # Project: x @ projection_matrix -> [B, H, T, num_features]
        projection = torch.einsum("bhtd,dm->bhtm", x, self.projection_matrix)
        
        # Compute ||x||² for each query/key
        x_squared = (x ** 2).sum(dim=-1, keepdim=True) / 2.0  # [B, H, T, 1]
        
        # FAVOR+ feature map:
        # exp(projection - ||x||²/2 - log(m)/2) / sqrt(m)
        # The division by sqrt(m) is for normalization
        features = torch.exp(projection - x_squared - 0.5 * math.log(self.num_features))
        
        return features  # [B, H, T, num_features]
    
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        
        Q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        K = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        
        # Apply feature map
        phiQ = self._phi(Q)  # [B, H, T, m]
        phiK = self._phi(K)  # [B, H, T, m]
        
        # Causal linear attention via cumulative state
        # state_kv: running sum of φ(K)^T V
        # state_k: running sum of φ(K)
        
        m = self.num_features
        state_kv = torch.zeros(B, self.h, m, self.dk, device=x.device, dtype=Q.dtype)
        state_k = torch.zeros(B, self.h, m, device=x.device, dtype=Q.dtype)
        
        out_chunks = []
        for cs in range(0, T, self.chunk_size):
            ce = min(cs + self.chunk_size, T)
            pK = phiK[:, :, cs:ce, :]  # [B, H, chunk, m]
            pQ = phiQ[:, :, cs:ce, :]  # [B, H, chunk, m]
            vC = V[:, :, cs:ce, :]     # [B, H, chunk, d_k]
            
            # Within-chunk causal attention via cumsum
            # kv_outer: [B, H, chunk, m, d_k]
            kv_outer = torch.einsum("bhtm,bhtd->bhtmd", pK, vC)
            kv_local = torch.cumsum(kv_outer, dim=2)
            k_local = torch.cumsum(pK, dim=2)
            
            # Add state from previous chunks
            kv_at_t = state_kv.unsqueeze(2) + kv_local  # [B, H, chunk, m, d_k]
            k_at_t = state_k.unsqueeze(2) + k_local     # [B, H, chunk, m]
            
            # Compute output: φ(Q) @ state_kv / (φ(Q) @ state_k)
            num = torch.einsum("bhtm,bhtmd->bhtd", pQ, kv_at_t)
            den = torch.einsum("bhtm,bhtm->bht", pQ, k_at_t).unsqueeze(-1) + self.eps
            out_chunks.append(num / den)
            
            # Update state with last position in chunk
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
    """Causal sortLSH sparse attention via HyperAttention's recursive split.

    Base case uses ONLY SDPA (no Triton flash).
    """

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
        # Always SDPA — no Triton flash
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
# SECTION 7: CAUSAL ELSAA — shared QKV + separate W_O per branch
# ============================================================================

class CausalELSAAAttention(nn.Module):
    """Causal ELSAA.

    Shared Q/K/V across branches (so both branches approximate the SAME
    attention operation), separate W_O per branch, post-projection fusion.
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

        # SHARED Q/K/V
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)

        # SEPARATE output projections per branch
        self.o_sparse = nn.Linear(d, d)
        self.o_race = nn.Linear(d, d)
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

        # Branch forwards (per-head outputs + denominator proxies)
        head_sparse, log_d_sparse = self.sparse.forward_core(Q, K, V, return_lse=True)
        head_race, d_race = self.race.forward_core(Q, K, V, return_den=True)

        # SEPARATE W_O per branch
        merged_sparse = head_sparse.transpose(1, 2).contiguous().view(B, T, H * Dk)
        merged_race = head_race.transpose(1, 2).contiguous().view(B, T, H * Dk)
        out_sparse = self.o_sparse(self.out_drop(merged_sparse))
        out_race = self.o_race(self.out_drop(merged_race))

        # Lambda
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
        m_sparse = torch.exp(log_d_sparse_det - log_den).to(out_sparse.dtype)

        # Gates
        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)
        g_sparse = gates[..., 0:1]
        g_race = gates[..., 1:2]

        # Post-projection fusion
        out = g_sparse * m_sparse * out_sparse + g_race *(1 - m_sparse)* out_race

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
# SECTION 9: CAUSAL LANGUAGE MODEL WITH INTERPOLATED POSITIONAL EMBEDDINGS
# ============================================================================

class CausalLM(nn.Module):
    """Causal language model with positional embedding interpolation
    for length extrapolation."""

    def __init__(self, cfg, attn_type, device="cpu"):
        super().__init__()
        self.cfg = cfg
        d = cfg["embed_dim"]
        self.vocab_size = cfg["vocab_size"]
        # Training-time positional embedding table size
        self.train_pos_size = cfg["train_pos_size"]

        self.tok_emb = nn.Embedding(self.vocab_size, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.train_pos_size, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        self.drop = nn.Dropout(cfg["drop_rate"])

        self.layers = nn.ModuleList([
            CausalTransformerBlock(cfg, attn_type, device=device)
            for _ in range(cfg["num_layers"])
        ])

        self.norm = nn.LayerNorm(d)
        # Tied output head
        self.head = nn.Linear(d, self.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def _get_pos_emb(self, T):
        """Return positional embeddings for length T.
        If T <= train_pos_size, return first T positions.
        If T > train_pos_size, linearly interpolate.
        """
        if T <= self.train_pos_size:
            return self.pos_emb[:, :T, :]
        # Interpolate for longer sequences
        pe = self.pos_emb.transpose(1, 2)  # [1, d, train_pos_size]
        pe = F.interpolate(pe, size=T, mode="linear", align_corners=False)
        return pe.transpose(1, 2)

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x)
        h = h + self._get_pos_emb(T)
        h = self.drop(h)

        for blk in self.layers:
            h = blk(h, mask=None)

        h = self.norm(h)
        return self.head(h)


# ============================================================================
# SECTION 10: COPY TASK DATA
# ============================================================================

class CopyDataset(Dataset):
    """Synthetic Copy task.

    Each example is a sequence of length 2*K + 1:
        [t_1, t_2, ..., t_K, SEP, t_1, t_2, ..., t_K]

    where:
      - t_i are i.i.d. uniform from {0, ..., vocab_size - 2}
      - SEP = vocab_size - 1  (reserved separator token)

    During training, the model learns next-token prediction over the WHOLE
    sequence. Perplexity is computed ONLY over the COPY half (positions
    K+1 to 2K), because the source half is unpredictable noise.

    Returns:
        x : [2K] input tokens (sequence[:-1])
        y : [2K] target tokens (sequence[1:])
        copy_mask : [2K] bool — True for positions whose TARGET lies in copy half
    """

    def __init__(self, num_examples, K, vocab_size, seed=0):
        super().__init__()
        self.K = K
        self.vocab_size = vocab_size
        self.num_examples = num_examples
        self.SEP = vocab_size - 1  # reserved separator
        self.seq_len = 2 * K + 1

        # Pre-generate to make epochs deterministic and fast
        rng = np.random.RandomState(seed)
        # source tokens are in [0, vocab_size - 1) so SEP never appears in source
        self.sources = rng.randint(
            low=0, high=vocab_size - 1, size=(num_examples, K), dtype=np.int64
        )

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        src = self.sources[idx]                          # [K]
        seq = np.empty(self.seq_len, dtype=np.int64)
        seq[:self.K] = src
        seq[self.K] = self.SEP
        seq[self.K + 1:] = src

        seq_t = torch.from_numpy(seq)
        x = seq_t[:-1]                                   # [2K]
        y = seq_t[1:]                                    # [2K]

        # Loss mask: include positions whose TARGET is in the copy half
        # The copy half occupies original positions [K+1 .. 2K],
        # so for shifted-by-one labels, the corresponding x-positions are
        # [K .. 2K-1]. We mask positions x[K:].
        copy_mask = torch.zeros_like(x, dtype=torch.bool)
        copy_mask[self.K:] = True

        return x, y, copy_mask


# ============================================================================
# SECTION 11: PERPLEXITY EVALUATION WITH OOM HANDLING
# ============================================================================

def check_memory_for_length(model, seq_len, vocab_size, device=DEVICE):
    """Dry run to check if seq_len fits in memory."""
    if device != "cuda":
        return True
    
    try:
        print(f"    [memory check] Testing seq_len={seq_len} ...")
        dummy_x = torch.randint(0, vocab_size, (1, seq_len), device=device)
        with torch.no_grad():
            _ = model(dummy_x)
        del dummy_x
        torch.cuda.empty_cache()
        print(f"    [memory check] ✓ seq_len={seq_len} fits in memory")
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    [memory check] ✗ seq_len={seq_len} OOM - will skip this length")
            torch.cuda.empty_cache()
            return False
        else:
            raise e


@torch.no_grad()
def evaluate_copy_perplexity(model, loader, desc="eval", handle_oom=True):
    """Average loss over copy-half positions only -> perplexity.
    
    With OOM handling: processes examples one at a time with try-catch.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    failed_examples = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y, mask in pbar:
        try:
            x = x.to(DEVICE); y = y.to(DEVICE); mask = mask.to(DEVICE)

            logits = model(x)                                  # [B, T, V]
            per_pos_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="none",
            ).reshape_as(y)                                   # [B, T]

            loss_sum = (per_pos_loss * mask.float()).sum().item()
            n_tokens = mask.sum().item()
            total_loss += loss_sum
            total_tokens += n_tokens

            # Free memory immediately
            del x, y, mask, logits, per_pos_loss
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            pbar.set_postfix({"ppl": f"{math.exp(min(total_loss / max(1, total_tokens), 50)):.2f}"})
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and handle_oom:
                failed_examples += 1
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                # Continue with remaining examples
                continue
            else:
                raise e

    if failed_examples > 0:
        print(f"    [warn] {failed_examples} examples failed due to OOM")
    
    if total_tokens == 0:
        return float('inf')  # All examples failed
    
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 50))


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


def train_one_run(model, train_dl, valid_dl, optimizer, cfg, attn_type,
                  num_epochs, grad_accum_steps, log_to_wandb=True):
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
        running_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0

        pbar = tqdm(train_dl, desc=f"Ep{epoch} train", leave=False)
        for x, y, mask in pbar:
            try:
                x = x.to(DEVICE); y = y.to(DEVICE); mask = mask.to(DEVICE)

                logits = model(x)
                per_pos_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    reduction="none",
                ).reshape_as(y)
                mask_f = mask.float()
                denom = mask_f.sum().clamp_min(1.0)
                loss = (per_pos_loss * mask_f).sum() / denom

                (loss / grad_accum_steps).backward()
                accum += 1
                running_loss += (per_pos_loss * mask_f).sum().item()
                running_tokens += mask.sum().item()

                if accum == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    global_update += 1

                avg = running_loss / max(1, running_tokens)
                pbar.set_postfix({"loss": avg, "ppl": math.exp(min(avg, 20))})
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[warn] OOM during training batch, skipping...")
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    # Reset gradients and continue
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    continue
                else:
                    raise e

        if accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        tr_loss = running_loss / max(1, running_tokens)
        tr_ppl = math.exp(min(tr_loss, 20))

        # --- VALIDATION ---
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        attn_stats_acc = defaultdict(list)
        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0
        with torch.no_grad():
            for x, y, mask in tqdm(valid_dl, desc=f"Ep{epoch} val", leave=False):
                try:
                    x = x.to(DEVICE); y = y.to(DEVICE); mask = mask.to(DEVICE)
                    logits = model(x)
                    per_pos_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        reduction="none",
                    ).reshape_as(y)
                    val_loss_sum += (per_pos_loss * mask.float()).sum().item()
                    val_tokens += mask.sum().item()
                    s = collect_attn_stats(model, attn_type)
                    for k, v in s.items():
                        attn_stats_acc[k].append(v)
                    
                    # Free memory
                    del x, y, mask, logits, per_pos_loss
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n[warn] OOM during validation batch, skipping...")
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        val_time = time.time() - t1
        va_loss = val_loss_sum / max(1, val_tokens)
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

        print(f"Ep{epoch:3d} | tr_loss {tr_loss:.4f} ppl {tr_ppl:.3f} ({train_time:.1f}s) "
              f"| va_loss {va_loss:.4f} ppl {va_ppl:.3f} ({val_time:.1f}s) "
              f"| best {best_val_ppl:.3f} | lr {cur_lr:.2e}")

    return best_val_ppl


# ============================================================================
# SECTION 13: LENGTH EXTRAPOLATION STUDY WITH ROBUST OOM HANDLING
# ============================================================================

def length_extrapolation_study(
    attn_type,
    train_K=512,
    eval_K_values=[256, 512, 1024, 2048, 4096],
    num_epochs=40,
    batch_size=64,
    grad_accum_steps=1,
    lr=3e-4,
    weight_decay=0.01,
    vocab_size=64,
    num_train_examples=100_000,
    num_valid_examples=2_000,
    num_test_examples=2_000,
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    mlp_dim=1024,
    wandb_project="ELSAA_Copy_Scaling",
    results_dir="./copy_results",
):
    """Train at one K, evaluate at multiple K values.
    
    This tests length extrapolation: how well does the model generalize
    to longer sequences than it was trained on?
    
    Includes robust OOM handling:
    - Pre-checks memory for each eval length
    - Processes examples with batch=1 if needed
    - Gracefully skips lengths that don't fit
    - Continues with remaining lengths even if some fail
    """
    
    # Maximum sequence length we'll ever see
    max_K = max(eval_K_values)
    max_seq_len = 2 * max_K + 1
    
    # Training sequence length
    train_seq_len = 2 * train_K + 1
    
    print(f"\n{'='*70}")
    print(f"  LENGTH EXTRAPOLATION STUDY: {attn_type}")
    print(f"{'='*70}")
    print(f"  Train at K={train_K} (seq_len={train_seq_len})")
    print(f"  Evaluate at K={eval_K_values}")
    print(f"  Max seq_len={max_seq_len} (for positional embeddings)")
    print(f"{'='*70}\n")
    
    # Build config with LARGE max_seq_len to support all eval lengths
    cfg = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "mlp_dim": mlp_dim,
        "num_layers": num_layers,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "vocab_size": vocab_size,
        "train_pos_size": max_seq_len,  # LARGE to support interpolation
        # Sparse branch
        "hyper_num_bits": 5,
        "hyper_block_size": 64,
        "hyper_min_seq_len": 256,
        # RACE branch
        "race_num_bits": 4,
        "race_num_tables": 4,
        "race_chunk_size": 64,
        # Linear baseline
        "linear_chunk_size": 128,
        # Performer baseline
        "performer_num_features": None,
        "performer_chunk_size": 128,
        "performer_ortho_scaling": False,
        # ELSAA
        "mexact_eps": 1e-6,
        "mexact_lambda_init": 1.0,
        "lambda_offset_init": 0.3,
        "lambda_init_target": 0.8,
        "gate_hidden_dim": 128,
    }
    
    # ========== PHASE 1: TRAIN AT K=train_K ==========
    print(f"\n[PHASE 1] Training {attn_type} at K={train_K}")
    
    try:
        train_ds = CopyDataset(
            num_examples=num_train_examples,
            K=train_K,
            vocab_size=vocab_size,
            seed=0
        )
        valid_ds = CopyDataset(
            num_examples=num_valid_examples,
            K=train_K,
            vocab_size=vocab_size,
            seed=1
        )
        
        train_dl = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=2, pin_memory=(DEVICE == "cuda"),
        )
        valid_dl = DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=(DEVICE == "cuda"),
        )
        
        model = CausalLM(cfg, attn_type, device=DEVICE).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] {n_params/1e6:.2f}M params, layers={num_layers}, "
              f"d={embed_dim}, h={num_heads}")
        print(f"[model] train_pos_size={cfg['train_pos_size']} (supports up to K={max_K})")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
        
        log_wandb = HAS_WANDB
        if log_wandb:
            run_name = f"copy_{attn_type}_trainK{train_K}_scaling"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={**cfg, "attn_type": attn_type,
                        "train_K": train_K,
                        "eval_K_values": eval_K_values,
                        "lr": lr, "weight_decay": weight_decay,
                        "epochs": num_epochs, "grad_accum_steps": grad_accum_steps,
                        "n_params": n_params},
                reinit=True,
            )
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("eval_K/*")
            wandb.define_metric("lr", step_metric="epoch")
        
        # Train
        best_ppl = train_one_run(
            model, train_dl, valid_dl, optimizer, cfg, attn_type,
            num_epochs=num_epochs, grad_accum_steps=grad_accum_steps,
            log_to_wandb=log_wandb,
        )
        print(f"\n[PHASE 1 DONE] Best validation ppl at K={train_K}: {best_ppl:.3f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[PHASE 1 FAILED] OOM during training at K={train_K}")
            print(f"[ERROR] {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            return {"error": f"OOM during training at K={train_K}"}
        else:
            raise e
    except Exception as e:
        print(f"\n[PHASE 1 FAILED] Unexpected error during training")
        print(f"[ERROR] {e}")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return {"error": f"Training failed: {str(e)}"}
    
    # ========== PHASE 2: EVALUATE AT ALL K VALUES ==========
    print(f"\n[PHASE 2] Evaluating {attn_type} at multiple K values")
    
    results = {}
    skipped_K = []
    
    for K in eval_K_values:
        seq_len = 2 * K + 1
        print(f"\n  Evaluating at K={K} (seq_len={seq_len})...")
        
        # Memory check BEFORE creating dataset
        try:
            if not check_memory_for_length(model, seq_len, vocab_size):
                print(f"    [skip] K={K} doesn't fit in memory, skipping")
                skipped_K.append(K)
                results[K] = None  # Mark as skipped
                continue
        except Exception as e:
            print(f"    [skip] Memory check failed for K={K}: {e}")
            skipped_K.append(K)
            results[K] = None
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue
        
        # Create test dataset at this K
        try:
            test_ds = CopyDataset(
                num_examples=num_test_examples,
                K=K,
                vocab_size=vocab_size,
                seed=42
            )
            
            # Use batch_size=1 for very long sequences to avoid OOM
            eval_batch_size = 1 if seq_len > 4096 else min(batch_size, 32)
            
            test_dl = DataLoader(
                test_ds, batch_size=eval_batch_size, shuffle=False,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=False,  # Safer for long sequences
            )
            
            # Evaluate with OOM handling
            ppl = evaluate_copy_perplexity(
                model, test_dl, 
                desc=f"  K={K}",
                handle_oom=True
            )
            
            if ppl == float('inf'):
                print(f"    [skip] K={K} all examples failed (OOM)")
                skipped_K.append(K)
                results[K] = None
            else:
                results[K] = ppl
                print(f"    ✓ K={K:4d}  seq_len={seq_len:5d}  ppl={ppl:.3f}")
                
                if log_wandb:
                    wandb.log({f"eval_K/K{K}_ppl": ppl})
            
            # Always clear cache between evaluations
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    [skip] K={K} OOM during evaluation setup")
                skipped_K.append(K)
                results[K] = None
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                print(f"    [error] Unexpected error at K={K}: {e}")
                skipped_K.append(K)
                results[K] = None
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue
        except Exception as e:
            print(f"    [error] Unexpected error at K={K}: {e}")
            skipped_K.append(K)
            results[K] = None
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue
    
    # ========== SAVE RESULTS ==========
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"copy_scaling_{attn_type}_trainK{train_K}.json")
    
    # Filter out None values for saving
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    save_data = {
        "attn_type": attn_type,
        "train_K": train_K,
        "best_val_ppl": best_ppl if 'best_ppl' in locals() else None,
        "eval_results": {f"K{k}": ppl for k, ppl in valid_results.items()},
        "eval_K_values": eval_K_values,
        "skipped_K": skipped_K,
        "successful_K": list(valid_results.keys()),
    }
    
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[saved] {results_path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY: {attn_type}")
    print(f"{'='*70}")
    print(f"  {'K':>6}  {'seq_len':>8}  {'perplexity':>12}  {'status':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*10}")
    for K in eval_K_values:
        ppl = results.get(K)
        seq_len = 2 * K + 1
        marker = " (train)" if K == train_K else ""
        if ppl is None:
            status = "SKIPPED"
            ppl_str = "---"
        else:
            status = "OK"
            ppl_str = f"{ppl:.3f}"
        print(f"  {K:6d}  {seq_len:8d}  {ppl_str:>12}{marker}  {status:>10}")
    
    if skipped_K:
        print(f"\n  [summary] Skipped K values due to OOM: {skipped_K}")
    print(f"{'='*70}\n")
    
    if log_wandb:
        wandb.finish()
    
    return results


# ============================================================================
# SECTION 14: EXPERIMENT RUNNER WITH ROBUST ERROR HANDLING
# ============================================================================

def run_all_methods_scaling_study(
    train_K=512,
    eval_K_values=[256, 512, 1024, 2048, 4096],
    num_epochs=40,
    methods=["exact", "linear", "performer", "causal_race", "causal_sparse", "elsaa"],
):
    """Run length extrapolation study for all methods with robust error handling.
    
    Each method is wrapped in try-except so that if one fails, the others
    continue. Final summary shows which succeeded and which failed.
    """
    
    all_results = {}
    failed_methods = []
    successful_methods = []
    
    for i, method in enumerate(methods, 1):
        print(f"\n{'#'*70}")
        print(f"#  METHOD {i}/{len(methods)}: {method}")
        print(f"{'#'*70}\n")
        
        try:
            results = length_extrapolation_study(
                attn_type=method,
                train_K=train_K,
                eval_K_values=eval_K_values,
                num_epochs=num_epochs,
                batch_size=64,
                grad_accum_steps=1,
                lr=3e-4,
                weight_decay=0.01,
            )
            
            # Check if training succeeded
            if isinstance(results, dict) and "error" in results:
                print(f"\n[FAILED] {method}: {results['error']}")
                failed_methods.append(method)
                all_results[method] = None
            else:
                all_results[method] = results
                successful_methods.append(method)
                print(f"\n[SUCCESS] {method} completed")
        
        except Exception as e:
            print(f"\n[FAILED] {method}: Unexpected error")
            print(f"[ERROR] {e}")
            failed_methods.append(method)
            all_results[method] = None
            
            # Clear CUDA cache and continue
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    
    # ========== PRINT COMPARATIVE SUMMARY ==========
    print(f"\n{'='*70}")
    print(f"  COMPARATIVE SUMMARY (train_K={train_K})")
    print(f"{'='*70}")
    
    if not successful_methods:
        print("  [ERROR] All methods failed!")
        print(f"{'='*70}\n")
        return all_results
    
    # Build header with only successful methods
    header = f"  {'K':>6}  {'seq_len':>8}"
    for m in successful_methods:
        header += f"  {m:>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    
    # Print results for each K
    for K in eval_K_values:
        seq_len = 2 * K + 1
        row = f"  {K:6d}  {seq_len:8d}"
        
        for m in successful_methods:
            results = all_results.get(m)
            if results is None:
                row += f"  {'FAILED':>12}"
            else:
                ppl = results.get(K)
                if ppl is None:
                    row += f"  {'SKIPPED':>12}"
                else:
                    row += f"  {ppl:12.3f}"
        print(row)
    
    print(f"{'='*70}")
    
    # Print status summary
    print(f"\n  Status Summary:")
    print(f"  ✓ Successful methods ({len(successful_methods)}): {', '.join(successful_methods)}")
    if failed_methods:
        print(f"  ✗ Failed methods ({len(failed_methods)}): {', '.join(failed_methods)}")
    print(f"{'='*70}\n")
    
    return all_results


# ============================================================================
# SECTION 15: MAIN
# ============================================================================

def main():
    """Main entry point.
    
    Default: Train at K=512, evaluate at [256, 512, 1024, 2048, 4096]
    """
    
    # Quick test (uncomment for fast verification)
    # run_all_methods_scaling_study(
    #     train_K=256,
    #     eval_K_values=[128, 256, 512],
    #     num_epochs=5,
    #     methods=["exact", "linear", "elsaa"],
    # )
    
    # Full experiment (default)
    run_all_methods_scaling_study(
        train_K=512,
        eval_K_values=[256, 512, 1024, 2048, 4096],
        num_epochs=40,
        methods=["elsaa"],
    )


if __name__ == "__main__":
    main()