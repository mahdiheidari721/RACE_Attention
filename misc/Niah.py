"""
============================================================================
ELSAA CAUSAL — Needle In A Haystack (NIAH) Retrieval
============================================================================

Synthetic retrieval task. The model sees:
    [filler ... KEY VAL ... filler] [SEP] [KEY]
and must predict VAL at the last position.

Training:
    - Uniform random needle position in [0.05, 0.95] for every example
    - Single training length (e.g. 512)
    - 200K training examples for confident generalization

Validation:
    - Same training length, uniform needle position
    - Just measures whether the model learned the task

Final evaluation:
    - 2D grid over (seq_len, needle_fraction)
    - Tests length extrapolation by validating at longer lengths than trained
    - Outputs an accuracy heatmap per attention method

Vocabulary layout:
    [0 .. FILLER_VOCAB-1]                              filler tokens
    [FILLER_VOCAB .. FILLER_VOCAB+NUM_PAIRS-1]         key tokens
    [FILLER_VOCAB+NUM_PAIRS .. FILLER_VOCAB+2*PAIRS-1] value tokens
    [FILLER_VOCAB+2*NUM_PAIRS]                         SEP token

Attention types tested:
    - elsaa            : Causal sortLSH sparse + Causal RACE + m_sparse fusion
    - causal_race      : RACE branch only
    - exact            : SDPA causal exact attention
    - linear           : Causal linear attention (ELU+1 kernel)
    - performer        : Causal Performer (FAVOR+ random Fourier features)

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
    """Causal ELSAA: shared Q/K/V, separate W_O per branch, post-projection fusion."""

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
        out = g_sparse * m_sparse * out_sparse + g_race * (1 - m_sparse) * out_race

        self.last_gates = gates.detach()
        self.last_m_sparse = m_sparse.detach()
        self.last_lambda = lam.detach() if isinstance(lam, torch.Tensor) else lam.detach()
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
# SECTION 9: CAUSAL LM WITH INTERPOLATED POSITIONAL EMBEDDINGS
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
        self.head = nn.Linear(d, self.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def _get_pos_emb(self, T):
        """Return positional embeddings for length T.
        If T <= train_pos_size, return first T positions.
        If T > train_pos_size, linearly interpolate.
        """
        if T <= self.train_pos_size:
            return self.pos_emb[:, :T, :]
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
# SECTION 10: NIAH DATASET
# ============================================================================

NIAH_FILLER_VOCAB = 50
NIAH_NUM_PAIRS    = 64
NIAH_SEP          = NIAH_FILLER_VOCAB + 2 * NIAH_NUM_PAIRS
NIAH_VOCAB_SIZE   = NIAH_FILLER_VOCAB + 2 * NIAH_NUM_PAIRS + 1


def niah_key(pair_idx):
    return NIAH_FILLER_VOCAB + pair_idx


def niah_val(pair_idx):
    return NIAH_FILLER_VOCAB + NIAH_NUM_PAIRS + pair_idx


class NIAHDataset(Dataset):
    """Needle In A Haystack dataset with configurable number of needles.

    For num_needles=1 (default):
        Total raw sequence = seq_len + 2
        = filler_len (= seq_len - 2) + needle (2 tokens) + SEP + query KEY
        Returns: x, y, last_mask (True only at last position)

    For num_needles > 1:
        Total raw sequence = seq_len + (num_needles + 1)
        = filler + (KEY VAL) * num_needles + SEP + query KEY
        The query KEY is randomly selected from one of the num_needles keys
        Returns: x, y, last_mask (True only at last position)

    needle_pos_frac:
        None  -> uniform random in [0.05, 0.95] per example
        float -> fixed fraction across all examples
    """

    def __init__(
        self,
        num_examples,
        seq_len,
        num_needles=1,          # NEW: configurable number of needles
        needle_pos_frac=None,
        seed=0,
        needle_pos_low=0.05,
        needle_pos_high=0.95,
    ):
        super().__init__()
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.num_needles = num_needles
        self.needle_pos_frac = needle_pos_frac
        self.needle_pos_low = needle_pos_low
        self.needle_pos_high = needle_pos_high

        # Adjust filler length based on number of needles
        # Each needle takes 2 tokens (KEY, VAL), plus 1 SEP, plus 1 query KEY
        self.filler_len = seq_len - (2 * num_needles)
        assert self.filler_len > 0, f"seq_len must be > {2 * num_needles}"

        self._build(seed)

    def _build(self, seed):
        rng = np.random.RandomState(seed)
        self.examples = []

        for _ in range(self.num_examples):
            # Sample num_needles distinct key-value pairs
            pair_indices = rng.choice(NIAH_NUM_PAIRS, size=self.num_needles, replace=False)
            
            # Generate filler
            filler = rng.randint(0, NIAH_FILLER_VOCAB, size=self.filler_len).tolist()
            
            # Determine positions for each needle
            if self.needle_pos_frac is None:
                # Uniform random positions for each needle
                fracs = sorted(rng.uniform(self.needle_pos_low, self.needle_pos_high, size=self.num_needles))
            else:
                # For multi-needle with fixed frac, space them evenly around that fraction
                if self.num_needles == 1:
                    fracs = [self.needle_pos_frac]
                else:
                    # Space needles evenly in range [frac-0.1, frac+0.1]
                    spread = 0.2 / (self.num_needles - 1) if self.num_needles > 1 else 0
                    fracs = [self.needle_pos_frac - 0.1 + i * spread for i in range(self.num_needles)]
                    fracs = [max(self.needle_pos_low, min(self.needle_pos_high, f)) for f in fracs]
            
            # Build sequence with needles inserted at positions
            seq_parts = []
            last_pos = 0
            
            for i, (pair_idx, frac) in enumerate(zip(pair_indices, fracs)):
                # Calculate position in remaining filler
                remaining_filler = self.filler_len - last_pos
                if i < self.num_needles - 1:
                    # Not the last needle, place proportionally
                    rel_pos = int(frac * self.filler_len)
                    needle_pos = max(last_pos, min(rel_pos, self.filler_len - (self.num_needles - i) * 2))
                else:
                    # Last needle, use remaining space
                    needle_pos = self.filler_len - 2
                
                # Add filler before this needle
                seq_parts.extend(filler[last_pos:needle_pos])
                
                # Add needle
                key_tok = niah_key(pair_idx)
                val_tok = niah_val(pair_idx)
                seq_parts.extend([key_tok, val_tok])
                
                last_pos = needle_pos
            
            # Add remaining filler
            seq_parts.extend(filler[last_pos:])
            
            # Add SEP and query (randomly select one of the needles to query)
            query_idx = rng.randint(0, self.num_needles)
            query_key = niah_key(pair_indices[query_idx])
            query_val = niah_val(pair_indices[query_idx])
            
            seq_parts.append(NIAH_SEP)
            seq_parts.append(query_key)
            
            # Verify sequence length
            expected_len = self.seq_len + 2  # +1 for SEP, +1 for query
            assert len(seq_parts) == expected_len, f"Expected {expected_len}, got {len(seq_parts)}"
            
            x = torch.tensor(seq_parts[:-1], dtype=torch.long)
            y = torch.tensor(seq_parts[1:], dtype=torch.long)
            
            # Mask: True only at last position (where target = query_val)
            last_mask = torch.zeros(self.seq_len + 1, dtype=torch.bool)
            last_mask[-1] = True
            
            self.examples.append((x, y, last_mask))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================================
# SECTION 11: TRAIN/VALIDATION + GRID EVALUATION
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


def _run_loader(model, loader, train=False, optimizer=None, scheduler=None,
                grad_accum_steps=1, attn_type=None, attn_stats_acc=None,
                desc=""):
    """Unified train/eval inner loop. Returns (avg_loss, accuracy)."""
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    n_examples = 0

    if train:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
    else:
        model.eval()

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y, mask in pbar:
        x = x.to(DEVICE); y = y.to(DEVICE); mask = mask.to(DEVICE)

        if train:
            logits = model(x)
        else:
            with torch.no_grad():
                logits = model(x)

        per_pos_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape_as(y)
        mask_f = mask.float()
        denom = mask_f.sum().clamp_min(1.0)
        loss = (per_pos_loss * mask_f).sum() / denom

        # Accuracy: prediction at last position vs target at last position
        with torch.no_grad():
            pred = logits[:, -1, :].argmax(dim=-1)
            target = y[:, -1]
            correct += (pred == target).sum().item()
            n_examples += x.size(0)

        if train:
            (loss / grad_accum_steps).backward()
            accum += 1
            if accum == grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

        total_loss += (per_pos_loss * mask_f).sum().item()
        total_tokens += mask.sum().item()

        if attn_stats_acc is not None and not train:
            s = collect_attn_stats(model, attn_type)
            for k, v in s.items():
                attn_stats_acc[k].append(v)

        avg = total_loss / max(1, total_tokens)
        acc = correct / max(1, n_examples)
        pbar.set_postfix({"loss": avg, "acc": acc})

    if train and accum > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, total_tokens)
    accuracy = correct / max(1, n_examples)
    return avg_loss, accuracy


def train_one_run(model, train_dl, valid_dl, optimizer, cfg, attn_type,
                  num_epochs, grad_accum_steps, log_to_wandb=True):
    steps_per_epoch = len(train_dl)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates = num_epochs * updates_per_epoch
    warmup_updates = max(1, int(0.02 * total_updates))
    scheduler = LinearWarmupLR(optimizer, warmup_updates, total_updates)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # TRAIN
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        tr_loss, tr_acc = _run_loader(
            model, train_dl, train=True,
            optimizer=optimizer, scheduler=scheduler,
            grad_accum_steps=grad_accum_steps,
            desc=f"Ep{epoch} train",
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0

        # VALIDATE
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        attn_stats_acc = defaultdict(list)
        va_loss, va_acc = _run_loader(
            model, valid_dl, train=False,
            attn_type=attn_type, attn_stats_acc=attn_stats_acc,
            desc=f"Ep{epoch} val",
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        val_time = time.time() - t1

        best_val_acc = max(best_val_acc, va_acc)
        cur_lr = scheduler.get_last_lr()[0]

        log = {
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss": va_loss, "val/acc": va_acc,
            "val/best_acc": best_val_acc,
            "lr": cur_lr,
            "time/train_sec": train_time, "time/val_sec": val_time,
        }
        for k, vlist in attn_stats_acc.items():
            log[k] = float(np.mean(vlist))

        if log_to_wandb and HAS_WANDB:
            wandb.log(log, step=epoch)

        print(f"Ep{epoch:3d} | tr_loss {tr_loss:.4f} acc {tr_acc:.3f} ({train_time:.1f}s) "
              f"| va_loss {va_loss:.4f} acc {va_acc:.3f} ({val_time:.1f}s) "
              f"| best {best_val_acc:.3f} | lr {cur_lr:.2e}")

    return best_val_acc


def check_memory_for_length(model, seq_len, vocab_size, device=DEVICE):
    """Dry run to check if seq_len fits in memory."""
    if device != "cuda":
        return True
    
    try:
        print(f"[memory check] Testing seq_len={seq_len} ...")
        dummy_x = torch.randint(0, vocab_size, (1, seq_len + 1), device=device)
        with torch.no_grad():
            _ = model(dummy_x)
        del dummy_x
        torch.cuda.empty_cache()
        print(f"[memory check] ✓ seq_len={seq_len} fits in memory")
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[memory check] ✗ seq_len={seq_len} OOM - skipping this length")
            torch.cuda.empty_cache()
            return False
        else:
            raise e


@torch.no_grad()
def evaluate_grid(model, cfg, attn_type, eval_seq_lengths, eval_needle_fracs,
                  num_eval_per_cell=500, log_to_wandb=True):
    """Memory-efficient 2D evaluation grid with robust OOM handling.
    
    Processes ONE grid cell at a time with batch=1 to maximize sequence length.
    All 500 examples are used per cell for statistical confidence.
    
    If a sequence length causes OOM, that length is skipped entirely and
    evaluation continues with the next length.
    
    Also tracks ELSAA parameters (gates, lambda, m_sparse, denominators) per length.
    """
    model.eval()
    print("\n[grid] 2D evaluation: (seq_len x needle_frac) -> accuracy")
    print(f"  sequence lengths: {eval_seq_lengths}")
    print(f"  needle fractions: {eval_needle_fracs}")
    print(f"  examples per cell: {num_eval_per_cell}, batch_size: 1 (maximizing seq len)")

    results = {}
    skipped_lengths = []
    
    # NEW: Track parameters per length for ELSAA
    params_per_length = defaultdict(lambda: defaultdict(list))

    for L in eval_seq_lengths:
        # Memory check before processing this length
        try:
            if not check_memory_for_length(model, L, cfg["vocab_size"]):
                print(f"[skip] seq_len={L} doesn't fit in memory, skipping all positions at this length")
                skipped_lengths.append(L)
                continue
        except Exception as e:
            print(f"[skip] Memory check failed for seq_len={L}: {e}")
            skipped_lengths.append(L)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue
        
        # Flag to track if this length should be skipped after first OOM
        length_oom = False
        
        for frac in eval_needle_fracs:
            if length_oom:
                # Skip remaining fractions for this length
                continue
                
            print(f"\n[grid] Processing seq_len={L}, needle_frac={frac:.2f} ...")
            
            try:
                # Generate dataset for this cell
                ds = NIAHDataset(
                    num_examples=num_eval_per_cell,
                    seq_len=L,
                    num_needles=cfg.get("num_needles", 1),
                    needle_pos_frac=frac,
                    seed=10_000 + L * 100 + int(frac * 1000),
                )
                
                # ALWAYS batch=1 to maximize seq_len capacity
                dl = DataLoader(
                    ds, 
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )

                correct = 0
                total = 0
                
                pbar = tqdm(dl, desc=f"  L={L} f={frac:.2f}", leave=False)
                for x, y, _ in pbar:
                    try:
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)

                        logits = model(x)
                        pred = logits[0, -1, :].argmax()
                        target = y[0, -1]
                        
                        correct += (pred == target).item()
                        total += 1
                        
                        # NEW: Collect ELSAA parameters if applicable
                        if attn_type in ("elsaa", "elsaa_lambda"):
                            for li, blk in enumerate(model.layers):
                                att = getattr(blk, "att", None)
                                if att is None:
                                    continue
                                
                                # Gates
                                if hasattr(att, "last_gates") and att.last_gates is not None:
                                    g = att.last_gates
                                    params_per_length[L][f"layer{li}_gate_sparse"].append(g[..., 0].mean().item())
                                    params_per_length[L][f"layer{li}_gate_race"].append(g[..., 1].mean().item())
                                
                                # m_sparse
                                if hasattr(att, "last_m_sparse") and att.last_m_sparse is not None:
                                    params_per_length[L][f"layer{li}_m_sparse"].append(att.last_m_sparse.mean().item())
                                
                                # Lambda
                                if hasattr(att, "last_lambda") and att.last_lambda is not None:
                                    if isinstance(att.last_lambda, torch.Tensor):
                                        params_per_length[L][f"layer{li}_lambda"].append(att.last_lambda.float().mean().item())
                                    else:
                                        params_per_length[L][f"layer{li}_lambda"].append(float(att.last_lambda))
                                
                                # Denominators
                                if hasattr(att, "last_d_sparse_mean") and att.last_d_sparse_mean is not None:
                                    params_per_length[L][f"layer{li}_d_sparse"].append(att.last_d_sparse_mean.item())
                                
                                if hasattr(att, "last_d_race_mean") and att.last_d_race_mean is not None:
                                    params_per_length[L][f"layer{li}_d_race"].append(att.last_d_race_mean.item())
                        
                        pbar.set_postfix({"acc": f"{correct/total:.3f}"})
                        
                        # Free memory immediately
                        del x, y, logits, pred, target
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"\n[OOM] seq_len={L} frac={frac:.2f} - stopping this length")
                            length_oom = True
                            if L not in skipped_lengths:
                                skipped_lengths.append(L)
                            if DEVICE == "cuda":
                                torch.cuda.empty_cache()
                            break
                        else:
                            raise e

                if not length_oom and total > 0:
                    acc = correct / total
                    results[(L, frac)] = acc
                    print(f"  ✓ seq_len={L:6d}  needle@{frac:.2f}  acc={acc:.3f}  ({correct}/{total})")

                    if log_to_wandb and HAS_WANDB:
                        wandb.log({f"grid_acc/L{L}_f{frac:.2f}": acc})
                
                # Cache clear between cells
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] seq_len={L} frac={frac:.2f} during setup - skipping this cell")
                    length_oom = True
                    if L not in skipped_lengths:
                        skipped_lengths.append(L)
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"[error] Unexpected error at seq_len={L} frac={frac:.2f}: {e}")
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    continue
            except Exception as e:
                print(f"[error] Unexpected error at seq_len={L} frac={frac:.2f}: {e}")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue

    if skipped_lengths:
        print(f"\n[summary] Skipped lengths due to OOM: {skipped_lengths}")
    
    return results, params_per_length


def print_heatmap(results, seq_lengths, needle_fracs):
    """Pretty-print accuracy heatmap."""
    print("\n" + "=" * 70)
    print("  NIAH Accuracy Heatmap")
    print("=" * 70)
    header = "  seq_len \\ needle_frac"
    for f in needle_fracs:
        header += f"   {f:.2f}"
    print(header)
    for L in seq_lengths:
        row = f"  {L:>10d}        "
        for f in needle_fracs:
            acc = results.get((L, f), 0.0)
            row += f"  {acc:.3f}"
        print(row)
    print("=" * 70 + "\n")


def plot_params_vs_length(params_per_length, attn_type, results_dir="./niah_results"):
    """Plot ELSAA parameters (gates, lambda, m_sparse, denominators) vs sequence length.
    
    Creates a multi-panel figure showing how each parameter evolves with context length.
    Saves both the plot and raw data to results_dir.
    """
    if not params_per_length or attn_type not in ("elsaa", "elsaa_lambda"):
        print("[plot] No parameter data to plot (not ELSAA)")
        return
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("[plot] matplotlib not installed, skipping parameter plots")
        return
    
    # Extract sequence lengths (sorted)
    seq_lengths = sorted(params_per_length.keys())
    if not seq_lengths:
        print("[plot] No data collected for plotting")
        return
    
    # Identify layers by looking at parameter keys
    sample_keys = list(params_per_length[seq_lengths[0]].keys())
    layers = set()
    for key in sample_keys:
        if key.startswith("layer"):
            layer_num = int(key.split("_")[0].replace("layer", ""))
            layers.add(layer_num)
    layers = sorted(layers)
    
    print(f"\n[plot] Generating parameter plots for {len(seq_lengths)} lengths, {len(layers)} layers")
    
    # Prepare data structures: {param_name: {layer: {length: mean_value}}}
    param_categories = ["gate_sparse", "gate_race", "m_sparse", "lambda", "d_sparse", "d_race"]
    data = {cat: {li: {} for li in layers} for cat in param_categories}
    
    for L in seq_lengths:
        for li in layers:
            for cat in param_categories:
                key = f"layer{li}_{cat}"
                if key in params_per_length[L] and params_per_length[L][key]:
                    # Average over all examples at this length
                    data[cat][li][L] = np.mean(params_per_length[L][key])
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"ELSAA Parameters vs Sequence Length ({attn_type})", fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
    
    # Plot each parameter category
    plot_configs = [
        ("gate_sparse", "Gate (Sparse Branch)", axes[0, 0]),
        ("gate_race", "Gate (RACE Branch)", axes[0, 1]),
        ("m_sparse", "m_sparse (Denominator Correction)", axes[1, 0]),
        ("lambda", "Lambda (Branch Balance)", axes[1, 1]),
        ("d_sparse", "Denominator (Sparse)", axes[2, 0]),
        ("d_race", "Denominator (RACE)", axes[2, 1]),
    ]
    
    for cat, title, ax in plot_configs:
        for li, color in zip(layers, colors):
            if data[cat][li]:
                lengths = sorted(data[cat][li].keys())
                values = [data[cat][li][L] for L in lengths]
                ax.plot(lengths, values, marker='o', label=f"Layer {li}", 
                       color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel("Sequence Length", fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Format x-axis to show powers of 2
        from matplotlib.ticker import FuncFormatter
        def format_func(value, tick_number):
            if value >= 1024:
                return f"{int(value/1024)}K"
            else:
                return f"{int(value)}"
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save plot
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f"params_vs_length_{attn_type}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[plot] Saved parameter plot: {plot_path}")
    plt.close()
    
    # Save raw data to JSON
    data_path = os.path.join(results_dir, f"params_vs_length_{attn_type}.json")
    # Convert to serializable format
    data_serializable = {}
    for cat in param_categories:
        data_serializable[cat] = {}
        for li in layers:
            data_serializable[cat][f"layer{li}"] = {str(L): v for L, v in data[cat][li].items()}
    
    with open(data_path, "w") as f:
        json.dump(data_serializable, f, indent=2)
    print(f"[plot] Saved parameter data: {data_path}")
    
    # Print summary statistics
    print(f"\n[plot] Parameter summary across lengths:")
    for cat in param_categories:
        print(f"\n  {cat}:")
        for li in layers:
            if data[cat][li]:
                all_vals = list(data[cat][li].values())
                print(f"    Layer {li}: min={min(all_vals):.4f}, max={max(all_vals):.4f}, "
                      f"mean={np.mean(all_vals):.4f}, std={np.std(all_vals):.4f}")
    
    # Log to wandb if available
    if HAS_WANDB and wandb.run is not None:
        wandb.log({"param_plot": wandb.Image(plot_path)})
        print("[plot] Logged plot to wandb")


# ============================================================================
# SECTION 12: EXPERIMENT RUNNER
# ============================================================================

DEFAULT_CFG = {
    # Transformer
    "embed_dim": 256,
    "num_heads": 4,
    "mlp_dim": 1024,
    "num_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    # NIAH task
    "train_seq_len": 512,           # length used during training
    "num_train_examples": 200_000,  # large to ensure confident generalization
    "num_valid_examples": 2_000,
    "train_pos_size": 513,          # = train_seq_len + 1 (positional embedding table)
    "vocab_size": NIAH_VOCAB_SIZE,
    "num_needles": 1,               # NEW: number of needles per sequence (1 = single-needle)
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
    # Performer baseline (NEW)
    "performer_num_features": None,  # None = same as d_k (default), or specify int
    "performer_chunk_size": 128,
    "performer_ortho_scaling": False,
    # ELSAA
    "mexact_eps": 1e-6,
    "mexact_lambda_init": 1.0,
    "lambda_offset_init": 0.3,
    "lambda_init_target": 0.8,
    "gate_hidden_dim": 128,
}

# Final grid evaluation
EVAL_SEQ_LENGTHS  = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
EVAL_NEEDLE_FRACS = [0.5]


def run_experiment(attn_type, num_epochs=20, batch_size=64,
                   grad_accum_steps=1, lr=3e-4, weight_decay=0.01,
                   wandb_project="ELSAA_NIAH",
                   results_dir="./niah_results", **overrides):
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    cfg["batch_size"] = batch_size

    train_seq_len = cfg["train_seq_len"]
    cfg["train_pos_size"] = train_seq_len + 1

    print(f"[niah] vocab_size={NIAH_VOCAB_SIZE}  "
          f"filler={NIAH_FILLER_VOCAB} key/val_pairs={NIAH_NUM_PAIRS}")
    print(f"[niah] train_seq_len={train_seq_len}, "
          f"num_train={cfg['num_train_examples']}, "
          f"num_valid={cfg['num_valid_examples']}, "
          f"num_needles={cfg.get('num_needles', 1)}")

    # Build training data (uniform random needle position)
    train_ds = NIAHDataset(
        num_examples=cfg["num_train_examples"],
        seq_len=train_seq_len,
        num_needles=cfg.get("num_needles", 1),
        needle_pos_frac=None,  # uniform
        seed=0,
    )
    # Validation at training length (uniform random needle position)
    valid_ds = NIAHDataset(
        num_examples=cfg["num_valid_examples"],
        seq_len=train_seq_len,
        num_needles=cfg.get("num_needles", 1),
        needle_pos_frac=None,  # uniform
        seed=1,
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=(DEVICE == "cuda"),
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(DEVICE == "cuda"),
    )

    print(f"\n=== NIAH | Method: {attn_type} | "
          f"train_seq={train_seq_len} ===")
    model = CausalLM(cfg, attn_type, device=DEVICE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params/1e6:.2f}M params, layers={cfg['num_layers']}, "
          f"d={cfg['embed_dim']}, h={cfg['num_heads']}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    log_wandb = HAS_WANDB
    if log_wandb:
        run_name = f"niah_{attn_type}_L{train_seq_len}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={**cfg, "attn_type": attn_type,
                    "lr": lr, "weight_decay": weight_decay,
                    "epochs": num_epochs, "grad_accum_steps": grad_accum_steps,
                    "n_params": n_params,
                    "eval_seq_lengths": list(EVAL_SEQ_LENGTHS),
                    "eval_needle_fracs": list(EVAL_NEEDLE_FRACS)},
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

    try:
        best_acc = train_one_run(
            model, train_dl, valid_dl, optimizer, cfg, attn_type,
            num_epochs=num_epochs, grad_accum_steps=grad_accum_steps,
            log_to_wandb=log_wandb,
        )
        print(f"\n[done] {attn_type}: best valid acc = {best_acc:.3f}")

        # Final 2D grid evaluation on test sets
        grid_results, params_per_length = evaluate_grid(
            model, cfg, attn_type,
            eval_seq_lengths=EVAL_SEQ_LENGTHS,
            eval_needle_fracs=EVAL_NEEDLE_FRACS,
            num_eval_per_cell=500,  # always 500 examples per cell
            log_to_wandb=log_wandb,
        )
        print_heatmap(grid_results, EVAL_SEQ_LENGTHS, EVAL_NEEDLE_FRACS)
        
        # NEW: Plot parameters vs length for ELSAA
        plot_params_vs_length(params_per_length, attn_type, results_dir=results_dir)

        # Save results to JSON
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"niah_{attn_type}_L{train_seq_len}.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "attn_type": attn_type,
                    "best_val_acc": best_acc,
                    "grid": {f"{L}_{f:.2f}": acc for (L, f), acc in grid_results.items()},
                    "eval_seq_lengths": EVAL_SEQ_LENGTHS,
                    "eval_needle_fracs": EVAL_NEEDLE_FRACS,
                },
                f, indent=2,
            )
        print(f"[saved] results -> {results_path}")
    finally:
        if log_wandb:
            wandb.finish()


# ============================================================================
# SECTION 13: EXPERIMENT LIST
# ============================================================================

SHARED_CFG = dict(
    epochs              = 10,          # ← RECOMMENDED: 20 epochs for final results
    lr                  = 3e-4,
    weight_decay        = 0.01,
    batch_size          = 4,
    embed_dim           = 256,
    num_layers          = 4,           # ← RECOMMENDED: 4 layers for final results
    num_heads           = 4,
    mlp_dim             = 1024,
    train_seq_len       = 1024,
    num_train_examples  = 200_000,
    num_valid_examples  = 2_000,
    num_needles         = 2,           # ← TUNE THIS: 1 for single-needle, >1 for multi-needle
    hyper_min_seq_len   = 256,
    wandb_project       = "ELSAA_NIAH",
    results_dir         = "./niah_results",
)

EXPERIMENTS = [
    #dict(method="exact",        grad_accum_steps=16),
    #dict(method="elsaa",        grad_accum_steps=16),
    dict(method="causal_race",  grad_accum_steps=16),
    #dict(method="linear",       grad_accum_steps=16),
    dict(method="performer",    grad_accum_steps=16),  # NEW: Performer (RFF kernel)
    dict(method="causal_sparse",grad_accum_steps=16),
    # dict(method="elsaa_lambda", grad_accum_steps=16),
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
        results_dir       = cfg.pop("results_dir")

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
            results_dir=results_dir,
            **cfg,
        )


if __name__ == "__main__":
    main()