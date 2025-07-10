import torch
import numpy as np
from torch import nn
from torch.nn import GELU
import torch.nn.functional as F

# ------------ Tensor Implementation ------------
class ACE:
    def __init__(self, D_dim, K, L, device='cpu', seed=None):
        self.K = K
        self.L = L
        self.D_dim = D_dim
        self.hash_size = 2 ** K
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        # Hash planes: [L, K, D]
        self.hash_planes = torch.randn(L, K, D_dim, device=device)

        # Count arrays: [L, 2^K]
        self.arrays = torch.zeros(L, self.hash_size, device=device)

        self.n = 0
        self.mu = 0.0

    def hash(self, x):
        """Hash a single vector x: [D] → [L]"""
        projections = torch.einsum('lkd,d->lk', self.hash_planes, x)
        bits = (projections > 0).int()  # [L, K]
        powers = 2 ** torch.arange(self.K, device=self.device)
        return (bits * powers).sum(dim=-1).int()  # [L]

    def hash_batch(self, X):
        """Hash a batch X: [B, D] → [B, L]"""
        projections = torch.einsum('lkd,bd->blk', self.hash_planes, X)  # [B, L, K]
        bits = (projections > 0).int()
        powers = 2 ** torch.arange(self.K, device=self.device).view(1, 1, self.K)  # [1,1,K]
        return (bits * powers).sum(dim=-1).int()  # [B, L]

    def add(self, x):
        """Add a single vector x: [D]"""
        indices = self.hash(x)  # [L]
        incr = 0.0
        for j in range(self.L):
            h = indices[j].item()
            self.arrays[j, h] += 1
            incr += (2 * self.arrays[j, h].item() + 1) / self.L
        self.mu = (self.n * self.mu + incr) / (self.n + 1)
        self.n += 1

    def add_batch(self, X):
        """Add a batch of vectors X: [B, D]"""
        B = X.shape[0]
        indices = self.hash_batch(X)  # [B, L]
        incrs = torch.zeros(B, device=self.device)

        for j in range(self.L):
            idx = indices[:, j]  # [B]
            # Increment count array in-place
            self.arrays[j].index_add_(0, idx, torch.ones_like(idx, dtype=self.arrays.dtype))
            # Fetch updated values
            values = self.arrays[j][idx].float()
            incrs += (2 * values + 1) / self.L

        total_incr = incrs.sum().item()
        self.mu = (self.n * self.mu + total_incr) / (self.n + B)
        self.n += B

    def score(self, q):
        """Score a single query q: [D]"""
        indices = self.hash(q)  # [L]
        counts = self.arrays[torch.arange(self.L), indices]  # [L]
        return counts.float().mean().item()

    def is_anomaly(self, q, alpha):
        return self.score(q) < self.mu - alpha

    def clear(self):
        self.arrays.zero_()
        self.mu = 0.0
        self.n = 0


class RACE:
    def __init__(self, D_dim, K, L, N_M, D_out, device="cpu", seed=None):
        self.N_M = N_M
        self.D_out = D_out
        self.device = device
        self.L = L
        self.hash_size = 2 ** K
        self.K = K

        self.aces = [
            ACE(D_dim, K=K, L=L, device=device, seed=(seed if seed is not None else None))
            for _ in range(N_M)
        ]

        # Value accumulators: [N_M, L, hash_size, D_out]
        self.ases = torch.zeros(N_M, L, self.hash_size, D_out, dtype=torch.float32, device=device)

    def add(self, x, v):
        """Add single key-value pair to all ACEs."""
        for m, ace in enumerate(self.aces):
            indices = ace.hash(x)  # [L]
            for l in range(ace.L):
                h = indices[l].item()
                ace.arrays[l, h] += 1
                self.ases[m, l, h] += v

    def add_batch(self, keys, values):
        """
        Add a batch of keys and corresponding values.
        keys:   [B, D]
        values: [B, D_out]
        """
        B = keys.shape[0]
        assert values.shape[0] == B

        for m, ace in enumerate(self.aces):
            indices = ace.hash_batch(keys)  # [B, L]

            for l in range(self.L):
                idx_l = indices[:, l]  # [B]
                # Increment ACE counters
                ace.arrays[l].index_add_(0, idx_l, torch.ones_like(idx_l, dtype=ace.arrays.dtype))

                # Accumulate values: for each b in [B], add values[b] to self.ases[m, l, idx_l[b]]
                self.ases[m, l].index_add_(0, idx_l, values)

    def score(self, q):
        """
        Query vector q. Returns median of per-ACE (v_sum / count) estimates.
        """
        per_ace_estimates = []

        for m, ace in enumerate(self.aces):
            indices = ace.hash(q)  # [L]
            v_sum = torch.zeros(self.D_out, device=self.device)
            count = 0
            for l in range(self.L):
                h = indices[l].item()
                count += ace.arrays[l, h].item()
                v_sum += self.ases[m, l, h]
            avg_v = v_sum / (count + 1e-6)
            per_ace_estimates.append(avg_v)

        per_ace_estimates = torch.stack(per_ace_estimates, dim=0)  # [N_M, D_out]
        return per_ace_estimates.median(dim=0).values  # [D_out]


    def clear(self):
        for ace in self.aces:
            ace.clear()
        self.ases.zero_()

# ----------------------------------------------------------

# ------------------ NUMPY Implementation ------------------
class ACENumpy:
    def __init__(self, D_dim, K, L, seed=None):
        self.K = K
        self.L = L
        self.D_dim = D_dim
        self.hash_size = 2 ** K

        if seed is not None:
            np.random.seed(seed)

        self.hash_planes = np.random.randn(L, K, D_dim).astype(np.float32)  # [L, K, D]
        self.arrays = np.zeros((L, self.hash_size), dtype=np.int32)

        self.n = 0
        self.mu = 0.0

    def hash(self, x):
        projections = np.einsum('lkd,d->lk', self.hash_planes, x)  # [L, K]
        bits = (projections > 0).astype(np.int32)
        powers = (2 ** np.arange(self.K)).astype(np.int32)
        return np.sum(bits * powers, axis=-1)  # [L]

    def add(self, x):
        indices = self.hash(x)  # [L]
        incr = 0.0
        for j in range(self.L):
            h = indices[j]
            self.arrays[j, h] += 1
            incr += (2 * self.arrays[j, h] + 1) / self.L
        self.mu = (self.n * self.mu + incr) / (self.n + 1)
        self.n += 1

    def score(self, q):
        indices = self.hash(q)  # [L]
        counts = np.array([self.arrays[j, indices[j]] for j in range(self.L)], dtype=np.float32)
        return counts.mean()

    def is_anomaly(self, q, alpha):
        return self.score(q) < self.mu - alpha

    def clear(self):
        self.arrays.fill(0)
        self.mu = 0.0
        self.n = 0


class RACENumpy:
    def __init__(self, D_dim, K, L, N_M, D_out, seed=None):
        self.N_M = N_M
        self.D_out = D_out
        self.L = L
        self.K = K
        self.hash_size = 2 ** K

        self.aces = [ACENumpy(D_dim, K=K, L=L, seed=seed) for _ in range(N_M)]
        self.ases = np.zeros((N_M, L, self.hash_size, D_out), dtype=np.float32)

    def add(self, x, v):
        for m, ace in enumerate(self.aces):
            indices = ace.hash(x)  # [L]
            for l in range(self.L):
                h = indices[l]
                ace.arrays[l, h] += 1
                self.ases[m, l, h] += v

    def score(self, q):
        estimates = []
        for m, ace in enumerate(self.aces):
            indices = ace.hash(q)  # [L]
            v_sum = np.zeros(self.D_out, dtype=np.float32)
            count = 0
            for l in range(self.L):
                h = indices[l]
                count += ace.arrays[l, h]
                v_sum += self.ases[m, l, h]
            avg_v = v_sum / (count + 1e-6)
            estimates.append(avg_v)

        estimates = np.stack(estimates, axis=0)  # [N_M, D_out]
        return np.median(estimates, axis=0)  # [D_out]

    def clear(self):
        for ace in self.aces:
            ace.clear()
        self.ases.fill(0)

# ----------------------------------------------------------

# ------------------ RACE Model ----------------------------
class RACEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[RACEBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        _ , seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class RACEAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(B, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(B, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(B, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        context_vec = torch.zeros_like(queries)  # (B, H, T, D_h)
        sketches = [
            RACENumpy(D_dim=self.d_out, K=18, L=8, N_M=10, D_out=self.d_out)
            for _ in range(B)
        ]

        # Collapse heads for sketching: reshape to [B, T, D]
        q_flat = queries.transpose(1, 2).contiguous().view(B, num_tokens, self.d_out)
        k_flat = keys.transpose(1, 2).contiguous().view(B, num_tokens, self.d_out)
        v_flat = values.transpose(1, 2).contiguous().view(B, num_tokens, self.d_out)

        k_flat_np = k_flat.detach().numpy()  # shape [B, T, D]
        v_flat_np = v_flat.detach().numpy()
        q_flat_np = q_flat.detach().numpy()
        for b in range(B):
            sketch = sketches[b]
            for t in range(num_tokens):
                k_np = k_flat_np[b, t]
                v_np = v_flat_np[b, t]
                q_np = q_flat_np[b, t]

                sketch.add(k_np, v_np)
                context_np = sketch.score(q_np)  # [D_out] numpy array

                # Convert back to torch tensor and write to context_vec
                context_tensor = torch.tensor(context_np, device=x.device, dtype=x.dtype)
                context_vec[b, :, t, :] = context_tensor.view(self.num_heads, self.head_dim)
            
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(B, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection
        return context_vec
    
class RACEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = RACEAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ## Expansion
            GELU(), ## Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ## Contraction
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
# -------------------------------------------------------------------
