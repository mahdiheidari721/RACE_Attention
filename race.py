import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import GELU
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import torch

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


def calc_loss_acc_batch_race(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch, use_sketches=True)  # Forward pass with sketches
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    # Compute accuracy
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)  # Get indices of max logit
        correct = (predictions == target_batch).float()
        acc = correct.mean().item()  # Convert to scalar float
    return loss, acc

def calc_loss_acc_loader_race(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    total_acc = 0.0
    num_batches = num_batches or len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches: break
        loss, acc = calc_loss_acc_batch_race(input_batch, target_batch, model, device)
        total_loss += loss.item()
        total_acc += acc
    return total_loss / num_batches, total_acc / num_batches
