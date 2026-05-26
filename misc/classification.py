# ==================================================
# IMDB long classification with Softmax / Angular / RACE /
# Hyper-LSH / Hyper-RACE / Hyper-RACE-mexact
# ==================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import random
import math
import time
import itertools
from math import ceil
from collections import Counter
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.attention import sdpa_kernel, SDPBackend
from datasets import load_dataset
import wandb

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ==================================================
# 0) User-facing hyperparameters
# ==================================================
VOCAB_LIMIT = 50_000
MAX_LEN = 512
BATCH = 32
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_WANDB = True

# Default experiment list. Edit this list as you like.
RUN_ATTENTION_TYPES = [
    #"race",
    #"hyper_lsh",
    #"hyper_race_mexact",
     "hyper_race",
    # "race",
    # "softmax",
    # "angular",
]

DEFAULT_EPOCHS = 150
DEFAULT_LR = 1e-5
DEFAULT_WD = 5e-5

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ==================================================
# 1) basic_english tokenizer
# ==================================================
_basic_english_re = re.compile(
    r"""([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])   # punctuation
     |(\d+[%]?)                                  # numbers
     |([A-Za-z]+(?:'[A-Za-z]+)?)                 # words
    """,
    re.VERBOSE,
)


def basic_english_tokenizer(text: str) -> List[str]:
    text = text.lower()
    tokens = []
    for punc, num, word in _basic_english_re.findall(text):
        if punc:
            tokens.append(punc)
        elif num:
            tokens.append(num)
        elif word:
            tokens.append(word)
    return tokens


tok = basic_english_tokenizer


# ==================================================
# 2) EDA augmenters
# ==================================================
def eda_random_deletion(tokens, p=0.05):
    if len(tokens) == 1:
        return tokens
    out = [t for t in tokens if random.random() > p]
    return out or [random.choice(tokens)]


def eda_random_swap(tokens, n_swaps=3):
    toks = tokens.copy()
    if len(toks) < 2:
        return toks
    for _ in range(n_swaps):
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    return toks


# ==================================================
# 3) Load IMDB and build vocab
# ==================================================
print("Loading IMDB dataset...")
raw = load_dataset("imdb")
train_examples = list(zip(raw["train"]["label"], raw["train"]["text"]))
test_examples = list(zip(raw["test"]["label"], raw["test"]["text"]))

print("Building vocabulary...")
counter = Counter()
for lbl, txt in train_examples:
    counter.update(tok(txt))
most_common = [w for w, _ in counter.most_common(VOCAB_LIMIT)]
stoi = {w: i + 2 for i, w in enumerate(most_common)}
stoi["<pad>"] = 0
stoi["<unk>"] = 1
PAD_IDX, UNK_IDX = 0, 1
VOCAB_SIZE = len(stoi)
print(f"Vocab size: {VOCAB_SIZE}")


# ==================================================
# 4) Augmented IMDB dataset
# ==================================================
class AugmentedIMDB(Dataset):
    def __init__(self, examples, max_len, augment=False):
        self.examples = examples
        self.max_len = int(max_len)
        self.augment = augment
        self.pad_idx = PAD_IDX

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        lbl, txt = self.examples[idx]
        toks = tok(txt)
        if self.augment:
            op = random.choice(["del", "swap", None])
            if op == "del":
                toks = eda_random_deletion(toks)
            elif op == "swap":
                toks = eda_random_swap(toks)
        toks = toks[: self.max_len]
        ids = [stoi.get(t, UNK_IDX) for t in toks]
        if len(ids) < self.max_len:
            ids += [PAD_IDX] * (self.max_len - len(ids))
        return int(lbl), torch.tensor(ids, dtype=torch.long)

    def collate_fn(self, batch):
        labels, texts = zip(*batch)
        texts = torch.stack(texts, dim=0)
        masks = (texts != PAD_IDX).long()
        return texts, masks, torch.tensor(labels, dtype=torch.long)


# ==================================================
# 5) Make long IMDB subsets
# ==================================================
def make_long_subsets(examples):
    random.seed(SEED)
    real_long, short_pos, short_neg, super_long = [], [], [], []
    for lbl, txt in examples:
        L = len(tok(txt))
        if MAX_LEN <= L < 2 * MAX_LEN:
            real_long.append((lbl, txt))
        elif L >= 2 * MAX_LEN:
            super_long.append((lbl, txt))
        else:
            (short_pos if lbl == 1 else short_neg).append(txt)

    split_long = []
    for lbl, txt in super_long:
        toks = tok(txt)
        step = max(1, MAX_LEN // 2)
        num_windows = ceil((len(toks) - MAX_LEN) / step) + 1
        for w in range(num_windows):
            start = min(w * step, len(toks) - MAX_LEN)
            window = toks[start : start + MAX_LEN]
            split_long.append((lbl, " ".join(window)))
            if start + MAX_LEN >= len(toks):
                break

    random.seed(SEED)
    random.shuffle(short_pos)
    random.seed(SEED)
    random.shuffle(short_neg)

    new_long = []
    for pool, label in [(short_pos, 1), (short_neg, 0)]:
        i = 0
        while i + 1 < len(pool):
            combo = tok(pool[i]) + tok(pool[i + 1])
            if len(combo) >= MAX_LEN:
                new_long.append((label, " ".join(combo)))
                i += 2
            else:
                if i + 2 < len(pool):
                    combo3 = combo + tok(pool[i + 2])
                    if len(combo3) >= MAX_LEN:
                        new_long.append((label, " ".join(combo3)))
                        i += 3
                        continue
                i += 1

    long_train = real_long + split_long + new_long
    random.seed(SEED)
    random.shuffle(long_train)

    long_test = [(lbl, txt) for lbl, txt in examples if len(tok(txt)) >= MAX_LEN]
    random.seed(SEED)
    random.shuffle(long_test)

    return long_train, long_test


def get_data():
    long_train, _ = make_long_subsets(train_examples)
    print(f"--> FINAL long_train: {len(long_train)}")

    _, long_test = make_long_subsets(test_examples)
    print(f"--> FINAL long_test:  {len(long_test)}")

    train_ds = AugmentedIMDB(long_train, MAX_LEN, augment=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        pin_memory=(DEVICE == "cuda"),
        num_workers=4,
        collate_fn=train_ds.collate_fn,
        generator=torch.Generator().manual_seed(SEED),
    )

    test_ds = AugmentedIMDB(long_test, MAX_LEN, augment=False)
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH,
        shuffle=False,
        pin_memory=(DEVICE == "cuda"),
        num_workers=2,
        collate_fn=test_ds.collate_fn,
    )
    return train_dl, test_dl


# ==================================================
# 6) Attention helpers
# ==================================================
def _gray_code_order(num_bits: int, device):
    if num_bits == 1:
        return torch.tensor([0, 1], device=device, dtype=torch.long)

    def rec(n):
        if n == 1:
            return torch.tensor([0, 1], device=device, dtype=torch.long)
        a = rec(n - 1)
        return torch.cat([a, torch.flip(a, dims=[0]) + (1 << (n - 1))], dim=0)

    return rec(num_bits)


def _gather_tokens_3d(x: torch.Tensor, idx: torch.Tensor):
    """x: [H,T,D], idx: [H,S] -> [H,S,D]."""
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def _run_exact_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """Q,K,V: [B',H',T,D]."""
    if Q.device.type == "cuda":
        try:
            Q16, K16, V16 = [t.to(dtype=torch.float16) for t in (Q, K, V)]
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    Q16, K16, V16,
                    dropout_p=0.0,
                    is_causal=False,
                )
            return out.to(Q.dtype)
        except Exception:
            return F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=0.0,
                is_causal=False,
            )
    return F.scaled_dot_product_attention(
        Q, K, V,
        dropout_p=0.0,
        is_causal=False,
    )


class AngularLSHGray(nn.Module):
    def __init__(self, num_bits: int, dim: int, device="cpu"):
        super().__init__()
        self.num_bits = int(num_bits)
        self.R = 1 << self.num_bits
        proj_dir = torch.randn(dim, self.num_bits, device=device)
        perm = _gray_code_order(self.num_bits, device=device)
        enc_vec = (2 ** torch.arange(self.num_bits, device=device, dtype=torch.long)).view(1, 1, self.num_bits)
        self.register_buffer("proj_dir", proj_dir, persistent=False)
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("enc_vec", enc_vec, persistent=False)

    def hash(self, mat: torch.Tensor):
        # mat: [..., T, D]
        proj = torch.einsum("...td,dr->...tr", mat, self.proj_dir)
        bits = (proj > 0).to(torch.long)
        bin_ids = (bits * self.enc_vec).sum(dim=-1)
        return self.perm[bin_ids]


# ==================================================
# 7) Baseline attentions
# ==================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        K = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            pad = mask[:, None, None, :]
            scores = scores.masked_fill(pad == 0, torch.finfo(scores.dtype).min)
        W = torch.softmax(scores, dim=-1)
        W = self.drop(W)
        out = (W @ V).transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class AngularAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        B, T, _ = x.shape
        Q = F.normalize(self.q(x).view(B, T, self.h, self.dk).transpose(1, 2), dim=-1)
        K = F.normalize(self.k(x).view(B, T, self.h, self.dk).transpose(1, 2), dim=-1)
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        sim = (Q @ K.transpose(-2, -1)).clamp(-0.999, 0.999)
        scores = 1.0 - torch.acos(sim) / math.pi
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, 0.0)
        W = scores.clamp(min=1e-6).pow(18)
        W = W / (W.sum(-1, keepdim=True) + 1e-6)
        W = self.drop(W)
        out = (W @ V).transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


# ==================================================
# 8) Hyper-LSH exact sparse attention
# ==================================================
class HyperLSHExactAttentionText(nn.Module):
    def __init__(
        self,
        d,
        h,
        drop,
        num_bits=5,
        block_size=32,
        min_seq_len=256,
        neighbor_blocks=0,
        qkv_bias=False,
        device="cpu",
    ):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.block_size = int(block_size)
        self.min_seq_len = int(min_seq_len)
        self.neighbor_blocks = int(neighbor_blocks)

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.dk, device=device)

    def _full_sdpa_fallback(self, Qh, Kh, Vh):
        return _run_exact_sdpa(Qh.unsqueeze(0), Kh.unsqueeze(0), Vh.unsqueeze(0))[0]

    def _same_block_exact(self, Qs, Ks, Vs, valid_T):
        H, _, D = Qs.shape
        bsz = self.block_size
        num_full_blocks = valid_T // bsz
        rem = valid_T % bsz
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
            q_last = Qs[:, num_full_blocks * bsz :, :]
            k_last = Ks[:, num_full_blocks * bsz :, :]
            v_last = Vs[:, num_full_blocks * bsz :, :]
            o_last = _run_exact_sdpa(q_last.unsqueeze(0), k_last.unsqueeze(0), v_last.unsqueeze(0))[0]
            out_sorted[:, num_full_blocks * bsz :, :] = o_last
        return out_sorted

    def _neighbor_block_exact(self, Qs, Ks, Vs, valid_T):
        H, _, D = Qs.shape
        bsz = self.block_size
        num_blocks = math.ceil(valid_T / bsz)
        out_sorted = torch.zeros_like(Qs)

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
            o_blk = _run_exact_sdpa(q_blk.unsqueeze(0), k_blk.unsqueeze(0), v_blk.unsqueeze(0))[0]
            out_sorted[:, q0:q1, :] = o_blk
        return out_sorted

    def forward(self, x, mask):
        B, T, _ = x.shape
        H, D = self.h, self.dk
        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q, K, V = Q * keep, K * keep, V * keep

        out = torch.zeros_like(Q)
        for b in range(B):
            valid_T = int(mask[b].sum().item()) if mask is not None else T
            if valid_T == 0:
                continue
            Qh = Q[b, :, :valid_T, :]
            Kh = K[b, :, :valid_T, :]
            Vh = V[b, :, :valid_T, :]

            if valid_T < self.min_seq_len:
                out[b, :, :valid_T, :] = self._full_sdpa_fallback(Qh, Kh, Vh)
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
                O_sorted = self._same_block_exact(Qs, Ks, Vs, valid_T)
            else:
                O_sorted = self._neighbor_block_exact(Qs, Ks, Vs, valid_T)

            O_unsorted = O_sorted.gather(1, q_sort_inv.unsqueeze(-1).expand(-1, -1, D))
            out[b, :, :valid_T, :] = O_unsorted

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.drop(out)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


# ==================================================
# 9) RACE attention
# ==================================================
class BatchedACE(nn.Module):
    def __init__(self, d_k, K, L, M, device="cpu", share_planes=False):
        super().__init__()
        self.d_k = int(d_k)
        self.K = int(K)
        self.L = int(L)
        self.M = int(M)
        self.R = 1 << self.K
        self.share_planes = bool(share_planes)

        if self.share_planes:
            planes = torch.randn(self.L, self.K, self.d_k, device=device)
            self.register_buffer("planes_T", planes.view(self.L * self.K, self.d_k).T)
        else:
            planes = torch.randn(self.M, self.L, self.K, self.d_k, device=device)
            planes = planes.view(self.M, self.L * self.K, self.d_k).transpose(1, 2)
            self.register_buffer("planes_T", planes)

        corners = torch.tensor(list(itertools.product([-1.0, +1.0], repeat=self.K)), device=device)
        self.register_buffer("protos_T", corners.T)
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0, device=device)))

    def _probs_and_values(self, Khf, Vhf, Qhf):
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)

        if self.share_planes:
            N = M * B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2 = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            projK = Kh2 @ self.planes_T
            projQ = Qh2 @ self.planes_T
        else:
            BH = B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2 = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            projK = torch.einsum("mbtd,mds->mbts", Kh2, self.planes_T)
            projQ = torch.einsum("mbtd,mds->mbts", Qh2, self.planes_T)
            projK = projK.contiguous().view(M * BH, T, self.L * self.K)
            projQ = projQ.contiguous().view(M * BH, T, self.L * self.K)
            V2 = V2.contiguous().view(M * BH, T, dk)
            N = M * BH

        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)
        logitsK = (projK.tanh().div(scale) @ self.protos_T)
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)
        probsK = F.softmax(logitsK, dim=-1).contiguous().view(N, T, S)
        probsQ = F.softmax(logitsQ, dim=-1).contiguous().view(N, T, S)
        return probsK, probsQ, V2, N, S

    def forward(self, Khf, Vhf, Qhf, mask=None, eps=1e-6):
        M, B, T, H, dk = Khf.shape
        probsK, probsQ, V2, N, S = self._probs_and_values(Khf, Vhf, Qhf)

        if mask is not None:
            mask_bh = mask[:, None, :].expand(B, H, T).contiguous().view(B * H, T)
            mask_N = mask_bh.unsqueeze(0).expand(M, -1, -1).contiguous().view(N, T).to(probsK.dtype)
            probsK = probsK * mask_N.unsqueeze(-1)
            probsQ = probsQ * mask_N.unsqueeze(-1)
            V2 = V2 * mask_N.unsqueeze(-1)

        b_sum = probsK.transpose(1, 2).bmm(V2)
        A = probsK.sum(dim=1)
        E = b_sum / (A.unsqueeze(-1) + eps)
        out2 = probsQ.bmm(E)
        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4).contiguous()
        return out


class RACEAttention(nn.Module):
    def __init__(self, d, h, drop, M=2, K=3, L=2, qkv_bias=False, device="cpu"):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.M = h, d // h, M
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(self.dk, K, L, M, device=device)

    def forward(self, x, mask):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk)
        K = self.k(x).view(B, T, self.h, self.dk)
        V = self.v(x).view(B, T, self.h, self.dk)

        if mask is not None:
            m = mask[:, :, None, None].to(Q.dtype)
            Q, K, V = Q * m, K * m, V * m

        def pack(z):
            return z.unsqueeze(0).expand(self.M, -1, -1, -1, -1)

        out_m = self.ace(pack(K), pack(V), pack(Q), mask=mask)  # [M,B,T,H,dk]
        out = out_m.mean(dim=0)  # [B,T,H,dk]
        out = out.contiguous().view(B, T, self.h * self.dk)
        out = self.drop(self.o(out))
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class RACEAttentionWithDenom(nn.Module):
    def __init__(self, d, h, drop, M=2, K=3, L=2, qkv_bias=False, device="cpu"):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.M = h, d // h, M
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(self.dk, K, L, M, device=device)

    def _ace_with_denom(self, Khf, Vhf, Qhf, mask=None, eps=1e-6):
        M, B, T, H, dk = Khf.shape
        probsK, probsQ, V2, N, S = self.ace._probs_and_values(Khf, Vhf, Qhf)

        if mask is not None:
            mask_bh = mask[:, None, :].expand(B, H, T).contiguous().view(B * H, T)
            mask_N = mask_bh.unsqueeze(0).expand(M, -1, -1).contiguous().view(N, T).to(probsK.dtype)
            probsK = probsK * mask_N.unsqueeze(-1)
            probsQ = probsQ * mask_N.unsqueeze(-1)
            V2 = V2 * mask_N.unsqueeze(-1)

        total_num = probsK.transpose(1, 2).bmm(V2)
        total_den = probsK.sum(dim=1)
        E = total_num / (total_den.unsqueeze(-1) + eps)
        out2 = probsQ.bmm(E)
        d2 = torch.einsum("nts,ns->nt", probsQ, total_den).clamp_min(eps)
        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4).contiguous()  # [M,B,T,H,dk]
        den = d2.view(M, B, H, T).permute(0, 1, 3, 2).contiguous()  # [M,B,T,H]
        return out, den

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk)
        K = self.k(x).view(B, T, self.h, self.dk)
        V = self.v(x).view(B, T, self.h, self.dk)

        if mask is not None:
            m = mask[:, :, None, None].to(Q.dtype)
            Q, K, V = Q * m, K * m, V * m

        def pack(z):
            return z.unsqueeze(0).expand(self.M, -1, -1, -1, -1)

        out_m, den_m = self._ace_with_denom(pack(K), pack(V), pack(Q), mask=mask)
        out_heads = out_m.mean(dim=0)  # [B,T,H,dk]
        out = out_heads.contiguous().view(B, T, self.h * self.dk)
        out = self.drop(self.o(out))
        d_race_heads = den_m.mean(dim=0)  # [B,T,H]
        d_race_token = d_race_heads.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out, d_race_token


# ==================================================
# 10) Hyper-RACE baseline
# ==================================================
class HyperRaceGatedAttentionText(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["emb_dim"]
        gate_hidden = cfg.get("gate_hidden_dim", 64)

        self.hyper = HyperLSHExactAttentionText(
            d=d,
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )
        self.race = RACEAttention(
            d=d,
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            M=cfg.get("M", 2),
            K=cfg.get("K", 3),
            L=cfg.get("L", 2),
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

    def forward(self, x, mask):
        out_hyper = self.hyper(x, mask)
        out_race = self.race(x, mask)
        gates = torch.sigmoid(self.gate_mlp(x))
        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        self.last_gates = gates.detach()
        out = gates[..., 0:1] * out_hyper + gates[..., 1:2] * out_race
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


# ==================================================
# 11) Hyper-LSH with log denominator for mexact
# ==================================================
class HyperLSHExactWithLogDenomAttentionText(nn.Module):
    def __init__(
        self,
        d,
        h,
        drop,
        num_bits=5,
        block_size=32,
        min_seq_len=256,
        neighbor_blocks=0,
        qkv_bias=False,
        device="cpu",
        mexact_eps=1e-6,
    ):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dk = d // h
        self.block_size = int(block_size)
        self.min_seq_len = int(min_seq_len)
        self.neighbor_blocks = int(neighbor_blocks)
        self.mexact_eps = mexact_eps
        self.scale = 1.0 / math.sqrt(self.dk)

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.lsh = AngularLSHGray(num_bits=num_bits, dim=self.dk, device=device)

    def _full_sdpa_fallback_with_lse(self, Qh, Kh, Vh):
        out_h = _run_exact_sdpa(Qh.unsqueeze(0), Kh.unsqueeze(0), Vh.unsqueeze(0))[0]
        with torch.no_grad():
            logits = torch.einsum("hqd,hkd->hqk", Qh, Kh) * self.scale
            lse_h = torch.logsumexp(logits.float(), dim=-1).to(Qh.dtype)
        return out_h, lse_h

    def _same_block_exact_with_lse(self, Qs, Ks, Vs, valid_T):
        H, _, D = Qs.shape
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
            O_flat = _run_exact_sdpa(Q_flat, K_flat, V_flat)
            O_full = O_flat.reshape(H, num_full_blocks, bsz, D).reshape(H, T_full, D)
            out_sorted[:, :T_full, :] = O_full

            with torch.no_grad():
                lse_chunks = []
                block_chunk = 64
                for bs in range(0, num_full_blocks, block_chunk):
                    be = min(bs + block_chunk, num_full_blocks)
                    logits = torch.einsum(
                        "hnqd,hnkd->hnqk",
                        Q_full[:, bs:be, :, :],
                        K_full[:, bs:be, :, :],
                    ) * self.scale
                    lse_blk = torch.logsumexp(logits.float(), dim=-1).to(Qs.dtype)
                    lse_chunks.append(lse_blk)
                lse_full = torch.cat(lse_chunks, dim=1).reshape(H, T_full)
                lse_sorted[:, :T_full] = lse_full

        if rem > 0:
            q_last = Qs[:, num_full_blocks * bsz :, :]
            k_last = Ks[:, num_full_blocks * bsz :, :]
            v_last = Vs[:, num_full_blocks * bsz :, :]
            o_last = _run_exact_sdpa(q_last.unsqueeze(0), k_last.unsqueeze(0), v_last.unsqueeze(0))[0]
            out_sorted[:, num_full_blocks * bsz :, :] = o_last
            with torch.no_grad():
                logits = torch.einsum("hqd,hkd->hqk", q_last, k_last) * self.scale
                lse_last = torch.logsumexp(logits.float(), dim=-1).to(Qs.dtype)
                lse_sorted[:, num_full_blocks * bsz :] = lse_last

        return out_sorted, lse_sorted

    def _neighbor_block_exact_with_lse(self, Qs, Ks, Vs, valid_T):
        H, _, D = Qs.shape
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
            o_blk = _run_exact_sdpa(q_blk.unsqueeze(0), k_blk.unsqueeze(0), v_blk.unsqueeze(0))[0]
            out_sorted[:, q0:q1, :] = o_blk
            with torch.no_grad():
                logits = torch.einsum("hqd,hkd->hqk", q_blk, k_blk) * self.scale
                lse_blk = torch.logsumexp(logits.float(), dim=-1).to(Qs.dtype)
                lse_sorted[:, q0:q1] = lse_blk
        return out_sorted, lse_sorted

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        H, D = self.h, self.dk
        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Q, K, V = Q * keep, K * keep, V * keep

        out_heads = torch.zeros_like(Q)
        lse_heads = torch.full((B, H, T), float("-inf"), device=x.device, dtype=Q.dtype)

        for b in range(B):
            valid_T = int(mask[b].sum().item()) if mask is not None else T
            if valid_T == 0:
                continue
            Qh = Q[b, :, :valid_T, :]
            Kh = K[b, :, :valid_T, :]
            Vh = V[b, :, :valid_T, :]

            if valid_T < self.min_seq_len:
                out_b, lse_b = self._full_sdpa_fallback_with_lse(Qh, Kh, Vh)
                out_heads[b, :, :valid_T, :] = out_b
                lse_heads[b, :, :valid_T] = lse_b
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
                O_sorted, LSE_sorted = self._same_block_exact_with_lse(Qs, Ks, Vs, valid_T)
            else:
                O_sorted, LSE_sorted = self._neighbor_block_exact_with_lse(Qs, Ks, Vs, valid_T)

            O_unsorted = O_sorted.gather(1, q_sort_inv.unsqueeze(-1).expand(-1, -1, D))
            LSE_unsorted = LSE_sorted.gather(1, q_sort_inv)
            out_heads[b, :, :valid_T, :] = O_unsorted
            lse_heads[b, :, :valid_T] = LSE_unsorted

        with torch.no_grad():
            log_d_exact_token = (torch.logsumexp(lse_heads.float(), dim=1) - math.log(H)).to(Q.dtype).unsqueeze(-1)

        out = out_heads.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.drop(out)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out, log_d_exact_token


class HyperRaceMExactAttentionText(nn.Module):
    """
    Hyper-RACE with m_exact correction and learnable scalar lambda:

        m_exact_i = d_exact_i / (d_exact_i + lambda * d_race_i + eps)
        lambda = exp(log_lambda)

    By default mexact_lambda_learnable=True and mexact_lambda_init=1.0.
    """
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["emb_dim"]
        h = cfg["n_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        gate_hidden = cfg.get("gate_hidden_dim", 64)
        self.mexact_eps = cfg.get("mexact_eps", 1e-6)

        self.mexact_lambda_learnable = bool(cfg.get("mexact_lambda_learnable", True))
        lambda_init = float(cfg.get("mexact_lambda_init", 1.0))
        if lambda_init <= 0:
            raise ValueError("mexact_lambda_init must be > 0")
        if self.mexact_lambda_learnable:
            self.log_mexact_lambda = nn.Parameter(torch.tensor(math.log(lambda_init), dtype=torch.float32))
        else:
            self.register_buffer("log_mexact_lambda", torch.tensor(math.log(lambda_init), dtype=torch.float32), persistent=False)

        self.hyper = HyperLSHExactWithLogDenomAttentionText(
            d=d,
            h=h,
            drop=drop,
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=qkv_bias,
            device=device,
            mexact_eps=self.mexact_eps,
        )
        self.race = RACEAttentionWithDenom(
            d=d,
            h=h,
            drop=drop,
            M=cfg.get("M", 2),
            K=cfg.get("K", 3),
            L=cfg.get("L", 2),
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

    def forward(self, x, mask=None):
        out_hyper, log_d_exact = self.hyper(x, mask)
        out_race, d_race = self.race(x, mask)

        log_d_exact_det = log_d_exact.detach().float()
        log_d_race_det = torch.log(d_race.detach().float().clamp_min(self.mexact_eps))
        log_lambda = self.log_mexact_lambda.float()
        log_eps = torch.full_like(log_d_exact_det, math.log(self.mexact_eps))

        log_den = torch.logsumexp(
            torch.stack([log_d_exact_det, log_lambda + log_d_race_det, log_eps], dim=0),
            dim=0,
        )
        m_exact = torch.exp(log_d_exact_det - log_den).to(out_hyper.dtype)

        gates = torch.sigmoid(self.gate_mlp(x))
        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        out = gates[..., 0:1] * m_exact * out_hyper + gates[..., 1:2] * out_race
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)

        self.last_gates = gates.detach()
        self.last_m_exact = m_exact.detach()
        self.last_mexact_lambda = self.log_mexact_lambda.detach().exp()
        if mask is not None and bool(mask.any().item()):
            valid = mask.bool().unsqueeze(-1)
            de = torch.exp(log_d_exact.detach().clamp(max=20.0))
            dr = d_race.detach()
            self.last_d_exact_mean = de[valid].mean()
            self.last_d_race_mean = dr[valid].mean()
        else:
            self.last_d_exact_mean = torch.exp(log_d_exact.detach().clamp(max=20.0)).mean()
            self.last_d_race_mean = d_race.detach().mean()
        return out


# ==================================================
# 12) Transformer blocks
# ==================================================
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = MultiHeadAttention(d=d, h=cfg["n_heads"], drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, pad_mask)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = AngularAttention(d=d, h=cfg["n_heads"], drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, pad_mask)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class HyperLSHExactBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = HyperLSHExactAttentionText(
            d=d,
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, pad_mask)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class RACEBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = RACEAttention(
            d=d,
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            M=cfg.get("M", 2),
            K=cfg.get("K", 3),
            L=cfg.get("L", 2),
            qkv_bias=cfg.get("qkv_bias", False),
            device=device,
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, pad_mask)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class HyperRaceGatedBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = HyperRaceGatedAttentionText(cfg, device=device)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, pad_mask)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class HyperRaceMExactBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = HyperRaceMExactAttentionText(cfg, device=device)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, pad_mask)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


# ==================================================
# 13) Classifier
# ==================================================
class Classifier(nn.Module):
    def __init__(self, cfg: Dict[str, Any], kind: str):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.ModuleList()
        for _ in range(cfg["n_layers"]):
            if kind == "softmax":
                self.blocks.append(TransformerBlock(cfg))
            elif kind == "angular":
                self.blocks.append(AngularBlock(cfg))
            elif kind == "race":
                self.blocks.append(RACEBlock(cfg, device=DEVICE))
            elif kind == "hyper_lsh":
                self.blocks.append(HyperLSHExactBlock(cfg, device=DEVICE))
            elif kind == "hyper_race":
                self.blocks.append(HyperRaceGatedBlock(cfg, device=DEVICE))
            elif kind == "hyper_race_mexact":
                self.blocks.append(HyperRaceMExactBlock(cfg, device=DEVICE))
            else:
                raise ValueError(f"Unsupported attention kind: {kind}")

        self.norm = nn.LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], 2)

    def forward(self, x, mask):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.norm(h)
        return self.head(h[:, 0])


# ==================================================
# 14) Experiment loop
# ==================================================
def build_cfg():
    return {
        "vocab_size": VOCAB_SIZE,
        "context_length": MAX_LEN,
        "emb_dim": 128,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.1,
        "qkv_bias": False,
        # RACE config
        "M": 2,
        "K": 3,
        "L": 2,
        # Hyper-LSH sparse branch
        "hyper_num_bits": 5,
        "hyper_block_size": 32,
        "hyper_min_seq_len": 256,
        "hyper_neighbor_blocks": 0,
        # Gate MLP
        "gate_hidden_dim": 64,
        "gate_normalize": False,
        # mexact scalar lambda
        "mexact_eps": 1e-6,
        "mexact_lambda_learnable": True,
        "mexact_lambda_init": 1.0,
    }


def _collect_extra_logs(model, kind, ep):
    extra_logs = {}
    if kind not in {"hyper_race", "hyper_race_mexact"}:
        return extra_logs

    hyper_means, race_means, lambda_vals, m_means = [], [], [], []
    for layer_idx, layer in enumerate(model.blocks):
        if not hasattr(layer, "att"):
            continue
        att = layer.att
        if hasattr(att, "last_gates") and att.last_gates is not None:
            gates = att.last_gates.detach().float()
            hyper_g = gates[..., 0].reshape(-1)
            race_g = gates[..., 1].reshape(-1)
            extra_logs[f"gates/layer{layer_idx}_hyper_mean"] = hyper_g.mean().item()
            extra_logs[f"gates/layer{layer_idx}_race_mean"] = race_g.mean().item()
            extra_logs[f"gates/layer{layer_idx}_hyper_std"] = hyper_g.std(unbiased=False).item()
            extra_logs[f"gates/layer{layer_idx}_race_std"] = race_g.std(unbiased=False).item()
            hyper_means.append(hyper_g.mean().item())
            race_means.append(race_g.mean().item())
            if USE_WANDB and ep % 5 == 0:
                extra_logs[f"gates_hist/layer{layer_idx}_hyper"] = wandb.Histogram(hyper_g.cpu().numpy())
                extra_logs[f"gates_hist/layer{layer_idx}_race"] = wandb.Histogram(race_g.cpu().numpy())

        if hasattr(att, "last_m_exact") and att.last_m_exact is not None:
            m = att.last_m_exact.detach().float().reshape(-1)
            extra_logs[f"m_exact/layer{layer_idx}_mean"] = m.mean().item()
            extra_logs[f"m_exact/layer{layer_idx}_std"] = m.std(unbiased=False).item()
            extra_logs[f"m_exact/layer{layer_idx}_min"] = m.min().item()
            extra_logs[f"m_exact/layer{layer_idx}_max"] = m.max().item()
            m_means.append(m.mean().item())

        if hasattr(att, "last_mexact_lambda") and att.last_mexact_lambda is not None:
            lam = att.last_mexact_lambda.detach().float().reshape(-1)
            extra_logs[f"lambda/layer{layer_idx}_mean"] = lam.mean().item()
            extra_logs[f"lambda/layer{layer_idx}_std"] = lam.std(unbiased=False).item()
            extra_logs[f"lambda/layer{layer_idx}_min"] = lam.min().item()
            extra_logs[f"lambda/layer{layer_idx}_max"] = lam.max().item()
            lambda_vals.append(lam.mean().item())

        if hasattr(att, "last_d_exact_mean") and att.last_d_exact_mean is not None:
            extra_logs[f"den/layer{layer_idx}_exact_mean"] = float(att.last_d_exact_mean.detach().cpu())
        if hasattr(att, "last_d_race_mean") and att.last_d_race_mean is not None:
            extra_logs[f"den/layer{layer_idx}_race_mean"] = float(att.last_d_race_mean.detach().cpu())

    if hyper_means:
        extra_logs["gates/global_hyper_mean"] = sum(hyper_means) / len(hyper_means)
        extra_logs["gates/global_race_mean"] = sum(race_means) / len(race_means)
    if lambda_vals:
        extra_logs["lambda/global_mean"] = sum(lambda_vals) / len(lambda_vals)
    if m_means:
        extra_logs["m_exact/global_mean"] = sum(m_means) / len(m_means)
    return extra_logs


def run_experiment(attn_types, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, wd=DEFAULT_WD):
    cfg = build_cfg()
    train_dl, test_dl = get_data()

    for kind in attn_types:
        if USE_WANDB:
            run = wandb.init(
                project="RACE",
                name=f"imdb_{kind}_{cfg['context_length']}",
                config={
                    "dataset": "IMDB-long",
                    "attn_type": kind,
                    "context_length": cfg["context_length"],
                    "emb_dim": cfg["emb_dim"],
                    "n_heads": cfg["n_heads"],
                    "n_layers": cfg["n_layers"],
                    "drop_rate": cfg["drop_rate"],
                    "batch_size": BATCH,
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": wd,
                    "M": cfg["M"],
                    "K": cfg["K"],
                    "L": cfg["L"],
                    "hyper_num_bits": cfg["hyper_num_bits"],
                    "hyper_block_size": cfg["hyper_block_size"],
                    "hyper_min_seq_len": cfg["hyper_min_seq_len"],
                    "hyper_neighbor_blocks": cfg["hyper_neighbor_blocks"],
                    "gate_hidden_dim": cfg["gate_hidden_dim"],
                    "gate_normalize": cfg["gate_normalize"],
                    "mexact_eps": cfg["mexact_eps"],
                    "mexact_lambda_learnable": cfg["mexact_lambda_learnable"],
                    "mexact_lambda_init": cfg["mexact_lambda_init"],
                },
            )
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("time/*", step_metric="epoch")
            wandb.define_metric("gates/*", step_metric="epoch")
            wandb.define_metric("m_exact/*", step_metric="epoch")
            wandb.define_metric("lambda/*", step_metric="epoch")
            wandb.define_metric("den/*", step_metric="epoch")
            wandb.define_metric("val/acc", summary="max")
            wandb.define_metric("val/loss", summary="min")

        print(f"\n=== Training {kind.upper()} for {epochs} epochs ===")
        model = Classifier(cfg, kind).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        for ep in range(1, epochs + 1):
            # ---------------- TRAIN ----------------
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            model.train()
            tl, ta = 0.0, 0.0

            for x, mask, y in train_dl:
                x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                logits = model(x, mask)
                loss = F.cross_entropy(logits, y)
                acc = (logits.argmax(-1) == y).float().mean().item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item()
                ta += acc

            tr_l = tl / max(1, len(train_dl))
            tr_a = ta / max(1, len(train_dl))
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            train_time = time.time() - t0

            # ---------------- VALID ----------------
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            model.eval()
            vl, va = 0.0, 0.0
            with torch.no_grad():
                for x, mask, y in test_dl:
                    x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                    logits = model(x, mask)
                    vl += F.cross_entropy(logits, y).item()
                    va += (logits.argmax(-1) == y).float().mean().item()

            va_l = vl / max(1, len(test_dl))
            va_a = va / max(1, len(test_dl))
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            val_time = time.time() - t1

            extra_logs = _collect_extra_logs(model, kind, ep)
            if USE_WANDB:
                wandb.log(
                    {
                        "epoch": ep,
                        "train/loss": tr_l,
                        "train/acc": tr_a,
                        "val/loss": va_l,
                        "val/acc": va_a,
                        "time/train_sec": train_time,
                        "time/val_sec": val_time,
                        **extra_logs,
                    },
                    step=ep,
                )

            print(
                f"Ep{ep:3d} | "
                f"train_loss {tr_l:.3f}, acc {tr_a:.3f} ({train_time:.1f}s) | "
                f"val_loss {va_l:.3f}, acc {va_a:.3f} ({val_time:.1f}s)"
            )

        if USE_WANDB:
            wandb.finish()


if __name__ == "__main__":
    run_experiment(RUN_ATTENTION_TYPES, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, wd=DEFAULT_WD)
