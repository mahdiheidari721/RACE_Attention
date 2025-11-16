# ==================================================
# 0) Imports & Global Config
# ==================================================
import re, random, math, time, itertools, os
from collections import Counter
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

import re
import random
from collections import Counter, defaultdict
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ==================================================
# 0) Config
# ==================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "ccdv/arxiv-classification"
TEXT_FIELD   = "text"
LABEL_FIELD  = "label"

# ---- Target sequence length ----
SEQ_LEN_TARGET     = 32_000         # model context length (pad/truncate to this)
MIN_LEN_FOR_TRAIN  = 32_000         # require raw token length >= this

DESIRED_TRAIN_TOTAL = 7000          # desired; actual will be smaller
DESIRED_TEST_TOTAL  = 1000          # desired; actual will be smaller

# --------------------------------------------------
# High-level config
# --------------------------------------------------
TEXT_CONFIG = {
    # data / model
    "max_len": SEQ_LEN_TARGET,      # 32K context now
    "vocab_limit": 50_000,
    "embed_dim": 256,
    "num_heads": 4,
    "mlp_dim": 1024,
    "num_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,

    # RACE params
    "K": 2,
    "L": 2,
    "M": 1,

    # Performer params
    "m_features": 256,
    "favor_seed": None,

    # training
    "batch_size": 2,
    "epochs": 100,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_accum_steps": 16,
}

# ==================================================
# 1) Tokenizer (basic_english)
# ==================================================
_basic_english_re = re.compile(
    r"""([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])   # punctuation
     |(\d+[%]?)                                  # numbers (and percent)
     |([A-Za-z]+(?:'[A-Za-z]+)?)                 # words w/ optional apos
    """,
    re.VERBOSE,
)

def basic_english_tokenizer(text: str) -> List[str]:
    text = text.lower()
    tokens = []
    for punc, num, word in _basic_english_re.findall(text):
        if punc:   tokens.append(punc)
        elif num:  tokens.append(num)
        elif word: tokens.append(word)
    return tokens

tok = basic_english_tokenizer

# ==================================================
# 2) EDA augmenters (optional)
# ==================================================
def eda_random_deletion(tokens, p=0.05):
    if len(tokens) == 1:
        return tokens
    out = [t for t in tokens if random.random() > p]
    return out or [random.choice(tokens)]

def eda_random_swap(tokens, n_swaps=3):
    toks = tokens.copy()
    for _ in range(n_swaps):
        if len(toks) < 2:
            break
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    return toks

# ==================================================
# 3) Stats helpers
# ==================================================
def compute_raw_token_lengths(examples):
    """examples: list[(label, text)]"""
    lengths = []
    for lbl, txt in examples:
        toks = tok(str(txt))
        lengths.append(len(toks))
    return np.array(lengths, dtype=np.int32)

def print_length_stats(arr: np.ndarray, name: str, thresholds=()):
    print("--------------------------------------------------")
    print(f"Raw token length stats ({name})")
    print("--------------------------------------------------")
    print(f"Count:      {len(arr)}")
    print(f"Min:        {int(arr.min())}")
    print(f"Max:        {int(arr.max())}")
    print(f"Mean:       {float(arr.mean()):.1f}")
    print(f"Median:     {int(np.median(arr))}")
    print(f"90th pct:   {int(np.percentile(arr, 90))}")
    print(f"95th pct:   {int(np.percentile(arr, 95))}")
    print(f"99th pct:   {int(np.percentile(arr, 99))}")
    for thr in thresholds:
        frac = float((arr >= thr).mean())
        print(f"Frac >= {thr:6d}: {frac:.3f}")
    print()

def compute_effective_lengths_from_loader(dl, num_batches=100):
    """Use attention masks (non-PAD count) to get effective sequence length."""
    all_lengths = []
    for i, (tokens, masks, labels) in enumerate(dl):
        lens = masks.sum(dim=1).cpu().numpy()
        all_lengths.append(lens)
        if i + 1 >= num_batches:
            break
    if not all_lengths:
        return np.array([], dtype=np.int32)
    return np.concatenate(all_lengths).astype(np.int32)

def print_effective_length_stats(arr: np.ndarray, max_len: int, name: str, thresholds=()):
    print("--------------------------------------------------")
    print(f"Effective sequence length stats ({name})")
    print("--------------------------------------------------")
    print(f"Count (sampled): {len(arr)}")
    if len(arr) == 0:
        print("No data collected.")
        print()
        return
    print(f"Min:             {int(arr.min())}")
    print(f"Max:             {int(arr.max())}  (max_len = {max_len})")
    print(f"Mean:            {float(arr.mean()):.1f}")
    print(f"Median:          {int(np.median(arr))}")
    for thr in thresholds:
        frac = float((arr >= thr).mean())
        print(f"Frac >= {thr:6d}: {frac:.3f}")
    print()

# ==================================================
# 4) Length-filtered, class-balanced subset builder
# ==================================================
def make_balanced_long_examples(split, desired_total, min_len, seed=SEED, name="train"):
    """
    Create a class-balanced subset where each example has
    raw token length >= min_len.
    Returns:
      examples: list[(label, text)]
      num_classes: int
    """
    labels = list(split[LABEL_FIELD])
    texts  = list(split[TEXT_FIELD])

    print(f"\nBuilding balanced LONG-{name} subset with min_len = {min_len}...")
    print(f"Original {name} split size: {len(labels)}")

    # Precompute lengths
    lengths = []
    for txt in texts:
        toks = tok(str(txt))
        lengths.append(len(toks))
    lengths = np.array(lengths, dtype=np.int32)

    # Bucket long examples by class
    buckets = defaultdict(list)
    for idx, (y, L) in enumerate(zip(labels, lengths)):
        if L >= min_len:
            buckets[int(y)].append(idx)

    num_classes = len(buckets)
    if num_classes == 0:
        raise ValueError(f"No examples meet min_len={min_len} in {name} split!")

    print(f"Found {num_classes} classes with at least one example >= min_len.")
    for y in sorted(buckets.keys()):
        print(f"  Class {y}: {len(buckets[y])} examples >= {min_len}")

    # per-class limit
    max_possible_per_class = min(len(idxs) for idxs in buckets.values())
    desired_per_class      = desired_total // num_classes
    per_class              = min(max_possible_per_class, desired_per_class)
    actual_total           = per_class * num_classes

    print(f"\nDesired total {name} examples: {desired_total}")
    print(f"Max possible per class (given min_len): {max_possible_per_class}")
    print(f"Using per_class = {per_class}, so actual total = {actual_total}")

    rng = random.Random(seed)
    chosen_idx = []
    for y, idxs in buckets.items():
        rng.shuffle(idxs)
        chosen_idx.extend(idxs[:per_class])
    rng.shuffle(chosen_idx)

    examples = [(int(labels[i]), texts[i]) for i in chosen_idx]

    # Raw stats for this final subset
    chosen_lengths = lengths[chosen_idx]
    print_length_stats(
        chosen_lengths,
        name=f"{name} (balanced, length-filtered)",
        thresholds=(min_len, TEXT_CONFIG["max_len"]),
    )

    return examples, num_classes

# ==================================================
# 5) Load raw dataset and build long subsets
# ==================================================
raw = load_dataset(DATASET_NAME)

# pick train / test split
if "validation" in raw:
    train_split = raw["train"]
    test_split  = raw["validation"]
elif "test" in raw:
    train_split = raw["train"]
    test_split  = raw["test"]
else:
    tmp = raw["train"].train_test_split(test_size=0.2, seed=SEED)
    train_split, test_split = tmp["train"], tmp["test"]

train_examples, num_classes_train = make_balanced_long_examples(
    train_split,
    desired_total=DESIRED_TRAIN_TOTAL,
    min_len=MIN_LEN_FOR_TRAIN,
    seed=SEED,
    name="train",
)
test_examples,  num_classes_test  = make_balanced_long_examples(
    test_split,
    desired_total=DESIRED_TEST_TOTAL,
    min_len=MIN_LEN_FOR_TRAIN,
    seed=SEED,
    name="test",
)

assert num_classes_train == num_classes_test
num_classes = num_classes_train

# Extra raw-length stats directly from examples:
train_lengths = compute_raw_token_lengths(train_examples)
test_lengths  = compute_raw_token_lengths(test_examples)
print_length_stats(
    train_lengths,
    name="train_examples (recomputed)",
    thresholds=(MIN_LEN_FOR_TRAIN, TEXT_CONFIG["max_len"]),
)
print_length_stats(
    test_lengths,
    name="test_examples (recomputed)",
    thresholds=(MIN_LEN_FOR_TRAIN, TEXT_CONFIG["max_len"]),
)

# ==================================================
# 6) Build vocab from (balanced, long) train texts
# ==================================================
counter = Counter()
for lbl, txt in train_examples:
    counter.update(tok(str(txt)))

most_common = [w for w,_ in counter.most_common(TEXT_CONFIG["vocab_limit"])]
stoi = {w: i+2 for i, w in enumerate(most_common)}
stoi["<pad>"] = 0
stoi["<unk>"] = 1
PAD_IDX, UNK_IDX = 0, 1
VOCAB_SIZE = len(stoi)
TEXT_CONFIG["vocab_size"]   = VOCAB_SIZE
TEXT_CONFIG["num_classes"]  = num_classes

print(f"Vocab size: {VOCAB_SIZE}, num_classes: {num_classes}")
print(f"Balanced LONG train examples: {len(train_examples)}, test examples: {len(test_examples)}")

# ==================================================
# 7) Dataset for 32K sequences
# ==================================================
class ArxivDataset(Dataset):
    def __init__(self, examples, max_len, augment=False):
        self.examples = examples
        self.max_len  = max_len
        self.augment  = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        lbl, txt = self.examples[idx]
        toks = tok(str(txt))

        if self.augment:
            op = random.choice(["del", "swap", None])
            if op == "del":
                toks = eda_random_deletion(toks)
            elif op == "swap":
                toks = eda_random_swap(toks)

        toks = toks[: self.max_len]
        ids  = [stoi.get(t, UNK_IDX) for t in toks]
        if len(ids) < self.max_len:
            ids += [PAD_IDX] * (self.max_len - len(ids))
        return lbl, torch.tensor(ids, dtype=torch.long)

    def collate_fn(self, batch):
        labels, tokens = zip(*batch)
        tokens = torch.stack(tokens, dim=0)
        masks  = (tokens != PAD_IDX).long()
        return tokens, masks, torch.tensor(labels, dtype=torch.long)

max_len  = TEXT_CONFIG["max_len"]   # 32_000 now
batch_sz = TEXT_CONFIG["batch_size"]

train_ds = ArxivDataset(train_examples, max_len, augment=True)
test_ds  = ArxivDataset(test_examples,  max_len, augment=False)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_sz,
    shuffle=True,
    drop_last=True,
    pin_memory=(DEVICE == "cuda"),
    num_workers=4,
    collate_fn=train_ds.collate_fn,
)

test_dl = DataLoader(
    test_ds,
    batch_size=batch_sz,
    shuffle=False,
    pin_memory=(DEVICE == "cuda"),
    num_workers=2,
    collate_fn=test_ds.collate_fn,
)

print(f"Train batches: {len(train_dl)}, Test batches: {len(test_dl)}")

# ==================================================
# 8) Effective length stats from DataLoader
# ==================================================
train_eff_lengths = compute_effective_lengths_from_loader(
    train_dl,
    num_batches=100,   # sample first 100 batches
)
print_effective_length_stats(
    train_eff_lengths,
    max_len=max_len,
    name="train_dl (sampled)",
    thresholds=(int(0.8 * max_len), max_len),
)


# ==================================================
# 5) Attention modules (all baselines from vision)
#     – text version is pad-mask aware
# ==================================================

class MultiHeadAttention(nn.Module):
    """Standard softmax MH attention with pad mask."""
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
        h, dk = self.h, self.dk

        Q = self.q(x).view(B, T, h, dk).transpose(1, 2)
        K = self.k(x).view(B, T, h, dk).transpose(1, 2)
        V = self.v(x).view(B, T, h, dk).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            pad = mask[:, None, None, :]  # (B,1,1,T)
            scores = scores.masked_fill(pad == 0, float("-inf"))

        W = torch.softmax(scores, dim=-1)
        W = self.drop(W)
        out = (W @ V).transpose(1, 2).contiguous().view(B, T, h * dk)
        return self.o(out)

class AngularAttention(nn.Module):
    """Angular (cosine) attention."""
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
        h, dk = self.h, self.dk

        Q = self.q(x).view(B, T, h, dk).transpose(1, 2)
        K = self.k(x).view(B, T, h, dk).transpose(1, 2)
        V = self.v(x).view(B, T, h, dk).transpose(1, 2)

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        sim = (Q @ K.transpose(-2, -1)).clamp(-0.999, 0.999)
        scores = 1.0 - torch.acos(sim) / math.pi
        if mask is not None:
            pad = mask[:, None, None, :]
            scores = scores.masked_fill(pad == 0, 0.0)

        W = scores.clamp(min=1e-6).pow(8)
        W = W / (W.sum(-1, keepdim=True) + 1e-6)
        W = self.drop(W)

        out = (W @ V).transpose(1, 2).contiguous().view(B, T, h * dk)
        return self.o(out)

class BatchedACE(nn.Module):
    """Non-causal ACE used inside RACE, adapted from vision."""
    def __init__(self, d_k, K, L, M, device="cpu", share_planes=False):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer("planes_T", planes.view(L * K, d_k).T)
        else:
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)
            self.register_buffer("planes_T", planes)

        corners = torch.tensor(list(itertools.product([-1., +1.], repeat=K)), device=device)
        self.register_buffer("protos_T", corners.T)

        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, Khf, Vhf, Qhf, eps=1e-6):
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)

        if self.share_planes:
            N = M * B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            projK = Kh2 @ self.planes_T
            projQ = Qh2 @ self.planes_T
        else:
            BH = B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            projK = torch.einsum("mbtd,mds->mbts", Kh2, self.planes_T)
            projQ = torch.einsum("mbtd,mds->mbts", Qh2, self.planes_T)

            projK = projK.contiguous().view(M * BH, T, self.L * self.K)
            projQ = projQ.contiguous().view(M * BH, T, self.L * self.K)
            V2    = V2.view(M * BH, T, dk)
            N     = M * BH

        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)

        logitsK = (projK.tanh().div(scale) @ self.protos_T)   # [N,T,L,R]
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)
        probsK  = F.softmax(logitsK, dim=-1)
        probsQ  = F.softmax(logitsQ, dim=-1)

        probsK_S = probsK.contiguous().view(N, T, S)
        probsQ_S = probsQ.contiguous().view(N, T, S)

        b_sum = probsK_S.transpose(1, 2).bmm(V2)      # [N,S,dk]
        A     = probsK_S.sum(dim=1)                   # [N,S]
        E     = b_sum / (A.unsqueeze(-1) + eps)       # [N,S,dk]

        out2 = probsQ_S.bmm(E)                        # [N,T,dk]
        out  = out2.view(M, B, H, T, dk).permute(0, 1, 2, 3, 4)
        return out

class RACEAttention(nn.Module):
    def __init__(self, d, h, drop, K, L, M, qkv_bias=False, device="cpu"):
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
        B, T, d = x.shape
        h, dk, M = self.h, self.dk, self.M

        Q = self.q(x).view(B, T, h, dk)
        K = self.k(x).view(B, T, h, dk)
        V = self.v(x).view(B, T, h, dk)

        if mask is not None:
            m = mask.unsqueeze(-1).unsqueeze(-1).to(Q.dtype)
            Q, K, V = Q * m, K * m, V * m

        def pack(z):
            return z.unsqueeze(0).expand(M, -1, -1, -1, -1)

        out_m = self.ace(pack(K), pack(V), pack(Q))   # [M,B,H,T,dk]
        out   = out_m.mean(dim=0)                     # [B,H,T,dk]
        out   = out.permute(0, 2, 1, 3).contiguous().view(B, T, h * dk)
        return self.drop(self.o(out))

# ---- FAVOR+ (Performer) ----
def favorplus_features(x, proj, eps=1e-6):
    xw = torch.einsum("bhtd,hmd->bhtm", x, proj)
    xw = xw - xw.max(dim=-1, keepdim=True).values
    exp_part  = torch.exp(xw)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    base      = torch.exp(-0.5 * x_norm_sq)
    return exp_part * base + eps

class FavorPlusAttention(nn.Module):
    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
        super().__init__()
        assert d % h == 0
        self.h  = h
        self.dk = d // h
        self.m  = m_features

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        if seed is not None:
            torch.manual_seed(seed)
        proj = torch.nn.init.orthogonal_(torch.randn(h, m_features, self.dk))
        self.register_buffer("proj", proj)
        self.eps = 1e-6

    def forward(self, x, mask=None):
        B, T, d = x.shape
        h, dk, m = self.h, self.dk, self.m

        Q = self.q(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        Qs = Q / math.sqrt(dk)
        Ks = K / math.sqrt(dk)

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Ks   = Ks * keep
            V    = V  * keep

        phiQ = favorplus_features(Qs, self.proj, eps=self.eps) / math.sqrt(m)
        phiK = favorplus_features(Ks, self.proj, eps=self.eps) / math.sqrt(m)

        if mask is not None:
            keep_m = mask[:, None, :, None].to(phiK.dtype)
            phiK   = phiK * keep_m

        KV   = torch.einsum("bhtm,bhtd->bhmd", phiK, V)
        Ksum = phiK.sum(dim=2)

        num = torch.einsum("bhtm,bhmd->bhtd", phiQ, KV)
        den = torch.einsum("bhtm,bhm->bht",   phiQ, Ksum).unsqueeze(-1) + self.eps
        out_heads = num / den

        merged = out_heads.transpose(1, 2).contiguous().view(B, T, h * dk)
        merged = self.drop(merged)
        return self.o(merged)

# ---- Linear attention (ELU kernel) ----
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
        return F.elu(x) + 1

    def forward(self, x, mask=None):
        B, T, _ = x.size()
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2)
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2)

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            K = K * keep
            V = V * keep

        Q = self.kernel(Q)
        K = self.kernel(K)

        KV = torch.einsum("bhtd,bhte->bhde", K, V)  # [B,H,D,D]
        K_sum = K.sum(dim=2)                       # [B,H,D]

        Z = torch.einsum("bhtd,bhd->bht", Q, K_sum) + self.eps
        context = torch.einsum("bhtd,bhde->bhte", Q, KV)
        out = context / Z.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.dropout(out)
        return self.out_proj(out)

# ---- Linformer attention ----
class LinformerAttention(nn.Module):
    def __init__(self, d, dropout, num_heads, qkv_bias, k_proj_dim, max_seq_len):
        super().__init__()
        assert d % num_heads == 0
        self.h  = num_heads
        self.dk = d // num_heads
        self.k_proj_dim = k_proj_dim
        self.max_seq_len = max_seq_len

        self.W_query = nn.Linear(d, d, bias=qkv_bias)
        self.W_key   = nn.Linear(d, d, bias=qkv_bias)
        self.W_value = nn.Linear(d, d, bias=qkv_bias)

        self.E_k = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        self.E_v = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        nn.init.xavier_uniform_(self.E_k)
        nn.init.xavier_uniform_(self.E_v)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        assert T <= self.max_seq_len
        h, dk, k = self.h, self.dk, self.k_proj_dim

        Q = self.W_query(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.W_key(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            K = K * keep
            V = V * keep

        Ek = self.E_k[:T]  # (T,k)
        Ev = self.E_v[:T]

        K_proj = torch.einsum("bhtd,tk->bhkd", K, Ek)  # [B,h,k,dk]
        V_proj = torch.einsum("bhtd,tk->bhkd", V, Ev)

        scale = 1.0 / math.sqrt(dk)
        scores = torch.einsum("bhtd,bhkd->bhtk", Q, K_proj) * scale
        attn = F.softmax(scores, dim=-1)

        ctx = torch.einsum("bhtk,bhkd->bhtd", attn, V_proj)
        out = ctx.transpose(1, 2).contiguous().view(B, T, h * dk)
        out = self.dropout(out)
        return self.out_proj(out)

# ==================================================
# 6) Transformer blocks (one per baseline)
# ==================================================
class SoftmaxBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.att  = MultiHeadAttention(d, h, drop, qkv_bias)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.att  = AngularAttention(d, h, drop, qkv_bias)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class RACEBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.att = RACEAttention(
            d=d, h=h, drop=drop,
            K=cfg["K"], L=cfg["L"], M=cfg["M"],
            qkv_bias=qkv_bias, device=device,
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class LinearBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        h = cfg["num_heads"]
        self.att  = LinearAttention(
            d_in=d, d_out=d, dropout=drop, num_heads=h, qkv_bias=qkv_bias
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class LinformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        h = cfg["num_heads"]
        k_proj_dim = 128
        self.att  = LinformerAttention(
            d=d,
            dropout=drop,
            num_heads=h,
            qkv_bias=qkv_bias,
            k_proj_dim=k_proj_dim,
            max_seq_len=cfg["max_len"],
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class PerformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        self.att = FavorPlusAttention(
            d=d,
            h=h,
            m_features=cfg["m_features"],
            drop=drop,
            qkv_bias=cfg["qkv_bias"],
            seed=cfg["favor_seed"],
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

# ==================================================
# 7) Text Transformer classifier (ViT-style structure)
# ==================================================
class TextTransformerClassifier(nn.Module):
    def __init__(self, cfg, attn_type: str):
        super().__init__()
        self.cfg = cfg
        vocab_size = cfg["vocab_size"]
        max_len    = cfg["max_len"]
        d          = cfg["embed_dim"]

        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.drop    = nn.Dropout(cfg["drop_rate"])

        if attn_type == "softmax":
            Block = SoftmaxBlock
        elif attn_type == "race":
            Block = lambda c: RACEBlock(c, device=DEVICE)
        elif attn_type == "angular":
            Block = AngularBlock
        elif attn_type == "linear":
            Block = LinearBlock
        elif attn_type == "linformer":
            Block = LinformerBlock
        elif attn_type == "performer":
            Block = PerformerBlock
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

        self.layers = nn.ModuleList(
            [Block(cfg) for _ in range(cfg["num_layers"])]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["num_classes"])

    def forward(self, x, mask):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.layers:
            h = blk(h, mask)
        h = self.norm(h)
        # CLS-style: use position 0
        logits = self.head(h[:, 0])
        return logits

# ==================================================
# 8) Scheduler & training loop (like vision file)
# ==================================================
class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(self.warmup_steps + 1, int(total_steps))
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

def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    cfg,
    attn_type,
    grad_accum_steps: int = 1,
):
    steps_per_epoch   = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
    total_updates     = num_epochs * updates_per_epoch
    warmup_updates    = max(1, int(0.1 * total_updates))

    scheduler = LinearWarmupLR(
        optimizer,
        warmup_steps=warmup_updates,
        total_steps=total_updates,
    )

    out_path = f"arxiv_{attn_type}_32K.txt"

    def _log(fp, msg):
        print(msg)
        fp.write(msg + "\n")
        fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Attn: {attn_type}, Epochs: {num_epochs}")
        _log(f, "-" * 80)
        global_update = 0

        for epoch in range(1, num_epochs + 1):
            # ---- TRAIN ----
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t0 = time.time()

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = 0.0
            running_correct = 0
            running_total = 0
            accum_count = 0

            train_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch} [train]",
                leave=False,
            )

            for tokens, masks, labels in train_iter:
                tokens  = tokens.to(device)
                masks   = masks.to(device)
                labels  = labels.to(device)

                logits = model(tokens, masks)
                loss   = F.cross_entropy(logits, labels)

                (loss / grad_accum_steps).backward()
                accum_count += 1

                preds = logits.argmax(dim=-1)
                running_correct += (preds == labels).sum().item()
                running_total   += labels.size(0)
                running_loss    += loss.item()

                # Update tqdm with running stats
                train_iter.set_postfix({
                    "loss": running_loss / max(1, len(train_iter)),
                    "acc":  running_correct / max(1, running_total),
                })

                if accum_count == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_update += 1

            if accum_count > 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            if "cuda" in str(device):
                torch.cuda.synchronize()
            train_time = time.time() - t0

            tr_l = running_loss / len(train_loader)
            tr_a = running_correct / max(1, running_total)

            # ---- VAL ----
            if "cuda" in str(device):
                torch.cuda.synchronize()
            t1 = time.time()

            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total   = 0

            val_iter = tqdm(
                val_loader,
                desc=f"Epoch {epoch} [val]",
                leave=False,
            )

            with torch.no_grad():
                for tokens, masks, labels in val_iter:
                    tokens = tokens.to(device)
                    masks  = masks.to(device)
                    labels = labels.to(device)

                    logits = model(tokens, masks)
                    loss   = F.cross_entropy(logits, labels)
                    val_loss_total += loss.item()

                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)

                    val_iter.set_postfix({
                        "loss": val_loss_total / max(1, len(val_iter)),
                        "acc":  val_correct / max(1, val_total),
                    })

            if "cuda" in str(device):
                torch.cuda.synchronize()
            val_time = time.time() - t1

            va_l = val_loss_total / len(val_loader)
            va_a = val_correct / max(1, val_total)
            curr_lr = scheduler.get_last_lr()[0]

            _log(
                f,
                (f"Ep{epoch:3d} | "
                 f"train_loss {tr_l:.4f}, acc {tr_a:.4f} ({train_time:.1f}s) | "
                 f"val_loss {va_l:.4f}, acc {va_a:.4f} ({val_time:.1f}s) | "
                 f"lr {curr_lr:.3e} | updates {global_update}/{total_updates}")
            )

        _log(f, "-" * 80)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")


# ==================================================
# 9) Run all baselines (like vision)
# ==================================================
def run_experiment(attn_types, cfg):
    for attn_type in attn_types:
        print(f"\n=== Training {attn_type.upper()} on Arxiv 32K ===")
        model = TextTransformerClassifier(cfg, attn_type).to(DEVICE)
        opt   = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        train_model_simple(
            model=model,
            train_loader=train_dl,
            val_loader=test_dl,
            optimizer=opt,
            device=DEVICE,
            num_epochs=cfg["epochs"],
            cfg=cfg,
            attn_type=attn_type,
            grad_accum_steps=cfg["grad_accum_steps"],
        )

if __name__ == "__main__":
    # same baseline set as Vision file
    run_experiment(
        ["race"],
        TEXT_CONFIG,
    )
