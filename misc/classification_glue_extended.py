# ==================================================
# Text classification with IMDB / SST-2 / QNLI
# Supports: softmax, exact_flash, race, hyper_lsh,
#           hyper_race, hyper_plus_race, fixed_hyper_plus_race, angular
# ==================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re, random, math, time, itertools
from math import ceil
from collections import Counter

import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.attention import sdpa_kernel, SDPBackend
from datasets import load_dataset
from tqdm import tqdm

# ==================================================
# 0) Global setup
# ==================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_IDX, UNK_IDX, CLS_IDX = 0, 1, 2

if DEVICE == "cuda":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cudnn.benchmark = True

def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ==================================================
# 1) Task configs
# ==================================================
TASK_CONFIGS = {
    "imdb": {
        "dataset_display_name": "IMDB-long",
        "max_len": 512,
        "batch_size": 32,
        "epochs": 150,
        "lr": 1e-5,
        "weight_decay": 5e-5,
        "emb_dim": 128,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.1,
        "num_classes": 2,
        "qkv_bias": False,
        "M": 2,
        "K": 3,
        "L": 2,
        "hyper_num_bits": 5,
        "hyper_block_size": 32,
        "hyper_min_seq_len": 256,
        "hyper_neighbor_blocks": 0,
        "gate_hidden_dim": 64,
        "gate_normalize": False,
        "hyper_plus_race_q_chunk_size": 256,
        "vocab_limit": 50_000,
        "use_long_imdb": True,
        "train_augment": True,
    },
    "sst2": {
        "dataset_display_name": "GLUE-SST2",
        "max_len": 1024,
        "batch_size": 32,
        "epochs": 100,
        "lr": 1e-5,
        "weight_decay": 5e-5,
        "emb_dim": 384,
        "n_heads": 8,
        "n_layers": 4,
        "drop_rate": 0.1,
        "num_classes": 2,
        "qkv_bias": False,
        "M": 2,
        "K": 3,
        "L": 2,
        "hyper_num_bits": 5,
        "hyper_block_size": 32,
        "hyper_min_seq_len": 256,
        "hyper_neighbor_blocks": 0,
        "gate_hidden_dim": 64,
        "gate_normalize": False,
        "hyper_plus_race_q_chunk_size": 256,
        "vocab_limit": 50_000,
        "use_long_imdb": False,
        "train_augment": False,
    },
    "qnli": {
        "dataset_display_name": "GLUE-QNLI",
        "max_len": 2048,
        "batch_size": 32,
        "epochs": 100,
        "lr": 1e-5,
        "weight_decay": 5e-5,
        "emb_dim": 384,
        "n_heads": 8,
        "n_layers": 4,
        "drop_rate": 0.1,
        "num_classes": 2,
        "qkv_bias": False,
        "M": 2,
        "K": 3,
        "L": 2,
        "hyper_num_bits": 5,
        "hyper_block_size": 32,
        "hyper_min_seq_len": 256,
        "hyper_neighbor_blocks": 0,
        "gate_hidden_dim": 64,
        "gate_normalize": False,
        "hyper_plus_race_q_chunk_size": 256,
        "vocab_limit": 50_000,
        "use_long_imdb": False,
        "train_augment": False,
    },
}

# Change this list if you only want a subset.
DATASETS_TO_RUN = [ "sst2", "qnli"]   # "imdb"

# ==================================================
# 2) “basic_english” tokenizer drop‑in
# ==================================================
_basic_english_re = re.compile(
    r"""([!"#$%&'()*+,\-./:;<=>?@[\\]^_`{|}~])   # any punctuation
     |(\d+[%]?)                                    # numbers (and percent)
     |([A-Za-z]+(?:'[A-Za-z]+)?)                   # words w/ optional apos
    """,
    re.VERBOSE,
)

def basic_english_tokenizer(text: str) -> list[str]:
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
# 3) EDA augmenters
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
# 4) Dataset / vocab utils
# ==================================================
def _join_qnli(question: str, sentence: str) -> str:
    # single-token separator under basic_english
    return f"{question} septoken {sentence}"


def load_task_examples(task_name: str):
    task_name = task_name.lower()

    if task_name == "imdb":
        raw = load_dataset("imdb")
        train_examples = list(zip(raw["train"]["label"], raw["train"]["text"]))
        test_examples  = list(zip(raw["test"]["label"],  raw["test"]["text"]))
        return train_examples, test_examples

    if task_name == "sst2":
        raw = load_dataset("glue", "sst2")
        train_examples = list(zip(raw["train"]["label"], raw["train"]["sentence"]))
        test_examples  = list(zip(raw["validation"]["label"], raw["validation"]["sentence"]))
        return train_examples, test_examples

    if task_name == "qnli":
        raw = load_dataset("glue", "qnli")
        train_examples = [
            (int(lbl), _join_qnli(q, s))
            for lbl, q, s in zip(raw["train"]["label"], raw["train"]["question"], raw["train"]["sentence"])
        ]
        test_examples = [
            (int(lbl), _join_qnli(q, s))
            for lbl, q, s in zip(raw["validation"]["label"], raw["validation"]["question"], raw["validation"]["sentence"])
        ]
        return train_examples, test_examples

    raise ValueError(f"Unsupported task_name: {task_name}")


def build_vocab(train_examples, vocab_limit: int):
    counter = Counter()
    for _, txt in train_examples:
        counter.update(tok(txt))

    stoi = {"<pad>": PAD_IDX, "<unk>": UNK_IDX, "<cls>": CLS_IDX}
    for w, _ in counter.most_common(vocab_limit):
        if w not in stoi:
            stoi[w] = len(stoi)
    return stoi


class TextClassificationDataset(Dataset):
    def __init__(self, examples, max_len, stoi, augment=False):
        self.examples = examples
        self.max_len = max_len
        self.stoi = stoi
        self.augment = augment

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

        toks = toks[: self.max_len - 1]
        ids = [CLS_IDX] + [self.stoi.get(t, UNK_IDX) for t in toks]
        if len(ids) < self.max_len:
            ids += [PAD_IDX] * (self.max_len - len(ids))

        return int(lbl), torch.tensor(ids, dtype=torch.long)

    def collate_fn(self, batch):
        labels, texts = zip(*batch)
        texts = torch.stack(texts, dim=0)
        masks = (texts != PAD_IDX).long()
        return texts, masks, torch.tensor(labels, dtype=torch.long)


def make_long_subsets(examples, max_len: int, seed: int = SEED):
    random.seed(seed)
    real_long, short_pos, short_neg, super_long = [], [], [], []
    for lbl, txt in examples:
        L = len(tok(txt))
        if max_len <= L < 2 * max_len:
            real_long.append((lbl, txt))
        elif L >= 2 * max_len:
            super_long.append((lbl, txt))
        else:
            (short_pos if lbl == 1 else short_neg).append(txt)

    split_long = []
    for lbl, txt in super_long:
        toks = tok(txt)
        step = max_len // 2
        num_windows = ceil((len(toks) - max_len) / step) + 1
        for w in range(num_windows):
            start = min(w * step, len(toks) - max_len)
            window = toks[start : start + max_len]
            split_long.append((lbl, " ".join(window)))
            if start + max_len >= len(toks):
                break

    random.seed(seed)
    random.shuffle(short_pos)
    random.seed(seed)
    random.shuffle(short_neg)

    new_long = []
    for pool, label in [(short_pos, 1), (short_neg, 0)]:
        i = 0
        while i + 1 < len(pool):
            combo = tok(pool[i]) + tok(pool[i + 1])
            if len(combo) >= max_len:
                new_long.append((label, " ".join(combo)))
                i += 2
            else:
                if i + 2 < len(pool):
                    combo3 = combo + tok(pool[i + 2])
                    if len(combo3) >= max_len:
                        new_long.append((label, " ".join(combo3)))
                        i += 3
                        continue
                i += 1

    long_train = real_long + split_long + new_long
    random.seed(seed)
    random.shuffle(long_train)

    long_test = [(lbl, txt) for lbl, txt in examples if len(tok(txt)) >= max_len]
    random.seed(seed)
    random.shuffle(long_test)

    return long_train, long_test


def get_data(task_name: str):
    task_name = task_name.lower()
    task_cfg = TASK_CONFIGS[task_name]

    train_examples, test_examples = load_task_examples(task_name)
    stoi = build_vocab(train_examples, task_cfg["vocab_limit"])
    vocab_size = len(stoi)

    max_len = task_cfg["max_len"]
    batch_size = task_cfg["batch_size"]

    if task_name == "imdb" and task_cfg["use_long_imdb"]:
        long_train, _ = make_long_subsets(train_examples, max_len=max_len, seed=SEED)
        _, long_test = make_long_subsets(test_examples, max_len=max_len, seed=SEED)

        print(f"--> FINAL long_train: {len(long_train)}")
        print(f"--> FINAL long_test:  {len(long_test)}")

        train_ds = TextClassificationDataset(long_train, max_len, stoi, augment=task_cfg["train_augment"])
        test_ds  = TextClassificationDataset(long_test,  max_len, stoi, augment=False)
    else:
        train_local = list(train_examples)
        test_local  = list(test_examples)
        random.seed(SEED); random.shuffle(train_local)
        random.seed(SEED); random.shuffle(test_local)

        print(f"--> FINAL train: {len(train_local)}")
        print(f"--> FINAL test:  {len(test_local)}")

        train_ds = TextClassificationDataset(train_local, max_len, stoi, augment=task_cfg["train_augment"])
        test_ds  = TextClassificationDataset(test_local,  max_len, stoi, augment=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(DEVICE == "cuda"),
        num_workers=4,
        collate_fn=train_ds.collate_fn,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(DEVICE == "cuda"),
        num_workers=2,
        collate_fn=test_ds.collate_fn,
    )

    data_info = {
        "task_cfg": task_cfg,
        "vocab_size": vocab_size,
        "context_length": max_len,
        "num_classes": task_cfg["num_classes"],
        "dataset_display_name": task_cfg["dataset_display_name"],
    }
    return train_dl, test_dl, data_info

# ==================================================
# 7) Attention blocks (pad‑mask aware)
# ==================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
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
            pad = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad == 0, float("-inf"))
        W = torch.softmax(scores, -1)
        W = self.drop(W)
        out = (W @ V).transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

class AngularAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
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
        scores = 1 - torch.acos(sim) / math.pi
        if mask is not None:
            pad = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad == 0, 0.0)
        W = scores.clamp(min=1e-6).pow(18)
        W = W / (W.sum(-1, keepdim=True) + 1e-6)
        W = self.drop(W)
        out = (W @ V).transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

# ==================================================
# Hyper-LSH exact sparse attention helpers
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
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))

def _run_exact_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
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

def _run_exact_sdpa_masked(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None):
    attn_mask = None
    if mask is not None:
        attn_mask = mask[:, None, None, :].bool()  # True = keep key

    if Q.device.type == "cuda":
        Q16, K16, V16 = [t.to(dtype=torch.float16) for t in (Q, K, V)]
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    Q16, K16, V16,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
            return out.to(Q.dtype)
        except RuntimeError:
            pass

    return F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )

class AngularLSHGray(nn.Module):
    def __init__(self, num_bits: int, dim: int, device="cpu"):
        super().__init__()
        self.num_bits = num_bits
        self.R = 1 << num_bits

        proj_dir = torch.randn(dim, num_bits, device=device)
        perm = _gray_code_order(num_bits, device=device)

        self.register_buffer("proj_dir", proj_dir, persistent=False)
        self.register_buffer("perm", perm, persistent=False)

    def hash(self, mat: torch.Tensor):
        proj = torch.einsum("htd,dr->htr", mat, self.proj_dir)
        bits = (proj > 0).to(torch.long)

        enc = (2 ** torch.arange(self.num_bits, device=mat.device, dtype=torch.long)).view(
            1, 1, self.num_bits
        )
        bin_ids = (bits * enc).sum(dim=-1)
        return self.perm[bin_ids]

class ExactFlashAttentionText(nn.Module):
    """Exact attention via SDPA; uses Flash backend when available."""
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
        Q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2).contiguous()

        out = _run_exact_sdpa_masked(Q, K, V, mask)
        if mask is not None:
            out = out * mask[:, None, :, None].to(out.dtype)

        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.dk)
        out = self.drop(out)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

class ExactFlashBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = ExactFlashAttentionText(
            d=cfg["emb_dim"],
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
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

        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x

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
        return _run_exact_sdpa(
            Qh.unsqueeze(0),
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

    def forward(self, x, mask):
        B, T, _ = x.shape
        H, D = self.h, self.dk

        Q = self.q(x).view(B, T, H, D).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, H, D).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, H, D).transpose(1, 2).contiguous()

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

            O_unsorted = O_sorted.gather(
                1, q_sort_inv.unsqueeze(-1).expand(-1, -1, D)
            )
            out[b, :, :valid_T, :] = O_unsorted

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.drop(out)
        out = self.o(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

class HyperLSHExactBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        self.att = HyperLSHExactAttentionText(
            d=cfg["emb_dim"],
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            num_bits=cfg.get("hyper_num_bits", 5),
            block_size=cfg.get("hyper_block_size", 32),
            min_seq_len=cfg.get("hyper_min_seq_len", 256),
            neighbor_blocks=cfg.get("hyper_neighbor_blocks", 0),
            qkv_bias=cfg["qkv_bias"],
            device=device,
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
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
        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x

class BatchedACE(nn.Module):
    def __init__(self, d_k, K, L, M):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        planes = torch.randn(L, K, d_k)
        self.register_buffer("planes_T", planes.view(L * K, d_k).T)
        corners = torch.tensor(list(itertools.product([-1., +1.], repeat=K)))
        self.register_buffer("protos_T", corners.T)

    def forward(self, Kh, Vh, Qh):
        M, B, T, H, dk = Kh.shape
        assert M == self.M and dk == self.d_k

        flat_K = Kh.contiguous().view(-1, dk)
        projK_flat = flat_K @ self.planes_T
        projK = projK_flat.view(-1, self.L, self.K)

        logitsK = (projK.tanh().div(dk ** 0.5).view(-1, self.K) @ self.protos_T).view(
            M, B, T, H, self.L, self.R
        )
        probsK = logitsK.softmax(dim=-1)

        MBH = M * B * H
        probs_flat = probsK.permute(0, 1, 3, 2, 4, 5).contiguous().view(MBH, T, self.L * self.R)
        V_flat = Vh.permute(0, 1, 3, 2, 4).contiguous().view(MBH, T, dk)
        b_sum = probs_flat.transpose(1, 2).bmm(V_flat)
        A = probs_flat.sum(dim=1, keepdim=True)
        E_flat = b_sum / (A.transpose(1, 2) + 1e-6)

        flat_Q = Qh.contiguous().view(-1, dk)
        projQ = (flat_Q @ self.planes_T).view(-1, self.L, self.K)
        logitsQ = ((projQ.tanh().div(dk ** 0.5).view(-1, self.K) @ self.protos_T).view(
            M, B, T, H, self.L, self.R
        ))
        probsQ = logitsQ.softmax(dim=-1)
        probsQ_flat = probsQ.permute(0, 1, 3, 2, 4, 5).contiguous().view(MBH, T, self.L * self.R)

        out_flat = probsQ_flat.bmm(E_flat)
        return out_flat.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4)

class RACEAttention(nn.Module):
    """Pad-aware RACEAttention. This matters for SST-2 / QNLI."""
    def __init__(self, d, h, drop, M=2, K=3, L=2, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.M = h, d // h, M
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(self.dk, K, L, M)

    def forward(self, x, mask):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk)
        K = self.k(x).view(B, T, self.h, self.dk)
        V = self.v(x).view(B, T, self.h, self.dk)

        out = torch.zeros(B, T, self.h, self.dk, device=x.device, dtype=x.dtype)

        def pack(z):
            return z.unsqueeze(0).expand(self.M, -1, -1, -1, -1)

        for b in range(B):
            valid_T = int(mask[b].sum().item()) if mask is not None else T
            if valid_T == 0:
                continue

            Qb = Q[b:b+1, :valid_T]     # [1,Tv,H,dk]
            Kb = K[b:b+1, :valid_T]
            Vb = V[b:b+1, :valid_T]

            out_m = self.ace(pack(Kb), pack(Vb), pack(Qb))   # [M,1,Tv,H,dk]
            out[b, :valid_T] = out_m.mean(dim=0)[0]

        out = out.contiguous().view(B, T, self.h * self.dk)
        out = self.drop(self.o(out))
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

def race_bucket_probs_from_qk(attn: RACEAttention, Q: torch.Tensor, K: torch.Tensor):
    """
    Reproduce the bucket probability part of the current RACEAttention.
    Inputs:
        Q, K: [B, T, H, dk]
    Returns:
        probsQ, probsK: [M, B, T, H, L, R]
    """
    ace = attn.ace
    M = attn.M
    B, T, H, dk = Q.shape

    def pack(z):
        return z.unsqueeze(0).expand(M, -1, -1, -1, -1)

    Qhf = pack(Q)
    Khf = pack(K)

    flat_Q = Qhf.contiguous().view(-1, dk)
    flat_K = Khf.contiguous().view(-1, dk)

    projQ = (flat_Q @ ace.planes_T).view(-1, ace.L, ace.K)
    projK = (flat_K @ ace.planes_T).view(-1, ace.L, ace.K)

    logitsQ = (projQ.tanh().div(dk ** 0.5).view(-1, ace.K) @ ace.protos_T).view(
        M, B, T, H, ace.L, ace.R
    )
    logitsK = (projK.tanh().div(dk ** 0.5).view(-1, ace.K) @ ace.protos_T).view(
        M, B, T, H, ace.L, ace.R
    )

    probsQ = logitsQ.softmax(dim=-1)
    probsK = logitsK.softmax(dim=-1)
    return probsQ, probsK

class BucketExcludedRACEAttentionText(RACEAttention):
    """
    Basic hyper_plus_race RACE branch for text:
    same hard Hyper bucket keys are excluded from the RACE numerator,
    but kept in the denominator.
    """
    def __init__(
        self,
        d,
        h,
        drop,
        M=2,
        K=3,
        L=2,
        hard_num_bits=5,
        q_chunk_size=256,
        qkv_bias=False,
    ):
        super().__init__(d=d, h=h, drop=drop, M=M, K=K, L=L, qkv_bias=qkv_bias)
        self.hard_num_bits = hard_num_bits
        self.hard_R = 1 << hard_num_bits
        self.q_chunk_size = q_chunk_size

    def _single_valid_forward(self, Qv, Kv, Vv, q_hard_ids_valid, k_hard_ids_valid):
        """
        Qv, Kv, Vv: [Tv, H, dk]
        q_hard_ids_valid, k_hard_ids_valid: [H, Tv]
        """
        Tv, H, dk = Qv.shape
        B = 1
        S = self.ace.L * self.ace.R
        N = self.M * B * H

        Q = Qv.unsqueeze(0)   # [1,Tv,H,dk]
        K = Kv.unsqueeze(0)
        V = Vv.unsqueeze(0)

        probsQ, probsK = race_bucket_probs_from_qk(self, Q, K)   # [M,1,T,H,L,R], [M,1,T,H,L,R]
        probsQ_S = probsQ.permute(0, 1, 3, 2, 4, 5).contiguous().view(N, Tv, S)
        probsK_S = probsK.permute(0, 1, 3, 2, 4, 5).contiguous().view(N, Tv, S)

        Vhf = V.unsqueeze(0).expand(self.M, -1, -1, -1, -1)  # [M,1,Tv,H,dk]
        V2 = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, Tv, dk)

        total_num = probsK_S.transpose(1, 2).bmm(V2)         # [N,S,dk]
        total_den = probsK_S.sum(dim=1)                      # [N,S]
        denom = total_den.unsqueeze(-1) + 1e-6

        E_all = total_num / denom
        out2 = probsQ_S.bmm(E_all)                           # [N,Tv,dk]

        q_ids = q_hard_ids_valid.unsqueeze(0).expand(self.M, -1, -1).contiguous().view(N, Tv)
        k_ids = k_hard_ids_valid.unsqueeze(0).expand(self.M, -1, -1).contiguous().view(N, Tv)

        for b_id in range(self.hard_R):
            qmask_b = (q_ids == b_id)
            if not bool(qmask_b.any().item()):
                continue

            kmask_b = (k_ids == b_id).to(probsK_S.dtype)

            same_num_b = torch.einsum(
                "nts,nt,ntd->nsd",
                probsK_S,
                kmask_b,
                V2,
            )   # [N,S,dk]

            E_same_b = same_num_b / denom

            for qs in range(0, Tv, self.q_chunk_size):
                qe = min(qs + self.q_chunk_size, Tv)
                qmask_chunk = qmask_b[:, qs:qe]
                if not bool(qmask_chunk.any().item()):
                    continue

                remove_chunk = probsQ_S[:, qs:qe, :].bmm(E_same_b)
                out2[:, qs:qe, :] = out2[:, qs:qe, :] - (
                    qmask_chunk.unsqueeze(-1).to(out2.dtype) * remove_chunk
                )

        out = out2.view(self.M, H, Tv, dk).mean(dim=0)   # [H,Tv,dk]
        out = out.permute(1, 0, 2).contiguous()          # [Tv,H,dk]
        return out

    def forward(self, x, mask, q_hard_bucket_ids, k_hard_bucket_ids):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk)
        K = self.k(x).view(B, T, self.h, self.dk)
        V = self.v(x).view(B, T, self.h, self.dk)

        out = torch.zeros(B, T, self.h, self.dk, device=x.device, dtype=x.dtype)

        for b in range(B):
            valid_T = int(mask[b].sum().item()) if mask is not None else T
            if valid_T == 0:
                continue

            out[b, :valid_T] = self._single_valid_forward(
                Q[b, :valid_T],
                K[b, :valid_T],
                V[b, :valid_T],
                q_hard_bucket_ids[b, :, :valid_T],
                k_hard_bucket_ids[b, :, :valid_T],
            )

        out = out.contiguous().view(B, T, self.h * self.dk)
        out = self.drop(self.o(out))
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

class RACEBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        self.att = RACEAttention(
            d=cfg["emb_dim"],
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            M=cfg.get("M", 2),
            K=cfg.get("K", 3),
            L=cfg.get("L", 2),
            qkv_bias=cfg.get("qkv_bias", False),
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
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

        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x

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
        out_race  = self.race(x, mask)

        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)

        if self.normalize_gates:
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)

        self.last_gates = gates.detach()

        g_hyper = gates[..., 0:1]
        g_race  = gates[..., 1:2]

        out = g_hyper * out_hyper + g_race * out_race
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

class HyperRaceGatedBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        self.att = HyperRaceGatedAttentionText(cfg, device=device)

        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
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

        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x

class FixedHyperPlusRaceAttentionText(nn.Module):
    """
    Fixed hyper_plus_race (global scalars a,b).
    """
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["emb_dim"]

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
        self.shared_lsh = self.hyper.lsh

        self.race = BucketExcludedRACEAttentionText(
            d=d,
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            M=cfg.get("M", 2),
            K=cfg.get("K", 3),
            L=cfg.get("L", 2),
            hard_num_bits=cfg.get("hyper_num_bits", 5),
            q_chunk_size=cfg.get("hyper_plus_race_q_chunk_size", 256),
            qkv_bias=cfg["qkv_bias"],
        )

        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def _shared_hard_bucket_ids(self, x, mask):
        B, T, _ = x.shape
        H, D = self.hyper.h, self.hyper.dk

        Qh = self.hyper.q(x).view(B, T, H, D).transpose(1, 2).contiguous()
        Kh = self.hyper.k(x).view(B, T, H, D).transpose(1, 2).contiguous()

        q_ids = torch.zeros(B, H, T, device=x.device, dtype=torch.long)
        k_ids = torch.zeros(B, H, T, device=x.device, dtype=torch.long)

        for b in range(B):
            valid_T = int(mask[b].sum().item()) if mask is not None else T
            if valid_T == 0:
                continue
            q_ids[b, :, :valid_T] = self.shared_lsh.hash(Qh[b, :, :valid_T, :])
            k_ids[b, :, :valid_T] = self.shared_lsh.hash(Kh[b, :, :valid_T, :])

        return q_ids, k_ids

    def forward(self, x, mask):
        q_hard_bucket_ids, k_hard_bucket_ids = self._shared_hard_bucket_ids(x, mask)

        out_hyper = self.hyper(x, mask)
        out_race  = self.race(x, mask, q_hard_bucket_ids, k_hard_bucket_ids)

        out = self.a * out_race + self.b * out_hyper
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

class FixedHyperPlusRaceBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        self.att = FixedHyperPlusRaceAttentionText(cfg, device=device)

        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
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

        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x

class HyperPlusRaceAttentionText(nn.Module):
    """
    Basic hyper_plus_race with input-dependent sigmoid gates.
    """
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
        self.shared_lsh = self.hyper.lsh

        self.race = BucketExcludedRACEAttentionText(
            d=d,
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            M=cfg.get("M", 2),
            K=cfg.get("K", 3),
            L=cfg.get("L", 2),
            hard_num_bits=cfg.get("hyper_num_bits", 5),
            q_chunk_size=cfg.get("hyper_plus_race_q_chunk_size", 256),
            qkv_bias=cfg["qkv_bias"],
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )
        self.last_gates = None

    @torch.no_grad()
    def _shared_hard_bucket_ids(self, x, mask):
        B, T, _ = x.shape
        H, D = self.hyper.h, self.hyper.dk

        Qh = self.hyper.q(x).view(B, T, H, D).transpose(1, 2).contiguous()
        Kh = self.hyper.k(x).view(B, T, H, D).transpose(1, 2).contiguous()

        q_ids = torch.zeros(B, H, T, device=x.device, dtype=torch.long)
        k_ids = torch.zeros(B, H, T, device=x.device, dtype=torch.long)

        for b in range(B):
            valid_T = int(mask[b].sum().item()) if mask is not None else T
            if valid_T == 0:
                continue
            q_ids[b, :, :valid_T] = self.shared_lsh.hash(Qh[b, :, :valid_T, :])
            k_ids[b, :, :valid_T] = self.shared_lsh.hash(Kh[b, :, :valid_T, :])

        return q_ids, k_ids

    def forward(self, x, mask):
        q_hard_bucket_ids, k_hard_bucket_ids = self._shared_hard_bucket_ids(x, mask)

        out_hyper = self.hyper(x, mask)
        out_race  = self.race(x, mask, q_hard_bucket_ids, k_hard_bucket_ids)

        gate_logits = self.gate_mlp(x)
        gates = torch.sigmoid(gate_logits)
        self.last_gates = gates.detach()

        g_hyper = gates[..., 0:1]
        g_race  = gates[..., 1:2]

        out = g_hyper * out_hyper + g_race * out_race
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

class HyperPlusRaceBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        self.att = HyperPlusRaceAttentionText(cfg, device=device)

        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
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

        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d=cfg["emb_dim"],
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
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

        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x

class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = AngularAttention(
            d=cfg["emb_dim"],
            h=cfg["n_heads"],
            drop=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
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

        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        return x

# ==================================================
# 8) Model & run_experiment
# ==================================================
class Classifier(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop    = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.ModuleList()
        for _ in range(cfg["n_layers"]):
            if kind == "softmax":
                self.blocks.append(TransformerBlock(cfg))
            elif kind == "exact_flash":
                self.blocks.append(ExactFlashBlock(cfg))
            elif kind == "angular":
                self.blocks.append(AngularBlock(cfg))
            elif kind == "race":
                self.blocks.append(RACEBlock(cfg, device=DEVICE))
            elif kind == "hyper_lsh":
                self.blocks.append(HyperLSHExactBlock(cfg, device=DEVICE))
            elif kind == "hyper_race":
                self.blocks.append(HyperRaceGatedBlock(cfg, device=DEVICE))
            elif kind == "hyper_plus_race":
                self.blocks.append(HyperPlusRaceBlock(cfg, device=DEVICE))
            elif kind == "fixed_hyper_plus_race":
                self.blocks.append(FixedHyperPlusRaceBlock(cfg, device=DEVICE))
            else:
                raise ValueError(kind)

        self.norm = nn.LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["num_classes"])

    def forward(self, x, mask):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)

        for blk in self.blocks:
            h = blk(h, mask)

        h = self.norm(h)

        # Use explicit CLS token at position 0
        return self.head(h[:, 0])



def run_experiment(attn_types, datasets=None, epochs=None, lr=None, wd=None):
    if datasets is None:
        datasets = DATASETS_TO_RUN

    for task_name in datasets:
        if task_name not in TASK_CONFIGS:
            raise ValueError(f"Unsupported dataset in run_experiment: {task_name}")

        print(f"\n{'=' * 20} DATASET: {task_name.upper()} {'=' * 20}")
        train_dl, test_dl, data_info = get_data(task_name)
        task_cfg = data_info["task_cfg"]

        cfg = {
            "task_name": task_name,
            "dataset_display_name": data_info["dataset_display_name"],
            "vocab_size": data_info["vocab_size"],
            "context_length": data_info["context_length"],
            "num_classes": data_info["num_classes"],
            "emb_dim": task_cfg["emb_dim"],
            "n_heads": task_cfg["n_heads"],
            "n_layers": task_cfg["n_layers"],
            "drop_rate": task_cfg["drop_rate"],
            "qkv_bias": task_cfg["qkv_bias"],
            "M": task_cfg["M"],
            "K": task_cfg["K"],
            "L": task_cfg["L"],
            "hyper_num_bits": task_cfg["hyper_num_bits"],
            "hyper_block_size": task_cfg["hyper_block_size"],
            "hyper_min_seq_len": task_cfg["hyper_min_seq_len"],
            "hyper_neighbor_blocks": task_cfg["hyper_neighbor_blocks"],
            "gate_hidden_dim": task_cfg["gate_hidden_dim"],
            "gate_normalize": task_cfg["gate_normalize"],
            "hyper_plus_race_q_chunk_size": task_cfg["hyper_plus_race_q_chunk_size"],
        }

        task_epochs = task_cfg["epochs"] if epochs is None else epochs
        task_lr = task_cfg["lr"] if lr is None else lr
        task_wd = task_cfg["weight_decay"] if wd is None else wd
        task_batch = task_cfg["batch_size"]

        for kind in attn_types:
            run = wandb.init(
                project="RACE",
                name=f"{task_name}_{kind}_{cfg['context_length']}",
                config={
                    "dataset": cfg["dataset_display_name"],
                    "attn_type": kind,
                    "context_length": cfg["context_length"],
                    "emb_dim": cfg["emb_dim"],
                    "n_heads": cfg["n_heads"],
                    "n_layers": cfg["n_layers"],
                    "drop_rate": cfg["drop_rate"],
                    "batch_size": task_batch,
                    "epochs": task_epochs,
                    "lr": task_lr,
                    "weight_decay": task_wd,
                    "M": cfg["M"],
                    "K": cfg["K"],
                    "L": cfg["L"],
                    "hyper_num_bits": cfg["hyper_num_bits"],
                    "hyper_block_size": cfg["hyper_block_size"],
                    "hyper_min_seq_len": cfg["hyper_min_seq_len"],
                    "hyper_neighbor_blocks": cfg["hyper_neighbor_blocks"],
                    "gate_hidden_dim": cfg["gate_hidden_dim"],
                    "gate_normalize": cfg["gate_normalize"],
                    "hyper_plus_race_q_chunk_size": cfg["hyper_plus_race_q_chunk_size"],
                }
            )

            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("time/*", step_metric="epoch")
            wandb.define_metric("gates/*", step_metric="epoch")
            wandb.define_metric("mix/*", step_metric="epoch")
            wandb.define_metric("val/acc", summary="max")
            wandb.define_metric("val/loss", summary="min")

            print(f"\n=== Training {kind.upper()} on {task_name.upper()} for {task_epochs} epochs ===")
            model = Classifier(cfg, kind).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=task_lr, weight_decay=task_wd)

            for ep in range(1, task_epochs + 1):
                # ---------------- TRAIN ----------------
                if "cuda" in str(DEVICE):
                    torch.cuda.synchronize()
                t0 = time.time()

                model.train()
                tl = 0.0
                ta = 0.0
                loop_t0 = time.time()

                train_pbar = tqdm(train_dl, desc=f"Epoch {ep} [train]", leave=True)
                for step, (x, mask, y) in enumerate(train_pbar, start=1):
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
                    elapsed = time.time() - loop_t0
                    train_pbar.set_postfix({
                        "loss": f"{tl / step:.4f}",
                        "acc": f"{ta / step:.4f}",
                        "s/it": f"{elapsed / step:.3f}",
                    })

                if "cuda" in str(DEVICE):
                    torch.cuda.synchronize()
                train_time = time.time() - t0
                tr_l = tl / len(train_dl)
                tr_a = ta / len(train_dl)

                # ---------------- VALID ----------------
                if "cuda" in str(DEVICE):
                    torch.cuda.synchronize()
                t1 = time.time()

                model.eval()
                vl = 0.0
                va = 0.0
                loop_t1 = time.time()

                val_pbar = tqdm(test_dl, desc=f"Epoch {ep} [val]", leave=True)
                with torch.no_grad():
                    for step, (x, mask, y) in enumerate(val_pbar, start=1):
                        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                        logits = model(x, mask)
                        loss = F.cross_entropy(logits, y)
                        acc = (logits.argmax(-1) == y).float().mean().item()

                        vl += loss.item()
                        va += acc
                        elapsed = time.time() - loop_t1
                        val_pbar.set_postfix({
                            "loss": f"{vl / step:.4f}",
                            "acc": f"{va / step:.4f}",
                            "s/it": f"{elapsed / step:.3f}",
                        })

                if "cuda" in str(DEVICE):
                    torch.cuda.synchronize()
                val_time = time.time() - t1
                va_l = vl / len(test_dl)
                va_a = va / len(test_dl)

                extra_logs = {}

                if kind in {"hyper_race", "hyper_plus_race"}:
                    hyper_means = []
                    race_means = []
                    for layer_idx, layer in enumerate(model.blocks):
                        if hasattr(layer, "att") and hasattr(layer.att, "last_gates"):
                            gates = layer.att.last_gates
                            if gates is not None:
                                hyper_g = gates[..., 0]
                                race_g  = gates[..., 1]

                                extra_logs[f"gates/layer{layer_idx}_hyper_mean"] = hyper_g.mean().item()
                                extra_logs[f"gates/layer{layer_idx}_race_mean"]  = race_g.mean().item()
                                extra_logs[f"gates/layer{layer_idx}_hyper_std"]  = hyper_g.std().item()
                                extra_logs[f"gates/layer{layer_idx}_race_std"]   = race_g.std().item()
                                extra_logs[f"gates/layer{layer_idx}_hyper_min"]  = hyper_g.min().item()
                                extra_logs[f"gates/layer{layer_idx}_race_min"]   = race_g.min().item()
                                extra_logs[f"gates/layer{layer_idx}_hyper_max"]  = hyper_g.max().item()
                                extra_logs[f"gates/layer{layer_idx}_race_max"]   = race_g.max().item()

                                hyper_means.append(hyper_g.mean().item())
                                race_means.append(race_g.mean().item())

                                if ep % 5 == 0:
                                    extra_logs[f"gates_hist/layer{layer_idx}_hyper"] = wandb.Histogram(
                                        hyper_g.detach().cpu().flatten().numpy()
                                    )
                                    extra_logs[f"gates_hist/layer{layer_idx}_race"] = wandb.Histogram(
                                        race_g.detach().cpu().flatten().numpy()
                                    )

                    if len(hyper_means) > 0:
                        extra_logs["gates/global_hyper_mean"] = sum(hyper_means) / len(hyper_means)
                        extra_logs["gates/global_race_mean"] = sum(race_means) / len(race_means)

                if kind == "fixed_hyper_plus_race":
                    a_vals, b_vals = [], []
                    for layer_idx, layer in enumerate(model.blocks):
                        if hasattr(layer, "att"):
                            att = layer.att
                            if hasattr(att, "a") and hasattr(att, "b"):
                                a_val = att.a.detach().item()
                                b_val = att.b.detach().item()
                                extra_logs[f"mix/layer{layer_idx}_a"] = a_val
                                extra_logs[f"mix/layer{layer_idx}_b"] = b_val
                                a_vals.append(a_val)
                                b_vals.append(b_val)
                    if len(a_vals) > 0:
                        extra_logs["mix/global_a_mean"] = sum(a_vals) / len(a_vals)
                        extra_logs["mix/global_b_mean"] = sum(b_vals) / len(b_vals)

                wandb.log({
                    "epoch": ep,
                    "train/loss": tr_l,
                    "train/acc": tr_a,
                    "val/loss": va_l,
                    "val/acc": va_a,
                    "time/train_sec": train_time,
                    "time/val_sec": val_time,
                    **extra_logs,
                }, step=ep)

                print(
                    f"Ep{ep:3d} | "
                    f"train_loss {tr_l:.4f}, acc {tr_a:.4f} ({train_time:.1f}s) | "
                    f"val_loss {va_l:.4f}, acc {va_a:.4f} ({val_time:.1f}s)"
                )

            wandb.finish()


if __name__ == "__main__":
    run_experiment(
        attn_types=["exact_flash", "race", "hyper_lsh", "hyper_race", "hyper_plus_race"],
        datasets=DATASETS_TO_RUN,
    )
