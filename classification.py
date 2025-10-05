# ==================================================
# 0) Imports & Hyper‑params
# ==================================================
import re, random, math, time, itertools, os
from math import ceil
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==================================================
# Adjust these!
# ==================================================
VOCAB_LIMIT = 50_000
MAX_LEN     = 512
BATCH       = 32
SEED        = 42
EPOCHS      = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------

if DEVICE == "cuda":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cudnn.benchmark = True


# ==================================================
# 1) “basic_english” tokenizer drop‑in
# ==================================================
_basic_english_re = re.compile(
    r"""([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])   # any punctuation
     |(\d+[%]?)                                  # numbers (and percent)
     |([A-Za-z]+(?:'[A-Za-z]+)?)                 # words w/ optional apos
    """,
    re.VERBOSE,
)
def basic_english_tokenizer(text: str) -> list[str]:
    text = text.lower()
    tokens = []
    for punc, num, word in _basic_english_re.findall(text):
        if punc:   tokens.append(punc)
        elif num:  tokens.append(num)
        elif word: tokens.append(word)
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
    for _ in range(n_swaps):
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    return toks


# ==================================================
# 3) Load IMDB and build vocab
# ==================================================
raw = load_dataset("imdb")
train_examples = list(zip(raw["train"]["label"], raw["train"]["text"]))
test_examples  = list(zip(raw["test"]["label"],  raw["test"]["text"]))

counter = Counter()
for lbl, txt in train_examples:
    counter.update(tok(txt))
most_common = [w for w,_ in counter.most_common(VOCAB_LIMIT)]
stoi = {w:i+2 for i,w in enumerate(most_common)}
stoi["<pad>"] = 0; stoi["<unk>"] = 1
PAD_IDX, UNK_IDX = 0, 1
VOCAB_SIZE = len(stoi)


# ==================================================
# 4) AugmentedIMDB Dataset
# ==================================================
class AugmentedIMDB(Dataset):
    def __init__(self, examples, max_len, augment=False):
        self.examples = examples
        self.max_len  = max_len
        self.augment  = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        lbl, txt = self.examples[idx]
        toks = tok(txt)
        if self.augment:
            op = random.choice(["del","swap",None])
            if   op=="del":  toks = eda_random_deletion(toks)
            elif op=="swap": toks = eda_random_swap(toks)
        toks = toks[: self.max_len]
        ids  = [stoi.get(t, UNK_IDX) for t in toks]
        if len(ids) < self.max_len:
            ids += [PAD_IDX] * (self.max_len - len(ids))
        return lbl, torch.tensor(ids, dtype=torch.long)

    def collate_fn(self, batch):
        labels, texts = zip(*batch)
        texts = torch.stack(texts, dim=0)
        masks = (texts != PAD_IDX).long()
        return texts, masks, torch.tensor(labels, dtype=torch.long)


# ==============================================================
# 5) Exactly‑copied “long” splitting with deterministic seeding
# ==============================================================
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
        step = MAX_LEN // 2
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
    for pool, label in [(short_pos,1),(short_neg,0)]:
        i = 0
        while i + 1 < len(pool):
            combo = tok(pool[i]) + tok(pool[i+1])
            if len(combo) >= MAX_LEN:
                new_long.append((label, " ".join(combo)))
                i += 2
            else:
                if i + 2 < len(pool):
                    combo3 = combo + tok(pool[i+2])
                    if len(combo3) >= MAX_LEN:
                        new_long.append((label, " ".join(combo3)))
                        i += 3; continue
                i += 1

    long_train = real_long + split_long + new_long
    random.seed(SEED); random.shuffle(long_train)

    long_test = [(lbl,txt) for lbl,txt in examples if len(tok(txt)) >= MAX_LEN]
    random.seed(SEED); random.shuffle(long_test)

    return long_train, long_test


# ==================================================
# 6) DataLoaders
# ==================================================
def get_data():
    long_train, _ = make_long_subsets(train_examples)
    print(f"--> FINAL long_train: {len(long_train)}")

    _, long_test   = make_long_subsets(test_examples)
    print(f"--> FINAL long_test:  {len(long_test)}")
    train_ds = AugmentedIMDB(long_train, MAX_LEN, augment=True)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, drop_last=True,
        pin_memory=(DEVICE == "cuda"), num_workers=4,
        collate_fn=train_ds.collate_fn,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_ds = AugmentedIMDB(long_test, MAX_LEN, augment=False)
    test_dl = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        pin_memory=(DEVICE == "cuda"), num_workers=2,
        collate_fn=test_ds.collate_fn,
    )
    return train_dl, test_dl
    

# ==================================================
# 7) Attention blocks (pad‑mask aware)
# ==================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        self.h, self.dk = h, d//h
        self.q = nn.Linear(d,d, bias=qkv_bias)
        self.k = nn.Linear(d,d, bias=qkv_bias)
        self.v = nn.Linear(d,d, bias=qkv_bias)
        self.o = nn.Linear(d,d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        B,T,_ = x.shape
        Q = self.q(x).view(B,T,self.h,self.dk).transpose(1,2)
        K = self.k(x).view(B,T,self.h,self.dk).transpose(1,2)
        V = self.v(x).view(B,T,self.h,self.dk).transpose(1,2)
        
        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.dk)
        if mask is not None:
            pad = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad==0, float("-inf"))
        W = torch.softmax(scores, -1)
        W = self.drop(W)
        out = (W @ V).transpose(1,2).contiguous().view(B,T,self.h*self.dk)
        return self.o(out)


class AngularAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        self.h, self.dk = h, d//h
        self.q = nn.Linear(d,d, bias=qkv_bias)
        self.k = nn.Linear(d,d, bias=qkv_bias)
        self.v = nn.Linear(d,d, bias=qkv_bias)
        self.o = nn.Linear(d,d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        B,T,_ = x.shape
        Q = F.normalize(self.q(x).view(B,T,self.h,self.dk).transpose(1,2), dim=-1)
        K = F.normalize(self.k(x).view(B,T,self.h,self.dk).transpose(1,2), dim=-1)
        V = self.v(x).view(B,T,self.h,self.dk).transpose(1,2)
        sim = (Q @ K.transpose(-2,-1)).clamp(-0.999,0.999)
        scores = 1 - torch.acos(sim)/math.pi
        if mask is not None:
            pad = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad==0, 0.0)
        W = scores.clamp(min=1e-6).pow(18)
        W = W / (W.sum(-1,keepdim=True)+1e-6)
        W = self.drop(W)
        out = (W @ V).transpose(1,2).contiguous().view(B,T,self.h*self.dk)
        return self.o(out)


class BatchedACE(nn.Module):
    def __init__(self, d_k, K, L, M):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        # planes: [L*K, d_k]  → project any [*,d_k] → [*,L,K]
        planes = torch.randn(L, K, d_k)
        self.register_buffer("planes_T", planes.view(L*K, d_k).T)    # [d_k, L*K]
        # protos:  [K, R]      → project any [*,K]   → [*,R]
        corners = torch.tensor(list(itertools.product([-1.,+1.],repeat=K)))
        self.register_buffer("protos_T", corners.T)                  # [K, R]

    def forward(self, Kh, Vh, Qh):
        # Kh,Vh,Qh: [M, B, T, H, d_k]
        M,B,T,H,dk = Kh.shape
        assert M==self.M and dk==self.d_k

        # 1) flatten out to a big batch for keys:
        #    flat_K: [M*B*T*H, d_k]
        flat_K = Kh.contiguous().view(-1, dk)
        #    projK_flat: [M*B*T*H, L*K]
        projK_flat = flat_K @ self.planes_T
        #    → [M*B*T*H, L, K]
        projK = projK_flat.view(-1, self.L, self.K)

        # 2) compute soft‑hash logits & probs:
        #    [M*B*T*H*L, K] @ [K, R] → [M*B*T*H*L, R]
        logitsK = (projK.tanh().div(dk**0.5).view(-1, self.K) @ self.protos_T).view(M, B, T, H, self.L, self.R)
        probsK  = logitsK.softmax(dim=-1)     # [M,B,T,H,L,R]

        # 3) build your bucket‐summaries E via two small batched bmm’s:
        #    - collapse M,B,H into one dim so we can bmm along T
        MBH = M*B*H
        #    probs_flat: [MBH, T, L*R]
        probs_flat = probsK.permute(0,1,3,2,4,5).contiguous().view(MBH, T, self.L*self.R)
        #    V_flat:      [MBH, T, d_k]
        V_flat     = Vh.permute(0,1,3,2,4).contiguous().view(MBH, T, dk)
        #    b_sum:      [MBH, L*R, d_k]
        b_sum = probs_flat.transpose(1,2).bmm(V_flat)
        #    A:          [MBH, 1, L*R]
        A = probs_flat.sum(dim=1, keepdim=True)
        #    E_flat:     [MBH, L*R, d_k] normalized
        E_flat = b_sum / (A.transpose(1,2) + 1e-6)

        # 4) same for queries → final outputs
        #    projQ → probsQ exactly like projK/logitsK/probsK
        flat_Q = Qh.contiguous().view(-1, dk)
        projQ  = (flat_Q @ self.planes_T).view(-1, self.L, self.K)
        logitsQ= ((projQ.tanh().div(dk**0.5).view(-1, self.K) @ self.protos_T).view(M, B, T, H, self.L, self.R))
        probsQ = logitsQ.softmax(dim=-1)
        #    probsQ_flat: [MBH, T, L*R]
        probsQ_flat = probsQ.permute(0,1,3,2,4,5).contiguous().view(MBH, T, self.L*self.R)

        # 5) expected‐value lookup via one more bmm:
        #    [MBH, T, L*R] @ [MBH, L*R, d_k] → [MBH, T, d_k]
        out_flat = probsQ_flat.bmm(E_flat)    # [MBH, T, dk]
        # 6) un‐flatten & return
        return out_flat.view(M, B, H, T, dk).permute(0,1,3,2,4)



class RACEAttention(nn.Module):
    """Bidirectional RACEAttention using non‑causal ACE."""
    def __init__(self, d, h, drop, M=2, K=3, L=2, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.M = h, d//h, M
        self.q = nn.Linear(d,d, bias=qkv_bias)
        self.k = nn.Linear(d,d, bias=qkv_bias)
        self.v = nn.Linear(d,d, bias=qkv_bias)
        self.o = nn.Linear(d,d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(self.dk, K, L, M)

    def forward(self, x, mask):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk)
        K = self.k(x).view(B, T, self.h, self.dk)
        V = self.v(x).view(B, T, self.h, self.dk)

        # zero‑out pad tokens
        if mask is not None:
            m = mask.unsqueeze(-1).unsqueeze(-1)
            Q, K, V = Q*m, K*m, V*m

        # pack for ACE: [M,B,T,H,dk]
        def pack(z):
            return z.unsqueeze(0).expand(self.M, -1, -1, -1, -1)

        out_m = self.ace(pack(K), pack(V), pack(Q))     # [M,B,T,H,dk]
        out   = out_m.mean(dim=0)                       # [B,T,H,dk]
        out   = out.transpose(1,2).contiguous().view(B, T, -1)     # [B,T,d]
        return self.drop(self.o(out))


class RACEBlock(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.att = RACEAttention(
            d = cfg["emb_dim"],
            h = cfg["n_heads"],
            drop = cfg["drop_rate"],
            M = 2,          # number of ensembles
            K = 3,          # hash bits
            L = 2,          # number of hash tables
            qkv_bias = cfg.get("qkv_bias", False)
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

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

class TransformerBlock(nn.Module):
    """Standard softmax‐attention Transformer block, pad‐mask aware."""
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(d=cfg["emb_dim"], h=cfg["n_heads"], drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

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
    """Angular (cosine)‐attention block, pad‐mask aware."""
    def __init__(self, cfg):
        super().__init__()
        self.att = AngularAttention(d=cfg["emb_dim"], h=cfg["n_heads"], drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])

        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

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
# 8) Model & run_experiment
# ==================================================
class Classifier(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop    = nn.Dropout(cfg["drop_rate"])

        # Build the blocks list correctly
        self.blocks = nn.ModuleList()
        for _ in range(cfg["n_layers"]):
            if kind == "softmax":
                self.blocks.append( TransformerBlock(cfg) )
            elif kind == "angular":
                self.blocks.append( AngularBlock(cfg) )
            elif kind == "race":
                self.blocks.append( RACEBlock(cfg, device=DEVICE) )
            else:
                raise ValueError(kind)

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
        return self.head(h[:,0])



def run_experiment(attn_types, epochs=5, lr=1e-5, wd=5e-05):
    cfg = {
      "vocab_size": VOCAB_SIZE,
      "context_length": MAX_LEN,
      "emb_dim": 256,
      "n_heads": 2,
      "n_layers": 1,
      "drop_rate": 0.1,
      "qkv_bias": False,      # <— add this
    }


    for kind in attn_types:
        print(f"\n=== Training {kind.upper()} for {epochs} epochs ===")
        model = torch.compile(Classifier(cfg, kind)).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        train_dl, test_dl = get_data()
        for ep in range(1, epochs+1):

            # --- train timing ---
            t0= time.time()
            model.train(); tl=ta=0
            for x,mask,y in train_dl:
                x,mask,y = x.to(DEVICE),mask.to(DEVICE),y.to(DEVICE)
                opt.zero_grad()
                logits = model(x,mask)
                loss   = F.cross_entropy(logits,y)
                acc    = (logits.argmax(-1)==y).float().mean().item()
                loss.backward(); opt.step()
                tl += loss.item(); ta += acc
            tr_l, tr_a = tl/len(train_dl), ta/len(train_dl)
            train_time = time.time() - t0
            
            # --- eval timing ---
            model.eval()
            t1 = time.time()
            vl=va=0
            with torch.no_grad():
                for x,mask,y in test_dl:
                    x,mask,y = x.to(DEVICE),mask.to(DEVICE),y.to(DEVICE)
                    logits = model(x,mask)
                    vl += F.cross_entropy(logits,y).item()
                    va += (logits.argmax(-1)==y).float().mean().item()
            va_l, va_a = vl/len(test_dl), va/len(test_dl)
            val_time = time.time() - t1

            print(
                f"Ep{ep:2d} | "
                f"train_loss {tr_l:.3f}, acc {tr_a:.3f} "
                f"({train_time:.1f}s) | "
                f"val_loss   {va_l:.3f}, acc {va_a:.3f} "
                f"({val_time:.1f}s)"
            )

if __name__ == "__main__":
    run_experiment(["softmax", "race"], epochs=10)
