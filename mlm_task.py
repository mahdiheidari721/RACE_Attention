from torch.utils.data import DataLoader
import itertools
import math
import torch
import torch.nn as nn
from datasets import load_dataset
import time
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_float32_matmul_precision('high')

# ------------ CONSTANTS ------------
SAMPLE_SIZE = 10000  # Reduced dataset size

BERT_CONFIG = {
    "vocab_size": 30522,    # Vocabulary size
    "context_length": 128, # Context length
    "emb_dim": 312,         # Embedding dimension
    "n_heads": 8,          # Number of attention heads
    "n_layers": 4,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "K": 3,
    "L": 6,
    "M": 2
}

# ------------------------------------

# ------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj= nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"], num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x); x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x); x = self.drop(x) + h
        return x

# ----------------------------------------------------------

# ---------------------- Angular Model ----------------------
class AngularAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj= nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_query(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        K = self.W_key(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        V = self.W_value(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        Q = F.normalize(Q, dim=-1, eps=1e-6)
        K = F.normalize(K, dim=-1, eps=1e-6)

        cos_sim = (Q @ K.transpose(-2,-1)).clamp(-0.999,0.999)
        scores  = 1 - torch.acos(cos_sim)/torch.pi
        W = scores.clamp(min=1e-6).pow(16.0) # Change exponent to adjust attention sharpness
        W = W / (W.sum(-1,keepdim=True)+1e-6)
        W = self.dropout(W)

        out = (W @ V).transpose(1,2).reshape(B, T, -1)
        return self.out_proj(out)

class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = AngularAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            dropout=cfg["drop_rate"], num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x); x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x); x = self.drop(x) + h
        return x
# ------------------------

# ------------- RACE ------------------
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
        flat_K = Kh.reshape(-1, dk)
        #    projK_flat: [M*B*T*H, L*K]
        projK_flat = flat_K @ self.planes_T
        #    → [M*B*T*H, L, K]
        projK = projK_flat.view(-1, self.L, self.K)

        # 2) compute soft‑hash logits & probs:
        #    [M*B*T*H*L, K] @ [K, R] → [M*B*T*H*L, R]
        logitsK = (projK.tanh().div(dk**0.5)
                        .reshape(-1, self.K)
                   @ self.protos_T
                  ).view(M, B, T, H, self.L, self.R)
        probsK  = logitsK.softmax(dim=-1)     # [M,B,T,H,L,R]

        # 3) build your bucket‐summaries E via two small batched bmm’s:
        #    - collapse M,B,H into one dim so we can bmm along T
        MBH = M*B*H
        #    probs_flat: [MBH, T, L*R]
        probs_flat = probsK.permute(0,1,3,2,4,5).reshape(MBH, T, self.L*self.R)
        #    V_flat:      [MBH, T, d_k]
        V_flat     = Vh.permute(0,1,3,2,4).reshape(MBH, T, dk)
        #    b_sum:      [MBH, L*R, d_k]
        b_sum = probs_flat.transpose(1,2).bmm(V_flat)
        #    A:          [MBH, 1, L*R]
        A = probs_flat.sum(dim=1, keepdim=True)
        #    E_flat:     [MBH, L*R, d_k] normalized
        E_flat = b_sum / (A.transpose(1,2) + 1e-6)

        # 4) same for queries → final outputs
        #    projQ → probsQ exactly like projK/logitsK/probsK
        flat_Q = Qh.reshape(-1, dk)
        projQ  = (flat_Q @ self.planes_T).view(-1, self.L, self.K)
        logitsQ= ((projQ.tanh().div(dk**0.5)
                      .reshape(-1, self.K)
                   @ self.protos_T
                  ).view(M, B, T, H, self.L, self.R))
        probsQ = logitsQ.softmax(dim=-1)
        #    probsQ_flat: [MBH, T, L*R]
        probsQ_flat = probsQ.permute(0,1,3,2,4,5).reshape(MBH, T, self.L*self.R)

        # 5) expected‐value lookup via one more bmm:
        #    [MBH, T, L*R] @ [MBH, L*R, d_k] → [MBH, T, d_k]
        out_flat = probsQ_flat.bmm(E_flat)    # [MBH, T, dk]
        # 6) un‐flatten & return
        return out_flat.view(M, B, H, T, dk).permute(0,1,3,2,4)



class RACEAttention(nn.Module):
    """Bidirectional RACEAttention using non‑causal ACE."""
    def __init__(self, d, h, drop, M, K, L, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.M = h, d//h, M
        self.q = nn.Linear(d,d, bias=qkv_bias)
        self.k = nn.Linear(d,d, bias=qkv_bias)
        self.v = nn.Linear(d,d, bias=qkv_bias)
        self.o = nn.Linear(d,d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(self.dk, K, L, M)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.q(x).view(B, T, self.h, self.dk)
        K = self.k(x).view(B, T, self.h, self.dk)
        V = self.v(x).view(B, T, self.h, self.dk)

        # pack for ACE: [M,B,T,H,dk]
        def pack(z):
            return z.unsqueeze(0).expand(self.M, -1, -1, -1, -1)

        out_m = self.ace(pack(K), pack(V), pack(Q))     # [M,B,T,H,dk]
        out   = out_m.mean(dim=0)                       # [B,T,H,dk]
        out   = out.transpose(1,2).reshape(B, T, -1)     # [B,T,d]
        return self.drop(self.o(out))


class RACEBlock(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.att   = RACEAttention(
            d=cfg["emb_dim"],
            drop=cfg["drop_rate"],
            h=cfg["n_heads"], qkv_bias=cfg["qkv_bias"],
            L=cfg["L"], K=cfg["K"], M=cfg["M"]
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x); x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x); x = self.drop(x) + h
        return x


# --------------------------------------

# ------------ MODEL DEFINITION -----------

class LMModel(nn.Module):
    def __init__(self, cfg, attn_type="gpt", device="cuda"):
        """
        attn_type ∈ {"gpt","angular","race"}
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb= nn.Dropout(cfg["drop_rate"])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # choose attention class
        if attn_type == "bert":
            AttnBlock = TransformerBlock
        elif attn_type == "angular":
            AttnBlock = AngularBlock
        elif attn_type == "race":
            # our custom RACEBlock needs device
            AttnBlock = lambda c: RACEBlock(c, device)
        else:
            raise ValueError(attn_type)

        # build n_layers of whichever block
        self.blocks = nn.Sequential(
            *[AttnBlock(cfg) for _ in range(cfg["n_layers"])]
        )

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.drop_emb(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

# ------------------------------------

def load_small_tinystories():
    dataset = load_dataset("roneneldan/TinyStories")
    raw_texts = dataset["train"].select(range(SAMPLE_SIZE))
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=BERT_CONFIG["context_length"], padding="max_length")

    tokenized = raw_texts.map(tokenize_function, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // BERT_CONFIG["context_length"]) * BERT_CONFIG["context_length"]
        result = {
            k: [t[i:i + BERT_CONFIG["context_length"]] for i in range(0, total_len, BERT_CONFIG["context_length"])]
            for k, t in concatenated.items()
        }
        return result

    grouped = tokenized.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    def tuple_collate_fn(features):
        # Let HuggingFace collator do the masking and batch creation
        batch = data_collator(features)
        return batch["input_ids"], batch["labels"]

    train_size = int(0.9 * len(grouped))
    train_dataset = grouped.select(range(train_size))
    val_dataset = grouped.select(range(train_size, len(grouped)))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=tuple_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=tuple_collate_fn)

    return train_loader, val_loader


# ------------ TRAINING ------------
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_times, val_times = [], []
    train_ppls, val_ppls = [], []

    for epoch in range(1, num_epochs + 1):
        # === TRAIN ===
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            logits = outputs  # [B, T, V]

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Accuracy (only on masked tokens)
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

        train_time = time.time() - t0
        train_times.append(train_time)
        tr_l = total_loss / len(train_loader)
        tr_a = total_correct / total_tokens
        tr_p = math.exp(tr_l)
        train_losses.append(tr_l)
        train_accs.append(tr_a)
        train_ppls.append(tr_p)

        # === VALIDATION ===
        t1 = time.time()
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_tokens = 0

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                outputs = model(input_ids)
                logits = outputs

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                val_loss_total += loss.item()

                preds = logits.argmax(dim=-1)
                mask = labels != -100
                val_correct += ((preds == labels) & mask).sum().item()
                val_tokens += mask.sum().item()

        val_time = time.time() - t1
        val_times.append(val_time)
        va_l = val_loss_total / len(val_loader)
        va_a = val_correct / val_tokens
        va_p = math.exp(va_l)
        val_losses.append(va_l)
        val_accs.append(va_a)
        val_ppls.append(va_p)

        print(
            f"Ep{epoch:2d} | "
            f"train loss {tr_l:.3f}, acc {tr_a:.3f}, ppl {tr_p:.2f} "
            f"(train_time {train_time:.1f}s) | "
            f"val loss   {va_l:.3f}, acc {va_a:.3f}, ppl {va_p:.2f} "
            f"(val_time {val_time:.1f}s)"
        )

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_ppl": train_ppls,
        "val_ppl": val_ppls,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_time": train_times,
        "val_time": val_times,
    }


def start_experiment():
    device = "cuda:2"
    train_loader, val_loader = load_small_tinystories()
    num_epochs = 20
    # ------------------ TRAINING MODELS -----------------
    # print("Training BERT model...")
    # torch.manual_seed(123)
    # model_bert = torch.compile(LMModel(BERT_CONFIG, "bert"))
    # model_bert.to(device)
    # optimizer_bert = torch.optim.AdamW(model_bert.parameters(), lr=0.0001, weight_decay=0.1)

    # metrics_bert = train_model_simple(
    #     model_bert, train_loader, val_loader, optimizer_bert, device,
    #     num_epochs=num_epochs)
    
    print("Training RACE model...")
    torch.manual_seed(123)
    model_race = torch.compile(LMModel(BERT_CONFIG, "race"))
    model_race.to(device)
    optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=0.0001, weight_decay=0.1)

    metrics_race = train_model_simple(
        model_race, train_loader, val_loader, optimizer_race, device,
        num_epochs=num_epochs)
    

    # print("Training Angular model...")
    # torch.manual_seed(123)
    # model_angular = torch.compile(LMModel(BERT_CONFIG, "angular"))
    # model_angular.to(device)
    # optimizer_angular = torch.optim.AdamW(model_angular.parameters(), lr=0.0001, weight_decay=0.1)

    # metrics_race = train_model_simple(
    #     model_angular, train_loader, val_loader, optimizer_angular, device,
    #     num_epochs=num_epochs)
    



start_experiment()
