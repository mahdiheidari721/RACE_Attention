import os 
# lm_task.py — put these at the very top
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")      # start with 1 to rule out OMP deadlocks
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")        # harmless on libomp, helps if iomp is present

from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
from torch import nn
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from gpt import TransformerBlock, AngularBlock
import torch.nn.functional as F
import math
from race import RACEBlock
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# ------------ CONSTANTS ------------
SAMPLE_SIZE = 7000  # Reduced dataset size

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 512, # Context length
    "emb_dim": 128,         # Embedding dimension
    "n_heads": 2,          # Number of attention heads
    "n_layers": 8,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "K": 2,
    "L": 2,
    "M": 1
}

# ------------ Unified Model Class ------------
class LMModel(nn.Module):
    def __init__(self, cfg, attn_type="gpt", device="cpu"):
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
        if attn_type == "gpt":
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

# ----------------------------------------------------

# ------------ DATA LOADING ------------

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size, max_length, 
                         stride, shuffle=True, drop_last=True):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0
    )

    return dataloader

# ------------------------------------

# ------------ EXAMPLE DATA ------------

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_iter=None):
    train_losses, val_losses = [], []
    train_ppls, val_ppls = [], []
    train_accs, val_accs = [], []
    train_times, val_times = [], []

    for epoch in range(1, num_epochs + 1):
        # === TRAIN ===
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # accuracy
            acc = (logits.argmax(-1) == y).float().mean().item()
            total_loss += loss.item()
            total_acc += acc
            
        train_time = time.time() - t0
        train_times.append(train_time)
        tr_l = total_loss / len(train_loader)
        tr_a = total_acc / len(train_loader)
        tr_p = math.exp(tr_l)

        # === VALIDATION ===
        t1 = time.time()
        model.eval()
        val_loss_total = 0.0
        val_acc_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                acc = (logits.argmax(-1) == y).float().mean().item()
                val_loss_total += loss.item()
                val_acc_total += acc
        val_time = time.time() - t1
        val_times.append(val_time)
        va_l = val_loss_total / len(val_loader)
        va_a = val_acc_total / len(val_loader)
        va_p = math.exp(va_l)

        # Store
        train_losses.append(tr_l)
        val_losses.append(va_l)
        train_ppls.append(tr_p)
        val_ppls.append(va_p)
        train_accs.append(tr_a)
        val_accs.append(va_a)

        print(
            f"Ep{epoch:2d} | "
            f"train loss {tr_l:.3f}, acc {tr_a:.3f}, ppl {tr_p:.2f} "
            f"(train_time {train_time:.1f}s) | "
            f"val loss {va_l:.3f}, acc {va_a:.3f}, ppl {va_p:.2f} "
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


def load_small_tinystories():
    dataset = load_dataset("roneneldan/TinyStories")
    text_data = dataset["train"]["text"][:SAMPLE_SIZE]
    text_data = " ".join(text_data)  # Join all text into a single string
    tokenizer = tiktoken.get_encoding("gpt2")
    char_count = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Character count: {char_count}")
    print(f"Token count: {total_tokens}")

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=16,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"] // 2,
        drop_last=True,
        shuffle=False
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=16,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"] // 2,
        drop_last=True,
        shuffle=False
    )

    print(f"Train data size: {len(train_loader.dataset)}")
    print(f"Validation data size: {len(val_loader.dataset)}")
    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "decrease the `training_ratio`")

    return train_loader, val_loader

def start_experiment():
    device = "cpu"
    train_loader, val_loader = load_small_tinystories()
    num_epochs = 4

    # ------------------ TRAINING GPT -----------------
    # print("Training GPT model...")
    # torch.manual_seed(123)
    # model_gpt = torch.compile(LMModel(GPT_CONFIG_124M, attn_type="gpt"))
    # model_gpt.to(device)
    # optimizer_gpt = torch.optim.AdamW(model_gpt.parameters(), lr=1e-5, weight_decay=1e-2)

    # metrics_gpt = train_model_simple(
    #     model_gpt, train_loader, val_loader, optimizer_gpt, device,
    #     num_epochs=num_epochs, eval_iter=None
    # )
  
    # ------------------ TRAINING RACE -----------------
    print("Training RACE model...")
    torch.manual_seed(123)
    model_race = LMModel(GPT_CONFIG_124M, attn_type="race")
    model_race.to(device)
    optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=1e-5, weight_decay=1e-2)

    metrics_race = train_model_simple(
        model_race, train_loader, val_loader, optimizer_race, device,
        num_epochs=num_epochs, eval_iter=None
    )

    # ------------------ Training Angular ---------------
    # print("Training Angular model...")
    # torch.manual_seed(123)
    # model_angular = LMModel(GPT_CONFIG_124M, attn_type="angular")
    # model_angular.to(device)
    # optimizer_angular = torch.optim.AdamW(model_angular.parameters(), lr=1e-5, weight_decay=0.01)

    # metrics_angular = train_model_simple(
    #     model_angular, train_loader, val_loader, optimizer_angular, device,
    #     num_epochs=num_epochs, eval_iter=None
    # )
    
    plot_comparison_metrics(metrics_race, metrics_gpt, f"race_{GPT_CONFIG_124M['context_length']}.png")

def plot_comparison_metrics(metrics_race, metrics_gpt, save_path, seq_len=GPT_CONFIG_124M["context_length"], K=GPT_CONFIG_124M["K"], L=GPT_CONFIG_124M["L"], M=GPT_CONFIG_124M["M"]):
    epochs = range(1, len(metrics_race["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.3)

    def plot_metric(ax, metric_key, ylabel, title):
        # RACE
        ax.plot(epochs, metrics_race[f"train_{metric_key}"], label="RACE - Train", color="#1f77b4", marker='o', markersize=4, linewidth=2)
        ax.plot(epochs, metrics_race[f"val_{metric_key}"], label="RACE - Val", color="#1f77b4", linestyle='--', marker='x', markersize=4, linewidth=2)

        # # AngularAttention
        # ax.plot(epochs, metrics_angular[f"train_{metric_key}"], label="Angular - Train", color="#ff7f0e", marker='s', markersize=4, linewidth=2)
        # ax.plot(epochs, metrics_angular[f"val_{metric_key}"], label="Angular - Val", color="#ff7f0e", linestyle='--', marker='^', markersize=4, linewidth=2)

        # GPT (Softmax)
        ax.plot(epochs, metrics_gpt[f"train_{metric_key}"], label="GPT - Train", color="#2ca02c", marker='D', markersize=4, linewidth=2)
        ax.plot(epochs, metrics_gpt[f"val_{metric_key}"], label="GPT - Val", color="#2ca02c", linestyle='--', marker='v', markersize=4, linewidth=2)

        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, loc="best")

    plot_metric(axes[0], "loss", "Cross-Entropy Loss", "Loss (Train vs Val)")
    plot_metric(axes[1], "ppl", "Perplexity", "Perplexity (Train vs Val)")

    # Compose extra info string
    extra_info = []
    if seq_len is not None:
        extra_info.append(f"Seq Len = {seq_len}")
    if K is not None:
        extra_info.append(f"K = {K}")
    if L is not None:
        extra_info.append(f"L = {L}")
    if M is not None:
        extra_info.append(f"M = {M}")
    info_str = " | ".join(extra_info)

    fig.suptitle(f"RACE vs GPT Attention\n{info_str}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    start_experiment()
