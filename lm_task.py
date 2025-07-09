from torch.utils.data import Dataset, DataLoader
import os
import tiktoken
import torch
import torch.nn as nn
from torch.nn import GELU
import numpy as np
import random
import urllib.request
from datasets import load_dataset
import time
from tqdm import tqdm
from maxk import MaxkModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import json
import math
from race import build_race_sketches, calc_loss_acc_loader_race, RACE

torch.autograd.set_detect_anomaly(True)

# ------------ CONSTANTS ------------
SAMPLE_SIZE = 15  # Reduced dataset size

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 64, # Context length
    "emb_dim": 512,         # Embedding dimension
    "n_heads": 8,          # Number of attention heads
    "n_layers": 4,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# ------------------------------------

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
                         stride, shuffle=True, drop_last=True,
                         num_workers=0):

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
        num_workers=num_workers
    )

    return dataloader

# ------------------------------------

# ------------ MODEL DEFINITION ------------
class MultiHeadAttention(nn.Module):
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
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
   
        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

class RACEAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim
        self.norm = nn.LayerNorm(d_in)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x, use_sketches=False):
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

        # Normalize for cosine similarity
        queries = F.normalize(queries, dim=-1, p=2, eps=1e-6)
        keys = F.normalize(keys, dim=-1, p=2, eps=1e-6)

        context_vec = torch.zeros_like(queries)  # (B, H, T, D_h)
        if use_sketches:
            sketches = [
                RACE(D_dim=self.d_out, K=8, L=5, N_M=3, D_out=self.d_out, device=x.device)
                for _ in range(B)
            ]

            # Collapse heads for sketching: reshape to [B, T, D]
            q_flat = queries.transpose(1, 2).contiguous().view(B, num_tokens, self.d_out)
            k_flat = keys.transpose(1, 2).contiguous().view(B, num_tokens, self.d_out)
            v_flat = values.transpose(1, 2).contiguous().view(B, num_tokens, self.d_out)

            for b in range(B):
                sketch = sketches[b]
                for t in range(num_tokens):
                    sketch.add(k_flat[b, t], v_flat[b, t])
                    context = sketch.score(q_flat[b, t])
                    context_vec[b, :, t, :] = context.view(self.num_heads, self.head_dim)

        else:
            # Cosine similarity: [B, H, T_q, T_k]
            cos_sim = queries @ keys.transpose(2, 3)
            cos_sim = cos_sim.clamp(min=-0.999, max=0.999) 
            attn_scores = 1 - torch.acos(cos_sim) / torch.pi
            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, 0.0)
            attn_weights = attn_scores.clamp(min=1e-6).pow(18.0)  # sharpen with γ
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)

            attn_weights = self.dropout(attn_weights)
            if torch.isnan(attn_weights).any():
                with open("nan_log.txt", "a") as f:
                    f.write("Null\n")
            context_vec = (attn_weights @ values)
            
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(B, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

    def get_qkv(self, x):
        # x: [B, T, D]
        b, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        return queries, keys, values

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
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
        # 2*4*768
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
        # 2*4*768

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


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

    def forward(self, x, use_sketches=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_sketches)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        # 2*4*768
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

class RACEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.ModuleList(
            [RACEBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx, use_sketches=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, use_sketches=use_sketches)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
# ------------------------------------

# ------------ EXAMPLE DATA ------------

def calc_loss_acc_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    # Compute accuracy
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)  # Get indices of max logit
        correct = (predictions == target_batch).float()
        acc = correct.mean().item()  # Convert to scalar float
    return loss, acc


def calc_loss_acc_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    total_acc = 0.0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss, acc = calc_loss_acc_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            total_acc += acc
        else:
            break
    return total_loss / num_batches, total_acc / num_batches


# ------------ TRAINING ------------
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_iter, tokenizer, gpt=False):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses = [], []
    train_ppls, val_ppls = [], []
    train_accs, val_accs = [], []

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss, _ = calc_loss_acc_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # Update model weights using loss gradients

        train_loss, val_loss, train_acc, val_acc = evaluate_model(
            model, train_loader, val_loader, device, eval_iter, use_sketches=True, gpt=gpt)
        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Ep {epoch+1}): "
                f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} | Train PPL {train_ppl:.3f}, Val PPL {val_ppl:.3f} | Train acc {train_acc:.3f}, Val acc {val_acc:.3f}")  

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_ppl": train_ppls,
        "val_ppl": val_ppls,
        "train_acc": train_accs,
        "val_acc": val_accs,
    }

def evaluate_model(model, train_loader, val_loader, device, eval_iter, use_sketches=False, gpt=False):
    model.eval()
    with torch.no_grad():
        train_loss, train_acc = calc_loss_acc_loader(train_loader, model, device, num_batches=eval_iter)
        if use_sketches is not False and gpt is False:
            val_loss, val_acc = calc_loss_acc_loader_race(val_loader, model, device, num_batches=eval_iter)
        else:
            val_loss, val_acc = calc_loss_acc_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss, train_acc, val_acc

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
        batch_size=8,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"] // 2,
        drop_last=True,
        shuffle=False,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=8,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"] // 2,
        drop_last=True,
        shuffle=False,
        num_workers=0
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
    device = "cuda"
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader = load_small_tinystories()
    num_epochs = 30

    # ------------------ TRAINING GPT -----------------
    print("Training GPT model...")
    torch.manual_seed(123)
    model_gpt = GPTModel(GPT_CONFIG_124M)
    model_gpt.to(device)
    optimizer_gpt = torch.optim.AdamW(model_gpt.parameters(), lr=0.0001, weight_decay=0.1)

    metrics_gpt = train_model_simple(
        model_gpt, train_loader, val_loader, optimizer_gpt, device,
        num_epochs=num_epochs, eval_iter=None, tokenizer=tokenizer, gpt=True
    )
  
    # ------------------ TRAINING RACE -----------------
    print("Training RACE model...")
    torch.manual_seed(123)
    model_race = RACEModel(GPT_CONFIG_124M)
    model_race.to(device)
    optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=0.0001, weight_decay=0.1)

    metrics_race = train_model_simple(
        model_race, train_loader, val_loader, optimizer_race, device,
        num_epochs=num_epochs, eval_iter=None, tokenizer=tokenizer, gpt=False
    )

    plot_comparison_metrics(metrics_race, metrics_gpt, "race.png")

def plot_comparison_metrics(metrics_race, metrics_gpt, save_path):
    epochs = range(1, len(metrics_race["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.3)

    def plot_metric(ax, metric_key, ylabel, title):
        # Plot MaxKAttention
        ax.plot(epochs, metrics_race[f"train_{metric_key}"], label="Race - Train", color="#1f77b4", marker='o', markersize=4, linewidth=2)
        ax.plot(epochs, metrics_race[f"val_{metric_key}"], label="Race - Val", color="#800080", linestyle='--', marker='x', markersize=4, linewidth=2)

        # Plot GPT (Softmax)
        ax.plot(epochs, metrics_gpt[f"train_{metric_key}"], label="GPT - Train", color="#2ca02c", marker='s', markersize=4, linewidth=2)
        ax.plot(epochs, metrics_gpt[f"val_{metric_key}"], label="GPT - Val", color="#d62728", linestyle='--', marker='^', markersize=4, linewidth=2)

        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10, loc="best")

    plot_metric(axes[0], "loss", "Cross-Entropy Loss", "Loss (Train vs Val)")
    plot_metric(axes[1], "acc", "Accuracy", "Accuracy (Train vs Val)")
    plot_metric(axes[2], "ppl", "Perplexity", "Perplexity (Train vs Val)")

    fig.suptitle(f"RACEAttention vs MultiHeadAttention", fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300)
    plt.show()

start_experiment()
