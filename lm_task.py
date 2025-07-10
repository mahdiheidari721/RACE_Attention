from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from race import RACEModel
from gpt import GPTModel, AngularModel

torch.autograd.set_detect_anomaly(True)

# ------------ CONSTANTS ------------
SAMPLE_SIZE = 12  # Reduced dataset size

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
        print("Data loader is empty for loss calculation.")
        return float("nan"), float("nan")
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
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_iter):
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
            model, train_loader, val_loader, device, eval_iter)
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

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss, train_acc = calc_loss_acc_loader(train_loader, model, device, num_batches=eval_iter)
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
        batch_size=4,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"] // 2,
        drop_last=True,
        shuffle=False,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=4,
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
    device = "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader = load_small_tinystories()
    num_epochs = 20

    # ------------------ TRAINING GPT -----------------
    print("Training GPT model...")
    torch.manual_seed(123)
    model_gpt = GPTModel(GPT_CONFIG_124M)
    model_gpt.to(device)
    optimizer_gpt = torch.optim.AdamW(model_gpt.parameters(), lr=0.0001, weight_decay=0.1)

    metrics_gpt = train_model_simple(
        model_gpt, train_loader, val_loader, optimizer_gpt, device,
        num_epochs=num_epochs, eval_iter=None
    )
  
    # ------------------ TRAINING RACE -----------------
    print("Training RACE model...")
    torch.manual_seed(123)
    model_race = RACEModel(GPT_CONFIG_124M)
    model_race.to(device)
    optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=0.0001, weight_decay=0.1)

    metrics_race = train_model_simple(
        model_race, train_loader, val_loader, optimizer_race, device,
        num_epochs=num_epochs, eval_iter=None
    )

    # ------------------ Training Angular ---------------
    print("Training Angular model...")
    torch.manual_seed(123)
    model_angular = AngularModel(GPT_CONFIG_124M)
    model_angular.to(device)
    optimizer_angular = torch.optim.AdamW(model_angular.parameters(), lr=0.0001, weight_decay=0.1)

    metrics_angular = train_model_simple(
        model_angular, train_loader, val_loader, optimizer_angular, device,
        num_epochs=num_epochs, eval_iter=None
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
