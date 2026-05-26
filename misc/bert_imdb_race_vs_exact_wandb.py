# ============================================================
# IMDb + pretrained light BERT + exact-vs-RACE ROW-BY-ROW plots
# with LSH same-bucket keys highlighted.
#
# This script does EXACTLY this:
#   - load one random IMDb review (preferably full 512 tokens)
#   - load a light pretrained encoder-only BERT from Hugging Face
#   - for each layer:
#       * build exact normalized attention matrix from QK softmax
#       * build RACE normalized attention matrix from the SAME Q/K
#       * average over heads first  --> [T, T]
#       * build a SAME-BUCKET mask from hard angular LSH (5 planes by default)
#         using majority vote across heads
#       * for EVERY row r in [0, valid_len-1]:
#             plot exact_row[r, :] and race_row[r, :] on the SAME FIGURE
#             highlight SAME-BUCKET keys in red
#             log that FIGURE to Weights & Biases
#
# Notes
# -----
# 1) Flash-attention kernels do not expose the full attention matrix directly,
#    so the "exact flash attention matrix" here is reconstructed explicitly
#    from the same QK softmax distribution that exact / flash attention uses.
# 2) The row-by-row plots are head-averaged (Option B).
# 3) Same-bucket highlighting uses majority vote across heads because the plots
#    are head-averaged.
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Config
# ============================================================
@dataclass
class CFG:
    # public, light, encoder-only pretrained model
    model_name: str = "google/bert_uncased_L-4_H-256_A-4"

    dataset_name: str = "imdb"
    dataset_split: str = "test"

    max_len: int = 512
    require_full_length: bool = True
    max_sampling_trials: int = 3000
    seed: int = 42
    topk_keys_to_mark: int = 64
    # RACE parameters
    race_K: int = 3
    race_L: int = 2
    race_M: int = 2
    race_logit_temp: float = 1.0
    race_eps: float = 1e-6

    # LSH bucket visualization
    lsh_num_bits: int = 5
    #same_bucket_vote_threshold: float = 0.5  # majority vote across heads
    exclude_self_from_same_bucket: bool = False

    # W&B
    wandb_project: str = "RACE_attention_ablation"
    wandb_mode: str = "online"   # "online" or "offline"

    # logging
    row_plot_figsize: tuple = (12, 4)
    rows_per_log_batch: int = 32


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transpose_for_scores(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    x: [B, T, H*D]
    returns: [B, H, T, D]
    """
    B, T, _ = x.shape
    return x.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def maybe_average_heads(attn_mat: torch.Tensor) -> torch.Tensor:
    """
    Input:
      [H, T, T] -> returns [T, T]
      [T, T]    -> returns [T, T]
    """
    if attn_mat.dim() == 3:
        return attn_mat.mean(dim=0)
    elif attn_mat.dim() == 2:
        return attn_mat
    else:
        raise ValueError(f"Unexpected attention matrix shape: {tuple(attn_mat.shape)}")


# ============================================================
# Exact attention matrix from Q/K
# ============================================================
@torch.no_grad()
def build_exact_attention_matrix_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    q, k: [B, H, T, D]
    attention_mask: [B, T] with 1 for valid tokens, 0 for padding

    Returns:
        exact normalized attention matrix of shape [H, T, T] for B=1

    This is the exact row-softmax matrix used by full attention.
    """
    assert q.size(0) == 1 and k.size(0) == 1, "This script expects batch size 1."

    _, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale   # [1, H, T, T]

    # mask padded keys before softmax
    key_mask = attention_mask[:, None, None, :].to(torch.bool)  # [1,1,1,T]
    scores = scores.masked_fill(~key_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)  # [1,H,T,T]

    # zero padded query rows for cleanliness
    query_mask = attention_mask[:, None, :, None].to(attn.dtype)  # [1,1,T,1]
    attn = attn * query_mask

    # renormalize valid rows
    row_sum = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    attn = attn / row_sum

    return attn[0]  # [H,T,T]


# ============================================================
# RACE matrix builder
# ============================================================
class RaceMatrixBuilder(torch.nn.Module):
    """
    Build a row-stochastic RACE-induced attention matrix from Q/K.

    For one ensemble m, the implicit weight is:
        w_ij = sum_s probsQ(i,s) * probsK(j,s) / A_s
    where
        A_s = sum_j probsK(j,s)

    Then we average over M ensembles and row-normalize.
    """
    def __init__(self, num_heads: int, head_dim: int, K: int, L: int, M: int,
                 logit_temp: float, device: torch.device, seed: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.K = K
        self.L = L
        self.M = M
        self.R = 1 << K
        self.S = L * self.R
        self.device = device
        self.logit_temp = logit_temp

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        planes = torch.randn(M, head_dim, L * K, generator=gen)
        self.register_buffer("planes_T", planes.to(device), persistent=False)  # [M, D, L*K]

        corners = torch.tensor(
            list(__import__("itertools").product([-1.0, +1.0], repeat=K)),
            dtype=torch.float32,
        )  # [R, K]
        self.register_buffer("protos_T", corners.t().to(device), persistent=False)  # [K, R]

    def _probs(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, T, D]
        attention_mask: [B, T]
        returns: [M, B, H, T, S]
        """
        B, H, T, D = x.shape
        assert H == self.num_heads and D == self.head_dim

        proj = torch.einsum("bhtd,mds->mbhts", x, self.planes_T)
        proj = proj.view(self.M, B, H, T, self.L, self.K)

        logits = torch.einsum(
            "mbhtlk,kr->mbhtlr",
            proj.tanh().div(self.logit_temp),
            self.protos_T,
        )  # [M,B,H,T,L,R]

        probs = F.softmax(logits, dim=-1).reshape(self.M, B, H, T, self.S)

        token_mask = attention_mask[None, :, None, :, None].to(probs.dtype)
        probs = probs * token_mask
        return probs

    @torch.no_grad()
    def build_matrix(self, q: torch.Tensor, k: torch.Tensor, attention_mask: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
        """
        q, k: [B, H, T, D]
        attention_mask: [B, T]

        returns: [H, T, T] for B=1
        """
        assert q.size(0) == 1 and k.size(0) == 1, "This script expects batch size 1."

        probsQ = self._probs(q, attention_mask)   # [M,1,H,T,S]
        probsK = self._probs(k, attention_mask)   # [M,1,H,T,S]

        A = probsK.sum(dim=3)  # [M,1,H,S]
        Knorm = probsK / (A.unsqueeze(3) + eps)  # [M,1,H,T,S]

        W = torch.matmul(probsQ, Knorm.transpose(-1, -2))  # [M,1,H,T,T]
        W = W.mean(dim=0)[0]  # [H,T,T]

        valid = attention_mask[0].to(W.dtype)
        W = W * valid.view(1, -1, 1) * valid.view(1, 1, -1)

        row_sum = W.sum(dim=-1, keepdim=True).clamp_min(eps)
        W = W / row_sum
        return W


# ============================================================
# LSH helpers for highlighting same-bucket keys
# ============================================================
def _gray_code_order(num_bits: int, device):
    """
    Gray-code order so adjacent bucket IDs differ by one bit.
    """
    if num_bits == 1:
        return torch.tensor([0, 1], device=device, dtype=torch.long)

    def rec(n):
        if n == 1:
            return torch.tensor([0, 1], device=device, dtype=torch.long)
        a = rec(n - 1)
        return torch.cat([a, torch.flip(a, dims=[0]) + (1 << (n - 1))], dim=0)

    return rec(num_bits)


class AngularLSHGray(torch.nn.Module):
    """
    Hard angular LSH with Gray-code bucket ordering.
    Input:
        [..., T, D]
    Output:
        [..., T] integer bucket IDs
    """
    def __init__(self, num_bits: int, dim: int, device="cpu", seed: int = 0):
        super().__init__()
        self.num_bits = num_bits
        self.R = 1 << num_bits

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        proj_dir = torch.randn(dim, num_bits, generator=gen)
        self.register_buffer("proj_dir", proj_dir.to(device), persistent=False)
        self.register_buffer("perm", _gray_code_order(num_bits, device), persistent=False)

    def hash(self, mat: torch.Tensor):
        """
        mat: [..., T, D]
        returns: [..., T]
        """
        proj = torch.einsum("...td,dr->...tr", mat, self.proj_dir)
        bits = (proj > 0).to(torch.long)

        enc = (2 ** torch.arange(
            self.num_bits, device=mat.device, dtype=torch.long
        )).view(*([1] * (bits.ndim - 1)), self.num_bits)

        bin_ids = (bits * enc).sum(dim=-1)
        return self.perm[bin_ids]



@torch.no_grad()
def build_same_bucket_mask_union(
    q_heads: torch.Tensor,
    k_heads: torch.Tensor,
    num_bits: int,
    seed: int,
    exclude_self: bool = False,
):
    """
    q_heads, k_heads: [H, T, D]

    Returns
    -------
    same_bucket_mask: [T, T] bool
        same_bucket_mask[i, j] = True iff query row i and key j are in the
        same hard LSH bucket in AT LEAST ONE head.
    q_ids: [H, T]
    k_ids: [H, T]
    """
    assert q_heads.dim() == 3 and k_heads.dim() == 3
    H, T, D = q_heads.shape

    lsh = AngularLSHGray(
        num_bits=num_bits,
        dim=D,
        device=q_heads.device,
        seed=seed,
    )

    q_ids = lsh.hash(q_heads)   # [H, T]
    k_ids = lsh.hash(k_heads)   # [H, T]

    # [H, T, T]
    same = q_ids.unsqueeze(-1) == k_ids.unsqueeze(-2)

    # UNION across heads: if any head matches, mark it
    same_bucket_mask = same.any(dim=0)   # [T, T]

    if exclude_self:
        same_bucket_mask.fill_diagonal_(False)

    return same_bucket_mask, q_ids, k_ids


    #=====================================================
@torch.no_grad()
def topk_indices_from_row(row_tensor: torch.Tensor, k: int):
    """
    row_tensor: [T]
    returns: numpy array of top-k indices
    """
    k_eff = min(k, row_tensor.numel())
    idx = torch.topk(row_tensor, k=k_eff, largest=True).indices
    return idx.detach().cpu().numpy()    
# ============================================================
# Data sampling
# ============================================================
@torch.no_grad()
def pick_random_imdb_sample(tokenizer, cfg: CFG):
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    best = None
    best_len = -1

    for _ in range(cfg.max_sampling_trials):
        idx = random.randrange(len(ds))
        text = ds[idx]["text"]

        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_len,
            return_tensors="pt",
        )

        valid_len = int(enc["attention_mask"][0].sum().item())

        if valid_len > best_len:
            best = (idx, text, enc, valid_len)
            best_len = valid_len

        if not cfg.require_full_length:
            return idx, text, enc, valid_len

        if valid_len == cfg.max_len:
            return idx, text, enc, valid_len

    if cfg.require_full_length:
        if best is None:
            raise RuntimeError("Failed to sample an IMDb example.")
        if best_len < cfg.max_len:
            raise RuntimeError(
                f"Could not find a full-length example of {cfg.max_len} tokens "
                f"after {cfg.max_sampling_trials} trials. Best valid length: {best_len}."
            )

    return best


# ============================================================
# Main analysis
# ============================================================
@torch.no_grad()
def analyze_one_sample(cfg: CFG):
    set_seed(cfg.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModel.from_pretrained(cfg.model_name).to(device)
    model.eval()

    sample_idx, text, enc, valid_len = pick_random_imdb_sample(tokenizer, cfg)

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads

    race_builders = [
        RaceMatrixBuilder(
            num_heads=num_heads,
            head_dim=head_dim,
            K=cfg.race_K,
            L=cfg.race_L,
            M=cfg.race_M,
            logit_temp=cfg.race_logit_temp,
            device=device,
            seed=cfg.seed + 1000 + layer_idx,
        ).to(device)
        for layer_idx in range(num_layers)
    ]

    exact_attn_per_layer = []
    race_attn_per_layer = []
    same_bucket_masks_per_layer = []

    # ------------------------------------------------------------
    # Build matrices and same-bucket masks layer-by-layer
    # ------------------------------------------------------------
    build_pbar = tqdm(range(num_layers), desc="Building exact/RACE/LSH", leave=True)
    for layer_idx in build_pbar:
        layer_mod = model.encoder.layer[layer_idx].attention.self
        x = hidden_states[layer_idx]  # [1,T,D_model]

        q_lin = layer_mod.query(x)
        k_lin = layer_mod.key(x)

        q = transpose_for_scores(q_lin, num_heads, head_dim)  # [1,H,T,D]
        k = transpose_for_scores(k_lin, num_heads, head_dim)  # [1,H,T,D]

        exact_mat = build_exact_attention_matrix_from_qk(q, k, attention_mask)  # [H,T,T]
        race_mat = race_builders[layer_idx].build_matrix(q, k, attention_mask, eps=cfg.race_eps)  # [H,T,T]

        same_bucket_mask, _, _ = build_same_bucket_mask_union(
                q_heads=q[0],
                k_heads=k[0],
                num_bits=cfg.lsh_num_bits,
                seed=cfg.seed + 5000 + layer_idx,
                exclude_self=cfg.exclude_self_from_same_bucket,
            )

        exact_attn_per_layer.append(exact_mat)
        race_attn_per_layer.append(race_mat)
        same_bucket_masks_per_layer.append(same_bucket_mask)

    # ------------------------------------------------------------
    # W&B init
    # ------------------------------------------------------------
    model_short = cfg.model_name.split("/")[-1]
    run_name = f"imdb_{model_short}_sample{sample_idx}_rows_headavg_lsh"

    wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        mode=cfg.wandb_mode,
        config={
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
            "dataset_split": cfg.dataset_split,
            "sample_index": sample_idx,
            "max_len": cfg.max_len,
            "valid_len": valid_len,
            "race_K": cfg.race_K,
            "race_L": cfg.race_L,
            "race_M": cfg.race_M,
            "race_logit_temp": cfg.race_logit_temp,
            "lsh_num_bits": cfg.lsh_num_bits,
            "aggregate_heads": "mean",
            #"same_bucket_vote_threshold": cfg.same_bucket_vote_threshold,
        },
    )

    # sample info table (string-safe)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:valid_len].tolist())

    sample_table = wandb.Table(columns=["name", "value_str"])
    sample_table.add_data("sample_index", str(sample_idx))
    sample_table.add_data("valid_length", str(valid_len))
    sample_table.add_data("text_preview", text[:3000])
    sample_table.add_data("tokens", " ".join(tokens))
    wandb.log({"sample_info": sample_table})

    # locate SEP tokens for optional vertical markers
    sep_positions = [i for i, tok in enumerate(tokens) if tok == "[SEP]"]

    # ------------------------------------------------------------
    # Per-layer logging
    # ------------------------------------------------------------
    layer_pbar = tqdm(range(num_layers), desc="Logging row plots", leave=True)
    for layer_idx in layer_pbar:
        exact_mat = maybe_average_heads(exact_attn_per_layer[layer_idx])  # [T,T]
        race_mat  = maybe_average_heads(race_attn_per_layer[layer_idx])   # [T,T]
        same_bucket_mat = same_bucket_masks_per_layer[layer_idx]          # [T,T]

        exact_mat = exact_mat[:valid_len, :valid_len]
        race_mat  = race_mat[:valid_len, :valid_len]
        same_bucket_mat = same_bucket_mat[:valid_len, :valid_len]

        batch_log = {}
        row_pbar = tqdm(range(valid_len), desc=f"Layer {layer_idx} rows", leave=False)
        for r in row_pbar:
            qtok = tokens[r] if r < len(tokens) else str(r)

            exact_row_t = exact_mat[r].detach().cpu().float()   # [T]
            race_row_t  = race_mat[r].detach().cpu().float()    # [T]

            exact_row = exact_row_t.numpy()
            race_row  = race_row_t.numpy()

            same_idx = torch.nonzero(
                same_bucket_mat[r], as_tuple=False
            ).squeeze(-1).cpu().numpy()

            topk_exact_idx = topk_indices_from_row(
                exact_row_t, cfg.topk_keys_to_mark
            )
            topk_race_idx = topk_indices_from_row(
                race_row_t, cfg.topk_keys_to_mark
            )

            fig, ax = plt.subplots(figsize=cfg.row_plot_figsize)

            # main curves
            ax.plot(
                exact_row,
                label="exact_attention",
                linewidth=2,
                color="tab:blue",
            )
            ax.plot(
                race_row,
                label="race_attention",
                linewidth=2,
                color="tab:orange",
            )

            # --------------------------------------------------
            # same-bucket keys (union over heads)
            # --------------------------------------------------
            if same_idx.size > 0:
                ax.scatter(
                    same_idx,
                    exact_row[same_idx],
                    color="red",
                    s=22,
                    alpha=0.90,
                    zorder=5,
                    label="same-bucket keys (exact)",
                )
                ax.scatter(
                    same_idx,
                    race_row[same_idx],
                    color="darkred",
                    marker="x",
                    s=22,
                    alpha=0.90,
                    zorder=6,
                    label="same-bucket keys (race)",
                )

            # --------------------------------------------------
            # top-k exact keys
            # --------------------------------------------------
            if topk_exact_idx.size > 0:
                ax.scatter(
                    topk_exact_idx,
                    exact_row[topk_exact_idx],
                    facecolors="none",
                    edgecolors="green",
                    marker="s",
                    s=34,
                    linewidths=1.2,
                    alpha=0.95,
                    zorder=7,
                    label=f"top-{cfg.topk_keys_to_mark} exact keys",
                )

            # --------------------------------------------------
            # top-k race keys
            # --------------------------------------------------
            if topk_race_idx.size > 0:
                ax.scatter(
                    topk_race_idx,
                    race_row[topk_race_idx],
                    color="purple",
                    marker="x",
                    s=26,
                    alpha=0.95,
                    zorder=8,
                    label=f"top-{cfg.topk_keys_to_mark} race keys",
                )
            set_same = set(same_idx.tolist()) if same_idx.size > 0 else set()
            set_exact = set(topk_exact_idx.tolist()) if topk_exact_idx.size > 0 else set()
            set_race = set(topk_race_idx.tolist()) if topk_race_idx.size > 0 else set()

            overlap_exact_race = len(set_exact & set_race)
            overlap_same_exact = len(set_same & set_exact)
            overlap_same_race = len(set_same & set_race)                
            # mark special positions
            ax.axvline(0, linestyle="--", linewidth=1, alpha=0.5, color="gray")  # CLS
            for sp in sep_positions:
                ax.axvline(sp, linestyle=":", linewidth=1, alpha=0.4, color="gray")

            ax.set_title(
                f"Layer {layer_idx} | Row {r} | Token: {qtok} | "
                f"Head-averaged | red=same-bucket, green=top-{cfg.topk_keys_to_mark} exact, "
                f"purple=top-{cfg.topk_keys_to_mark} race"
                f"\nOverlap: exact∩race={overlap_exact_race}, "
                f"same∩exact={overlap_same_exact}, same∩race={overlap_same_race}"
            )
            ax.set_xlabel("Key index")
            ax.set_ylabel("Normalized attention value")
            ax.set_xlim(0, valid_len - 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            batch_log[f"layer_{layer_idx}/row_{r:03d}"] = wandb.Image(fig)
            plt.close(fig)

            if len(batch_log) >= cfg.rows_per_log_batch:
                wandb.log(batch_log)
                batch_log = {}

        if batch_log:
            wandb.log(batch_log)

    wandb.finish()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    cfg = CFG()
    analyze_one_sample(cfg)
