# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RACE Attention is an ICLR 2026 research project implementing a strictly linear-time attention mechanism that replaces quadratic Softmax attention with randomized hash-based aggregation. Paper: https://arxiv.org/abs/2510.04008

## Setup & Dependencies

```bash
pip install -r requirements.txt
# Optional: flash-attn for baseline comparisons
pip install flash-attn
```

## Building Kernels

**CPU kernel (OpenMP):**
```bash
cd kernels/cpu
python setup.py          # builds and runs benchmark
python race_ext.py       # Python wrapper via torch.utils.cpp_extension
```

**GPU kernel (CUDA):** compiled automatically via `torch.utils.cpp_extension.load()` when imported.

**Triton kernel:** no build step; pure Python in [misc/race_kernel.py](misc/race_kernel.py).

## Running Experiments

All training scripts in `misc/` are self-contained and run directly:
```bash
python misc/classification.py   # IMDB text classification
python misc/vit.py              # Vision Transformer (FashionMNIST, Oxford-IIIT-Pet)
python misc/arxiv_64K.py        # Large-scale arXiv abstracts (64K tokens)
python misc/lm.py               # Causal language modeling (WikiText)
python misc/mlm.py              # Masked language modeling (BERT-style)
```

Quick-start notebooks in [notebooks/](notebooks/) are the recommended entry point for each task type.

Benchmarking across attention types:
```bash
python scaling/benchmark_time.py
```

## Core Architecture

### RACE Attention (`misc/race.py`)

Three layered classes:

1. **`BatchedACE`** — the core algorithm. Takes `[M, B, T, H, d_k]` tensors (M ensembles). Projects queries/keys onto L×K random hyperplanes, soft-hashes into R=2^K buckets via `tanh(proj/scale) @ prototypes`, computes causal prefix sums `cumsum(probs*V) / cumsum(probs)` for O(T) attention.

2. **`RACEAttention`** — multi-head wrapper. Splits into heads, expands to M ensembles, averages ensemble outputs. Drop-in replacement for Softmax attention.

3. **`RACEBlock`** — standard Transformer layer (attention + LayerNorm + Dropout + FFN).

**Key hyperparameters:** `K` (log₂ of bucket count), `L` (hash tables), `M` (ensembles, usually 1–2).

### Attention Variants

All training scripts support swappable `attention_type` argument:

| Type | Description |
|------|-------------|
| `race` | Core RACE linear attention |
| `softmax` | Standard dense (baseline) |
| `angular_lsh` | Hash-based LSH attention |
| `hyper_lsh` | Hybrid local+global sparse |
| `hyper_race` | RACE with sparse local fallback |
| `hyper_race_mexact` | RACE with learnable mixing toward exact attention |
| `linear` | Sequential recurrence baseline |

### Kernel Implementations

- **CPU** ([kernels/cpu/](kernels/cpu/)): OpenMP prefix-sum kernels, platform-aware (macOS libomp / Linux OpenMP). Handles `NTSD` and flat tensor layouts. Gradients included.
- **GPU** ([kernels/gpu/](kernels/gpu/)): CUDA forward/backward kernels. fp16 state compression, atomic ops. Tested on NVIDIA GH200 (up to 12M tokens).
- **Triton** ([misc/race_kernel.py](misc/race_kernel.py)): Chunk-wise forward pass with inter-chunk state and intra-chunk causality.

## Experiment Tracking

Scripts use [Weights & Biases](https://wandb.ai) for logging. Set `WANDB_PROJECT` or configure inside scripts. Output `.txt` log files in the repo root capture trial results.

## Configuration Pattern

Training scripts use centralized config dicts (e.g., `VISION_CONFIG` in [misc/vit.py](misc/vit.py)) rather than CLI argparse. Edit the config dict at the top of each script to change hyperparameters.
