# RACE Attention

[![ICLR 2026](https://img.shields.io/badge/ICLR%202026-Accepted-blue.svg)](https://openreview.net/forum?id=RR8Lh8RHgA)
[![arXiv](https://img.shields.io/badge/arXiv-2510.04008-b31b1b.svg)](https://arxiv.org/abs/2510.04008)

**RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training**  
✅ Accepted at **ICLR 2026**

📄 [Paper (OpenReview)](https://openreview.net/forum?id=RR8Lh8RHgA) | 
📄 [arXiv](https://arxiv.org/abs/2510.04008)

---

## TL;DR

**RACE Attention** is a strictly linear-time attention mechanism for scalable long-sequence training.

- Replaces quadratic Softmax attention with randomized similarity aggregation
- Enables training at substantially longer sequence lengths
- Maintains competitive accuracy with modern attention baselines
- Backed by theoretical guarantees and scalable kernel implementations

👉 Start with the notebooks in `notebooks/` to run examples in minutes.

---

## Overview

RACE Attention introduces a randomized formulation of attention that replaces dense similarity computation with hash-based aggregation. By leveraging angular similarity and randomized collision statistics, the method achieves strictly linear computational complexity while preserving the inductive behavior of Softmax attention.

### Key Features

- ⚡ **Strictly linear-time attention in sequence length**
- 🔁 Drop-in replacement for Softmax Attention
- 🎯 Competitive accuracy with FlashAttention and linear attention baselines
- 📈 **Demonstrated scalability:**  
  – Single attention layer evaluated in a forward–backward pass  
  – Processes up to **12M tokens** on an NVIDIA GH200 (96GB) GPU  
  – Processes up to **75M tokens** on an Intel Xeon® Gold 5220R CPU  
  – Exceeds practical limits of current state-of-the-art attention implementations
- 🧠 Theoretically grounded randomized attention mechanism
---

<img width="4404" height="1808" alt="Comparing_Softmax_and_RACE_Attention" src="https://github.com/user-attachments/assets/e39986a3-c95b-4075-b53c-524a5ab79c56" />


## Quick Start

Example notebooks demonstrating RACE Attention are located in:

```
notebooks/
```

These notebooks provide end-to-end training examples across multiple domains using smaller datasets and moderate sequence lengths, intended as illustrative examples rather than full-scale experimental runs.

| Notebook | Task | Dataset | Sequence Length |
|---|---|---|---|
| `ClassificationTask.ipynb` | Text classification | AG News | 512 |
| `LanguageModelling.ipynb` | Autoregressive language modeling | WikiText-2 | 128 |
| `MaskedLanguageModelling.ipynb` | Masked language modeling | TinyStories (BERT-style) | 512 |
| `VisionTask.ipynb` | Image classification (ViT) | MNIST | 784 |

The notebooks demonstrate how to:

- Train models using RACE Attention as a drop-in replacement
- Compare against Softmax and alternative attention mechanisms
- Reproduce representative experimental settings from the paper
- Analyze qualitative and quantitative behavior of attention mechanisms

---

## Installation

Clone the repository:
```bash
git clone https://github.com/sahiljoshi515/RACE_Attention.git
cd RACE_Attention
```

Install dependencies:

```bash
pip install -r requirements.txt
```

We recommend using a virtual environment:

```bash
python -m venv race_env
source race_env/bin/activate   # Linux / Mac
pip install -r requirements.txt
```

---

## Repository Layout

This repository is a **research artifact accompanying the paper**, containing experiment scripts, attention kernels, and benchmarking code.

```
RACE_Attention/
├── notebooks/   # Quick-start examples (recommended starting point)
├── misc/        # Training scripts used in paper experiments
├── kernels/     # CPU/CUDA implementations of RACE Attention
└── scaling/     # Long-context runtime benchmarks
```
New users are encouraged to begin with the notebooks before exploring task scripts or kernel implementations.

---

## Citation

If you use RACE Attention in your research, please cite:

```bibtex
@inproceedings{joshi2026raceattention,
  title     = {RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training},
  author    = {Joshi, Sahil and Chowdhury, Agniva and Kanakamedala, Amar and Singh, Ekam and Tu, Evan and Shrivastava, Anshumali},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=RR8Lh8RHgA}
}
```

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

---


## Contact

For questions, bug reports, or collaboration inquiries, please open a GitHub issue or contact [Sahil Joshi](https://www.sahiljoshi.org/).
