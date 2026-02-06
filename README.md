# RACE Attention

We use introduce [RACE](https://dl.acm.org/doi/abs/10.1145/3366423.3380244) Attention:

- A linear time and memory attention implementation that approximates the exponentiated angular kernel.
- Drop-in for Softmax Attention during pre-training for diverse tasks.
- Achieves competetive accuracy as other Linear Attenion baselines and quadratic FlashAttention.
- Scales to 75 million tokens on Intel Xeon® Gold 5220R CPU and to 12 million tokens on NVVIDA GH200 (96GB) GPU when using 1 layer of multihead attention.
- For more information read the [Pre-print](https://arxiv.org/pdf/2510.04008).

# User-Guide

- Use the custom Python notebooks to try out our RACE Attention algorithm on different tasks.

# Softmax Attention vs. RACE Attention
<img width="4685" height="2165" alt="Copy of Comparing Softmax and RACE Attention" src="https://github.com/user-attachments/assets/162cbf2a-5be7-4345-8cec-9ac49ba0e0ab" />

# Complete Workflow of RACE Attention Algorithm
<img width="4525" height="2955" alt="Detailed Flowchart" src="https://github.com/user-attachments/assets/a139f39e-b93c-4398-8dbc-37c1cef92c9a" />

# Intuitive schematic of RACE Attention
<img width="1988" height="1616" alt="Query attending to keys" src="https://github.com/user-attachments/assets/f91fe505-29ac-4038-b526-30691e8c6ff3" />

# Citation

If you use the code above, please cite:

```
@article{
    joshi2025raceattention, title = {Replacing Softmax Similarity with a Sharpened Angular Similarity: Theory and Practice of Scaling To Billion-Context Attention},
    author = {Sahil Joshi and Agniva Chowdhury and Amar Kanakamedala and Ekam Singh and Evan Tu and Anshumali Shrivastava},
    journal = {arXiv preprint arXiv:2510.04008},
    year = {2025},
    url = {https://arxiv.org/abs/2510.04008}
  }
```
