# RACE Attention

We use introduce [RACE](https://dl.acm.org/doi/abs/10.1145/3366423.3380244) Attention:

- A linear time and memory attention implementation that approximates the exponentiated angular kernel.
- Drop-in for Softmax Attention during pre-training for diverse tasks.
- Achieves competetive accuracy as other Linear Attenion baselines and quadratic FlashAttention.
- Scales to 75 million tokens on Intel Xeon® Gold 5220R CPU and to 12 million tokens on NVVIDA GH200 (96GB) GPU when using a single layer of multihead attention.
- For more information read the [Paper](https://arxiv.org/pdf/2510.04008).

# User-Guide

- Use the custom Python notebooks to try out our RACE Attention algorithm on different tasks.

# Softmax Attention vs. RACE Attention
<img width="5130" height="2305" alt="Comparing Softmax and RACE Attention" src="https://github.com/user-attachments/assets/f4b3a169-0231-45b8-8d7f-e34d51aaf2ee" />


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
