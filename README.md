# RACE_Attention

We use introduce [RACE](https://dl.acm.org/doi/abs/10.1145/3366423.3380244) Attention:

- A linear time and memory attention implementation that approximates the exponentiated angular kernel.
- Drop-in for Softmax Attention during pre-training.
- Achieves competetive accuracy as other Linear Attenion baselines and quadratic Softmax Attention.
- Scales to 75 million tokens on Intel Xeon® Gold 5220R CPU and to 12 million tokens on NVVIDA GH200 GPU. 


# User-Guide

- classification.py contains the experiment code for classification tasks. 
- vit.py contains the experiment code for image classification tasks.
- lm.py contains the experiment code for Language Modeling tasks.
- mlm.py contains the experiment code for Masked Language Modeling tasks.
- Our custom OpenMP kernel can be found in race_pref.cpp. If you wish to run on CPU, this will be way faster than torch.cumsum.
- You can simply replace any Attention method with our RACEAttention class and test with it.

# Softmax Attention vs. RACE Attention
<img width="4685" height="2165" alt="Copy of Comparing Softmax and RACE Attention" src="https://github.com/user-attachments/assets/162cbf2a-5be7-4345-8cec-9ac49ba0e0ab" />

# Complete Workflow of RACE Attention Algorithm
<img width="4525" height="2955" alt="Detailed Flowchart" src="https://github.com/user-attachments/assets/a139f39e-b93c-4398-8dbc-37c1cef92c9a" />

# Intuitive schematic of RACE Attention
<img width="1988" height="1616" alt="Query attending to keys" src="https://github.com/user-attachments/assets/f91fe505-29ac-4038-b526-30691e8c6ff3" />
