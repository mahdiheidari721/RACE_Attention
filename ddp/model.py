from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
import numpy
import torch
import time
import math
from datasets import DatasetDict, concatenate_datasets
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
import numpy
import torch
import time
import math
from datasets import DatasetDict, concatenate_datasets
from tqdm.auto import tqdm



import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
# import tiktoken
import itertools
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler import schedule
import argparse



import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
# import tiktoken
import itertools
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler import schedule
import argparse

# ==================================================
# 7) Attention blocks (pad‑mask aware)
# ==================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        self.h, self.dk = h, d//h
        self.q = nn.Linear(d,d, bias=qkv_bias)
        self.k = nn.Linear(d,d, bias=qkv_bias)
        self.v = nn.Linear(d,d, bias=qkv_bias)
        self.o = nn.Linear(d,d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        B,T,_ = x.shape

        Q = self.q(x).contiguous().view(B,T,self.h,self.dk)
        K = self.k(x).contiguous().view(B,T,self.h,self.dk)
        V = self.v(x).contiguous().view(B,T,self.h,self.dk)
        
        if mask is not None:
            m = mask.unsqueeze(-1).unsqueeze(-1)
            Q, K, V = Q*m, K*m, V*m
        
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.dk)

        W = torch.softmax(scores, -1)
        W = self.drop(W)
        out = (W @ V).transpose(1,2).reshape(B,T,self.h*self.dk)
        # mask = mask.unsqueeze(1).unsqueeze(2)
        # out = F.scaled_dot_product_attention(Q, K, V, attn_mask = pad_mask, dropout_p = self.drop if self.training else 0.0, is_causal=False, scale = 1/math.sqrt(self.dk))
        return self.o(out)
        
class TransformerBlock(nn.Module):
    """Standard softmax‐attention Transformer block, pad‐mask aware."""
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(d=cfg["emb_dim"], h=cfg["n_heads"], drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, pad_mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, pad_mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop    = nn.Dropout(cfg["drop_rate"]) 

        kind = cfg["kind"]

        # Build the blocks list correctly
        self.blocks = nn.ModuleList()
        # # elif kind == "angular":
        # #     A_Block = AngularBlock(cfg) )
        # elif kind == "race":
        #     R_block = RACEBlock(cfg, device=DEVICE)
        # else:
        #     raise ValueError(kind)

        for _ in range(cfg["n_layers"]):
            if kind == "softmax":
                self.blocks.append(TransformerBlock(cfg))
            # elif kind == "angular":
            #     self.blocks.append( AngularBlock(cfg) )
            # elif kind == "race":
            #     self.blocks.append(RACEBlock(cfg, device=cfg["device"]))
            # elif kind == "linear":
            #     self.blocks.append(LinearBlock(cfg, device = cfg["device"]))
            # elif kind == "angular":
            #     self.blocks.append(AngularBlock(cfg))
            # elif kind == "linformer":
            #     self.blocks.append(LinformerBlock(cfg))
            # elif kind == "performer":
            #     self.blocks.append(PerformerBlock(cfg))
            # else:
            #     raise ValueError(kind)

        self.norm = nn.LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["output_dim"])

    def forward(self, x, mask):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        # x: B x T -> B x T x emb_dim 
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.norm(h)
        # WHy just the 0th element?
        return self.head(h[:,0])