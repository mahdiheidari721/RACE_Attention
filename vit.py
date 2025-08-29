import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch 
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as dataloader
import torch.nn as nn
import itertools
import math
import time
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
torch.set_float32_matmul_precision('high')

VISION_CONFIG = {
    "batch_size": 16,
    "img_size": 32,
    "patch_size": 1,
    "num_channels": 3,
    "num_patches": 1024,
    "num_heads": 4,
    "embed_dim": 384,
    "mlp_dim": 32,
    "transformer_units": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "K": 3,
    "L": 3,
    "M": 1
}


def get_data(cfg):
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
    # normalize = transforms.Normalize(
    #     mean=(0.2860,),   # FashionMNIST mean
    #     std=(0.3530,)     # FashionMNIST std
    # )
    train_tfms = transforms.Compose([
        transforms.Resize(cfg["img_size"], antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    test_tfms = transforms.Compose([
        transforms.Resize(cfg["img_size"], antialias=True),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_tfms)
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_tfms)

    train_data = dataloader.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_data = dataloader.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    return train_data, val_data


class PatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = nn.Conv2d(cfg["num_channels"], cfg["embed_dim"], kernel_size=cfg["patch_size"], stride=cfg["patch_size"])

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.dropout_p = dropout

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj= nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape   
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=False, dropout_p=self.dropout_p)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)
    
class TransformerArchitecture(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(cfg["embed_dim"])
        self.self_attention = MultiHeadAttention(d_in=cfg["embed_dim"], d_out=cfg["embed_dim"], dropout=cfg["drop_rate"], num_heads=cfg["num_heads"], qkv_bias=cfg["qkv_bias"])
        self.layer_norm_2 = nn.LayerNorm(cfg["embed_dim"])
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
        )

    def forward(self, x):
        residual_1 = x
        attention_output = self.self_attention(self.layer_norm_1(x))
        x = attention_output + residual_1
        residual_2 = x
        mlp_output = self.multi_layer_perceptron(self.layer_norm_2(x))
        x = mlp_output + residual_2
        return x

class BatchedACE(nn.Module):
    """
    Non-causal BatchedACE with optional shared planes.
    Inputs:
      Khf, Vhf, Qhf : [M, B, T, H, d_k]
    """
    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = False):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            # Shared planes [L, K, d_k] --> [d_k, (L*K)]
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer('planes_T', planes.view(L * K, d_k).T)   # [d_k, L*K]
        else:
            # Independent planes [M, L, K, d_k] --> [M, d_k, (L*K)]
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)           # [M, d_k, L*K]
            self.register_buffer('planes_T', planes)

        # Prototypes (corners of {-1,+1}^K): [K, R]
        corners = torch.tensor(list(itertools.product([-1., +1.], repeat=K)), device=device)
        self.register_buffer('protos_T', corners.T)                        # [K, R]

        # learnable temperature
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
        # Khf, Vhf, Qhf: [M, B, T, H, d_k]
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 10.0) # uncomment when you make temp learnable

        if self.share_planes:
            # Collapse M·B·H → N
            N = M * B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)  # [N,T,dk]
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            # Projections to L*K
            projK = Kh2 @ self.planes_T                                     # [N,T,L*K]
            projQ = Qh2 @ self.planes_T                                     # [N,T,L*K]
        else:
            # Keep ensembles separate; collapse only B·H
            BH = B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)  # [M,BH,T,dk]
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            # One GEMM per ensemble
            projK = torch.einsum('mbtd,mds->mbts', Kh2, self.planes_T)        # [M,BH,T,L*K]
            projQ = torch.einsum('mbtd,mds->mbts', Qh2, self.planes_T)
            # Merge M,BH → N
            projK = projK.contiguous().view(M * BH, T, self.L * self.K)       # [N,T,L*K]
            projQ = projQ.contiguous().view(M * BH, T, self.L * self.K)
            V2    = V2.view(M * BH, T, dk)
            N     = M * BH

        # Reshape to [N,T,L,K] and soft-hash → probs over R buckets
        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)

        logitsK = (projK.tanh().div(scale) @ self.protos_T)                   # [N,T,L,R]
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)                   # [N,T,L,R]
        probsK  = F.softmax(logitsK, dim=-1)                                   # [N,T,L,R]
        probsQ  = F.softmax(logitsQ, dim=-1)                                   # [N,T,L,R]

        # -------- Non-causal bucket summaries over the full sequence --------
        # Collapse buckets L,R → S
        probsK_S = probsK.contiguous().view(N, T, S)                           # [N,T,S]
        probsQ_S = probsQ.contiguous().view(N, T, S)                           # [N,T,S]

        # Weighted sums across time:
        #   b_sum = probsK^T @ V   → [N,S,dk]
        b_sum = probsK_S.transpose(1, 2).bmm(V2)                               # [N,S,dk]
        #   A = sum_t probsK_t     → [N,S]
        A = probsK_S.sum(dim=1)                                                # [N,S]
        #   E = b_sum / (A + eps)  → [N,S,dk]
        E = b_sum / (A.unsqueeze(-1) + eps)                                    # [N,S,dk]

        # Query lookup per time (no prefix): [N,T,S] @ [N,S,dk] → [N,T,dk]
        out2 = probsQ_S.bmm(E)                                                 # [N,T,dk]

        # Unflatten back to [M,B,T,H,dk]
        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4)                 # [M,B,T,H,dk]
        return out

class RACEAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout,
                 num_heads, L, K, N_M, qkv_bias=False, device='cpu'):
        super().__init__()
        assert d_in % num_heads == 0
        self.H   = num_heads
        self.d_k = d_in // num_heads
        self.M   = N_M

        self.q_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.k_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.v_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.out    = nn.Linear(d_in, d_out)
        self.drop   = nn.Dropout(dropout)
        self.ace = BatchedACE(self.d_k, K, L, N_M, device=device)

    def forward(self, x):
        B, T, _ = x.shape
        H, d_k, M = self.H, self.d_k, self.M

        # 1) project & reshape for ACE
        Q = self.q_proj(x).view(B, T, H, d_k)
        K = self.k_proj(x).view(B, T, H, d_k)
        V = self.v_proj(x).view(B, T, H, d_k)

        # shape --> [M, B, T, H, d_k] by explicit unsqueeze
        def pack(Z):
            Zm = Z.unsqueeze(0).expand(M, -1, -1, -1, -1)
            return Zm

        Khf = pack(K)
        Vhf = pack(V)
        Qhf = pack(Q)

        # 2) run ACE
        out_hm = self.ace(Khf, Vhf, Qhf)  # [M,B,T,H,d_k]

        # 3) average ensembles & merge heads
        out = out_hm.mean(dim=0)          # [B,T,H,d_k]
        out = out.permute(0,2,1,3).reshape(B, T, H * d_k)

        # 4) final proj + dropout
        return self.drop(self.out(out))
    
class RACEBlock(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.att   = RACEAttention(
            d_in=cfg["embed_dim"], d_out=cfg["embed_dim"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"], qkv_bias=cfg["qkv_bias"],
            L=cfg["L"], K=cfg["K"], N_M=cfg["M"], device=device
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class LinearAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.eps = eps

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def kernel(self, x):
        # φ(x): positive-valued kernel feature map
        return F.elu(x) + 1  # [B, H, T, D]

    def forward(self, x):
        B, T, _ = x.size()

        # Linear projections
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, T, D]
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        # Apply kernel φ
        Q = self.kernel(Q)  # [B, H, T, D]
        K = self.kernel(K)  # [B, H, T, D]

        # Compute KV^T: [B, H, D, D]
        KV = torch.einsum('bhtd,bhte->bhde', K, V)  # [B, H, D, D]

        # Compute normalization factor: Z = Q * sum(K)
        K_sum = K.sum(dim=2)  # [B, H, D]
        Z = torch.einsum('bhtd,bhd->bht', Q, K_sum) + self.eps  # [B, H, T]

        # Compute output: Q @ (KV)
        context = torch.einsum('bhtd,bhde->bhte', Q, KV)  # [B, H, T, D]
        out = context / Z.unsqueeze(-1)  # [B, H, T, D]

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, H*D]
        return self.out_proj(out)

class LinearBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = LinearAttention(
            d_in=cfg["embed_dim"], d_out=cfg["embed_dim"],
            dropout=cfg["drop_rate"], num_heads=cfg["num_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"],4*cfg["embed_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["embed_dim"],cfg["embed_dim"])
                        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x); x = self.drop(x) + h
        return x
    
class AngularAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        self.h, self.dk = h, d//h
        self.q = nn.Linear(d,d, bias=qkv_bias)
        self.k = nn.Linear(d,d, bias=qkv_bias)
        self.v = nn.Linear(d,d, bias=qkv_bias)
        self.o = nn.Linear(d,d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B,T,_ = x.shape
        Q = F.normalize(self.q(x).view(B,T,self.h,self.dk).transpose(1,2), dim=-1)
        K = F.normalize(self.k(x).view(B,T,self.h,self.dk).transpose(1,2), dim=-1)
        V = self.v(x).view(B,T,self.h,self.dk).transpose(1,2)
        sim = (Q @ K.transpose(-2,-1)).clamp(-0.999,0.999)
        scores = 1 - torch.acos(sim)/math.pi
        W = scores.clamp(min=1e-6).pow(8)
        W = W / (W.sum(-1,keepdim=True)+1e-6)
        W = self.drop(W)
        out = (W @ V).transpose(1,2).contiguous().view(B,T,self.h*self.dk)
        return self.o(out)
    
class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = AngularAttention(d=cfg["embed_dim"], h=cfg["num_heads"], drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])

        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], 4*cfg["embed_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["embed_dim"], cfg["embed_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
    
class LinformerAttention(nn.Module):
    """
    Linformer-style attention: project K,V along sequence length T -> k (k << T),
    then do standard scaled dot-product attention with softmax over k.

    Shapes:
      x: (B, T, d_in)
      returns: (B, T, d_out)
    """
    def __init__(
        self,
        d: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        k_proj_dim: int,      # low-rank sequence dim
        max_seq_len: int    # allocate E up to this T, slice at runtime
    ):
        super().__init__()
        assert d % num_heads == 0, "d_out must be divisible by num_heads"
        self.h = num_heads
        self.dk = d // num_heads
        self.k_proj_dim = k_proj_dim
        self.max_seq_len = max_seq_len

        # token projections
        self.W_query = nn.Linear(d,  d, bias=qkv_bias)
        self.W_key   = nn.Linear(d,  d, bias=qkv_bias)
        self.W_value = nn.Linear(d,  d, bias=qkv_bias)

        # learnable sequence projections E_k, E_v: [T_max, k]
        self.E_k = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        self.E_v = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))

        nn.init.xavier_uniform_(self.E_k)
        nn.init.xavier_uniform_(self.E_v)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"
        h, dk, k = self.h, self.dk, self.k_proj_dim

        # Linear projections -> (B, h, T, dk)
        Q = self.W_query(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.W_key(  x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        # Sequence down-projection (T -> k) using E_k/E_v sliced to current T
        Ek = self.E_k[:T]  # (T, k)
        Ev = self.E_v[:T]  # (T, k)

        # K_proj, V_proj: (B, h, k, dk)
        # Contract over sequence axis
        K_proj = torch.einsum("bhtd,tk->bhkd", K, Ek)
        V_proj = torch.einsum("bhtd,tk->bhkd", V, Ev)

        # Scaled dot-product attention over compressed length k
        # scores: (B, h, T, k)
        scale = 1.0 / math.sqrt(dk)
        scores = torch.einsum("bhtd,bhkd->bhtk", Q, K_proj) * scale
        attn = F.softmax(scores, dim=-1)

        # Context: (B, h, T, dk)
        ctx = torch.einsum("bhtk,bhkd->bhtd", attn, V_proj)

        # Merge heads -> (B, T, d_out)
        out = ctx.transpose(1, 2).contiguous().view(B, T, h * dk)
        return self.out_proj(self.dropout(out))


class LinformerBlock(nn.Module):
    """
    Drop-in analogue of your LinearBlock but using LinformerAttention.
    Non-causal, no kernel; just K,V low-rank sequence projection.
    """
    def __init__(self, cfg):
        super().__init__()
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        k_proj_dim = 128

        self.att  = LinformerAttention(
            d=cfg["embed_dim"], dropout=drop, num_heads=cfg["num_heads"], qkv_bias=qkv_bias,
            k_proj_dim=k_proj_dim, max_seq_len=cfg["num_patches"] + 1
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
                        nn.GELU(),
                        nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"]),
                     )
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        # Attn sublayer
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        # FFN sublayer
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class VisionTransformer(nn.Module):
    def __init__(self, cfg, attn_type, device='cpu'):
        super().__init__()
        self.patch_embedding = PatchEmbedding(cfg)

        G = cfg["img_size"] // cfg["patch_size"]
        assert G * cfg["patch_size"] == cfg["img_size"], "img_size must be divisible by patch_size"
        num_patches = G * G
        d = cfg["embed_dim"]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # pick block
        if attn_type == "softmax":
            AttnBlock = TransformerArchitecture
        elif attn_type == "race":
            AttnBlock = lambda c: RACEBlock(c, device)
        elif attn_type == "angular":
            AttnBlock = AngularBlock
        elif attn_type == "linear":
            AttnBlock = LinearBlock
        elif attn_type == "linformer":
            AttnBlock = LinformerBlock
        else:
            raise ValueError("Unsupported attention type")

        self.transformer_layers = nn.Sequential(
            *[AttnBlock(cfg) for _ in range(cfg["transformer_units"])]
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 10))

    def forward(self, x):
        x = self.patch_embedding(x)                 # [B, N, d], N=G*G
        B, N, d = x.shape
        cls = self.cls_token.expand(B, -1, -1)      # [B,1,d]
        x = torch.cat([cls, x], dim=1)              # [B, N+1, d]
        x = x + self.pos_embed[:, :x.size(1), :]    # safe slice
        x = self.transformer_layers(x)
        x = x[:, 0]                                 # CLS
        return self.mlp_head(x)

    
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, cfg):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_times, val_times = [], []
    K, L, M = cfg.get("K", None), cfg.get("L", None), cfg.get("M", None)
    out_path = f"trial_K{K}_L{L}_M{M}_1024.txt"

    # helper logger
    def _log(fp, msg):
        print(msg)
        fp.write(msg + "\n")
        fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Epochs: {num_epochs}")
        _log(f, "-" * 72)
        
        for epoch in range(1, num_epochs + 1):
            # === TRAIN ===
            t0 = time.time()
            model.train()
            total_loss = 0.0
            correct_epoch = 0
            total_epoch = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)

                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct_epoch += (preds == labels).sum().item()
                total_epoch += labels.size(0)

            train_time = time.time() - t0
            train_times.append(train_time)

            tr_l = total_loss / len(train_loader)
            tr_a = correct_epoch / total_epoch
            train_losses.append(tr_l)
            train_accs.append(tr_a)

            # === VALIDATION ===
            t1 = time.time()
            model.eval()
            val_loss_total = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                    val_loss_total += loss.item()
                    preds = outputs.argmax(dim=1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            val_time = time.time() - t1
            val_times.append(val_time)

            va_l = val_loss_total / len(val_loader)
            va_a = correct_val / total_val
            val_losses.append(va_l)
            val_accs.append(va_a)

            # log in same style
            _log(
                f,
                (
                    f"Ep{epoch:2d} | "
                    f"train_loss {tr_l:.3f}, acc {tr_a:.3f} ({train_time:.1f}s) | "
                    f"val_loss   {va_l:.3f}, acc {va_a:.3f} ({val_time:.1f}s)"
                )
            )

        _log(f, "-" * 72)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_time": train_times,
        "val_time": val_times,
    }

def plot_comparison_metrics(metrics_race, metrics_gpt, save_path, K=VISION_CONFIG["K"], L=VISION_CONFIG["L"], M=VISION_CONFIG["M"]):
    epochs = range(1, len(metrics_race["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.3)

    def plot_metric(ax, metric_key, ylabel, title):
        # RACE
        ax.plot(epochs, metrics_race[f"train_{metric_key}"], label="RACE - Train", color="#1f77b4", marker='o', markersize=4, linewidth=2)
        ax.plot(epochs, metrics_race[f"val_{metric_key}"], label="RACE - Val", color="#1f77b4", linestyle='--', marker='x', markersize=4, linewidth=2)

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
    plot_metric(axes[1], "acc", "Accuracy", "Accuracy (Train vs Val)")

        # Compose extra info string
    extra_info = []
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

def start_experiment():
    device = "cpu"
    train_loader, val_loader = get_data(VISION_CONFIG)
    num_epochs = 1
    
    print("Training Softmax model...")
    torch.manual_seed(123)
    model_gpt = VisionTransformer(VISION_CONFIG, "softmax")
    model_gpt.to(device)
    optimizer_gpt = torch.optim.AdamW(model_gpt.parameters(), lr=1e-5, weight_decay=5e-5)

    metrics_gpt = train_model_simple(
        model_gpt, train_loader, val_loader, optimizer_gpt, device,
        num_epochs=num_epochs, cfg=VISION_CONFIG
    )

    print("Training RACE model...")
    torch.manual_seed(123)
    model_race = VisionTransformer(VISION_CONFIG, "race")
    model_race.to(device)
    optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=1e-5, weight_decay=5e-5)

    metrics_race = train_model_simple(
        model_race, train_loader, val_loader, optimizer_race, device,
        num_epochs=num_epochs, cfg=VISION_CONFIG
    )

    print("Training Linformer model...")
    torch.manual_seed(123)
    model_linformer = VisionTransformer(VISION_CONFIG, "linformer")
    model_linformer.to(device)
    optimizer_race = torch.optim.AdamW(model_linformer.parameters(), lr=1e-5, weight_decay=5e-5)

    metrics_race = train_model_simple(
        model_linformer, train_loader, val_loader, optimizer_race, device,
        num_epochs=num_epochs, cfg=VISION_CONFIG
    )

    print("Training LinearAttention...")
    torch.manual_seed(123)
    model_linear = VisionTransformer(VISION_CONFIG, "linear")
    model_linear.to(device)
    optimizer_linear = torch.optim.AdamW(model_linear.parameters(), lr=1e-5, weight_decay=5e-5)

    metrics_linear = train_model_simple(
        model_linear, train_loader, val_loader, optimizer_linear, device,
        num_epochs=num_epochs, cfg=VISION_CONFIG
    )

    print("Training Angular Attention....")
    torch.manual_seed(123)
    model_angular = torch.compile(VisionTransformer(VISION_CONFIG, "angular"))
    model_angular.to(device)
    optimizer_angular = torch.optim.AdamW(model_angular.parameters(), lr=1e-5, weight_decay=5e-5)

    metrics_angular = train_model_simple(
        model_angular, train_loader, val_loader, optimizer_angular, device,
        num_epochs=num_epochs, cfg=VISION_CONFIG
    )

    # plot_comparison_metrics(metrics_race, metrics_gpt, f"Vision_Plots.png")

start_experiment()
