# model.py
import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            cfg["num_channels"], cfg["embed_dim"],
            kernel_size=cfg["patch_size"], stride=cfg["patch_size"]
        )

    def forward(self, x):
        x = self.patch_embed(x)    # [B, d, H', W']
        x = x.flatten(2)           # [B, d, H'*W']
        x = x.transpose(1, 2)      # [B, N, d]
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
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            Q, K, V, is_causal=False, dropout_p=self.dropout_p
        )  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class TransformerArchitecture(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(cfg["embed_dim"])
        self.self_attention = MultiHeadAttention(
            d_in=cfg["embed_dim"], d_out=cfg["embed_dim"],
            dropout=cfg["drop_rate"], num_heads=cfg["num_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
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
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer('planes_T', planes.view(L * K, d_k).T)   # [d_k, L*K]
        else:
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)           # [M, d_k, L*K]
            self.register_buffer('planes_T', planes)

        corners = torch.tensor(list(itertools.product([-1., +1.], repeat=K)), device=device)
        self.register_buffer('protos_T', corners.T)                        # [K, R]

        # learnable temperature
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)

        if self.share_planes:
            N = M * B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            projK = Kh2 @ self.planes_T
            projQ = Qh2 @ self.planes_T
        else:
            BH = B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            projK = torch.einsum('mbtd,mds->mbts', Kh2, self.planes_T)
            projQ = torch.einsum('mbtd,mds->mbts', Qh2, self.planes_T)
            projK = projK.contiguous().view(M * BH, T, self.L * self.K)
            projQ = projQ.contiguous().view(M * BH, T, self.L * self.K)
            V2    = V2.view(M * BH, T, dk)
            N     = M * BH

        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)

        logitsK = (projK.tanh().div(scale) @ self.protos_T)
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)
        probsK  = F.softmax(logitsK, dim=-1)
        probsQ  = F.softmax(logitsQ, dim=-1)

        probsK_S = probsK.contiguous().view(N, T, S)
        probsQ_S = probsQ.contiguous().view(N, T, S)

        b_sum = probsK_S.transpose(1, 2).bmm(V2)
        A = probsK_S.sum(dim=1)
        E = b_sum / (A.unsqueeze(-1) + eps)

        out2 = probsQ_S.bmm(E)
        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4)
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

        Q = self.q_proj(x).view(B, T, H, d_k)
        K = self.k_proj(x).view(B, T, H, d_k)
        V = self.v_proj(x).view(B, T, H, d_k)

        def pack(Z):
            return Z.unsqueeze(0).expand(M, -1, -1, -1, -1)

        Khf = pack(K)
        Vhf = pack(V)
        Qhf = pack(Q)

        out_hm = self.ace(Khf, Vhf, Qhf)  # [M,B,T,H,d_k]

        out = out_hm.mean(dim=0)          # [B,T,H,d_k]
        out = out.permute(0, 2, 1, 3).reshape(B, T, H * d_k)
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


def favorplus_features(x, proj, eps=1e-6):
    """
    FAVOR+ positive random features for softmax kernel.
    x:    [B,H,T,D]
    proj: [H,M,D]  (one matrix per head; rows ~ N(0, I))
    ->    [B,H,T,M]  (non-negative)
    """
    xw = torch.einsum('bhtd,hmd->bhtm', x, proj)
    xw = xw - xw.max(dim=-1, keepdim=True).values

    exp_part  = torch.exp(xw)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    base      = torch.exp(-0.5 * x_norm_sq)
    return exp_part * base + eps


class FavorPlusAttention(nn.Module):
    """
    Non-causal FAVOR+ (Performer) attention (softmax kernel via positive RF).
    """
    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
        super().__init__()
        assert d % h == 0, "Embedding dim must be divisible by num_heads"
        self.h  = h
        self.dk = d // h
        self.m  = m_features

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        if seed is not None:
            torch.manual_seed(seed)
        proj = torch.nn.init.orthogonal_(torch.randn(h, m_features, self.dk))
        self.register_buffer("proj", proj)

        self.ctx = None
        self.eps = 1e-6

    def forward(self, x):
        B, T, d = x.shape
        h, dk, m = self.h, self.dk, self.m

        Q = self.q(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        Qs = Q / math.sqrt(dk)
        Ks = K / math.sqrt(dk)

        phiQ = favorplus_features(Qs, self.proj, eps=self.eps) / math.sqrt(m)
        phiK = favorplus_features(Ks, self.proj, eps=self.eps) / math.sqrt(m)

        KV   = torch.einsum("bhtm,bhtd->bhmd", phiK, V)
        Ksum = phiK.sum(dim=2)

        num = torch.einsum("bhtm,bhmd->bhtd", phiQ, KV)
        den = torch.einsum("bhtm,bhm->bht",   phiQ, Ksum).unsqueeze(-1) + self.eps
        out_heads = num / den

        merged = out_heads.transpose(1, 2).contiguous().view(B, T, h * dk)
        self.ctx = merged

        merged = self.drop(merged)
        return self.o(merged)


class PerformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = FavorPlusAttention(
            d=cfg["embed_dim"],
            h=cfg["num_heads"],
            m_features=cfg.get("m_features", 256),
            drop=cfg["drop_rate"],
            qkv_bias=cfg.get("qkv_bias", False),
            seed=cfg.get("favor_seed", None),
        )
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
        return F.elu(x) + 1  # [B, H, T, D]

    def forward(self, x):
        B, T, _ = x.size()

        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = self.kernel(Q)
        K = self.kernel(K)

        KV = torch.einsum('bhtd,bhte->bhde', K, V)

        K_sum = K.sum(dim=2)
        Z = torch.einsum('bhtd,bhd->bht', Q, K_sum) + self.eps

        context = torch.einsum('bhtd,bhde->bhte', Q, KV)
        out = context / Z.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
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
        self.att = AngularAttention(
            d=cfg["embed_dim"], h=cfg["num_heads"],
            drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"]
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


class LinformerAttention(nn.Module):
    """
    Linformer-style attention: project K,V along sequence length T -> k (k << T),
    then do standard scaled dot-product attention with softmax over k.
    """
    def __init__(
        self,
        d: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        k_proj_dim: int,
        max_seq_len: int
    ):
        super().__init__()
        assert d % num_heads == 0, "d_out must be divisible by num_heads"
        self.h = num_heads
        self.dk = d // num_heads
        self.k_proj_dim = k_proj_dim
        self.max_seq_len = max_seq_len

        self.W_query = nn.Linear(d,  d, bias=qkv_bias)
        self.W_key   = nn.Linear(d,  d, bias=qkv_bias)
        self.W_value = nn.Linear(d,  d, bias=qkv_bias)

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

        Q = self.W_query(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.W_key(  x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        Ek = self.E_k[:T]
        Ev = self.E_v[:T]

        K_proj = torch.einsum("bhtd,tk->bhkd", K, Ek)
        V_proj = torch.einsum("bhtd,tk->bhkd", V, Ev)

        scale = 1.0 / math.sqrt(dk)
        scores = torch.einsum("bhtd,bhkd->bhtk", Q, K_proj) * scale
        attn = F.softmax(scores, dim=-1)

        ctx = torch.einsum("bhtk,bhkd->bhtd", attn, V_proj)

        out = ctx.transpose(1, 2).contiguous().view(B, T, h * dk)
        return self.out_proj(self.dropout(out))


class LinformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        k_proj_dim = 128

        self.att  = LinformerAttention(
            d=cfg["embed_dim"], dropout=drop, num_heads=cfg["num_heads"],
            qkv_bias=qkv_bias, k_proj_dim=k_proj_dim,
            max_seq_len=cfg["num_patches"] + 1
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
                        nn.GELU(),
                        nn.Linear(cfg["mlp_dim"], cfg["embed_dim"]),
                     )
        self.drop  = nn.Dropout(drop)

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


class VisionTransformer(nn.Module):
    def __init__(self, cfg, attn_type: str, device: str = "cpu"):
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
        elif attn_type == "performer":
            AttnBlock = PerformerBlock
        else:
            raise ValueError("Unsupported attention type")

        self.transformer_layers = nn.Sequential(
            *[AttnBlock(cfg) for _ in range(cfg["transformer_units"])]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg["mlp_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(cfg["mlp_dim"], cfg["num_classes"])
        )

    def forward(self, x):
        x = self.patch_embedding(x)                 # [B, N, d], N=G*G
        B, N, d = x.shape
        cls = self.cls_token.expand(B, -1, -1)      # [B,1,d]
        x = torch.cat([cls, x], dim=1)              # [B, N+1, d]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer_layers(x)
        x = x[:, 0]
        return self.mlp_head(x)
