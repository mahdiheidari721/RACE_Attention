import os, platform, torch
from torch.utils.cpp_extension import load
import time, statistics as stats
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

src = "/Users/sahiljoshi/Documents/Research/race_pref.cpp" # Path to the .cpp file

extra_cflags = ["-O3"]
extra_ldflags = []

if platform.system() == "Darwin":
    # Where Homebrew installs libomp
    omp_prefix = os.environ.get("LIBOMP_PREFIX")
    if not omp_prefix:
        # try common locations
        for p in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
            if os.path.isdir(p):
                omp_prefix = p
                break
    if not omp_prefix:
        raise SystemExit("libomp not found. Run: brew install libomp "
                         "or set LIBOMP_PREFIX=/path/to/libomp")

    extra_cflags += ["-Xpreprocessor", "-fopenmp", f"-I{omp_prefix}/include"]
    extra_ldflags += [f"-L{omp_prefix}/lib", "-lomp", f"-Wl,-rpath,{omp_prefix}/lib"]
else:
    # Linux/WSL
    extra_cflags += ["-fopenmp"]
    extra_ldflags += ["-fopenmp"]

ext = load(
    name="race_pref",
    sources=[src],
    extra_cflags=extra_cflags,
    extra_ldflags=extra_ldflags,
    verbose=True,
)

print("OK:", dir(ext))

# ---------- reference in PyTorch ----------
def rfa_prefix_mean_ref(probsK, V, eps=1e-6):
    # probsK: [N,T,S], V: [N,T,D] -> [N,T,S,D]
    A_pref = probsK.cumsum(dim=1)                                        # [N,T,S]
    B_pref = (probsK.unsqueeze(-1) * V.unsqueeze(2)).cumsum(dim=1)       # [N,T,S,D]
    return B_pref / (A_pref.unsqueeze(-1) + eps)                         # [N,T,S,D]

# ---------- helpers ----------
def median_time(fn, *args, warmup=3, repeat=10):
    # warmup
    for _ in range(warmup):
        _ = fn(*args)
    # measure
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = fn(*args)
        times.append(time.perf_counter() - t0)
    return stats.median(times)

def bench_once(N=32, T=512, L=4, R=4, D=64, eps=1e-6, device="cpu"):
    torch.manual_seed(0)
    S = L * R

    probsK = torch.rand(N, T, S, device=device, dtype=torch.float32).contiguous()
    V      = torch.randn(N, T, D, device=device, dtype=torch.float32).contiguous()

    # Reference
    tref = median_time(rfa_prefix_mean_ref, probsK, V, eps)

    # OpenMP (N,T,S,D) version
    tntsd = median_time(ext.rfa_prefix_mean_ntsd, probsK, V, eps)

    # ---------- FIXED FLAT VERSION ----------
    # Flatten streams as (n, s) with time last -> [NS, T]
    probsK_flat = probsK.permute(0, 2, 1).contiguous().view(N*S, T)
    # Duplicate V across S, align time axis, then flatten -> [NS, T, D]
    V_flat = (
        V.unsqueeze(2)                 # [N,T,1,D]
         .expand(N, T, S, D)           # [N,T,S,D]
         .permute(0, 2, 1, 3)          # [N,S,T,D]
         .contiguous()
         .view(N*S, T, D)
    )
    tflat = median_time(ext.rfa_prefix_mean_flat, probsK_flat, V_flat, eps)
    # ---------------------------------------

    # correctness checks
    E_ref  = rfa_prefix_mean_ref(probsK, V, eps)                   # [N,T,S,D]
    E_ntsd = ext.rfa_prefix_mean_ntsd(probsK, V, eps)              # [N,T,S,D]

    # Unflatten back to [N,T,S,D] for comparison
    E_ext_flat = ext.rfa_prefix_mean_flat(probsK_flat, V_flat, eps)              # [NS,T,D]
    E_flat = (
        E_ext_flat.view(N, S, T, D)        # [N,S,T,D]
                .permute(0, 2, 1, 3)       # [N,T,S,D]
                .contiguous()
    )

    print(f"Shapes  N={N} T={T} S={S} D={D} | OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')} | torch threads={torch.get_num_threads()}")
    print(f"Ref (cumsum):     {tref*1e3:7.2f} ms")
    print(f"NTSD (OpenMP):    {tntsd*1e3:7.2f} ms  | speedup vs ref: {tref/tntsd:6.1f}×")
    print(f"Flat (OpenMP):    {tflat*1e3:7.2f} ms  | speedup vs ref: {tref/tflat:6.1f}×")
    print("Max |diff| ntsd:", (E_ref - E_ntsd).abs().max().item())
    print("Max |diff| flat:", (E_ref - E_flat).abs().max().item())


# ---------- run a couple of sizes ----------
bench_once(N=32, T=64,  L=2, R=2, D=32)     # small sanity
bench_once(N=16, T=256, L=4, R=4, D=64)     # moderate
bench_once(N=8,  T=512, L=4, R=8, D=64)     # larger S



# --- Autograd Function (flat) ---
class RFAPrefixMeanFlatFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probsK_flat, V_flat, eps: float):
        probsK_flat = probsK_flat.contiguous()
        V_flat = V_flat.contiguous()
        E = ext.rfa_prefix_mean_flat(probsK_flat, V_flat, float(eps))
        ctx.save_for_backward(probsK_flat, V_flat)
        ctx.eps = float(eps)
        return E

    @staticmethod
    def backward(ctx, gradE):
        probsK_flat, V_flat = ctx.saved_tensors
        gradE = gradE.contiguous()
        gW, gV = ext.rfa_prefix_mean_flat_bw(probsK_flat, V_flat, gradE, ctx.eps)
        return gW, gV, None

def rfa_prefix_mean_flat_ref(w, v, eps=1e-6):
    # w: [NS,T], v:[NS,T,D] -> [NS,T,D]
    A = w.cumsum(dim=1)                   # [NS,T]
    B = (w.unsqueeze(-1) * v).cumsum(dim=1)  # [NS,T,D]
    return B / (A.unsqueeze(-1) + eps)

# --- gradient check against PyTorch autograd on the ref ---
torch.manual_seed(0)

NS, T, D = 256, 1024, 256
eps = 1e-6
w = torch.rand(NS, T, dtype=torch.float32, requires_grad=True)
v = torch.randn(NS, T, D, dtype=torch.float32, requires_grad=True)

# same random weights for the loss so both see identical upstream gradients
Wloss = torch.randn(NS, T, D)

# Reference
E_ref = rfa_prefix_mean_flat_ref(w, v, eps)
loss_ref = (E_ref * Wloss).sum()
loss_ref.backward()
gW_ref, gV_ref = w.grad.detach().clone(), v.grad.detach().clone()

# Extension
w2 = w.detach().clone().requires_grad_(True)
v2 = v.detach().clone().requires_grad_(True)
E_ext = RFAPrefixMeanFlatFn.apply(w2, v2, eps)
loss_ext = (E_ext * Wloss).sum()
loss_ext.backward()
gW_ext, gV_ext = w2.grad, v2.grad

print("max|E diff|:", (E_ref - E_ext).abs().max().item())
print("max|gW diff|:", (gW_ref - gW_ext).abs().max().item())
print("max|gV diff|:", (gV_ref - gV_ext).abs().max().item())

def relerr(a, b, eps=1e-12):
    return (a - b).abs().max().item() / (b.abs().max().item() + eps)

print("rel |gW|:", relerr(gW_ext, gW_ref))
print("rel |gV|:", relerr(gV_ext, gV_ref))
print("rel |E| :", relerr(E_ext,  E_ref))


# torch.manual_seed(0)
# M,B,T,H,dk = 1,16,64,1,128
# K,L = 2,2
# ace = BatchedACE(dk, K, L, M, device="cpu")   # CPU!
# Khf = torch.randn(M,B,T,H,dk, requires_grad=True)
# Qhf = torch.randn(M,B,T,H,dk, requires_grad=True)
# Vhf = torch.randn(M,B,T,H,dk, requires_grad=True)

# out = ace(Khf, Vhf, Qhf)           # should return in ms
# loss = out.pow(2).mean()
# loss.backward()                     # should also be quick
# print("ok", out.shape)
