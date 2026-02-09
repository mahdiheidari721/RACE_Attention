#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <stdint.h>

using at::Tensor;

// ============================================================================
// Fused forward:
//  out[n,t,d] += sum_s probsQ[n,t,s] * (B_{n,s,d}(t) / (A_{n,s}(t)+eps))
// where:
//  A_{n,s}(t) = sum_{τ<=t} probsK[n,τ,s]
//  B_{n,s,d}(t) = sum_{τ<=t} probsK[n,τ,s] * V2[n,τ,d]
// Also writes final states to be used oin backward noscan:
//  A_final[n,s] = A_{n,s}(T-1)
//  B_final[n,s,d] = B_{n,s,d}(T-1)   (stored fp16 to save memory)
// ============================================================================
__global__ void race_fused_fwd_cuda(
    const float *__restrict__ probsK, // [N,T,S]
    const float *__restrict__ probsQ, // [N,T,S]
    const float *__restrict__ V2,     // [N,T,D]
    float *__restrict__ out,          // [N,T,D] (must be zeroed)
    float *__restrict__ A_final,      // [N,S]
    __half *__restrict__ B_final,     // [N,S,D] (fp16)
    int N, int T, int S, int D,
    float eps)
{
    int n = blockIdx.y;
    int s = blockIdx.x;
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (n >= N || s >= S || d >= D)
        return;

    float A = 0.0f;
    float B = 0.0f;

    for (int t = 0; t < T; ++t)
    {
        size_t idxKS = ((size_t)n * (size_t)T + (size_t)t) * (size_t)S + (size_t)s;
        float pk = probsK[idxKS];
        float pq = probsQ[idxKS];

        size_t idxV = ((size_t)n * (size_t)T + (size_t)t) * (size_t)D + (size_t)d;
        float v = V2[idxV];

        A += pk;
        B += pk * v;

        float e = B / (A + eps);
        atomicAdd(&out[idxV], pq * e);
    }

    // write final A once per (n,s)
    if (d == 0)
    {
        A_final[(size_t)n * (size_t)S + (size_t)s] = A;
    }
    // write final B per d
    size_t idxBf = ((size_t)n * (size_t)S + (size_t)s) * (size_t)D + (size_t)d;
    B_final[idxBf] = __float2half(B);
}

std::vector<Tensor> race_fused_fwd(Tensor probsK, Tensor probsQ, Tensor V2, float eps)
{
    TORCH_CHECK(probsK.is_cuda() && probsQ.is_cuda() && V2.is_cuda(), "CUDA only");
    TORCH_CHECK(probsK.scalar_type() == at::kFloat, "probsK must be fp32");
    TORCH_CHECK(probsQ.scalar_type() == at::kFloat, "probsQ must be fp32");
    TORCH_CHECK(V2.scalar_type() == at::kFloat, "V2 must be fp32");
    TORCH_CHECK(probsK.dim() == 3 && probsQ.dim() == 3 && V2.dim() == 3, "shapes must be [N,T,S],[N,T,S],[N,T,D]");

    int N = probsK.size(0);
    int T = probsK.size(1);
    int S = probsK.size(2);
    int D = V2.size(2);
    TORCH_CHECK(probsQ.size(0) == N && probsQ.size(1) == T && probsQ.size(2) == S, "probsQ mismatch");
    TORCH_CHECK(V2.size(0) == N && V2.size(1) == T, "V2 mismatch");

    auto out = torch::zeros({N, T, D}, torch::TensorOptions().device(probsK.device()).dtype(at::kFloat));
    auto A_final = torch::empty({N, S}, torch::TensorOptions().device(probsK.device()).dtype(at::kFloat));
    auto B_final = torch::empty({N, S, D}, torch::TensorOptions().device(probsK.device()).dtype(at::kHalf));

    int threads = 256;
    int block_x = (D < threads) ? D : threads;
    dim3 block(block_x, 1, 1);
    dim3 grid(S, N, (D + block_x - 1) / block_x);

    race_fused_fwd_cuda<<<grid, block>>>(
        probsK.data_ptr<float>(),
        probsQ.data_ptr<float>(),
        V2.data_ptr<float>(),
        out.data_ptr<float>(),
        A_final.data_ptr<float>(),
        (__half *)B_final.data_ptr<at::Half>(),
        N, T, S, D, eps);
    return {out, A_final, B_final};
}