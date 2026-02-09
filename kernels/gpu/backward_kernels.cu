#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <stdint.h>
using at::Tensor;

// -------------------- warp+block reductions (float) --------------------
__inline__ __device__ float warp_reduce_sum(float val)
{
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}
__inline__ __device__ float block_reduce_sum(float val)
{
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? shared[lane] : 0.0f;
    if (wid == 0)
        val = warp_reduce_sum(val);
    return val;
}

// ============================================================================
// Computes gradK and gradQ without a linear timee forward scan:
//   init A,B from A_final/B_final, then do a reverse scan across the token dimension (T)
// Outputs:
//   gradK [N,T,S], gradQ [N,T,S]
// ============================================================================
__global__ void race_bwd_kq_noscan_cuda(
    const float *__restrict__ probsK,   // [N,T,S]
    const float *__restrict__ probsQ,   // [N,T,S]
    const float *__restrict__ V2,       // [N,T,D]
    const float *__restrict__ grad_out, // [N,T,D]
    const float *__restrict__ A_final,  // [N,S]
    const __half *__restrict__ B_final, // [N,S,D] fp16
    float *__restrict__ gradK,          // [N,T,S]
    float *__restrict__ gradQ,          // [N,T,S]
    int N, int T, int S, int D,
    float eps)
{
    int n = blockIdx.y;
    int s = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N || s >= S)
        return;

    extern __shared__ float sh[];
    float *sh_B = sh;         // [D]
    float *sh_gBn = sh_B + D; // [D]
    __shared__ float sh_A;
    __shared__ float sh_gAn;

    // init from finals
    for (int d = tid; d < D; d += blockDim.x)
    {
        size_t idxBf = ((size_t)n * (size_t)S + (size_t)s) * (size_t)D + (size_t)d;
        sh_B[d] = __half2float(B_final[idxBf]);
        sh_gBn[d] = 0.0f;
    }
    if (tid == 0)
    {
        sh_A = A_final[(size_t)n * (size_t)S + (size_t)s];
        sh_gAn = 0.0f;
    }
    __syncthreads();

    for (int t = T - 1; t >= 0; --t)
    {
        size_t idxKS = ((size_t)n * (size_t)T + (size_t)t) * (size_t)S + (size_t)s;
        float pk = probsK[idxKS];
        float pq = probsQ[idxKS];

        float A = sh_A;
        float denom = A + eps;
        float inv = 1.0f / denom;
        float inv2 = inv * inv;

        float gradA_loc_part = 0.0f;
        float gradQ_part = 0.0f;
        float sum_gBv_part = 0.0f;

        size_t baseVD = ((size_t)n * (size_t)T + (size_t)t) * (size_t)D;

        for (int d = tid; d < D; d += blockDim.x)
        {
            float B = sh_B[d];
            float v = V2[baseVD + (size_t)d];
            float go = grad_out[baseVD + (size_t)d];

            float E = B * inv;
            gradQ_part += go * E;

            float U = pq * go;
            float gB_loc = U * inv;
            gradA_loc_part += U * (-B) * inv2;

            float gB = gB_loc + sh_gBn[d];
            sh_gBn[d] = gB;

            sum_gBv_part += gB * v;
        }

        float gradA_loc = block_reduce_sum(gradA_loc_part);
        float gradQ_val = block_reduce_sum(gradQ_part);
        float sum_gBv = block_reduce_sum(sum_gBv_part);

        if (tid == 0)
        {
            float gA = gradA_loc + sh_gAn;
            gradK[idxKS] = gA + sum_gBv;
            gradQ[idxKS] = gradQ_val;
            sh_gAn = gA;
        }
        __syncthreads();

        // update to t-1
        if (tid == 0)
            sh_A -= pk;
        for (int d = tid; d < D; d += blockDim.x)
        {
            float v = V2[baseVD + (size_t)d];
            sh_B[d] -= pk * v;
        }
        __syncthreads();
    }
}

// ============================================================================
// Contains the necessary logic to invoke race_bwd_kq_noscan_cuda
//     and computes the gradient of K and the gradient of Q
// Outputs:
//   gradK [N,T,S], gradQ [N,T,S]
// ============================================================================
std::vector<Tensor> race_bwd_kq_noscan(
    Tensor probsK, Tensor probsQ, Tensor V2, Tensor grad_out,
    Tensor A_final, Tensor B_final, float eps)
{
    TORCH_CHECK(probsK.is_cuda() && probsQ.is_cuda() && V2.is_cuda() && grad_out.is_cuda(), "CUDA only");
    TORCH_CHECK(A_final.is_cuda() && B_final.is_cuda(), "CUDA only");
    TORCH_CHECK(probsK.scalar_type() == at::kFloat && probsQ.scalar_type() == at::kFloat && V2.scalar_type() == at::kFloat, "fp32 only");
    TORCH_CHECK(grad_out.scalar_type() == at::kFloat, "grad_out fp32");
    TORCH_CHECK(A_final.scalar_type() == at::kFloat, "A_final fp32");
    TORCH_CHECK(B_final.scalar_type() == at::kHalf, "B_final fp16");

    int N = probsK.size(0), T = probsK.size(1), S = probsK.size(2), D = V2.size(2);

    auto gK = torch::empty_like(probsK);
    auto gQ = torch::empty_like(probsQ);

    dim3 grid(S, N, 1);
    int threads = 256;
    size_t shmem = (size_t)(2 * D) * sizeof(float);

    race_bwd_kq_noscan_cuda<<<grid, threads, shmem>>>(
        probsK.data_ptr<float>(),
        probsQ.data_ptr<float>(),
        V2.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        A_final.data_ptr<float>(),
        (const __half *)B_final.data_ptr<at::Half>(),
        gK.data_ptr<float>(),
        gQ.data_ptr<float>(),
        N, T, S, D, eps);
    return {gK, gQ};
}

// ============================================================================
// Backward for gradV (NO forward scan):
// Uses only A_final[n,s], reverse scan over t, reduces over s in-block.
// One block per (n,d), threads over s (S <= 1024).
// ============================================================================
__global__ void race_bwd_v_noscan_cuda(
    const float *__restrict__ probsK,   // [N,T,S]
    const float *__restrict__ probsQ,   // [N,T,S]
    const float *__restrict__ grad_out, // [N,T,D]
    const float *__restrict__ A_final,  // [N,S]
    float *__restrict__ gradV,          // [N,T,D]
    int N, int T, int S, int D,
    float eps)
{
    int n = blockIdx.y;
    int d = blockIdx.x;
    int s = threadIdx.x;
    if (n >= N || d >= D || s >= S)
        return;

    float A = A_final[(size_t)n * (size_t)S + (size_t)s];
    float gBn = 0.0f;

    for (int t = T - 1; t >= 0; --t)
    {
        size_t idxKS = ((size_t)n * (size_t)T + (size_t)t) * (size_t)S + (size_t)s;
        float pk = probsK[idxKS];
        float pq = probsQ[idxKS];

        size_t idxVD = ((size_t)n * (size_t)T + (size_t)t) * (size_t)D + (size_t)d;
        float go = grad_out[idxVD];

        float inv = 1.0f / (A + eps);
        float gB_loc = (pq * go) * inv;
        float gB = gB_loc + gBn;

        float contrib = gB * pk;
        float total = block_reduce_sum(contrib);
        if (s == 0)
            gradV[idxVD] = total;

        __syncthreads();
        gBn = gB;
        A -= pk;
        __syncthreads();
    }
}

// ============================================================================
// Contains the necessary logic to invoke race_bwd_v_noscan_cuda
//     and computes the gradient of V
// Outputs:
//   gradV [N,T,D]
// ============================================================================
Tensor race_bwd_v_noscan(
    Tensor probsK, Tensor probsQ, Tensor grad_out, Tensor A_final, float eps)
{
    TORCH_CHECK(probsK.is_cuda() && probsQ.is_cuda() && grad_out.is_cuda() && A_final.is_cuda(), "CUDA only");
    TORCH_CHECK(probsK.scalar_type() == at::kFloat && probsQ.scalar_type() == at::kFloat, "fp32 only");
    TORCH_CHECK(grad_out.scalar_type() == at::kFloat, "grad_out fp32");
    TORCH_CHECK(A_final.scalar_type() == at::kFloat, "A_final fp32");

    int N = probsK.size(0), T = probsK.size(1), S = probsK.size(2), D = grad_out.size(2);
    TORCH_CHECK(S <= 1024, "S must be <= 1024 for this kernel");

    auto gV = torch::empty_like(grad_out);

    dim3 grid(D, N, 1);
    dim3 block(S, 1, 1);

    race_bwd_v_noscan_cuda<<<grid, block>>>(
        probsK.data_ptr<float>(),
        probsQ.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        A_final.data_ptr<float>(),
        gV.data_ptr<float>(),
        N, T, S, D, eps);
    return gV;
}