import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _race_chunk_fwd_kernel(
    phiQ_ptr, phiK_ptr, V_ptr,
    state_A_ptr, state_B_ptr,
    num_ptr, den_ptr,
    # phiQ/phiK strides: [B, H, L, T, R]
    sQ_b, sQ_h, sQ_l, sQ_t,
    sK_b, sK_h, sK_l, sK_t,
    # V strides: [B, H, T, D]  (no L)
    sV_b, sV_h, sV_t,
    # state_A: [B, H, L, n, R]; state_B: [B, H, L, n, R, D]
    sSA_b, sSA_h, sSA_l, sSA_n,
    sSB_b, sSB_h, sSB_l, sSB_n, sSB_r,
    # num: [B, H, T, D]; den: [B, H, T]  (sum over L; L cancels in out = num/den)
    sN_b, sN_h, sN_t,
    sD_b, sD_h, sD_t,
    T, n_chunks,
    H: tl.constexpr, L: tl.constexpr,
    R: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)         # (b * H + h)
    pid_nm = tl.program_id(1)         # n_chunk * (C//BLOCK_M) + m_block_idx

    m_blocks_per_chunk = C // BLOCK_M
    pid_n = pid_nm // m_blocks_per_chunk
    pid_m = pid_nm % m_blocks_per_chunk

    b = pid_bh // H
    h = pid_bh % H

    chunk_start = pid_n * C
    m_start = chunk_start + pid_m * BLOCK_M

    offs_m = m_start + tl.arange(0, BLOCK_M)         # global token index within sequence
    offs_d = tl.arange(0, D)
    offs_r = tl.arange(0, R)
    offs_n_base = tl.arange(0, BLOCK_N)

    m_in_chunk = offs_m - chunk_start                 # token index within the chunk

    # V doesn't have L dim — base pointer once
    V_base = V_ptr + b * sV_b + h * sV_h

    # Accumulators over L
    num_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    den_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over L tables (unrolled at compile time)
    for l in tl.static_range(L):
        phiQ_base = phiQ_ptr + b * sQ_b + h * sQ_h + l * sQ_l
        phiK_base = phiK_ptr + b * sK_b + h * sK_h + l * sK_l
        sA_off = state_A_ptr + b * sSA_b + h * sSA_h + l * sSA_l + pid_n * sSA_n
        sB_off = state_B_ptr + b * sSB_b + h * sSB_h + l * sSB_l + pid_n * sSB_n

        # Load phiQ_block [BLOCK_M, R]
        phiQ_block = tl.load(
            phiQ_base + offs_m[:, None] * sQ_t + offs_r[None, :],
            mask=offs_m[:, None] < T, other=0.0,
        )

        # Load state_A_l [R] and state_B_l [R, D]
        state_A_l = tl.load(sA_off + offs_r)
        state_B_l = tl.load(sB_off + offs_r[:, None] * sSB_r + offs_d[None, :])

        # ----- INTER-CHUNK -----
        inter_num = tl.dot(phiQ_block, state_B_l)                     # [BLOCK_M, D]
        inter_den = tl.sum(phiQ_block * state_A_l[None, :], axis=1)   # [BLOCK_M]

        # ----- INTRA-CHUNK (causal within chunk, j <= i) -----
        intra_num = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        intra_den = tl.zeros([BLOCK_M], dtype=tl.float32)

        # End of K column loop = end of this M-block within chunk
        end_n_in_chunk = (pid_m + 1) * BLOCK_M
        for nj in range(0, end_n_in_chunk, BLOCK_N):
            offs_n_in_chunk = nj + offs_n_base
            offs_n = chunk_start + offs_n_in_chunk

            phiK_block = tl.load(
                phiK_base + offs_n[:, None] * sK_t + offs_r[None, :],
                mask=offs_n[:, None] < T, other=0.0,
            )                                                          # [BLOCK_N, R]

            # M = phiQ_block @ phiK_block^T  [BLOCK_M, BLOCK_N], fp32 accumulator
            M_ij = tl.dot(phiQ_block, tl.trans(phiK_block))

            # Causal mask within chunk
            causal_mask = m_in_chunk[:, None] >= offs_n_in_chunk[None, :]
            M_ij = tl.where(causal_mask, M_ij, 0.0)

            # V doesn't depend on L — load once per inner iter (cache will hit on L loop)
            V_block = tl.load(
                V_base + offs_n[:, None] * sV_t + offs_d[None, :],
                mask=offs_n[:, None] < T, other=0.0,
            )                                                          # [BLOCK_N, D]

            intra_num += tl.dot(M_ij.to(V_block.dtype), V_block)
            intra_den += tl.sum(M_ij, axis=1)

        num_acc += inter_num + intra_num
        den_acc += inter_den + intra_den

    # Store sum-over-L outputs (L division cancels in out = num/den)
    num_base = num_ptr + b * sN_b + h * sN_h
    den_base = den_ptr + b * sD_b + h * sD_h
    tl.store(
        num_base + offs_m[:, None] * sN_t + offs_d[None, :],
        num_acc,
        mask=offs_m[:, None] < T,
    )
    tl.store(
        den_base + offs_m * sD_t,
        den_acc,
        mask=offs_m < T,
    )


def race_chunkwise_triton_forward(phiQ, phiK, V, state_A, state_B, C):
    """
    phiQ, phiK: [B, H, L, T, R]   (must be contiguous in last dim)
    V:          [B, H, T, D]
    state_A:    [B, H, L, n_chunks, R]    (exclusive cumsum over chunks)
    state_B:    [B, H, L, n_chunks, R, D]
    Returns:
        num: [B, H, T, D]  (sum over L)
        den: [B, H, T]     (sum over L)
    """
    B, H, L, T, R = phiQ.shape
    _, _, _, D = V.shape
    n_chunks = state_A.shape[3]
    assert T == n_chunks * C, f"T={T} must equal n_chunks*C={n_chunks*C}"

    num = torch.empty(B, H, T, D, dtype=torch.float32, device=phiQ.device)
    den = torch.empty(B, H, T,    dtype=torch.float32, device=phiQ.device)

    BLOCK_M = 64
    BLOCK_N = 64
    m_blocks_per_chunk = C // BLOCK_M

    grid = (B * H, n_chunks * m_blocks_per_chunk)
    _race_chunk_fwd_kernel[grid](
        phiQ, phiK, V, state_A, state_B, num, den,
        phiQ.stride(0), phiQ.stride(1), phiQ.stride(2), phiQ.stride(3),
        phiK.stride(0), phiK.stride(1), phiK.stride(2), phiK.stride(3),
        V.stride(0),    V.stride(1),    V.stride(2),
        state_A.stride(0), state_A.stride(1), state_A.stride(2), state_A.stride(3),
        state_B.stride(0), state_B.stride(1), state_B.stride(2), state_B.stride(3), state_B.stride(4),
        num.stride(0), num.stride(1), num.stride(2),
        den.stride(0), den.stride(1), den.stride(2),
        T, n_chunks,
        H=H, L=L, R=R, D=D, C=C,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return num, den