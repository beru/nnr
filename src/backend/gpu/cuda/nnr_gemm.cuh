// NNR CUDA GEMM library — optimized kernels for all shape classes.
//
// Three kernel tiers:
//   1. sgemm_large:  128×128 tile, cp.async + float4 + double-buf (for M,N ≥ 128)
//   2. sgemm_medium: 64×64 tile, float4 + register prefetch (for M,N ≥ 32)
//   3. sgemm_gemv:   warp-parallel reduction along K (for M ≤ 16)
//
// Plus split-K in a single launch (3D grid) for few-block cases.
//
// Dispatch: nnr_gemm::sgemm() picks the best kernel automatically.

#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

namespace nnr_gemm {

// ===========================================================================
// Tier 3: GEMV kernel for M ≤ 16
// ===========================================================================
// Each thread block handles one row of C and reduces along K.
// Grid: (ceil(N/BN), M), Block: (BN, 1)
// Each thread computes one element of C by accumulating A[m,k]*B[k,n] for all k.

template <int BN>
__global__ void sgemv_kernel(
    const int M, const int N, const int K,
    const float alpha,
    const float* __restrict__ A, const int lda,
    const float* __restrict__ B, const int ldb,
    const float beta,
    float* __restrict__ C, const int ldc)
{
    const int gm = blockIdx.y;
    const int gn = blockIdx.x * BN + threadIdx.x;
    if (gm >= M || gn >= N) return;

    const float* a_row = &A[gm * lda];
    float sum = 0.f;

    // Vectorized K reduction — 4 elements at a time
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float4 a4 = *reinterpret_cast<const float4*>(&a_row[k]);
        sum += a4.x * B[(k+0) * ldb + gn]
             + a4.y * B[(k+1) * ldb + gn]
             + a4.z * B[(k+2) * ldb + gn]
             + a4.w * B[(k+3) * ldb + gn];
    }
    for (; k < K; k++)
        sum += a_row[k] * B[k * ldb + gn];

    float val = alpha * sum;
    if (beta != 0.f) val += beta * C[gm * ldc + gn];
    C[gm * ldc + gn] = val;
}

// ===========================================================================
// Tier 2: Medium kernel — 64×64 with float4 loads + register prefetch
// ===========================================================================
// For shapes where 128×128 would create too few blocks.
// BM=64, BN=64, BK=16, TM=4, TN=4 → 16×16 = 256 threads.

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(256)
void sgemm_medium_kernel(
    const int M, const int N, const int K,
    const float alpha,
    const float* __restrict__ A, const int lda,
    const float* __restrict__ B, const int ldb,
    const float beta,
    float* __restrict__ C, const int ldc)
{
    constexpr int TPB_X = BN / TN;
    constexpr int TPB_Y = BM / TM;
    constexpr int NUM_T = TPB_X * TPB_Y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TPB_X + tx;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    __shared__ float As[BK][BM];  // transposed
    __shared__ float Bs[BK][BN];

    float acc[TM][TN] = {};

    constexpr int A_TOT = BM * BK;
    constexpr int B_TOT = BK * BN;
    // Number of float4 loads per thread for A and B
    constexpr int A_F4 = (A_TOT / 4 + NUM_T - 1) / NUM_T;
    constexpr int B_F4 = (B_TOT / 4 + NUM_T - 1) / NUM_T;
    float4 a_buf[A_F4], b_buf[B_F4];

    const int num_tiles = (K + BK - 1) / BK;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int k_base = tile * BK;

        // Load A into registers (float4 where aligned, scalar fallback)
        #pragma unroll
        for (int ld = 0; ld < A_F4; ld++) {
            int flat = (tid + ld * NUM_T) * 4;
            if (flat < A_TOT) {
                int am = flat / BK, ak = flat % BK;
                int gm = bm + am, gk = k_base + ak;
                const float* src = &A[gm * lda + gk];
                bool ok = (gm < M && gk + 3 < K);
                bool aligned = ok && ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
                if (aligned) {
                    a_buf[ld] = __ldg(reinterpret_cast<const float4*>(src));
                } else {
                    float t[4];
                    for (int i = 0; i < 4; i++)
                        t[i] = (gm < M && gk+i < K) ? A[gm*lda+gk+i] : 0.f;
                    a_buf[ld] = *reinterpret_cast<float4*>(t);
                }
            }
        }

        // Load B (float4)
        #pragma unroll
        for (int ld = 0; ld < B_F4; ld++) {
            int flat = (tid + ld * NUM_T) * 4;
            if (flat < B_TOT) {
                int bk = flat / BN, bni = flat % BN;
                int gk = k_base + bk, gn = bn + bni;
                const float* src = &B[gk * ldb + gn];
                bool ok = (gk < K && gn + 3 < N);
                bool aligned = ok && ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
                if (aligned) {
                    b_buf[ld] = __ldg(reinterpret_cast<const float4*>(src));
                } else {
                    float t[4];
                    for (int i = 0; i < 4; i++)
                        t[i] = (gk < K && gn+i < N) ? B[gk*ldb+gn+i] : 0.f;
                    b_buf[ld] = *reinterpret_cast<float4*>(t);
                }
            }
        }

        // Store A transposed
        #pragma unroll
        for (int ld = 0; ld < A_F4; ld++) {
            int flat = (tid + ld * NUM_T) * 4;
            if (flat < A_TOT) {
                int am = flat / BK, ak = flat % BK;
                As[ak+0][am] = a_buf[ld].x;
                As[ak+1][am] = a_buf[ld].y;
                As[ak+2][am] = a_buf[ld].z;
                As[ak+3][am] = a_buf[ld].w;
            }
        }

        // Store B
        #pragma unroll
        for (int ld = 0; ld < B_F4; ld++) {
            int flat = (tid + ld * NUM_T) * 4;
            if (flat < B_TOT) {
                int bk = flat / BN, bni = flat % BN;
                *reinterpret_cast<float4*>(&Bs[bk][bni]) = b_buf[ld];
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float af[TM], bf[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) af[i] = As[k][ty*TM+i];
            #pragma unroll
            for (int j = 0; j < TN; j++) bf[j] = Bs[k][tx*TN+j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += af[i] * bf[j];
        }
        __syncthreads();
    }

    // Store
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = bm + ty*TM + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = bn + tx*TN + j;
            if (gn < N) {
                float val = alpha * acc[i][j];
                if (beta != 0.f) val += beta * C[gm*ldc+gn];
                C[gm*ldc+gn] = val;
            }
        }
    }
}

// ===========================================================================
// Tier 1: Large kernel — 128×128 with cp.async + float4 (= v4)
// ===========================================================================
// Imported from sgemm_v4.cuh — the best kernel for large shapes.
// Already defined in nnr_sgemm_v4 namespace.

// ===========================================================================
// Single-launch Split-K (3D grid, no cudaMalloc)
// ===========================================================================
// For medium shapes with few M×N blocks but large K.
// grid.z = splitk, each z-slice computes partial sum, then atomicAdd to output.

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_splitk_atomic_kernel(
    const int M, const int N, const int K, const int splitk,
    const float alpha,
    const float* __restrict__ A, const int lda,
    const float* __restrict__ B, const int ldb,
    const float beta,
    float* __restrict__ C, const int ldc)
{
    constexpr int TPB_X = BN / TN;
    constexpr int TPB_Y = BM / TM;
    constexpr int NUM_T = TPB_X * TPB_Y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TPB_X + tx;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    // K-slice for this z-block
    const int slice = (K + splitk - 1) / splitk;
    const int k_start = blockIdx.z * slice;
    const int k_end = min(k_start + slice, K);
    if (k_start >= K) return;

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    float acc[TM][TN] = {};
    const int slice_tiles = (k_end - k_start + BK - 1) / BK;

    for (int tile = 0; tile < slice_tiles; tile++) {
        const int k_base = k_start + tile * BK;

        for (int idx = tid; idx < BM * BK; idx += NUM_T) {
            int am = idx / BK, ak = idx % BK;
            int gm = bm + am, gk = k_base + ak;
            As[ak][am] = (gm < M && gk < k_end) ? A[gm*lda+gk] : 0.f;
        }
        for (int idx = tid; idx < BK * BN; idx += NUM_T) {
            int bk = idx / BN, bni = idx % BN;
            int gk = k_base + bk, gn = bn + bni;
            Bs[bk][bni] = (gk < k_end && gn < N) ? B[gk*ldb+gn] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float af[TM], bf[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) af[i] = As[k][ty*TM+i];
            #pragma unroll
            for (int j = 0; j < TN; j++) bf[j] = Bs[k][tx*TN+j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += af[i] * bf[j];
        }
        __syncthreads();
    }

    // Atomic add to output (all splitk slices contribute)
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = bm + ty*TM + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = bn + tx*TN + j;
            if (gn < N) {
                float val = alpha * acc[i][j];
                if (splitk == 1) {
                    if (beta != 0.f) val += beta * C[gm*ldc+gn];
                    C[gm*ldc+gn] = val;
                } else {
                    atomicAdd(&C[gm*ldc+gn], val);
                }
            }
        }
    }
}

// ===========================================================================
// Dispatcher
// ===========================================================================

inline void sgemm(
    int M, int N, int K,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc,
    cudaStream_t stream = 0)
{
    // Tier 3: GEMV for tiny M
    if (M <= 16) {
        constexpr int BN = 256;
        dim3 grid((N + BN - 1) / BN, M);
        dim3 block(BN);
        sgemv_kernel<BN><<<grid, block, 0, stream>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // Compute block counts for different tile sizes
    auto nblocks = [](int M, int N, int bm, int bn) {
        return ((M+bm-1)/bm) * ((N+bn-1)/bn);
    };

    const int b128 = nblocks(M, N, 128, 128);
    const int b64  = nblocks(M, N, 64, 64);

    // If 128×128 gives enough blocks (≥ 32), use v4 (Tier 1)
    if (b128 >= 32) {
        nnr_sgemm_v4::sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, stream);
        return;
    }

    // Try split-K with 64×64 tile to increase parallelism
    if (b64 < 64 && K >= 256) {
        int splitk = 1;
        while (b64 * splitk < 64 && splitk * 2 <= K / 16)
            splitk *= 2;
        if (splitk > 1) {
            // Clear C first (atomicAdd needs zero-initialized output)
            if (beta == 0.f)
                cudaMemsetAsync(C, 0, (size_t)M * N * sizeof(float), stream);
            // else: beta != 0 with splitk is tricky — first slice applies beta, rest atomicAdd
            // For simplicity, only use splitk when beta == 0
            if (beta == 0.f) {
                dim3 grid((N+63)/64, (M+63)/64, splitk);
                dim3 block(64/4, 64/4);  // 16×16 = 256
                sgemm_splitk_atomic_kernel<64,64,16,4,4><<<grid, block, 0, stream>>>(
                    M, N, K, splitk, alpha, A, lda, B, ldb, 0.f, C, ldc);
                return;
            }
        }
    }

    // Tier 2: 64×64 medium kernel with float4
    if (b64 > b128 * 2) {
        dim3 grid((N+63)/64, (M+63)/64);
        dim3 block(64/4, 64/4);
        sgemm_medium_kernel<64,64,16,4,4><<<grid, block, 0, stream>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // Fallback: v4 (128×128) — still best even with few blocks
    nnr_sgemm_v4::sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

} // namespace nnr_gemm
