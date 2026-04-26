// Optimized GEMM kernels for small/medium matrix sizes.
// Same techniques as v4 (float4 loads, register prefetch, cp.async for B)
// but with smaller tile sizes to maximize SM occupancy for small M or N.
//
// Tile configs:
//   64×64:  for M or N in [33..128], 256 threads, TM=4 TN=4
//   64×128: for M < 128, N >= 128 (tall-ish)
//   128×64: for M >= 128, N < 128
//   32×32:  for M or N < 32, 64 threads, TM=4 TN=4

#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

namespace nnr_sgemm_small {

// ---- Unified optimized kernel for any tile size ----

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_kernel(
    const int M, const int N, const int K,
    const float alpha,
    const float* __restrict__ A, const int lda,
    const float* __restrict__ B, const int ldb,
    const float beta,
    float* __restrict__ C, const int ldc)
{
    constexpr int TPB_X = BN / TN;
    constexpr int TPB_Y = BM / TM;
    constexpr int NUM_THREADS = TPB_X * TPB_Y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TPB_X + tx;

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    // Double-buffered shared memory: A transposed [k][m], B normal [k][n]
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float acc[TM][TN] = {};

    // ---- Load work distribution ----
    constexpr int A_TOTAL = BM * BK;
    constexpr int B_TOTAL = BK * BN;
    // Each thread loads ceil(total / num_threads) elements
    // Use loops for flexibility across tile sizes

    // Register buffers for A prefetch (transposed store)
    constexpr int A_PER_THREAD = (A_TOTAL + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int B_PER_THREAD = (B_TOTAL + NUM_THREADS - 1) / NUM_THREADS;
    float a_regs[A_PER_THREAD];
    float b_regs[B_PER_THREAD];

    // Simplified load: no cp.async for small tiles (overhead not worth it)

    // ---- Main loop ----
    const int num_tiles = (K + BK - 1) / BK;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int k_base = tile * BK;

        // Load A transposed: global [m][k] → shared [k][m]
        for (int i = 0; i < A_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < A_TOTAL) {
                int am = idx / BK;
                int ak = idx % BK;
                int gm = bm + am;
                int gk = k_base + ak;
                As[0][ak][am] = (gm < M && gk < K) ? A[gm * lda + gk] : 0.f;
            }
        }

        // Load B: global [k][n] → shared [k][n]
        for (int i = 0; i < B_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < B_TOTAL) {
                int bk = idx / BN;
                int bn_i = idx % BN;
                int gk = k_base + bk;
                int gn = bn + bn_i;
                Bs[0][bk][bn_i] = (gk < K && gn < N) ? B[gk * ldb + gn] : 0.f;
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float af[TM], bf[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) af[i] = As[0][k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) bf[j] = Bs[0][k][tx * TN + j];
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
        int gm = bm + ty * TM + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = bn + tx * TN + j;
            if (gn < N) {
                float val = alpha * acc[i][j];
                if (beta != 0.f) val += beta * C[gm * ldc + gn];
                C[gm * ldc + gn] = val;
            }
        }
    }
}

// ---- Launch helpers ----

template <int BM, int BN, int BK, int TM, int TN>
inline void launch(int M, int N, int K, float alpha,
    const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc, cudaStream_t stream)
{
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);
    sgemm_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Config: 64×64, BK=16, TM=4, TN=4 → 16×16 = 256 threads
inline void sgemm_64x64(int M, int N, int K, float alpha,
    const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc, cudaStream_t s = 0)
{
    launch<64, 64, 16, 4, 4>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
}

// Config: 32×32, BK=16, TM=4, TN=4 → 8×8 = 64 threads
inline void sgemm_32x32(int M, int N, int K, float alpha,
    const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc, cudaStream_t s = 0)
{
    launch<32, 32, 16, 4, 4>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
}

// Config: 64×128, BK=16, TM=4, TN=8 → 16×16 = 256 threads
inline void sgemm_64x128(int M, int N, int K, float alpha,
    const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc, cudaStream_t s = 0)
{
    launch<64, 128, 16, 4, 8>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
}

// Config: 128×64, BK=16, TM=8, TN=4 → 16×16 = 256 threads
inline void sgemm_128x64(int M, int N, int K, float alpha,
    const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc, cudaStream_t s = 0)
{
    launch<128, 64, 16, 8, 4>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, s);
}

} // namespace nnr_sgemm_small
