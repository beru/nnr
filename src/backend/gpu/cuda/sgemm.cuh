// High-performance FP32 GEMM for Ampere (SM 8.x).
// Row-major: C[M×N] = alpha * A[M×K] × B[K×N] + beta * C[M×N]
//
// Optimizations:
//   1. float4 vectorized global loads (128-bit)
//   2. Register-level prefetching (global → registers → shared, overlaps compute)
//   3. Large register tile TM×TN = 8×8 (64 FMAs per k step)
//   4. Transposed A in shared memory (bank-conflict-free)
//   5. __launch_bounds__ for register allocation control
//
// BM=128, BN=128, BK=16, TM=8, TN=8
//   256 threads (16×16), each 8×8 = 64 outputs
//   Shared memory: (128×16 + 16×128) × 4 = 16 KB
//   Compute per tile: 128 × 128 × 16 × 2 = 524,288 FLOPs

#pragma once
#include <cuda_runtime.h>

namespace nnr_sgemm {

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(256, 2)
void sgemm_kernel(
    const int M, const int N, const int K,
    const float alpha,
    const float* __restrict__ A, const int lda,
    const float* __restrict__ B, const int ldb,
    const float beta,
    float* __restrict__ C, const int ldc)
{
    constexpr int TPB_X = BN / TN;  // 16
    constexpr int TPB_Y = BM / TM;  // 16
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TPB_X + tx;

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    // Shared memory: A transposed [BK][BM], B normal [BK][BN]
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    // Accumulators
    float acc[TM][TN] = {};

    // ---- Load index precomputation ----
    // A: BM×BK = 128×16 = 2048 floats → 512 float4 → 256 threads × 2 loads each
    // B: BK×BN = 16×128 = 2048 floats → 512 float4 → 256 threads × 2 loads each
    constexpr int A_FLOAT4_PER_ROW = BK / 4;        // 4
    constexpr int A_TOTAL_FLOAT4 = BM * A_FLOAT4_PER_ROW;  // 512
    constexpr int A_LOADS_PER_THREAD = A_TOTAL_FLOAT4 / 256; // 2

    constexpr int B_FLOAT4_PER_ROW = BN / 4;        // 32
    constexpr int B_TOTAL_FLOAT4 = BK * B_FLOAT4_PER_ROW;  // 512
    constexpr int B_LOADS_PER_THREAD = B_TOTAL_FLOAT4 / 256; // 2

    // Registers for prefetching global → registers (then write to shared)
    float4 a_prefetch[A_LOADS_PER_THREAD];
    float4 b_prefetch[B_LOADS_PER_THREAD];

    const int num_tiles = (K + BK - 1) / BK;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int k_base = tile * BK;

        // ---- Global → Registers (prefetch) ----
        #pragma unroll
        for (int ld = 0; ld < A_LOADS_PER_THREAD; ld++) {
            int flat = tid + ld * 256;
            int a_m = flat / A_FLOAT4_PER_ROW;
            int a_k4 = (flat % A_FLOAT4_PER_ROW) * 4;
            int gm = bm + a_m;
            int gk = k_base + a_k4;
            {
                const float* src_a = &A[gm * lda + gk];
                bool aa = (reinterpret_cast<uintptr_t>(src_a) & 15) == 0;
                if (gm < M && gk + 3 < K && aa)
                    a_prefetch[ld] = __ldg(reinterpret_cast<const float4*>(src_a));
                else {
                    float tmp[4];
                    for (int i = 0; i < 4; i++)
                        tmp[i] = (gm < M && gk + i < K) ? A[gm * lda + gk + i] : 0.0f;
                    a_prefetch[ld] = *reinterpret_cast<float4*>(tmp);
                }
            }
        }

        #pragma unroll
        for (int ld = 0; ld < B_LOADS_PER_THREAD; ld++) {
            int flat = tid + ld * 256;
            int b_k = flat / B_FLOAT4_PER_ROW;
            int b_n4 = (flat % B_FLOAT4_PER_ROW) * 4;
            int gk = k_base + b_k;
            int gn = bn + b_n4;
            {
                const float* src_b = &B[gk * ldb + gn];
                bool ba = (reinterpret_cast<uintptr_t>(src_b) & 15) == 0;
                if (gk < K && gn + 3 < N && ba)
                    b_prefetch[ld] = __ldg(reinterpret_cast<const float4*>(src_b));
                else {
                    float tmp[4];
                    for (int i = 0; i < 4; i++)
                        tmp[i] = (gk < K && gn + i < N) ? B[gk * ldb + gn + i] : 0.0f;
                    b_prefetch[ld] = *reinterpret_cast<float4*>(tmp);
                }
            }
        }

        // ---- Registers → Shared (A transposed) ----
        #pragma unroll
        for (int ld = 0; ld < A_LOADS_PER_THREAD; ld++) {
            int flat = tid + ld * 256;
            int a_m = flat / A_FLOAT4_PER_ROW;
            int a_k4 = (flat % A_FLOAT4_PER_ROW) * 4;
            As[a_k4 + 0][a_m] = a_prefetch[ld].x;
            As[a_k4 + 1][a_m] = a_prefetch[ld].y;
            As[a_k4 + 2][a_m] = a_prefetch[ld].z;
            As[a_k4 + 3][a_m] = a_prefetch[ld].w;
        }

        #pragma unroll
        for (int ld = 0; ld < B_LOADS_PER_THREAD; ld++) {
            int flat = tid + ld * 256;
            int b_k = flat / B_FLOAT4_PER_ROW;
            int b_n4 = (flat % B_FLOAT4_PER_ROW) * 4;
            *reinterpret_cast<float4*>(&Bs[b_k][b_n4]) = b_prefetch[ld];
        }

        __syncthreads();

        // ---- Compute: BK outer products ----
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_frag[TM], b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_frag[i] = As[k][ty * TM + i];

            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_frag[j] = Bs[k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }

        __syncthreads();
    }

    // ---- Store ----
    const int c_m = bm + ty * TM;
    const int c_n = bn + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const int gm = c_m + i;
        if (gm >= M) continue;

        float* row = &C[gm * ldc + c_n];
        bool c_aligned = (reinterpret_cast<uintptr_t>(row) & 15) == 0;
        if (c_n + TN <= N && beta == 0.0f && c_aligned) {
            float4 v0 = make_float4(
                alpha * acc[i][0], alpha * acc[i][1],
                alpha * acc[i][2], alpha * acc[i][3]);
            float4 v1 = make_float4(
                alpha * acc[i][4], alpha * acc[i][5],
                alpha * acc[i][6], alpha * acc[i][7]);
            *reinterpret_cast<float4*>(&row[0]) = v0;
            *reinterpret_cast<float4*>(&row[4]) = v1;
        } else {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int gn = c_n + j;
                if (gn < N) {
                    float val = alpha * acc[i][j];
                    if (beta != 0.0f) val += beta * C[gm * ldc + gn];
                    C[gm * ldc + gn] = val;
                }
            }
        }
    }
}

inline void sgemm(
    int M, int N, int K,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc,
    cudaStream_t stream = 0)
{
    constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);
    sgemm_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

} // namespace nnr_sgemm
