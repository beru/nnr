// NNR CUDA GEMM micro-kernel — shared memory tiled with register blocking and fused epilogue.
//
// Row-major C[M×N] = A[M×K] × B[K×N] + bias[N] + post_op
//
// Template parameters:
//   BM, BN — thread block tile size (shared memory)
//   TM, TN — register tile per thread (work-per-thread)
//
// Thread block: (BN/TN, BM/TM) threads
// Grid: ((N+BN-1)/BN, (M+BM-1)/BM)
//
// Designed for NVRTC compilation — M, N, K can be baked as constants
// for better register allocation.

#pragma once

#include <cuda_runtime.h>

// Post-op type — fused into GEMM epilogue to avoid separate kernel launch
enum class GemmPostOp : int {
    NONE = 0,
    RELU,
    CLIP,     // requires clip_min, clip_max
    SIGMOID,
};

// GEMM kernel parameters passed via push-constant / kernel args
struct GemmParams {
    int M, N, K;
    const float* __restrict__ A;    // [M × K] row-major
    int lda;                        // leading dim of A (typically K)
    const float* __restrict__ B;    // [K × N] row-major
    int ldb;                        // leading dim of B (typically N)
    float* __restrict__ C;          // [M × N] row-major
    int ldc;                        // leading dim of C (typically N)
    float alpha;
    float beta;
    const float* __restrict__ bias; // [N] per-column bias, or nullptr
    GemmPostOp post_op;
    float clip_min, clip_max;       // only used when post_op == CLIP
};

// --- The kernel ---

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_kernel(GemmParams p)
{
    const int M = p.M, N = p.N, K = p.K;

    // Thread position within block
    const int tx = threadIdx.x;  // 0..BN/TN-1
    const int ty = threadIdx.y;  // 0..BM/TM-1

    // Global row/col base for this thread's register tile
    const int row_base = blockIdx.y * BM + ty * TM;
    const int col_base = blockIdx.x * BN + tx * TN;

    // Shared memory for A and B tiles
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Register accumulators — TM × TN per thread
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    // Number of K-tiles
    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; t++) {
        // Cooperatively load A tile [BM × BK] from global to shared
        // Each thread loads multiple elements to cover the tile
        {
            const int total_A = BM * BK;
            constexpr int threads_per_block = (BM / TM) * (BN / TN);
            const int tid = ty * (BN / TN) + tx;
            for (int idx = tid; idx < total_A; idx += threads_per_block) {
                const int sm = idx / BK;
                const int sk = idx % BK;
                const int gm = blockIdx.y * BM + sm;
                const int gk = t * BK + sk;
                As[sm][sk] = (gm < M && gk < K) ? p.A[gm * p.lda + gk] : 0.0f;
            }
        }

        // Cooperatively load B tile [BK × BN] from global to shared
        {
            const int total_B = BK * BN;
            constexpr int threads_per_block = (BM / TM) * (BN / TN);
            const int tid = ty * (BN / TN) + tx;
            for (int idx = tid; idx < total_B; idx += threads_per_block) {
                const int sk = idx / BN;
                const int sn = idx % BN;
                const int gk = t * BK + sk;
                const int gn = blockIdx.x * BN + sn;
                Bs[sk][sn] = (gk < K && gn < N) ? p.B[gk * p.ldb + gn] : 0.0f;
            }
        }

        __syncthreads();

        // Compute TM × TN partial products from shared memory
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A column into registers
            float a_reg[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = As[ty * TM + i][k];

            // Load B row into registers
            float b_reg[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = Bs[k][tx * TN + j];

            // Outer product
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    // --- Epilogue: scale, bias, post-op, store ---

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const int gm = row_base + i;
        if (gm >= M) continue;

        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int gn = col_base + j;
            if (gn >= N) continue;

            float val = p.alpha * acc[i][j];

            // Beta * existing C
            if (p.beta != 0.0f)
                val += p.beta * p.C[gm * p.ldc + gn];

            // Bias (per-column)
            if (p.bias)
                val += p.bias[gn];

            // Fused post-op
            switch (p.post_op) {
            case GemmPostOp::RELU:
                val = fmaxf(val, 0.0f);
                break;
            case GemmPostOp::CLIP:
                val = fminf(fmaxf(val, p.clip_min), p.clip_max);
                break;
            case GemmPostOp::SIGMOID:
                val = 1.0f / (1.0f + expf(-val));
                break;
            default:
                break;
            }

            p.C[gm * p.ldc + gn] = val;
        }
    }
}

// --- Launch helper ---

// Default tile sizes tuned for Ampere (RTX 3090, SM 8.6):
//   BM=64, BN=64, BK=16, TM=4, TN=4
//   → 16×16 = 256 threads per block, each computing 4×4 = 16 outputs
//   → 64*64 = 4096 outputs per block
//
// For large GEMM (M,N >= 2048): BM=128, BN=128, BK=16, TM=8, TN=8
//   → 16×16 = 256 threads, each computing 8×8 = 64 outputs
//   → 128*128 = 16384 outputs per block

inline void launch_gemm_kernel(GemmParams& p, cudaStream_t stream)
{
    if (p.M >= 2048 || p.N >= 2048) {
        // Large GEMM
        constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
        dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);
        dim3 block(BN / TN, BM / TM);  // 16×16 = 256
        gemm_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(p);
    } else {
        // Default
        constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
        dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);
        dim3 block(BN / TN, BM / TM);  // 16×16 = 256
        gemm_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(p);
    }
}
