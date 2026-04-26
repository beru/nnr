// sgemm v4: cp.async DMA + double-buffered shared memory
//
// Key change: Global → Shared via hardware DMA (cp.async), not registers.
// CUDA cores are 100% free for FMA during memory transfers.
// Requires SM 8.0+ (Ampere).
//
// BM=128, BN=128, BK=16, TM=8, TN=8
// Double-buffered shared memory: compute on buf[i] while DMA fills buf[1-i]

#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

namespace nnr_sgemm_v4 {

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
    constexpr int TPB_X = BN / TN;   // 16
    constexpr int TPB_Y = BM / TM;   // 16
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TPB_X + tx;

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    // Double-buffered shared memory
    // A: TRANSPOSED [k][m] — bank-conflict-free sequential reads along m
    //    Loaded via registers (transpose during store)
    // B: [k][n] — loaded via cp.async DMA (contiguous, no register usage)
    __shared__ float As[2][BK][BM];   // A transposed
    __shared__ float Bs[2][BK][BN];   // B normal

    float acc[TM][TN] = {};

    // --- Load addressing ---
    // A tile: BM × BK = 128 × 16 = 2048 floats = 512 float4
    // 256 threads → 2 float4 per thread
    constexpr int A_F4_PER_ROW = BK / 4;              // 4
    constexpr int A_LOADS = (BM * A_F4_PER_ROW) / 256; // 2

    // B tile: BK × BN = 16 × 128 = 2048 floats = 512 float4
    constexpr int B_F4_PER_ROW = BN / 4;              // 32
    constexpr int B_LOADS = (BK * B_F4_PER_ROW) / 256; // 2

    const int num_tiles = (K + BK - 1) / BK;
    int buf = 0;

    // ---- Issue cp.async for tile 0 into buffer 0 ----
    // Hybrid: A through registers (transposed), B via cp.async DMA
    float4 a_buf[A_LOADS];  // register staging for A transpose

    auto issue_load = [&](int tile, int b) {
        const int k_base = tile * BK;

        // A: global → registers (float4 vectorized)
        #pragma unroll
        for (int ld = 0; ld < A_LOADS; ld++) {
            int flat = tid + ld * 256;
            int am = flat / A_F4_PER_ROW;
            int ak4 = (flat % A_F4_PER_ROW) * 4;
            int gm = bm + am;
            int gk = k_base + ak4;
            {
                const float* src = &A[gm * lda + gk];
                bool a_aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
                if (gm < M && gk + 3 < K && a_aligned)
                    a_buf[ld] = __ldg(reinterpret_cast<const float4*>(src));
                else {
                    float t[4];
                    for (int i = 0; i < 4; i++)
                        t[i] = (gm < M && gk + i < K) ? A[gm * lda + gk + i] : 0.f;
                    a_buf[ld] = *reinterpret_cast<float4*>(t);
                }
            }
        }

        // B: cp.async DMA — global → shared directly, no register usage
        #pragma unroll
        for (int ld = 0; ld < B_LOADS; ld++) {
            int flat = tid + ld * 256;
            int bk = flat / B_F4_PER_ROW;
            int bn4 = (flat % B_F4_PER_ROW) * 4;
            int gk = k_base + bk;
            int gn = bn + bn4;
            float* dst = &Bs[b][bk][bn4];
            const float* src = &B[gk * ldb + gn];
            // cp.async requires 16-byte aligned source. Check alignment.
            bool aligned = (reinterpret_cast<uintptr_t>(src) & 15) == 0;
            if (gk < K && gn + 3 < N && aligned)
                __pipeline_memcpy_async(dst, src, 16);
            else
                for (int i = 0; i < 4; i++)
                    dst[i] = (gk < K && gn + i < N) ? src[i] : 0.f;
        }
        __pipeline_commit();
    };

    auto store_a_transposed = [&](int b) {
        // Registers → Shared (transposed: As[k][m])
        #pragma unroll
        for (int ld = 0; ld < A_LOADS; ld++) {
            int flat = tid + ld * 256;
            int am = flat / A_F4_PER_ROW;
            int ak4 = (flat % A_F4_PER_ROW) * 4;
            As[b][ak4 + 0][am] = a_buf[ld].x;
            As[b][ak4 + 1][am] = a_buf[ld].y;
            As[b][ak4 + 2][am] = a_buf[ld].z;
            As[b][ak4 + 3][am] = a_buf[ld].w;
        }
    };

    // ---- Load first tile ----
    issue_load(0, 0);
    __pipeline_wait_prior(0);  // wait for B's cp.async
    store_a_transposed(0);      // write A registers → shared (transposed)
    __syncthreads();

    for (int tile = 0; tile < num_tiles; tile++) {
        int next_buf = 1 - buf;

        // Issue loads for next tile (A → registers, B → shared via DMA)
        // These overlap with the compute below.
        if (tile + 1 < num_tiles) {
            issue_load(tile + 1, next_buf);
        }

        // ---- Compute with double-buffered fragments (CUTLASS technique) ----
        // While computing outer product on frag[p], load next frag[1-p] from smem.
        // This overlaps smem reads with FMA within a single warp.
        float a_frag[2][TM], b_frag[2][TN];

        // Preload k=0 fragments
        #pragma unroll
        for (int i = 0; i < TM; i++)
            a_frag[0][i] = As[buf][0][ty * TM + i];
        #pragma unroll
        for (int j = 0; j < TN; j++)
            b_frag[0][j] = Bs[buf][0][tx * TN + j];

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load next k's fragments (into the other buffer) while computing current
            if (k + 1 < BK) {
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    a_frag[(k+1)%2][i] = As[buf][k+1][ty * TM + i];
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    b_frag[(k+1)%2][j] = Bs[buf][k+1][tx * TN + j];
            }

            // Outer product on current fragments
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[k%2][i] * b_frag[k%2][j];
        }

        // Wait for next tile's B DMA + store next A transposed
        if (tile + 1 < num_tiles) {
            __pipeline_wait_prior(0);
            store_a_transposed(next_buf);
        }
        __syncthreads();

        buf = next_buf;
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

} // namespace nnr_sgemm_v4

// Extern linkage for dispatch
namespace nnr_sgemm_best {
inline void sgemm_v4_launch(int M, int N, int K, float alpha,
    const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc, cudaStream_t stream) {
    nnr_sgemm_v4::sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}
}
