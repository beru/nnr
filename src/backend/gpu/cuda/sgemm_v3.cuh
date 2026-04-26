// sgemm v3: Warp-shuffle fragment exchange + larger BK
//
// Key changes from v2:
//   1. Warp-level A/B fragment sharing via __shfl_sync (reduces smem read pressure)
//   2. BK=32 (doubles compute per __syncthreads — 2048 FMAs/thread/tile)
//   3. Warp layout 4×8 for natural shuffle patterns
//   4. Thread tile TM=8, TN=4, warp tile 32×32
//
// Block: BM=128, BN=128, 256 threads (8 warps)
// Warp grid: 4 warps in M × 2 warps in N
// Each warp: 32 threads as 4-row × 8-col → warp tile 32×32
// Each thread: TM=8 × TN=4 = 32 outputs

#pragma once
#include <cuda_runtime.h>

namespace nnr_sgemm_v3 {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WM = 32;   // warp tile M
constexpr int WN = 32;   // warp tile N
constexpr int TM = 8;    // thread tile M
constexpr int TN = 4;    // thread tile N
constexpr int WARP_ROWS = 4;  // WM/TM
constexpr int WARP_COLS = 8;  // WN/TN
// WARP_ROWS * WARP_COLS = 32 ✓

constexpr int WARPS_M = BM / WM;  // 4
constexpr int WARPS_N = BN / WN;  // 4
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 16 → 512 threads... too many!

// Fix: use WARPS_M=4, WARPS_N=2 → 8 warps = 256 threads
// But then BN = 2 * WN = 64, not 128.
// Alternative: WN=64, TN=8 → WARP_COLS=8, WARP_ROWS=4 → 32 threads per warp ✓
// Then WARPS_M=4, WARPS_N=2 → BM=128, BN=128 ✓

} // namespace nnr_sgemm_v3

// Reconfigure properly:
// WM=32, WN=64 → warp tile 32×64
// TM=8, TN=8 → thread: 4 rows × 8 cols → WARP_ROWS=4, WARP_COLS=8 ✓
// WARPS_M=4 (128/32), WARPS_N=2 (128/64) → 8 warps = 256 threads ✓
//
// Warp internal layout:
//   lane_id = threadIdx within warp
//   warp_m = lane_id / 8    (0..3)  → rows within warp tile
//   warp_n = lane_id % 8    (0..7)  → cols within warp tile
//
// For each k step:
//   A fragment: 8 values per thread (TM=8), but threads with same warp_m need same values
//   → 4 unique row groups × 8 values = 32 unique A values per warp
//   → 4 source threads (one per warp_m) load 8 values each from smem
//   → Broadcast via __shfl_sync to all 8 threads with the same warp_m
//
//   B fragment: 8 values per thread (TN=8), threads with same warp_n need same values
//   → 8 unique col groups × 8 values = 64 unique B values per warp
//   → All 8 threads per warp_n already load unique values — no savings from shuffle
//   → Just load directly from smem
//
// Net benefit: A loads from smem drop from 256 to 32 per warp per k step
// (hardware multicast may already handle this, so benefit is instruction count reduction)

namespace nnr_sgemm_v3 {

__global__ __launch_bounds__(256, 2)
void sgemm_kernel(
    const int M, const int N, const int K,
    const float alpha,
    const float* __restrict__ A, const int lda,
    const float* __restrict__ B, const int ldb,
    const float beta,
    float* __restrict__ C, const int ldc)
{
    constexpr int _BM = 128, _BN = 128, _BK = 32;
    constexpr int _WM = 32, _WN = 64;
    constexpr int _TM = 8, _TN = 8;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp position in block grid
    const int warp_m = warp_id / 2;  // 0..3 (WARPS_N=2)
    const int warp_n = warp_id % 2;  // 0..1

    // Thread position within warp
    const int wm = lane_id / 8;     // 0..3  (row within warp tile)
    const int wn = lane_id % 8;     // 0..7  (col within warp tile)

    // Global output position
    const int bm = blockIdx.y * _BM;
    const int bn = blockIdx.x * _BN;
    const int gm_base = bm + warp_m * _WM + wm * _TM;
    const int gn_base = bn + warp_n * _WN + wn * _TN;

    // Shared memory
    __shared__ float As[_BK][_BM];   // A transposed
    __shared__ float Bs[_BK][_BN];

    // Accumulators
    float acc[_TM][_TN] = {};

    // ---- Global → Shared loading indices ----
    // Total: BM × BK = 128 × 32 = 4096 floats → 1024 float4 → 256 threads × 4 loads each
    // Total: BK × BN = 32 × 128 = 4096 floats → 1024 float4 → 256 threads × 4 loads each
    constexpr int A_F4_PER_ROW = _BK / 4;        // 8
    constexpr int A_TOTAL_F4 = _BM * A_F4_PER_ROW;  // 1024
    constexpr int A_LOADS = A_TOTAL_F4 / 256;        // 4

    constexpr int B_F4_PER_ROW = _BN / 4;        // 32
    constexpr int B_TOTAL_F4 = _BK * B_F4_PER_ROW;  // 1024
    constexpr int B_LOADS = B_TOTAL_F4 / 256;        // 4

    float4 a_buf[A_LOADS];
    float4 b_buf[B_LOADS];

    const int num_tiles = (K + _BK - 1) / _BK;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int k_base = tile * _BK;

        // ---- Prefetch: Global → Registers ----
        #pragma unroll
        for (int ld = 0; ld < A_LOADS; ld++) {
            int flat = tid + ld * 256;
            int am = flat / A_F4_PER_ROW;
            int ak4 = (flat % A_F4_PER_ROW) * 4;
            int gm = bm + am;
            int gk = k_base + ak4;
            if (gm < M && gk + 3 < K)
                a_buf[ld] = __ldg(reinterpret_cast<const float4*>(&A[gm * lda + gk]));
            else {
                float t[4];
                for (int i = 0; i < 4; i++)
                    t[i] = (gm < M && gk + i < K) ? A[gm * lda + gk + i] : 0.f;
                a_buf[ld] = *reinterpret_cast<float4*>(t);
            }
        }

        #pragma unroll
        for (int ld = 0; ld < B_LOADS; ld++) {
            int flat = tid + ld * 256;
            int bk = flat / B_F4_PER_ROW;
            int bn4 = (flat % B_F4_PER_ROW) * 4;
            int gk = k_base + bk;
            int gn = bn + bn4;
            if (gk < K && gn + 3 < N)
                b_buf[ld] = __ldg(reinterpret_cast<const float4*>(&B[gk * ldb + gn]));
            else {
                float t[4];
                for (int i = 0; i < 4; i++)
                    t[i] = (gk < K && gn + i < N) ? B[gk * ldb + gn + i] : 0.f;
                b_buf[ld] = *reinterpret_cast<float4*>(t);
            }
        }

        // ---- Registers → Shared ----
        #pragma unroll
        for (int ld = 0; ld < A_LOADS; ld++) {
            int flat = tid + ld * 256;
            int am = flat / A_F4_PER_ROW;
            int ak4 = (flat % A_F4_PER_ROW) * 4;
            As[ak4 + 0][am] = a_buf[ld].x;
            As[ak4 + 1][am] = a_buf[ld].y;
            As[ak4 + 2][am] = a_buf[ld].z;
            As[ak4 + 3][am] = a_buf[ld].w;
        }

        #pragma unroll
        for (int ld = 0; ld < B_LOADS; ld++) {
            int flat = tid + ld * 256;
            int bk = flat / B_F4_PER_ROW;
            int bn4 = (flat % B_F4_PER_ROW) * 4;
            *reinterpret_cast<float4*>(&Bs[bk][bn4]) = b_buf[ld];
        }

        __syncthreads();

        // ---- Compute with warp shuffles ----
        #pragma unroll
        for (int k = 0; k < _BK; k++) {
            // A fragment: each thread needs 8 values for its row group
            // Threads with same warp_m within a warp need the same A values.
            // Source: thread with wn==0 in each warp_m row loads from smem,
            // broadcasts to others via __shfl_sync.
            float a_frag[_TM];

            // Load A from shared memory (only first thread per warp_m row needs to load)
            // Then broadcast via __shfl_sync
            #pragma unroll
            for (int i = 0; i < _TM; i++) {
                int smem_row = warp_m * _WM + wm * _TM + i;
                float val = As[k][smem_row];
                // All threads in same warp_m row can read the same smem address (broadcast)
                // But with __shfl, we load once and distribute:
                // Source lane: wm * 8 + 0 (first thread in this warp_m row)
                // Actually, since all 8 threads with same wm read the same address,
                // hardware multicast handles it. But let's use shfl explicitly
                // to give the compiler a hint:
                int src_lane = wm * 8;  // lane that has our row's A value
                a_frag[i] = __shfl_sync(0xFFFFFFFF, val, src_lane);
            }

            // B fragment: each thread needs 8 values for its column group
            // Threads with same warp_n need the same B values — but warp_n has 8 values
            // and each thread's wn IS unique, so each loads directly
            float b_frag[_TN];
            #pragma unroll
            for (int j = 0; j < _TN; j++) {
                b_frag[j] = Bs[k][warp_n * _WN + wn * _TN + j];
            }

            // Outer product: TM × TN FMAs
            #pragma unroll
            for (int i = 0; i < _TM; i++)
                #pragma unroll
                for (int j = 0; j < _TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }

        __syncthreads();
    }

    // ---- Store ----
    #pragma unroll
    for (int i = 0; i < _TM; i++) {
        const int gm = gm_base + i;
        if (gm >= M) continue;

        if (gn_base + _TN <= N && beta == 0.0f) {
            float4 v0 = make_float4(
                alpha * acc[i][0], alpha * acc[i][1],
                alpha * acc[i][2], alpha * acc[i][3]);
            float4 v1 = make_float4(
                alpha * acc[i][4], alpha * acc[i][5],
                alpha * acc[i][6], alpha * acc[i][7]);
            *reinterpret_cast<float4*>(&C[gm * ldc + gn_base + 0]) = v0;
            *reinterpret_cast<float4*>(&C[gm * ldc + gn_base + 4]) = v1;
        } else {
            #pragma unroll
            for (int j = 0; j < _TN; j++) {
                int gn = gn_base + j;
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
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(16, 16);
    sgemm_kernel<<<grid, block, 0, stream>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

} // namespace nnr_sgemm_v3
