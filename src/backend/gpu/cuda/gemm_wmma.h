#pragma once
// WMMA TensorCore GEMM kernels for the NVRTC module.
// Ampere (sm_80+) uses TF32 tensor cores for f32 → f32 matmul:
//   fragment shape M=16, N=16, K=8, precision::tf32, accumulate in float.
// A kernel per (transA, transB) layout pair + a scalar fallback for any shape.

#if defined(NNR_USE_CUDA)

namespace nnr::gpu {

inline const char* gemm_source() {
    return R"CUDA(
#include <mma.h>
#include <cuda_pipeline_primitives.h>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Scalar fallback — shared-memory tiled. Any M, N, K; any transpose flags.
// Block: 16×16 threads = 256 per block. Tile: 16×16 output, K-step 16.
// Each thread: 1 output element; cooperatively loads 1 A + 1 B per K-step
// into __shared__; dot product across K-tile from shared (K × 2 loads/thread
// per K-tile vs K × 2 global loads in naive kernel).
// ---------------------------------------------------------------------------
extern "C" __global__
void gemm_scalar_f32(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K,
                     int ta, int tb)
{
    constexpr int TS = 16;
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TS + ty;
    int col = blockIdx.x * TS + tx;

    float acc = 0.f;
    int k_tiles = (K + TS - 1) / TS;
    for (int t = 0; t < k_tiles; ++t) {
        int a_col = t * TS + tx;
        int b_row = t * TS + ty;
        // Load A[row, a_col] into As[ty][tx], zero-pad on out-of-range.
        if (row < M && a_col < K) {
            As[ty][tx] = ta ? A[(size_t)a_col * M + row]
                            : A[(size_t)row * K + a_col];
        } else As[ty][tx] = 0.f;
        // Load B[b_row, col] into Bs[ty][tx].
        if (b_row < K && col < N) {
            Bs[ty][tx] = tb ? B[(size_t)col * K + b_row]
                            : B[(size_t)b_row * N + col];
        } else Bs[ty][tx] = 0.f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TS; ++kk)
            acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }

    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// TF32 TensorCore kernel (slow reference) — 1 warp per 16×16 output tile.
// Requires M % 16 == 0, N % 16 == 0, K % 8 == 0.
// Used for tiny/unaligned-for-blocked shapes. No shared mem, no reuse.
// ---------------------------------------------------------------------------
extern "C" __global__
void gemm_tc_tf32(const float* __restrict__ A,
                  const float* __restrict__ B,
                  float* __restrict__ C,
                  int M, int N, int K)
{
    const int m_tile = blockIdx.y * 16;
    const int n_tile = blockIdx.x * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    wmma::fill_fragment(c_frag, 0.f);

    for (int k = 0; k < K; k += 8) {
        wmma::load_matrix_sync(a_frag, A + (size_t)m_tile * K + k, K);
        wmma::load_matrix_sync(b_frag, B + (size_t)k * N + n_tile, N);
        #pragma unroll
        for (int i = 0; i < a_frag.num_elements; ++i)
            a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
        #pragma unroll
        for (int i = 0; i < b_frag.num_elements; ++i)
            b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (size_t)m_tile * N + n_tile, c_frag, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// TF32 TensorCore kernel (blocked, any shape) — 4 warps / 128 threads per
// block, 64×64 output tile, K-chunk 16. Each warp owns a 32×32 subtile via
// a 2×2 grid of 16×16 WMMA fragments.
//
// Handles unaligned M/N/K: A and B loads zero-pad out-of-range elements; store
// goes through a shared-mem scratch with bounds check so partial edge tiles
// work too.
//
// Shared memory: A_tile[64][16] + B_tile[16][64] + C_scratch[64][64] ≈ 24 KB.
// Launch: grid=(ceil(N/64), ceil(M/64)), block=128.
// ---------------------------------------------------------------------------
extern "C" __global__
void gemm_tc_tf32_blocked(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K)
{
    constexpr int BM = 64, BN = 64, BK = 16;

    __shared__ float A_tile [BM][BK];
    __shared__ float B_tile [BK][BN];
    __shared__ float C_scratch[BM][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / 2;
    const int warp_n  = warp_id % 2;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(c_frag[i][j], 0.f);

    for (int k = 0; k < K; k += BK) {
        // Load A[block_m:+BM, k:+BK] into shared. 128 threads × 8 elts = 1024.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int lin = tid * 8 + i;
            int row = lin / BK, col = lin % BK;
            int gr = block_m + row, gc = k + col;
            A_tile[row][col] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.f;
        }
        // Load B[k:+BK, block_n:+BN] into shared.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int lin = tid * 8 + i;
            int row = lin / BN, col = lin % BN;
            int gr = k + row, gc = block_n + col;
            B_tile[row][col] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(a_frag[i], &A_tile[warp_m*32 + i*16][kk], BK);
                #pragma unroll
                for (int e = 0; e < a_frag[i].num_elements; ++e)
                    a_frag[i].x[e] = wmma::__float_to_tf32(a_frag[i].x[e]);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(b_frag[j], &B_tile[kk][warp_n*32 + j*16], BN);
                #pragma unroll
                for (int e = 0; e < b_frag[j].num_elements; ++e)
                    b_frag[j].x[e] = wmma::__float_to_tf32(b_frag[j].x[e]);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        __syncthreads();
    }

    // Store each fragment to C_scratch (in shared), then copy to global with
    // per-element bounds check.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int sm_row = warp_m * 32 + i * 16;
            int sm_col = warp_n * 32 + j * 16;
            wmma::store_matrix_sync(&C_scratch[sm_row][sm_col],
                                    c_frag[i][j], BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // 128 threads × 32 elts = 4096 = BM*BN.
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int lin = tid * 32 + i;
        int row = lin / BN, col = lin % BN;
        int gr = block_m + row, gc = block_n + col;
        if (gr < M && gc < N) C[(size_t)gr * N + gc] = C_scratch[row][col];
    }
}

)CUDA"
R"CUDA(
// ---------------------------------------------------------------------------
// Double-buffered cp.async variant of gemm_tc_tf32_blocked.
//
// Prefetches A[k+BK] and B[k+BK] into buffer[1-cur] via cp.async while the
// WMMA compute consumes buffer[cur]. Edge-tile zero padding handled by
// __pipeline_memcpy_async(..., zfill) — zfill==16 skips the global load
// entirely so out-of-range rows/cols never touch global memory.
//
// Shared memory: 2× A_tile + 2× B_tile + C_scratch
//   = 2×64×16 + 2×16×64 + 64×64 = 8 + 8 + 16 = 32 KB.
// ---------------------------------------------------------------------------
extern "C" __global__
void gemm_tc_tf32_blocked_async(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K)
{
    constexpr int BM = 64, BN = 64, BK = 16;

    __shared__ float A_tile[2][BM][BK];
    __shared__ float B_tile[2][BK][BN];
    __shared__ float C_scratch[BM][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / 2;
    const int warp_n  = warp_id % 2;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(c_frag[i][j], 0.f);

    // Each thread issues 2 × float4 for A and 2 × float4 for B per tile.
    // cp.async.16 requires BOTH src and dst to be 16-byte aligned. For strides
    // like K=27 (Cin=3, 3×3) or odd N, global alignment is not guaranteed.
    // Hybrid: issue cp.async when aligned + fully-in-bounds; otherwise scalar
    // write to shared. Scalar writes complete immediately; cp.async writes
    // complete at __pipeline_wait_prior. The __syncthreads below guarantees
    // both are visible before compute.
    auto issue_tile = [&](int k_base, int buf) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int lin = tid * 8 + i * 4;
            int row = lin / BK;
            int col = lin % BK;
            int gr = block_m + row;
            int gc = k_base + col;
            float*       dst = &A_tile[buf][row][col];
            const float* src = &A[(size_t)gr * K + gc];
            bool in_row = (gr < M);
            bool all_in = in_row && (gc + 3 < K);
            bool aligned = ((reinterpret_cast<uintptr_t>(src) & 15u) == 0);
            if (all_in && aligned) {
                __pipeline_memcpy_async(dst, src, 16);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    dst[j] = (in_row && (gc + j) < K) ? src[j] : 0.f;
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int lin = tid * 8 + i * 4;
            int row = lin / BN;
            int col = lin % BN;
            int gr = k_base + row;
            int gc = block_n + col;
            float*       dst = &B_tile[buf][row][col];
            const float* src = &B[(size_t)gr * N + gc];
            bool in_row = (gr < K);
            bool all_in = in_row && (gc + 3 < N);
            bool aligned = ((reinterpret_cast<uintptr_t>(src) & 15u) == 0);
            if (all_in && aligned) {
                __pipeline_memcpy_async(dst, src, 16);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    dst[j] = (in_row && (gc + j) < N) ? src[j] : 0.f;
            }
        }
    };

    // Kick off tile 0 into buffer 0.
    issue_tile(0, 0);
    __pipeline_commit();

    int cur = 0;
    for (int k = 0; k < K; k += BK) {
        int nxt = 1 - cur;
        int k_next = k + BK;

        // Prefetch tile k+BK into the other buffer while we consume this one.
        if (k_next < K) {
            issue_tile(k_next, nxt);
            __pipeline_commit();
            // Wait for the current tile's commit; leave the prefetch in flight.
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(a_frag[i], &A_tile[cur][warp_m*32 + i*16][kk], BK);
                #pragma unroll
                for (int e = 0; e < a_frag[i].num_elements; ++e)
                    a_frag[i].x[e] = wmma::__float_to_tf32(a_frag[i].x[e]);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(b_frag[j], &B_tile[cur][kk][warp_n*32 + j*16], BN);
                #pragma unroll
                for (int e = 0; e < b_frag[j].num_elements; ++e)
                    b_frag[j].x[e] = wmma::__float_to_tf32(b_frag[j].x[e]);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        __syncthreads();
        cur = nxt;
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int sm_row = warp_m * 32 + i * 16;
            int sm_col = warp_n * 32 + j * 16;
            wmma::store_matrix_sync(&C_scratch[sm_row][sm_col],
                                    c_frag[i][j], BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int lin = tid * 32 + i;
        int row = lin / BN, col = lin % BN;
        int gr = block_m + row, gc = block_n + col;
        if (gr < M && gc < N) C[(size_t)gr * N + gc] = C_scratch[row][col];
    }
}

// ---------------------------------------------------------------------------
// 128×128 WMMA TF32 kernel with cp.async double-buffering. 8 warps (256 threads),
// BM=BN=128, BK=16. Each warp owns a 32×64 subtile = 2×4 grid of 16×16 fragments.
// Higher arithmetic intensity than the 64×64 kernel (AI ≈ 128 vs 64 FLOP/byte),
// so it halves HBM bandwidth per FLOP — the win for big M,N,K matrices.
//
// Shared memory (96 KB) exceeds sm_80's default 48 KB cap. Uses dynamic shared
// memory. Caller must:
//   1. cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 98304)
//   2. cuLaunchKernel(..., sharedMemBytes = 98304, ...)
// ---------------------------------------------------------------------------
extern "C" __global__
void gemm_tc_tf32_128(const float* __restrict__ A,
                      const float* __restrict__ B,
                      float* __restrict__ C,
                      int M, int N, int K)
{
    constexpr int BM = 128, BN = 128, BK = 16;

    extern __shared__ float smem[];
    float (*A_tile)[BM][BK]  = reinterpret_cast<float(*)[BM][BK]>(smem);
    float (*B_tile)[BK][BN]  = reinterpret_cast<float(*)[BK][BN]>(smem + 2 * BM * BK);
    float (*C_scratch)[BN]   = reinterpret_cast<float(*)[BN]>(smem + 2 * BM * BK + 2 * BK * BN);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id >> 1;  // 0..3
    const int warp_n  = warp_id & 1;   // 0..1

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            wmma::fill_fragment(c_frag[i][j], 0.f);

    auto issue_tile = [&](int k_base, int buf) {
        // A: BM×BK = 128×16 = 2048 floats = 512 float4. 256 threads × 2 float4.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int lin = tid * 8 + i * 4;
            int row = lin / BK;
            int col = lin % BK;
            int gr = block_m + row;
            int gc = k_base + col;
            float*       dst = &A_tile[buf][row][col];
            const float* src = &A[(size_t)gr * K + gc];
            bool in_row = (gr < M);
            bool all_in = in_row && (gc + 3 < K);
            bool aligned = ((reinterpret_cast<uintptr_t>(src) & 15u) == 0);
            if (all_in && aligned) {
                __pipeline_memcpy_async(dst, src, 16);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    dst[j] = (in_row && (gc + j) < K) ? src[j] : 0.f;
            }
        }
        // B: BK×BN = 16×128 = 2048 floats = 512 float4. 256 threads × 2 float4.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int lin = tid * 8 + i * 4;
            int row = lin / BN;
            int col = lin % BN;
            int gr = k_base + row;
            int gc = block_n + col;
            float*       dst = &B_tile[buf][row][col];
            const float* src = &B[(size_t)gr * N + gc];
            bool in_row = (gr < K);
            bool all_in = in_row && (gc + 3 < N);
            bool aligned = ((reinterpret_cast<uintptr_t>(src) & 15u) == 0);
            if (all_in && aligned) {
                __pipeline_memcpy_async(dst, src, 16);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    dst[j] = (in_row && (gc + j) < N) ? src[j] : 0.f;
            }
        }
    };

    issue_tile(0, 0);
    __pipeline_commit();

    int cur = 0;
    for (int k = 0; k < K; k += BK) {
        int nxt = 1 - cur;
        int k_next = k + BK;

        if (k_next < K) {
            issue_tile(k_next, nxt);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(a_frag[i], &A_tile[cur][warp_m * 32 + i * 16][kk], BK);
                #pragma unroll
                for (int e = 0; e < a_frag[i].num_elements; ++e)
                    a_frag[i].x[e] = wmma::__float_to_tf32(a_frag[i].x[e]);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(b_frag[j], &B_tile[cur][kk][warp_n * 64 + j * 16], BN);
                #pragma unroll
                for (int e = 0; e < b_frag[j].num_elements; ++e)
                    b_frag[j].x[e] = wmma::__float_to_tf32(b_frag[j].x[e]);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        __syncthreads();
        cur = nxt;
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int sm_row = warp_m * 32 + i * 16;
            int sm_col = warp_n * 64 + j * 16;
            wmma::store_matrix_sync(&C_scratch[sm_row][sm_col],
                                    c_frag[i][j], BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        int lin = tid * 64 + i;
        int row = lin / BN, col = lin % BN;
        int gr = block_m + row, gc = block_n + col;
        if (gr < M && gc < N) C[(size_t)gr * N + gc] = C_scratch[row][col];
    }
}

// ---------------------------------------------------------------------------
// Epilogue: Y = alpha * Y + beta * C_broadcast_per_row_or_per_elem, optional.
// Used by ONNX Gemm's (alpha, beta, C) post-op. Applied after gemm_tc_tf32
// or gemm_scalar_f32 writes Y = A*B.
//   bias_kind: 0 none, 1 elementwise (C same shape as Y), 2 row-broadcast
//              (C is (N,)), 3 col-broadcast (C is (M,)).
// ---------------------------------------------------------------------------
extern "C" __global__
void gemm_epilogue_f32(float* __restrict__ Y,
                       const float* __restrict__ C,
                       int M, int N,
                       float alpha, float beta,
                       int bias_kind)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (i >= total) return;
    float v = alpha * Y[i];
    if (bias_kind) {
        int col = i % N;
        int row = i / N;
        float b = 0.f;
        if      (bias_kind == 1) b = C[i];
        else if (bias_kind == 2) b = C[col];
        else if (bias_kind == 3) b = C[row];
        v += beta * b;
    }
    Y[i] = v;
}

)CUDA"
R"CUDA(
// ---------------------------------------------------------------------------
// Int8 WMMA TensorCore GEMM — signed and unsigned 8-bit A*B → int32 C.
// Fragment shape: M=N=K=16 (one mma_sync per K-chunk of 16).
// 4 warps × 64×64 output tile; each warp owns a 32×32 subtile = 2×2 grid of
// 16×16 fragments.
//
// Zero-pad on load + bounds-checked store via shared scratch → any M,N,K works,
// not just aligned multiples. Shared memory: A[64][16] + B[16][64] (2 KB) +
// C_scratch[64][64] int32 (16 KB) ≈ 18 KB.
//
// Used by QLinearConv_cuda to replace the scalar int8 kernel on sm_80+.
// A is row-major (M,K), B is row-major (K,N), C is row-major int32 (M,N).
// ---------------------------------------------------------------------------
#define NNR_GEMM_INT8_IMPL(KERNAME, ELT)                                       \
extern "C" __global__                                                          \
void KERNAME(const ELT* __restrict__ A,                                        \
             const ELT* __restrict__ B,                                        \
             int* __restrict__ C,                                              \
             int M, int N, int K)                                              \
{                                                                              \
    constexpr int BM = 64, BN = 64, BK = 16;                                   \
    __shared__ ELT A_tile[BM][BK];                                             \
    __shared__ ELT B_tile[BK][BN];                                             \
    __shared__ int C_scratch[BM][BN];                                          \
    const int tid = threadIdx.x;                                               \
    const int warp_id = tid / 32;                                              \
    const int warp_m  = warp_id / 2;                                           \
    const int warp_n  = warp_id % 2;                                           \
    const int block_m = blockIdx.y * BM;                                       \
    const int block_n = blockIdx.x * BN;                                       \
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag[2][2];           \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 2; ++i)                                                \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j)                                            \
            wmma::fill_fragment(c_frag[i][j], 0);                              \
    for (int k = 0; k < K; k += BK) {                                          \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 8; ++i) {                                          \
            int lin = tid * 8 + i;                                             \
            int row = lin / BK, col = lin % BK;                                \
            int gr = block_m + row, gc = k + col;                              \
            A_tile[row][col] = (gr < M && gc < K)                              \
                ? A[(size_t)gr * K + gc] : (ELT)0;                             \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 8; ++i) {                                          \
            int lin = tid * 8 + i;                                             \
            int row = lin / BN, col = lin % BN;                                \
            int gr = k + row, gc = block_n + col;                              \
            B_tile[row][col] = (gr < K && gc < N)                              \
                ? B[(size_t)gr * N + gc] : (ELT)0;                             \
        }                                                                      \
        __syncthreads();                                                       \
        wmma::fragment<wmma::matrix_a, 16, 16, 16, ELT, wmma::row_major> a_frag[2]; \
        wmma::fragment<wmma::matrix_b, 16, 16, 16, ELT, wmma::row_major> b_frag[2]; \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 2; ++i)                                            \
            wmma::load_matrix_sync(a_frag[i],                                  \
                &A_tile[warp_m * 32 + i * 16][0], BK);                         \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j)                                            \
            wmma::load_matrix_sync(b_frag[j],                                  \
                &B_tile[0][warp_n * 32 + j * 16], BN);                         \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < 2; ++i)                                            \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 2; ++j)                                        \
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]); \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 2; ++i) {                                              \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 2; ++j) {                                          \
            int sm_row = warp_m * 32 + i * 16;                                 \
            int sm_col = warp_n * 32 + j * 16;                                 \
            wmma::store_matrix_sync(&C_scratch[sm_row][sm_col],                \
                                    c_frag[i][j], BN, wmma::mem_row_major);   \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 32; ++i) {                                             \
        int lin = tid * 32 + i;                                                \
        int row = lin / BN, col = lin % BN;                                    \
        int gr = block_m + row, gc = block_n + col;                            \
        if (gr < M && gc < N) C[(size_t)gr * N + gc] = C_scratch[row][col];    \
    }                                                                          \
}

NNR_GEMM_INT8_IMPL(gemm_tc_s8s32_blocked, signed char)
NNR_GEMM_INT8_IMPL(gemm_tc_u8s32_blocked, unsigned char)

#undef NNR_GEMM_INT8_IMPL

)CUDA";
}

} // namespace nnr::gpu

#endif // NNR_USE_CUDA
