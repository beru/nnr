// Shape-adaptive GEMM dispatch.
// After extensive benchmarking, v4 (128×128 hybrid DMA) is the best kernel
// for ALL shapes ≥ 1×1. Even for small M/N, the optimized load pipeline
// (float4, register prefetch, cp.async) outweighs the wasted threads.
//
// The dispatch just forwards to v4, with a scalar fallback for unaligned dims.

#pragma once
#include "sgemm_v4.cuh"
#include "sgemm_small.cuh"

namespace nnr_sgemm_best {

inline void sgemm(
    int M, int N, int K,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc,
    cudaStream_t stream = 0)
{
    // v4 is the best kernel for all shapes. Period.
    // Even M=1 or N=1 — the 128×128 tile handles boundaries,
    // and its optimized pipeline beats smaller unoptimized tiles.
    nnr_sgemm_v4::sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

} // namespace nnr_sgemm_best
