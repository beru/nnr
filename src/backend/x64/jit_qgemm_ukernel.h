#pragma once
// JIT-compiled int8 GEMM micro-kernel for AVX-512 VNNI.
// 6×48 tile: 6 rows × 48 columns, 18 VPDPBUSD per K-step sharing 3 B loads.
// Bakes ZeroMode at codegen time to eliminate hot-loop branches.

#include "cpu_features.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include "jit_kernel.h"

namespace nnr::int8 {

// JIT micro-kernel function signature.
// packed_A: [PackedCountK × MR × 4] uint8, interleaved by row.
// packed_B: [PackedCountK × NR × 4] uint8, VNNI panel layout.
// C: [MR rows × ldc columns] int32 output (row-major, strided by ldc).
// PackedCountK: number of VNNI groups (K/4 rounded up).
// ldc: C row stride in int32 elements.
// RowSumBuffer: [MR] int32 row sums (for zero-point compensation).
// ColumnSumBuffer: [NR] int32 column sums.
using jit_qgemm_ukernel_fn_t = void(*)(
    const uint8_t* packed_A,
    const uint8_t* packed_B,
    int32_t* C,
    int PackedCountK,
    int ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer);

constexpr int QGEMM_MR = 6;
constexpr int QGEMM_NR = 48;

struct jit_qgemm_ukernel_key_t {
    bool zero_mode;     // true = zero accumulators, false = accumulate into C
    int col_count;      // 48, 32, 16, or tail (determines masking)
    bool operator==(const jit_qgemm_ukernel_key_t& o) const {
        return zero_mode == o.zero_mode && col_count == o.col_count;
    }
};

struct jit_qgemm_ukernel_hash_t {
    size_t operator()(const jit_qgemm_ukernel_key_t& k) const {
        // zero_mode: bit 0. col_count class: bits 1-2 (0=48, 1=32, 2=16, 3=tail)
        int cc = (k.col_count >= 48) ? 0 : (k.col_count >= 32) ? 1 : (k.col_count >= 16) ? 2 : 3;
        return (size_t)k.zero_mode | ((size_t)cc << 1);
    }
};

struct jit_qgemm_ukernel_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NCHW special=[GEMM,JIT] tiling=[MR,NR]
    jit_qgemm_ukernel_avx512_t(bool zero_mode, int col_count);
};

} // namespace nnr::int8

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
