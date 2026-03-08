#pragma once
// JIT-compiled fused im2col + int8 GEMM micro-kernel for AVX-512 VNNI.
// MR=6 rows × NR=16 columns. Software-pipelined K-loop: loads/interleave for
// K-group k+1 overlap with VPDPBUSD chain for K-group k.
//
// Reads input pixels directly (no im2col buffer or B-packing).

#include "cpu_features.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include "jit_kernel.h"
#include <cstddef>
#include <cstdint>

namespace nnr::int8 {

// JIT micro-kernel function signature.
// W:          weight row base: W + i0 * CHW (uint8, rows are CHW apart)
// x_pad:      pre-padded input base + ow0 (uint8)
// Y:          output base: Y + i0 * y_stride + oh_rel * oW + ow0 (int32)
// k_off:      [K4] byte offsets into x_pad per K sub-index (includes oh contribution)
// Kgroups:    number of VNNI groups (K4 / 4)
// CHW:        weight row stride in bytes
// MR_actual:  actual rows <= 6 (for tail M-block)
// y_stride:   output row stride in int32 elements
// col_sum:    [16] output: column sums (int32, for zero-point compensation by caller)
// mask:       16-bit mask for tail ow columns (0xFFFF for full width)
// zero_mode:  1=zero accumulators (first KC block), 0=load from Y (subsequent blocks)
using jit_conv_int8_ukernel_fn_t = void(*)(
    const uint8_t* W,
    const uint8_t* x_pad,
    int32_t* Y,
    const size_t* k_off,
    int Kgroups,
    int CHW,
    int MR_actual,
    int y_stride,
    int32_t* col_sum,
    uint16_t mask,
    int zero_mode);

constexpr int CONV_INT8_MR = 6;          // intrinsics fallback
constexpr int CONV_INT8_MR_JIT = 12;     // NR=16 JIT: zmm14-25
constexpr int CONV_INT8_MR_JIT_NR32 = 10;// NR=32 JIT: zmm0-9 col0, zmm10-19 col1
constexpr int CONV_INT8_NR = 16;

// Cache key: MR_actual (1-6).
// Mask is passed as a runtime parameter (not baked at JIT time).
struct jit_conv_int8_key_t {
    int mr;
    bool operator==(const jit_conv_int8_key_t& o) const { return mr == o.mr; }
};

struct jit_conv_int8_hash_t {
    size_t operator()(const jit_conv_int8_key_t& k) const { return (size_t)k.mr; }
};

struct jit_conv_int8_ukernel_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[Direct,JIT] tiling=[MR,NR]
    jit_conv_int8_ukernel_avx512_t(int mr_actual);
};

// NR=32 fused im2col: two 16-col groups per K-step, 2×MR VPDPBUSDs.
// col_sum[32]: [0..15]=col0 sums, [16..31]=col1 sums (at ptr+64).
using jit_conv_int8_nr32_ukernel_fn_t = void(*)(
    const uint8_t* W, const uint8_t* x_pad, int32_t* Y,
    const size_t* k_off, int Kgroups, int CHW, int MR_actual,
    int y_stride, int32_t* col_sum, uint16_t mask0, uint16_t mask1,
    int zero_mode);

struct jit_conv_int8_nr32_ukernel_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[Direct,JIT] tiling=[MR,NR]
    jit_conv_int8_nr32_ukernel_avx512_t(int mr_actual);
};

// ── Packed-input GEMM JIT ────────────────────────────────────────────
// ORT-style: reads pre-interleaved VNNI data via sequential zmm loads.
// MR up to 12 rows × NR=16 columns. No interleave in K-loop.
// Colsum computed during packing, not here.
//
// W:          weight row base (uint8, rows CHW apart)
// packed:     pre-interleaved VNNI data [Kgroups × 64 bytes], sequential
// Y:          output base (int32, rows y_stride apart)
// Kgroups:    number of VNNI groups
// CHW:        weight row stride in bytes
// y_stride:   output row stride in int32 elements
// mask:       16-bit mask for tail columns
using jit_packed_gemm_int8_fn_t = void(*)(
    const uint8_t* W,
    const void* packed,
    int32_t* Y,
    int Kgroups,
    int CHW,
    int y_stride,
    uint16_t mask);

struct jit_packed_gemm_int8_ukernel_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[GEMM,JIT] tiling=[MR,NR]
    jit_packed_gemm_int8_ukernel_avx512_t(int mr_actual);
};

// ── Packed-input GEMM JIT, NR=32 (2 ZMM columns) ────────────────────
// MR up to 6 rows × NR=32 columns. 12 accumulators (6 per column group).
// Handles zero_mode, row_sums, col_sums in epilogue.
//
// A:          weight row base (uint8, rows lda apart, running pointer)
// packed:     [Kgroups × 128 bytes] = 2 zmms per K-group (col0 + col1)
// C:          output (int32, rows ldc apart)
// Kgroups:    number of VNNI groups in this K-block
// lda:        weight row stride in bytes
// ldc:        output row stride in int32 elements
// row_sums:   [MR] int32 row sum corrections
// col_sums:   [32] int32 column sum corrections (lo16 + hi16)
// mask_lo:    16-bit mask for columns 0-15
// mask_hi:    16-bit mask for columns 16-31
// zero_mode:  1 = zero accumulators, 0 = load from C
// Always accumulates (loads C, adds GEMM result + row/col sums, stores).
// Caller must zero C before the first K-block.
using jit_packed_gemm_nr32_fn_t = void(*)(
    const uint8_t* A,
    const void* packed,
    int32_t* C,
    int Kgroups,
    int lda,
    int ldc,
    const int32_t* row_sums,
    const int32_t* col_sums,
    uint16_t mask_lo,
    uint16_t mask_hi,
    int ldb);   // packed B K-group stride in bytes (= N*4 for standard packing)

struct jit_packed_gemm_nr32_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[GEMM,JIT] tiling=[MR,NR] fusion=qdq
    jit_packed_gemm_nr32_avx512_t(int mr_actual);
};

// ── Gather-GEMM JIT: pre-packed weights, NR=48 (3 ZMM columns) ───────
// Transposed GEMM: M=spatial, N=OC, K=CHW.
// A-side (broadcast): input gathered via k_off (4 byte loads → uint32 → vpbroadcastd)
// B-side (sequential): pre-packed weights (3 zmm loads per K-step)
// VPDPBUSD: src1=input_uint8(unsigned), src2=weight_int8(signed). No shifts.
constexpr int GATHER_GEMM_MR = 6;
constexpr int GATHER_GEMM_NR = 48;

// Fused gather-GEMM + requantize. Pre-biased accumulators, writes uint8 directly.
// adj_col: [48] int32, pre-computed -x_zp * col_sum_w (loaded as initial accumulators)
// rq_params: packed struct { float* scale, float* bias, float inv_y_scale, float y_zp,
//            float qmin, float qmax, uint8_t* Y_out, int y_stride }
struct gather_rq_t {
    const float* scale;      // [NR] combined_scales per channel
    const float* bias;       // [NR] bias per channel (or nullptr → 0)
    float inv_y_scale;
    float y_zp;
    float qmin, qmax;
    uint8_t* Y_out;          // output base (NHWC: [spatial * OC])
    int y_stride;             // OC (NHWC stride between spatial rows)
};

using jit_gather_gemm_nr48_fn_t = void(*)(
    const uint8_t* x_pad,       // input base (x_pad + ow0)
    const int8_t* W_packed,     // packed weight NR-block base
    const int32_t* adj_col,     // [48] ZP correction (init accumulators)
    int Kgroups,
    const size_t* k_off,        // [K4] offsets into x_pad (oh-adjusted)
    const gather_rq_t* rq,      // requantize params
    uint16_t mask0, uint16_t mask1, uint16_t mask2);  // channel masks

struct jit_gather_gemm_nr48_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[GEMM,JIT] tiling=[MR,NR] fusion=qdq
    jit_gather_gemm_nr48_avx512_t(int mr_actual);
};

// Requantize parameters for JIT packed GEMM NR=48 epilogue.
// Passed as nullptr for non-last K-blocks (store int32 to C).
// Filled by caller for last K-block (requantize and store uint8).
struct jit_rq_params_t {
    const float* output_scales;    // [OC] per-channel: x_scale * w_scale / y_scale
    const int32_t* bias_int32;     // [OC] integer bias, or nullptr
    const int32_t* w_col_sums;     // [OC] column sums, or nullptr (x_zp==0)
    int32_t x_zp_neg;             // -(int32_t)x_zp
    int32_t y_zp_int;             // output zero point
    float rq_qmin;                // 0.0f - y_zp
    float rq_qmax;                // 255.0f - y_zp
    int32_t row_corr[6];          // per-row ZP correction
    uint8_t* y_rows[6];           // per-row Y_out pointers
};

// ── Packed GEMM JIT, NR=48 (3 sub-panels of 16, ORT-style) ──────────
// A (broadcast): row-major [MR x K4], vpbroadcastd 4 bytes, 6 independent row pointers.
// B (sequential): 3 sub-panels [Kgroups x 64], stride sub_stride between sub-panels.
// 18 accumulators: zmm14-19 (sub0), zmm20-25 (sub1), zmm26-31 (sub2).
// K-loop: 3 zmm loads + 6 broadcasts + 18 VPDPBUSDs = 27 instructions.
// Zero/accumulate mode. Stores int32 to C[MR x ldc], or requantizes to uint8 if rq != null.
//
// A:          packed A base (uint8, rows lda apart)
// B:          packed B sub-panel 0 base
// C:          output (int32, rows ldc apart)
// Kgroups:    VNNI groups in this K-block
// lda:        A row stride in bytes (= K4 for this K-block)
// ldc:        C row stride in int32 elements
// sub_stride: bytes between B sub-panels (= Kgroups_full * 64)
// zero_mode:  1 = zero accumulators, 0 = load from C
// nr_tail:    0=none, 1=16 cols, 2=32 cols after full NR-blocks
// rq:         nullptr = store int32 to C; non-null = requantize and store uint8
using jit_packed_gemm_nr48_fn_t = void(*)(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    int Kgroups,
    int lda,
    int ldc,
    size_t sub_stride,
    int zero_mode,
    int num_nr_blocks,
    int nr_tail,
    const jit_rq_params_t* rq);

struct jit_packed_gemm_nr48_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[GEMM,JIT] tiling=[MR,NR] fusion=qdq
    jit_packed_gemm_nr48_avx512_t(int mr_actual);
};

// Partial NR kernels for N%48 remainder (16 or 32 columns).
// Same calling convention as NR=48 but process exactly 1 block.
// num_nr_blocks parameter is ignored (always 1).
using jit_packed_gemm_partial_fn_t = void(*)(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    int Kgroups,
    int lda,
    int ldc,
    size_t sub_stride,
    int zero_mode);

struct jit_packed_gemm_partial16_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[GEMM,JIT] tiling=[MR,NR]
    jit_packed_gemm_partial16_avx512_t(int mr_actual);
};

struct jit_packed_gemm_partial32_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[GEMM,JIT] tiling=[MR,NR]
    jit_packed_gemm_partial32_avx512_t(int mr_actual);
};

// ── Memcpy-free NHWC-direct GEMM JIT, NR=48 ─────────────────────────
// Reads x_pad_nhwc directly via per-row base pointers + per-kernel-pixel
// byte offsets — no pack_a memcpy floor. Forked from
// jit_packed_gemm_nr48_avx512_t with the addressing rewritten for NHWC
// channel-minor layout. K-order is (kh, kw, ic) matching pack_weights_nr48_nhwc.
//
// One call processes a single MR×NR=48 panel for the full kernel K
// (kHW × IC4 k4-steps). Always zero_mode (caller invokes once per K-block —
// for the SSD-12 use case the entire K fits in one call, so single-K is
// sufficient for the prototype). int32 output; requantize is the caller's
// responsibility.
//
// row_bases:  [MR] per-row pointers into pre-padded NHWC input.
//             For MR_actual < 6 the tail entries should still be valid
//             pointers (e.g. duplicate row 0) — the kernel issues all
//             MR_actual broadcasts unconditionally.
// pixel_off:  [kHW] int32 byte offsets per kernel pixel (shared across rows).
// kHW:        number of kernel pixels (kH * kW).
// IC4:        IC / 4 (number of VNNI k4-steps per kernel pixel).
// B:          packed B sub-panel-0 base pointer for this N-block
//             (sub-stride layout — same as jit_packed_gemm_nr48_avx512_t).
// sub_stride: bytes between consecutive B sub-panels (= Kgroups * 64).
// C:          int32 output [MR x NR=48], row-stride ldc. Used only when
//             rq == nullptr; ignored when rq != nullptr.
// ldc:        C row stride in int32 elements.
// rq:         requantize params, or nullptr.
//             - nullptr: store int32 accumulators to C[MR x NR=48].
//             - non-null: requantize in-register and store uint8 directly to
//               rq->y_rows[r] at column offsets 0/16/32 for the 3 sub-panels.
//               Caller is responsible for filling rq with N-block-relative
//               pointers (output_scales + nb*NR, w_col_sums + nb*NR, etc.)
//               and per-row state (row_corr[r], y_rows[r]).
using jit_nhwc_direct_gemm_nr48_fn_t = void(*)(
    const uint8_t* const* row_bases,
    const int32_t* pixel_off,
    int kHW,
    int IC4,
    const uint8_t* B,
    size_t sub_stride,
    int32_t* C,
    int ldc,
    const jit_rq_params_t* rq);

struct jit_nhwc_direct_gemm_nr48_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[Direct,JIT] tiling=[MR,NR] fusion=qdq
    jit_nhwc_direct_gemm_nr48_avx512_t(int mr_actual);
};

// ── NHWC-direct GEMM JIT, NR=16 ─────────────────────────────────────
// Memcpy-free counterpart of `MlasConvSymKernelAvx512Vnni` (ORT's SSD-12
// int8 kernel). 6 rows × 16 cols panel, 6 accumulator zmms, indirection
// buffer for A-side addressing.
//
// Call processes one MR × NR=16 panel for the full kernel K (kHW × IC4).
// The K-loop is nested (kp × k4) with 6 vpbroadcastd + 1 vmovdqu32 +
// 6 vpdpbusd per k4-step. Lower per-k4 µop count (13 vs 27 for NR=48)
// trades off against 3× more kernel calls per panel.
//
// ptrs_kp_major: [kHW × MR] flat indirection buffer. Layout is kp-major
//                so that at each kp iteration the kernel loads 6 row
//                pointers from a contiguous 48-byte stride. Caller builds
//                this per M-block (~432 bytes, one cache line for 3×3).
//                For MR < 6 the tail entries should still be valid ptrs
//                (pad with row 0).
// kHW:           number of kernel pixels (kH * kW).
// IC4:           IC / 4 (number of VNNI k4-steps per kernel pixel).
// B:             packed B panel for this 16-OC block. Layout is flat
//                [Kgroups × 64 bytes] — reuses one sub-panel of the
//                existing `pack_weights_nr48_nhwc` output (each NR=48
//                sub-panel is already an NR=16 panel).
// C:             int32 output [MR × NR=16], row-stride ldc.
//                Used only when rq == nullptr.
// ldc:           C row stride in int32 elements.
// rq:            requantize params, or nullptr. Same semantics as the
//                NR=48 variant but processes a single sub-panel.
using jit_nhwc_direct_gemm_nr16_fn_t = void(*)(
    const uint8_t* const* ptrs_kp_major,
    int kHW,
    int IC4,
    const uint8_t* B,
    int32_t* C,
    int ldc,
    const jit_rq_params_t* rq);

struct jit_nhwc_direct_gemm_nr16_avx512_t : jit_kernel_t {
    // @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NHWC special=[Direct,JIT] tiling=[MR,NR] fusion=qdq
    jit_nhwc_direct_gemm_nr16_avx512_t(int mr_actual);
};

} // namespace nnr::int8

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
