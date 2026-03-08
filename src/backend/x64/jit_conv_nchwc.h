#pragma once
// JIT-compiled AVX-512 NCHWc direct convolution tile kernels.
//
// Two kernel families:
//   1x14 (legacy): 1 OC block × up to 14 spatial positions.
//   4x6:           up to 4 OC blocks × up to 6 spatial positions.
//                  Amortizes input broadcasts across F filter loads.
//
// Weight layout: [OCb, ICb, KH, KW, 16ic, 16oc] (IC-blocked).

#include "cpu_features.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include "jit_kernel.h"
#include <functional>

namespace nnr {

// =========================================================================
// Legacy 1x14 tile kernel (kept for pointwise 1x1 path)
// =========================================================================

using jit_nchwc_tile_fn_t = void(*)(
    float* out,
    const float* in_base,
    const float* w_ob,
    int64_t ICb,
    int64_t IW_bytes,
    int64_t blk_bytes,
    const float* bias);

struct jit_nchwc_key_t {
    int KH, KW;
    int strideW;
    int ow_tile;
    bool operator==(const jit_nchwc_key_t& o) const {
        return KH == o.KH && KW == o.KW && strideW == o.strideW && ow_tile == o.ow_tile;
    }
};

struct jit_nchwc_hash_t {
    size_t operator()(const jit_nchwc_key_t& k) const {
        return (size_t)k.KH * 31 + k.KW * 7 + k.strideW * 3 + k.ow_tile;
    }
};

struct jit_nchwc_conv_t : jit_kernel_t {
    // @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[NCHWc,JIT] tiling=[MR,NR] fusion=post_op
    jit_nchwc_conv_t(int KH, int KW, int strideW, int ow_tile);
};

// @nnr-meta isa=scalar dtype=fp32 layout=BLOCKED_16 special=NCHWc
inline bool jit_nchwc_eligible(int KH, int KW, int strideW, int ow_tile) {
    (void)strideW;
    size_t body = (size_t)KH * KW * 16 * (8 + (size_t)ow_tile * 14);
    size_t estimated = body + 512;
    return estimated < 60000;
}

// @nnr-meta isa=scalar dtype=fp32 layout=BLOCKED_16 special=NCHWc
jit_nchwc_tile_fn_t resolve_jit_nchwc(int KH, int KW, int strideW, int ow_tile);

// =========================================================================
// 4x6 tile kernel: F OC blocks × ow_tile spatial positions.
//
// ZMM register layout (F=4, ow_tile=6):
//   zmm0-5:   OC block 0 accumulators
//   zmm6-11:  OC block 1 accumulators
//   zmm12-17: OC block 2 accumulators
//   zmm18-23: OC block 3 accumulators
//   zmm24:    weight vector (loaded per OC block)
//   zmm25-30: broadcast input scalars (reused across F filter loads)
//   zmm31:    spare
//
// Inner loop structure (per kh,kw position):
//   for il = 0..15:
//     broadcast ow_tile inputs → zmm25..zmm(25+ow_tile-1)
//     for f = 0..F-1:
//       load filter[f] → zmm24
//       FMA: acc[f][s] += zmm24 * zmm(25+s) for s = 0..ow_tile-1
//
// KH×KW: fully unrolled.  il=0..15: runtime loop.
// Code size: ~250 bytes per (kh,kw) step → fits all practical kernel sizes.
// =========================================================================

using jit_nchwc_4x6_fn_t = void(*)(
    float* out,              // output base for first OC block at ow_start
    const float* in_base,    // input at (oh*sH-pH, ow_start*sW-pW), ICb=0
    const float* w_base,     // weight for first OC block
    int64_t ICb,             // IC block count
    int64_t IW_bytes,        // IW * 64 (byte row stride in input)
    int64_t blk_bytes,       // IH * IW * 64 (byte stride per ICb block)
    const float* bias,       // [F*16] bias or nullptr
    int64_t w_ob_stride,     // byte stride between OC blocks in weight
    int64_t out_ob_stride);  // byte stride between OC blocks in output

struct jit_nchwc_4x6_key_t {
    int KH, KW, strideW;
    int F;         // FilterCount (1-4)
    int ow_tile;   // spatial tile width (1-6)
    bool operator==(const jit_nchwc_4x6_key_t& o) const {
        return KH == o.KH && KW == o.KW && strideW == o.strideW
            && F == o.F && ow_tile == o.ow_tile;
    }
};

struct jit_nchwc_4x6_hash_t {
    size_t operator()(const jit_nchwc_4x6_key_t& k) const {
        return (size_t)k.KH * 131 + k.KW * 37 + k.strideW * 11
             + k.F * 7 + k.ow_tile;
    }
};

struct jit_nchwc_4x6_conv_t : jit_kernel_t {
    // @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=[NCHWc,JIT] tiling=[MR,NR] fusion=post_op
    jit_nchwc_4x6_conv_t(int KH, int KW, int strideW, int F, int ow_tile);
};

// @nnr-meta isa=scalar dtype=fp32 layout=BLOCKED_16 special=NCHWc
inline bool jit_nchwc_4x6_eligible(int KH, int KW, int /*strideW*/,
                                    int F, int ow_tile) {
    // il loop is runtime (not unrolled), so code is compact.
    // Per (kh,kw): ~(5 + ow_tile + F + F*ow_tile) instructions × 7 bytes
    size_t per_step = (size_t)(5 + ow_tile + F + F * ow_tile) * 7;
    size_t body = (size_t)KH * KW * per_step;
    size_t estimated = body + 1024;  // prologue, epilogue, ICb loop, bias init
    return F >= 1 && F <= 4 && ow_tile >= 1 && ow_tile <= 6
        && estimated < 60000;
}

// @nnr-meta isa=scalar dtype=fp32 layout=BLOCKED_16 special=NCHWc
jit_nchwc_4x6_fn_t resolve_jit_nchwc_4x6(int KH, int KW, int strideW,
                                           int F, int ow_tile);

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
