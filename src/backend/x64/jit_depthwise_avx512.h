#pragma once
// JIT-compiled depthwise 3x3 stride=1 kernel (AVX-512).
// Bakes shape, boundary masks, and activation at codegen time.
// Eliminates: kH/kW dispatch, row pointer ternaries, mask computation, post-op pass.

#include "cpu_features.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include "jit_kernel.h"
#include "jit_epilogue.h"

namespace nnr {

// JIT depthwise function: processes one channel (all oH × oW output).
using jit_dw_fn_t = void(*)(
    const float* input,     // channel input (iH × iW floats)
    float* output,          // channel output (oH × oW floats)
    const float* weights,   // 9 floats (3×3 kernel)
    const float* bias_ptr   // pointer to 1 float, or nullptr
);

struct jit_dw_key_t {
    int iH, iW, oH, oW, pH, pW;
    jit_activation_t activation;
    bool operator==(const jit_dw_key_t& o) const {
        return iH == o.iH && iW == o.iW && oH == o.oH && oW == o.oW
            && pH == o.pH && pW == o.pW && activation == o.activation;
    }
};

struct jit_dw_hash_t {
    size_t operator()(const jit_dw_key_t& k) const {
        size_t h = (size_t)k.iH;
        h = h * 131 + k.iW;
        h = h * 131 + k.oH;
        h = h * 131 + k.oW;
        h = h * 131 + k.pH;
        h = h * 131 + k.pW;
        h = h * 131 + (size_t)k.activation;
        return h;
    }
};

struct jit_depthwise_avx512_t : jit_kernel_t {
    // @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=[DW,JIT]
    jit_depthwise_avx512_t(int iH, int iW, int oH, int oW, int pH, int pW,
                            jit_activation_t activation,
                            const jit_epilogue_params_t& params);
};

// Check if JIT depthwise compilation is expected to succeed.
// The unrolled ow-row body scales with oW; large spatial overflows the code buffer.
// @nnr-meta isa=scalar dtype=fp32 special=[DW,JIT]
inline bool jit_dw_eligible(int iH, int iW, int oH, int oW, int pH, int pW) {
    (void)iH; (void)pH; (void)pW;
    // Each ow position emits ~3 FMA instructions (~30 bytes) for the interior path,
    // plus boundary handling. The oh loop wraps the ow body.
    // Top/bottom boundary rows are fully unrolled (oH rows total code).
    size_t ow_body = (size_t)oW * 40;           // interior row
    size_t boundary = (size_t)(oW + iW) * 50;   // boundary rows (conservative)
    size_t estimated = ow_body * 2 + boundary * 4 + 1024;  // oh loop + edges + overhead
    return estimated < 60000;
}

// Resolve JIT depthwise kernel (thread-safe, cached).
// Caller should check jit_dw_eligible() first.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=[DW,JIT]
jit_dw_fn_t resolve_jit_depthwise(
    int iH, int iW, int oH, int oW, int pH, int pW,
    jit_activation_t activation, const jit_epilogue_params_t& params);

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
