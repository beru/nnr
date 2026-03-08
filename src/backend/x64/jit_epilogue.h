#pragma once
// JIT fused epilogue: emits bias + activation instructions into a CodeGenerator.
// Used by GEMM and depthwise JIT kernels to fuse post-ops at register level.

#include "cpu_features.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include <cstdint>

namespace Xbyak { class CodeGenerator; struct Zmm; struct Label; }

namespace nnr {

// Activation types for JIT epilogue (superset of post_op_kind).
enum class jit_activation_t : uint8_t {
    none,
    relu,           // max(x, 0)
    clip,           // clamp(x, lo, hi)
    leaky_relu,     // x >= 0 ? x : x * alpha
    // Future: sigmoid, silu, hardswish, etc.
};

// Parameters for JIT epilogue generation.
struct jit_epilogue_params_t {
    jit_activation_t activation = jit_activation_t::none;
    float fmin = 0.0f;       // clip lower bound (relu: 0, clip: user-specified)
    float fmax = 0.0f;       // clip upper bound (clip: user-specified)
    float alpha = 0.0f;      // leaky_relu slope
};

// Emit activation instructions on zmm[first..first+count-1].
// Uses zmm_tmp0, zmm_tmp1, zmm_tmp2 as scratch (caller must not use them).
// data_labels: RIP-relative labels for embedded constants (emitted by emit_epilogue_data).
// The caller must call emit_epilogue_data() after ret() to place the constants.
struct jit_epilogue_data_t {
    Xbyak::Label* fmin_label = nullptr;
    Xbyak::Label* fmax_label = nullptr;
    Xbyak::Label* alpha_label = nullptr;
};

// Emit activation code for zmm[first..first+count-1].
// zmm_tmp0/1/2 are scratch registers (indices, not Zmm objects).
// @nnr-meta isa=AVX512 dtype=fp32 special=JIT fusion=post_op
void emit_epilogue_activation(
    Xbyak::CodeGenerator& c,
    jit_activation_t activation,
    int first, int count,
    int zmm_tmp0, int zmm_tmp1, int zmm_tmp2,
    jit_epilogue_data_t& data);

// Emit embedded constant data (after ret). Call once per kernel.
// @nnr-meta isa=scalar special=JIT fusion=post_op
void emit_epilogue_data(
    Xbyak::CodeGenerator& c,
    const jit_epilogue_params_t& params,
    jit_epilogue_data_t& data);

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
