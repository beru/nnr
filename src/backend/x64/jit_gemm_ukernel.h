#pragma once
// JIT-compiled GEMM micro-kernel for AVX-512.
// Specializes on (zero_init, activation) at codegen time.
// Eliminates branches and enables open-ended post-op fusion.

#include "cpu_features.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include "jit_kernel.h"
#include "jit_epilogue.h"
#include "backend/cpu/kernel/post_ops.h"

namespace nnr {

// Function signature: zero_init, activation, and fmin/fmax/alpha are baked in.
// bp is only read when activation != none; ignored otherwise.
using jit_ukernel_fn_t = void(*)(
    int kc,
    const float* pa,
    const float* pb,
    int pb_stride,
    float* const pc[8],
    int v,
    const float* bp);

struct jit_ukernel_key_t {
    bool zero_init;
    bool do_fuse;
    bool operator==(const jit_ukernel_key_t& o) const {
        return zero_init == o.zero_init && do_fuse == o.do_fuse;
    }
};

struct jit_ukernel_hash_t {
    size_t operator()(const jit_ukernel_key_t& k) const {
        return (size_t)k.zero_init | ((size_t)k.do_fuse << 1);
    }
};

struct jit_ukernel_avx512_t : jit_kernel_t {
    // Construct with activation type and parameters.
    jit_ukernel_avx512_t(bool zero_init, jit_activation_t activation,
                          const jit_epilogue_params_t& params);
    // Legacy: construct from fmin/fmax (maps to clip activation).
    // @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW special=[GEMM,JIT] tiling=[MR,NR] fusion=post_op
    jit_ukernel_avx512_t(bool zero_init, bool do_fuse, float fmin, float fmax);
};

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
