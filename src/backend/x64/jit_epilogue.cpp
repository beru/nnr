// JIT fused epilogue implementation.
// Emits AVX-512 activation instructions into a CodeGenerator.

#include "jit_epilogue.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include <xbyak/xbyak.h>
#include <cstring>

namespace nnr {

static uint32_t float_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

void emit_epilogue_activation(
    Xbyak::CodeGenerator& c,
    jit_activation_t activation,
    int first, int count,
    int zmm_tmp0, int zmm_tmp1, int zmm_tmp2,
    jit_epilogue_data_t& data)
{
    using namespace Xbyak;

    switch (activation) {
    case jit_activation_t::none:
        break;

    case jit_activation_t::relu:
        // max(x, 0)
        c.vxorps(Zmm(zmm_tmp0), Zmm(zmm_tmp0), Zmm(zmm_tmp0));
        for (int i = 0; i < count; i++)
            c.vmaxps(Zmm(first + i), Zmm(first + i), Zmm(zmm_tmp0));
        break;

    case jit_activation_t::clip:
        // clamp(x, fmin, fmax)
        c.vbroadcastss(Zmm(zmm_tmp0), c.dword [c.rip + *data.fmin_label]);
        c.vbroadcastss(Zmm(zmm_tmp1), c.dword [c.rip + *data.fmax_label]);
        for (int i = 0; i < count; i++)
            c.vmaxps(Zmm(first + i), Zmm(first + i), Zmm(zmm_tmp0));
        for (int i = 0; i < count; i++)
            c.vminps(Zmm(first + i), Zmm(first + i), Zmm(zmm_tmp1));
        break;

    case jit_activation_t::leaky_relu: {
        // x >= 0 ? x : x * alpha
        c.vbroadcastss(Zmm(zmm_tmp0), c.dword [c.rip + *data.alpha_label]);
        c.vxorps(Zmm(zmm_tmp1), Zmm(zmm_tmp1), Zmm(zmm_tmp1));
        for (int i = 0; i < count; i++) {
            c.vcmpps(c.k1, Zmm(first + i), Zmm(zmm_tmp1),
                     1 /* _CMP_LT_OS */);
            c.vmulps(Zmm(zmm_tmp2), Zmm(first + i), Zmm(zmm_tmp0));
            c.vmovaps(Zmm(first + i) | c.k1, Zmm(zmm_tmp2));
        }
        break;
    }
    }
}

void emit_epilogue_data(
    Xbyak::CodeGenerator& c,
    const jit_epilogue_params_t& params,
    jit_epilogue_data_t& data)
{
    c.align(4);

    if (data.fmin_label) {
        c.L(*data.fmin_label);
        c.dd(float_bits(params.fmin));
    }
    if (data.fmax_label) {
        c.L(*data.fmax_label);
        c.dd(float_bits(params.fmax));
    }
    if (data.alpha_label) {
        c.L(*data.alpha_label);
        c.dd(float_bits(params.alpha));
    }
}

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
