// JIT-compiled AVX-512 GEMM micro-kernel (8x16).
// Equivalent to ukernel_nchw() in gemm_ukernel_avx512.h but with
// zero_init and activation baked in at codegen time.

#include "jit_gemm_ukernel.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include "jit_epilogue.h"
#include <xbyak/xbyak.h>
#include <cfloat>

namespace nnr {

// Legacy constructor: maps fmin/fmax to clip activation
jit_ukernel_avx512_t::jit_ukernel_avx512_t(
    bool zero_init, bool do_fuse, float fmin, float fmax)
    : jit_ukernel_avx512_t(zero_init,
        do_fuse ? jit_activation_t::clip : jit_activation_t::none,
        {do_fuse ? jit_activation_t::clip : jit_activation_t::none, fmin, fmax, 0.0f})
{}

// Main constructor: generates JIT code for given activation
jit_ukernel_avx512_t::jit_ukernel_avx512_t(
    bool zero_init, jit_activation_t activation,
    const jit_epilogue_params_t& params)
    : jit_kernel_t(8192)
{
    using namespace Xbyak;
    auto& c = gen();
    const bool do_fuse = (activation != jit_activation_t::none);

    // Prologue: saves callee-saved GPRs + xmm6-15 on Win64.
    // rdi(7)/rsi(6) auto-saved on Win64.
    emit_prologue((1u<<3)|(1u<<12)|(1u<<13)|(1u<<14));  // rbx, r12, r13, r14; save xmm6-15

    // Named register aliases (platform-neutral after prologue)
    const Reg64 reg_kc(Operand::RDI);    // arg0
    const Reg64 reg_pa(Operand::RSI);    // arg1
    const Reg64 reg_pb(10);              // r10 ← arg2
    const Reg64 reg_stride(11);          // r11 ← arg3 (int, sign-extended)
    const Reg64 reg_pc_arr(12);          // r12 ← arg4
    const Reg64 reg_v(13);               // r13 ← arg5 (int, sign-extended)
    const Reg64 reg_bp(14);              // r14 ← arg6
    const Reg64 reg_tmp(Operand::RBX);
    const Reg64 reg_tmp2(Operand::RAX);

    // Load arguments into working registers (no #ifdef — load_arg handles ABI)
    load_arg(0, Operand::RDI);
    load_arg(1, Operand::RSI);
    load_arg(2, 10);
    load_arg_i32(3, 11);
    c.shl(reg_stride, 2);              // int stride → byte stride
    load_arg(4, 12);
    load_arg_i32(5, 13);
    c.shl(reg_v, 2);                   // int v → byte offset
    if (do_fuse)
        load_arg(6, 14);

    // --- Load or zero accumulators ---
    if (zero_init) {
        for (int r = 0; r < 8; r++)
            c.vxorps(Zmm(r), Zmm(r), Zmm(r));
    } else {
        for (int r = 0; r < 8; r++) {
            c.mov(reg_tmp, c.ptr [reg_pc_arr + r * 8]);
            c.vmovups(Zmm(r), c.ptr [reg_tmp + reg_v]);
        }
    }

    // --- K-4x unrolled main loop ---
    Label k_loop_4x, k_loop_1x, k_done;

    c.cmp(reg_kc, 4);
    c.jl(k_loop_1x, CodeGenerator::T_NEAR);

    c.L(k_loop_4x);
    {
        c.vmovups(Zmm(8), c.ptr [reg_pb]);
        c.lea(reg_tmp2, c.ptr [reg_pb + reg_stride]);
        c.vmovups(Zmm(9), c.ptr [reg_tmp2]);
        c.lea(reg_tmp2, c.ptr [reg_tmp2 + reg_stride]);
        c.vmovups(Zmm(10), c.ptr [reg_tmp2]);
        c.lea(reg_tmp2, c.ptr [reg_tmp2 + reg_stride]);
        c.vmovups(Zmm(11), c.ptr [reg_tmp2]);

        for (int ki = 0; ki < 4; ki++) {
            for (int r = 0; r < 8; r++) {
                c.vbroadcastss(Zmm(12), c.dword [reg_pa + (ki * 8 + r) * 4]);
                c.vfmadd231ps(Zmm(r), Zmm(12), Zmm(8 + ki));
            }
        }

        c.add(reg_pa, 8 * 4 * 4);
        c.lea(reg_pb, c.ptr [reg_pb + reg_stride * 4]);
        c.sub(reg_kc, 4);
        c.cmp(reg_kc, 4);
        c.jge(k_loop_4x);
    }

    // --- K-1x remainder ---
    c.L(k_loop_1x);
    c.test(reg_kc, reg_kc);
    c.jz(k_done, CodeGenerator::T_NEAR);

    {
        Label k_tail;
        c.L(k_tail);
        c.vmovups(Zmm(8), c.ptr [reg_pb]);
        for (int r = 0; r < 8; r++) {
            c.vbroadcastss(Zmm(12), c.dword [reg_pa + r * 4]);
            c.vfmadd231ps(Zmm(r), Zmm(12), Zmm(8));
        }
        c.add(reg_pa, 8 * 4);
        c.add(reg_pb, reg_stride);
        c.dec(reg_kc);
        c.jnz(k_tail);
    }

    c.L(k_done);

    // --- Fused epilogue: bias + activation ---
    // Allocate labels for embedded constants
    Label L_fmin, L_fmax, L_alpha;
    jit_epilogue_data_t epi_data;

    if (do_fuse) {
        // Bias add: acc[r] += broadcast(bp[r])
        for (int r = 0; r < 8; r++) {
            c.vbroadcastss(Zmm(12), c.dword [reg_bp + r * 4]);
            c.vaddps(Zmm(r), Zmm(r), Zmm(12));
        }

        // Set up labels for constants used by activation
        if (activation == jit_activation_t::clip) {
            epi_data.fmin_label = &L_fmin;
            epi_data.fmax_label = &L_fmax;
        } else if (activation == jit_activation_t::leaky_relu) {
            epi_data.alpha_label = &L_alpha;
        }

        // Emit activation on zmm0-zmm7, using zmm13/14/15 as scratch
        emit_epilogue_activation(c, activation, 0, 8, 13, 14, 15, epi_data);
    }

    // --- Store accumulators ---
    for (int r = 0; r < 8; r++) {
        c.mov(reg_tmp, c.ptr [reg_pc_arr + r * 8]);
        c.vmovups(c.ptr [reg_tmp + reg_v], Zmm(r));
    }

    emit_epilogue();

    // --- Embedded constant data (after ret) ---
    if (do_fuse)
        emit_epilogue_data(c, params, epi_data);

    finalize();
}

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
