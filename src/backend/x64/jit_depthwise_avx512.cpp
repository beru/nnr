// JIT-compiled depthwise 3x3 stride=1 kernel (AVX-512).
// Generates per-shape code with fused activation, baked boundary masks,
// and no row-pointer ternaries.

#include "jit_depthwise_avx512.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include <xbyak/xbyak.h>
#include <algorithm>
#include <cfloat>

namespace nnr {

// ---------------------------------------------------------------------------
// Register allocation
// ---------------------------------------------------------------------------
// GPR:
//   r12  = input base (const)
//   r13  = output base (const)
//   rdi  = row 0 pointer (per oh)
//   rsi  = row 1 pointer (per oh)
//   rbx  = row 2 pointer (per oh)
//   r10  = ow byte offset (interior ow loop)
//   r11  = output row pointer (per oh)
//   r14  = oh counter (interior oh loop)
//   rax, rcx, rdx = scratch
//
// ZMM:
//   0-2  = accumulators (a0=bias, a1=0, a2=0)
//   3    = bias broadcast (const)
//   4-12 = 9 weight broadcasts (const): zmm(4+kh*3+kw) = w[kh][kw]
//   13   = scratch (leaky_relu blend)
//   14   = fmin or alpha (const, activation)
//   15   = fmax or zeros (const, activation)
//   16   = temp for masked loads

using namespace Xbyak;

// Emit activation on zmm0. Constants pre-loaded in zmm14/zmm15.
static void emit_act(CodeGenerator& c, jit_activation_t act) {
    switch (act) {
    case jit_activation_t::none: break;
    case jit_activation_t::relu:
        c.vmaxps(Zmm(0), Zmm(0), Zmm(15));
        break;
    case jit_activation_t::clip:
        c.vmaxps(Zmm(0), Zmm(0), Zmm(14));
        c.vminps(Zmm(0), Zmm(0), Zmm(15));
        break;
    case jit_activation_t::leaky_relu:
        c.vcmpps(c.k1, Zmm(0), Zmm(15), 1);
        c.vmulps(Zmm(13), Zmm(0), Zmm(14));
        c.vmovaps(Zmm(0) | c.k1, Zmm(13));
        break;
    }
}

// Emit one fixed-position ow tile (boundary or interior, 1-16 elements).
// row pointers: rdi(r0), rsi(r1), rbx(r2). Output row: r11.
static void emit_tile_fixed(
    CodeGenerator& c, int ow, int count, int iW, int pW,
    bool r0_ok, bool r1_ok, bool r2_ok,
    jit_activation_t act)
{
    const Reg64 rows[3] = {Reg64(Operand::RDI), Reg64(Operand::RSI), Reg64(Operand::RBX)};
    const bool valid[3] = {r0_ok, r1_ok, r2_ok};

    // Init accumulators
    c.vmovaps(Zmm(0), Zmm(3));   // a0 = bias
    if (r1_ok) c.vxorps(Zmm(1), Zmm(1), Zmm(1));
    if (r2_ok) c.vxorps(Zmm(2), Zmm(2), Zmm(2));

    for (int kh = 0; kh < 3; kh++) {
        if (!valid[kh]) continue;
        for (int kw = 0; kw < 3; kw++) {
            int iw = ow - pW + kw;
            int mf = std::max(0, -iw);
            int ml = std::min(count, iW - iw);
            uint16_t mask = (ml > mf)
                ? (uint16_t)((0xFFFFu >> (16 - ml + mf)) << mf) : 0;
            int w_zmm = 4 + kh * 3 + kw;
            int disp = iw * 4;

            if (mask == 0) {
                // skip
            } else if (mask == 0xFFFF && count == 16) {
                c.vfmadd231ps(Zmm(kh), Zmm(w_zmm), c.ptr[rows[kh] + disp]);
            } else {
                c.mov(c.eax, (int)(uint32_t)mask);
                c.kmovw(c.k1, c.eax);
                c.vmovups(Zmm(16) | c.k1 | c.T_z, c.ptr[rows[kh] + disp]);
                c.vfmadd231ps(Zmm(kh), Zmm(w_zmm), Zmm(16));
            }
        }
    }

    // Reduce
    if (r1_ok) c.vaddps(Zmm(0), Zmm(0), Zmm(1));
    if (r2_ok) c.vaddps(Zmm(0), Zmm(0), Zmm(2));

    emit_act(c, act);

    // Store
    if (count < 16) {
        uint16_t sm = (uint16_t)((1u << count) - 1);
        c.mov(c.eax, (int)(uint32_t)sm);
        c.kmovw(c.k1, c.eax);
        c.vmovups(c.ptr[Reg64(11) + ow * 4] | c.k1, Zmm(0));
    } else {
        c.vmovups(c.ptr[Reg64(11) + ow * 4], Zmm(0));
    }
}

// Emit one register-indexed ow tile (interior ow loop body).
// r10 = ow byte offset. All 3 rows valid, all loads in bounds.
static void emit_tile_indexed(
    CodeGenerator& c, int pW, jit_activation_t act)
{
    const Reg64 r_ow(10);
    const Reg64 rows[3] = {Reg64(Operand::RDI), Reg64(Operand::RSI), Reg64(Operand::RBX)};

    c.vmovaps(Zmm(0), Zmm(3));
    c.vxorps(Zmm(1), Zmm(1), Zmm(1));
    c.vxorps(Zmm(2), Zmm(2), Zmm(2));

    for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
            int disp = (kw - pW) * 4;
            int w_zmm = 4 + kh * 3 + kw;
            c.vfmadd231ps(Zmm(kh), Zmm(w_zmm), c.ptr[rows[kh] + r_ow + disp]);
        }
    }

    c.vaddps(Zmm(0), Zmm(0), Zmm(1));
    c.vaddps(Zmm(0), Zmm(0), Zmm(2));
    emit_act(c, act);
    c.vmovups(c.ptr[Reg64(11) + r_ow], Zmm(0));
}

// Emit all ow tiles for one oh row.
static void emit_ow_row(
    CodeGenerator& c, int oW, int iW, int pW,
    bool r0_ok, bool r1_ok, bool r2_ok,
    jit_activation_t act)
{
    const int ow_int_start = pW;
    const int ow_int_end = iW + pW - 17;

    int ow = 0;

    // Left boundary tiles (unrolled, baked masks)
    while (ow + 16 <= oW && ow < ow_int_start) {
        emit_tile_fixed(c, ow, 16, iW, pW, r0_ok, r1_ok, r2_ok, act);
        ow += 16;
    }

    // Interior ow tiles
    int loop_start = ow;
    int loop_end = loop_start;
    while (loop_end + 16 <= oW && loop_end < ow_int_end)
        loop_end += 16;

    // Only use a loop for the interior when all 3 rows are valid
    // (boundary oh rows have some rows invalid — still unroll for them)
    bool all_valid = r0_ok && r1_ok && r2_ok;
    int num_interior = (loop_end - loop_start) / 16;

    if (num_interior > 0) {
        if (all_valid && num_interior > 2) {
            // Tight interior ow loop
            Reg64 r_ow(10);
            Label L_ow;
            c.mov(r_ow.cvt32(), loop_start * 4);
            c.L(L_ow);
            emit_tile_indexed(c, pW, act);
            c.add(r_ow.cvt32(), 64);
            c.cmp(r_ow.cvt32(), loop_end * 4);
            c.jl(L_ow);
        } else {
            // Unroll interior tiles
            for (int t = loop_start; t < loop_end; t += 16)
                emit_tile_fixed(c, t, 16, iW, pW, r0_ok, r1_ok, r2_ok, act);
        }
        ow = loop_end;
    }

    // Right boundary tiles (unrolled)
    while (ow + 16 <= oW) {
        emit_tile_fixed(c, ow, 16, iW, pW, r0_ok, r1_ok, r2_ok, act);
        ow += 16;
    }

    // Tail (masked store)
    int tail = oW - ow;
    if (tail > 0)
        emit_tile_fixed(c, ow, tail, iW, pW, r0_ok, r1_ok, r2_ok, act);
}

// ---------------------------------------------------------------------------
// Constructor — generates the full JIT kernel
// ---------------------------------------------------------------------------
jit_depthwise_avx512_t::jit_depthwise_avx512_t(
    int iH, int iW, int oH, int oW, int pH, int pW,
    jit_activation_t activation, const jit_epilogue_params_t& params)
    : jit_kernel_t(32768)
{
    auto& c = gen();
    const int iW_bytes = iW * 4;
    const int oW_bytes = oW * 4;

    // Interior oh range: all 3 input rows in bounds
    // ih0 = oh - pH, need ih0 >= 0 and ih0+2 < iH
    const int oh_int_start = pH;
    const int oh_int_end = iH + pH - 2;  // exclusive

    // Prologue — rbx(3), r12-r14
    emit_prologue((1u << 3) | (1u << 12) | (1u << 13) | (1u << 14));

    const Reg64 r_input(12);
    const Reg64 r_output(13);
    const Reg64 r_r0(Operand::RDI);
    const Reg64 r_r1(Operand::RSI);
    const Reg64 r_r2(Operand::RBX);
    const Reg64 r_ow(10);
    const Reg64 r_out_row(11);
    const Reg64 r_oh(14);

    // Load arguments
    load_arg(0, 12);   // r12 = input
    load_arg(1, 13);   // r13 = output

    // Broadcast 9 weights from arg2
    load_arg(2, Operand::RAX);
    for (int i = 0; i < 9; i++)
        c.vbroadcastss(Zmm(4 + i), c.dword[c.rax + i * 4]);

    // Broadcast bias from arg3 (may be null)
    Label L_no_bias, L_bias_done;
    load_arg(3, Operand::RAX);
    c.test(c.rax, c.rax);
    c.jz(L_no_bias);
    c.vbroadcastss(Zmm(3), c.dword[c.rax]);
    c.jmp(L_bias_done);
    c.L(L_no_bias);
    c.vxorps(Zmm(3), Zmm(3), Zmm(3));
    c.L(L_bias_done);

    // Pre-load activation constants into zmm14/zmm15
    Label L_fmin, L_fmax, L_alpha;
    if (activation == jit_activation_t::relu) {
        c.vxorps(Zmm(15), Zmm(15), Zmm(15));
    } else if (activation == jit_activation_t::clip) {
        c.vbroadcastss(Zmm(14), c.dword[c.rip + L_fmin]);
        c.vbroadcastss(Zmm(15), c.dword[c.rip + L_fmax]);
    } else if (activation == jit_activation_t::leaky_relu) {
        c.vbroadcastss(Zmm(14), c.dword[c.rip + L_alpha]);
        c.vxorps(Zmm(15), Zmm(15), Zmm(15));
    }

    // === Top boundary oh rows (unrolled) ===
    for (int oh = 0; oh < std::min(oh_int_start, oH); oh++) {
        int ih0 = oh - pH;
        bool r0_ok = (ih0 >= 0 && ih0 < iH);
        bool r1_ok = (ih0 + 1 >= 0 && ih0 + 1 < iH);
        bool r2_ok = (ih0 + 2 >= 0 && ih0 + 2 < iH);

        // Compute valid row pointers
        if (r0_ok) {
            c.lea(r_r0, c.ptr[r_input + ih0 * iW_bytes]);
        }
        if (r1_ok) {
            c.lea(r_r1, c.ptr[r_input + (ih0 + 1) * iW_bytes]);
        }
        if (r2_ok) {
            c.lea(r_r2, c.ptr[r_input + (ih0 + 2) * iW_bytes]);
        }

        // Output row pointer
        c.lea(r_out_row, c.ptr[r_output + oh * oW_bytes]);

        emit_ow_row(c, oW, iW, pW, r0_ok, r1_ok, r2_ok, activation);
    }

    // === Interior oh loop ===
    if (oh_int_start < oh_int_end && oh_int_start < oH) {
        int actual_end = std::min(oh_int_end, oH);

        // r_r0 = input + (oh_int_start - pH) * iW_bytes = input + 0 (when oh_int_start == pH)
        c.lea(r_r0, c.ptr[r_input + (oh_int_start - pH) * iW_bytes]);
        c.lea(r_out_row, c.ptr[r_output + oh_int_start * oW_bytes]);
        c.mov(r_oh.cvt32(), oh_int_start);

        Label L_oh, L_oh_done;
        c.L(L_oh);
        c.cmp(r_oh.cvt32(), actual_end);
        c.jge(L_oh_done, c.T_NEAR);

        // r1 = r0 + iW_bytes, r2 = r0 + 2*iW_bytes
        c.lea(r_r1, c.ptr[r_r0 + iW_bytes]);
        c.lea(r_r2, c.ptr[r_r0 + iW_bytes * 2]);

        emit_ow_row(c, oW, iW, pW, true, true, true, activation);

        // Advance
        c.add(r_r0, iW_bytes);
        c.add(r_out_row, oW_bytes);
        c.inc(r_oh.cvt32());
        c.jmp(L_oh, c.T_NEAR);

        c.L(L_oh_done);
    }

    // === Bottom boundary oh rows (unrolled) ===
    for (int oh = std::max(oh_int_end, oh_int_start); oh < oH; oh++) {
        int ih0 = oh - pH;
        bool r0_ok = (ih0 >= 0 && ih0 < iH);
        bool r1_ok = (ih0 + 1 >= 0 && ih0 + 1 < iH);
        bool r2_ok = (ih0 + 2 >= 0 && ih0 + 2 < iH);

        if (r0_ok) c.lea(r_r0, c.ptr[r_input + ih0 * iW_bytes]);
        if (r1_ok) c.lea(r_r1, c.ptr[r_input + (ih0 + 1) * iW_bytes]);
        if (r2_ok) c.lea(r_r2, c.ptr[r_input + (ih0 + 2) * iW_bytes]);
        c.lea(r_out_row, c.ptr[r_output + oh * oW_bytes]);

        emit_ow_row(c, oW, iW, pW, r0_ok, r1_ok, r2_ok, activation);
    }

    emit_epilogue();

    // === Embedded constant data (after ret) ===
    c.align(4);
    auto float_bits = [](float f) -> uint32_t {
        uint32_t u; memcpy(&u, &f, 4); return u;
    };
    if (activation == jit_activation_t::clip) {
        c.L(L_fmin); c.dd(float_bits(params.fmin));
        c.L(L_fmax); c.dd(float_bits(params.fmax));
    } else if (activation == jit_activation_t::leaky_relu) {
        c.L(L_alpha); c.dd(float_bits(params.alpha));
    }

    finalize();
}

// ---------------------------------------------------------------------------
// Resolve (cached)
// ---------------------------------------------------------------------------
static jit_cache_t<jit_dw_key_t, jit_depthwise_avx512_t, jit_dw_hash_t>& dw_cache() {
    static jit_cache_t<jit_dw_key_t, jit_depthwise_avx512_t, jit_dw_hash_t> c;
    return c;
}

jit_dw_fn_t resolve_jit_depthwise(
    int iH, int iW, int oH, int oW, int pH, int pW,
    jit_activation_t activation, const jit_epilogue_params_t& params)
{
    jit_dw_key_t key{iH, iW, oH, oW, pH, pW, activation};
    auto* k = dw_cache().get_or_create(key, iH, iW, oH, oW, pH, pW, activation, params);
    return k->fn<jit_dw_fn_t>();
}

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
