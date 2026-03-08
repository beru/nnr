// JIT-compiled AVX-512 NCHWc direct convolution tile kernel.
//
// Key optimizations vs the intrinsics kernel (conv_nchwc_avx512.h):
// 1. Fully unrolled KH×KW×16 inner loop — zero loop overhead
// 2. Embedded broadcast FMA: vfmadd231ps zmm, zmm, [mem]{1to16}
//    saves one µop vs separate vbroadcastss + vfmadd
// 3. Weight loads interleaved with FMAs for OoO execution
// 4. Input row pointers pre-computed per kh (no recomputation per il)
// 5. Software prefetch for next ICb weight tile

#include "jit_conv_nchwc.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include <xbyak/xbyak.h>

namespace nnr {

jit_nchwc_conv_t::jit_nchwc_conv_t(int KH, int KW, int strideW, int ow_tile)
    : jit_kernel_t(65536)  // up to 64KB code (fully unrolled loops are large)
{
    using namespace Xbyak;
    using namespace Xbyak::util;
    auto& c = gen();

    const int kSpatial = KH * KW;
    const int tile_stride = strideW * 64;  // byte stride between tile positions in input
    const int w_icb_stride = kSpatial * 256 * 4;  // weight bytes per ICb iteration

    // ZMM register assignment
    // zmm0 .. zmm(ow_tile-1): output accumulators
    // zmm(ow_tile):           weight vector
    // zmm(ow_tile+1):         broadcast scratch (for separate vbroadcastss path)
    const int W_REG = ow_tile;

    // GPR assignment (after prologue loads args into these)
    const Reg64 reg_out     (Operand::R13);  // arg0: output pointer
    const Reg64 reg_in_base (Operand::RSI);  // arg1: input base (advances per ICb)
    const Reg64 reg_w_ptr   (Operand::R10);  // arg2: weight pointer (advances per ICb)
    const Reg64 reg_icb     (Operand::RDI);  // arg3: ICb counter
    const Reg64 reg_IW_bytes(Operand::R11);  // arg4: IW * 64
    const Reg64 reg_blk     (Operand::R12);  // arg5: IH * IW * 64
    const Reg64 reg_bias    (Operand::R14);  // arg6: bias pointer
    const Reg64 reg_row     (Operand::RBX);  // scratch: input row for kh > 0
    const Reg64 reg_tmp     (Operand::RAX);  // scratch

    // Callee-saved GPRs we use: rbx(3), r12(12), r13(13), r14(14)
    emit_prologue((1u << 3) | (1u << 12) | (1u << 13) | (1u << 14));

    // Load arguments into working registers
    load_arg(0, Operand::R13);   // out
    load_arg(1, Operand::RSI);   // in_base
    load_arg(2, Operand::R10);   // w_ob
    load_arg_i32(3, Operand::RDI);  // ICb (sign-extend to 64-bit)
    load_arg(4, Operand::R11);   // IW_bytes
    load_arg(5, Operand::R12);   // blk_bytes
    load_arg(6, Operand::R14);   // bias

    // --- Initialize accumulators from bias (or zero) ---
    // Test bias != nullptr
    c.test(reg_bias, reg_bias);
    Label no_bias;
    c.jz(no_bias);
    // Load bias into all accumulators
    c.vmovups(Zmm(0), ptr[reg_bias]);
    for (int t = 1; t < ow_tile; t++)
        c.vmovaps(Zmm(t), Zmm(0));
    Label bias_done;
    c.jmp(bias_done);
    c.L(no_bias);
    for (int t = 0; t < ow_tile; t++)
        c.vxorps(Zmm(t), Zmm(t), Zmm(t));
    c.L(bias_done);

    // --- ICb reduction loop ---
    Label icb_loop;
    c.L(icb_loop);

    // Unrolled KH × KW × 16 body.
    // For each (kh, kw), emit 16 weight loads + 16 × ow_tile FMAs.
    int w_offset = 0;  // byte offset from reg_w_ptr (accumulates through kh,kw)
    for (int kh = 0; kh < KH; kh++) {
        // Compute input row base for this kh.
        // kh=0: use reg_in_base directly
        // kh>0: reg_row = reg_in_base + kh * IW_bytes
        Reg64 row_base = (kh == 0) ? reg_in_base : reg_row;
        if (kh == 1) {
            c.lea(reg_row, ptr[reg_in_base + reg_IW_bytes]);
        } else if (kh == 2) {
            c.lea(reg_row, ptr[reg_in_base + reg_IW_bytes * 2]);
        } else if (kh > 2) {
            c.mov(reg_tmp, reg_IW_bytes);
            c.imul(reg_tmp, reg_tmp, kh);
            c.lea(reg_row, ptr[reg_in_base + reg_tmp]);
        }

        for (int kw = 0; kw < KW; kw++) {
            // Prefetch next weight tile (256*4 = 1024 bytes ahead)
            if (kh == 0 && kw == 0 && kSpatial > 1) {
                // Prefetch the second (kh=0,kw=1) tile into L1
                for (int pf = 0; pf < 16; pf += 4)
                    c.prefetcht0(ptr[reg_w_ptr + w_icb_stride + pf * 64]);
            }

            // For each input channel lane within this IC block
            const int B_REG = W_REG + 1;  // broadcast scratch
            for (int il = 0; il < 16; il++) {
                // Load weight vector: 16 OC values for this (icb, kh, kw, il)
                c.vmovups(Zmm(W_REG), ptr[reg_w_ptr + w_offset + il * 64]);

                // FMA for each tile position
                for (int t = 0; t < ow_tile; t++) {
                    int in_off = (t * strideW + kw) * 64 + il * 4;
                    c.vbroadcastss(Zmm(B_REG), dword[row_base + in_off]);
                    c.vfmadd231ps(Zmm(t), Zmm(W_REG), Zmm(B_REG));
                }
            }

            w_offset += 256 * 4;  // advance to next (kh,kw) weight tile: 1024 bytes
        }
    }

    // Advance pointers for next ICb iteration
    c.add(reg_in_base, reg_blk);           // in_base += IH * IW * 64
    c.add(reg_w_ptr, w_icb_stride);        // w_ptr += kSpatial * 1024
    c.dec(reg_icb);
    c.jnz(icb_loop, c.T_NEAR);

    // --- Store accumulators ---
    for (int t = 0; t < ow_tile; t++)
        c.vmovups(ptr[reg_out + t * 64], Zmm(t));

    emit_epilogue();
    finalize();
}

// Global cache for JIT NCHWc kernels
static jit_cache_t<jit_nchwc_key_t, jit_nchwc_conv_t, jit_nchwc_hash_t> g_nchwc_cache;

jit_nchwc_tile_fn_t resolve_jit_nchwc(int KH, int KW, int strideW, int ow_tile)
{
    jit_nchwc_key_t key{KH, KW, strideW, ow_tile};
    auto* k = g_nchwc_cache.get_or_create(key, KH, KW, strideW, ow_tile);
    return k->fn<jit_nchwc_tile_fn_t>();
}

// =========================================================================
// 4x6 JIT kernel: F OC blocks × ow_tile spatial positions.
//
// Key difference from 1x14: broadcasts are shared across F filter loads.
// Per (kh,kw,il): S broadcasts + F loads + F×S FMAs.
// KH×KW unrolled, il=0..15 is a runtime loop.
// =========================================================================

jit_nchwc_4x6_conv_t::jit_nchwc_4x6_conv_t(
    int KH, int KW, int strideW, int F, int ow_tile)
    : jit_kernel_t(65536)
{
    using namespace Xbyak;
    using namespace Xbyak::util;
    auto& c = gen();

    const int kSpatial = KH * KW;
    const int w_icb_stride = kSpatial * 256 * 4;  // weight bytes per ICb

    // ZMM layout: acc[f][s] = Zmm(f * ow_tile + s)
    auto acc = [&](int f, int s) -> Zmm { return Zmm(f * ow_tile + s); };
    const int WREG = F * ow_tile;                  // weight vector
    auto bcast = [&](int s) -> Zmm { return Zmm(WREG + 1 + s); };

    // GPR assignment
    const Reg64 reg_out     (Operand::R13);   // arg0
    const Reg64 reg_in_base (Operand::RSI);   // arg1 (advances per ICb)
    const Reg64 reg_w0      (Operand::R10);   // arg2 (advances per ICb)
    const Reg64 reg_icb     (Operand::RDI);   // arg3
    const Reg64 reg_IW_bytes(Operand::R11);   // arg4
    const Reg64 reg_blk     (Operand::R12);   // arg5
    const Reg64 reg_bias    (Operand::R14);   // arg6 (only used for init)
    const Reg64 reg_w_stride(Operand::R15);   // arg7: w_ob_stride
    const Reg64 reg_out_str (Operand::RBP);   // arg8: out_ob_stride
    const Reg64 reg_row     (Operand::RBX);   // scratch: input row
    const Reg64 reg_tmp     (Operand::RAX);   // scratch
    const Reg64 reg_il      (Operand::R9);    // il counter (0..15)
    // Weight pointers for OC blocks 1-3 (computed from w0 + f * w_stride)
    const Reg64 reg_w1      (Operand::RCX);
    const Reg64 reg_w2      (Operand::RDX);
    const Reg64 reg_w3      (Operand::R8);

    // Callee-saved: rbx(3), rbp(5), rsi(6), rdi(7), r12, r13, r14, r15
    uint32_t gpr_mask = (1u<<3)|(1u<<5)|(1u<<6)|(1u<<7)
                       |(1u<<12)|(1u<<13)|(1u<<14)|(1u<<15);

    // XMM save mask: callee-saved xmm6-xmm15 overlap with zmm6-zmm15.
    // Determine which zmm regs in 6-15 range we actually clobber.
    uint32_t xmm_mask = 0;
    int max_zmm = WREG + 1 + ow_tile - 1;  // highest zmm used
    for (int i = 6; i <= std::min(max_zmm, 15); i++)
        xmm_mask |= (1u << (i - 6));

    emit_prologue(gpr_mask, xmm_mask);

    // Load 9 arguments
    load_arg(0, Operand::R13);      // out
    load_arg(1, Operand::RSI);      // in_base
    load_arg(2, Operand::R10);      // w_base → w0
    load_arg_i32(3, Operand::RDI);  // ICb
    load_arg(4, Operand::R11);      // IW_bytes
    load_arg(5, Operand::R12);      // blk_bytes
    load_arg(6, Operand::R14);      // bias
    load_arg(7, Operand::R15);      // w_ob_stride
    load_arg(8, Operand::RBP);      // out_ob_stride

    // Compute w1, w2, w3 from w0
    if (F >= 2) c.lea(reg_w1, ptr[reg_w0 + reg_w_stride]);
    if (F >= 3) c.lea(reg_w2, ptr[reg_w0 + reg_w_stride * 2]);
    if (F >= 4) {
        c.lea(reg_w3, ptr[reg_w1 + reg_w_stride * 2]);  // w0+3*stride
    }

    // --- Initialize accumulators from bias (or zero) ---
    c.test(reg_bias, reg_bias);
    Label no_bias;
    c.jz(no_bias, c.T_NEAR);
    for (int f = 0; f < F; f++) {
        c.vmovups(acc(f, 0), ptr[reg_bias + f * 64]);
        for (int s = 1; s < ow_tile; s++)
            c.vmovaps(acc(f, s), acc(f, 0));
    }
    Label bias_done;
    c.jmp(bias_done, c.T_NEAR);
    c.L(no_bias);
    for (int f = 0; f < F; f++)
        for (int s = 0; s < ow_tile; s++)
            c.vxorps(acc(f, s), acc(f, s), acc(f, s));
    c.L(bias_done);

    // --- ICb reduction loop ---
    Label icb_loop;
    c.L(icb_loop);

    // Unrolled KH × KW body, with runtime il=0..15 loop
    int w_byte_off = 0;
    for (int kh = 0; kh < KH; kh++) {
        // Compute input row base for this kh
        Reg64 row_base = (kh == 0) ? reg_in_base : reg_row;
        if (kh == 1) {
            c.lea(reg_row, ptr[reg_in_base + reg_IW_bytes]);
        } else if (kh == 2) {
            c.lea(reg_row, ptr[reg_in_base + reg_IW_bytes * 2]);
        } else if (kh > 2) {
            c.mov(reg_tmp, reg_IW_bytes);
            c.imul(reg_tmp, reg_tmp, kh);
            c.lea(reg_row, ptr[reg_in_base + reg_tmp]);
        }

        for (int kw = 0; kw < KW; kw++) {
            // il loop: iterate over 16 input channels within one IC block
            c.xor_(reg_il, reg_il);
            Label il_loop;
            c.L(il_loop);

            // Compute weight byte offset for il: reg_tmp = il * 64
            c.mov(reg_tmp, reg_il);
            c.shl(reg_tmp, 6);

            // Broadcast ow_tile input scalars (reused across F filter loads)
            for (int s = 0; s < ow_tile; s++) {
                int in_off = (s * strideW + kw) * 64;  // byte offset for spatial pos s
                c.vbroadcastss(bcast(s), dword[row_base + reg_il * 4 + in_off]);
            }

            // For each OC block: load filter, FMA with all spatial positions
            Reg64 w_regs[] = { reg_w0, reg_w1, reg_w2, reg_w3 };
            for (int f = 0; f < F; f++) {
                c.vmovups(Zmm(WREG), ptr[w_regs[f] + reg_tmp + w_byte_off]);
                for (int s = 0; s < ow_tile; s++)
                    c.vfmadd231ps(acc(f, s), Zmm(WREG), bcast(s));
            }

            c.inc(reg_il);
            c.cmp(reg_il, 16);
            c.jb(il_loop, c.T_NEAR);

            w_byte_off += 256 * 4;  // 1024 bytes per (kh,kw) tile
        }
    }

    // Advance pointers for next ICb
    c.add(reg_in_base, reg_blk);
    if (w_icb_stride < 128) {
        c.add(reg_w0, w_icb_stride);
        if (F >= 2) c.add(reg_w1, w_icb_stride);
        if (F >= 3) c.add(reg_w2, w_icb_stride);
        if (F >= 4) c.add(reg_w3, w_icb_stride);
    } else {
        c.mov(reg_tmp, w_icb_stride);
        c.add(reg_w0, reg_tmp);
        if (F >= 2) c.add(reg_w1, reg_tmp);
        if (F >= 3) c.add(reg_w2, reg_tmp);
        if (F >= 4) c.add(reg_w3, reg_tmp);
    }
    c.dec(reg_icb);
    c.jnz(icb_loop, c.T_NEAR);

    // --- Store accumulators ---
    // OC block 0: directly at [reg_out + s * 64]
    for (int s = 0; s < ow_tile; s++)
        c.vmovups(ptr[reg_out + s * 64], acc(0, s));

    if (F >= 2) {
        // Reuse reg_w0 as out1 = reg_out + out_ob_stride
        c.lea(reg_w0, ptr[reg_out + reg_out_str]);
        for (int s = 0; s < ow_tile; s++)
            c.vmovups(ptr[reg_w0 + s * 64], acc(1, s));
    }
    if (F >= 3) {
        // out2 = reg_out + 2 * out_ob_stride
        c.lea(reg_w1, ptr[reg_out + reg_out_str * 2]);
        for (int s = 0; s < ow_tile; s++)
            c.vmovups(ptr[reg_w1 + s * 64], acc(2, s));
    }
    if (F >= 4) {
        // out3 = out1 + 2 * out_ob_stride = reg_out + 3 * out_ob_stride
        c.lea(reg_w2, ptr[reg_w0 + reg_out_str * 2]);
        for (int s = 0; s < ow_tile; s++)
            c.vmovups(ptr[reg_w2 + s * 64], acc(3, s));
    }

    emit_epilogue();
    finalize();
}

// Global cache for 4x6 JIT NCHWc kernels
static jit_cache_t<jit_nchwc_4x6_key_t, jit_nchwc_4x6_conv_t,
                    jit_nchwc_4x6_hash_t> g_nchwc_4x6_cache;

jit_nchwc_4x6_fn_t resolve_jit_nchwc_4x6(
    int KH, int KW, int strideW, int F, int ow_tile)
{
    jit_nchwc_4x6_key_t key{KH, KW, strideW, F, ow_tile};
    auto* k = g_nchwc_4x6_cache.get_or_create(key, KH, KW, strideW, F, ow_tile);
    return k->fn<jit_nchwc_4x6_fn_t>();
}

} // namespace nnr

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
