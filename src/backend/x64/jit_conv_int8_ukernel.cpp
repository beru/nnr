// JIT-compiled fused im2col + int8 GEMM micro-kernel for AVX-512 VNNI.
// MR=6 rows × NR=16 columns. Software-pipelined K-loop: loads/interleave
// for K-group k+1 overlap with VPDPBUSD chain for K-group k.
//
// Reads input pixels directly from pre-padded buffer — no im2col, no B-packing.
//
// Caller requirements:
//   - k_off[k] for k >= CHW must point to bytes with value 128 (so that after
//     vpsubb(-128), the signed value is 0 → zero contribution to VPDPBUSD and colsum).
//   - Weight buffer must be zero-padded at positions [CHW, K4) per row (uint8 = 0).
//   - x_pad must have ≥16 bytes of slack past the last valid pixel address.

#include "jit_conv_int8_ukernel.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include <xbyak/xbyak.h>

namespace nnr::int8 {

jit_conv_int8_ukernel_avx512_t::jit_conv_int8_ukernel_avx512_t(int mr_actual)
    : jit_kernel_t(16384)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;  // 1..6, baked at JIT time

    // ── Register map ───────────────────────────────────────────────────
    // GPRs:
    //   rdi(7)   W base, advances +4 per K-group
    //   rsi(6)   x_pad base (constant)
    //   r10      Y output base
    //   r11      k_off pointer, advances +32 per K-group
    //   r12      Kgroups counter (counts down)
    //   r13      CHW (setup only — freed after row offset computation)
    //   r14      y_stride (bytes)
    //   r15      col_sum output pointer
    //   rbx      scratch
    //   rax      scratch
    //
    // ZMM:
    //   zmm0     VNNI-interleaved input (current K-group)
    //   zmm1-4   XMM sub-regs: input loads + interleave scratch
    //   zmm5     column sum accumulator
    //   zmm6     constant 0x01 bytes (vpmaddubsw)
    //   zmm7     constant 0x0001 words (vpmaddwd)
    //   zmm8     constant 0x80 bytes (vpsubb shift)
    //   zmm9-12  scratch (broadcast, interleave temps)
    //   zmm14-25 row accumulators [0..MR-1] (zmm14+r; max zmm25 for MR=12)
    //
    //   k1       opmask for tail-column masked stores

    // Callee-saved: rbx(3), r12(12), r13(13), r14(14), r15(15)
    emit_prologue((1u << 3) | (1u << 12) | (1u << 13) | (1u << 14) | (1u << 15));

    const Reg64 rW(Operand::RDI), rX(Operand::RSI);
    const Reg64 rY(10), rKo(11), rKc(12), rYs(14), rCs(15);
    const Reg64 rT(Operand::RBX), rT2(Operand::RAX);

    // ── Load all arguments (before manual rsp adjustment) ──────────────
    load_arg(0, Operand::RDI);      // W
    load_arg(1, Operand::RSI);      // x_pad
    load_arg(2, 10);                // Y
    load_arg(3, 11);                // k_off
    load_arg_i32(4, 12);            // Kgroups
    load_arg_i32(5, 13);            // CHW (temporary, for row offsets)
    // arg6 = MR_actual: skip (baked at JIT time)
    load_arg_i32(7, 14);            // y_stride (int32 elements)
    load_arg(8, 15);                // col_sum
    load_arg_i32(9, Operand::RAX);  // mask (uint16)
    load_arg_i32(10, Operand::RBX); // zero_mode → rbx (callee-saved, already saved)

    c.shl(rYs, 2);                  // int32 elements → bytes
    c.kmovw(Opmask(1), Reg32(Operand::RAX));

    // ── Stack frame: row offsets ───────────────────────────────────────
    // [rsp + (r-1)*8] = r × CHW for r = 1..MR-1
    // Weight for row r at current K-group: [rW + row_offset[r]]
    const Reg64 rChw(13);
    const int FRAME = ((MR > 1 ? (MR - 1) * 8 : 0) + 15) & ~15;
    if (FRAME > 0) {
        c.sub(c.rsp, FRAME);
        for (int r = 1; r < MR; r++) {
            c.imul(rT2, rChw, r);
            c.mov(c.qword[c.rsp + (r - 1) * 8], rT2);
        }
    }

    // ── Constants ──────────────────────────────────────────────────────
    // vpternlogd(zmm, zmm, zmm, 0xFF) → all ones (AVX-512 idiom for vpcmpeqd)
    c.vpternlogd(Zmm(6), Zmm(6), Zmm(6), 0xFF);
    c.vpabsb(Zmm(6), Zmm(6));              // zmm6 = 0x01 bytes
    c.vpternlogd(Zmm(7), Zmm(7), Zmm(7), 0xFF);
    c.vpsrlw(Zmm(7), Zmm(7), 15);          // zmm7 = 0x0001 words
    c.mov(Reg32(Operand::RAX), 0x80808080);
    c.vmovd(Xmm(8), Reg32(Operand::RAX));
    c.vpbroadcastd(Zmm(8), Xmm(8));        // zmm8 = 0x80 bytes

    // ── Init accumulators ────────────────────────────────────────────
    // zero_mode=1: zero accumulators (first KC block)
    // zero_mode=0: load from Y (accumulate across KC blocks)
    c.vpxord(Zmm(5), Zmm(5), Zmm(5));      // colsum accumulator always starts at 0
    c.test(Reg32(Operand::RBX), Reg32(Operand::RBX));
    c.jnz("zm_zero", CodeGenerator::T_NEAR);
    // Load accumulators from Y (accumulate mode)
    {
        c.mov(rT2, rY);
        c.vmovdqu32(Zmm(14), c.ptr[rT2]);
        for (int r = 1; r < MR; r++) {
            c.add(rT2, rYs);
            c.vmovdqu32(Zmm(14 + r), c.ptr[rT2]);
        }
    }
    c.jmp("zm_done", CodeGenerator::T_NEAR);
    c.L("zm_zero");
    for (int r = 0; r < MR; r++)
        c.vpxord(Zmm(14 + r), Zmm(14 + r), Zmm(14 + r));
    c.L("zm_done");

    // ── Codegen helpers ────────────────────────────────────────────────

    // Load 4 input sub-indices, subtract 128, interleave → zmm0.
    // koff_off: byte offset from rKo to the first of 4 size_t offsets.
    auto emit_load_sub_interleave = [&](int koff_off) {
        for (int i = 0; i < 4; i++) {
            c.mov(rT2, c.qword[rKo + koff_off + i * 8]);
            c.vmovdqu32(Xmm(1 + i), c.ptr[rX + rT2]);
            c.vpsubb(Xmm(1 + i), Xmm(1 + i), Xmm(8));
        }
        // Byte interleave: 4×16 → 16 VNNI uint32
        c.vpunpcklbw(Xmm(9), Xmm(1), Xmm(2));    // lo01
        c.vpunpckhbw(Xmm(10), Xmm(1), Xmm(2));   // hi01
        c.vpunpcklbw(Xmm(11), Xmm(3), Xmm(4));   // lo23
        c.vpunpckhbw(Xmm(12), Xmm(3), Xmm(4));   // hi23
        c.vpunpcklwd(Xmm(1), Xmm(9), Xmm(11));   // q0
        c.vpunpckhwd(Xmm(2), Xmm(9), Xmm(11));   // q1
        c.vpunpcklwd(Xmm(3), Xmm(10), Xmm(12));  // q2
        c.vpunpckhwd(Xmm(4), Xmm(10), Xmm(12));  // q3
        c.vpxord(Zmm(0), Zmm(0), Zmm(0));         // break false dep
        c.vinserti32x4(Zmm(0), Zmm(0), Xmm(1), 0);
        c.vinserti32x4(Zmm(0), Zmm(0), Xmm(2), 1);
        c.vinserti32x4(Zmm(0), Zmm(0), Xmm(3), 2);
        c.vinserti32x4(Zmm(0), Zmm(0), Xmm(4), 3);
    };

    // VPDPBUSD for all MR rows using current zmm0 and weight at rW.
    // vmovd+vpbroadcastd(xmm) uses ports 0/1/5 instead of port 10,
    // freeing port 10 from the vpbroadcastd bottleneck.
    auto emit_vpdpbusd_all = [&]() {
        c.vmovd(Xmm(9), c.dword[rW]);
        c.vpbroadcastd(Zmm(9), Xmm(9));
        c.vpdpbusd(Zmm(14), Zmm(9), Zmm(0));
        for (int r = 1; r < MR; r++) {
            c.mov(rT, c.qword[c.rsp + (r - 1) * 8]);
            c.vmovd(Xmm(9), c.dword[rW + rT]);
            c.vpbroadcastd(Zmm(9), Xmm(9));
            c.vpdpbusd(Zmm(14 + r), Zmm(9), Zmm(0));
        }
    };

    // Column sum: accumulate sum of signed bytes in zmm0 → zmm5.
    auto emit_colsum = [&]() {
        c.vpmaddubsw(Zmm(10), Zmm(6), Zmm(0));   // ones8 × vnni → 16-bit
        c.vpmaddwd(Zmm(10), Zmm(10), Zmm(7));     // → 32-bit
        c.vpaddd(Zmm(5), Zmm(5), Zmm(10));
    };

    // ── K-loop ─────────────────────────────────────────────────────────

    c.test(rKc, rKc);
    c.jz("k_done", CodeGenerator::T_NEAR);

    // Prologue: load K-group 0 → zmm0
    emit_load_sub_interleave(0);

    c.dec(rKc);
    c.jz("k_last", CodeGenerator::T_NEAR);

    // Main loop: process current zmm0 + load next → zmm0 (pipelined)
    Label k_loop;
    c.L(k_loop);
    {
        // VPDPBUSD for MR rows, interleaved with loading next K-group.
        // Each row emits: weight broadcast + VPDPBUSD.
        // For rows 0..min(MR,4)-1: also emit one input load+sub for next group.
        for (int r = 0; r < MR; r++) {
            // Weight load + register-form broadcast (avoids port 10 bottleneck)
            if (r == 0) {
                c.vmovd(Xmm(9), c.dword[rW]);
            } else {
                c.mov(rT, c.qword[c.rsp + (r - 1) * 8]);
                c.vmovd(Xmm(9), c.dword[rW + rT]);
            }
            c.vpbroadcastd(Zmm(9), Xmm(9));
            c.vpdpbusd(Zmm(14 + r), Zmm(9), Zmm(0));

            // Interleave one next-group load+sub per VPDPBUSD row
            if (r < 4) {
                c.mov(rT2, c.qword[rKo + 32 + r * 8]);
                c.vmovdqu32(Xmm(1 + r), c.ptr[rX + rT2]);
                c.vpsubb(Xmm(1 + r), Xmm(1 + r), Xmm(8));
            }
        }
        // Remaining loads+subs if MR < 4
        for (int r = MR; r < 4; r++) {
            c.mov(rT2, c.qword[rKo + 32 + r * 8]);
            c.vmovdqu32(Xmm(1 + r), c.ptr[rX + rT2]);
            c.vpsubb(Xmm(1 + r), Xmm(1 + r), Xmm(8));
        }

        // Column sum for current K-group (must read zmm0 before overwrite)
        emit_colsum();

        // Interleave next K-group → zmm0
        c.vpunpcklbw(Xmm(9), Xmm(1), Xmm(2));
        c.vpunpckhbw(Xmm(10), Xmm(1), Xmm(2));
        c.vpunpcklbw(Xmm(11), Xmm(3), Xmm(4));
        c.vpunpckhbw(Xmm(12), Xmm(3), Xmm(4));
        c.vpunpcklwd(Xmm(1), Xmm(9), Xmm(11));
        c.vpunpckhwd(Xmm(2), Xmm(9), Xmm(11));
        c.vpunpcklwd(Xmm(3), Xmm(10), Xmm(12));
        c.vpunpckhwd(Xmm(4), Xmm(10), Xmm(12));
        c.vpxord(Zmm(0), Zmm(0), Zmm(0));
        c.vinserti32x4(Zmm(0), Zmm(0), Xmm(1), 0);
        c.vinserti32x4(Zmm(0), Zmm(0), Xmm(2), 1);
        c.vinserti32x4(Zmm(0), Zmm(0), Xmm(3), 2);
        c.vinserti32x4(Zmm(0), Zmm(0), Xmm(4), 3);

        // Advance pointers
        c.add(rW, 4);
        c.add(rKo, 32);
        c.dec(rKc);
        c.jnz(k_loop);
    }

    // Epilogue: process last K-group (already in zmm0, no next to load)
    c.L("k_last");
    emit_vpdpbusd_all();
    emit_colsum();

    c.L("k_done");

    // ── Store results ──────────────────────────────────────────────────

    // Column sums (unmasked — 16 lanes; caller masks valid ones)
    c.vmovdqu32(c.ptr[rCs], Zmm(5));

    // Accumulators: masked store via k1, running pointer for rows 1+
    c.vmovdqu32(c.ptr[rY] | Opmask(1), Zmm(14));
    if (MR > 1) {
        c.mov(rT2, rY);
        for (int r = 1; r < MR; r++) {
            c.add(rT2, rYs);
            c.vmovdqu32(c.ptr[rT2] | Opmask(1), Zmm(14 + r));
        }
    }

    // ── Cleanup ────────────────────────────────────────────────────────
    if (FRAME > 0)
        c.add(c.rsp, FRAME);

    emit_epilogue();
    finalize();
}

// ── Packed-input GEMM JIT ────────────────────────────────────────────
// ORT-style: reads pre-interleaved VNNI data from sequential buffer.
// K-loop: zmm load + MR broadcasts + MR VPDPBUSDs. No interleave.
// Accumulates into Y (load + add + store) for K-blocking support.

jit_packed_gemm_int8_ukernel_avx512_t::jit_packed_gemm_int8_ukernel_avx512_t(int mr_actual)
    : jit_kernel_t(16384)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;

    // Callee-saved: rbx(3), r13(13), r14(14)
    emit_prologue((1u << 3) | (1u << 13) | (1u << 14));

    const Reg64 rW(Operand::RDI), rP(Operand::RSI);
    const Reg64 rY(10), rKc(11), rChw(13), rYs(14);
    const Reg64 rWsave(Operand::RBX), rT2(Operand::RAX);

    load_arg(0, Operand::RDI);      // W
    load_arg(1, Operand::RSI);      // packed
    load_arg(2, 10);                // Y
    load_arg_i32(3, 11);            // Kgroups
    load_arg_i32(4, 13);            // CHW
    load_arg_i32(5, 14);            // y_stride
    load_arg_i32(6, Operand::RAX);  // mask

    c.shl(rYs, 2);
    c.kmovw(Opmask(1), Reg32(Operand::RAX));

    // Load accumulators from Y (for K-block accumulation)
    c.vmovdqu32(Zmm(14) | Opmask(1), c.ptr[rY]);
    if (MR > 1) {
        c.mov(rT2, rY);
        for (int r = 1; r < MR; r++) {
            c.add(rT2, rYs);
            c.vmovdqu32(Zmm(14 + r) | Opmask(1), c.ptr[rT2]);
        }
    }

    // K-loop: running W pointer
    c.test(rKc, rKc);
    c.jz("k_done", CodeGenerator::T_NEAR);

    Label k_loop;
    c.L(k_loop);
    {
        c.vmovdqu32(Zmm(0), c.ptr[rP]);
        c.mov(rWsave, rW);
        for (int r = 0; r < MR; r++) {
            c.vpbroadcastd(Zmm(9), c.dword[rW]);
            c.vpdpbusd(Zmm(14 + r), Zmm(9), Zmm(0));
            if (r < MR - 1)
                c.add(rW, rChw);
        }
        c.lea(rW, c.ptr[rWsave + 4]);
        c.add(rP, 64);
        c.dec(rKc);
        c.jnz(k_loop);
    }

    c.L("k_done");

    // Store accumulators
    c.vmovdqu32(c.ptr[rY] | Opmask(1), Zmm(14));
    if (MR > 1) {
        c.mov(rT2, rY);
        for (int r = 1; r < MR; r++) {
            c.add(rT2, rYs);
            c.vmovdqu32(c.ptr[rT2] | Opmask(1), Zmm(14 + r));
        }
    }

    emit_epilogue();
    finalize();
}

// ── Packed GEMM JIT, NR=32 (2 ZMM columns) ──────────────────────────
// MR=1..6 baked at JIT time. 12 accumulators: zmm14-19 (col0), zmm20-25 (col1).
// K-loop: 2 zmm loads from packed B, MR broadcasts from A, 2×MR VPDPBUSDs.
// Running A pointer (advance by lda per row, reset per K-group).
// Epilogue: add row_sums + col_sums, masked store.

jit_packed_gemm_nr32_avx512_t::jit_packed_gemm_nr32_avx512_t(int mr_actual)
    : jit_kernel_t(16384)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;  // 1..6

    // Callee-saved: rbx(3), r12(12), r13(13), r14(14), r15(15)
    emit_prologue((1u << 3) | (1u << 12) | (1u << 13) | (1u << 14) | (1u << 15));

    // GPR assignments — 6 independent A-row pointers eliminate serial
    // dependency chain in K-loop (old: running rA + add rLda per row)
    const Reg64 rP(Operand::RSI);    // packed B pointer (running)
    const Reg64 rC(10);              // C output base
    const Reg64 rKc(11);             // Kgroups counter
    const Reg64 rLdc(14);            // ldc (C row stride in bytes)
    const Reg64 rRs(15);             // row_sums pointer
    const Reg64 rT(Operand::RAX);    // scratch
    const Reg64 rCs(12);             // col_sums pointer

    // A-row registers: one per row, all advance by 4 independently per K-step.
    // After load_arg, rcx/rdx/r8/r9 are free (Win64 arg regs, values moved out).
    const Reg64 rArow[6] = {
        Reg64(Operand::RDI),   // row 0 (= A arg)
        Reg64(Operand::RCX),   // row 1
        Reg64(Operand::RDX),   // row 2
        Reg64(8),              // row 3 (r8)
        Reg64(9),              // row 4 (r9)
        Reg64(Operand::RBX),   // row 5 (callee-saved, already pushed)
    };
    const Reg64 rLda(13);    // lda — only needed for row-pointer setup

    // Load arguments
    load_arg(0, Operand::RDI);       // A → rArow[0]
    load_arg(1, Operand::RSI);       // packed
    load_arg(2, 10);                 // C
    load_arg_i32(3, 11);             // Kgroups
    load_arg_i32(4, 13);             // lda (temporary)
    load_arg_i32(5, 14);             // ldc
    load_arg(6, 15);                 // row_sums
    load_arg(7, 12);                 // col_sums
    load_arg_i32(8, Operand::RAX);   // mask_lo
    c.kmovw(Opmask(1), Reg32(Operand::RAX));
    load_arg_i32(9, Operand::RAX);   // mask_hi
    c.kmovw(Opmask(2), Reg32(Operand::RAX));

    c.shl(rLdc, 2);                  // int32 elements → bytes

    // Set up independent A-row pointers: Ar = A + r * lda
    for (int r = 1; r < MR; r++)
        c.lea(rArow[r], c.ptr[rArow[r - 1] + rLda]);
    // rLda (r13) no longer needed — reload as ldb (packed B K-group stride in bytes)
    load_arg_i32(10, 13);   // ldb → r13

    // Zero-initialize accumulators (single K-pass, no K-blocking)
    for (int r = 0; r < MR; r++) {
        c.vpxord(Zmm(14 + r), Zmm(14 + r), Zmm(14 + r));   // col0 accumulators
        c.vpxord(Zmm(20 + r), Zmm(20 + r), Zmm(20 + r));   // col1 accumulators
    }

    // K-loop: all row broadcasts are independent (no serial dependency)
    c.test(rKc, rKc);
    c.jz("k_done", CodeGenerator::T_NEAR);

    Label k_loop;
    c.L(k_loop);
    {
        c.vmovdqu32(Zmm(0), c.ptr[rP]);         // B col0
        c.vmovdqu32(Zmm(1), c.ptr[rP + 64]);    // B col1

        for (int r = 0; r < MR; r++) {
            c.vpbroadcastd(Zmm(9), c.dword[rArow[r]]);
            c.vpdpbusd(Zmm(14 + r), Zmm(9), Zmm(0));
            c.vpdpbusd(Zmm(20 + r), Zmm(9), Zmm(1));
            c.add(rArow[r], 4);  // independent: no cross-row dependency
        }

        c.add(rP, rLda);   // advance packed B by ldb bytes (r13 reloaded as ldb)
        c.dec(rKc);
        c.jnz(k_loop);
    }
    c.L("k_done");

    // Epilogue: add col_sums + row_sums, masked store
    {
        c.vmovdqu32(Zmm(2) | Opmask(1), c.ptr[rCs]);        // col_sums lo
        c.vmovdqu32(Zmm(3) | Opmask(2), c.ptr[rCs + 64]);   // col_sums hi

        c.mov(rT, rC);
        for (int r = 0; r < MR; r++) {
            c.vpbroadcastd(Zmm(9), c.dword[rRs + r * 4]);   // row_sums[r]
            // col0
            c.vpaddd(Zmm(14 + r), Zmm(14 + r), Zmm(2));
            c.vpaddd(Zmm(14 + r), Zmm(14 + r), Zmm(9));
            c.vmovdqu32(c.ptr[rT] | Opmask(1), Zmm(14 + r));
            // col1
            c.vpaddd(Zmm(20 + r), Zmm(20 + r), Zmm(3));
            c.vpaddd(Zmm(20 + r), Zmm(20 + r), Zmm(9));
            c.vmovdqu32(c.ptr[rT + 64] | Opmask(2), Zmm(20 + r));
            if (r < MR - 1) c.add(rT, rLdc);
        }
    }

    emit_epilogue();
    finalize();
}

// ── Fused im2col NR=32 (2 ZMM column groups) ─────────────────────────
// MR=1..10 baked at JIT time.
// ZMM layout (all 32):
//   zmm0-9   col0 accumulators  zmm10-19  col1 accumulators
//   zmm20    colsum col0        zmm21     colsum col1
//   zmm22    ones8   zmm23      ones16    zmm24   0x80 bytes
//   zmm25    A broadcast / colsum temp
//   zmm26    B col0 interleaved VNNI     zmm27  B col1
//   zmm28-31 interleave temporaries

jit_conv_int8_nr32_ukernel_avx512_t::jit_conv_int8_nr32_ukernel_avx512_t(int mr_actual)
    : jit_kernel_t(32768)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;

    emit_prologue((1u<<3)|(1u<<12)|(1u<<13)|(1u<<14)|(1u<<15));

    const Reg64 rW(Operand::RDI), rX(Operand::RSI);
    const Reg64 rY(10), rKo(11), rKc(12), rYs(14), rCs(15);
    const Reg64 rT(Operand::RBX), rT2(Operand::RAX);

    load_arg(0, Operand::RDI);
    load_arg(1, Operand::RSI);
    load_arg(2, 10);
    load_arg(3, 11);
    load_arg_i32(4, 12);
    load_arg_i32(5, 13);   // CHW → r13 (temp for frame)
    // arg6 = MR_actual: skip
    load_arg_i32(7, 14);   // y_stride
    load_arg(8, 15);       // col_sum base
    load_arg_i32(9, Operand::RAX);
    c.kmovw(Opmask(1), Reg32(Operand::RAX));   // mask0
    load_arg_i32(10, Operand::RAX);
    c.kmovw(Opmask(2), Reg32(Operand::RAX));   // mask1
    load_arg_i32(11, Operand::RBX);             // zero_mode → rbx

    c.shl(rYs, 2);

    const int FRAME = ((MR > 1 ? (MR-1)*8 : 0) + 15) & ~15;
    if (FRAME > 0) {
        c.sub(c.rsp, FRAME);
        const Reg64 rChw(13);
        for (int r = 1; r < MR; r++) {
            c.imul(rT2, rChw, r);
            c.mov(c.qword[c.rsp + (r-1)*8], rT2);
        }
    }

    // Constants
    c.vpternlogd(Zmm(22), Zmm(22), Zmm(22), 0xFF);
    c.vpabsb(Zmm(22), Zmm(22));
    c.vpternlogd(Zmm(23), Zmm(23), Zmm(23), 0xFF);
    c.vpsrlw(Zmm(23), Zmm(23), 15);
    c.mov(Reg32(Operand::RAX), 0x80808080);
    c.vmovd(Xmm(24), Reg32(Operand::RAX));
    c.vpbroadcastd(Zmm(24), Xmm(24));

    // Init accumulators + colsums (zero_mode=1: zero, zero_mode=0: load from Y)
    c.vpxord(Zmm(20), Zmm(20), Zmm(20));   // colsums always start at 0
    c.vpxord(Zmm(21), Zmm(21), Zmm(21));
    c.test(Reg32(Operand::RBX), Reg32(Operand::RBX));
    c.jnz("zm_zero32", CodeGenerator::T_NEAR);
    // Accumulate mode: load from Y
    {
        c.mov(rT2, rY);
        c.vmovdqu32(Zmm(0), c.ptr[rT2]);
        c.vmovdqu32(Zmm(10), c.ptr[rT2 + 64]);
        for (int r = 1; r < MR; r++) {
            c.add(rT2, rYs);
            c.vmovdqu32(Zmm(r), c.ptr[rT2]);
            c.vmovdqu32(Zmm(10 + r), c.ptr[rT2 + 64]);
        }
    }
    c.jmp("zm_done32", CodeGenerator::T_NEAR);
    c.L("zm_zero32");
    for (int r = 0; r < MR; r++) {
        c.vpxord(Zmm(r),    Zmm(r),    Zmm(r));
        c.vpxord(Zmm(10+r), Zmm(10+r), Zmm(10+r));
    }
    c.L("zm_done32");

    // 4 xmm loads+sub into xmm28-31 from k_off[koff_base..+3] + col_off.
    auto emit_raw_loads = [&](int koff_base, int col_off) {
        for (int i = 0; i < 4; i++) {
            c.mov(rT2, c.qword[rKo + koff_base + i*8]);
            if (col_off == 0)
                c.vmovdqu32(Xmm(28+i), c.ptr[rX + rT2]);
            else
                c.vmovdqu32(Xmm(28+i), c.ptr[rX + rT2 + col_off]);
            c.vpsubb(Xmm(28+i), Xmm(28+i), Xmm(24));
        }
    };
    // xmm28-31 → zmm(dest). Uses zmm25 and xmm(dest) as temps.
    auto emit_unpack = [&](int dest) {
        c.vpunpcklbw(Xmm(25),   Xmm(28), Xmm(29));
        c.vpunpckhbw(Xmm(dest), Xmm(28), Xmm(29));
        c.vpunpcklbw(Xmm(28),   Xmm(30), Xmm(31));
        c.vpunpckhbw(Xmm(29),   Xmm(30), Xmm(31));
        c.vpunpcklwd(Xmm(30), Xmm(25),   Xmm(28));
        c.vpunpckhwd(Xmm(31), Xmm(25),   Xmm(28));
        c.vpunpcklwd(Xmm(25), Xmm(dest), Xmm(29));
        c.vpunpckhwd(Xmm(28), Xmm(dest), Xmm(29));
        c.vpxord(Zmm(dest), Zmm(dest), Zmm(dest));
        c.vinserti32x4(Zmm(dest), Zmm(dest), Xmm(30), 0);
        c.vinserti32x4(Zmm(dest), Zmm(dest), Xmm(31), 1);
        c.vinserti32x4(Zmm(dest), Zmm(dest), Xmm(25), 2);
        c.vinserti32x4(Zmm(dest), Zmm(dest), Xmm(28), 3);
    };
    auto emit_interleave = [&](int dest, int koff_base, int col_off) {
        emit_raw_loads(koff_base, col_off);
        emit_unpack(dest);
    };
    // VPDPBUSDs for all rows using zmm26/27. During rows 0..nload-1, also load
    // next[i] = x_pad[k_off_next[i] + col_off] - 0x80 into xmm(28+i).
    auto emit_rows_with_loads = [&](int nload, int koff_next, int col_off) {
        for (int r = 0; r < MR; r++) {
            if (r == 0) {
                c.vmovd(Xmm(25), c.dword[rW]);
            } else {
                c.mov(rT, c.qword[c.rsp + (r-1)*8]);
                c.vmovd(Xmm(25), c.dword[rW + rT]);
            }
            c.vpbroadcastd(Zmm(25), Xmm(25));
            c.vpdpbusd(Zmm(r),    Zmm(25), Zmm(26));
            c.vpdpbusd(Zmm(10+r), Zmm(25), Zmm(27));
            if (r < nload) {
                c.mov(rT2, c.qword[rKo + koff_next + r*8]);
                if (col_off == 0)
                    c.vmovdqu32(Xmm(28+r), c.ptr[rX + rT2]);
                else
                    c.vmovdqu32(Xmm(28+r), c.ptr[rX + rT2 + col_off]);
                c.vpsubb(Xmm(28+r), Xmm(28+r), Xmm(24));
            }
        }
        // Any remaining loads if MR < nload
        for (int r = MR; r < nload; r++) {
            c.mov(rT2, c.qword[rKo + koff_next + r*8]);
            if (col_off == 0)
                c.vmovdqu32(Xmm(28+r), c.ptr[rX + rT2]);
            else
                c.vmovdqu32(Xmm(28+r), c.ptr[rX + rT2 + col_off]);
            c.vpsubb(Xmm(28+r), Xmm(28+r), Xmm(24));
        }
    };
    auto emit_colsums = [&]() {
        c.vpmaddubsw(Zmm(25), Zmm(22), Zmm(26));
        c.vpmaddwd(Zmm(25), Zmm(25), Zmm(23));
        c.vpaddd(Zmm(20), Zmm(20), Zmm(25));
        c.vpmaddubsw(Zmm(25), Zmm(22), Zmm(27));
        c.vpmaddwd(Zmm(25), Zmm(25), Zmm(23));
        c.vpaddd(Zmm(21), Zmm(21), Zmm(25));
    };
    auto emit_vpdpbusd_all = [&]() {
        for (int r = 0; r < MR; r++) {
            if (r == 0) c.vmovd(Xmm(25), c.dword[rW]);
            else { c.mov(rT, c.qword[c.rsp + (r-1)*8]); c.vmovd(Xmm(25), c.dword[rW + rT]); }
            c.vpbroadcastd(Zmm(25), Xmm(25));
            c.vpdpbusd(Zmm(r),    Zmm(25), Zmm(26));
            c.vpdpbusd(Zmm(10+r), Zmm(25), Zmm(27));
        }
    };

    c.test(rKc, rKc);
    c.jz("k_done", CodeGenerator::T_NEAR);

    // Prologue: load K[0] → zmm26, zmm27
    emit_interleave(26, 0,  0);
    emit_interleave(27, 0, 16);

    c.dec(rKc);
    c.jz("k_last", CodeGenerator::T_NEAR);

    // Pipelined main loop:
    // - colsums from current zmm26/27
    // - VPDPBUSDs for all rows; during rows 0-3 load next col0 → xmm28-31
    // - unpack col0 → zmm26 (next K)
    // - VPDPBUSDs again for all rows (col1); during rows 0-3 load next col1
    // - unpack col1 → zmm27 (next K)
    // Note: two VPDPBUSD passes per K-step doubles broadcasts but avoids register conflicts.
    Label k_loop;
    c.L(k_loop);
    {
        emit_colsums();
        emit_rows_with_loads(4, 32, 0);   // col0+col1 rows, embed 4 next-col0 loads
        emit_unpack(26);                   // xmm28-31 → zmm26 (next col0)
        emit_raw_loads(32, 16);            // load next col1 into xmm28-31
        emit_unpack(27);                   // → zmm27 (next col1)

        c.add(rW, 4);
        c.add(rKo, 32);
        c.dec(rKc);
        c.jnz(k_loop);
    }

    // Epilogue: last K-group in zmm26/27, no next to load
    c.L("k_last");
    emit_colsums();
    emit_vpdpbusd_all();

    c.L("k_done");

    // Store colsums: col0 at [rCs], col1 at [rCs+64]
    c.vmovdqu32(c.ptr[rCs],      Zmm(20));
    c.vmovdqu32(c.ptr[rCs + 64], Zmm(21));

    // Store accumulators
    c.vmovdqu32(c.ptr[rY]      | Opmask(1), Zmm(0));
    c.vmovdqu32(c.ptr[rY + 64] | Opmask(2), Zmm(10));
    if (MR > 1) {
        c.mov(rT2, rY);
        for (int r = 1; r < MR; r++) {
            c.add(rT2, rYs);
            c.vmovdqu32(c.ptr[rT2]      | Opmask(1), Zmm(r));
            c.vmovdqu32(c.ptr[rT2 + 64] | Opmask(2), Zmm(10+r));
        }
    }

    if (FRAME > 0) c.add(c.rsp, FRAME);
    emit_epilogue();
    finalize();
}

// ── Gather-GEMM NR=48 (3 ZMM columns, pre-packed weights) ────────────
// MR=1..6 baked at JIT time.  Transposed GEMM: M=spatial, N=OC, K=CHW.
// ZMM layout:
//   zmm0-5   accum col0 (ch 0-15)   zmm6-11  accum col1 (ch 16-31)
//   zmm12-17 accum col2 (ch 32-47)
//   zmm18    B col0   zmm19  B col1   zmm20  B col2  (packed weight loads)
//   zmm21    A broadcast (gathered input uint8)
// GPR: rsi=x_pad, rdi=W(running), r10=Y, r11=Kgroups, rbx=k_off(running),
//      r14=ldc_bytes, rcx/rdx/r8/r9=k_off[0..3], rax/rbp=scratch

jit_gather_gemm_nr48_avx512_t::jit_gather_gemm_nr48_avx512_t(int mr_actual)
    : jit_kernel_t(32768)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;

    // Callee-saved: rbx, rbp, rsi, rdi, r12, r13, r14, r15
    emit_prologue((1u<<3)|(1u<<5)|(1u<<6)|(1u<<7)|(1u<<12)|(1u<<13)|(1u<<14)|(1u<<15));

    const Reg64 rX(Operand::RSI);     // x_pad base
    const Reg64 rW(Operand::RDI);     // W_packed running pointer
    const Reg64 rAdj(12);             // adj_col pointer
    const Reg64 rKc(11);              // Kgroups counter
    const Reg64 rKo(Operand::RBX);    // k_off running pointer
    const Reg64 rRq(13);              // gather_rq_t* pointer
    const Reg64 rK0(Operand::RCX), rK1(Operand::RDX), rK2(8), rK3(9);
    const Reg64 rT(Operand::RAX);
    const Reg64 rT2(Operand::RBP);

    // Load arguments (9 args, Win64 ABI)
    load_arg(0, Operand::RSI);        // x_pad
    load_arg(1, Operand::RDI);        // W_packed
    load_arg(2, 12);                  // adj_col → r12
    load_arg_i32(3, 11);              // Kgroups
    load_arg(4, Operand::RBX);        // k_off
    load_arg(5, 13);                  // rq → r13
    load_arg_i32(6, Operand::RAX);    // mask0
    c.kmovw(Opmask(1), Reg32(Operand::RAX));
    load_arg_i32(7, Operand::RAX);    // mask1
    c.kmovw(Opmask(2), Reg32(Operand::RAX));
    load_arg_i32(8, Operand::RAX);    // mask2
    c.kmovw(Opmask(3), Reg32(Operand::RAX));

    // Pre-bias accumulators with adj_col (ZP correction folded into init)
    for (int r = 0; r < MR; r++) {
        c.vmovdqu32(Zmm(r),    c.ptr[rAdj]);        // col0: adj_col[0..15]
        c.vmovdqu32(Zmm(6+r),  c.ptr[rAdj + 64]);   // col1: adj_col[16..31]
        c.vmovdqu32(Zmm(12+r), c.ptr[rAdj + 128]);  // col2: adj_col[32..47]
    }

    // K-loop (unchanged — gather + broadcast + 3x VPDPBUSD)
    c.test(rKc, rKc);
    c.jz("k_done", CodeGenerator::T_NEAR);

    Label k_loop;
    c.L(k_loop);
    {
        c.vmovdqu32(Zmm(18), c.ptr[rW]);
        c.vmovdqu32(Zmm(19), c.ptr[rW + 64]);
        c.vmovdqu32(Zmm(20), c.ptr[rW + 128]);
        c.mov(rK0, c.qword[rKo]);
        c.mov(rK1, c.qword[rKo + 8]);
        c.mov(rK2, c.qword[rKo + 16]);
        c.mov(rK3, c.qword[rKo + 24]);

        for (int r = 0; r < MR; r++) {
            c.movzx(Reg32(Operand::RAX), c.byte[rX + rK0 + r]);
            c.movzx(Reg32(Operand::RBP), c.byte[rX + rK1 + r]);
            c.shl(Reg32(Operand::RBP), 8);
            c.or_(Reg32(Operand::RAX), Reg32(Operand::RBP));
            c.movzx(Reg32(Operand::RBP), c.byte[rX + rK2 + r]);
            c.shl(Reg32(Operand::RBP), 16);
            c.or_(Reg32(Operand::RAX), Reg32(Operand::RBP));
            c.movzx(Reg32(Operand::RBP), c.byte[rX + rK3 + r]);
            c.shl(Reg32(Operand::RBP), 24);
            c.or_(Reg32(Operand::RAX), Reg32(Operand::RBP));
            c.vmovd(Xmm(21), Reg32(Operand::RAX));
            c.vpbroadcastd(Zmm(21), Xmm(21));
            c.vpdpbusd(Zmm(r),    Zmm(21), Zmm(18));
            c.vpdpbusd(Zmm(6+r),  Zmm(21), Zmm(19));
            c.vpdpbusd(Zmm(12+r), Zmm(21), Zmm(20));
        }

        c.add(rW, 192);
        c.add(rKo, 32);
        c.dec(rKc);
        c.jnz(k_loop);
    }
    c.L("k_done");

    // ── Fused requantize epilogue ────────────────────────────────────────
    // accum[r][col] now holds corrected int32 values (pre-biased with adj_col).
    // Convert: float(accum) * scale + bias → round → clamp → uint8 → NHWC store.
    // rq struct: scale(0), bias(8), inv_y_scale(16), y_zp(20), qmin(24), qmax(28),
    //            Y_out(32), y_stride(40)

    // Load scalar constants from rq struct
    c.vbroadcastss(Zmm(22), c.dword[rRq + offsetof(gather_rq_t, inv_y_scale)]);
    c.vbroadcastss(Zmm(23), c.dword[rRq + offsetof(gather_rq_t, y_zp)]);
    c.vbroadcastss(Zmm(24), c.dword[rRq + offsetof(gather_rq_t, qmin)]);
    c.vbroadcastss(Zmm(25), c.dword[rRq + offsetof(gather_rq_t, qmax)]);

    // Load per-channel scale and bias into zmm18-20 (scale) and zmm26-28 (bias)
    const Reg64 rScale(Operand::RAX);
    c.mov(rScale, c.qword[rRq + offsetof(gather_rq_t, scale)]);
    c.vmovups(Zmm(18), c.ptr[rScale]);        // scale col0
    c.vmovups(Zmm(19), c.ptr[rScale + 64]);   // scale col1
    c.vmovups(Zmm(20), c.ptr[rScale + 128]);  // scale col2
    c.mov(rT, c.qword[rRq + offsetof(gather_rq_t, bias)]);
    c.test(rT, rT);
    c.jz("no_bias", CodeGenerator::T_NEAR);
    c.vmovups(Zmm(26), c.ptr[rT]);
    c.vmovups(Zmm(27), c.ptr[rT + 64]);
    c.vmovups(Zmm(28), c.ptr[rT + 128]);
    c.jmp("bias_done", CodeGenerator::T_NEAR);
    c.L("no_bias");
    c.vpxord(Zmm(26), Zmm(26), Zmm(26));
    c.vpxord(Zmm(27), Zmm(27), Zmm(27));
    c.vpxord(Zmm(28), Zmm(28), Zmm(28));
    c.L("bias_done");

    // Load output pointer and stride
    const Reg64 rOut(14);    // reuse r14
    const Reg64 rStride(15); // reuse r15
    c.mov(rOut, c.qword[rRq + offsetof(gather_rq_t, Y_out)]);
    c.movsxd(rStride, c.dword[rRq + offsetof(gather_rq_t, y_stride)]);

    // Per row: convert int32→float, fma(scale,bias), round, clamp, pack→uint8, store
    for (int r = 0; r < MR; r++) {
        // col0: zmm(r) → float → fma → round → clamp
        c.vcvtdq2ps(Zmm(r), Zmm(r));
        c.vfmadd213ps(Zmm(r), Zmm(18), Zmm(26));    // accum * scale + bias
        c.vmulps(Zmm(r), Zmm(r), Zmm(22));           // * inv_y_scale
        c.vrndscaleps(Zmm(r), Zmm(r), 0x00);         // round to nearest
        c.vaddps(Zmm(r), Zmm(r), Zmm(23));           // + y_zp
        c.vmaxps(Zmm(r), Zmm(r), Zmm(24));           // clamp min
        c.vminps(Zmm(r), Zmm(r), Zmm(25));           // clamp max
        c.vcvtps2dq(Zmm(r), Zmm(r));                 // → int32
        // col1
        c.vcvtdq2ps(Zmm(6+r), Zmm(6+r));
        c.vfmadd213ps(Zmm(6+r), Zmm(19), Zmm(27));
        c.vmulps(Zmm(6+r), Zmm(6+r), Zmm(22));
        c.vrndscaleps(Zmm(6+r), Zmm(6+r), 0x00);
        c.vaddps(Zmm(6+r), Zmm(6+r), Zmm(23));
        c.vmaxps(Zmm(6+r), Zmm(6+r), Zmm(24));
        c.vminps(Zmm(6+r), Zmm(6+r), Zmm(25));
        c.vcvtps2dq(Zmm(6+r), Zmm(6+r));
        // col2
        c.vcvtdq2ps(Zmm(12+r), Zmm(12+r));
        c.vfmadd213ps(Zmm(12+r), Zmm(20), Zmm(28));
        c.vmulps(Zmm(12+r), Zmm(12+r), Zmm(22));
        c.vrndscaleps(Zmm(12+r), Zmm(12+r), 0x00);
        c.vaddps(Zmm(12+r), Zmm(12+r), Zmm(23));
        c.vmaxps(Zmm(12+r), Zmm(12+r), Zmm(24));
        c.vminps(Zmm(12+r), Zmm(12+r), Zmm(25));
        c.vcvtps2dq(Zmm(12+r), Zmm(12+r));

        // Pack 3×16 int32 → 48 uint8 and store contiguous NHWC row
        // vpmovdb: 16 int32 → 16 uint8 with saturation (unsigned)
        c.vpmovusdb(Xmm(r),    Zmm(r));       // col0 → xmm(r) bytes [0..15]
        c.vpmovusdb(Xmm(6+r),  Zmm(6+r));     // col1 → xmm(6+r)
        c.vpmovusdb(Xmm(12+r), Zmm(12+r));    // col2 → xmm(12+r)

        // Store 48 bytes (3×16) at Y_out[r * stride], masked per column group
        c.vmovdqu8(c.ptr[rOut]      | Opmask(1), Xmm(r));
        c.vmovdqu8(c.ptr[rOut + 16] | Opmask(2), Xmm(6+r));
        c.vmovdqu8(c.ptr[rOut + 32] | Opmask(3), Xmm(12+r));
        if (r < MR - 1) c.add(rOut, rStride);
    }

    emit_epilogue();
    finalize();
}


// -- Packed GEMM JIT, NR=48 (3 sub-panels, ORT-style, inner N-loop) --
// MR=1..6 baked at JIT time. 18 accumulators. 4x K-unrolled.
// Inner N-loop processes num_nr_blocks x 48 columns per call.
// A pointer reset between N-blocks keeps A hot in L1.

jit_packed_gemm_nr48_avx512_t::jit_packed_gemm_nr48_avx512_t(int mr_actual)
    : jit_kernel_t(65536)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;

    // Only save xmm14-15: zmm14-31 are accumulators, zmm0-3 temps; zmm6-13 unused.
    emit_prologue((1u<<3)|(1u<<5)|(1u<<12)|(1u<<13)|(1u<<14)|(1u<<15), (1u<<8)|(1u<<9));

    const Reg64 rA0(Operand::RCX), rA1(Operand::RBX), rB(Operand::RDX), rC(8);
    const Reg64 rKrem(Operand::RSI), rLda(9), rSub(14), rLdc(15);
    const Reg64 rKsave(13), rNrem(12), rZm(10);
    const Reg64 rA0save(Operand::RDI), rA1save(Operand::RBP), rT(Operand::RAX);

    load_arg(0, Operand::RDI);
    load_arg(1, Operand::RDX);
    load_arg(2, 8);
    load_arg_i32(3, Operand::RSI);
    load_arg_i32(4, 9);
    load_arg_i32(5, 15);
    load_arg(6, 14);
    load_arg_i32(7, 10);
    load_arg_i32(8, 12);

    // Spill zero_mode (r10) to shadow slot, free r10 for column offset counter
    int zm_slot = arg_stack_offset(4) - 32;  // Win64 shadow slot 0
    c.mov(c.dword[c.rsp + zm_slot], Reg32(10));

    c.shl(rKrem, 2);
    c.mov(rKsave, rKrem);
    c.shl(rLdc, 2);

    c.mov(rA0, rA0save);
    if (MR > 3) {
        c.lea(rA1save, c.ptr[rLda + rLda * 2]);
        c.add(rA1save, rA0save);
        c.mov(rA1, rA1save);
    }

    // Initialize column element offset for requantize epilogue
    c.xor_(Reg64(10), Reg64(10));

    // --- Requantize epilogue emitter ---
    // Processes num_sub sub-panels (16 cols each) from accumulator registers.
    // On entry: rT(rax) = rq pointer (already tested non-null).
    // Uses zmm0-5, rKrem(rsi), rA0(rcx), rA1(rbx), rT(rax). r10 = col element offset.
    auto emit_requantize = [&](int num_sub, const int* bases) {
        c.mov(rKrem, rT);  // save rq ptr in rKrem

        // Pre-load float constants from rq struct
        c.vbroadcastss(Zmm(2), c.dword[rKrem + 32]);  // rq_qmin
        c.vbroadcastss(Zmm(3), c.dword[rKrem + 36]);  // rq_qmax
        c.vpbroadcastd(Zmm(4), c.dword[rKrem + 28]);  // y_zp_int

        for (int s = 0; s < num_sub; s++) {
            int base_zmm = bases[s];
            int col_u8 = s * 16;     // uint8 element offset within N-block
            int col_i32 = s * 64;    // byte offset for int32/float arrays

            // Per-column ZP correction: x_zp_neg * w_col_sums[col] → zmm0
            Label no_wcs, wcs_done;
            c.mov(rA1, c.qword[rKrem + 16]);  // w_col_sums ptr
            c.test(rA1, rA1);
            c.jz(no_wcs, CodeGenerator::T_NEAR);
            c.vpbroadcastd(Zmm(5), c.dword[rKrem + 24]);  // x_zp_neg
            c.vmovdqu32(Zmm(0), c.ptr[rA1 + Reg64(10) * 4 + col_i32]);
            c.vpmulld(Zmm(0), Zmm(0), Zmm(5));
            c.jmp(wcs_done, CodeGenerator::T_NEAR);
            c.L(no_wcs);
            c.vpxord(Zmm(0), Zmm(0), Zmm(0));
            c.L(wcs_done);

            // Per-column bias → zmm1
            Label no_bias, bias_done;
            c.mov(rA1, c.qword[rKrem + 8]);  // bias_int32 ptr
            c.test(rA1, rA1);
            c.jz(no_bias, CodeGenerator::T_NEAR);
            c.vmovdqu32(Zmm(1), c.ptr[rA1 + Reg64(10) * 4 + col_i32]);
            c.jmp(bias_done, CodeGenerator::T_NEAR);
            c.L(no_bias);
            c.vpxord(Zmm(1), Zmm(1), Zmm(1));
            c.L(bias_done);

            // output_scales base pointer
            c.mov(rA1, c.qword[rKrem + 0]);  // output_scales

            for (int r = 0; r < MR; r++) {
                int acc = base_zmm + r;

                c.vpaddd(Zmm(acc), Zmm(acc), Zmm(0));  // +zp_col_corr
                c.vpbroadcastd(Zmm(5), c.dword[rKrem + 40 + r * 4]);  // row_corr[r]
                c.vpaddd(Zmm(acc), Zmm(acc), Zmm(5));
                c.vpaddd(Zmm(acc), Zmm(acc), Zmm(1));  // +bias

                c.vcvtdq2ps(Zmm(acc), Zmm(acc));
                c.vmulps(Zmm(acc), Zmm(acc), c.ptr[rA1 + Reg64(10) * 4 + col_i32]);
                c.vmaxps(Zmm(acc), Zmm(acc), Zmm(2));
                c.vminps(Zmm(acc), Zmm(acc), Zmm(3));
                c.vcvtps2dq(Zmm(acc), Zmm(acc));
                c.vpaddd(Zmm(acc), Zmm(acc), Zmm(4));  // +y_zp

                // int32 → uint8 (unsigned saturate) → store 16 bytes
                c.vpmovusdb(Xmm(5), Zmm(acc));
                c.mov(rA0, c.qword[rKrem + 64 + r * 8]);  // y_rows[r]
                c.vmovups(c.ptr[rA0 + Reg64(10) + col_u8], Xmm(5));
            }
        }
    };

    c.test(rNrem, rNrem);
    c.jz("tail_start", CodeGenerator::T_NEAR);

    Label n_loop;
    c.L(n_loop);
    {
        c.mov(Reg64(11), rB);   // save B start for this N-block in r11
        c.mov(rA0, rA0save);
        if (MR > 3) c.mov(rA1, rA1save);
        c.mov(rKrem, rKsave);

        c.cmp(c.dword[c.rsp + zm_slot], 0);
        c.jz("load_c", CodeGenerator::T_NEAR);
        for (int r = 0; r < MR; r++) {
            c.vpxord(Zmm(26+r), Zmm(26+r), Zmm(26+r));
            c.vpxord(Zmm(20+r), Zmm(20+r), Zmm(20+r));
            c.vpxord(Zmm(14+r), Zmm(14+r), Zmm(14+r));
        }
        c.jmp("acc_ready", CodeGenerator::T_NEAR);

        c.L("load_c");
        c.mov(rT, rC);
        for (int r = 0; r < MR; r++) {
            c.vmovdqu32(Zmm(26+r), c.ptr[rT]);
            c.vmovdqu32(Zmm(20+r), c.ptr[rT + 64]);
            c.vmovdqu32(Zmm(14+r), c.ptr[rT + 128]);
            if (r < MR-1) c.add(rT, rLdc);
        }
        c.L("acc_ready");

        auto emit_block = [&](int voff, int boff) {
            c.vmovdqu32(Zmm(0), c.ptr[rB + voff]);
            c.vmovdqu32(Zmm(1), c.ptr[rB + rSub + voff]);
            c.vmovdqu32(Zmm(2), c.ptr[rB + rSub * 2 + voff]);
            for (int r = 0; r < std::min(MR, 3); r++) {
                if (r==0)      c.vpbroadcastd(Zmm(3), c.dword[rA0 + boff]);
                else if (r==1) c.vpbroadcastd(Zmm(3), c.dword[rA0 + rLda + boff]);
                else           c.vpbroadcastd(Zmm(3), c.dword[rA0 + rLda * 2 + boff]);
                c.vpdpbusd(Zmm(26+r), Zmm(3), Zmm(0));
                c.vpdpbusd(Zmm(20+r), Zmm(3), Zmm(1));
                c.vpdpbusd(Zmm(14+r), Zmm(3), Zmm(2));
            }
            for (int r = 3; r < MR; r++) {
                int rr = r-3;
                if (rr==0)      c.vpbroadcastd(Zmm(3), c.dword[rA1 + boff]);
                else if (rr==1) c.vpbroadcastd(Zmm(3), c.dword[rA1 + rLda + boff]);
                else            c.vpbroadcastd(Zmm(3), c.dword[rA1 + rLda * 2 + boff]);
                c.vpdpbusd(Zmm(26+r), Zmm(3), Zmm(0));
                c.vpdpbusd(Zmm(20+r), Zmm(3), Zmm(1));
                c.vpdpbusd(Zmm(14+r), Zmm(3), Zmm(2));
            }
        };

        c.cmp(rKrem, 16);
        c.jb("k_by1_start", CodeGenerator::T_NEAR);
        Label k_by4;
        c.L(k_by4);
        emit_block(0, 0); emit_block(64, 4); emit_block(128, 8); emit_block(192, 12);
        c.add(rA0, 16);
        if (MR > 3) c.add(rA1, 16);
        c.add(rB, 256);
        c.sub(rKrem, 16);
        c.cmp(rKrem, 16);
        c.jae(k_by4);

        c.L("k_by1_start");
        c.test(rKrem, rKrem);
        c.jz("k_done", CodeGenerator::T_NEAR);
        Label k_by1;
        c.L(k_by1);
        emit_block(0, 0);
        c.add(rA0, 4);
        if (MR > 3) c.add(rA1, 4);
        c.add(rB, 64);
        c.sub(rKrem, 4);
        c.jnz(k_by1);
        c.L("k_done");

        // --- Store or requantize ---
        c.mov(rT, c.qword[c.rsp + arg_stack_offset(10)]);  // rq ptr
        c.test(rT, rT);
        Label rq_n, store_done_n;
        c.jnz(rq_n, CodeGenerator::T_NEAR);

        // Normal int32 store
        c.mov(rT, rC);
        for (int r = 0; r < MR; r++) {
            c.vmovdqu32(c.ptr[rT],       Zmm(26+r));
            c.vmovdqu32(c.ptr[rT + 64],  Zmm(20+r));
            c.vmovdqu32(c.ptr[rT + 128], Zmm(14+r));
            if (r < MR-1) c.add(rT, rLdc);
        }
        c.jmp(store_done_n, CodeGenerator::T_NEAR);

        // Requantize: accumulators → uint8 directly to Y_out
        c.L(rq_n);
        { int bases[] = {26, 20, 14}; emit_requantize(3, bases); }
        c.add(Reg64(10), 48);  // advance column element offset

        c.L(store_done_n);

        // Advance B to next NR-block: B_next = B_saved + 3*sub_stride
        c.lea(rB, c.ptr[Reg64(11) + rSub]);    // B_saved + sub_stride
        c.lea(rB, c.ptr[rB + rSub * 2]);       // + 2*sub_stride = B_saved + 3*sub_stride
        c.add(rC, 192);
        c.dec(rNrem);
        c.jnz(n_loop);
    }
    // ---- Tail: handle partial last NR-block (nr_tail sub-panels) ----
    c.L("tail_start");
    // nr_tail is arg 9 (stack arg on Win64)
    c.movsxd(rT, c.dword[c.rsp + arg_stack_offset(9)]);
    c.test(rT, rT);
    c.jz("done", CodeGenerator::T_NEAR);

    // At this point: rB points past the last full NR-block (correct for tail B start).
    // rC is advanced by num_full_nr * 192 bytes (correct for tail C start).
    // rA0save/rA1save still hold original A pointers.
    // rT = nr_tail (1=16 cols, 2=32 cols)

    // Save nr_tail, restore A pointers and K counter
    c.mov(rNrem, rT);  // reuse rNrem for nr_tail
    c.mov(rA0, rA0save);
    if (MR > 3) c.mov(rA1, rA1save);
    c.mov(rKrem, rKsave);

    // Zero or load C for tail columns
    c.cmp(c.dword[c.rsp + zm_slot], 0);
    c.jz("tail_load_c", CodeGenerator::T_NEAR);
    for (int r = 0; r < MR; r++) {
        c.vpxord(Zmm(26+r), Zmm(26+r), Zmm(26+r));  // cols 0-15 (always)
        c.vpxord(Zmm(20+r), Zmm(20+r), Zmm(20+r));  // cols 16-31 (might not be used)
    }
    c.jmp("tail_acc_ready", CodeGenerator::T_NEAR);

    c.L("tail_load_c");
    c.mov(rT, rC);
    c.cmp(rNrem, 2);
    c.je("tail_load_c_32", CodeGenerator::T_NEAR);
    // Load 16 cols only
    for (int r = 0; r < MR; r++) {
        c.vmovdqu32(Zmm(26+r), c.ptr[rT]);
        if (r < MR-1) c.add(rT, rLdc);
    }
    c.jmp("tail_acc_ready", CodeGenerator::T_NEAR);
    c.L("tail_load_c_32");
    for (int r = 0; r < MR; r++) {
        c.vmovdqu32(Zmm(26+r), c.ptr[rT]);
        c.vmovdqu32(Zmm(20+r), c.ptr[rT + 64]);
        if (r < MR-1) c.add(rT, rLdc);
    }
    c.L("tail_acc_ready");

    // K-loop for tail: always load sub-panel 0, conditionally load sub-panel 1
    c.cmp(rNrem, 2);
    c.je("tail_k_nr32", CodeGenerator::T_NEAR);

    // --- Tail NR=16: 1 sub-panel ---
    {
        auto emit_tail16 = [&](int boff, int aoff) {
            c.vmovdqu32(Zmm(0), c.ptr[rB + boff]);
            for (int r = 0; r < std::min(MR, 3); r++) {
                if (r==0)      c.vpbroadcastd(Zmm(1), c.dword[rA0 + aoff]);
                else if (r==1) c.vpbroadcastd(Zmm(1), c.dword[rA0 + rLda + aoff]);
                else           c.vpbroadcastd(Zmm(1), c.dword[rA0 + rLda * 2 + aoff]);
                c.vpdpbusd(Zmm(26+r), Zmm(1), Zmm(0));
            }
            for (int r = 3; r < MR; r++) {
                int rr = r-3;
                if (rr==0)      c.vpbroadcastd(Zmm(1), c.dword[rA1 + aoff]);
                else if (rr==1) c.vpbroadcastd(Zmm(1), c.dword[rA1 + rLda + aoff]);
                else            c.vpbroadcastd(Zmm(1), c.dword[rA1 + rLda * 2 + aoff]);
                c.vpdpbusd(Zmm(26+r), Zmm(1), Zmm(0));
            }
        };

        c.cmp(rKrem, 16);
        c.jb("tail16_k1_start", CodeGenerator::T_NEAR);
        Label tail16_k4;
        c.L(tail16_k4);
        emit_tail16(0, 0); emit_tail16(64, 4); emit_tail16(128, 8); emit_tail16(192, 12);
        c.add(rA0, 16); if (MR > 3) c.add(rA1, 16);
        c.add(rB, 256); c.sub(rKrem, 16); c.cmp(rKrem, 16); c.jae(tail16_k4);

        c.L("tail16_k1_start");
        c.test(rKrem, rKrem);
        c.jz("tail16_store", CodeGenerator::T_NEAR);
        Label tail16_k1;
        c.L(tail16_k1);
        emit_tail16(0, 0);
        c.add(rA0, 4); if (MR > 3) c.add(rA1, 4);
        c.add(rB, 64); c.sub(rKrem, 4); c.jnz(tail16_k1);

        c.L("tail16_store");
        c.mov(rT, c.qword[c.rsp + arg_stack_offset(10)]);  // rq ptr
        c.test(rT, rT);
        Label rq_t16;
        c.jnz(rq_t16, CodeGenerator::T_NEAR);
        c.mov(rT, rC);
        for (int r = 0; r < MR; r++) {
            c.vmovdqu32(c.ptr[rT], Zmm(26+r));
            if (r < MR-1) c.add(rT, rLdc);
        }
        c.jmp("done", CodeGenerator::T_NEAR);
        c.L(rq_t16);
        { int bases[] = {26}; emit_requantize(1, bases); }
        c.jmp("done", CodeGenerator::T_NEAR);
    }

    // --- Tail NR=32: 2 sub-panels ---
    c.L("tail_k_nr32");
    {
        auto emit_tail32 = [&](int boff, int aoff) {
            c.vmovdqu32(Zmm(0), c.ptr[rB + boff]);
            c.vmovdqu32(Zmm(1), c.ptr[rB + rSub + boff]);
            for (int r = 0; r < std::min(MR, 3); r++) {
                if (r==0)      c.vpbroadcastd(Zmm(2), c.dword[rA0 + aoff]);
                else if (r==1) c.vpbroadcastd(Zmm(2), c.dword[rA0 + rLda + aoff]);
                else           c.vpbroadcastd(Zmm(2), c.dword[rA0 + rLda * 2 + aoff]);
                c.vpdpbusd(Zmm(26+r), Zmm(2), Zmm(0));
                c.vpdpbusd(Zmm(20+r), Zmm(2), Zmm(1));
            }
            for (int r = 3; r < MR; r++) {
                int rr = r-3;
                if (rr==0)      c.vpbroadcastd(Zmm(2), c.dword[rA1 + aoff]);
                else if (rr==1) c.vpbroadcastd(Zmm(2), c.dword[rA1 + rLda + aoff]);
                else            c.vpbroadcastd(Zmm(2), c.dword[rA1 + rLda * 2 + aoff]);
                c.vpdpbusd(Zmm(26+r), Zmm(2), Zmm(0));
                c.vpdpbusd(Zmm(20+r), Zmm(2), Zmm(1));
            }
        };

        c.cmp(rKrem, 16);
        c.jb("tail32_k1_start", CodeGenerator::T_NEAR);
        Label tail32_k4;
        c.L(tail32_k4);
        emit_tail32(0, 0); emit_tail32(64, 4); emit_tail32(128, 8); emit_tail32(192, 12);
        c.add(rA0, 16); if (MR > 3) c.add(rA1, 16);
        c.add(rB, 256); c.sub(rKrem, 16); c.cmp(rKrem, 16); c.jae(tail32_k4);

        c.L("tail32_k1_start");
        c.test(rKrem, rKrem);
        c.jz("tail32_store", CodeGenerator::T_NEAR);
        Label tail32_k1;
        c.L(tail32_k1);
        emit_tail32(0, 0);
        c.add(rA0, 4); if (MR > 3) c.add(rA1, 4);
        c.add(rB, 64); c.sub(rKrem, 4); c.jnz(tail32_k1);

        c.L("tail32_store");
        c.mov(rT, c.qword[c.rsp + arg_stack_offset(10)]);  // rq ptr
        c.test(rT, rT);
        Label rq_t32;
        c.jnz(rq_t32, CodeGenerator::T_NEAR);
        c.mov(rT, rC);
        for (int r = 0; r < MR; r++) {
            c.vmovdqu32(c.ptr[rT],      Zmm(26+r));
            c.vmovdqu32(c.ptr[rT + 64], Zmm(20+r));
            if (r < MR-1) c.add(rT, rLdc);
        }
        c.jmp("done", CodeGenerator::T_NEAR);
        c.L(rq_t32);
        { int bases[] = {26, 20}; emit_requantize(2, bases); }
    }

    c.L("done");

    emit_epilogue();
    finalize();
}

// ---------------------------------------------------------------------------
// NR=16 partial kernel: 1 sub-panel, 16 columns
// Accumulators: zmm26..zmm(26+MR-1) for cols 0-15
// ---------------------------------------------------------------------------
jit_packed_gemm_partial16_avx512_t::jit_packed_gemm_partial16_avx512_t(int mr_actual)
    : jit_kernel_t(65536)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;

    // Only zmm26-31 used as accumulators — no callee-saved XMMs clobbered
    emit_prologue((1u<<3)|(1u<<5)|(1u<<13)|(1u<<15), 0);

    const Reg64 rA0(Operand::RCX), rA1(Operand::RBX), rB(Operand::RDX), rC(8);
    const Reg64 rKrem(Operand::RSI), rLda(9), rLdc(15);
    const Reg64 rKsave(13), rZm(10);
    const Reg64 rA0save(Operand::RDI), rA1save(Operand::RBP), rT(Operand::RAX);

    load_arg(0, Operand::RDI);   // A
    load_arg(1, Operand::RDX);   // B
    load_arg(2, 8);              // C
    load_arg_i32(3, Operand::RSI); // Kgroups
    load_arg_i32(4, 9);           // lda
    load_arg_i32(5, 15);          // ldc
    // arg6 = sub_stride (unused for NR=16)
    load_arg_i32(7, 10);          // zero_mode

    c.shl(rKrem, 2);
    c.mov(rKsave, rKrem);
    c.shl(rLdc, 2);

    c.mov(rA0, rA0save);
    if (MR > 3) {
        c.lea(rA1save, c.ptr[rLda + rLda * 2]);
        c.add(rA1save, rA0save);
        c.mov(rA1, rA1save);
    }

    // Zero or load C
    c.test(rZm, rZm);
    c.jz("load_c", CodeGenerator::T_NEAR);
    for (int r = 0; r < MR; r++)
        c.vpxord(Zmm(26+r), Zmm(26+r), Zmm(26+r));
    c.jmp("acc_ready", CodeGenerator::T_NEAR);

    c.L("load_c");
    c.mov(rT, rC);
    for (int r = 0; r < MR; r++) {
        c.vmovdqu32(Zmm(26+r), c.ptr[rT]);
        if (r < MR-1) c.add(rT, rLdc);
    }
    c.L("acc_ready");

    // K-loop: 1 sub-panel only
    auto emit_block = [&](int boff, int aoff) {
        c.vmovdqu32(Zmm(0), c.ptr[rB + boff]);
        for (int r = 0; r < std::min(MR, 3); r++) {
            if (r==0)      c.vpbroadcastd(Zmm(1), c.dword[rA0 + aoff]);
            else if (r==1) c.vpbroadcastd(Zmm(1), c.dword[rA0 + rLda + aoff]);
            else           c.vpbroadcastd(Zmm(1), c.dword[rA0 + rLda * 2 + aoff]);
            c.vpdpbusd(Zmm(26+r), Zmm(1), Zmm(0));
        }
        for (int r = 3; r < MR; r++) {
            int rr = r-3;
            if (rr==0)      c.vpbroadcastd(Zmm(1), c.dword[rA1 + aoff]);
            else if (rr==1) c.vpbroadcastd(Zmm(1), c.dword[rA1 + rLda + aoff]);
            else            c.vpbroadcastd(Zmm(1), c.dword[rA1 + rLda * 2 + aoff]);
            c.vpdpbusd(Zmm(26+r), Zmm(1), Zmm(0));
        }
    };

    c.cmp(rKrem, 16);
    c.jb("k_by1_start", CodeGenerator::T_NEAR);
    Label k_by4;
    c.L(k_by4);
    emit_block(0, 0); emit_block(64, 4); emit_block(128, 8); emit_block(192, 12);
    c.add(rA0, 16);
    if (MR > 3) c.add(rA1, 16);
    c.add(rB, 256);
    c.sub(rKrem, 16);
    c.cmp(rKrem, 16);
    c.jae(k_by4);

    c.L("k_by1_start");
    c.test(rKrem, rKrem);
    c.jz("k_done", CodeGenerator::T_NEAR);
    Label k_by1;
    c.L(k_by1);
    emit_block(0, 0);
    c.add(rA0, 4);
    if (MR > 3) c.add(rA1, 4);
    c.add(rB, 64);
    c.sub(rKrem, 4);
    c.jnz(k_by1);
    c.L("k_done");

    // Store C
    c.mov(rT, rC);
    for (int r = 0; r < MR; r++) {
        c.vmovdqu32(c.ptr[rT], Zmm(26+r));
        if (r < MR-1) c.add(rT, rLdc);
    }

    emit_epilogue();
    finalize();
}

// ---------------------------------------------------------------------------
// NR=32 partial kernel: 2 sub-panels, 32 columns
// Accumulators: zmm26..zmm(26+MR-1) for cols 0-15
//               zmm20..zmm(20+MR-1) for cols 16-31
// ---------------------------------------------------------------------------
jit_packed_gemm_partial32_avx512_t::jit_packed_gemm_partial32_avx512_t(int mr_actual)
    : jit_kernel_t(65536)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;

    // zmm20-25 clobbers xmm(20-25) — but those aren't callee-saved on Win64
    // zmm26-31 are xmm26-31 — also not callee-saved (only xmm6-15 are)
    // So: no XMM saves needed!
    emit_prologue((1u<<3)|(1u<<5)|(1u<<13)|(1u<<14)|(1u<<15), 0);

    const Reg64 rA0(Operand::RCX), rA1(Operand::RBX), rB(Operand::RDX), rC(8);
    const Reg64 rKrem(Operand::RSI), rLda(9), rSub(14), rLdc(15);
    const Reg64 rKsave(13), rZm(10);
    const Reg64 rA0save(Operand::RDI), rA1save(Operand::RBP), rT(Operand::RAX);

    load_arg(0, Operand::RDI);   // A
    load_arg(1, Operand::RDX);   // B
    load_arg(2, 8);              // C
    load_arg_i32(3, Operand::RSI); // Kgroups
    load_arg_i32(4, 9);           // lda
    load_arg_i32(5, 15);          // ldc
    load_arg(6, 14);              // sub_stride
    load_arg_i32(7, 10);          // zero_mode

    c.shl(rKrem, 2);
    c.mov(rKsave, rKrem);
    c.shl(rLdc, 2);

    c.mov(rA0, rA0save);
    if (MR > 3) {
        c.lea(rA1save, c.ptr[rLda + rLda * 2]);
        c.add(rA1save, rA0save);
        c.mov(rA1, rA1save);
    }

    // Zero or load C
    c.test(rZm, rZm);
    c.jz("load_c", CodeGenerator::T_NEAR);
    for (int r = 0; r < MR; r++) {
        c.vpxord(Zmm(26+r), Zmm(26+r), Zmm(26+r));
        c.vpxord(Zmm(20+r), Zmm(20+r), Zmm(20+r));
    }
    c.jmp("acc_ready", CodeGenerator::T_NEAR);

    c.L("load_c");
    c.mov(rT, rC);
    for (int r = 0; r < MR; r++) {
        c.vmovdqu32(Zmm(26+r), c.ptr[rT]);
        c.vmovdqu32(Zmm(20+r), c.ptr[rT + 64]);
        if (r < MR-1) c.add(rT, rLdc);
    }
    c.L("acc_ready");

    // K-loop: 2 sub-panels
    auto emit_block = [&](int boff, int aoff) {
        c.vmovdqu32(Zmm(0), c.ptr[rB + boff]);
        c.vmovdqu32(Zmm(1), c.ptr[rB + rSub + boff]);
        for (int r = 0; r < std::min(MR, 3); r++) {
            if (r==0)      c.vpbroadcastd(Zmm(2), c.dword[rA0 + aoff]);
            else if (r==1) c.vpbroadcastd(Zmm(2), c.dword[rA0 + rLda + aoff]);
            else           c.vpbroadcastd(Zmm(2), c.dword[rA0 + rLda * 2 + aoff]);
            c.vpdpbusd(Zmm(26+r), Zmm(2), Zmm(0));
            c.vpdpbusd(Zmm(20+r), Zmm(2), Zmm(1));
        }
        for (int r = 3; r < MR; r++) {
            int rr = r-3;
            if (rr==0)      c.vpbroadcastd(Zmm(2), c.dword[rA1 + aoff]);
            else if (rr==1) c.vpbroadcastd(Zmm(2), c.dword[rA1 + rLda + aoff]);
            else            c.vpbroadcastd(Zmm(2), c.dword[rA1 + rLda * 2 + aoff]);
            c.vpdpbusd(Zmm(26+r), Zmm(2), Zmm(0));
            c.vpdpbusd(Zmm(20+r), Zmm(2), Zmm(1));
        }
    };

    c.cmp(rKrem, 16);
    c.jb("k_by1_start", CodeGenerator::T_NEAR);
    Label k_by4;
    c.L(k_by4);
    emit_block(0, 0); emit_block(64, 4); emit_block(128, 8); emit_block(192, 12);
    c.add(rA0, 16);
    if (MR > 3) c.add(rA1, 16);
    c.add(rB, 256);
    c.sub(rKrem, 16);
    c.cmp(rKrem, 16);
    c.jae(k_by4);

    c.L("k_by1_start");
    c.test(rKrem, rKrem);
    c.jz("k_done", CodeGenerator::T_NEAR);
    Label k_by1;
    c.L(k_by1);
    emit_block(0, 0);
    c.add(rA0, 4);
    if (MR > 3) c.add(rA1, 4);
    c.add(rB, 64);
    c.sub(rKrem, 4);
    c.jnz(k_by1);
    c.L("k_done");

    // Store C
    c.mov(rT, rC);
    for (int r = 0; r < MR; r++) {
        c.vmovdqu32(c.ptr[rT],      Zmm(26+r));
        c.vmovdqu32(c.ptr[rT + 64], Zmm(20+r));
        if (r < MR-1) c.add(rT, rLdc);
    }

    emit_epilogue();
    finalize();
}

// ───────────────────────────────────────────────────────────────────────────
// NHWC-direct GEMM JIT, NR=48
//
// Forked from jit_packed_gemm_nr48_avx512_t. Replaces the flat
// [packed_A + m*lda + k4_off] addressing with
// [row_bases[m] + pixel_off[kp] + k4_off_within_pixel].
//
// Register layout (x64 / Win64+SysV after load_arg helpers):
//   zmm14..zmm19 — sub-panel 0 accumulators (cols  0..15) per row
//   zmm20..zmm25 — sub-panel 1 accumulators (cols 16..31) per row
//   zmm26..zmm31 — sub-panel 2 accumulators (cols 32..47) per row
//   zmm0..zmm2   — B sub-panel loads (live one k4-step)
//   zmm3         — A broadcast scratch (one row at a time)
//
//   rA0..rA5     — 6 row-base GPRs (pre-loaded from row_bases[0..5])
//   r_xoff       — running byte offset into x_pad (init = pixel_off[kp],
//                  then += 4 per k4-step)
//   rB           — running B sub-panel-0 pointer (advanced 64B per k4)
//   rSub         — sub-panel byte stride (B[1] = B + rSub, B[2] = B + 2*rSub)
//   rIC4         — IC4 constant (re-loaded into rk4 at start of each kp)
//   rk4          — per-kp k4 countdown
//   rPxOff       — pointer into pixel_off[] (advanced 4B per kp)
//   rkHW         — kp countdown
//   rT           — scratch (epilogue)
//
// For MR<6 the unused row bases must still be valid — caller pads with
// row 0. The trailing accumulators stay zero and are not stored.
// ───────────────────────────────────────────────────────────────────────────
jit_nhwc_direct_gemm_nr48_avx512_t::jit_nhwc_direct_gemm_nr48_avx512_t(int mr_actual)
    : jit_kernel_t(32768)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;

    // Save callee-saved GPRs we touch: rbx(3), rbp(5), r12(12), r13(13),
    // r14(14), r15(15). On Win64, rsi/rdi are also callee-saved and the
    // helper auto-saves them. xmm save: zmm14..zmm14+MR-1 may alias xmm14/15.
    // For MR≥2 we touch zmm14 and zmm15; save both. For MR=1 only zmm14;
    // save only xmm14. Either way safe to save both — pay 16 bytes.
    emit_prologue(
        (1u << 3) | (1u << 5) | (1u << 12) | (1u << 13) | (1u << 14) | (1u << 15),
        (1u << 8) | (1u << 9));

    const Reg64 rA0(Operand::RCX), rA1(Operand::RDX), rA2(8), rA3(9);
    const Reg64 rA4(10), rA5(11);
    const Reg64 r_xoff(Operand::RBX);
    const Reg64 rB(Operand::RSI), rSub(Operand::RDI);
    const Reg64 rIC4(12), rk4(13);
    const Reg64 rPxOff(14), rkHW(15);
    const Reg64 rT(Operand::RAX);
    const Reg64 rTmp(Operand::RBP);  // only used in the prologue to deref row_bases

    // ── Argument loading ────────
    // arg0 row_bases  → rTmp (rbp). We can't put it directly in rA0=rcx because
    //                  rcx is the destination for row_bases[0].
    // arg1 pixel_off  → rPxOff (r14)
    // arg2 kHW (i32)  → rkHW (r15)
    // arg3 IC4 (i32)  → rIC4 (r12)
    // arg4 B          → rB (rsi)
    // arg5 sub_stride → rSub (rdi)
    // arg6 C          → loaded in epilogue from stack
    // arg7 ldc (i32)  → loaded in epilogue from stack
    load_arg(0, Operand::RBP);
    load_arg(1, 14);
    load_arg_i32(2, 15);
    load_arg_i32(3, 12);
    load_arg(4, Operand::RSI);
    load_arg(5, Operand::RDI);

    // Dereference 6 row bases. Order high→low so we can safely overwrite
    // the lower-numbered registers last.
    c.mov(rA5, c.qword[rTmp + 40]);
    c.mov(rA4, c.qword[rTmp + 32]);
    c.mov(rA3, c.qword[rTmp + 24]);
    c.mov(rA2, c.qword[rTmp + 16]);
    c.mov(rA1, c.qword[rTmp +  8]);
    c.mov(rA0, c.qword[rTmp +  0]);
    // rTmp (rbp) is now dead — not used in K-loop.

    // ── Init 18 accumulators ────────
    for (int s = 0; s < 3; s++) {
        for (int r = 0; r < MR; r++) {
            int acc = 14 + s * 6 + r;
            c.vpxord(Zmm(acc), Zmm(acc), Zmm(acc));
        }
    }

    // ── K-loop: outer kp, inner k4 ────────
    // Inner loop is k_by4 unrolled (4 k4-steps per branch), mirroring
    // jit_packed_gemm_nr48_avx512_t's pattern. This amortizes the loop
    // branch over 108 instructions instead of 27, and gives the scheduler
    // a wider window for the AGU + VPDPBUSD chain.
    //
    // emit_block emits one k4-step worth of work using compile-time
    // immediate offsets (b_off into rB, x_off into r_xoff base).
    const Reg64 rArr[6] = { rA0, rA1, rA2, rA3, rA4, rA5 };
    auto emit_block = [&](int b_off, int x_off) {
        c.vmovdqu32(Zmm(0), c.ptr[rB + b_off]);
        c.vmovdqu32(Zmm(1), c.ptr[rB + rSub + b_off]);
        c.vmovdqu32(Zmm(2), c.ptr[rB + rSub * 2 + b_off]);
        for (int r = 0; r < MR; r++) {
            c.vpbroadcastd(Zmm(3), c.dword[rArr[r] + r_xoff + x_off]);
            c.vpdpbusd(Zmm(14 + 0 * 6 + r), Zmm(3), Zmm(0));
            c.vpdpbusd(Zmm(14 + 1 * 6 + r), Zmm(3), Zmm(1));
            c.vpdpbusd(Zmm(14 + 2 * 6 + r), Zmm(3), Zmm(2));
        }
    };

    Label kp_loop, k4_by4, k4_tail_start, k4_tail, kp_advance, kp_done;
    c.test(rkHW, rkHW);
    c.jz(kp_done, CodeGenerator::T_NEAR);

    c.L(kp_loop);
    {
        // r_xoff = pixel_off[kp] (starting byte offset for this kernel pixel)
        c.movsxd(r_xoff, c.dword[rPxOff]);
        c.add(rPxOff, 4);

        // k4 inner loop, unrolled by 4
        c.mov(rk4, rIC4);
        c.cmp(rk4, 4);
        c.jb(k4_tail_start, CodeGenerator::T_NEAR);

        c.L(k4_by4);
        emit_block(  0,  0);
        emit_block( 64,  4);
        emit_block(128,  8);
        emit_block(192, 12);
        c.add(rB, 256);
        c.add(r_xoff, 16);
        c.sub(rk4, 4);
        c.cmp(rk4, 4);
        c.jae(k4_by4);

        // single-step tail for IC4 % 4 != 0
        c.L(k4_tail_start);
        c.test(rk4, rk4);
        c.jz(kp_advance, CodeGenerator::T_NEAR);
        c.L(k4_tail);
        emit_block(0, 0);
        c.add(rB, 64);
        c.add(r_xoff, 4);
        c.dec(rk4);
        c.jnz(k4_tail);

        c.L(kp_advance);
        c.dec(rkHW);
        c.jnz(kp_loop);
    }
    c.L(kp_done);

    // ── Epilogue: either requantize+store-u8 (rq != null) or int32 store
    // After K-loop, rA0..rA5, r_xoff, rB, rSub, rIC4, rk4, rPxOff, rkHW all
    // free. We reuse rT (rax), rB (rsi) and rSub (rdi) as scratch pointers.
    Label do_rq, done_store;
    c.mov(rT, c.qword[c.rsp + arg_stack_offset(8)]);   // rq ptr
    c.test(rT, rT);
    c.jnz(do_rq, CodeGenerator::T_NEAR);

    // ── Int32 store path (rq == null) ────────
    {
        const Reg64 rC_out = rB;       // reuse rsi
        const Reg64 rLdc   = rA0;       // reuse rcx
        c.mov(rC_out, c.qword[c.rsp + arg_stack_offset(6)]);     // C ptr
        c.movsxd(rLdc, c.dword[c.rsp + arg_stack_offset(7)]);    // ldc i32→64
        c.shl(rLdc, 2);                                           // bytes

        for (int r = 0; r < MR; r++) {
            c.vmovdqu32(c.ptr[rC_out +   0], Zmm(14 + 0 * 6 + r));
            c.vmovdqu32(c.ptr[rC_out +  64], Zmm(14 + 1 * 6 + r));
            c.vmovdqu32(c.ptr[rC_out + 128], Zmm(14 + 2 * 6 + r));
            if (r < MR - 1) c.add(rC_out, rLdc);
        }
        c.jmp(done_store, CodeGenerator::T_NEAR);
    }

    // ── Requantize path (rq != null) ────────
    // Structurally identical to jit_packed_gemm_nr48_avx512_t::emit_requantize,
    // but specialized for a single MR × NR=48 panel (no col offset accumulator,
    // since this kernel processes exactly one N-block per call).
    //
    // zmm0 = col_corr (x_zp_neg * w_col_sums[sub] or 0)
    // zmm1 = bias (or 0)
    // zmm2 = qmin (broadcast float)
    // zmm3 = qmax (broadcast float)
    // zmm4 = y_zp (broadcast int32)
    // zmm5 = scratch (x_zp_neg, row_corr, xmm for final store)
    c.L(do_rq);
    {
        const Reg64 rRq   = rB;          // rsi — rq pointer
        const Reg64 rTmp1 = rSub;        // rdi — working pointer (scales/bias/col_sums)
        const Reg64 rTmp2 = rA0;         // rcx — y_rows[r] load

        c.mov(rRq, rT);                   // rq ptr → rsi

        c.vbroadcastss(Zmm(2), c.dword[rRq + 32]);     // rq_qmin
        c.vbroadcastss(Zmm(3), c.dword[rRq + 36]);     // rq_qmax
        c.vpbroadcastd(Zmm(4), c.dword[rRq + 28]);     // y_zp_int

        const int sub_acc_base[3] = { 14, 20, 26 };
        for (int s = 0; s < 3; s++) {
            const int base_zmm = sub_acc_base[s];
            const int col_u8   = s * 16;
            const int col_i32  = s * 64;

            // col_corr → zmm0
            Label no_wcs, wcs_done;
            c.mov(rTmp1, c.qword[rRq + 16]);           // w_col_sums
            c.test(rTmp1, rTmp1);
            c.jz(no_wcs, CodeGenerator::T_NEAR);
            c.vpbroadcastd(Zmm(5), c.dword[rRq + 24]); // x_zp_neg
            c.vmovdqu32(Zmm(0), c.ptr[rTmp1 + col_i32]);
            c.vpmulld(Zmm(0), Zmm(0), Zmm(5));
            c.jmp(wcs_done, CodeGenerator::T_NEAR);
            c.L(no_wcs);
            c.vpxord(Zmm(0), Zmm(0), Zmm(0));
            c.L(wcs_done);

            // bias → zmm1
            Label no_bias, bias_done;
            c.mov(rTmp1, c.qword[rRq + 8]);            // bias_int32
            c.test(rTmp1, rTmp1);
            c.jz(no_bias, CodeGenerator::T_NEAR);
            c.vmovdqu32(Zmm(1), c.ptr[rTmp1 + col_i32]);
            c.jmp(bias_done, CodeGenerator::T_NEAR);
            c.L(no_bias);
            c.vpxord(Zmm(1), Zmm(1), Zmm(1));
            c.L(bias_done);

            // output_scales base
            c.mov(rTmp1, c.qword[rRq + 0]);            // output_scales

            for (int r = 0; r < MR; r++) {
                const int acc = base_zmm + r;

                c.vpaddd(Zmm(acc), Zmm(acc), Zmm(0));  // + col_corr
                c.vpbroadcastd(Zmm(5), c.dword[rRq + 40 + r * 4]);  // row_corr[r]
                c.vpaddd(Zmm(acc), Zmm(acc), Zmm(5));  // + row_corr
                c.vpaddd(Zmm(acc), Zmm(acc), Zmm(1));  // + bias

                c.vcvtdq2ps(Zmm(acc), Zmm(acc));
                c.vmulps(Zmm(acc), Zmm(acc), c.ptr[rTmp1 + col_i32]);
                c.vmaxps(Zmm(acc), Zmm(acc), Zmm(2));  // qmin clamp
                c.vminps(Zmm(acc), Zmm(acc), Zmm(3));  // qmax clamp
                c.vcvtps2dq(Zmm(acc), Zmm(acc));
                c.vpaddd(Zmm(acc), Zmm(acc), Zmm(4));  // + y_zp

                c.vpmovusdb(Xmm(5), Zmm(acc));         // saturate → u8
                c.mov(rTmp2, c.qword[rRq + 64 + r * 8]);  // y_rows[r]
                c.vmovups(c.ptr[rTmp2 + col_u8], Xmm(5));
            }
        }
    }
    c.L(done_store);

    emit_epilogue();
    finalize();
}

// ───────────────────────────────────────────────────────────────────────────
// NHWC-direct GEMM JIT, NR=16
//
// Memcpy-free NR=16 kernel modeled on ORT's MlasConvSymKernelAvx512Vnni.
// 6 rows × 1 NR=16 col group, 6 accumulator zmms, indirection-buffer A-side.
//
// Register layout:
//   zmm14..zmm19 — 6 accumulators (one per row)
//   zmm0         — B sub-panel (live one k4-step)
//   zmm1         — A broadcast scratch (per row)
//
//   rA0..rA5     — 6 row-base GPRs, re-loaded per kp iteration from
//                  ptrs_kp_major[kp*MR + r]
//   r_xoff       — running ic byte offset (0, 4, 8, ..., IC-4) within kp
//   rB           — running B sub-panel pointer (advanced 64B per k4)
//   rIC4         — IC4 constant
//   rk4_rem      — k4 countdown
//   rPtrs        — indirection pointer (advanced 48B per kp — 6 ptrs × 8B)
//   rkp_rem      — kp countdown
//
// For MR<6 the unused row bases must still be valid; trailing accumulators
// stay zero and are not stored.
// ───────────────────────────────────────────────────────────────────────────
jit_nhwc_direct_gemm_nr16_avx512_t::jit_nhwc_direct_gemm_nr16_avx512_t(int mr_actual)
    : jit_kernel_t(32768)
{
    using namespace Xbyak;
    auto& c = gen();
    const int MR = mr_actual;

    // Save callee-saved: rbx(3), rbp(5), r12(12), r13(13), r14(14), r15(15).
    // XMM (Win64: xmm6-xmm15 callee-saved; mask bit i → xmm(6+i)):
    //   bit 0 → xmm6   (used as zmm6 = bias reg in the requantize epilogue)
    //   bit 8 → xmm14  (zmm14 accumulator)
    //   bit 9 → xmm15  (zmm15 accumulator)
    emit_prologue(
        (1u << 3) | (1u << 5) | (1u << 12) | (1u << 13) | (1u << 14) | (1u << 15),
        (1u << 0) | (1u << 8) | (1u << 9));

    const Reg64 rPtrs(Operand::RCX);
    const Reg64 rkp_rem(Operand::RDX);
    const Reg64 rIC4(8);           // r8
    const Reg64 rB(9);             // r9
    const Reg64 rA0(Operand::RAX);
    const Reg64 rA1(Operand::RBX);
    const Reg64 rA2(10);
    const Reg64 rA3(11);
    const Reg64 rA4(12);
    const Reg64 rA5(13);
    const Reg64 r_xoff(14);
    const Reg64 rk4_rem(15);
    const Reg64 rTmp(Operand::RBP);   // scratch (epilogue)

    // ── Argument loading ────────
    load_arg(0, Operand::RCX);          // ptrs_kp_major
    load_arg_i32(1, Operand::RDX);      // kHW
    load_arg_i32(2, 8);                 // IC4
    load_arg(3, 9);                     // B
    // args 4 (C), 5 (ldc), 6 (rq) loaded from stack in the epilogue.

    // ── Init 6 accumulators ────────
    for (int r = 0; r < MR; r++)
        c.vpxord(Zmm(14 + r), Zmm(14 + r), Zmm(14 + r));

    // ── K-loop: outer kp, inner k4 ────────
    Label kp_loop, k4_loop, kp_done;
    c.test(rkp_rem, rkp_rem);
    c.jz(kp_done, CodeGenerator::T_NEAR);

    c.L(kp_loop);
    {
        // Load 6 row pointers for this kp.
        // ptrs_kp_major is kp-major, so 6 ptrs = 48 contiguous bytes.
        c.mov(rA0, c.qword[rPtrs +  0]);
        c.mov(rA1, c.qword[rPtrs +  8]);
        c.mov(rA2, c.qword[rPtrs + 16]);
        c.mov(rA3, c.qword[rPtrs + 24]);
        c.mov(rA4, c.qword[rPtrs + 32]);
        c.mov(rA5, c.qword[rPtrs + 40]);
        c.add(rPtrs, 48);

        // k4 loop
        c.xor_(r_xoff, r_xoff);
        c.mov(rk4_rem, rIC4);

        c.L(k4_loop);
        {
            // 1 B load
            c.vmovdqu32(Zmm(0), c.ptr[rB]);
            c.add(rB, 64);

            // 6 rows × (broadcast + vpdpbusd)
            const Reg64 rArr[6] = { rA0, rA1, rA2, rA3, rA4, rA5 };
            for (int r = 0; r < MR; r++) {
                c.vpbroadcastd(Zmm(1), c.dword[rArr[r] + r_xoff]);
                c.vpdpbusd(Zmm(14 + r), Zmm(1), Zmm(0));
            }

            c.add(r_xoff, 4);
            c.dec(rk4_rem);
            c.jnz(k4_loop);
        }

        c.dec(rkp_rem);
        c.jnz(kp_loop);
    }
    c.L(kp_done);

    // ── Epilogue: either requantize+store-u8 (rq != null) or int32 store
    Label do_rq, done_store;
    c.mov(rTmp, c.qword[c.rsp + arg_stack_offset(6)]);  // rq
    c.test(rTmp, rTmp);
    c.jnz(do_rq, CodeGenerator::T_NEAR);

    // ── Int32 store path (rq == null) ────────
    {
        const Reg64 rC_out(Operand::RAX);    // rax (rA0 dead)
        const Reg64 rLdc  (Operand::RBX);    // rbx (rA1 dead)
        c.mov(rC_out, c.qword[c.rsp + arg_stack_offset(4)]);     // C
        c.movsxd(rLdc, c.dword[c.rsp + arg_stack_offset(5)]);    // ldc (i32→64)
        c.shl(rLdc, 2);                                            // * 4 bytes

        for (int r = 0; r < MR; r++) {
            c.vmovdqu32(c.ptr[rC_out], Zmm(14 + r));
            if (r < MR - 1) c.add(rC_out, rLdc);
        }
        c.jmp(done_store, CodeGenerator::T_NEAR);
    }

    // ── Requantize path (rq != null) ────────
    // Single sub-panel version of jit_packed_gemm_nr48_avx512_t::emit_requantize.
    //
    // zmm0 = col_corr  (x_zp_neg * w_col_sums or 0)
    // zmm2 = qmin
    // zmm3 = qmax
    // zmm4 = y_zp_int
    // zmm5 = scratch (x_zp_neg, row_corr, u8 store xmm)
    // zmm6 = bias
    c.L(do_rq);
    {
        const Reg64 rRq(Operand::RBP);       // rq pointer (already loaded)
        const Reg64 rTmp1(Operand::RAX);     // working (scales/bias/csums ptrs, y_rows)
        // rTmp (rbp) already has the rq pointer.

        c.vbroadcastss(Zmm(2), c.dword[rRq + 32]);  // rq_qmin
        c.vbroadcastss(Zmm(3), c.dword[rRq + 36]);  // rq_qmax
        c.vpbroadcastd(Zmm(4), c.dword[rRq + 28]);  // y_zp_int

        // col_corr → zmm0 (single sub-panel → col_i32 offset is 0)
        Label no_wcs, wcs_done;
        c.mov(rTmp1, c.qword[rRq + 16]);             // w_col_sums
        c.test(rTmp1, rTmp1);
        c.jz(no_wcs, CodeGenerator::T_NEAR);
        c.vpbroadcastd(Zmm(5), c.dword[rRq + 24]);   // x_zp_neg
        c.vmovdqu32(Zmm(0), c.ptr[rTmp1]);
        c.vpmulld(Zmm(0), Zmm(0), Zmm(5));
        c.jmp(wcs_done, CodeGenerator::T_NEAR);
        c.L(no_wcs);
        c.vpxord(Zmm(0), Zmm(0), Zmm(0));
        c.L(wcs_done);

        // bias → zmm6
        Label no_bias, bias_done;
        c.mov(rTmp1, c.qword[rRq + 8]);              // bias_int32
        c.test(rTmp1, rTmp1);
        c.jz(no_bias, CodeGenerator::T_NEAR);
        c.vmovdqu32(Zmm(6), c.ptr[rTmp1]);
        c.jmp(bias_done, CodeGenerator::T_NEAR);
        c.L(no_bias);
        c.vpxord(Zmm(6), Zmm(6), Zmm(6));
        c.L(bias_done);

        // output_scales base
        c.mov(rTmp1, c.qword[rRq + 0]);

        for (int r = 0; r < MR; r++) {
            const int acc = 14 + r;

            c.vpaddd(Zmm(acc), Zmm(acc), Zmm(0));       // +col_corr
            c.vpbroadcastd(Zmm(5), c.dword[rRq + 40 + r * 4]);
            c.vpaddd(Zmm(acc), Zmm(acc), Zmm(5));       // +row_corr
            c.vpaddd(Zmm(acc), Zmm(acc), Zmm(6));       // +bias

            c.vcvtdq2ps(Zmm(acc), Zmm(acc));
            c.vmulps(Zmm(acc), Zmm(acc), c.ptr[rTmp1]);
            c.vmaxps(Zmm(acc), Zmm(acc), Zmm(2));       // clamp low
            c.vminps(Zmm(acc), Zmm(acc), Zmm(3));       // clamp high
            c.vcvtps2dq(Zmm(acc), Zmm(acc));
            c.vpaddd(Zmm(acc), Zmm(acc), Zmm(4));       // +y_zp

            c.vpmovusdb(Xmm(5), Zmm(acc));
            // rTmp1 would be clobbered by loading y_rows[r], but we still
            // need it for subsequent rows. Use a different temp: reload
            // y_rows[r] via a short-lived use of rBx (rA1 is dead).
            const Reg64 rY(Operand::RBX);
            c.mov(rY, c.qword[rRq + 64 + r * 8]);
            c.vmovups(c.ptr[rY], Xmm(5));
        }
    }
    c.L(done_store);

    emit_epilogue();
    finalize();
}

} // namespace nnr::int8

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
