// JIT-compiled AVX-512 VNNI int8 GEMM micro-kernel (6x48).
// 6 rows × 48 columns = 18 ZMM accumulators, matching ORT's MLAS register allocation.
// Each K-step: 3 B loads + 6 A broadcasts + 18 VPDPBUSD.
//
// Bakes ZeroMode and column count at codegen time to eliminate hot-loop branches.

#include "jit_qgemm_ukernel.h"

#if defined(NNR_ARCH_X64) && defined(NNR_USE_XBYAK)

#include <xbyak/xbyak.h>

namespace nnr::int8 {

jit_qgemm_ukernel_avx512_t::jit_qgemm_ukernel_avx512_t(bool zero_mode, int col_count)
    : jit_kernel_t(8192)
{
    using namespace Xbyak;
    auto& c = gen();

    // Number of 16-col ZMM strips (1, 2, or 3)
    int nstrips = (col_count + 15) / 16;
    if (nstrips > 3) nstrips = 3;

    // Prologue: save callee-saved GPRs. Uses zmm0-zmm31 so save xmm6-15 on Win64.
    emit_prologue((1u<<3)|(1u<<12)|(1u<<13)|(1u<<14)|(1u<<15));

    // Register aliases (platform-neutral after prologue)
    const Reg64 reg_pa(Operand::RDI);     // arg0: packed_A
    const Reg64 reg_pb(Operand::RSI);     // arg1: packed_B
    const Reg64 reg_c(10);                // r10 ← arg2: C
    const Reg64 reg_kcount(11);           // r11 ← arg3: PackedCountK
    const Reg64 reg_ldc(12);              // r12 ← arg4: ldc (in int32 elements)
    const Reg64 reg_rowsum(13);           // r13 ← arg5: RowSumBuffer
    const Reg64 reg_colsum(14);           // r14 ← arg6: ColumnSumBuffer
    const Reg64 reg_tmp(Operand::RBX);
    const Reg64 reg_c3(15);              // r15: C + 3*ldc*4 (for rows 3-5)

    // Load arguments
    load_arg(0, Operand::RDI);
    load_arg(1, Operand::RSI);
    load_arg(2, 10);
    load_arg_i32(3, 11);
    load_arg_i32(4, 12);
    c.shl(reg_ldc, 2);                   // ldc: elements → bytes
    load_arg(5, 13);
    load_arg(6, 14);

    // Compute C row pointers: rows 3-5 use reg_c3 = C + 3*ldc
    c.lea(reg_tmp, c.ptr [reg_ldc + reg_ldc * 2]);  // tmp = 3*ldc_bytes
    c.lea(reg_c3, c.ptr [reg_c + reg_tmp]);          // c3 = C + 3*ldc_bytes

    // Accumulator register map:
    //   Row 0: zmm14, zmm20, zmm26  (col 0-15, 16-31, 32-47)
    //   Row 1: zmm15, zmm21, zmm27
    //   Row 2: zmm16, zmm22, zmm28
    //   Row 3: zmm17, zmm23, zmm29
    //   Row 4: zmm18, zmm24, zmm30
    //   Row 5: zmm19, zmm25, zmm31
    // B loads: zmm0, zmm1, zmm2
    // A broadcast: zmm3
    // Scratch: zmm4 (unused with VNNI)

    // --- Initialize accumulators ---
    if (zero_mode) {
        for (int r = 0; r < 6; r++) {
            c.vpxord(Zmm(14 + r), Zmm(14 + r), Zmm(14 + r));
            if (nstrips >= 2) c.vpxord(Zmm(20 + r), Zmm(20 + r), Zmm(20 + r));
            if (nstrips >= 3) c.vpxord(Zmm(26 + r), Zmm(26 + r), Zmm(26 + r));
        }
    } else {
        // Load existing C values into accumulators
        for (int r = 0; r < 6; r++) {
            // Row pointer: r < 3 ? C + r*ldc : C3 + (r-3)*ldc
            Reg64 base = (r < 3) ? reg_c : reg_c3;
            int row_in_group = (r < 3) ? r : r - 3;

            if (row_in_group == 0) {
                c.vmovdqu32(Zmm(14 + r), c.ptr [base]);
                if (nstrips >= 2) c.vmovdqu32(Zmm(20 + r), c.ptr [base + 64]);
                if (nstrips >= 3) c.vmovdqu32(Zmm(26 + r), c.ptr [base + 128]);
            } else if (row_in_group == 1) {
                c.vmovdqu32(Zmm(14 + r), c.ptr [base + reg_ldc]);
                if (nstrips >= 2) c.vmovdqu32(Zmm(20 + r), c.ptr [base + reg_ldc + 64]);
                if (nstrips >= 3) c.vmovdqu32(Zmm(26 + r), c.ptr [base + reg_ldc + 128]);
            } else {
                c.lea(reg_tmp, c.ptr [base + reg_ldc * 2]);
                c.vmovdqu32(Zmm(14 + r), c.ptr [reg_tmp]);
                if (nstrips >= 2) c.vmovdqu32(Zmm(20 + r), c.ptr [reg_tmp + 64]);
                if (nstrips >= 3) c.vmovdqu32(Zmm(26 + r), c.ptr [reg_tmp + 128]);
            }
        }
    }

    // --- K-loop ---
    Label k_loop;
    c.test(reg_kcount, reg_kcount);
    c.jz("k_done", CodeGenerator::T_NEAR);

    c.L(k_loop);
    {
        // Load B panel: 1-3 ZMM strips of 16 uint32 each
        c.vmovdqu32(Zmm(0), c.ptr [reg_pb]);
        if (nstrips >= 2) c.vmovdqu32(Zmm(1), c.ptr [reg_pb + 64]);
        if (nstrips >= 3) c.vmovdqu32(Zmm(2), c.ptr [reg_pb + 128]);

        // 6 rows: broadcast A[r] (4 bytes = 1 VNNI group) and VPDPBUSD
        for (int r = 0; r < 6; r++) {
            c.vpbroadcastd(Zmm(3), c.dword [reg_pa + r * 4]);
            c.vpdpbusd(Zmm(14 + r), Zmm(3), Zmm(0));
            if (nstrips >= 2) c.vpdpbusd(Zmm(20 + r), Zmm(3), Zmm(1));
            if (nstrips >= 3) c.vpdpbusd(Zmm(26 + r), Zmm(3), Zmm(2));
        }

        // Advance pointers
        c.add(reg_pa, QGEMM_MR * 4);    // 6 rows × 4 bytes per VNNI group
        c.add(reg_pb, QGEMM_NR * 4);    // 48 columns × 4 bytes per VNNI group
        c.dec(reg_kcount);
        c.jnz(k_loop);
    }
    c.L("k_done");

    // --- Zero-point compensation ---
    // C[i,j] += RowSumBuffer[i] + ColumnSumBuffer[j]
    // (Caller pre-computes: RowSum[i] = -b_zp * sum_A_row[i] - K*a_zp*b_zp + K*a_zp*b_zp
    //  and ColSum[j] = -a_zp * sum_B_col[j]. The +K*a_zp*b_zp term is folded into RowSum.)
    // So here we just add RowSum[i] + ColSum[j] to each C[i,j].

    // Load column sums into zmm4-zmm6
    c.vmovdqu32(Zmm(4), c.ptr [reg_colsum]);
    if (nstrips >= 2) c.vmovdqu32(Zmm(5), c.ptr [reg_colsum + 64]);
    if (nstrips >= 3) c.vmovdqu32(Zmm(6), c.ptr [reg_colsum + 128]);

    for (int r = 0; r < 6; r++) {
        // Broadcast row sum for this row
        c.vpbroadcastd(Zmm(3), c.dword [reg_rowsum + r * 4]);
        // Add row_sum + col_sum to accumulators
        c.vpaddd(Zmm(14 + r), Zmm(14 + r), Zmm(3));
        c.vpaddd(Zmm(14 + r), Zmm(14 + r), Zmm(4));
        if (nstrips >= 2) {
            c.vpaddd(Zmm(20 + r), Zmm(20 + r), Zmm(3));
            c.vpaddd(Zmm(20 + r), Zmm(20 + r), Zmm(5));
        }
        if (nstrips >= 3) {
            c.vpaddd(Zmm(26 + r), Zmm(26 + r), Zmm(3));
            c.vpaddd(Zmm(26 + r), Zmm(26 + r), Zmm(6));
        }
    }

    // --- Store accumulators to C ---
    for (int r = 0; r < 6; r++) {
        Reg64 base = (r < 3) ? reg_c : reg_c3;
        int row_in_group = (r < 3) ? r : r - 3;

        if (row_in_group == 0) {
            c.vmovdqu32(c.ptr [base], Zmm(14 + r));
            if (nstrips >= 2) c.vmovdqu32(c.ptr [base + 64], Zmm(20 + r));
            if (nstrips >= 3) c.vmovdqu32(c.ptr [base + 128], Zmm(26 + r));
        } else if (row_in_group == 1) {
            c.vmovdqu32(c.ptr [base + reg_ldc], Zmm(14 + r));
            if (nstrips >= 2) c.vmovdqu32(c.ptr [base + reg_ldc + 64], Zmm(20 + r));
            if (nstrips >= 3) c.vmovdqu32(c.ptr [base + reg_ldc + 128], Zmm(26 + r));
        } else {
            c.lea(reg_tmp, c.ptr [base + reg_ldc * 2]);
            c.vmovdqu32(c.ptr [reg_tmp], Zmm(14 + r));
            if (nstrips >= 2) c.vmovdqu32(c.ptr [reg_tmp + 64], Zmm(20 + r));
            if (nstrips >= 3) c.vmovdqu32(c.ptr [reg_tmp + 128], Zmm(26 + r));
        }
    }

    emit_epilogue();
    finalize();
}

} // namespace nnr::int8

#endif // NNR_ARCH_X64 && NNR_USE_XBYAK
