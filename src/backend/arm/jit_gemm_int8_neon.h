#pragma once
// JIT int8 SMMLA GEMM ukernel (AArch64 NEON, i8mm) via xbyak_aarch64.
//
// Handles the full MR=4 row block: one call computes all N/NR=16 col-tiles for
// an i-block, including the SMMLA accumulation, the 2×2-block → row-major
// extraction, the row-bias + column-sum zero-point compensation, and the
// direct store to the final int32 C tile.
//
// xbyak_aarch64 only exposes the SVE variant of SMMLA (ZRegS/ZRegB), so the
// NEON/Advanced-SIMD encoding is emitted directly via dd():
//   SMMLA Vd.4S, Vn.16B, Vm.16B  ==  0x4E80A400 | (Rm << 16) | (Rn << 5) | Rd
//
// A is expected pre-packed per-i-block with XOR 0x80 fused in (caller
// responsibility, matches gemm_int8_smmla's a_pack layout).
// B is the standard gemm_int8_smmla pack (2-col × 8-K groups, zero-padded).
// col_sums[N16] int32 precomputed by pack_b_int8_smmla_and_col_sums.
// aux->a_off = (128 - a_zp), aux->rb[r] = -b_zp * row_sums[i+r] + K*a_zp*b_zp.
//
// Final C formula (applied inline):
//   C[i+r, j+c] = acc_raw[r, c] + aux->rb[r] + aux->a_off * col_sums[j+c]

#ifdef NNR_USE_XBYAK_AARCH64
#if defined(__aarch64__) || defined(_M_ARM64)

#include <xbyak_aarch64/xbyak_aarch64.h>
#include <cstddef>
#include <cstdint>

namespace nnr::int8::neon_jit {

struct JitSmmlaAux {
    int32_t a_off;   // 128 - a_zp
    int32_t rb[4];   // row biases for the 4 rows of the current i-block
};

// Signature:
//   x0 = a_pack             (per-i-block packed A, XOR-0x80 applied)
//   x1 = b_pack              (SMMLA-packed B, pointing at the current
//                             i-block's column 0 block — i.e. the whole matrix)
//   x2 = col_sums            (int32[N16])
//   x3 = c_out               (int32[4, N16] output tile top-left)
//   x4 = k_groups            (K/8 groups, partial group zero-padded in packs)
//   x5 = n_iters             (N/16 j-tiles)
//   x6 = n16                 (column stride in int32 elements)
//   x7 = aux                 (JitSmmlaAux*)
using jit_smmla_4xN_t = void(*)(
    const int8_t* a_pack,
    const int8_t* b_pack,
    const int32_t* col_sums,
    int32_t* c_out,
    int k_groups,
    int n_iters,
    int n16,
    const JitSmmlaAux* aux);

class JitSmmla_4xN : public Xbyak_aarch64::CodeGenerator {
public:
    JitSmmla_4xN() : CodeGenerator(16384) {
        generate();
        ready();
    }

    jit_smmla_4xN_t getFunc() const { return getCode<jit_smmla_4xN_t>(); }

private:
    // Emit NEON SMMLA Vd.4S, Vn.16B, Vm.16B via raw encoding.
    // xbyak_aarch64's smmla mnemonic only covers the SVE variant.
    void smmla(int Vd, int Vn, int Vm) {
        dd(0x4E80A400u | ((uint32_t)(Vm & 0x1F) << 16)
                       | ((uint32_t)(Vn & 0x1F) << 5)
                       | (uint32_t)(Vd & 0x1F));
    }

    void generate() {
        using namespace Xbyak_aarch64;

        // ---------- Prologue ----------
        // Save d8-d15 (AAPCS64 callee-saved lower halves). 64 bytes on stack.
        stp(d8,  d9,  pre_ptr(sp, -64));
        stp(d10, d11, ptr(sp, 16));
        stp(d12, d13, ptr(sp, 32));
        stp(d14, d15, ptr(sp, 48));

        // ---------- Load aux (a_off, rb[0..3]) ----------
        // aux layout: struct { int32 a_off; int32 rb[4]; } = 20 bytes.
        ldr(WReg(8),  ptr(x7,  0));   // a_off
        ldr(WReg(9),  ptr(x7,  4));   // rb0
        ldr(WReg(10), ptr(x7,  8));   // rb1
        ldr(WReg(11), ptr(x7, 12));   // rb2
        ldr(WReg(12), ptr(x7, 16));   // rb3

        // Broadcast to vregs (persist across the whole kernel).
        dup(VReg4S(2), WReg(8));      // v2 = a_off_v
        dup(VReg4S(4), WReg(9));      // v4 = rb_v0
        dup(VReg4S(5), WReg(10));     // v5 = rb_v1
        dup(VReg4S(6), WReg(11));     // v6 = rb_v2
        dup(VReg4S(7), WReg(12));     // v7 = rb_v3

        // ---------- Precompute strides ----------
        // k_stride_bytes = n16 * 8 (bytes between K-groups in packed_B)
        // ldc_bytes      = n16 * 4 (bytes between rows in C, int32 ldc)
        lsl(x11, x6, 3);              // x11 = n16 * 8
        lsl(x12, x6, 2);              // x12 = n16 * 4

        // Save a_pack base (consumed per j-iter) — keep in x0, copy to x8 each j.
        // Save b_pack base — keep in x1, will add j*128 per j-iter (x9 = current b).
        // Save col_sums base — x2, add 64 per j-iter (or compute from j counter).
        // Save c_out base — x3, add 64 per j-iter.

        // ---------- J-loop ----------
        Label j_loop, j_done;
        cbz(x5, j_done);

        L(j_loop);
        {
            // Init 16 accs to zero (v16..v31).
            for (int i = 16; i < 32; i++)
                movi(VReg16B(i), 0);

            // Save per-j-tile a_pack/b_pack starts into scratch regs.
            mov(x8, x0);              // x8 = current a_pack pointer
            mov(x9, x1);              // x9 = current b_pack pointer
            mov(x10, x4);             // x10 = k_loop counter

            // Skip K-loop if k_groups==0 (defensive; shouldn't happen).
            Label k_loop, k_done;
            cbz(x10, k_done);

            L(k_loop);
            {
                // Load A: two 16-byte chunks per K-group.
                ldr(QReg(0),  ptr(x8,  0));
                ldr(QReg(1),  ptr(x8, 16));
                add(x8, x8, 32);

                // Load 8 B chunks (col-pairs 0..7). Batched-load pattern —
                // tried interleaving loads with SMMLAs and it measured 3%
                // slower; hardware already hides the load latency here.
                ldr(QReg(8),  ptr(x9,   0));
                ldr(QReg(9),  ptr(x9,  16));
                ldr(QReg(10), ptr(x9,  32));
                ldr(QReg(11), ptr(x9,  48));
                ldr(QReg(12), ptr(x9,  64));
                ldr(QReg(13), ptr(x9,  80));
                ldr(QReg(14), ptr(x9,  96));
                ldr(QReg(15), ptr(x9, 112));

                // Prefetch next K-group of B (hides L2 latency when B doesn't
                // fully reside in L1 — packed_B for 256×256 = 64 KB).
                prfm(PLDL1KEEP, ptr(x9, x11));

                // 16 SMMLAs. Row-pair 0 (v0) → v16..v23. Row-pair 1 (v1) → v24..v31.
                smmla(16, 0,  8);  smmla(24, 1,  8);
                smmla(17, 0,  9);  smmla(25, 1,  9);
                smmla(18, 0, 10);  smmla(26, 1, 10);
                smmla(19, 0, 11);  smmla(27, 1, 11);
                smmla(20, 0, 12);  smmla(28, 1, 12);
                smmla(21, 0, 13);  smmla(29, 1, 13);
                smmla(22, 0, 14);  smmla(30, 1, 14);
                smmla(23, 0, 15);  smmla(31, 1, 15);

                add(x9, x9, x11);     // advance b_pack to next K-group
                subs(x10, x10, 1);
                bne(k_loop);
            }
            L(k_done);

            // ---------- Epilogue: ZIP + compensation + store ----------
            // acc Vn.4S lanes = [C[r*2, c*2], C[r*2, c*2+1], C[r*2+1, c*2], C[r*2+1, c*2+1]]
            // zip1 Vd.2D, Va.2D, Vb.2D = [Va[0..1], Vb[0..1]]  (top row strip)
            // zip2 Vd.2D, Va.2D, Vb.2D = [Va[2..3], Vb[2..3]]  (bottom row strip)

            // ---- Row-pair 0 (rows 0, 1): accs v16..v23, biases v4/v5 ----
            // Produce row 0 cols 0..15 into v0, v8, v10, v12 (v1/v9/v11/v13 hold row 1).
            zip1(VReg2D(0),  VReg2D(16), VReg2D(17));   // row 0 cols 0-3
            zip2(VReg2D(1),  VReg2D(16), VReg2D(17));   // row 1 cols 0-3
            zip1(VReg2D(8),  VReg2D(18), VReg2D(19));   // row 0 cols 4-7
            zip2(VReg2D(9),  VReg2D(18), VReg2D(19));   // row 1 cols 4-7
            zip1(VReg2D(10), VReg2D(20), VReg2D(21));   // row 0 cols 8-11
            zip2(VReg2D(11), VReg2D(20), VReg2D(21));   // row 1 cols 8-11
            zip1(VReg2D(12), VReg2D(22), VReg2D(23));   // row 0 cols 12-15
            zip2(VReg2D(13), VReg2D(22), VReg2D(23));   // row 1 cols 12-15

            // Load col_sums[j..j+15] from (x2) — the caller advances x2 per j-iter.
            ldr(QReg(14), ptr(x2,   0));                // cs03
            ldr(QReg(15), ptr(x2,  16));                // cs47
            // v16-v17 are now free (accs consumed); reuse for cs8B, csCF.
            ldr(QReg(16), ptr(x2,  32));                // cs8B
            ldr(QReg(17), ptr(x2,  48));                // csCF

            // Row 0 compensation: add rb_v0 + a_off * col_sums.
            add(VReg4S(0),  VReg4S(0),  VReg4S(4));     // + rb_v0
            add(VReg4S(8),  VReg4S(8),  VReg4S(4));
            add(VReg4S(10), VReg4S(10), VReg4S(4));
            add(VReg4S(12), VReg4S(12), VReg4S(4));
            mla(VReg4S(0),  VReg4S(2),  VReg4S(14));    // += a_off * cs03
            mla(VReg4S(8),  VReg4S(2),  VReg4S(15));
            mla(VReg4S(10), VReg4S(2),  VReg4S(16));
            mla(VReg4S(12), VReg4S(2),  VReg4S(17));

            // Store row 0 (4 × QReg to c_out).
            str(QReg(0),  ptr(x3,  0));
            str(QReg(8),  ptr(x3, 16));
            str(QReg(10), ptr(x3, 32));
            str(QReg(12), ptr(x3, 48));

            // Row 1 compensation.
            add(VReg4S(1),  VReg4S(1),  VReg4S(5));     // + rb_v1
            add(VReg4S(9),  VReg4S(9),  VReg4S(5));
            add(VReg4S(11), VReg4S(11), VReg4S(5));
            add(VReg4S(13), VReg4S(13), VReg4S(5));
            mla(VReg4S(1),  VReg4S(2),  VReg4S(14));
            mla(VReg4S(9),  VReg4S(2),  VReg4S(15));
            mla(VReg4S(11), VReg4S(2),  VReg4S(16));
            mla(VReg4S(13), VReg4S(2),  VReg4S(17));

            // Store row 1 to c_out + ldc_bytes.
            add(x13, x3, x12);                          // x13 = c_out + ldc
            str(QReg(1),  ptr(x13,  0));
            str(QReg(9),  ptr(x13, 16));
            str(QReg(11), ptr(x13, 32));
            str(QReg(13), ptr(x13, 48));

            // ---- Row-pair 1 (rows 2, 3): accs v24..v31, biases v6/v7 ----
            // Produce row 2 cols 0..15 into v0, v8, v10, v12 (reusing now-dead slots).
            zip1(VReg2D(0),  VReg2D(24), VReg2D(25));   // row 2 cols 0-3
            zip2(VReg2D(1),  VReg2D(24), VReg2D(25));   // row 3 cols 0-3
            zip1(VReg2D(8),  VReg2D(26), VReg2D(27));
            zip2(VReg2D(9),  VReg2D(26), VReg2D(27));
            zip1(VReg2D(10), VReg2D(28), VReg2D(29));
            zip2(VReg2D(11), VReg2D(28), VReg2D(29));
            zip1(VReg2D(12), VReg2D(30), VReg2D(31));
            zip2(VReg2D(13), VReg2D(30), VReg2D(31));

            // col_sums still in v14-v17 — reuse.
            // Row 2 compensation.
            add(VReg4S(0),  VReg4S(0),  VReg4S(6));     // + rb_v2
            add(VReg4S(8),  VReg4S(8),  VReg4S(6));
            add(VReg4S(10), VReg4S(10), VReg4S(6));
            add(VReg4S(12), VReg4S(12), VReg4S(6));
            mla(VReg4S(0),  VReg4S(2),  VReg4S(14));
            mla(VReg4S(8),  VReg4S(2),  VReg4S(15));
            mla(VReg4S(10), VReg4S(2),  VReg4S(16));
            mla(VReg4S(12), VReg4S(2),  VReg4S(17));

            // Store row 2 to c_out + 2*ldc.
            add(x13, x3, x12);          // ldc
            add(x13, x13, x12);         // 2*ldc
            str(QReg(0),  ptr(x13,  0));
            str(QReg(8),  ptr(x13, 16));
            str(QReg(10), ptr(x13, 32));
            str(QReg(12), ptr(x13, 48));

            // Row 3 compensation.
            add(VReg4S(1),  VReg4S(1),  VReg4S(7));     // + rb_v3
            add(VReg4S(9),  VReg4S(9),  VReg4S(7));
            add(VReg4S(11), VReg4S(11), VReg4S(7));
            add(VReg4S(13), VReg4S(13), VReg4S(7));
            mla(VReg4S(1),  VReg4S(2),  VReg4S(14));
            mla(VReg4S(9),  VReg4S(2),  VReg4S(15));
            mla(VReg4S(11), VReg4S(2),  VReg4S(16));
            mla(VReg4S(13), VReg4S(2),  VReg4S(17));

            // Store row 3 to c_out + 3*ldc.
            add(x13, x13, x12);         // 3*ldc
            str(QReg(1),  ptr(x13,  0));
            str(QReg(9),  ptr(x13, 16));
            str(QReg(11), ptr(x13, 32));
            str(QReg(13), ptr(x13, 48));

            // ---------- Advance per j-iter ----------
            // b_pack base advances by 128 bytes per j-tile (8 col-pairs × 16 B).
            add(x1, x1, 128);
            // col_sums advances by 64 bytes (16 cols × 4 bytes).
            add(x2, x2, 64);
            // c_out advances by 64 bytes.
            add(x3, x3, 64);

            subs(x5, x5, 1);
            bne(j_loop);
        }
        L(j_done);

        // ---------- Epilogue: restore d8-d15 ----------
        ldp(d14, d15, ptr(sp, 48));
        ldp(d12, d13, ptr(sp, 32));
        ldp(d10, d11, ptr(sp, 16));
        ldp(d8,  d9,  post_ptr(sp, 64));
        ret();
    }
};

inline jit_smmla_4xN_t get_jit_smmla_4xN() {
    static JitSmmla_4xN kernel;
    return kernel.getFunc();
}

} // namespace nnr::int8::neon_jit

#endif // aarch64
#endif // NNR_USE_XBYAK_AARCH64
