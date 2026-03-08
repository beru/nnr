#pragma once
// JIT GEMM micro-kernel for ARM64 NEON using xbyak_aarch64.
// MR=6, NR=16: matches KleidiAI's tile geometry for maximum throughput.
//
// Two kernels generated:
//   beta0: C = A * B         (first K-block, init accumulators to zero)
//   beta1: C += A * B        (subsequent K-blocks, load from C first)
//
// Inner loop: K-unroll=4, 96 FMAs per iteration
// Register allocation: v8-v31 = 24 accumulators, v0-v5 = 6 LHS rows, v6-v7 = RHS

#ifdef NNR_USE_XBYAK_AARCH64

#include <xbyak_aarch64/xbyak_aarch64.h>
#include <cstddef>

namespace nnr {
namespace neon_jit {

static constexpr int JIT_MR = 6;
static constexpr int JIT_NR = 16;

// Signature: void(const float* A, const float* B_packed, float* C,
//                 int K, int lda, int ldc)
using jit_gemm_func_t = void(*)(const float* A, const float* B_packed, float* C,
                                 int K, int lda, int ldc);

class JitGemmMicroKernel6x16 : public Xbyak_aarch64::CodeGenerator {
    using XReg = Xbyak_aarch64::XReg;
    using VReg4S = Xbyak_aarch64::VReg4S;

    bool beta_one_;  // false = beta0 (init zero), true = beta1 (load from C)

public:
    // @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=[GEMM,JIT] tiling=[MR,NR,K]
    JitGemmMicroKernel6x16(bool beta_one)
        : CodeGenerator(8192), beta_one_(beta_one)
    {
        generate();
        ready();
    }

    // @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=[GEMM,JIT]
    jit_gemm_func_t getFunc() const { return getCode<jit_gemm_func_t>(); }

private:
    // Emit fmla for one K-step across all 6 rows and 4 NR-chunks.
    // Interleaved scheduling: load one chunk, do all 6 FMAs, then load next.
    // This hides the 4-cycle load latency (6 FMAs ≥ 3 cycles at 2 FMA/cycle).
    // @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=[GEMM,JIT] tiling=K
    void emit_k_step(int ks) {
        using namespace Xbyak_aarch64;

        // Chunk 0: load v6, 6 FMAs
        ldr(QReg(6), ptr(x1, ks * 64));
        for (int row = 0; row < 6; row++)
            fmla(VReg4S(8 + row * 4), VReg4S(6), VReg4S(row)[ks]);

        // Chunk 1: load v7, 6 FMAs
        ldr(QReg(7), ptr(x1, ks * 64 + 16));
        for (int row = 0; row < 6; row++)
            fmla(VReg4S(9 + row * 4), VReg4S(7), VReg4S(row)[ks]);

        // Chunk 2: load v6, 6 FMAs
        ldr(QReg(6), ptr(x1, ks * 64 + 32));
        for (int row = 0; row < 6; row++)
            fmla(VReg4S(10 + row * 4), VReg4S(6), VReg4S(row)[ks]);

        // Chunk 3: load v7, 6 FMAs
        ldr(QReg(7), ptr(x1, ks * 64 + 48));
        for (int row = 0; row < 6; row++)
            fmla(VReg4S(11 + row * 4), VReg4S(7), VReg4S(row)[ks]);
    }

    // @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=[GEMM,JIT] tiling=[K,MR,NR]
    void generate() {
        using namespace Xbyak_aarch64;

        // x0=A, x1=B, x2=C, x3=K, x4=lda(floats), x5=ldc(floats)
        const XReg& pA = x0, &pB = x1, &pC = x2;
        const XReg& reg_K = x3, &reg_lda = x4, &reg_ldc = x5;
        const XReg& pA1 = x19, &pA2 = x20, &pA3 = x21;
        const XReg& pA4 = x22, &pA5 = x23, &reg_Kloop = x24;
        const XReg& pC1 = x25;

        // Save callee-saved
        stp(x19, x20, pre_ptr(sp, -112));
        stp(x21, x22, ptr(sp, 16));
        stp(x23, x24, ptr(sp, 32));
        str(x25, ptr(sp, 48));
        stp(d8, d9, ptr(sp, 56));
        stp(d10, d11, ptr(sp, 72));
        stp(d12, d13, ptr(sp, 88));

        // Convert strides to bytes
        lsl(reg_lda, reg_lda, 2);
        lsl(reg_ldc, reg_ldc, 2);

        // Row pointers for A
        add(pA1, pA, reg_lda);
        add(pA2, pA1, reg_lda);
        add(pA3, pA2, reg_lda);
        add(pA4, pA3, reg_lda);
        add(pA5, pA4, reg_lda);

        // Initialize accumulators
        if (beta_one_) {
            // Load from C
            // pC points to row 0; compute row pointers
            add(pC1, pC, reg_ldc);  // temp for row iteration
            // Row 0
            ldr(QReg(8),  ptr(pC));
            ldr(QReg(9),  ptr(pC, 16));
            ldr(QReg(10), ptr(pC, 32));
            ldr(QReg(11), ptr(pC, 48));
            // Row 1
            ldr(QReg(12), ptr(pC1));
            ldr(QReg(13), ptr(pC1, 16));
            ldr(QReg(14), ptr(pC1, 32));
            ldr(QReg(15), ptr(pC1, 48));
            add(pC1, pC1, reg_ldc);
            // Row 2
            ldr(QReg(16), ptr(pC1));
            ldr(QReg(17), ptr(pC1, 16));
            ldr(QReg(18), ptr(pC1, 32));
            ldr(QReg(19), ptr(pC1, 48));
            add(pC1, pC1, reg_ldc);
            // Row 3
            ldr(QReg(20), ptr(pC1));
            ldr(QReg(21), ptr(pC1, 16));
            ldr(QReg(22), ptr(pC1, 32));
            ldr(QReg(23), ptr(pC1, 48));
            add(pC1, pC1, reg_ldc);
            // Row 4
            ldr(QReg(24), ptr(pC1));
            ldr(QReg(25), ptr(pC1, 16));
            ldr(QReg(26), ptr(pC1, 32));
            ldr(QReg(27), ptr(pC1, 48));
            add(pC1, pC1, reg_ldc);
            // Row 5
            ldr(QReg(28), ptr(pC1));
            ldr(QReg(29), ptr(pC1, 16));
            ldr(QReg(30), ptr(pC1, 32));
            ldr(QReg(31), ptr(pC1, 48));
        } else {
            for (int i = 8; i <= 31; i++)
                movi(VReg16B(i), 0);
        }

        mov(reg_Kloop, reg_K);

        // Software-pipelined main loop: K-unroll=4
        // Pre-load A before loop. At end of each iteration, reload A for
        // next iteration interleaved with the last chunk's FMAs.
        Label loop_k4, loop_k1, done_k, loop_k4_tail;
        cmp(reg_Kloop, 4);
        blt(loop_k1);

        // Prolog: load first A batch
        ldr(QReg(0), ptr(pA));    add(pA, pA, 16);
        ldr(QReg(1), ptr(pA1));   add(pA1, pA1, 16);
        ldr(QReg(2), ptr(pA2));   add(pA2, pA2, 16);
        ldr(QReg(3), ptr(pA3));   add(pA3, pA3, 16);
        ldr(QReg(4), ptr(pA4));   add(pA4, pA4, 16);
        ldr(QReg(5), ptr(pA5));   add(pA5, pA5, 16);
        sub(reg_Kloop, reg_Kloop, 4);
        cmp(reg_Kloop, 4);
        blt(loop_k4_tail);

        L(loop_k4);
        {
            prfm(PLDL1KEEP, ptr(pA, 128));
            prfm(PLDL1KEEP, ptr(pB, 256));

            // K-steps 0-2: standard interleaved scheduling
            emit_k_step(0);
            emit_k_step(1);
            emit_k_step(2);

            // K-step 3: chunks 0-2 normal, chunk 3 interleaved with A reloads
            {
                using namespace Xbyak_aarch64;
                // Chunks 0-2 of K-step 3
                ldr(QReg(6), ptr(x1, 3 * 64));
                for (int row = 0; row < 6; row++)
                    fmla(VReg4S(8 + row * 4), VReg4S(6), VReg4S(row)[3]);

                ldr(QReg(7), ptr(x1, 3 * 64 + 16));
                for (int row = 0; row < 6; row++)
                    fmla(VReg4S(9 + row * 4), VReg4S(7), VReg4S(row)[3]);

                ldr(QReg(6), ptr(x1, 3 * 64 + 32));
                for (int row = 0; row < 6; row++)
                    fmla(VReg4S(10 + row * 4), VReg4S(6), VReg4S(row)[3]);

                // Chunk 3: interleave A reloads after each row's last use
                ldr(QReg(7), ptr(x1, 3 * 64 + 48));
                fmla(VReg4S(11), VReg4S(7), VReg4S(0)[3]);  // last use of v0
                ldr(QReg(0), ptr(pA));                        // reload v0
                fmla(VReg4S(15), VReg4S(7), VReg4S(1)[3]);  // last use of v1
                ldr(QReg(1), ptr(pA1));                       // reload v1
                fmla(VReg4S(19), VReg4S(7), VReg4S(2)[3]);  // last use of v2
                ldr(QReg(2), ptr(pA2));                       // reload v2
                fmla(VReg4S(23), VReg4S(7), VReg4S(3)[3]);  // last use of v3
                ldr(QReg(3), ptr(pA3));                       // reload v3
                fmla(VReg4S(27), VReg4S(7), VReg4S(4)[3]);  // last use of v4
                ldr(QReg(4), ptr(pA4));                       // reload v4
                fmla(VReg4S(31), VReg4S(7), VReg4S(5)[3]);  // last use of v5
                ldr(QReg(5), ptr(pA5));                       // reload v5
            }

            add(pB, pB, 256);
            add(pA, pA, 16);
            add(pA1, pA1, 16);
            add(pA2, pA2, 16);
            add(pA3, pA3, 16);
            add(pA4, pA4, 16);
            add(pA5, pA5, 16);
            sub(reg_Kloop, reg_Kloop, 4);
            cmp(reg_Kloop, 4);
            bge(loop_k4);
        }

        // Tail: last K=4 block (A already loaded, no next-iter reload needed)
        L(loop_k4_tail);
        {
            emit_k_step(0);
            emit_k_step(1);
            emit_k_step(2);
            emit_k_step(3);
            add(pB, pB, 256);
        }

        // K remainder
        cbz(reg_Kloop, done_k);

        L(loop_k1);
        {
            // Load 1 scalar per row, broadcast to full vector
            ld1r(VReg4S(0), ptr(pA));    add(pA, pA, 4);
            ld1r(VReg4S(1), ptr(pA1));   add(pA1, pA1, 4);
            ld1r(VReg4S(2), ptr(pA2));   add(pA2, pA2, 4);
            ld1r(VReg4S(3), ptr(pA3));   add(pA3, pA3, 4);
            ld1r(VReg4S(4), ptr(pA4));   add(pA4, pA4, 4);
            ld1r(VReg4S(5), ptr(pA5));   add(pA5, pA5, 4);

            // 4 NR-chunks with interleaved loads
            ldr(QReg(6), ptr(pB));
            for (int row = 0; row < 6; row++)
                fmla(VReg4S(8 + row * 4), VReg4S(6), VReg4S(row));

            ldr(QReg(7), ptr(pB, 16));
            for (int row = 0; row < 6; row++)
                fmla(VReg4S(9 + row * 4), VReg4S(7), VReg4S(row));

            ldr(QReg(6), ptr(pB, 32));
            for (int row = 0; row < 6; row++)
                fmla(VReg4S(10 + row * 4), VReg4S(6), VReg4S(row));

            ldr(QReg(7), ptr(pB, 48));
            for (int row = 0; row < 6; row++)
                fmla(VReg4S(11 + row * 4), VReg4S(7), VReg4S(row));

            add(pB, pB, 64);
            subs(reg_Kloop, reg_Kloop, 1);
            bne(loop_k1);
        }

        L(done_k);

        // Store 6×16 to C
        str(QReg(8),  ptr(pC));
        str(QReg(9),  ptr(pC, 16));
        str(QReg(10), ptr(pC, 32));
        str(QReg(11), ptr(pC, 48));

        add(pC, pC, reg_ldc);
        str(QReg(12), ptr(pC));
        str(QReg(13), ptr(pC, 16));
        str(QReg(14), ptr(pC, 32));
        str(QReg(15), ptr(pC, 48));

        add(pC, pC, reg_ldc);
        str(QReg(16), ptr(pC));
        str(QReg(17), ptr(pC, 16));
        str(QReg(18), ptr(pC, 32));
        str(QReg(19), ptr(pC, 48));

        add(pC, pC, reg_ldc);
        str(QReg(20), ptr(pC));
        str(QReg(21), ptr(pC, 16));
        str(QReg(22), ptr(pC, 32));
        str(QReg(23), ptr(pC, 48));

        add(pC, pC, reg_ldc);
        str(QReg(24), ptr(pC));
        str(QReg(25), ptr(pC, 16));
        str(QReg(26), ptr(pC, 32));
        str(QReg(27), ptr(pC, 48));

        add(pC, pC, reg_ldc);
        str(QReg(28), ptr(pC));
        str(QReg(29), ptr(pC, 16));
        str(QReg(30), ptr(pC, 32));
        str(QReg(31), ptr(pC, 48));

        // Restore callee-saved
        ldp(d12, d13, ptr(sp, 88));
        ldp(d10, d11, ptr(sp, 72));
        ldp(d8, d9, ptr(sp, 56));
        ldr(x25, ptr(sp, 48));
        ldp(x23, x24, ptr(sp, 32));
        ldp(x21, x22, ptr(sp, 16));
        ldp(x19, x20, post_ptr(sp, 112));
        ret();
    }
};

// Singleton accessors
// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=[GEMM,JIT] tiling=[MR,NR]
inline jit_gemm_func_t get_jit_gemm_6x16_beta0() {
    static JitGemmMicroKernel6x16 kernel(false);
    return kernel.getFunc();
}

// @nnr-meta isa=NEON dtype=fp32 layout=NCHW special=[GEMM,JIT] tiling=[MR,NR]
inline jit_gemm_func_t get_jit_gemm_6x16_beta1() {
    static JitGemmMicroKernel6x16 kernel(true);
    return kernel.getFunc();
}

} // namespace neon_jit
} // namespace nnr

#endif // NNR_USE_XBYAK_AARCH64
