#pragma once
// Unified GEMM kernel shared by Conv, MatMul, and Gemm operators.
// Computes C[n×m] = A[n×o] × B[o×m] with optional post-processing.

#include "nnr.h"
#include "thread_pool.h"
#include "backend/cpu/kernel/post_ops.h"
#include "cpu_features.h"
#include "profiler.h"

#ifdef NNR_ARCH_X64
#include "backend/x64/vec_ops_avx512.h"
#include "backend/x64/gemm_avx512.h"
#include "backend/x64/vec_ops_avx2.h"
#include "backend/x64/gemm_avx2.h"
#include "backend/x64/conv_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/vec_ops_neon.h"
#include "backend/arm/gemm_neon.h"
#include "backend/arm/conv_neon.h"
#include "backend/arm/gemm_fp16_neon.h"
#endif

namespace nnr {

// fused_zero_bias is declared in gemm_avx512.h (x64) or gemm_neon.h (ARM).

// Post-processing applied per-row after GEMM tile computation.
// Adds per-row bias, then calls the fused post-op (if any) on L1-hot data.
// apply_rows() processes a block of rows in one call, hoisting null checks
// and ISA dispatch outside the row loop.
struct gemm_post_t {
    static constexpr bool per_row_bias = true;  // NCHW: bias[row], enables register fusion
    const float* bias = nullptr;
    int bias_off = 0;
    const float* c_base = nullptr;  // GEMM C output base pointer
    int c_base_offset = 0;          // offset of c_base within full output tensor
    operator_t::post_fn_t post_fn = nullptr;
    const operator_t* fused_op = nullptr;
    post_op_kind kind = post_op_kind::none;
    float clip_min = 0.0f;
    float clip_max = FLT_MAX;

    gemm_post_t() = default;
    gemm_post_t(const float* b, int off, const float* c, int c_off, operator_t* op)
        : bias(b), bias_off(off), c_base(c), c_base_offset(c_off)
        , post_fn(op->post_fn), fused_op(op->fused_op) { classify(); }

    void classify() {
        if (post_fn == relu_post_fn) {
            kind = post_op_kind::relu;
            clip_min = 0.0f;
            clip_max = FLT_MAX;
        } else if (post_fn == clip_post_fn) {
            kind = post_op_kind::clip;
            auto* p = reinterpret_cast<const clip_post_params_t*>(fused_op);
            clip_min = p->min_val;
            clip_max = p->max_val;
        } else if (!post_fn && bias) {
            kind = post_op_kind::bias_only;
            clip_min = -FLT_MAX;
            clip_max = FLT_MAX;
        } else {
            kind = post_op_kind::none;
        }
    }

    // Single-row apply (backward-compatible for call sites that process one row).
    void apply(int row, float* __restrict data, int len, int /*j0*/ = 0) const {
        if (!bias && !post_fn) return;
        float bv = bias ? bias[bias_off + row] : 0.0f;
        if (post_fn) {
            int offset = c_base_offset + (int)(data - c_base);
            post_fn(data, 1, len, len, fused_op, bias ? &bias[bias_off + row] : nullptr, offset);
        } else if (bv != 0.0f) {
#ifdef NNR_ARCH_X64
            if (has_avx512()) {
                avx512::bias_add(data, len, bv);
            } else if (detect_isa() == isa_t::avx2) {
                avx2::bias_add(data, len, bv);
            } else
#elifdef NNR_ARCH_ARM64
            if (has_neon()) {
                neon::bias_add(data, len, bv);
            } else
#endif
            {
                for (int i = 0; i < len; i++)
                    data[i] += bv;
            }
        }
    }

    // Batch apply: process rows [i0, ie) of C[n×m] starting at column j0, width jw.
    // Hoists null/type checks and ISA dispatch outside the row loop.
    void apply_rows(int i0, int ie, float* C, int m, int j0, int jw) const {
        if (!bias && !post_fn) return;
        if (post_fn) {
            float* data = C + (size_t)i0 * m + j0;
            int offset = c_base_offset + (int)(data - c_base);
            post_fn(data, ie - i0, jw, m, fused_op,
                    bias ? bias + bias_off + i0 : nullptr, offset);
        } else {
            // Bias-only: ISA dispatch once, loop rows inside
            auto do_bias = [&](auto bias_fn) {
                for (int i = i0; i < ie; i++) {
                    float bv = bias[bias_off + i];
                    if (bv != 0.0f)
                        bias_fn(C + (size_t)i * m + j0, jw, bv);
                }
            };
#ifdef NNR_ARCH_X64
            if (has_avx512())
                do_bias([](float* d, int n, float b) { avx512::bias_add(d, n, b); });
            else if (detect_isa() == isa_t::avx2)
                do_bias([](float* d, int n, float b) { avx2::bias_add(d, n, b); });
            else
#elifdef NNR_ARCH_ARM64
            if (has_neon())
                do_bias([](float* d, int n, float b) { neon::bias_add(d, n, b); });
            else
#endif
                do_bias([](float* d, int n, float b) { for (int i = 0; i < n; i++) d[i] += b; });
        }
    }
};

// NHWC post-processing: column-wise bias (bias[j]) + fused post-op on L1-hot data.
// Used with dgemm_packed_b for NHWC Conv where output rows are spatial positions
// and columns are output channels — bias is per-column, not per-row.
struct gemm_post_nhwc_t {
    static constexpr bool per_row_bias = false;  // NHWC: bias[col], fused in dgemm_nhwc
    const float* bias = nullptr;
    const float* c_base = nullptr;
    int c_base_offset = 0;
    operator_t::post_fn_t post_fn = nullptr;
    const operator_t* fused_op = nullptr;
    post_op_kind kind = post_op_kind::none;
    float clip_min = 0.0f;
    float clip_max = FLT_MAX;

    gemm_post_nhwc_t() = default;

    void classify() {
        if (post_fn == relu_post_fn) {
            kind = post_op_kind::relu;
            clip_min = 0.0f;
            clip_max = FLT_MAX;
        } else if (post_fn == clip_post_fn) {
            kind = post_op_kind::clip;
            auto* p = reinterpret_cast<const clip_post_params_t*>(fused_op);
            clip_min = p->min_val;
            clip_max = p->max_val;
        } else if (!post_fn && bias) {
            kind = post_op_kind::bias_only;
            clip_min = -FLT_MAX;
            clip_max = FLT_MAX;
        } else {
            kind = post_op_kind::none;
        }
        if (kind != post_op_kind::none && !bias)
            bias = fused_zero_bias;
    }

    void apply(int /*row*/, float* __restrict data, int len, int j0 = 0) const {
        if (!bias && !post_fn) return;
        if (bias) {
            const float* b = bias + j0;
            int j = 0;
#ifdef NNR_ARCH_X64
            j = col_bias_add_x64(data, b, len);
#elifdef NNR_ARCH_ARM64
            j = col_bias_add_neon(data, b, len);
#endif
            for (; j < len; j++)
                data[j] += b[j];
        }
        if (post_fn) {
            int offset = c_base_offset + (int)(data - c_base);
            post_fn(data, 1, len, len, fused_op, nullptr, offset);
        }
    }

    // Batch apply: process rows [i0, ie) of C[n×m] starting at column j0, width jw.
    void apply_rows(int i0, int ie, float* C, int m, int j0, int jw) const {
        if (!bias && !post_fn) return;
        if (bias) {
            const float* b = bias + j0;
            for (int i = i0; i < ie; i++) {
                float* data = C + (size_t)i * m + j0;
                int j = 0;
#ifdef NNR_ARCH_X64
                j = col_bias_add_x64(data, b, jw);
#elifdef NNR_ARCH_ARM64
                j = col_bias_add_neon(data, b, jw);
#endif
                for (; j < jw; j++)
                    data[j] += b[j];
            }
        }
        if (post_fn) {
            float* data = C + (size_t)i0 * m + j0;
            int offset = c_base_offset + (int)(data - c_base);
            post_fn(data, ie - i0, jw, m, fused_op, nullptr, offset);
        }
    }
};

// Unified GEMM kernel: C[n×m] = A[n×o] × B[o×m]
// A is row-major [n×o], B is row-major [o×m], C is row-major [n×m].
// post_fn.apply(row, data, len) is called per-row on L1-hot tile data.
template <typename T, typename PostFn = gemm_post_t>
inline void dgemm_generic(int n, int m, int o, const T* __restrict A, const T* __restrict B, T* __restrict C, const PostFn& post_fn = PostFn{})
{
    NNR_PROFILE_SCOPE("dgemm_generic");
    if constexpr (std::is_same_v<T, float>) {
#ifdef NNR_ARCH_X64
        if (has_avx512()) {
            avx512::dgemm(n, m, o, A, B, C, post_fn);
            return;
        }
        if (detect_isa() == isa_t::avx2) {
            avx2::dgemm(n, m, o, A, B, C, post_fn);
            return;
        }
#elifdef NNR_ARCH_ARM64
        if (has_neon()) {
            neon::dgemm(n, m, o, A, B, C, post_fn);
            return;
        }
#endif
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            C[i * m + j] = T(0);
    // Tiled GEMM: BLK×BLK blocks sized so A + B co-reside in L1d
    // (two matrices × BLK² × sizeof(T) ≤ L1d_bytes). For 32 KB L1d → BLK = 64
    // (A+B = 32 KB exactly); larger L1d rounds up to the next multiple of 16.
    // Loop order i-k-j: A[i,k] is reused across j, B[k,j] streams sequentially,
    // C[i,j] accumulates in-place. Post-op runs per i-block after all K tiles
    // are accumulated, so it sees the final result while C is still L1-hot.
    int BLK;
    {
        const size_t l1d_bytes = (size_t)cpu_features().l1d_kb * 1024;
        int s = 1;
        while ((size_t)(s + 16) * (s + 16) * 4 * 2 <= l1d_bytes) s += 16;
        BLK = s < 16 ? 16 : s;
    }
    for (int i0 = 0; i0 < n; i0 += BLK) {
        int ie = std::min(i0 + BLK, n);
        for (int k0 = 0; k0 < o; k0 += BLK)
            for (int j0 = 0; j0 < m; j0 += BLK) {
                int ke = std::min(k0 + BLK, o);
                int je = std::min(j0 + BLK, m);
                for (int i = i0; i < ie; ++i)
                    for (int k = k0; k < ke; ++k) {
                        T a = A[i * o + k];
                        for (int j = j0; j < je; ++j)
                            C[i * m + j] += a * B[k * m + j];
                    }
            }
        for (int i = i0; i < ie; ++i)
            post_fn.apply(i, (float*)(C + (size_t)i * m), m);
    }
}

// Weight pre-packing (A matrix): pre-arrange A panels to eliminate per-tile
// A-copy from the GEMM hot loop. Critical for NCHW Conv where A = weights.
inline size_t pack_a_size(int n, int o) {
#ifdef NNR_ARCH_X64
    if (has_avx512())
        return avx512::pack_a_size(n, o);
    if (detect_isa() == isa_t::avx2)
        return avx2::pack_a_size(n, o);
#elifdef NNR_ARCH_ARM64
    if (has_neon())
        return neon::pack_a_size(n, o);
#endif
    return 0;
}

inline void pack_a(float* dst, const float* A, int n, int o) {
    NNR_PROFILE_SCOPE("pack_a");
#ifdef NNR_ARCH_X64
    if (has_avx512())
        avx512::pack_a(dst, A, n, o);
    else if (detect_isa() == isa_t::avx2)
        avx2::pack_a(dst, A, n, o);
#elifdef NNR_ARCH_ARM64
    if (has_neon())
        neon::pack_a(dst, A, n, o);
#endif
}

// GEMM with pre-packed A panels. packed_A must be created by pack_a().
template <typename PostFn = gemm_post_t>
inline void dgemm_packed_a(int n, int m, int o, const float* packed_A, const float* B, float* C, const PostFn& post_fn = {}) {
    NNR_PROFILE_SCOPE("dgemm_packed_a");
#ifdef NNR_ARCH_X64
    if (has_avx512())
        avx512::dgemm_packed_a(n, m, o, packed_A, B, C, post_fn);
    else if (detect_isa() == isa_t::avx2)
        avx2::dgemm_packed_a(n, m, o, packed_A, B, C, post_fn);
#elifdef NNR_ARCH_ARM64
    if (has_neon())
        neon::dgemm_packed_a(n, m, o, packed_A, B, C, post_fn);
#endif
}

// Weight pre-packing (B matrix): pre-arrange B matrix panels to eliminate per-tile
// B-copy from the GEMM hot loop. Critical for NHWC Conv where B = weights.
inline size_t pack_b_size(int o, int m) {
#ifdef NNR_ARCH_X64
    if (has_avx512())
        return avx512::pack_b_size(o, m);
    if (detect_isa() == isa_t::avx2)
        return avx2::pack_b_size(o, m);
#elifdef NNR_ARCH_ARM64
    if (has_neon())
        return neon::pack_b_size(o, m);
#endif
    return 0;
}

inline void pack_b(float* dst, const float* B, int o, int m) {
    NNR_PROFILE_SCOPE("pack_b");
#ifdef NNR_ARCH_X64
    if (has_avx512())
        avx512::pack_b(dst, B, o, m);
    else if (detect_isa() == isa_t::avx2)
        avx2::pack_b(dst, B, o, m);
#elifdef NNR_ARCH_ARM64
    if (has_neon())
        neon::pack_b(dst, B, o, m);
#endif
}

// GEMM with pre-packed B panels. packed_B must be created by pack_b().
template <typename PostFn = gemm_post_t>
inline void dgemm_packed_b(int n, int m, int o, const float* A, const float* packed_B, float* C, const PostFn& post_fn = {}) {
    NNR_PROFILE_SCOPE("dgemm_packed_b");
#ifdef NNR_ARCH_X64
    if (has_avx512())
        avx512::dgemm_packed_b(n, m, o, A, packed_B, C, post_fn);
    else if (detect_isa() == isa_t::avx2)
        avx2::dgemm_packed_b(n, m, o, A, packed_B, C, post_fn);
#elifdef NNR_ARCH_ARM64
    if (has_neon())
        neon::dgemm_packed_b(n, m, o, A, packed_B, C, post_fn);
#endif
}

// FP16 GEMM with pre-packed B. FP16 inputs (uint16_t bit pattern), FP32 output.
// Caller is responsible for any FP32→FP16 conversion of C. Returns false when
// no FP16 hardware path is available; the caller should then fall back to the
// convert-to-FP32 path.
inline size_t pack_b_fp16_size(int o, int m) {
#ifdef NNR_ARCH_ARM64
    if (has_neon_fp16())
        return nnr::fp16::neon::pack_b_fp16_neon_size(o, m);
#else
    (void)o; (void)m;
#endif
    return 0;
}

inline void pack_b_fp16(uint16_t* dst, const uint16_t* B, int o, int m) {
    NNR_PROFILE_SCOPE("pack_b_fp16");
#ifdef NNR_ARCH_ARM64
    if (has_neon_fp16())
        nnr::fp16::neon::pack_b_fp16_neon(dst, B, o, m);
#else
    (void)dst; (void)B; (void)o; (void)m;
#endif
}

inline bool dgemm_fp16(int n, int m, int o,
                      const uint16_t* A, const uint16_t* packed_B, float* C) {
    NNR_PROFILE_SCOPE("dgemm_fp16");
#ifdef NNR_ARCH_ARM64
    if (has_neon_fp16())
        return nnr::fp16::neon::gemm_fp16_neon(n, m, o, A, packed_B, C);
#else
    (void)n; (void)m; (void)o; (void)A; (void)packed_B; (void)C;
#endif
    return false;
}

// NHWC-native GEMM with pre-packed B. packed_B must be created by pack_b().
// Optimized loop ordering for NHWC Conv: tiles spatial only, B stays L1-hot.
template <typename PostFn = gemm_post_t>
inline void dgemm_nhwc(int n, int m, int o, const float* A, const float* packed_B, float* C, const PostFn& post_fn = {}) {
    NNR_PROFILE_SCOPE("dgemm_nhwc");
#ifdef NNR_ARCH_X64
    if (has_avx512())
        avx512::dgemm_nhwc(n, m, o, A, packed_B, C, post_fn);
    else if (detect_isa() == isa_t::avx2)
        avx2::dgemm_nhwc(n, m, o, A, packed_B, C, post_fn);
    else
        dgemm_packed_b(n, m, o, A, packed_B, C, post_fn); // fallback
#elifdef NNR_ARCH_ARM64
    if (has_neon())
        neon::dgemm_nhwc(n, m, o, A, packed_B, C, post_fn);
    else
        dgemm_packed_b(n, m, o, A, packed_B, C, post_fn); // fallback
#else
    dgemm_packed_b(n, m, o, A, packed_B, C, post_fn); // fallback
#endif
}

// Batched GEMM for Winograd: performs 36 independent GEMMs in a single dispatch.
template <typename PostFn = gemm_post_t>
inline void dgemm_packed_a_batch36(int n, int m, int o,
    const float* const packed_A_batch[36],
    const float* const B_batch[36],
    float* const C_batch[36],
    const PostFn& post_fn = {})
{
#ifdef NNR_ARCH_X64
    if (has_avx512())
        avx512::dgemm_packed_a_batch36(n, m, o, packed_A_batch, B_batch, C_batch, post_fn);
    else if (detect_isa() == isa_t::avx2)
        avx2::dgemm_packed_a_batch36(n, m, o, packed_A_batch, B_batch, C_batch, post_fn);
    else
#elifdef NNR_ARCH_ARM64
    if (has_neon())
        neon::dgemm_packed_a_batch36(n, m, o, packed_A_batch, B_batch, C_batch, post_fn);
    else
#endif
    {
        // Fallback: call individual GEMMs
        for (int p = 0; p < 36; ++p)
            dgemm_packed_a(n, m, o, packed_A_batch[p], B_batch[p], C_batch[p], post_fn);
    }
}

// Batched GEMM for NHWC Winograd: performs 36 independent GEMMs in a single dispatch.
template <typename PostFn = gemm_post_t>
inline void dgemm_packed_b_batch36(int n, int m, int o,
    const float* const A_batch[36],
    const float* const packed_B_batch[36],
    float* const C_batch[36],
    const PostFn& post_fn = {})
{
#ifdef NNR_ARCH_X64
    if (has_avx512())
        avx512::dgemm_packed_b_batch36(n, m, o, A_batch, packed_B_batch, C_batch, post_fn);
    else if (detect_isa() == isa_t::avx2)
        avx2::dgemm_packed_b_batch36(n, m, o, A_batch, packed_B_batch, C_batch, post_fn);
    else
#elifdef NNR_ARCH_ARM64
    if (has_neon())
        neon::dgemm_packed_b_batch36(n, m, o, A_batch, packed_B_batch, C_batch, post_fn);
    else
#endif
    {
        // Fallback: call individual GEMMs
        for (int p = 0; p < 36; ++p)
            dgemm_packed_b(n, m, o, A_batch[p], packed_B_batch[p], C_batch[p], post_fn);
    }
}

// Gemm with transB: C[m×n] = alpha * A[m×k] × B^T[n×k] + beta * bias[n]
// Common FC layer pattern. B stored as [n×k] (each row is an output neuron's weights).
inline void gemm_transB(float* output, const float* A, const float* B, const float* bias,
    int m, int n, int k, float alpha = 1.0f, float beta = 1.0f)
{
    // Init output with beta * bias (broadcast per row)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            output[i * n + j] = bias ? beta * bias[j] : 0.0f;

#ifdef NNR_ARCH_X64
    if (has_avx512()) {
        nnr::for_static(0, m, (int64_t)m * n > 64, [&](int i) {
            const float* a_row = A + (size_t)i * k;
            float* out_row = output + (size_t)i * n;
            for (int j = 0; j < n; ++j) {
                const float* b_row = B + (size_t)j * k;
                out_row[j] += alpha * avx512::dot_product(a_row, b_row, k);
            }
        });
    } else if (detect_isa() == isa_t::avx2) {
        nnr::for_static(0, m, (int64_t)m * n > 64, [&](int i) {
            const float* a_row = A + (size_t)i * k;
            float* out_row = output + (size_t)i * n;
            for (int j = 0; j < n; ++j) {
                const float* b_row = B + (size_t)j * k;
                out_row[j] += alpha * avx2::dot_product(a_row, b_row, k);
            }
        });
    } else
#elifdef NNR_ARCH_ARM64
    if (has_neon()) {
        nnr::for_static(0, m, (int64_t)m * n > 64, [&](int i) {
            const float* a_row = A + (size_t)i * k;
            float* out_row = output + (size_t)i * n;
            for (int j = 0; j < n; ++j) {
                const float* b_row = B + (size_t)j * k;
                out_row[j] += alpha * neon::dot_product(a_row, b_row, k);
            }
        });
    } else
#endif
    {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                const float* a_row = A + (size_t)i * k;
                const float* b_row = B + (size_t)j * k;
                for (int w = 0; w < k; ++w)
                    sum += a_row[w] * b_row[w];
                output[i * n + j] += alpha * sum;
            }
    }
}

} // namespace nnr
