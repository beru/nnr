// Consolidated simple binary arithmetic operators.
// Each uses binary_arith_op_t CRTP base: init=is_inout_size(2,1), reshape=multi_broadcast, exec=binary_broadcast_exec.

#include "nnr.h"
#include "tensor_view.h"
#include "util.h"
#include "thread_pool.h"
#include <limits>
#ifdef NNR_ARCH_X64
#include "backend/x64/simd_math_avx512.h"
#include "backend/x64/simd_math_avx2.h"
#include "backend/x64/ops_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/conv_neon.h"
#endif

#define NNR_BINARY_OP(Name, fn_body, ...) \
namespace { struct Name##_op : public binary_arith_op_t<Name##_op, __VA_ARGS__> { \
    static auto fn(auto a, auto b) { return fn_body; } \
}; } \
operator_t* resolver_default_op_##Name(int opset, pool_t& pool) { return pool_new<Name##_op>(pool); }

namespace nnr {

// Threaded same-shape float Add for large tensors. Out-of-line to avoid
// code layout changes in binary_math.cpp affecting other ops (e.g. Mul).
static NNR_NOINLINE void add_float_threaded(
    const float* pa, const float* pb, float* py, size_t l)
{
    // ORT-style cost-based threading: float Add = 8B in (2×4B), 4B out, 1 compute
    int nt = nnr::elementwise_threads(l, 8, 4, 1);
    constexpr size_t BLOCK = 4096;
    int nblocks = (int)((l + BLOCK - 1) / BLOCK);
    nnr::for_dynamic(0, nblocks, nt, [&](int /*tid*/, int blk) {
        size_t start = (size_t)blk * BLOCK;
        size_t end = std::min(start + BLOCK, l);
#ifdef NNR_ARCH_X64
        if (has_avx512()) add_vec_avx512(py, pa, pb, start, end);
        else              add_vec_avx2(py, pa, pb, start, end);
#elifdef NNR_ARCH_ARM64
        add_vec_neon(py, pa, pb, start, end);
#else
        for (size_t i = start; i < end; ++i)
            py[i] = pa[i] + pb[i];
#endif
    });
}

// Threaded fused max(0, a+b). Same blocking/cost as add_float_threaded but
// applies Relu in-register before the store. One pass over the output.
static NNR_NOINLINE void add_relu_float_threaded(
    const float* pa, const float* pb, float* py, size_t l)
{
    int nt = nnr::elementwise_threads(l, 8, 4, 1);
    constexpr size_t BLOCK = 4096;
    int nblocks = (int)((l + BLOCK - 1) / BLOCK);
    nnr::for_dynamic(0, nblocks, nt, [&](int /*tid*/, int blk) {
        size_t start = (size_t)blk * BLOCK;
        size_t end = std::min(start + BLOCK, l);
#ifdef NNR_ARCH_X64
        if (has_avx512()) add_relu_vec_avx512(py, pa, pb, start, end);
        else              add_relu_vec_avx2(py, pa, pb, start, end);
#else
        for (size_t i = start; i < end; ++i) {
            float v = pa[i] + pb[i];
            py[i] = v > 0.0f ? v : 0.0f;
        }
#endif
    });
}

// Add is hand-written (not macro) to support binary fusion (Conv+Add skip connection).
namespace {
struct Add_op : public binary_arith_op_t<Add_op,
    opset_t<14, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t>,
    opset_t<13, uint32_t, uint64_t, int32_t, int64_t, float16_t, float, double, bfloat16_t>,
    opset_t<7, uint32_t, uint64_t, int32_t, int64_t, float16_t, float, double>>
{
    using binary_arith_op_t::exec; // unhide template exec<T>()
    static auto fn(auto a, auto b) { return a + b; }

    // Fused binary Add: data[i] += skip[offset+i] + bias
    // Then chain to next post-op if present (e.g., Add→Relu)
    static void apply_inplace(float* data, int rows, int cols, int stride,
                              const operator_t* self, const float* bias, int offset) {
        const float* skip = (const float*)self->fused_tensor->data;
        for (int r = 0; r < rows; r++) {
            float* row = data + (size_t)r * stride;
            float bv = bias ? bias[r] : 0.0f;
            int row_off = offset + r * stride;
#ifdef NNR_ARCH_ARM64
            add_skip_bias_neon(row, skip, cols, row_off, bv);
#else
            for (int i = 0; i < cols; ++i)
                row[i] += skip[row_off + i] + bv;
#endif
        }
        if (self->post_fn)
            self->post_fn(data, rows, cols, stride, self->fused_op, nullptr, offset);
    }

    bool init() override {
        if (!is_inout_size(2, 1)) return false;
        fusable_apply = &apply_inplace;
        layout_mask = LAYOUT_ALL;
        return true;
    }

    bool supports_strided_output(memory_layout_t format) const override {
        if (format != memory_layout_t::NHWC) return false;
        if (inputs.size() < 2 || outputs.empty() || !outputs[0]) return false;
        auto* a = inputs[0]; auto* b = inputs[1]; auto* y = outputs[0];
        if (!a || !b) return false;
        if (y->ndim != 4 || y->type != NNR_DATA_TYPE_FLOAT32) return false;
        // Only same-shape (no broadcast).
        if (a->ndata != y->ndata || b->ndata != y->ndata) return false;
        return true;
    }

    scroll_info_t scroll_info() const override {
        if (inputs.size() != 2 || outputs.size() != 1) return {};
        auto* a = inputs[0]; auto* b = inputs[1];
        if (a->ndim != 4 || b->ndim != 4) return {};
        if (a->type != NNR_DATA_TYPE_FLOAT32) return {};
        // Same shape on N, C, W (skip H which may differ under ring buffer)
        if (a->dims[0] != b->dims[0] || a->dims[1] != b->dims[1] ||
            a->dims[3] != b->dims[3]) return {};
        // exec_strip uses y->declared_layout for the per-channel-block
        // stride. Mismatched declared_layouts across a/b/y would mis-stride
        // the addressing because the ring buffer was sized using one of
        // them while exec_strip uses another.
        auto* y = outputs[0];
        if (a->declared_layout != y->declared_layout) return {};
        if (b->declared_layout != y->declared_layout) return {};
        return { .scrollable = true };
    }

    bool exec() override {
        auto* a = inputs[0]; auto* b = inputs[1]; auto* y = outputs[0];
        // Strided NHWC (Concat alias): output's spatial-major stride is
        // ldc = parent C, not local C. Same-shape, same-format inputs only.
        if (y->strides_set && y->format == memory_layout_t::NHWC && y->ndim == 4
            && a->ndata == y->ndata && b->ndata == y->ndata
            && a->format == memory_layout_t::NHWC && b->format == memory_layout_t::NHWC
            && a->type == NNR_DATA_TYPE_FLOAT32) {
            const int N = y->dims[0], C = y->dims[1], H = y->dims[2], W = y->dims[3];
            const int NHW = N * H * W;
            const int ldc = make_addr(y).elem_stride<float>(3);
            const float* pa = (const float*)a->data;
            const float* pb = (const float*)b->data;
            float* py = (float*)y->data;
            nnr::for_static(0, NHW, NHW > 16, [&](int s) {
                const float* sa = pa + (size_t)s * C;
                const float* sb = pb + (size_t)s * C;
                float* sy = py + (size_t)s * ldc;
                for (int c = 0; c < C; ++c) sy[c] = sa[c] + sb[c];
            });
            y->format = memory_layout_t::NHWC;
            if (post_fn)
                post_fn((float*)y->data, NHW, C, ldc, fused_op, nullptr, 0);
            return true;
        }
        // Same element count = effectively same-shape (broadcast dims are all 1)
        if (a->type == NNR_DATA_TYPE_FLOAT32
            && a->ndata == b->ndata && a->ndata == y->ndata && a->ndata > 1024) {
            // Single-pass Add+Relu fast path: when the fused unary post-op is
            // a leaf Relu (no further chain), compute max(0, a+b) in registers
            // before the store. Avoids the second full pass over y that a
            // separate post_fn would require.
            const bool fuse_relu = post_fn != nullptr
                && fused_op != nullptr
                && fused_op->op_type == "Relu"
                && fused_op->post_fn == nullptr;
#ifdef NNR_ARCH_X64
            // For small tensors (<4M elements / 16MB), single-threaded AVX-512
            // is faster than thread dispatch overhead (~50µs wake+sync).
            if (a->ndata < 4 * 1024 * 1024) {
                if (fuse_relu) {
                    if (has_avx512())
                        add_relu_vec_avx512((float*)y->data, (const float*)a->data,
                                            (const float*)b->data, 0, y->ndata);
                    else
                        add_relu_vec_avx2((float*)y->data, (const float*)a->data,
                                          (const float*)b->data, 0, y->ndata);
                } else {
                    if (has_avx512())
                        add_vec_avx512((float*)y->data, (const float*)a->data,
                                       (const float*)b->data, 0, y->ndata);
                    else
                        add_vec_avx2((float*)y->data, (const float*)a->data,
                                     (const float*)b->data, 0, y->ndata);
                }
            } else if (fuse_relu) {
                add_relu_float_threaded((const float*)a->data, (const float*)b->data,
                                        (float*)y->data, y->ndata);
            } else {
                add_float_threaded((const float*)a->data, (const float*)b->data,
                                   (float*)y->data, y->ndata);
            }
#else
            if (fuse_relu) {
                add_relu_float_threaded((const float*)a->data, (const float*)b->data,
                                        (float*)y->data, y->ndata);
            } else {
                add_float_threaded((const float*)a->data, (const float*)b->data,
                                   (float*)y->data, y->ndata);
            }
#endif
            y->format = a->format;
            if (post_fn && !fuse_relu)
                post_fn((float*)y->data, 1, (int)y->ndata, (int)y->ndata,
                        fused_op, nullptr, 0);
            return true;
        }
        // Fast path: bias broadcast [N] + [*,N] or [*,N] + [N]
        // (common in transformers: MatMul output + bias vector)
        if (a->type == NNR_DATA_TYPE_FLOAT32 && y->ndata > 1024) {
            const tensor_t* full = nullptr;
            const tensor_t* bias = nullptr;
            if (a->ndata == y->ndata && b->ndim == 1 && b->dims[0] == y->dims[y->ndim - 1])
                { full = a; bias = b; }
            else if (b->ndata == y->ndata && a->ndim == 1 && a->dims[0] == y->dims[y->ndim - 1])
                { full = b; bias = a; }
            if (full && bias) {
                const float* pf = (const float*)full->data;
                const float* pb = (const float*)bias->data;
                float* py = (float*)y->data;
                int N = bias->dims[0];
                int rows = (int)(y->ndata / N);
#ifdef NNR_ARCH_X64
                if (y->ndata < 4 * 1024 * 1024) {
                    if (has_avx512()) add_bias_broadcast_avx512(py, pf, pb, rows, N);
                    else              add_bias_broadcast_avx2(py, pf, pb, rows, N);
                } else {
                    nnr::for_static(0, rows, true, [&](int r) {
                        if (has_avx512())
                            add_bias_broadcast_avx512(py + (size_t)r * N,
                                pf + (size_t)r * N, pb, 1, N);
                        else
                            add_bias_broadcast_avx2(py + (size_t)r * N,
                                pf + (size_t)r * N, pb, 1, N);
                    });
                }
#else
                for (int r = 0; r < rows; r++) {
                    const float* src = pf + (size_t)r * N;
                    float* dst = py + (size_t)r * N;
                    for (int i = 0; i < N; i++) dst[i] = src[i] + pb[i];
                }
#endif
                y->format = full->format;
                if (post_fn)
                    post_fn(py, 1, (int)y->ndata, (int)y->ndata,
                            fused_op, nullptr, 0);
                return true;
            }
        }
        bool ok = binary_arith_op_t::exec();
        if (ok && post_fn && y->type == NNR_DATA_TYPE_FLOAT32 && y->data) {
            post_fn((float*)y->data, 1, (int)y->ndata, (int)y->ndata,
                    fused_op, nullptr, 0);
        }
        return ok;
    }

    bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) override
    {
        // inputs[0] = chain input (may be ring-buffered via virtual pointer)
        // inputs[1] = skip input  (always full tensor, absolute offsets)
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        tensor_t* y = outputs[0];

        // Both inputs must have the same physical layout for element-wise Add.
        // Mixed NCHW + BLOCKED_16 would silently produce wrong results.
        if (a->format != b->format) return false;

        int W = y->dims[3];
        int aH = a->dims[2];  // ring_H if ring-buffered
        int bH = b->dims[2];  // always full H
        int oH = y->dims[2];
        int clamp_H = ring_in.orig_H > 0 ? ring_in.orig_H : bH;
        int out_end = std::min(out_row_start + out_rows, clamp_H);
        int rows = out_end - out_row_start;
        if (rows <= 0) return true;

        // BLOCKED_16/8: row width is W*block per channel block.
        // NCHW: row width is W per channel.
        // Check declared_layout, not format — at first-run setup time the
        // ring buffer is sized using declared_layout (see scroll_chains.cpp),
        // and y->format may still be the default NCHW even though the
        // upstream Conv will write BLOCKED bytes. Mismatched block size here
        // mis-strides the addressing and overflows the ring buffer.
        bool blocked = (y->declared_layout == NATIVE_BLOCKED_FMT);
        int block = blocked ? NATIVE_BLOCK : 1;
        int NC = blocked
            ? y->dims[0] * (y->dims[1] / block)
            : y->dims[0] * y->dims[1];
        int row_elems = W * block;
        int count = rows * row_elems;

        const float* pa = (const float*)a->data;
        const float* pb = (const float*)b->data;
        float* py = (float*)y->data;
        post_fn_t pf = post_fn;
        const operator_t* fop = fused_op;
        nnr::for_static(0, NC, NC > 4, [&](int nc) {
            const float* sa = pa + (size_t)nc * aH * row_elems + (size_t)out_row_start * row_elems;
            const float* sb = pb + (size_t)nc * bH * row_elems + (size_t)out_row_start * row_elems;
            float* dst = py + (size_t)nc * oH * row_elems + (size_t)out_row_start * row_elems;
#ifdef NNR_ARCH_ARM64
            int i = 0;
            for (; i + 4 <= count; i += 4)
                vst1q_f32(dst + i, vaddq_f32(vld1q_f32(sa + i), vld1q_f32(sb + i)));
            for (; i < count; ++i)
                dst[i] = sa[i] + sb[i];
#else
            for (int i = 0; i < count; ++i)
                dst[i] = sa[i] + sb[i];
#endif
            // Fused unary post-op (e.g. Relu) over this channel-block's strip
            // rows. Hot data is in L1 from the just-completed adds; applying
            // here avoids a second pass over the whole output tensor.
            if (pf) pf(dst, 1, count, count, fop, nullptr, 0);
        });
        return true;
    }
};
}
// @nnr-meta-op op=Add mt=dynamic layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8] scroll=yes fusion=binary
operator_t* resolver_default_op_Add(int opset, pool_t& pool) { return pool_new<Add_op>(pool); }
namespace {
struct Sub_op : public binary_arith_op_t<Sub_op,
    opset_t<14, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<13, int32_t, int64_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<7, int32_t, int64_t, uint32_t, uint64_t, float16_t, float, double>>
{
    using binary_arith_op_t::exec;
    static auto fn(auto a, auto b) { return a - b; }

    bool exec() override {
        auto* a = inputs[0]; auto* b = inputs[1]; auto* y = outputs[0];
#ifdef NNR_ARCH_X64
        if (a->type == NNR_DATA_TYPE_FLOAT32
            && a->ndata == b->ndata && a->ndata == y->ndata && a->ndata > 1024) {
            if (has_avx512())
                sub_avx512((const float*)a->data, (const float*)b->data,
                           (float*)y->data, y->ndata);
            else
                sub_avx2((const float*)a->data, (const float*)b->data,
                         (float*)y->data, y->ndata);
            y->format = a->format;
            return true;
        }
        if (a->type == NNR_DATA_TYPE_FLOAT32 && y->type == NNR_DATA_TYPE_FLOAT32
            && a->type == b->type && y->ndata > 1024 && y->ndim > 0
            && y->ndim <= MAX_NDIM) {
            int a_bstr[MAX_NDIM], b_bstr[MAX_NDIM];
            compute_broadcast_strides(a, y, a_bstr);
            compute_broadcast_strides(b, y, b_bstr);
            if (has_avx512())
                sub_broadcast_avx512((const float*)a->data, (const float*)b->data,
                    (float*)y->data, y->dims, a_bstr, b_bstr, y->ndim);
            else
                sub_broadcast_avx2((const float*)a->data, (const float*)b->data,
                    (float*)y->data, y->dims, a_bstr, b_bstr, y->ndim);
            return true;
        }
#endif
        return binary_arith_op_t::exec();
    }
};
}
// @nnr-meta-op op=Sub mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_Sub(int opset, pool_t& pool) { return pool_new<Sub_op>(pool); }
// Mul is hand-written to add AVX-512 fast path for same-shape float multiply.
namespace {
struct Mul_op : public binary_arith_op_t<Mul_op,
    opset_t<14, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<13, int32_t, int64_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<7, int32_t, int64_t, uint32_t, uint64_t, float16_t, float, double>>
{
    using binary_arith_op_t::exec; // unhide template exec<T>()
    static auto fn(auto a, auto b) { return a * b; }

    bool exec() override {
        auto* a = inputs[0]; auto* b = inputs[1]; auto* y = outputs[0];
#ifdef NNR_ARCH_X64
        if (a->type == NNR_DATA_TYPE_FLOAT32
            && a->ndata == b->ndata && a->ndata == y->ndata && a->ndata > 1024) {
            if (has_avx512())
                mul_avx512((const float*)a->data, (const float*)b->data,
                           (float*)y->data, y->ndata);
            else
                mul_avx2((const float*)a->data, (const float*)b->data,
                         (float*)y->data, y->ndata);
            y->format = a->format;
            return true;
        }
        // Broadcast SIMD path for fp32: covers PER_CHANNEL/GENERAL kinds whose
        // scalar fallback in binary_broadcast_exec dominated YOLO post-proc
        // (mul_18 in yolov3-12-int8 was 7.3 ms scalar, single-threaded).
        if (a->type == NNR_DATA_TYPE_FLOAT32 && y->type == NNR_DATA_TYPE_FLOAT32
            && a->type == b->type && y->ndata > 1024 && y->ndim > 0
            && y->ndim <= MAX_NDIM) {
            int a_bstr[MAX_NDIM], b_bstr[MAX_NDIM];
            compute_broadcast_strides(a, y, a_bstr);
            compute_broadcast_strides(b, y, b_bstr);
            if (has_avx512())
                mul_broadcast_avx512((const float*)a->data, (const float*)b->data,
                    (float*)y->data, y->dims, a_bstr, b_bstr, y->ndim);
            else
                mul_broadcast_avx2((const float*)a->data, (const float*)b->data,
                    (float*)y->data, y->dims, a_bstr, b_bstr, y->ndim);
            return true;
        }
#elifdef NNR_ARCH_ARM64
        if (a->type == NNR_DATA_TYPE_FLOAT32
            && a->ndata == b->ndata && a->ndata == y->ndata && a->ndata > 1024) {
            const float* pa = (const float*)a->data;
            const float* pb = (const float*)b->data;
            float* py = (float*)y->data;
            size_t n = y->ndata, i = 0;
            for (; i + 4 <= n; i += 4)
                vst1q_f32(py + i, vmulq_f32(vld1q_f32(pa + i), vld1q_f32(pb + i)));
            for (; i < n; ++i) py[i] = pa[i] * pb[i];
            y->format = a->format;
            return true;
        }
#endif
        return binary_arith_op_t::exec();
    }
};
}
// @nnr-meta-op op=Mul mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_Mul(int opset, pool_t& pool) { return pool_new<Mul_op>(pool); }
namespace {
struct Div_op : public binary_arith_op_t<Div_op,
    opset_t<14, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<13, int32_t, int64_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
    opset_t<7, int32_t, int64_t, uint32_t, uint64_t, float16_t, float, double>>
{
    using binary_arith_op_t::exec;
    // Float division yields inf/NaN and stays defined; integer division
    // by zero (and the signed INT_MIN / -1 case) is UB and on x86 raises
    // SIGFPE. ONNX does not fully specify integer divide-by-zero, but the
    // runtime must not crash — return 0 in those cases.
    static auto fn(auto a, auto b) {
        using A = decltype(a);
        using B = decltype(b);
        if constexpr (std::is_integral_v<A> && std::is_integral_v<B>) {
            if (b == 0) return A(0);
            if constexpr (std::is_signed_v<A> && std::is_signed_v<B>) {
                // INT_MIN / -1 is UB on signed two's-complement integers.
                if (b == B(-1) && a == std::numeric_limits<A>::min())
                    return std::numeric_limits<A>::min();
            }
            return A(a / b);
        } else {
            return a / b;
        }
    }

    bool exec() override {
        auto* a = inputs[0]; auto* b = inputs[1]; auto* y = outputs[0];
#ifdef NNR_ARCH_X64
        if (a->type == NNR_DATA_TYPE_FLOAT32
            && a->ndata == b->ndata && a->ndata == y->ndata && a->ndata > 1024) {
            if (has_avx512())
                div_avx512((const float*)a->data, (const float*)b->data,
                           (float*)y->data, y->ndata);
            else
                div_avx2((const float*)a->data, (const float*)b->data,
                         (float*)y->data, y->ndata);
            y->format = a->format;
            return true;
        }
        if (a->type == NNR_DATA_TYPE_FLOAT32 && y->type == NNR_DATA_TYPE_FLOAT32
            && a->type == b->type && y->ndata > 1024 && y->ndim > 0
            && y->ndim <= MAX_NDIM) {
            int a_bstr[MAX_NDIM], b_bstr[MAX_NDIM];
            compute_broadcast_strides(a, y, a_bstr);
            compute_broadcast_strides(b, y, b_bstr);
            if (has_avx512())
                div_broadcast_avx512((const float*)a->data, (const float*)b->data,
                    (float*)y->data, y->dims, a_bstr, b_bstr, y->ndim);
            else
                div_broadcast_avx2((const float*)a->data, (const float*)b->data,
                    (float*)y->data, y->dims, a_bstr, b_bstr, y->ndim);
            return true;
        }
#endif
        return binary_arith_op_t::exec();
    }
};
}
// @nnr-meta-op op=Div mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_Div(int opset, pool_t& pool) { return pool_new<Div_op>(pool); }

// Bitwise
// @nnr-meta-op op=BitwiseAnd mt=no
NNR_BINARY_OP(BitwiseAnd, a & b, opset_t<18, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>)
// @nnr-meta-op op=BitwiseOr mt=no
NNR_BINARY_OP(BitwiseOr,  a | b, opset_t<18, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>)
// @nnr-meta-op op=BitwiseXor mt=no
NNR_BINARY_OP(BitwiseXor, a ^ b, opset_t<18, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>)

#undef NNR_BINARY_OP

} // namespace nnr
