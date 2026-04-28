#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/simd_math_avx512.h"
#include "backend/x64/simd_math_avx2.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/simd_math_neon.h"
#endif

namespace nnr {

namespace {

struct Sigmoid_operator : public operator_t {

    static void apply_inplace(float* data, int rows, int cols, int stride,
                              const operator_t* self, const float* bias, int offset) {
        const bool fused_silu = self->is_fused_silu;
#if defined(NNR_ARCH_X64) || defined(NNR_ARCH_ARM64)
        if (!bias && stride == cols) {
            size_t n = (size_t)rows * cols;
            // post_fn is invoked from worker threads (gemm post-ops, depthwise
            // strip, etc.); call the single-threaded `_kernel` variants to
            // avoid nested for_static deadlock.
#ifdef NNR_ARCH_X64
            if (has_avx512()) {
                if (fused_silu) silu_avx512_kernel(data, data, n);
                else            sigmoid_avx512_kernel(data, n);
            } else {
                if (fused_silu) silu_avx2_kernel(data, data, n);
                else            sigmoid_avx2_kernel(data, n);
            }
#else
            if (fused_silu) silu_neon_kernel(data, data, n);
            else            sigmoid_neon_kernel(data, n);
#endif
            if (self->post_fn)
                self->post_fn(data, rows, cols, stride, self->fused_op, nullptr, offset);
            return;
        }
#endif
        for (int r = 0; r < rows; r++) {
            float* row = data + (size_t)r * stride;
            float bv = bias ? bias[r] : 0.0f;
            if (fused_silu) {
                for (int i = 0; i < cols; ++i) {
                    float v = row[i] + bv;
                    row[i] = v / (1.0f + expf(-v));
                }
            } else {
                for (int i = 0; i < cols; ++i)
                    row[i] = 1.0f / (1.0f + expf(-(row[i] + bv)));
            }
        }
        if (self->post_fn)
            self->post_fn(data, rows, cols, stride, self->fused_op, nullptr, offset);
    }

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        fusable_apply = &apply_inplace;
        layout_mask = LAYOUT_ALL;
        return true;
    }

    scroll_info_t scroll_info() const override {
        if (inputs[0]->ndim < 3) return {};
        return { .scrollable = true };
    }

    bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) override
    {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        // Element-wise strip path addresses px/py with different per-channel
        // strides (iH*W vs oH*W). When the chain pairs a full-buffer input
        // with a ring-buffer output (yolov10n's stem-Conv → SiLU), x and y
        // also disagree on channel-block layout (NCHW vs BLOCKED_16) and the
        // per-channel offsets walk into the wrong heap region. Bail so the
        // segment is pruned and the op runs via the non-scroll exec() path.
        if (x->ndim >= 2 && y->ndim >= 2
            && x->dims[x->ndim - 2] != y->dims[y->ndim - 2]) {
            return false;
        }
#if defined(NNR_ARCH_X64) || defined(NNR_ARCH_ARM64)
        // SIMD fast path for scroll strip: process the strip directly with
        // vectorized sigmoid/silu, parallelized across outer NC dims.
        {
            int NC = 1;
            for (int d = 0; d < x->ndim - 2; d++) NC *= x->dims[d];
            int iH = x->dims[x->ndim - 2], W = x->dims[x->ndim - 1];
            int oH = y->dims[y->ndim - 2];
            int clamp_H = ring_out.orig_H > 0 ? ring_out.orig_H : oH;
            int out_end = std::min(out_row_start + out_rows, clamp_H);
            int count = (out_end - out_row_start) * W;
            if (count <= 0) return true;
            const float* px = (const float*)x->data;
            float* py = (float*)y->data;
            const bool fused_silu_local = is_fused_silu;
            nnr::for_static(0, NC, NC > 1, [&](int nc) {
                const float* src = px + (size_t)nc * iH * W + (size_t)out_row_start * W;
                float* dst = py + (size_t)nc * oH * W + (size_t)out_row_start * W;
                if (src != dst) memcpy(dst, src, count * sizeof(float));
#ifdef NNR_ARCH_X64
                if (has_avx512()) {
                    if (fused_silu_local) silu_avx512_kernel(dst, dst, count);
                    else                  sigmoid_avx512_kernel(dst, count);
                } else {
                    if (fused_silu_local) silu_avx2_kernel(dst, dst, count);
                    else                  sigmoid_avx2_kernel(dst, count);
                }
#else // NNR_ARCH_ARM64
                if (fused_silu_local) silu_neon_kernel(dst, dst, count);
                else                  sigmoid_neon_kernel(dst, count);
#endif
            });
            return true;
        }
#endif
        if (is_fused_silu) {
            return exec_strip_elementwise((const float*)x->data, (float*)y->data,
                x->ndata, x->dims, x->ndim, out_row_start, out_rows,
                [](float v) { float s = 1.0f / (1.0f + expf(-v)); return v * s; },
                ring_out.orig_H, y->dims, y->ndata);
        }
        return exec_strip_elementwise((const float*)x->data, (float*)y->data,
            x->ndata, x->dims, x->ndim, out_row_start, out_rows,
            [](float v) { return 1.0f / (1.0f + expf(-v)); }, ring_out.orig_H,
            y->dims, y->ndata);
    }

    template <typename T>
    bool exec() {
        if constexpr (std::is_same_v<T, float>) {
#ifdef NNR_ARCH_X64
            tensor_t* y = outputs[0];
            const tensor_t* x = inputs[0];
            if (is_fused_silu) {
                if (has_avx512()) silu_avx512((const float*)x->data, (float*)y->data, x->ndata);
                else              silu_avx2  ((const float*)x->data, (float*)y->data, x->ndata);
            } else {
                if (x->data != y->data)
                    memcpy(y->data, x->data, x->ndata * sizeof(float));
                if (has_avx512()) sigmoid_avx512((float*)y->data, y->ndata);
                else              sigmoid_avx2  ((float*)y->data, y->ndata);
            }
            return true;
#elifdef NNR_ARCH_ARM64
            tensor_t* y = outputs[0];
            const tensor_t* x = inputs[0];
            if (is_fused_silu) {
                silu_neon((const float*)x->data, (float*)y->data, x->ndata);
            } else {
                if (x->data != y->data)
                    memcpy(y->data, x->data, x->ndata * sizeof(float));
                sigmoid_neon((float*)y->data, y->ndata);
            }
            return true;
#endif
        }
        if (is_fused_silu) {
            foreach_tensor<T>([](auto x) {
                auto s = (T)1.0 / ((T)1.0 + (T)exp(-(double)x));
                return x * s;
            });
        } else {
            foreach_tensor<T>([](auto x) -> T {
                if (x >= 0) {
                    return (T)((T)1.0 / ((T)1.0 + (T)exp(-1 * x)));
                } else {
                    return (T)(exp(x) / ((T)1.0 + (T)exp(x)));
                }
            });
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Sigmoid_operator,
            opset_t<13, bfloat16_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8] scroll=yes fusion=post_op
operator_t* resolver_default_op_Sigmoid(int opset, pool_t& pool) { return pool_new<Sigmoid_operator>(pool); }

} // namespace nnr
