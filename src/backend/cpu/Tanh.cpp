#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/simd_math_avx512.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/simd_math_neon.h"
#endif

namespace nnr {

namespace {

struct Tanh_operator : public operator_t {

    static void apply_inplace(float* data, int rows, int cols, int stride,
                              const operator_t* self, const float* bias, int offset) {
#if defined(NNR_ARCH_X64) || defined(NNR_ARCH_ARM64)
        if (!bias && stride == cols) {
            size_t n = (size_t)rows * cols;
#ifdef NNR_ARCH_X64
            tanh_avx512_kernel(data, data, n);
#else
            tanh_neon_kernel(data, data, n);
#endif
            if (self->post_fn)
                self->post_fn(data, rows, cols, stride, self->fused_op, nullptr, offset);
            return;
        }
#endif
        for (int r = 0; r < rows; r++) {
            float* row = data + (size_t)r * stride;
            float bv = bias ? bias[r] : 0.0f;
            for (int i = 0; i < cols; ++i)
                row[i] = tanhf(row[i] + bv);
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
        if (x->ndim >= 2 && y->ndim >= 2
            && x->dims[x->ndim - 2] != y->dims[y->ndim - 2]) {
            return false;
        }
#if defined(NNR_ARCH_X64) || defined(NNR_ARCH_ARM64)
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
            nnr::for_static(0, NC, NC > 1, [&](int nc) {
                const float* src = px + (size_t)nc * iH * W + (size_t)out_row_start * W;
                float* dst = py + (size_t)nc * oH * W + (size_t)out_row_start * W;
#ifdef NNR_ARCH_X64
                tanh_avx512_kernel(src, dst, count);
#else
                tanh_neon_kernel(src, dst, count);
#endif
            });
            return true;
        }
#endif
        return exec_strip_elementwise((const float*)x->data, (float*)y->data,
            x->ndata, x->dims, x->ndim, out_row_start, out_rows,
            [](float v) { return tanhf(v); }, ring_out.orig_H,
            y->dims, y->ndata);
    }

    template <typename T>
    bool exec() {
        if constexpr (std::is_same_v<T, float>) {
            tensor_t* y = outputs[0];
            const tensor_t* x = inputs[0];
#ifdef NNR_ARCH_X64
            constexpr size_t CHUNK = 16384;
            size_t n = x->ndata;
            int nchunks = (int)((n + CHUNK - 1) / CHUNK);
            const float* px = (const float*)x->data;
            float* py = (float*)y->data;
            nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
                size_t start = (size_t)c * CHUNK;
                size_t end = std::min(start + CHUNK, n);
                tanh_avx512_kernel(px + start, py + start, end - start);
            });
            return true;
#elifdef NNR_ARCH_ARM64
            constexpr size_t CHUNK = 16384;
            size_t n = x->ndata;
            int nchunks = (int)((n + CHUNK - 1) / CHUNK);
            const float* px = (const float*)x->data;
            float* py = (float*)y->data;
            nnr::for_static(0, nchunks, nchunks > 1, [&](int c) {
                size_t start = (size_t)c * CHUNK;
                size_t end = std::min(start + CHUNK, n);
                tanh_neon_kernel(px + start, py + start, end - start);
            });
            return true;
#endif
        }
        foreach_tensor<T>([](auto x){ return tanh(x); });
        return true;
    }

    bool exec() override {
        return typed_exec<Tanh_operator,
            opset_t<13, bfloat16_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8] scroll=yes fusion=post_op
operator_t* resolver_default_op_Tanh(int opset, pool_t& pool)
{
    return pool_new<Tanh_operator>(pool);
}

} // namespace nnr
