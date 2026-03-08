#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/vec_ops_avx512.h"
#include "backend/x64/vec_ops_avx2.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/vec_ops_neon.h"
#endif
#include "cpu_features.h"

namespace nnr {

namespace {

template <typename Body>
static void dispatch_leaky_relu(Body body) {
#ifdef NNR_ARCH_X64
    const auto isa = detect_isa();
    if (isa == isa_t::avx512)
        body([](const float* s, float* d, int n, float a) { avx512::leaky_relu(s, d, n, a); });
    else if (isa == isa_t::avx2)
        body([](const float* s, float* d, int n, float a) { avx2::leaky_relu(s, d, n, a); });
    else
        body([](const float* s, float* d, int n, float a) {
            for (int i = 0; i < n; ++i) d[i] = s[i] >= 0 ? s[i] : s[i] * a;
        });
#elifdef NNR_ARCH_ARM64
    body([](const float* s, float* d, int n, float a) { neon::leaky_relu(s, d, n, a); });
#else
    body([](const float* s, float* d, int n, float a) {
        for (int i = 0; i < n; ++i) d[i] = s[i] >= 0 ? s[i] : s[i] * a;
    });
#endif
}

struct LeakyRelu_operator : public operator_t {
    float alpha;

    static void apply_inplace(float* data, int rows, int cols, int stride,
                              const operator_t* fused_op, const float* bias, int offset) {
        float a = static_cast<const LeakyRelu_operator*>(fused_op)->alpha;
        for (int r = 0; r < rows; r++) {
            float* row = data + (size_t)r * stride;
            float bv = bias ? bias[r] : 0.0f;
#ifdef NNR_ARCH_X64
            const auto isa = detect_isa();
            if (isa == isa_t::avx512) {
                if (bv != 0.0f)
                    avx512::bias_leaky_relu(row, cols, bv, a);
                else
                    avx512::leaky_relu(row, row, cols, a);
            } else if (isa == isa_t::avx2) {
                if (bv != 0.0f)
                    avx2::bias_leaky_relu(row, cols, bv, a);
                else
                    avx2::leaky_relu(row, row, cols, a);
            } else {
                for (int i = 0; i < cols; ++i) {
                    float v = row[i] + bv;
                    row[i] = v >= 0 ? v : v * a;
                }
            }
#elifdef NNR_ARCH_ARM64
            if (bv != 0.0f)
                neon::bias_leaky_relu(row, cols, bv, a);
            else
                neon::leaky_relu(row, row, cols, a);
#else
            {
                for (int i = 0; i < cols; ++i) {
                    float v = row[i] + bv;
                    row[i] = v >= 0 ? v : v * a;
                }
            }
#endif
        }
        if (fused_op->post_fn)
            fused_op->post_fn(data, rows, cols, stride, fused_op->fused_op, nullptr, offset);
    }

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        fusable_apply = &apply_inplace;
        layout_mask = LAYOUT_ALL;
        alpha = attribute(attr_key_t::alpha, 0.01f);
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
        const tensor_t* y = outputs[0];
        if (x->ndim < 3) return false;
        int iH = x->dims[x->ndim - 2];
        int W  = x->dims[x->ndim - 1];
        int outer = (int)(x->ndata / (iH * W));
        int oH = y->dims[y->ndim - 2];
        int clamp_H = ring_out.orig_H > 0 ? ring_out.orig_H : oH;
        int out_end = std::min(out_row_start + out_rows, clamp_H);
        int count = (out_end - out_row_start) * W;
        if (count <= 0) return true;
        const float* px = (const float*)x->data;
        float* py = (float*)y->data;
        float a = alpha;
        dispatch_leaky_relu([&](auto fn) {
            nnr::for_static(0, outer, outer > 4, [&](int nc) {
                const float* src = px + (size_t)nc * iH * W + (size_t)out_row_start * W;
                float* dst = py + (size_t)nc * oH * W + (size_t)out_row_start * W;
                fn(src, dst, count, a);
            });
        });
        return true;
    }

    template <typename T>
    bool exec() {
        if constexpr (std::is_same_v<T, float>) {
            const float* px = (const float*)inputs[0]->data;
            float* py = (float*)outputs[0]->data;
            int len = (int)outputs[0]->ndata;
            float a = alpha;
            dispatch_leaky_relu([&](auto fn) { fn(px, py, len, a); });
        } else {
            const tensor_t* x = inputs[0];
            tensor_t* y = outputs[0];
            const T* px = (const T*)x->data;
            T* py = (T*)y->data;
            for (size_t i = 0, l = y->ndata; i < l; ++i) {
                T v = px[i];
                if (v < 0) v *= alpha;
                py[i] = v;
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<LeakyRelu_operator,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=static layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8] scroll=yes
operator_t* resolver_default_op_LeakyRelu(int opset, pool_t& pool) { return pool_new<LeakyRelu_operator>(pool); }

} // namespace nnr
