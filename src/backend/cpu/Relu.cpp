#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#include "kernel/quant_exec.h"
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
static void dispatch_clip(Body body) {
#ifdef NNR_ARCH_X64
    const auto isa = detect_isa();
    if (isa == isa_t::avx512)
        body([](const float* s, float* d, int n) { avx512::clip(s, d, n, 0.0f, FLT_MAX); });
    else if (isa == isa_t::avx2)
        body([](const float* s, float* d, int n) { avx2::clip(s, d, n, 0.0f, FLT_MAX); });
    else
        body([](const float* s, float* d, int n) { for (int i = 0; i < n; ++i) d[i] = std::max(0.0f, s[i]); });
#elifdef NNR_ARCH_ARM64
    body([](const float* s, float* d, int n) { neon::clip(s, d, (size_t)n, 0.0f, FLT_MAX); });
#else
    body([](const float* s, float* d, int n) { for (int i = 0; i < n; ++i) d[i] = std::max(0.0f, s[i]); });
#endif
}

struct Relu_operator : public operator_t {

    static void apply_inplace(float* data, int rows, int cols, int stride,
                              const operator_t* self, const float* bias, int offset) {
        for (int r = 0; r < rows; r++) {
            float* row = data + (size_t)r * stride;
            float bv = bias ? bias[r] : 0.0f;
#ifdef NNR_ARCH_X64
            const auto isa = detect_isa();
            if (isa == isa_t::avx512) {
                if (bv != 0.0f)
                    avx512::bias_relu(row, cols, bv);
                else
                    avx512::clip(row, row, cols, 0.0f, FLT_MAX);
            } else if (isa == isa_t::avx2) {
                if (bv != 0.0f)
                    avx2::bias_relu(row, cols, bv);
                else
                    avx2::clip(row, row, cols, 0.0f, FLT_MAX);
            } else {
                for (int i = 0; i < cols; ++i)
                    row[i] = std::max(0.0f, row[i] + bv);
            }
#elifdef NNR_ARCH_ARM64
            if (bv != 0.0f)
                neon::bias_relu(row, cols, bv);
            else
                neon::clip(row, row, (size_t)cols, 0.0f, FLT_MAX);
#else
            {
                for (int i = 0; i < cols; ++i)
                    row[i] = std::max(0.0f, row[i] + bv);
            }
#endif
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

    size_t workspace_size() const override {
        if (inputs[0]->is_quantized())
            return inputs[0]->ndata * sizeof(float);  // dequantize buffer
        return 0;
    }

    scroll_info_t scroll_info() const override {
        if (inputs[0]->ndim < 3) return {};
        // Ring buffer is sized using x's declared_layout; exec_strip must
        // address y with the same per-channel-block stride. Bail on
        // mismatch — whole-tensor exec is layout-agnostic.
        if (outputs.empty() || !outputs[0]) return {};
        if (inputs[0]->declared_layout != outputs[0]->declared_layout) return {};
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
        int oH = y->dims[y->ndim - 2];
        int clamp_H = ring_out.orig_H > 0 ? ring_out.orig_H : oH;
        int out_end = std::min(out_row_start + out_rows, clamp_H);
        int rows = out_end - out_row_start;
        if (rows <= 0) return true;
        // BLOCKED layout interleaves block channels per spatial step; row
        // stride per channel block is W*block, outer iteration is N*(C/block).
        bool blocked = (y->declared_layout == NATIVE_BLOCKED_FMT);
        int block = blocked ? NATIVE_BLOCK : 1;
        int row_elems = W * block;
        int outer = blocked
            ? (int)(x->ndata / (iH * row_elems))
            : (int)(x->ndata / (iH * W));
        int count = rows * row_elems;
        const float* px = (const float*)x->data;
        float* py = (float*)y->data;
        dispatch_clip([&](auto clip_fn) {
            nnr::for_static(0, outer, outer > 4, [&](int nc) {
                const float* src = px + (size_t)nc * iH * row_elems + (size_t)out_row_start * row_elems;
                float* dst = py + (size_t)nc * oH * row_elems + (size_t)out_row_start * row_elems;
                clip_fn(src, dst, count);
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
            dispatch_clip([&](auto clip_fn) { clip_fn(px, py, len); });
        } else {
            foreach_tensor<T>([](auto x){return std::max((T)0, x);});
        }
        return true;
    }

    bool exec() override {
        // QDQ-fused path: int8/int16 input with quant metadata
        if (inputs[0]->is_quantized())
            return exec_quantized_unary(this, [](float x) { return std::max(0.0f, x); });
        return typed_exec<Relu_operator,
            opset_t<14, int8_t, int16_t, int32_t, int64_t, bfloat16_t, float16_t, float, double>,
            opset_t<13, bfloat16_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=static layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8] workspace=yes scroll=yes fusion=post_op
operator_t* resolver_default_op_Relu(int opset, pool_t& pool) { return pool_new<Relu_operator>(pool); }

} // namespace nnr
