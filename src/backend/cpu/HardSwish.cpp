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

struct HardSwish_operator : public operator_t {

    static void apply_inplace(float* data, int rows, int cols, int stride,
                              const operator_t* self, const float* bias, int offset) {
        for (int r = 0; r < rows; r++) {
            float* row = data + (size_t)r * stride;
            float bv = bias ? bias[r] : 0.0f;
#ifdef NNR_ARCH_X64
            const auto isa = detect_isa();
            if (isa == isa_t::avx512) {
                avx512::bias_hardswish(row, cols, bv);
            } else if (isa == isa_t::avx2) {
                avx2::bias_hardswish(row, cols, bv);
            } else {
                for (int i = 0; i < cols; ++i) {
                    float x = row[i] + bv;
                    row[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
                }
            }
#elifdef NNR_ARCH_ARM64
            neon::bias_hardswish(row, cols, bv);
#else
            for (int i = 0; i < cols; ++i) {
                float x = row[i] + bv;
                row[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
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

    scroll_info_t scroll_info() const override {
        if (inputs[0]->ndim < 3) return {};
        return { .scrollable = true };
    }

    bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) override
    {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->ndim < 3) return false;
        const float* px = (const float*)x->data;
        float* py = (float*)y->data;
        int iH = x->dims[x->ndim - 2];
        int W = x->dims[x->ndim - 1];
        int oH = y->dims[y->ndim - 2];
        int outer = (int)(x->ndata / (iH * W));
        int clamp_H = ring_out.orig_H > 0 ? ring_out.orig_H : oH;
        int out_end = std::min(out_row_start + out_rows, clamp_H);
        int count = (out_end - out_row_start) * W;
        if (count <= 0) return true;
        nnr::for_static(0, outer, outer > 4, [&](int nc) {
            const float* src = px + (size_t)nc * iH * W + (size_t)out_row_start * W;
            float* dst = py + (size_t)nc * oH * W + (size_t)out_row_start * W;
#ifdef NNR_ARCH_X64
            if (has_avx512())
                avx512::hardswish(src, dst, count);
            else if (detect_isa() == isa_t::avx2)
                avx2::hardswish(src, dst, count);
            else {
                for (int i = 0; i < count; ++i) {
                    float x = src[i];
                    dst[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
                }
            }
#elifdef NNR_ARCH_ARM64
            neon::hardswish(src, dst, count);
#else
            for (int i = 0; i < count; ++i) {
                float x = src[i];
                dst[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
            }
#endif
        });
        return true;
    }

    template <typename T>
    bool exec() {
        if constexpr (std::is_same_v<T, float>) {
            const float* px = (const float*)inputs[0]->data;
            float* py = (float*)outputs[0]->data;
            size_t len = outputs[0]->ndata;
#ifdef NNR_ARCH_X64
            if (has_avx512()) {
                avx512::hardswish(px, py, len);
                return true;
            }
            if (detect_isa() == isa_t::avx2) {
                avx2::hardswish(px, py, len);
                return true;
            }
#elifdef NNR_ARCH_ARM64
            neon::hardswish(px, py, len);
            return true;
#endif
            for (size_t i = 0; i < len; ++i) {
                float x = px[i];
                py[i] = x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f));
            }
        } else {
            foreach_tensor<T>([](auto x) {
                return (decltype(x))((float)x * std::max(0.0f, std::min(1.0f, (float)x / 6.0f + 0.5f)));
            });
        }
        return true;
    }

    bool exec() override {
        return typed_exec<HardSwish_operator,
            opset_t<14, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

    bool reshape() override {
        return outputs[0]->reshape_identity(inputs[0]);
    }
};

} // namespace {

// @nnr-meta-op mt=static layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8] scroll=yes fusion=post_op
operator_t* resolver_default_op_HardSwish(int opset, pool_t& pool) { return pool_new<HardSwish_operator>(pool); }

} // namespace nnr
