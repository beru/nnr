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

struct Clip_operator : public operator_t {
    void* pmin = nullptr;
    void* pmax = nullptr;

    static void apply_inplace(float* data, int rows, int cols, int stride,
                              const operator_t* op, const float* bias, int offset) {
        auto* self = static_cast<const Clip_operator*>(op);
        float lo = self->pmin ? *(float*)self->pmin : -FLT_MAX;
        float hi = self->pmax ? *(float*)self->pmax : FLT_MAX;
        for (int r = 0; r < rows; r++) {
            float* row = data + (size_t)r * stride;
            float bv = bias ? bias[r] : 0.0f;
#ifdef NNR_ARCH_X64
            const auto isa = detect_isa();
            if (isa == isa_t::avx512) {
                if (bv != 0.0f)
                    avx512::bias_clip(row, cols, bv, lo, hi);
                else
                    avx512::clip(row, row, cols, lo, hi);
            } else if (isa == isa_t::avx2) {
                if (bv != 0.0f)
                    avx2::bias_clip(row, cols, bv, lo, hi);
                else
                    avx2::clip(row, row, cols, lo, hi);
            } else {
                for (int i = 0; i < cols; ++i)
                    row[i] = std::clamp(row[i] + bv, lo, hi);
            }
#elifdef NNR_ARCH_ARM64
            if (bv != 0.0f)
                neon::bias_clip(row, cols, bv, lo, hi);
            else
                neon::clip(row, row, (size_t)cols, lo, hi);
#else
            {
                for (int i = 0; i < cols; ++i)
                    row[i] = std::clamp(row[i] + bv, lo, hi);
            }
#endif
        }
        if (op->post_fn)
            op->post_fn(data, rows, cols, stride, op->fused_op, nullptr, offset);
    }

    bool init() override {
        if (!(inputs.size() >= 1 && outputs.size() == 1)) {
            return false;
        }
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
        float lo = pmin ? *(float*)pmin : -FLT_MAX;
        float hi = pmax ? *(float*)pmax : FLT_MAX;
        int iH = x->dims[x->ndim - 2];
        int W = x->dims[x->ndim - 1];
        int oH = y->dims[y->ndim - 2];
        int outer = (int)(x->ndata / (iH * W));
        int clamp_H = ring_out.orig_H > 0 ? ring_out.orig_H : oH;
        int out_end = std::min(out_row_start + out_rows, clamp_H);
        int count = (out_end - out_row_start) * W;
        if (count <= 0) return true;
        auto strip_fn = [&](const float* src, float* dst, int n) {
#ifdef NNR_ARCH_ARM64
            neon::clip(src, dst, (size_t)n, lo, hi);
#else
            for (int i = 0; i < n; ++i)
                dst[i] = std::clamp(src[i], lo, hi);
#endif
        };
        nnr::for_static(0, outer, outer > 4, [&](int nc) {
            const float* src = px + (size_t)nc * iH * W + (size_t)out_row_start * W;
            float* dst = py + (size_t)nc * oH * W + (size_t)out_row_start * W;
            strip_fn(src, dst, count);
        });
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        pmin = nullptr;
        pmax = nullptr;
        // inputs[1] = min (optional), inputs[2] = max (optional)
        if (inputs.size() > 1 && inputs[1] && inputs[1]->ndata > 0)
            pmin = inputs[1]->data;
        if (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0)
            pmax = inputs[2]->data;
        return y->reshape_identity(x);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        T minv = pmin ? *(T*)pmin : std::numeric_limits<T>::lowest();
        T maxv = pmax ? *(T*)pmax : std::numeric_limits<T>::max();

#ifdef NNR_ARCH_X64
        if constexpr (std::is_same_v<T, float>) {
            if (has_avx512()) {
                avx512::clip(px, py, y->ndata, minv, maxv);
                return true;
            }
            if (detect_isa() == isa_t::avx2) {
                avx2::clip(px, py, y->ndata, minv, maxv);
                return true;
            }
        }
#elifdef NNR_ARCH_ARM64
        if constexpr (std::is_same_v<T, float>) {
            neon::clip(px, py, y->ndata, minv, maxv);
            return true;
        }
#endif
        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            T v = px[i];
            v = max(v, minv);
            v = min(v, maxv);
            py[i] = v;
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Clip_operator,
            opset_t<13, int8_t, int16_t, int32_t, int64_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                bfloat16_t, float16_t, float, double>,
            opset_t<12, int8_t, int16_t, int32_t, int64_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                float16_t, float, double>,
            opset_t<11, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=static layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8] scroll=yes fusion=post_op
operator_t* resolver_default_op_Clip(int opset, pool_t& pool) { return pool_new<Clip_operator>(pool); }

} // namespace nnr
