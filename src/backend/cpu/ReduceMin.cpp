#include <limits>
#include "reduce_base.h"

namespace nnr {

namespace {

struct ReduceMin_operator : reduce_base_t {
    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)outputs[0]->data;
        if (x->ndata == 0) {
            if constexpr (std::is_floating_point_v<T>) {
                for (size_t i = 0; i < outputs[0]->ndata; ++i) py[i] = std::numeric_limits<T>::infinity();
            } else if constexpr (std::is_constructible_v<T, float>) {
                for (size_t i = 0; i < outputs[0]->ndata; ++i) py[i] = T(std::numeric_limits<float>::infinity());
            } else if constexpr (std::is_arithmetic_v<T>) {
                for (size_t i = 0; i < outputs[0]->ndata; ++i) py[i] = std::numeric_limits<T>::max();
            }
            return true;
        }
        if (caxes.empty()) { memcpy(py, px, x->ndata * sizeof(T)); return true; }
        {
            auto plan = plan_reduce(x->dims, x->ndim, caxes.data(), (int)caxes.size());
            if (plan.contiguous) {
                int batch = plan.batch_size, red = plan.reduce_size, tail = plan.tail_size;
                for (int b = 0; b < batch; ++b)
                    for (int t = 0; t < tail; ++t) {
                        T acc = px[b * red * tail + t];
                        for (int r = 1; r < red; ++r) {
                            T v = px[(b * red + r) * tail + t];
                            if (v < acc) acc = v;
                        }
                        py[b * tail + t] = acc;
                    }
                return true;
            }
        }
        reduce_scatter_loop<T>(this, px, py, [](const T* px, int o, auto& iter, auto& strides, auto& maxes) -> T {
            T minv = px[o];
            do {
                T v = px[o + stride_offset(iter, strides)];
                if (minv > v) minv = v;
            } while (dim_next(iter, maxes));
            return minv;
        });
        return true;
    }

    bool exec() override {
        if (inputs[0]->is_quantized())
            return exec_quantized_via_float(this, [this]() { return exec<float>(); });
        return typed_exec<ReduceMin_operator,
            opset_t<20, bool_t, int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
            opset_t<13, int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
            opset_t<1, int32_t, int64_t, uint32_t, uint64_t, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ReduceMin(int opset, pool_t& pool) { return pool_new<ReduceMin_operator>(pool); }

} // namespace nnr
