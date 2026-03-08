#include "reduce_base.h"

namespace nnr {

namespace {

struct ReduceMean_operator : reduce_base_t {
    template <typename T>
    bool exec() {
        using AccT = typename acc_widen<T>::type;
        const tensor_t* x = inputs[0];
        int reduce_count = 1;
        for (size_t i = 0; i < caxes.size(); ++i) reduce_count *= x->dims[caxes[i]];
        mean_div<AccT> div(reduce_count);
        return reduce_exec_accum<T, acc_widen>(this, T(0), AccT(0),
            [](AccT acc, T v) { return acc + (AccT)v; },
            [div](AccT acc) { return (T)div(acc); });
    }

    bool exec() override {
        if (inputs[0]->is_quantized())
            return exec_quantized_via_float(this, [this]() { return exec<float>(); });
        return typed_exec<ReduceMean_operator,
            opset_t<13, uint8_t, uint32_t, uint64_t, int8_t, int32_t, int64_t, float16_t, float, double, bfloat16_t>,
            opset_t<1, uint8_t, uint32_t, uint64_t, int8_t, int32_t, int64_t, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ReduceMean(int opset, pool_t& pool) { return pool_new<ReduceMean_operator>(pool); }

} // namespace nnr
