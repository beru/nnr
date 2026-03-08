#include "reduce_base.h"

namespace nnr {

namespace {

struct ReduceProd_operator : reduce_base_t {
    template <typename T>
    bool exec() {
        using AccT = typename acc_float<T>::type;
        return reduce_exec_accum<T, acc_float>(this, T(1), AccT(1),
            [](AccT acc, T v) { return acc * (AccT)v; },
            [](AccT acc) { return (T)acc; });
    }

    bool exec() override {
        if (inputs[0]->is_quantized())
            return exec_quantized_via_float(this, [this]() { return exec<float>(); });
        return typed_exec<ReduceProd_operator,
            opset_t<13, uint8_t, uint32_t, uint64_t, int8_t, int32_t, int64_t, float16_t, float, double, bfloat16_t>,
            opset_t<1, uint8_t, uint32_t, uint64_t, int8_t, int32_t, int64_t, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ReduceProd(int opset, pool_t& pool) { return pool_new<ReduceProd_operator>(pool); }

} // namespace nnr
