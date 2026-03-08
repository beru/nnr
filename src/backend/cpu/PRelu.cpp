#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct PRelu_operator : public operator_t {

    bool init() override {
        layout_mask = LAYOUT_ALL;
        return is_inout_size(2, 1);
    }

    bool reshape() override {
        if (!outputs[0]->reshape_multi_broadcast(inputs[0], inputs[1], inputs[0]->type))
            return false;
        auto kind = classify_broadcast(inputs[0], inputs[1], outputs[0]);
        layout_mask = (kind == broadcast_kind::GENERAL) ? LAYOUT_NCHW : LAYOUT_ALL;
        return true;
    }

    template <typename T>
    bool exec() {
        return binary_broadcast_exec<T>(inputs[0], inputs[1], outputs[0],
            [](T a, T b) -> T { return a < T(0) ? T(a * b) : a; });
    }

    bool exec() override {
        return typed_exec<PRelu_operator,
            opset_t<9, int32_t, int64_t, uint32_t, uint64_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_PRelu(int opset, pool_t& pool) { return pool_new<PRelu_operator>(pool); }

} // namespace nnr
