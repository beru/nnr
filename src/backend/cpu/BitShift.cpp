#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct BitShift_operator : public operator_t {
    bool isleft;

    bool init() override {
        if (!is_inout_size(2, 1)) {
            return false;
        }
        isleft = attribute(attr_key_t::direction, "LEFT") == "LEFT";
        return true;
    }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        return y->reshape_multi_broadcast(a, b, a->type);
    }

    template <typename T>
    bool exec() {
        if (isleft)
            return binary_broadcast_exec<T>(inputs[0], inputs[1], outputs[0],
                [](T a, T b) -> T { return a << b; });
        else
            return binary_broadcast_exec<T>(inputs[0], inputs[1], outputs[0],
                [](T a, T b) -> T { return a >> b; });
    }

    bool exec() override {
        return typed_exec<BitShift_operator,
            opset_t<11, uint8_t, uint16_t, uint32_t, uint64_t>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_BitShift(int opset, pool_t& pool) { return pool_new<BitShift_operator>(pool); }

} // namespace nnr
