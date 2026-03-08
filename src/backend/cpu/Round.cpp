#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Round_operator : public operator_t {

    bool init() override {
        layout_mask = LAYOUT_ALL;
        return is_inout_size(1, 1);
    }

    template <typename T>
    bool exec() {
        int r = fegetround();
        if (r != FE_TONEAREST) {
            fesetround(FE_TONEAREST);
        }
        foreach_tensor<T>([](auto x){return nearbyint(x);});
        if (r != FE_TONEAREST) {
            fesetround(r);
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Round_operator,
            opset_t<11, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_Round(int opset, pool_t& pool) { return pool_new<Round_operator>(pool); }

} // namespace nnr
