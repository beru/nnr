#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Abs_operator : public operator_t {
    bool init() override {
        layout_mask = LAYOUT_ALL;
        return is_inout_size(1, 1);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
                // std::abs(INT_MIN) is UB — negating the signed min value
                // overflows. Go through the unsigned representation: the
                // result's bit pattern equals INT_MIN, matching NumPy /
                // ONNX reference behavior on signed overflow.
                using U = std::make_unsigned_t<T>;
                py[i] = (px[i] < 0) ? (T)(U(0) - (U)px[i]) : px[i];
            } else if constexpr (std::is_signed_v<T>) {
                py[i] = abs(px[i]);
            } else {
                py[i] = px[i];
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Abs_operator,
            opset_t<13, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t>,
            opset_t<6, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_Abs(int opset, pool_t& pool) { return pool_new<Abs_operator>(pool); }

} // namespace nnr
