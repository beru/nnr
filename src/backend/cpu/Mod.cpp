#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Mod_operator : public operator_t {
    int attr_fmod;

    bool init() override {
        if (!is_inout_size(2, 1)) {
            return false;
        }
        attr_fmod = attribute(attr_key_t::fmod, 0);
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
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        T* py = (T*)y->data;

        if (attr_fmod) {
            for (size_t i = 0, l = y->ndata; i < l; ++i) {
                const T* pa = (const T*)a->broadcast_map_address(y, i);
                const T* pb = (const T*)b->broadcast_map_address(y, i);
                py[i] = (T)fmod(*pa, *pb);
            }
        }else {
            for (size_t i = 0, l = y->ndata; i < l; ++i) {
                const T* pa = (const T*)a->broadcast_map_address(y, i);
                const T* pb = (const T*)b->broadcast_map_address(y, i);
                T t;
                if constexpr (std::is_integral_v<T>) {
                    t = *pa % *pb;
                    if (((t < 0) && (*pb > 0)) || ((t > 0) && (*pb < 0))) {
                        t += *pb;
                    }
                }else {
                    t = fmod(*pa, *pb);
                }
                py[i] = t;
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Mod_operator,
            opset_t<13, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
            opset_t<10, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Mod(int opset, pool_t& pool) { return pool_new<Mod_operator>(pool); }

} // namespace nnr
