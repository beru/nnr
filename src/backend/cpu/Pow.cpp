#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

double tensor_get_value(const void* p, data_type_t type)
{
    double v;

    switch (type) {
    case NNR_DATA_TYPE_BOOL:
        v = *((bool_t*)p);
        break;
    case NNR_DATA_TYPE_INT8:
        v = *((int8_t*)p);
        break;
    case NNR_DATA_TYPE_INT16:
        v = *((int16_t*)p);
        break;
    case NNR_DATA_TYPE_INT32:
        v = *((int32_t*)p);
        break;
    case NNR_DATA_TYPE_INT64:
        v = (double)*((int64_t*)p);
        break;
    case NNR_DATA_TYPE_UINT8:
        v = *((uint8_t*)p);
        break;
    case NNR_DATA_TYPE_UINT16:
        v = *((uint16_t*)p);
        break;
    case NNR_DATA_TYPE_UINT32:
        v = *((uint32_t*)p);
        break;
    case NNR_DATA_TYPE_UINT64:
        v = (double)*((uint64_t*)p);
        break;
    case NNR_DATA_TYPE_BFLOAT16:
        v = *((bfloat16_t*)p);
        break;
    case NNR_DATA_TYPE_FLOAT16:
        v = *((float16_t*)p);
        break;
    case NNR_DATA_TYPE_FLOAT32:
        v = *((float*)p);
        break;
    case NNR_DATA_TYPE_FLOAT64:
        v = *((double*)p);
        break;
    default:
        v = 0;
        break;
    }
    return v;
}

struct Pow_operator : public operator_t {

    bool init() override {
        return is_inout_size(2, 1);
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
        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            const T* pa = (const T*)a->broadcast_map_address(y, i);
            const void* pb = b->broadcast_map_address(y, i);
            T v = (T)tensor_get_value(pb, b->type);
            py[i] = (T)pow(*pa, v);
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Pow_operator,
            opset_t<13, int32_t, int64_t, bfloat16_t, float16_t, float, double>,
            opset_t<12, int32_t, int64_t, float16_t, float, double>,
            opset_t<7, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Pow(int opset, pool_t& pool) { return pool_new<Pow_operator>(pool); }

} // namespace nnr
