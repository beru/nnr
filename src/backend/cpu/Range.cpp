#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

double tensor_get_value(void* p, data_type_t type)
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

struct Range_operator : public operator_t {
    double start = 0;
    double limit = 0;
    double delta = 0;

    bool init() override {
        if (!is_inout_size(3, 1)) {
            return false;
        }
        return true;
    }

    bool reshape() override {
        tensor_t* y = outputs[0];
        start = tensor_get_value(inputs[0]->data, inputs[0]->type);
        limit = tensor_get_value(inputs[1]->data, inputs[1]->type);
        delta = tensor_get_value(inputs[2]->data, inputs[2]->type);
        int ndim = (int)fmax(ceil((limit - start) / delta), 0);
        int tmp[] = { ndim };
        return y->reshape(tmp, inputs[0]->type);
    }

    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        T* py = (T*)y->data;
        for (size_t i = 0, l = y->ndata; i < l; ++i) {
            py[i] = (T)(start + (delta * i));
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Range_operator,
            opset_t<11, int16_t, int32_t, int64_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Range(int opset, pool_t& pool) { return pool_new<Range_operator>(pool); }

} // namespace nnr
