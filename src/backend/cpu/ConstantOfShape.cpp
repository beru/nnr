#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

union scalar_t {
    uint8_t v_bool;
    int8_t v_int8;
    int16_t v_int16;
    int32_t v_int32;
    int64_t v_int64;
    uint8_t v_uint8;
    uint16_t v_uint16;
    uint32_t v_uint32;
    uint64_t v_uint64;
    uint16_t v_bfloat16;
    uint16_t v_float16;
    float v_float32;
    double v_float64;
    struct { float real; float imaginary; } v_complex64;
    struct { double real; double imaginary; } v_complex128;
};

struct ConstantOfShape_operator : public operator_t {
    data_type_t type;
    scalar_t scalar;
    size_t size;

    bool init() override {
        if (!is_inout_size(1, 1))
            return false;
        attr_t* a = find_attr("value");
        if (a && a->kind == attr_t::kind_t::TENSOR && a->tensor) {
            tensor_t* t = a->tensor;
            type = t->type;
            size = data_type_sizeof(type);
            memset(&scalar, 0, sizeof(scalar_t));
            if (t->data && t->ndata > 0 && size > 0)
                memcpy(&scalar, t->data, size);
        } else {
            type = NNR_DATA_TYPE_FLOAT32;
            size = sizeof(float);
            memset(&scalar, 0, sizeof(scalar_t));
        }
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        // Input is a 1D tensor of target dimensions. If data is available
        // (e.g. from a folded Constant), set the output shape now so
        // downstream nodes can reshape before exec runs.
        if (x->data && x->ndata > 0) {
            small_vector<int> dims((int)x->ndata);
            for (int i = 0; i < (int)x->ndata; ++i)
                dims[i] = (int)((const int64_t*)x->data)[i];
            return y->reshape(dims, type);
        }
        return true;
    }

    bool exec() override {
        if (opset >= 9) {
            const tensor_t* x = inputs[0];
            tensor_t* y = outputs[0];
            if (x->ndata > 0) {
                small_vector<int> dims((int)x->ndata);
                for (int i = 0; i < (int)x->ndata; ++i)
                    dims[i] = (int)((const int64_t*)x->data)[i];
                if (!y->reinit(type, dims)) return false;
            } else {
                if (!y->reinit(type, {})) return false;
            }
            char* p = (char*)y->data;
            for (size_t i = 0; i < y->ndata; ++i) {
                memcpy(p, &scalar, size);
                p += size;
            }
            return true;
        }
        return false;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ConstantOfShape(int opset, pool_t& pool) { return pool_new<ConstantOfShape_operator>(pool); }

} // namespace nnr
