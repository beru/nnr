#include "nnr.h"
#include "util.h"

namespace nnr {

// Declared in Cast.cpp
void Cast_array(
    data_type_t from_type, const void* from_data,
    data_type_t to_type, void* to_data,
    size_t ndata);
void Cast_set_saturate(bool sat);

namespace {

struct CastLike_operator : public operator_t {
    bool saturate = true;

    bool init() override {
        if (inputs.size() < 2 || outputs.size() != 1)
            return false;
        saturate = (bool)attribute(attr_key_t::saturate, (int32_t)1);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        const tensor_t* target = inputs[1];
        tensor_t* y = outputs[0];
        return y->reshape_identity(x, target->type);
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->ndata == 0) return true;
        Cast_set_saturate(saturate);
        Cast_array(x->type, x->data, y->type, y->data, y->ndata);
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_CastLike(int opset, pool_t& pool) { return pool_new<CastLike_operator>(pool); }

} // namespace nnr
