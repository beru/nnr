#include <cstring>

#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct BitCast_operator : public operator_t {
    data_type_t to;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        to = (data_type_t)attribute(attr_key_t::to, inputs[0]->type);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        const size_t src_size = data_type_sizeof(x->type);
        const size_t dst_size = data_type_sizeof(to);
        if (src_size == 0 || dst_size == 0) return false;

        if (src_size == dst_size) {
            return y->reshape_identity(x, to);
        }

        // Innermost-dim resize: total bytes preserved.
        if (x->ndim == 0) return false;
        int last = x->dims[x->ndim - 1];
        size_t last_bytes = (size_t)last * src_size;
        if (last_bytes % dst_size != 0) return false;
        int new_last = (int)(last_bytes / dst_size);

        int dims[MAX_NDIM];
        for (int i = 0; i < x->ndim - 1; ++i) dims[i] = x->dims[i];
        dims[x->ndim - 1] = new_last;
        return y->reshape(std::span<const int>(dims, x->ndim), to);
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        std::memcpy(y->data, x->data, x->ndata * data_type_sizeof(x->type));
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_BitCast(int opset, pool_t& pool)
{
    return pool_new<BitCast_operator>(pool);
}

} // namespace nnr
