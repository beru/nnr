#include "nnr.h"
#include "util.h"
#include "allocator.h"

namespace nnr {

namespace {

struct SplitToSequence_operator : public operator_t {
    int axis = 0;
    int keepdims = 1;

    bool init() override
    {
        axis = (int)attribute(attr_key_t::axis, (int64_t)0);
        keepdims = (int)attribute(attr_key_t::keepdims, (int64_t)1);
        return true;
    }

    bool reshape() override
    {
        return outputs[0]->reshape({}, NNR_DATA_TYPE_SEQUENCE);
    }

    bool exec() override
    {
        const tensor_t* x = inputs[0];
        if (!x || !x->data) return false;

        sequence_t* seq = tensor_get_sequence(outputs[0]);
        if (!seq) return false;
        for (auto* t : seq->tensors) delete t;
        seq->tensors.clear();
        seq->elem_type = x->type;

        int ndim = x->ndim;
        int ax = axis;
        if (ax < 0) ax += ndim;
        if (ax < 0 || ax >= ndim) return false;

        int dim_size = x->dims[ax];
        size_t elem_sz = data_type_sizeof(x->type);

        // Build split sizes
        arena_scope_t scope(ctx->arena);
        arena_vector<int> splits(arena_allocator<int>{scope.arena});
        if (inputs.size() >= 2 && inputs[1] && inputs[1]->data) {
            const tensor_t* split_t = inputs[1];
            // split can be a scalar (equal splits of that size) or a 1D vector
            if (split_t->ndim == 0 || split_t->ndata == 1) {
                int chunk = (int)*(const int64_t*)split_t->data;
                for (int i = 0; i < dim_size; i += chunk) {
                    splits.push_back(std::min(chunk, dim_size - i));
                }
            } else {
                for (size_t i = 0; i < split_t->ndata; ++i) {
                    splits.push_back((int)((const int64_t*)split_t->data)[i]);
                }
            }
        } else {
            // Default: split into individual slices of size 1
            for (int i = 0; i < dim_size; ++i) {
                splits.push_back(1);
            }
        }

        // Number of elements before and after axis
        int before = 1;
        for (int i = 0; i < ax; ++i) before *= x->dims[i];
        int after = 1;
        for (int i = ax + 1; i < ndim; ++i) after *= x->dims[i];

        int offset = 0;
        for (int chunk : splits) {
            // Build output dims for this slice
            small_vector<int> out_dims(ndim);
            for (int i = 0; i < ndim; ++i) out_dims[i] = x->dims[i];
            out_dims[ax] = chunk;

            small_vector<int> slice_dims;
            if (keepdims || chunk > 1) {
                slice_dims = out_dims;
            } else {
                // Remove the axis dimension
                for (int i = 0; i < ndim; ++i) {
                    if (i != ax) slice_dims.push_back(x->dims[i]);
                }
            }

            tensor_t* elem = new (std::nothrow) tensor_t("", x->type, slice_dims);
            if (!elem) return false;

            if (elem->data && x->data) {
                // Copy data for this slice
                size_t slice_after = after * chunk;
                char* dst = (char*)elem->data;
                const char* src = (const char*)x->data + offset * after * elem_sz;
                for (int b = 0; b < before; ++b) {
                    memcpy(dst, src, slice_after * elem_sz);
                    dst += slice_after * elem_sz;
                    src += (size_t)dim_size * after * elem_sz;
                }
            }
            seq->tensors.push_back(elem);
            offset += chunk;
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_SplitToSequence(int opset, pool_t& pool)
{
    if (opset >= 11) {
        return pool_new<SplitToSequence_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
