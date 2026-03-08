#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct ConcatFromSequence_operator : public operator_t {
    int axis = 0;
    int new_axis = 0;

    bool init() override
    {
        axis = (int)attribute(attr_key_t::axis, (int64_t)0);
        new_axis = (int)attribute(attr_key_t::new_axis, (int64_t)0);
        return true;
    }

    bool reshape() override
    {
        const sequence_t* seq = tensor_get_sequence(inputs[0]);
        if (!seq || seq->tensors.empty()) return true;

        const tensor_t* first = seq->tensors[0];
        if (!first) return false;
        int ndim = first->ndim;
        int ax = axis;
        if (ax < 0) ax += (new_axis ? ndim + 1 : ndim);

        small_vector<int> out_dims;
        if (new_axis) {
            // Stack: insert a new axis
            for (int i = 0; i < ax; ++i) out_dims.push_back(first->dims[i]);
            out_dims.push_back((int)seq->tensors.size());
            for (int i = ax; i < ndim; ++i) out_dims.push_back(first->dims[i]);
        } else {
            // Concat along axis
            out_dims.resize(ndim);
            for (int i = 0; i < ndim; ++i) out_dims[i] = first->dims[i];
            for (size_t k = 1; k < seq->tensors.size(); ++k) {
                const tensor_t* t = seq->tensors[k];
                if (!t) return false;
                out_dims[ax] += t->dims[ax];
            }
        }
        return outputs[0]->reshape(out_dims, first->type);
    }

    bool exec() override
    {
        const sequence_t* seq = tensor_get_sequence(inputs[0]);
        if (!seq || seq->tensors.empty()) return true;
        if (!outputs[0]->data) return false;

        const tensor_t* first = seq->tensors[0];
        int ndim = first->ndim;
        int ax = axis;
        if (ax < 0) ax += (new_axis ? ndim + 1 : ndim);
        size_t elem_sz = data_type_sizeof(first->type);

        if (new_axis) {
            // Stack: copy each tensor along new axis
            // Stride in output before and after ax
            int before = 1;
            for (int i = 0; i < ax; ++i) before *= first->dims[i];
            int after_elem = 1;
            for (int i = ax; i < ndim; ++i) after_elem *= first->dims[i];
            int n_seq = (int)seq->tensors.size();

            char* dst_base = (char*)outputs[0]->data;
            for (int k = 0; k < n_seq; ++k) {
                const tensor_t* t = seq->tensors[k];
                if (!t || !t->data) return false;
                const char* src = (const char*)t->data;
                for (int b = 0; b < before; ++b) {
                    char* dst = dst_base + (b * n_seq + k) * after_elem * elem_sz;
                    memcpy(dst, src + b * after_elem * elem_sz, after_elem * elem_sz);
                }
            }
        } else {
            // Concat: copy contiguous slices
            int before = 1;
            for (int i = 0; i < ax; ++i) before *= first->dims[i];
            int total_after_ax = outputs[0]->dims[ax]; // total concat dim in output
            int after = 1;
            for (int i = ax + 1; i < ndim; ++i) after *= first->dims[i];

            char* dst_base = (char*)outputs[0]->data;
            for (int b = 0; b < before; ++b) {
                int out_off = 0;
                for (const auto* t : seq->tensors) {
                    if (!t || !t->data) return false;
                    int chunk = t->dims[ax] * after;
                    const char* src = (const char*)t->data + b * t->dims[ax] * after * elem_sz;
                    char* dst = dst_base + (b * total_after_ax * after + out_off) * elem_sz;
                    memcpy(dst, src, chunk * elem_sz);
                    out_off += chunk;
                }
            }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_ConcatFromSequence(int opset, pool_t& pool)
{
    if (opset >= 11) {
        return pool_new<ConcatFromSequence_operator>(pool);
    }
    return nullptr;
}

} // namespace nnr
