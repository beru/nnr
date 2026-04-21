#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#include <vector>

namespace nnr {

namespace {

struct TopK_operator : public operator_t {
    int axis;
    int caxis;
    int largest;
    int sorted;

    bool init() override {
        if (!(outputs.size() == 2)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, (int32_t)-1);
        largest = attribute(attr_key_t::largest, (int32_t)1);
        sorted = attribute(attr_key_t::sorted, (int32_t)1);
        return true;
    }

    int get_k() {
        if (opset >= 11) {
            // K is input[1], an int64 scalar
            if (inputs.size() < 2 || !inputs[1]) {
                return -1;
            }
            const tensor_t* kt = inputs[1];
            if (!kt->data) return -1;
            return (int)(*(int64_t*)kt->data);
        }else {
            // K is an attribute for opset < 11
            return (int)attribute(attr_key_t::k, (int64_t)0);
        }
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* values = outputs[0];
        tensor_t* indices = outputs[1];
        const int ndim = x->ndim;

        int k = get_k();
        if (k <= 0) {
            return false;
        }

        caxis = axis;
        if (caxis < 0) {
            caxis += ndim;
        }
        if (caxis < 0 || caxis >= ndim) {
            return false;
        }

        small_vector<int> dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            dims[i] = x->dims[i];
        }
        dims[caxis] = k;

        if (!values->reshape(dims, x->type)) {
            return false;
        }
        if (!indices->reshape(dims, NNR_DATA_TYPE_INT64)) {
            return false;
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* val_out = outputs[0];
        tensor_t* idx_out = outputs[1];
        const T* px = (const T*)x->data;
        T* pv = (T*)val_out->data;
        int64_t* pi = (int64_t*)idx_out->data;

        int k = get_k();
        if (k <= 0) {
            return false;
        }

        const int ndim = x->ndim;
        const int axis_dim = x->dims[caxis];

        // Compute number of slices (product of all dims except axis)
        int num_slices = 1;
        for (int i = 0; i < ndim; ++i) {
            if (i != caxis) {
                num_slices *= x->dims[i];
            }
        }

        // For each slice along the axis, collect values, sort, take top K
        small_vector<int> idx(ndim);
        // We'll iterate over all positions in the output
        // but it's easier to iterate per-slice

        // Compute strides for iterating over non-axis dims
        // We iterate over all combinations of non-axis indices
        // For each such combination, we have a 1D slice of length axis_dim

        // Use a temporary buffer for sorting
        struct val_idx {
            T value;
            int64_t index;
        };

        // Check if axis is the last dim (contiguous access pattern)
        bool axis_last = (caxis == ndim - 1);
        int64_t inner_stride = axis_last ? 1 : x->strides[caxis];
        int outer_stride_x = axis_last ? axis_dim : 1;
        int outer_stride_v = axis_last ? k : 1;

        // Threaded + partial_sort for K << axis_dim
        nnr::for_static(0, num_slices, num_slices > 4, [&](int s) {
            // Thread-local buffer
            std::vector<val_idx> buf(axis_dim);

            // Compute base offset for this slice
            int64_t base_x, base_v;
            if (axis_last) {
                base_x = (int64_t)s * axis_dim;
                base_v = (int64_t)s * k;
            } else {
                // General: compute offset from slice index
                small_vector<int> outer(ndim);
                std::fill(outer.begin(), outer.end(), 0);
                int rem = s;
                for (int d = ndim - 1; d >= 0; --d) {
                    if (d == caxis) continue;
                    outer[d] = rem % x->dims[d];
                    rem /= x->dims[d];
                }
                base_x = x->indices_to_offset(outer);
                base_v = val_out->indices_to_offset(outer);
            }

            // Fill buffer
            for (int a = 0; a < axis_dim; ++a) {
                buf[a].value = px[base_x + a * inner_stride];
                buf[a].index = a;
            }

            // Use partial_sort when K < axis_dim (O(N log K) vs O(N log N))
            auto cmp_largest = [](const val_idx& a, const val_idx& b) {
                return (float)a.value > (float)b.value ||
                       ((float)a.value == (float)b.value && a.index < b.index);
            };
            auto cmp_smallest = [](const val_idx& a, const val_idx& b) {
                return (float)a.value < (float)b.value ||
                       ((float)a.value == (float)b.value && a.index < b.index);
            };
            if (k < axis_dim) {
                if (largest)
                    std::partial_sort(buf.begin(), buf.begin() + k, buf.end(), cmp_largest);
                else
                    std::partial_sort(buf.begin(), buf.begin() + k, buf.end(), cmp_smallest);
            } else {
                if (largest)
                    std::sort(buf.begin(), buf.end(), cmp_largest);
                else
                    std::sort(buf.begin(), buf.end(), cmp_smallest);
            }

            // Write top K
            int64_t v_stride = axis_last ? 1 : val_out->strides[caxis];
            for (int a = 0; a < k; ++a) {
                int64_t off_v = base_v + a * v_stride;
                pv[off_v] = buf[a].value;
                pi[off_v] = buf[a].index;
            }
        });

        return true;
    }

    bool exec() override {
        return typed_exec<TopK_operator,
            opset_t<11, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float16_t, float, double, bfloat16_t>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_TopK(int opset, pool_t& pool)
{
    return pool_new<TopK_operator>(pool);
}

} // namespace nnr
