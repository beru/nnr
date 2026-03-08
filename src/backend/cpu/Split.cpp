#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Split_operator : public operator_t {
    int axis;
    int caxis;
    small_vector<int> split_sizes;
    bool is_contiguous_view_ = false;

    bool init() override {
        if (!(inputs.size() >= 1 && outputs.size() >= 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, (int32_t)0);

        // For opset < 13, split sizes come from attributes
        if (opset < 13) {
            int64_t* ints;
            int n = attribute(attr_key_t::split, ints);
            if (n > 0) {
                split_sizes.resize(n);
                for (int i = 0; i < n; ++i) {
                    split_sizes[i] = static_cast<int>(ints[i]);
                }
            }
        }
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        const int ndim = x->ndim;
        if (ndim == 0) {
            return false;
        }

        caxis = axis;
        if (caxis < 0) {
            caxis += ndim;
        }
        if (caxis < 0 || caxis >= ndim) {
            return false;
        }

        // Determine split sizes
        small_vector<int> splits;
        int num_outputs = static_cast<int>(outputs.size());

        if (opset >= 13 && inputs.size() >= 2 && inputs[1] && inputs[1]->ndata > 0) {
            // Split sizes from input tensor (opset >= 13)
            const int64_t* ps = (const int64_t*)inputs[1]->data;
            int n = static_cast<int>(inputs[1]->ndata);
            splits.resize(n);
            for (int i = 0; i < n; ++i) {
                splits[i] = static_cast<int>(ps[i]);
            }
        } else if (split_sizes.size() > 0) {
            // Split sizes from attributes (opset < 13)
            splits = split_sizes;
        } else {
            // Equal split
            int dim = x->dims[caxis];
            int chunk = dim / num_outputs;
            int remainder = dim % num_outputs;
            splits.resize(num_outputs);
            for (int i = 0; i < num_outputs; ++i) {
                splits[i] = chunk + (i < remainder ? 1 : 0);
            }
        }

        if (splits.size() != num_outputs) {
            return false;
        }

        // Reshape each output
        for (int i = 0; i < num_outputs; ++i) {
            small_vector<int> dims(ndim);
            for (int j = 0; j < ndim; ++j) {
                dims[j] = x->dims[j];
            }
            dims[caxis] = splits[i];
            if (!outputs[i]->reshape({dims.data(), static_cast<size_t>(ndim)}, x->type)) {
                return false;
            }
        }

        // Check if outputs are contiguous views into the input.
        // When all dims before split axis multiply to 1, each output
        // is a contiguous block — can be a zero-copy view.
        is_contiguous_view_ = false;
        if (x->format == memory_layout_t::NCHW &&
            x->type != NNR_DATA_TYPE_STRING) {
            int outer = 1;
            for (int d = 0; d < caxis; d++) outer *= x->dims[d];
            if (outer == 1)
                is_contiguous_view_ = true;
        }
        return true;
    }

    int view_input_index() const override {
        return is_contiguous_view_ ? 0 : -1;
    }

    bool exec_impl() {
        const tensor_t* x = inputs[0];
        int num_outputs = static_cast<int>(outputs.size());

        // Zero-copy view: point each output into the input's contiguous data
        if (is_contiguous_view_) {
            const size_t sz = data_type_sizeof(x);
            size_t offset = 0;
            for (int i = 0; i < num_outputs; ++i) {
                tensor_t* y = outputs[i];
                if (!y->owns_data)
                    y->data = (char*)x->data + offset;
                else if (y->data != (char*)x->data + offset)
                    memcpy(y->data, (char*)x->data + offset, y->ndata * sz);
                offset += y->ndata * sz;
            }
            return true;
        }

        if (x->type == NNR_DATA_TYPE_STRING) {
            const std::string* px = (const std::string*)x->data;
            // Pitch: number of elements from caxis onward in input
            int xpitch = std::reduce(x->dims + caxis, x->dims + x->ndim, 1, std::multiplies<>{});
            // Elements before the split axis
            int outer = std::reduce(x->dims, x->dims + caxis, 1, std::multiplies<>{});
            // Elements after the split axis
            int inner = (caxis + 1 < x->ndim) ?
                std::reduce(x->dims + caxis + 1, x->dims + x->ndim, 1, std::multiplies<>{}) : 1;

            int offset_along_axis = 0;
            for (int i = 0; i < num_outputs; ++i) {
                tensor_t* y = outputs[i];
                std::string* py = (std::string*)y->data;
                int split_dim = y->dims[caxis];
                int ypitch = split_dim * inner;

                for (int o = 0; o < outer; ++o) {
                    for (int s = 0; s < split_dim; ++s) {
                        for (int k = 0; k < inner; ++k) {
                            int src = o * xpitch + (offset_along_axis + s) * inner + k;
                            int dst = o * ypitch + s * inner + k;
                            py[dst] = px[src];
                        }
                    }
                }
                offset_along_axis += split_dim;
            }
        } else {
            const char* px = (const char*)x->data;
            const size_t sz = data_type_sizeof(x);
            int xpitch = std::reduce(x->dims + caxis, x->dims + x->ndim, 1, std::multiplies<>{});
            int outer = std::reduce(x->dims, x->dims + caxis, 1, std::multiplies<>{});
            int inner = (caxis + 1 < x->ndim) ?
                std::reduce(x->dims + caxis + 1, x->dims + x->ndim, 1, std::multiplies<>{}) : 1;

            int offset_along_axis = 0;
            for (int i = 0; i < num_outputs; ++i) {
                tensor_t* y = outputs[i];
                char* py = (char*)y->data;
                int split_dim = y->dims[caxis];
                int ypitch = split_dim * inner;

                for (int o = 0; o < outer; ++o) {
                    // Copy one contiguous block per outer iteration
                    // Each block is split_dim * inner elements
                    const char* src = px + (o * xpitch + offset_along_axis * inner) * sz;
                    char* dst = py + o * ypitch * sz;
                    memcpy(dst, src, ypitch * sz);
                }
                offset_along_axis += split_dim;
            }
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        if (opset >= 13) {
            switch (type) {
            case NNR_DATA_TYPE_BOOL:
            case NNR_DATA_TYPE_INT8:
            case NNR_DATA_TYPE_INT16:
            case NNR_DATA_TYPE_INT32:
            case NNR_DATA_TYPE_INT64:
            case NNR_DATA_TYPE_UINT8:
            case NNR_DATA_TYPE_UINT16:
            case NNR_DATA_TYPE_UINT32:
            case NNR_DATA_TYPE_UINT64:
            case NNR_DATA_TYPE_BFLOAT16:
            case NNR_DATA_TYPE_FLOAT16:
            case NNR_DATA_TYPE_FLOAT32:
            case NNR_DATA_TYPE_FLOAT64:
            case NNR_DATA_TYPE_COMPLEX64:
            case NNR_DATA_TYPE_COMPLEX128:
            case NNR_DATA_TYPE_STRING:
                return exec_impl();
            default:
                break;
            }
        } else if (opset >= 11) {
            switch (type) {
            case NNR_DATA_TYPE_BOOL:
            case NNR_DATA_TYPE_INT8:
            case NNR_DATA_TYPE_INT16:
            case NNR_DATA_TYPE_INT32:
            case NNR_DATA_TYPE_INT64:
            case NNR_DATA_TYPE_UINT8:
            case NNR_DATA_TYPE_UINT16:
            case NNR_DATA_TYPE_UINT32:
            case NNR_DATA_TYPE_UINT64:
            case NNR_DATA_TYPE_FLOAT16:
            case NNR_DATA_TYPE_FLOAT32:
            case NNR_DATA_TYPE_FLOAT64:
            case NNR_DATA_TYPE_COMPLEX64:
            case NNR_DATA_TYPE_COMPLEX128:
            case NNR_DATA_TYPE_STRING:
                return exec_impl();
            default:
                break;
            }
        } else if (opset >= 2) {
            switch (type) {
            case NNR_DATA_TYPE_BOOL:
            case NNR_DATA_TYPE_INT8:
            case NNR_DATA_TYPE_INT16:
            case NNR_DATA_TYPE_INT32:
            case NNR_DATA_TYPE_INT64:
            case NNR_DATA_TYPE_UINT8:
            case NNR_DATA_TYPE_UINT16:
            case NNR_DATA_TYPE_UINT32:
            case NNR_DATA_TYPE_UINT64:
            case NNR_DATA_TYPE_FLOAT16:
            case NNR_DATA_TYPE_FLOAT32:
            case NNR_DATA_TYPE_FLOAT64:
            case NNR_DATA_TYPE_COMPLEX64:
            case NNR_DATA_TYPE_COMPLEX128:
            case NNR_DATA_TYPE_STRING:
                return exec_impl();
            default:
                break;
            }
        } else if (opset >= 1) {
            switch (type) {
            case NNR_DATA_TYPE_FLOAT16:
            case NNR_DATA_TYPE_FLOAT32:
            case NNR_DATA_TYPE_FLOAT64:
                return exec_impl();
            default:
                break;
            }
        }
        return false;
    }
};

} // namespace

// @nnr-meta-op mt=no inplace=yes
operator_t* resolver_default_op_Split(int opset, pool_t& pool) { return pool_new<Split_operator>(pool); }

} // namespace nnr
