#include "nnr.h"
#include "util.h"
#include "thread_pool.h"

namespace nnr {

namespace {

struct Slice_operator : public operator_t {
    small_vector<int> starts_;
    small_vector<int> ends_;
    small_vector<int> axes_;
    small_vector<int> steps_;
    bool is_contiguous_view_ = false;  // set in reshape()

    int view_input_index() const override {
        return is_contiguous_view_ ? 0 : -1;
    }

    bool init() override {
        if (outputs.size() != 1) {
            return false;
        }
        if (opset >= 10) {
            // inputs: data, starts, ends, [axes], [steps]
            if (inputs.size() < 3 || inputs.size() > 5) {
                return false;
            }
        }else {
            // opset 1-9: data input only, starts/ends/axes from attributes
            if (inputs.size() != 1) {
                return false;
            }
        }
        return true;
    }

    // Read a start/end value from a tensor element, keeping int64 precision.
    static int64_t read_index(const tensor_t* t, int i) {
        if (t->type == NNR_DATA_TYPE_INT32)
            return ((const int32_t*)t->data)[i];
        return ((const int64_t*)t->data)[i];
    }

    static int clamp_index(int64_t idx, int dim_size, int step) {
        // Clamp to valid range depending on step direction.
        // Use int64 to avoid overflow of ONNX sentinel values (e.g. INT64_MIN).
        if (step > 0) {
            return (int)std::clamp(idx, (int64_t)0, (int64_t)dim_size);
        } else {
            return (int)std::clamp(idx, (int64_t)-1, (int64_t)(dim_size - 1));
        }
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int ndim = x->ndim;

        if (ndim == 0) {
            return false;
        }

        // Initialize per-axis defaults: start=0, end=dim, step=1
        starts_.resize(ndim);
        ends_.resize(ndim);
        axes_.resize(ndim);
        steps_.resize(ndim);
        for (int i = 0; i < ndim; ++i) {
            starts_[i] = 0;
            ends_[i] = x->dims[i];
            axes_[i] = i;
            steps_[i] = 1;
        }

        if (opset >= 10) {
            // Read starts, ends from input tensors
            const tensor_t* t_starts = inputs[1];
            const tensor_t* t_ends = inputs[2];
            int nslices = static_cast<int>(t_starts->ndata);

            // Read axes (optional input 3)
            small_vector<int> axes(nslices);
            if (inputs.size() > 3 && inputs[3]) {
                const tensor_t* t_axes = inputs[3];
                for (int i = 0; i < nslices; ++i) {
                    int a = (t_axes->type == NNR_DATA_TYPE_INT32)
                        ? ((int32_t*)t_axes->data)[i]
                        : static_cast<int>(((int64_t*)t_axes->data)[i]);
                    if (a < 0) a += ndim;
                    axes[i] = a;
                }
            }else {
                for (int i = 0; i < nslices; ++i) {
                    axes[i] = i;
                }
            }

            // Read steps (optional input 4)
            small_vector<int> steps(nslices);
            if (inputs.size() > 4 && inputs[4]) {
                const tensor_t* t_steps = inputs[4];
                for (int i = 0; i < nslices; ++i)
                    steps[i] = (int)read_index(t_steps, i);
            }else {
                for (int i = 0; i < nslices; ++i) {
                    steps[i] = 1;
                }
            }

            // Apply each slice specification
            for (int i = 0; i < nslices; ++i) {
                int a = axes[i];
                int dim = x->dims[a];
                int step = steps[i];
                if (step == 0) return false;

                // Keep full int64 precision to avoid overflow of ONNX sentinels
                // (e.g. INT64_MIN used as "before beginning" with negative step).
                int64_t start = read_index(t_starts, i);
                int64_t end   = read_index(t_ends, i);

                // Handle negative indices (only when not sentinel values)
                if (start >= -(int64_t)dim && start < 0) start += dim;
                if (end   >= -(int64_t)dim && end   < 0) end   += dim;

                // Clamp
                starts_[a] = clamp_index(start, dim, step);
                ends_[a]   = clamp_index(end,   dim, step);
                steps_[a]  = step;
            }
        }else {
            // opset 1-9: read from attributes
            int64_t* attr_starts;
            int64_t* attr_ends;
            int64_t* attr_axes;
            int nstarts = attribute(attr_key_t::starts, attr_starts);
            int nends = attribute(attr_key_t::ends, attr_ends);
            int naxes = attribute(attr_key_t::axes, attr_axes);

            if (nstarts <= 0 || nends <= 0 || nstarts != nends) {
                return false;
            }

            small_vector<int> axes(nstarts);
            if (naxes == nstarts) {
                for (int i = 0; i < nstarts; ++i) {
                    int a = static_cast<int>(attr_axes[i]);
                    if (a < 0) a += ndim;
                    axes[i] = a;
                }
            }else {
                for (int i = 0; i < nstarts; ++i) {
                    axes[i] = i;
                }
            }

            for (int i = 0; i < nstarts; ++i) {
                int a = axes[i];
                int dim = x->dims[a];
                int start = static_cast<int>(attr_starts[i]);
                int end = static_cast<int>(attr_ends[i]);

                // Handle negative indices
                if (start < 0) start += dim;
                if (end < 0) end += dim;

                // Clamp (step is always 1 for opset < 10)
                start = clamp_index(start, dim, 1);
                end = clamp_index(end, dim, 1);

                starts_[a] = start;
                ends_[a] = end;
                // steps_[a] stays 1
            }
        }

        // Compute output dimensions
        small_vector<int> dims(ndim);
        for (int i = 0; i < ndim; ++i) {
            int step = steps_[i];
            int start = starts_[i];
            int end = ends_[i];
            int size;
            if (step > 0) {
                size = (end - start + step - 1) / step;
            }else {
                size = (end - start + step + 1) / step;
            }
            if (size < 0) size = 0;
            dims[i] = size;
        }

        if (!y->reshape(dims, x->type)) return false;

        // Check if this slice is a contiguous view (all steps==1, contiguous region).
        // Only for NCHW layout — tensor strides assume standard row-major order.
        // NHWC and BLOCKED_16 have different physical layouts not reflected in strides.
        is_contiguous_view_ = false;
        if (x->type != NNR_DATA_TYPE_STRING && ndim > 0
            && x->format == memory_layout_t::NCHW) {
            bool all_step1 = true;
            for (int d = 0; d < ndim; d++)
                if (steps_[d] != 1) { all_step1 = false; break; }
            if (all_step1) {
                // Check contiguity: sliced dims must have all inner dims fully covered
                is_contiguous_view_ = true;
                for (int d = ndim - 1; d >= 0; d--) {
                    bool full = (starts_[d] == 0 && ends_[d] == x->dims[d]);
                    if (!full) {
                        for (int d2 = d + 1; d2 < ndim; d2++) {
                            if (starts_[d2] != 0 || ends_[d2] != x->dims[d2]) {
                                is_contiguous_view_ = false;
                                break;
                            }
                        }
                        break;
                    }
                }
            }
        }
        return true;
    }

    // Zero-copy view: check if the sliced region is contiguous in memory.
    // If so, output is just a pointer into the input — no copy needed.
    // Contiguous when: for each dimension d, either the slice covers the
    // full extent OR all inner dimensions (d+1..ndim-1) are fully covered.
    bool try_view() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int ndim = x->ndim;
        const size_t sz = data_type_sizeof(x);

        // Check contiguity: from innermost dim outward, the slice must
        // cover the full extent until we hit a sliced dimension.
        // After that sliced dimension, ALL remaining inner dims must be full.
        bool contiguous = true;
        for (int d = ndim - 1; d >= 0; d--) {
            bool full = (starts_[d] == 0 && ends_[d] == x->dims[d]);
            if (!full) {
                // This dim is sliced — all dims below must be full
                for (int d2 = d + 1; d2 < ndim; d2++) {
                    if (starts_[d2] != 0 || ends_[d2] != x->dims[d2]) {
                        contiguous = false;
                        break;
                    }
                }
                break;
            }
        }

        if (!contiguous) return false;

        // Compute byte offset to the start of the sliced region
        size_t offset = 0;
        for (int d = 0; d < ndim; d++)
            offset += (size_t)starts_[d] * x->strides[d];

        // Point output directly into input data — zero copy
        y->data = (char*)x->data + offset * sz;
        return true;
    }

    // Fallback: contiguous inner block memcpy + threading.
    bool exec_fast_copy() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int ndim = x->ndim;
        const size_t sz = data_type_sizeof(x);

        // Find innermost contiguous run
        int outer_dims = 0;
        for (int d = ndim - 1; d >= 1; d--) {
            if (starts_[d] != 0 || ends_[d] != x->dims[d]) {
                outer_dims = d;
                break;
            }
        }

        size_t inner_bytes = sz;
        for (int d = outer_dims; d < ndim; d++)
            inner_bytes *= y->dims[d];

        int total_outer = 1;
        for (int d = 0; d < outer_dims; d++)
            total_outer *= y->dims[d];

        const char* px = (const char*)x->data;
        char* py = (char*)y->data;

        if (total_outer == 1) {
            size_t src_off = 0;
            for (int d = 0; d < ndim; d++)
                src_off += (size_t)starts_[d] * x->strides[d];
            memcpy(py, px + src_off * sz, inner_bytes);
            return true;
        }

        small_vector<int> outer_strides(outer_dims);
        if (outer_dims > 0) {
            outer_strides[outer_dims - 1] = 1;
            for (int d = outer_dims - 2; d >= 0; d--)
                outer_strides[d] = outer_strides[d + 1] * y->dims[d + 1];
        }

        nnr::for_static(0, total_outer, total_outer > 16, [&](int flat) {
            size_t src_off = 0;
            int rem = flat;
            for (int d = 0; d < outer_dims; d++) {
                int idx = rem / outer_strides[d];
                rem %= outer_strides[d];
                src_off += (size_t)(starts_[d] + idx) * x->strides[d];
            }
            for (int d = outer_dims; d < ndim; d++)
                src_off += (size_t)starts_[d] * x->strides[d];

            memcpy(py + (size_t)flat * inner_bytes, px + src_off * sz, inner_bytes);
        });
        return true;
    }

    bool exec_impl() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int ndim = x->ndim;

        // Zero-copy view: output points into input data (no allocation needed).
        // Safe after memory planning has excluded the view output from the pool
        // (owns_data=false, data=nullptr/view) and extended the source's lifetime.
        // On the first run (before planning), owns_data=true → use memcpy.
        if (is_contiguous_view_ && !y->owns_data) {
            const size_t sz = data_type_sizeof(x);
            size_t offset = 0;
            for (int d = 0; d < ndim; d++)
                offset += (size_t)starts_[d] * x->strides[d];
            y->data = (char*)x->data + offset * sz;
            return true;
        }

        // Fast memcpy path: all steps == 1, NCHW layout, non-string
        // (strides assume standard row-major order — not valid for NHWC/BLOCKED_16)
        if (x->type != NNR_DATA_TYPE_STRING && ndim > 0
            && x->format == memory_layout_t::NCHW) {
            bool all_step1 = true;
            for (int d = 0; d < ndim; d++)
                if (steps_[d] != 1) { all_step1 = false; break; }
            if (all_step1)
                return exec_fast_copy();
        }

        if (x->type == NNR_DATA_TYPE_STRING) {
            const std::string* px = (const std::string*)x->data;
            std::string* py = (std::string*)y->data;
            small_vector<int> iy(ndim);
            small_vector<int> ix(ndim);

            for (size_t oy = 0, l = y->ndata; oy < l; ++oy) {
                y->offset_to_indices(static_cast<int>(oy), iy);
                for (int d = 0; d < ndim; ++d) {
                    ix[d] = starts_[d] + iy[d] * steps_[d];
                }
                int ox = x->indices_to_offset(ix);
                py[oy] = px[ox];
            }
        }else {
            const char* px = (const char*)x->data;
            char* py = (char*)y->data;
            const size_t sz = data_type_sizeof(x);
            small_vector<int> iy(ndim);
            small_vector<int> ix(ndim);

            for (size_t oy = 0, l = y->ndata; oy < l; ++oy) {
                y->offset_to_indices(static_cast<int>(oy), iy);
                for (int d = 0; d < ndim; ++d) {
                    ix[d] = starts_[d] + iy[d] * steps_[d];
                }
                int ox = x->indices_to_offset(ix);
                memcpy(py + oy * sz, px + ox * sz, sz);
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
        }else if (opset >= 10) {
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
        }else if (opset >= 1) {
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
        }
        return false;
    }
};

} // namespace

// @nnr-meta-op mt=static inplace=yes
operator_t* resolver_default_op_Slice(int opset, pool_t& pool)
{
    return pool_new<Slice_operator>(pool);
}

} // namespace nnr
