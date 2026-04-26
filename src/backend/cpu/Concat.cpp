#include "nnr.h"
#include "util.h"
#include "thread_pool.h"

namespace nnr {

namespace {

struct Concat_operator : public operator_t {
    int axis;
    int caxis;
    // True when each input occupies one contiguous byte run in the output
    // under NHWC physical ordering [N, H, W, C]. Set in reshape() based on
    // dims and caxis; exec_impl uses a single-memcpy-per-input fast-path
    // when y->format == NHWC and this is set. Required because the existing
    // nblocks-loop path uses logical dim strides that mis-stride NHWC bytes
    // for caxis != 1.
    bool nhwc_single_contiguous = false;

    bool init() override {
        if (!(inputs.size() >= 1 && outputs.size() == 1)) {
            return false;
        }
        axis = attribute(attr_key_t::axis, 1);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int ndim = x->ndim;
        small_vector<int> dims(ndim);
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
        // Initialize dims from first input (required for single-input Concat,
        // since the loop below starts at i=1 and wouldn't populate dims at all).
        for (int j = 0; j < ndim; ++j)
            dims[j] = x->dims[j];
        int s = x->dims[caxis];
        for (size_t i = 1; i < inputs.size(); ++i) {
            const int* pdims = &inputs[i]->dims[0];
            for (int j = 0; j < ndim; ++j) {
                if (j == caxis) {
                    s += pdims[j];
                }else if (x->dims[j] != pdims[j]) {
                    return false;
                }
                dims[j] = pdims[j];
            }
        }
        dims[caxis] = s;

        layout_mask = LAYOUT_NCHW;
        // NHWC support requires exec_impl to produce correct NHWC bytes.
        // Two cases work:
        //   (a) caxis=1 channel concat — strided memcpy below at line ~93,
        //       handles arbitrary N/H/W with C-interleaved per spatial.
        //   (b) any caxis where each input is a single contiguous byte run
        //       in NHWC physical order — exec_impl falls into the
        //       single-memcpy-per-input fast-path. Holds when the product of
        //       *physical* NHWC dims [N, H, W, C] before the concat-axis's
        //       physical position is 1:
        //         caxis=0(N): outer=1 always
        //         caxis=1(C): outer=N*H*W (this case overlaps with (a))
        //         caxis=2(H): outer=N
        //         caxis=3(W): outer=N*H
        //   memory_planner uses the same physical-outer rule to gate
        //   Concat-alias on NHWC inputs.
        size_t nhwc_outer = 1;
        if (ndim == 4) {
            switch (caxis) {
            case 0: nhwc_outer = 1; break;
            case 1: nhwc_outer = (size_t)dims[0] * dims[2] * dims[3]; break;
            case 2: nhwc_outer = (size_t)dims[0]; break;
            case 3: nhwc_outer = (size_t)dims[0] * dims[2]; break;
            }
        }
        nhwc_single_contiguous = (ndim == 4) && (nhwc_outer == 1);
        if (ndim == 4 && caxis == 1 && x->type == NNR_DATA_TYPE_FLOAT32)
            layout_mask |= LAYOUT_NHWC;
        if (nhwc_single_contiguous)
            layout_mask |= LAYOUT_NHWC;
        if (!y->reshape(dims, x->type)) return false;
        // Concat is quantization-transparent: propagate if all inputs share same quant params
        if (x->is_quantized()) {
            bool same = true;
            for (size_t i = 1; i < inputs.size() && same; i++)
                same = inputs[i]->quant_scale == x->quant_scale
                    && inputs[i]->quant_zero_point == x->quant_zero_point;
            if (same) y->set_quant(x->quant_scale, x->quant_zero_point);
        }
        return true;
    }

    bool exec_impl() {
        tensor_t* y = outputs[0];
        if (!y->data || y->ndata == 0) return true;  // empty output — nothing to do

        // Zero-copy: if memory planner aliased inputs into the output buffer,
        // all data is already in the right place — nothing to copy.
        if (!inputs.empty() && inputs[0]->data != nullptr) {
            const size_t sz = data_type_sizeof(inputs[0]);
            char* py = (char*)y->data;
            size_t offset = 0;
            bool all_aliased = true;
            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                const tensor_t* x = inputs[idx];
                if ((const char*)x->data != py + offset) {
                    all_aliased = false;
                    break;
                }
                offset += x->ndata * sz;
            }
            if (all_aliased && offset == y->ndata * sz)
                return true;
        }

        // Strided alias (NHWC channel-axis with H>1 or W>1): each input's data
        // pointer lands inside `y` at start_C*elem_sz, and the producer Conv
        // wrote with ldc = parent C. Producers and consumer Concat share the
        // parent buffer; Concat is a no-op.
        if (y->format == memory_layout_t::NHWC && y->ndim == 4 && caxis == 1) {
            bool all_strided_aliased = !inputs.empty();
            for (auto* x : inputs) {
                if (!x || x->concat_parent != y || !x->strides_set) {
                    all_strided_aliased = false;
                    break;
                }
            }
            if (all_strided_aliased) return true;
        }

        // NHWC single-contiguous fast-path: each input is one contiguous
        // byte run in physical NHWC order. Layout-agnostic memcpy works.
        // Covers caxis=0 always; caxis=2 N=1; caxis=3 N=H=1; caxis=1 N=H=W=1.
        if (y->format == memory_layout_t::NHWC && nhwc_single_contiguous) {
            char* py = (char*)y->data;
            const size_t sz = data_type_sizeof(inputs[0]);
            size_t offset_bytes = 0;
            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                const tensor_t* x = inputs[idx];
                size_t bytes = (size_t)x->ndata * sz;
                memcpy(py + offset_bytes, x->data, bytes);
                offset_bytes += bytes;
            }
            return true;
        }

        // NHWC channel concat: axis=1 in NCHW → innermost dim in NHWC
        if (y->format == memory_layout_t::NHWC && y->ndim == 4 && caxis == 1) {
            char* py = (char*)y->data;
            const size_t sz = data_type_sizeof(inputs[0]);
            int yC = y->dims[1];
            int NHW = y->dims[0] * y->dims[2] * y->dims[3];
            int c_off = 0;
            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                const tensor_t* x = inputs[idx];
                int xC = x->dims[1];
                const char* px = (const char*)x->data;
                size_t xC_bytes = (size_t)xC * sz;
                size_t yC_bytes = (size_t)yC * sz;
                for (int s = 0; s < NHW; ++s)
                    memcpy(py + s * yC_bytes + (size_t)c_off * sz,
                        px + s * xC_bytes, xC_bytes);
                c_off += xC;
            }
            return true;
        }

        if (inputs[0]->type == NNR_DATA_TYPE_STRING) {
            std::string* py = (std::string*)y->data;
            int ypitch = std::reduce(y->dims + caxis, y->dims + y->ndim, 1, std::multiplies<>{});
            int ybase = 0;
            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                const tensor_t* x = inputs[idx];
                const std::string* px = (const std::string*)x->data;
                int xpitch = std::reduce(x->dims + caxis, x->dims + x->ndim, 1, std::multiplies<>{});
                for (int o = 0, j = 0, k = ybase, l = x->ndata; o < l; ++o) {
                    py[k + o] = px[o];
                    if (++j == xpitch) {
                        k += (ypitch - xpitch);
                        j = 0;
                    }
                }
                ybase += xpitch;
            }
        }else {
            char* py = (char*)y->data;
            const size_t sz = data_type_sizeof(inputs[0]);
            int ypitch = std::reduce(y->dims + caxis, y->dims + y->ndim, 1, std::multiplies<>{});
            int ybase = 0;
            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                const tensor_t* x = inputs[idx];
                const char* px = (const char*)x->data;
                int xpitch = std::reduce(x->dims + caxis, x->dims + x->ndim, 1, std::multiplies<>{});
                if (xpitch > 0 && x->ndata > 0) {
                    size_t xbytes = (size_t)xpitch * sz;
                    size_t ybytes = (size_t)ypitch * sz;
                    int nblocks = (int)(x->ndata / xpitch);
                    char* dst = py + (size_t)ybase * sz;
                    nnr::for_static(0, nblocks, nblocks > 16, [&](int b) {
                        memcpy(dst + (size_t)b * ybytes,
                               px + (size_t)b * xbytes, xbytes);
                    });
                }
                ybase += xpitch;
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
        }else if (opset >= 11) {
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
        }else if (opset >= 4) {
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

} // namespace {

// @nnr-meta-op mt=static layout=[NCHW,NHWC]
operator_t* resolver_default_op_Concat(int opset, pool_t& pool) { return pool_new<Concat_operator>(pool); }

} // namespace nnr
