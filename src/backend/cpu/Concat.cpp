#include "nnr.h"
#include "util.h"
#include "thread_pool.h"

namespace nnr {

namespace {

struct Concat_operator : public operator_t {
    int axis;
    int caxis;

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
        // NHWC support: channel concat (axis=1) on 4D float tensors
        if (ndim == 4 && caxis == 1 && x->type == NNR_DATA_TYPE_FLOAT32)
            layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
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
