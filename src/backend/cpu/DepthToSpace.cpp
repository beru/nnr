#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct DepthToSpace_operator : public operator_t {
    int blocksize;
    std::string_view mode;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        blocksize = attribute(attr_key_t::blocksize, (int32_t)0);
        if (blocksize <= 0) {
            return false;
        }
        mode = attribute(attr_key_t::mode, "DCR");
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        if (x->ndim != 4) {
            return false;
        }

        int N = x->dims[0];
        int C = x->dims[1];
        int H = x->dims[2];
        int W = x->dims[3];
        int bs2 = blocksize * blocksize;

        if (C % bs2 != 0) {
            return false;
        }

        small_vector<int> dims(4);
        dims[0] = N;
        dims[1] = C / bs2;
        dims[2] = H * blocksize;
        dims[3] = W * blocksize;
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        int N = x->dims[0];
        int C = x->dims[1];
        int H = x->dims[2];
        int W = x->dims[3];
        int bs = blocksize;
        int bs2 = bs * bs;
        int oC = C / bs2;
        int oH = H * bs;
        int oW = W * bs;

        // Output strides
        int y_stride_n = oC * oH * oW;
        int y_stride_c = oH * oW;
        int y_stride_h = oW;

        // Input strides
        int x_stride_n = C * H * W;
        int x_stride_c = H * W;
        int x_stride_h = W;

        bool is_dcr = (mode == "DCR");

        for (int n = 0; n < N; ++n) {
            for (int oc = 0; oc < oC; ++oc) {
                for (int oh = 0; oh < oH; ++oh) {
                    for (int ow = 0; ow < oW; ++ow) {
                        int h = oh / bs;
                        int bh = oh % bs;
                        int w = ow / bs;
                        int bw = ow % bs;
                        int ic;
                        if (is_dcr) {
                            // DCR: input channel = bh * bs * oC + bw * oC + oc
                            ic = bh * bs * oC + bw * oC + oc;
                        }else {
                            // CRD: input channel = oc * bs * bs + bh * bs + bw
                            ic = oc * bs2 + bh * bs + bw;
                        }
                        int src = n * x_stride_n + ic * x_stride_c + h * x_stride_h + w;
                        int dst = n * y_stride_n + oc * y_stride_c + oh * y_stride_h + ow;
                        py[dst] = px[src];
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<DepthToSpace_operator,
            opset_t<13, bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double, bfloat16_t,
                std::complex<float>, std::complex<double>,
                std::string>,
            opset_t<1, bool_t,
                uint8_t, uint16_t, uint32_t, uint64_t,
                int8_t, int16_t, int32_t, int64_t,
                float16_t, float, double,
                std::complex<float>, std::complex<double>,
                std::string>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_DepthToSpace(int opset, pool_t& pool)
{
    return pool_new<DepthToSpace_operator>(pool);
}

} // namespace nnr
