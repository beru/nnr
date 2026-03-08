#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct SpaceToDepth_operator : public operator_t {
    int blocksize;

    bool init() override {
        if (inputs.empty() || outputs.empty()) return false;
        blocksize = attribute(attr_key_t::blocksize, (int32_t)0);
        if (blocksize <= 0) return false;
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        // Input: [N, C, H, W] -> Output: [N, C*bs*bs, H/bs, W/bs]
        int bs = blocksize;
        small_vector<int> dims(4);
        dims[0] = x->dims[0];
        dims[1] = x->dims[1] * bs * bs;
        dims[2] = x->dims[2] / bs;
        dims[3] = x->dims[3] / bs;
        return y->reshape(dims, x->type);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;

        int N = x->dims[0], C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int bs = blocksize;
        int oH = H / bs, oW = W / bs;

        // SpaceToDepth is the inverse of DepthToSpace (DCR mode)
        // DCR: output channel = bh * bs * C + bw * C + c
        int oC = C * bs * bs;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < oH; ++oh) {
                    for (int ow = 0; ow < oW; ++ow) {
                        for (int bh = 0; bh < bs; ++bh) {
                            for (int bw = 0; bw < bs; ++bw) {
                                int ih = oh * bs + bh;
                                int iw = ow * bs + bw;
                                int oc = bh * bs * C + bw * C + c;
                                py[((n * oC + oc) * oH + oh) * oW + ow] =
                                    px[((n * C + c) * H + ih) * W + iw];
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<SpaceToDepth_operator,
            int8_t, int16_t, int32_t, int64_t,
            uint8_t, uint16_t, uint32_t, uint64_t,
            float16_t, bfloat16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_SpaceToDepth(int opset, pool_t& pool) { return pool_new<SpaceToDepth_operator>(pool); }

} // namespace nnr
