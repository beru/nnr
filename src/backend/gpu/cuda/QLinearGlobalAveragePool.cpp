#if defined(NNR_USE_CUDA)

// QLinearGlobalAveragePool on CUDA — reduces all spatial dims to 1×1 per
// (n, c). Formula: Y[n,c] = sat(rint(sum_spatial(X - x_zp) * rs) + y_zp)
// where rs = x_scale / (y_scale * spatial).
//
// Launch: one block per (n, c). 256 threads/block. Warp-reduce across lanes,
// block-reduce across warps via shared mem. Keeping GAP on device removes
// the D2H/H2D shuttle that sits between backbone Convs and the classifier
// head on int8 classification models.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"

#include <cstdint>

namespace nnr {

operator_t* resolver_default_op_QLinearGlobalAveragePool(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* qgap_source() {
    return R"CUDA(
extern "C" {

#define QGAP_BODY(ELT, Y_MIN, Y_MAX)                                           \
    __shared__ int warp_sums[8];                                               \
    int nc = blockIdx.x;                                                       \
    const ELT* src = X + (size_t)nc * spatial;                                 \
    int tid = threadIdx.x;                                                     \
    int lane = tid & 31;                                                       \
    int wid = tid >> 5;                                                        \
    int sum = 0;                                                               \
    for (int s = tid; s < spatial; s += 256)                                   \
        sum += (int)src[s];                                                    \
    for (int off = 16; off > 0; off >>= 1)                                     \
        sum += __shfl_xor_sync(0xffffffff, sum, off);                          \
    if (lane == 0) warp_sums[wid] = sum;                                       \
    __syncthreads();                                                           \
    if (wid == 0) {                                                            \
        int v = (lane < 8) ? warp_sums[lane] : 0;                              \
        for (int off = 4; off > 0; off >>= 1)                                  \
            v += __shfl_xor_sync(0xffffffff, v, off);                          \
        if (lane == 0) {                                                       \
            v -= x_zp * spatial;                                               \
            int q = (int)rintf((float)v * rs) + y_zp;                          \
            if (q < (Y_MIN)) q = (Y_MIN);                                      \
            if (q > (Y_MAX)) q = (Y_MAX);                                      \
            Y[nc] = (ELT)q;                                                    \
        }                                                                      \
    }

__global__ void qgap_u8(const unsigned char* __restrict__ X,
                        unsigned char* __restrict__ Y,
                        int spatial, int x_zp, int y_zp, float rs)
{
    QGAP_BODY(unsigned char, 0, 255)
}

__global__ void qgap_s8(const signed char* __restrict__ X,
                        signed char* __restrict__ Y,
                        int spatial, int x_zp, int y_zp, float rs)
{
    QGAP_BODY(signed char, -128, 127)
}
#undef QGAP_BODY

// NHWC variant. Layout in memory: X[n*spatial*C + s*C + c]. Block-per-(n,c)
// reduction needs strided spatial access: X[(n*spatial + s)*C + c].
#define QGAP_NHWC_BODY(ELT, Y_MIN, Y_MAX)                                      \
    __shared__ int warp_sums[8];                                               \
    int nc = blockIdx.x;                                                       \
    int n  = nc / C;                                                           \
    int c  = nc % C;                                                           \
    const ELT* base = X + (size_t)n * spatial * C + c;                         \
    int tid = threadIdx.x;                                                     \
    int lane = tid & 31;                                                       \
    int wid = tid >> 5;                                                        \
    int sum = 0;                                                               \
    for (int s = tid; s < spatial; s += 256)                                   \
        sum += (int)base[(size_t)s * C];                                       \
    for (int off = 16; off > 0; off >>= 1)                                     \
        sum += __shfl_xor_sync(0xffffffff, sum, off);                          \
    if (lane == 0) warp_sums[wid] = sum;                                       \
    __syncthreads();                                                           \
    if (wid == 0) {                                                            \
        int v = (lane < 8) ? warp_sums[lane] : 0;                              \
        for (int off = 4; off > 0; off >>= 1)                                  \
            v += __shfl_xor_sync(0xffffffff, v, off);                          \
        if (lane == 0) {                                                       \
            v -= x_zp * spatial;                                               \
            int q = (int)rintf((float)v * rs) + y_zp;                          \
            if (q < (Y_MIN)) q = (Y_MIN);                                      \
            if (q > (Y_MAX)) q = (Y_MAX);                                      \
            /* NHWC output Y[n,1,1,c] = Y[n*C + c] */                          \
            Y[nc] = (ELT)q;                                                    \
        }                                                                      \
    }

__global__ void qgap_nhwc_u8(const unsigned char* __restrict__ X,
                             unsigned char* __restrict__ Y,
                             int spatial, int C, int x_zp, int y_zp, float rs)
{
    QGAP_NHWC_BODY(unsigned char, 0, 255)
}

__global__ void qgap_nhwc_s8(const signed char* __restrict__ X,
                             signed char* __restrict__ Y,
                             int spatial, int C, int x_zp, int y_zp, float rs)
{
    QGAP_NHWC_BODY(signed char, -128, 127)
}
#undef QGAP_NHWC_BODY

} // extern "C"
)CUDA";
}

struct QLinearGlobalAveragePool_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    bool is_uint8 = true;
    int N = 0, C = 0, spatial = 0;
    int x_zp = 0, y_zp = 0;
    float rs = 0.f;
    const char* kernel_name = nullptr;

    bool init() override {
        if (!(inputs.size() == 5 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_QLinearGlobalAveragePool(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;

        const tensor_t* X = inputs[0];
        if (X->ndim < 3) return true;
        if (X->type != NNR_DATA_TYPE_UINT8 && X->type != NNR_DATA_TYPE_INT8) return true;
        if (X->type != outputs[0]->type) return true;
        is_uint8 = (X->type == NNR_DATA_TYPE_UINT8);

        const tensor_t* xs = inputs[1];
        const tensor_t* ys = inputs[3];
        if (xs->type != NNR_DATA_TYPE_FLOAT32 || xs->ndata != 1 || !xs->data) return true;
        if (ys->type != NNR_DATA_TYPE_FLOAT32 || ys->ndata != 1 || !ys->data) return true;
        float x_scale = *(const float*)xs->data;
        float y_scale = *(const float*)ys->data;
        if (y_scale == 0.f) return true;

        auto read_zp = [](const tensor_t* t) -> int {
            if (!t || t->ndata == 0 || !t->data) return 0;
            if (t->type == NNR_DATA_TYPE_UINT8) return (int)*(const uint8_t*)t->data;
            if (t->type == NNR_DATA_TYPE_INT8)  return (int)*(const int8_t*)t->data;
            return INT32_MIN;
        };
        if (inputs[2] && inputs[2]->ndata > 1) return true;
        if (inputs[4] && inputs[4]->ndata > 1) return true;
        x_zp = read_zp(inputs[2]);
        y_zp = read_zp(inputs[4]);
        if (x_zp == INT32_MIN || y_zp == INT32_MIN) return true;

        N = X->dims[0]; C = X->dims[1];
        spatial = 1;
        for (int d = 2; d < X->ndim; ++d) spatial *= X->dims[d];
        if (spatial <= 0) return true;
        rs = x_scale / (y_scale * (float)spatial);
        kernel_name = nullptr; // resolved at exec() based on layout

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) return fallback->exec();

        bool is_nhwc = (outputs[0]->format == memory_layout_t::NHWC);
        const char* kname = is_nhwc
            ? (is_uint8 ? "qgap_nhwc_u8" : "qgap_nhwc_s8")
            : (is_uint8 ? "qgap_u8"      : "qgap_s8");
        CUfunction f = be->nvrtc.get("nnr_qgap", qgap_source(),
                                     kname,
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) return fallback->exec();

        void* d_x = be->cache->ensure_device(inputs[0]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) return fallback->exec();

        int _sp = spatial, _xz = x_zp, _yz = y_zp, _C = C;
        float _rs = rs;
        unsigned grid = (unsigned)(N * C);
        unsigned block = 256;
        if (is_nhwc) {
            void* args[] = { &d_x, &d_y, &_sp, &_C, &_xz, &_yz, &_rs };
            if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args))
                return fallback->exec();
        } else {
            void* args[] = { &d_x, &d_y, &_sp, &_xz, &_yz, &_rs };
            if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args))
                return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_QLinearGlobalAveragePool(int opset, pool_t& pool) {
    return pool_new<QLinearGlobalAveragePool_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
