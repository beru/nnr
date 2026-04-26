#if defined(NNR_USE_CUDA)

// QLinearAveragePool on CUDA — 2D quantized average pool, one thread per
// output element. Scalar per-tensor scales/zps; NCHW only.
//
// Formula per output pixel:
//   sum = Σ (X[ih, iw] − x_zp)  over the window
//   count = number of valid positions  (or fixed kH·kW if count_include_pad)
//   y  = sat(round(sum · x_scale / count / y_scale) + y_zp)
// Pre-folded on the host:
//   cs = x_scale / y_scale     (constant per op)
// Kernel does: y = sat(round(cs * sum / count) + y_zp).

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

#include <cstdint>

namespace nnr {

operator_t* resolver_default_op_QLinearAveragePool(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* qavgpool_source() {
    return R"CUDA(
extern "C" {

#define QAVGPOOL_BODY(ELT, Y_MIN, Y_MAX)                                       \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = N * C * Ho * Wo;                                               \
    if (idx >= total) return;                                                  \
    int ow = idx % Wo; int t = idx / Wo;                                       \
    int oh = t   % Ho;     t = t   / Ho;                                       \
    int c  = t   % C;      t = t   / C;                                        \
    int n  = t;                                                                \
    const ELT* xn = x + ((size_t)n * C + c) * Hi * Wi;                         \
    int sum = 0;                                                               \
    int count = 0;                                                             \
    for (int kr = 0; kr < kH; ++kr) {                                          \
        int ih = oh * sH - pT + kr;                                            \
        bool row_ok = (ih >= 0 && ih < Hi);                                    \
        if (!row_ok) {                                                         \
            if (count_include_pad) count += kW;                                \
            continue;                                                          \
        }                                                                      \
        for (int kc = 0; kc < kW; ++kc) {                                      \
            int iw = ow * sW - pL + kc;                                        \
            if (iw < 0 || iw >= Wi) {                                          \
                if (count_include_pad) count += 1;                             \
                continue;                                                      \
            }                                                                  \
            sum += (int)xn[ih * Wi + iw] - x_zp;                               \
            count += 1;                                                        \
        }                                                                      \
    }                                                                          \
    if (count == 0) count = 1;                                                 \
    float v = cs * (float)sum / (float)count + (float)y_zp;                    \
    int q = (int)rintf(v);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    y[idx] = (ELT)q;

__global__ void qavgpool_u8(const unsigned char* __restrict__ x,
                            unsigned char* __restrict__ y,
                            int N, int C, int Hi, int Wi, int Ho, int Wo,
                            int kH, int kW, int sH, int sW,
                            int pT, int pL,
                            int x_zp, int y_zp, float cs,
                            int count_include_pad)
{ QAVGPOOL_BODY(unsigned char, 0, 255) }

__global__ void qavgpool_s8(const signed char* __restrict__ x,
                            signed char* __restrict__ y,
                            int N, int C, int Hi, int Wi, int Ho, int Wo,
                            int kH, int kW, int sH, int sW,
                            int pT, int pL,
                            int x_zp, int y_zp, float cs,
                            int count_include_pad)
{ QAVGPOOL_BODY(signed char, -128, 127) }
#undef QAVGPOOL_BODY

// NHWC variant: idx decode is (n, oh, ow, c); X read is strided by C.
#define QAVGPOOL_NHWC_BODY(ELT, Y_MIN, Y_MAX)                                  \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = N * Ho * Wo * C;                                               \
    if (idx >= total) return;                                                  \
    int c  = idx % C;             int t = idx / C;                             \
    int ow = t   % Wo;                t = t   / Wo;                            \
    int oh = t   % Ho;                t = t   / Ho;                            \
    int n  = t;                                                                \
    /* xn points to the (n, *, *, c) slab — strided spatial reads. */          \
    const ELT* xn = x + (size_t)n * Hi * Wi * C;                               \
    int sum = 0;                                                               \
    int count = 0;                                                             \
    for (int kr = 0; kr < kH; ++kr) {                                          \
        int ih = oh * sH - pT + kr;                                            \
        bool row_ok = (ih >= 0 && ih < Hi);                                    \
        if (!row_ok) {                                                         \
            if (count_include_pad) count += kW;                                \
            continue;                                                          \
        }                                                                      \
        for (int kc = 0; kc < kW; ++kc) {                                      \
            int iw = ow * sW - pL + kc;                                        \
            if (iw < 0 || iw >= Wi) {                                          \
                if (count_include_pad) count += 1;                             \
                continue;                                                      \
            }                                                                  \
            sum += (int)xn[((size_t)ih * Wi + iw) * C + c] - x_zp;             \
            count += 1;                                                        \
        }                                                                      \
    }                                                                          \
    if (count == 0) count = 1;                                                 \
    float v = cs * (float)sum / (float)count + (float)y_zp;                    \
    int q = (int)rintf(v);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    y[idx] = (ELT)q;

__global__ void qavgpool_nhwc_u8(const unsigned char* __restrict__ x,
                                 unsigned char* __restrict__ y,
                                 int N, int C, int Hi, int Wi, int Ho, int Wo,
                                 int kH, int kW, int sH, int sW,
                                 int pT, int pL,
                                 int x_zp, int y_zp, float cs,
                                 int count_include_pad)
{ QAVGPOOL_NHWC_BODY(unsigned char, 0, 255) }

__global__ void qavgpool_nhwc_s8(const signed char* __restrict__ x,
                                 signed char* __restrict__ y,
                                 int N, int C, int Hi, int Wi, int Ho, int Wo,
                                 int kH, int kW, int sH, int sW,
                                 int pT, int pL,
                                 int x_zp, int y_zp, float cs,
                                 int count_include_pad)
{ QAVGPOOL_NHWC_BODY(signed char, -128, 127) }
#undef QAVGPOOL_NHWC_BODY

} // extern "C"
)CUDA";
}

struct QLinearAveragePool_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    bool is_uint8 = true;

    int N=0, C=0, Hi=0, Wi=0, Ho=0, Wo=0;
    int kH=0, kW=0, sH=1, sW=1, pT=0, pL=0;
    int x_zp = 0, y_zp = 0;
    float cs = 0.f;
    int count_include_pad = 0;
    const char* kernel_name = nullptr;

    bool init() override {
        if (!(inputs.size() == 5 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_QLinearAveragePool(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;

        const tensor_t* X = inputs[0];
        tensor_t* Y = outputs[0];
        if (X->ndim != 4 || Y->ndim != 4) return true;
        if (X->type != Y->type) return true;
        if (X->type != NNR_DATA_TYPE_UINT8 && X->type != NNR_DATA_TYPE_INT8) return true;
        is_uint8 = (X->type == NNR_DATA_TYPE_UINT8);

        int64_t* ints = nullptr;
        int n = attribute(attr_key_t::kernel_shape, ints);
        if (n != 2) return true;
        kH = (int)ints[0]; kW = (int)ints[1];
        n = attribute(attr_key_t::strides, ints);
        sH = (n >= 1) ? (int)ints[0] : 1;
        sW = (n >= 2) ? (int)ints[1] : 1;
        n = attribute(attr_key_t::pads, ints);
        pT = (n >= 2) ? (int)ints[0] : 0;
        pL = (n >= 2) ? (int)ints[1] : 0;

        auto is_scalar_f32 = [](const tensor_t* t) {
            return t && t->type == NNR_DATA_TYPE_FLOAT32 && t->ndata == 1 && t->data;
        };
        if (!is_scalar_f32(inputs[1]) || !is_scalar_f32(inputs[3])) return true;

        float x_scale = *(const float*)inputs[1]->data;
        float y_scale = *(const float*)inputs[3]->data;
        if (y_scale == 0.f) return true;
        cs = x_scale / y_scale;

        auto read_zp = [](const tensor_t* t) -> int {
            if (!t || t->ndata == 0 || !t->data) return 0;
            if (t->type == NNR_DATA_TYPE_UINT8) return (int)*(const uint8_t*)t->data;
            if (t->type == NNR_DATA_TYPE_INT8)  return (int)*(const int8_t*)t->data;
            return INT32_MIN;
        };
        x_zp = read_zp(inputs[2]);
        y_zp = read_zp(inputs[4]);
        if (x_zp == INT32_MIN || y_zp == INT32_MIN) return true;

        N = X->dims[0]; C = X->dims[1]; Hi = X->dims[2]; Wi = X->dims[3];
        Ho = Y->dims[2]; Wo = Y->dims[3];
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
            ? (is_uint8 ? "qavgpool_nhwc_u8" : "qavgpool_nhwc_s8")
            : (is_uint8 ? "qavgpool_u8"      : "qavgpool_s8");
        CUfunction f = be->nvrtc.get("nnr_qavgpool", qavgpool_source(),
                                     kname, gpu::nvrtc_arch_option(be->device));
        if (!f) return fallback->exec();

        void* d_x = be->cache->ensure_device(inputs[0]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) return fallback->exec();

        int _N=N, _C=C, _Hi=Hi, _Wi=Wi, _Ho=Ho, _Wo=Wo;
        int _kH=kH, _kW=kW, _sH=sH, _sW=sW, _pT=pT, _pL=pL;
        int _xz=x_zp, _yz=y_zp;
        float _cs=cs;
        int _cip=count_include_pad;
        void* args[] = { &d_x, &d_y, &_N, &_C, &_Hi, &_Wi, &_Ho, &_Wo,
                         &_kH, &_kW, &_sH, &_sW, &_pT, &_pL,
                         &_xz, &_yz, &_cs, &_cip };
        unsigned long long total = (unsigned long long)N * C * Ho * Wo;
        unsigned block = 256;
        unsigned grid  = (unsigned)((total + block - 1) / block);
        if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args))
            return fallback->exec();
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_QLinearAveragePool(int opset, pool_t& pool) {
    return pool_new<QLinearAveragePool_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
