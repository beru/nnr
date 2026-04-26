#if defined(NNR_USE_CUDA)

// Pooling ops on CUDA via NVRTC:
//   GlobalAveragePool — block-per-channel; threads sum over H*W then divide.
//   AveragePool / MaxPool — thread-per-output-pixel 2D sliding window.
// Float32 only; NCHW only.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_GlobalAveragePool(int opset, pool_t& pool);
operator_t* resolver_default_op_AveragePool      (int opset, pool_t& pool);
operator_t* resolver_default_op_MaxPool          (int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

// --------------------------------------------------------------------
// NVRTC kernel source
// --------------------------------------------------------------------

static const char* pool_source() {
    return R"CUDA(
#ifndef INFINITY
#define INFINITY __int_as_float(0x7F800000)
#endif
extern "C" {

// Block per (n, c). Threads cooperatively sum HW elements, then thread 0
// writes the mean. Grid.x = N*C, block = fixed (typ. 256).
__global__ void gap_f32(const float* __restrict__ x,
                        float* __restrict__ y,
                        int HW)
{
    extern __shared__ float smem[];
    const int nc  = blockIdx.x;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const float* xc = x + (size_t)nc * HW;

    float acc = 0.f;
    for (int j = tid; j < HW; j += nth) acc += xc[j];
    smem[tid] = acc;
    __syncthreads();

    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    if (tid == 0) y[nc] = smem[0] / (float)HW;
}

// Generic 2D AveragePool / MaxPool. Thread per output element.
// NCHW layout. Kernel kH x kW, stride sH x sW, pad pT/pL at top/left,
// dilation dH/dW. count_include_pad for avg: 0 = exclude, 1 = include.
__global__ void avgpool_f32(const float* __restrict__ x,
                            float* __restrict__ y,
                            int N, int C, int Hi, int Wi, int Ho, int Wo,
                            int kH, int kW, int sH, int sW,
                            int pT, int pL, int dH, int dW,
                            int count_include_pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Ho * Wo;
    if (idx >= total) return;

    int ow = idx % Wo; int t = idx / Wo;
    int oh = t   % Ho;     t = t   / Ho;
    int c  = t   % C;      t = t   / C;
    int n  = t;

    const float* xn = x + ((size_t)n * C + c) * Hi * Wi;
    float sum = 0.f;
    int count = 0;

    for (int kr = 0; kr < kH; ++kr) {
        int ih = oh * sH - pT + kr * dH;
        if (ih < 0 || ih >= Hi) {
            if (count_include_pad) count += kW;
            continue;
        }
        for (int kc = 0; kc < kW; ++kc) {
            int iw = ow * sW - pL + kc * dW;
            if (iw < 0 || iw >= Wi) {
                if (count_include_pad) ++count;
                continue;
            }
            sum += xn[ih * Wi + iw];
            ++count;
        }
    }
    y[idx] = (count > 0) ? (sum / (float)count) : 0.f;
}

__global__ void maxpool_f32(const float* __restrict__ x,
                            float* __restrict__ y,
                            int N, int C, int Hi, int Wi, int Ho, int Wo,
                            int kH, int kW, int sH, int sW,
                            int pT, int pL, int dH, int dW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Ho * Wo;
    if (idx >= total) return;

    int ow = idx % Wo; int t = idx / Wo;
    int oh = t   % Ho;     t = t   / Ho;
    int c  = t   % C;      t = t   / C;
    int n  = t;

    const float* xn = x + ((size_t)n * C + c) * Hi * Wi;
    float best = -INFINITY;
    for (int kr = 0; kr < kH; ++kr) {
        int ih = oh * sH - pT + kr * dH;
        if (ih < 0 || ih >= Hi) continue;
        for (int kc = 0; kc < kW; ++kc) {
            int iw = ow * sW - pL + kc * dW;
            if (iw < 0 || iw >= Wi) continue;
            float v = xn[ih * Wi + iw];
            if (v > best) best = v;
        }
    }
    y[idx] = best;
}

// MaxPool on int8/uint8 — element-wise max commutes with the quantization
// transform (y = (x − zp) * scale is monotonic), so max over quantized inputs
// equals the quantized version of max over dequantized inputs. Out-of-bounds
// positions get the lowest representable value of the dtype so OOB is never
// selected. Matches ONNX MaxPool semantics for integer inputs.
#define MAXPOOL_INT_BODY(ELT, ELT_MIN)                                         \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = N * C * Ho * Wo;                                               \
    if (idx >= total) return;                                                  \
    int ow = idx % Wo; int t = idx / Wo;                                       \
    int oh = t   % Ho;     t = t   / Ho;                                       \
    int c  = t   % C;      t = t   / C;                                        \
    int n  = t;                                                                \
    const ELT* xn = x + ((size_t)n * C + c) * Hi * Wi;                         \
    int best = (int)(ELT_MIN);                                                 \
    for (int kr = 0; kr < kH; ++kr) {                                          \
        int ih = oh * sH - pT + kr * dH;                                       \
        if (ih < 0 || ih >= Hi) continue;                                      \
        for (int kc = 0; kc < kW; ++kc) {                                      \
            int iw = ow * sW - pL + kc * dW;                                   \
            if (iw < 0 || iw >= Wi) continue;                                  \
            int v = (int)xn[ih * Wi + iw];                                     \
            if (v > best) best = v;                                            \
        }                                                                      \
    }                                                                          \
    y[idx] = (ELT)best;

__global__ void maxpool_s8(const signed char* __restrict__ x,
                           signed char* __restrict__ y,
                           int N, int C, int Hi, int Wi, int Ho, int Wo,
                           int kH, int kW, int sH, int sW,
                           int pT, int pL, int dH, int dW)
{ MAXPOOL_INT_BODY(signed char, -128) }

__global__ void maxpool_u8(const unsigned char* __restrict__ x,
                           unsigned char* __restrict__ y,
                           int N, int C, int Hi, int Wi, int Ho, int Wo,
                           int kH, int kW, int sH, int sW,
                           int pT, int pL, int dH, int dW)
{ MAXPOOL_INT_BODY(unsigned char, 0) }
#undef MAXPOOL_INT_BODY

// NHWC variants: idx decoded as (n, oh, ow, c); X read is strided by C.
#define MAXPOOL_NHWC_BODY(ELT, ELT_MIN)                                        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int total = N * Ho * Wo * C;                                               \
    if (idx >= total) return;                                                  \
    int c  = idx % C;             int t = idx / C;                             \
    int ow = t   % Wo;                t = t   / Wo;                            \
    int oh = t   % Ho;                t = t   / Ho;                            \
    int n  = t;                                                                \
    const ELT* xn = x + (size_t)n * Hi * Wi * C;                               \
    int best = (int)(ELT_MIN);                                                 \
    for (int kr = 0; kr < kH; ++kr) {                                          \
        int ih = oh * sH - pT + kr * dH;                                       \
        if (ih < 0 || ih >= Hi) continue;                                      \
        for (int kc = 0; kc < kW; ++kc) {                                      \
            int iw = ow * sW - pL + kc * dW;                                   \
            if (iw < 0 || iw >= Wi) continue;                                  \
            int v = (int)xn[((size_t)ih * Wi + iw) * C + c];                   \
            if (v > best) best = v;                                            \
        }                                                                      \
    }                                                                          \
    y[idx] = (ELT)best;

__global__ void maxpool_nhwc_s8(const signed char* __restrict__ x,
                                signed char* __restrict__ y,
                                int N, int C, int Hi, int Wi, int Ho, int Wo,
                                int kH, int kW, int sH, int sW,
                                int pT, int pL, int dH, int dW)
{ MAXPOOL_NHWC_BODY(signed char, -128) }

__global__ void maxpool_nhwc_u8(const unsigned char* __restrict__ x,
                                unsigned char* __restrict__ y,
                                int N, int C, int Hi, int Wi, int Ho, int Wo,
                                int kH, int kW, int sH, int sW,
                                int pT, int pL, int dH, int dW)
{ MAXPOOL_NHWC_BODY(unsigned char, 0) }

__global__ void maxpool_nhwc_f32(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int N, int C, int Hi, int Wi, int Ho, int Wo,
                                 int kH, int kW, int sH, int sW,
                                 int pT, int pL, int dH, int dW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Ho * Wo * C;
    if (idx >= total) return;
    int c  = idx % C;             int t = idx / C;
    int ow = t   % Wo;                t = t   / Wo;
    int oh = t   % Ho;                t = t   / Ho;
    int n  = t;
    const float* xn = x + (size_t)n * Hi * Wi * C;
    float best = -INFINITY;
    for (int kr = 0; kr < kH; ++kr) {
        int ih = oh * sH - pT + kr * dH;
        if (ih < 0 || ih >= Hi) continue;
        for (int kc = 0; kc < kW; ++kc) {
            int iw = ow * sW - pL + kc * dW;
            if (iw < 0 || iw >= Wi) continue;
            float v = xn[((size_t)ih * Wi + iw) * C + c];
            if (v > best) best = v;
        }
    }
    y[idx] = best;
}
#undef MAXPOOL_NHWC_BODY

} // extern "C"
)CUDA";
}

static CUfunction get_pool_kernel(gpu::cuda_backend_t* be, const char* name) {
    return be->nvrtc.get("nnr_pool_f32",
                         pool_source(),
                         name,
                         gpu::nvrtc_arch_option(be->device));
}

// --------------------------------------------------------------------
// GlobalAveragePool
// --------------------------------------------------------------------

struct GlobalAveragePool_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int N = 0, C = 0, HW = 0;

    bool init() override {
        fallback = resolver_default_op_GlobalAveragePool(opset, ctx->attr_pool);
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false;
        if (inputs.size() == 1 && outputs.size() == 1) {
            const tensor_t* x = inputs[0];
            prim_valid = (x->type == NNR_DATA_TYPE_FLOAT32 && x->ndim == 4);
            if (prim_valid) {
                N = x->dims[0]; C = x->dims[1];
                HW = x->dims[2] * x->dims[3];
            }
        }
        device_tag = prim_valid ? static_cast<uint8_t>(backend_t::CUDA) : 0;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f = get_pool_kernel(be, "gap_f32");
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned block = 256;
        while ((int)block > HW && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = block * sizeof(float);
        int _HW = HW;
        void* args[] = { &d_x, &d_y, &_HW };
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)(N * C), 1, 1,
                               block, 1, 1, args, shared)) {
            return fallback->exec();
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// --------------------------------------------------------------------
// AveragePool / MaxPool
// --------------------------------------------------------------------

struct Pool2d_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;
    bool is_avg = false;
    int count_include_pad = 0;

    int N=0, C=0, Hi=0, Wi=0, Ho=0, Wo=0;
    int kH=0, kW=0, sH=1, sW=1, pT=0, pL=0, dH=1, dW=1;

    virtual operator_t* make_fallback() = 0;

    bool init() override {
        fallback = make_fallback();
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false;
        device_tag = 0;
        // f32 supported for both Max/Avg; int8/uint8 only for MaxPool (AvgPool
        // on quantized tensors goes through QLinearAveragePool instead).
        if (inputs.size() != 1 || outputs.size() != 1) return true;
        const tensor_t* x = inputs[0];
        const tensor_t* y = outputs[0];
        if (x->ndim != 4 || y->ndim != 4) return true;
        if (x->type != y->type) return true;
        bool is_int8  = (x->type == NNR_DATA_TYPE_INT8);
        bool is_uint8 = (x->type == NNR_DATA_TYPE_UINT8);
        if (x->type != NNR_DATA_TYPE_FLOAT32 && !is_int8 && !is_uint8) return true;
        if ((is_int8 || is_uint8) && is_avg) return true;  // no int AvgPool
        // Kernel name resolved at exec() based on output->format. Both NCHW
        // and NHWC variants are available for all dtypes (MaxPool & f32-Avg).
        kernel_name = nullptr;

        int64_t* ints = nullptr;
        int n = attribute(attr_key_t::kernel_shape, ints);
        if (n != 2) return true;
        kH = (int)ints[0]; kW = (int)ints[1];

        sH = sW = 1;
        n = attribute(attr_key_t::strides, ints);
        if (n == 2) { sH = (int)ints[0]; sW = (int)ints[1]; }

        dH = dW = 1;
        n = attribute(attr_key_t::dilations, ints);
        if (n == 2) { dH = (int)ints[0]; dW = (int)ints[1]; }

        pT = pL = 0;
        n = attribute(attr_key_t::pads, ints);
        if (n >= 2) { pT = (int)ints[0]; pL = (int)ints[1]; }

        if (is_avg) {
            count_include_pad = (int)attribute(attr_key_t::count_include_pad, (int64_t)0);
        }

        N = x->dims[0]; C = x->dims[1]; Hi = x->dims[2]; Wi = x->dims[3];
        Ho = y->dims[2]; Wo = y->dims[3];
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        // Only MaxPool has NHWC variants for all dtypes; f32 AvgPool stays
        // NCHW-only for now (chains end at AvgPool boundary anyway).
        if (!is_avg) {
            layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
        }
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        bool is_nhwc = (outputs[0]->format == memory_layout_t::NHWC);
        const tensor_t* x = inputs[0];
        const char* kname;
        if (is_avg) {
            kname = "avgpool_f32";
        } else {
            bool is_s8 = (x->type == NNR_DATA_TYPE_INT8);
            bool is_u8 = (x->type == NNR_DATA_TYPE_UINT8);
            if (is_nhwc) {
                kname = is_s8 ? "maxpool_nhwc_s8"
                              : is_u8 ? "maxpool_nhwc_u8" : "maxpool_nhwc_f32";
            } else {
                kname = is_s8 ? "maxpool_s8"
                              : is_u8 ? "maxpool_u8" : "maxpool_f32";
            }
        }
        CUfunction f = get_pool_kernel(be, kname);
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned block = 256;
        unsigned long long total = (unsigned long long)N * C * Ho * Wo;
        unsigned grid = (unsigned)((total + block - 1) / block);
        bool ok;
        if (is_avg) {
            void* args[] = { &d_x, &d_y, &N, &C, &Hi, &Wi, &Ho, &Wo,
                             &kH, &kW, &sH, &sW, &pT, &pL, &dH, &dW,
                             &count_include_pad };
            ok = gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args);
        } else {
            void* args[] = { &d_x, &d_y, &N, &C, &Hi, &Wi, &Ho, &Wo,
                             &kH, &kW, &sH, &sW, &pT, &pL, &dH, &dW };
            ok = gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args);
        }
        if (!ok) { return fallback->exec(); }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct AveragePool_cuda : Pool2d_cuda {
    AveragePool_cuda() { kernel_name = "avgpool_f32"; is_avg = true; }
    operator_t* make_fallback() override { return resolver_default_op_AveragePool(opset, ctx->attr_pool); }
};

struct MaxPool_cuda : Pool2d_cuda {
    MaxPool_cuda() { kernel_name = "maxpool_f32"; is_avg = false; }
    operator_t* make_fallback() override { return resolver_default_op_MaxPool(opset, ctx->attr_pool); }
};

} // namespace

operator_t* resolver_cuda_op_GlobalAveragePool(int opset, pool_t& pool) {
    return pool_new<GlobalAveragePool_cuda>(pool);
}
operator_t* resolver_cuda_op_AveragePool(int opset, pool_t& pool) {
    return pool_new<AveragePool_cuda>(pool);
}
operator_t* resolver_cuda_op_MaxPool(int opset, pool_t& pool) {
    return pool_new<MaxPool_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
