#if defined(NNR_USE_CUDA)

// QuantizeLinear / DequantizeLinear via NVRTC.
// Per-tensor scalar scale + zero-point only (f32 ↔ u8/i8).
// Per-axis / per-channel falls back to CPU.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_QuantizeLinear  (int opset, pool_t& pool);
operator_t* resolver_default_op_DequantizeLinear(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* quant_source() {
    return R"CUDA(
extern "C" {

// Quantize f32 -> u8
__global__ void quant_f32_to_u8(const float* __restrict__ x,
                                unsigned char* __restrict__ y,
                                unsigned long long n,
                                float scale, int zero_point)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = rintf(x[i] / scale) + (float)zero_point;
        if (v < 0.f)   v = 0.f;
        if (v > 255.f) v = 255.f;
        y[i] = (unsigned char)v;
    }
}

// Quantize f32 -> i8
__global__ void quant_f32_to_i8(const float* __restrict__ x,
                                signed char* __restrict__ y,
                                unsigned long long n,
                                float scale, int zero_point)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = rintf(x[i] / scale) + (float)zero_point;
        if (v < -128.f) v = -128.f;
        if (v >  127.f) v =  127.f;
        y[i] = (signed char)v;
    }
}

__global__ void dequant_u8_to_f32(const unsigned char* __restrict__ x,
                                  float* __restrict__ y,
                                  unsigned long long n,
                                  float scale, int zero_point)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = ((int)x[i] - zero_point) * scale;
}

__global__ void dequant_i8_to_f32(const signed char* __restrict__ x,
                                  float* __restrict__ y,
                                  unsigned long long n,
                                  float scale, int zero_point)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = ((int)x[i] - zero_point) * scale;
}

} // extern "C"
)CUDA";
}

struct QuantizeLinear_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;
    float scale = 1.f;
    int zero_point = 0;

    bool init() override {
        if (inputs.size() < 2 || outputs.size() != 1) return false;
        fallback = resolver_default_op_QuantizeLinear(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0; kernel_name = nullptr;

        const tensor_t* x = inputs[0];
        const tensor_t* s = inputs[1];
        const tensor_t* zp = (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0) ? inputs[2] : nullptr;
        if (x->type != NNR_DATA_TYPE_FLOAT32) return true;
        if (s->type != NNR_DATA_TYPE_FLOAT32 || s->ndata != 1) return true;   // per-tensor only
        if (!s->data) return true;
        scale = *(const float*)s->data;

        zero_point = 0;
        data_type_t out_type = outputs[0]->type;
        if (zp) {
            if (zp->ndata != 1 || !zp->data) return true;
            if (zp->type == NNR_DATA_TYPE_UINT8) zero_point = *(const uint8_t*)zp->data;
            else if (zp->type == NNR_DATA_TYPE_INT8) zero_point = *(const int8_t*)zp->data;
            else return true;
            if (zp->type != out_type) return true;
        }
        if (out_type == NNR_DATA_TYPE_UINT8)      kernel_name = "quant_f32_to_u8";
        else if (out_type == NNR_DATA_TYPE_INT8)  kernel_name = "quant_f32_to_i8";
        else return true;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = be->nvrtc.get("nnr_quant_f32", quant_source(),
                                     kernel_name, gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        void*  d_y = be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        float sc = scale; int zp = zero_point;
        void* args[] = { &d_x, &d_y, &n, &sc, &zp };
        unsigned block = 256;
        unsigned grid = (unsigned)((n + block - 1) / block);
        if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

struct DequantizeLinear_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    const char* kernel_name = nullptr;
    float scale = 1.f;
    int zero_point = 0;

    bool init() override {
        if (inputs.size() < 2 || outputs.size() != 1) return false;
        fallback = resolver_default_op_DequantizeLinear(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0; kernel_name = nullptr;

        const tensor_t* x = inputs[0];
        const tensor_t* s = inputs[1];
        const tensor_t* zp = (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0) ? inputs[2] : nullptr;
        if (outputs[0]->type != NNR_DATA_TYPE_FLOAT32) return true;
        if (s->type != NNR_DATA_TYPE_FLOAT32 || s->ndata != 1) return true;
        if (!s->data) return true;
        scale = *(const float*)s->data;

        zero_point = 0;
        if (zp) {
            if (zp->ndata != 1 || !zp->data) return true;
            if (zp->type == NNR_DATA_TYPE_UINT8) zero_point = *(const uint8_t*)zp->data;
            else if (zp->type == NNR_DATA_TYPE_INT8) zero_point = *(const int8_t*)zp->data;
            else return true;
        }

        if (x->type == NNR_DATA_TYPE_UINT8)      kernel_name = "dequant_u8_to_f32";
        else if (x->type == NNR_DATA_TYPE_INT8)  kernel_name = "dequant_i8_to_f32";
        else return true;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = be->nvrtc.get("nnr_quant_f32", quant_source(),
                                     kernel_name, gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        void*  d_x = be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        float sc = scale; int zp = zero_point;
        void* args[] = { &d_x, &d_y, &n, &sc, &zp };
        unsigned block = 256;
        unsigned grid = (unsigned)((n + block - 1) / block);
        if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_QuantizeLinear  (int opset, pool_t& pool) { return pool_new<QuantizeLinear_cuda>  (pool); }
operator_t* resolver_cuda_op_DequantizeLinear(int opset, pool_t& pool) { return pool_new<DequantizeLinear_cuda>(pool); }

} // namespace nnr

#endif // NNR_USE_CUDA
