#if defined(NNR_USE_CUDA)

// Softmax (opset 13) on the last axis via NVRTC.
// Layout interpreted as outer × current × inner; this kernel supports
// inner == 1 (row-wise softmax on last axis — the common case).
// Any other shape falls back to CPU.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_Softmax(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* softmax_source() {
    return R"CUDA(
#ifndef INFINITY
#define INFINITY __int_as_float(0x7F800000)
#endif
extern "C" __global__
void softmax_lastaxis_f32(const float* __restrict__ x,
                          float* __restrict__ y,
                          int current,
                          int outer)
{
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * current;
    float* yr       = y + (size_t)row * current;

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    extern __shared__ float smem[];

    // ----- pass 1: max -----
    float local_max = -INFINITY;
    for (int j = tid; j < current; j += nth) {
        float v = xr[j];
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float a = smem[tid];
            float b = smem[tid + off];
            smem[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    const float maxv = smem[0];

    // ----- pass 2: sum of exp -----
    float local_sum = 0.f;
    for (int j = tid; j < current; j += nth) {
        float v = __expf(xr[j] - maxv);
        yr[j] = v;            // stash exp result
        local_sum += v;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    const float sumv = smem[0];
    const float inv  = (sumv != 0.f) ? (1.f / sumv) : 0.f;

    // ----- pass 3: normalize -----
    for (int j = tid; j < current; j += nth) yr[j] *= inv;
}
)CUDA";
}

struct Softmax_cuda : public operator_t {
    int axis = -1;
    int caxis = 0, current = 0, outer = 0, inner = 0;
    bool prim_valid = false;
    operator_t* fallback = nullptr;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_Softmax(opset, ctx->attr_pool);
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        axis = attribute(attr_key_t::axis, -1);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        const tensor_t* x = inputs[0];

        caxis = axis < 0 ? axis + x->ndim : axis;
        outer = 1; inner = 1;
        if (caxis < 0 || caxis >= x->ndim) {
            prim_valid = false; device_tag = 0; return true;
        }
        for (int i = 0; i < x->ndim; ++i) {
            if (i == caxis)      current = x->dims[i];
            else if (i < caxis)  outer   *= x->dims[i];
            else                 inner   *= x->dims[i];
        }
        // Only support last-axis softmax with f32 for now.
        prim_valid = (x->type == NNR_DATA_TYPE_FLOAT32 && inner == 1 && current > 0);
        device_tag = prim_valid ? static_cast<uint8_t>(backend_t::CUDA) : 0;
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();

        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f = be->nvrtc.get("nnr_softmax_lastaxis_f32",
                                     softmax_source(),
                                     "softmax_lastaxis_f32",
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        // Block size: power of 2, clamped to [64, 256]. Shared mem scales with it.
        unsigned block = 256;
        while ((int)block > current && block > 64) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = block * sizeof(float);

        int _current = current;
        int _outer   = outer;
        void* args[] = { &d_x, &d_y, &_current, &_outer };
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)outer, 1, 1, block, 1, 1, args, shared)) {
            return fallback->exec();
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Softmax(int opset, pool_t& pool) {
    return pool_new<Softmax_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
