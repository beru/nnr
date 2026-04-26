#if defined(NNR_USE_CUDA)

// Gather along a single axis via NVRTC.
// data: (outer, axis_dim, inner). indices: (N,). out: (outer, N, inner).
// data: f32 or int64. indices: int64 only.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_Gather(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* gather_source() {
    return R"CUDA(
extern "C" {

#define GATHER_KERNEL(name, T)                                                  \
__global__                                                                      \
void name(const T* __restrict__ data,                                           \
          const long long* __restrict__ idx,                                    \
          T* __restrict__ y,                                                    \
          int outer, int axis_dim, int inner, int idx_size)                     \
{                                                                               \
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; \
    unsigned long long total = (unsigned long long)outer * idx_size * inner;    \
    if (i >= total) return;                                                     \
    int k  = (int)(i % (unsigned long long)inner);                              \
    unsigned long long t = i / (unsigned long long)inner;                       \
    int ii = (int)(t % (unsigned long long)idx_size);                           \
    int o  = (int)(t / (unsigned long long)idx_size);                           \
    long long xi = idx[ii];                                                     \
    if (xi < 0) xi += axis_dim;                                                 \
    if (xi < 0) xi = 0;                                                         \
    if (xi >= axis_dim) xi = axis_dim - 1;                                      \
    size_t src_off = ((size_t)o * axis_dim + (size_t)xi) * (size_t)inner + k;   \
    y[i] = data[src_off];                                                       \
}

GATHER_KERNEL(gather_axis_f32, float)
GATHER_KERNEL(gather_axis_i64, long long)

} // extern "C"
)CUDA";
}

struct Gather_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int axis_attr = 0;
    int outer = 0, axis_dim = 0, inner = 0, idx_size = 0;

    bool init() override {
        if (!(inputs.size() == 2 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_Gather(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        axis_attr = (int)attribute(attr_key_t::axis, (int64_t)0);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* data = inputs[0];
        const tensor_t* idx  = inputs[1];
        if (idx->type != NNR_DATA_TYPE_INT64) return true;
        if (data->type != NNR_DATA_TYPE_FLOAT32 && data->type != NNR_DATA_TYPE_INT64) return true;

        int caxis = axis_attr < 0 ? axis_attr + data->ndim : axis_attr;
        if (caxis < 0 || caxis >= data->ndim) return true;

        outer = 1; inner = 1;
        for (int d = 0; d < caxis; ++d)        outer *= data->dims[d];
        for (int d = caxis + 1; d < data->ndim; ++d) inner *= data->dims[d];
        axis_dim = data->dims[caxis];
        idx_size = (int)idx->ndata;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        const char* kn = (inputs[0]->type == NNR_DATA_TYPE_INT64)
                         ? "gather_axis_i64" : "gather_axis_f32";
        CUfunction f = be->nvrtc.get("nnr_gather",
                                     gather_source(),
                                     kn,
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        void*      d_data = be->cache->ensure_device(inputs[0]);
        long long* d_idx  = (long long*)be->cache->ensure_device(inputs[1]);
        void*      d_y    = be->cache->alloc_output(outputs[0]);
        if (!d_data || !d_idx || !d_y) { return fallback->exec(); }

        unsigned long long total = (unsigned long long)outer * idx_size * inner;
        int _o = outer, _a = axis_dim, _i = inner, _is = idx_size;
        void* args[] = { &d_data, &d_idx, &d_y, &_o, &_a, &_i, &_is };
        unsigned block = 256;
        unsigned grid = (unsigned)((total + block - 1) / block);
        if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Gather(int opset, pool_t& pool) {
    return pool_new<Gather_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
