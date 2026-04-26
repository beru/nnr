#if defined(NNR_USE_CUDA)

// Concat along a single axis via NVRTC.
// Strategy: interpret each tensor as (outer × axis_dim × inner). Launch one
// kernel per input with the running axis_offset; copies outer*src_axis*inner
// elements into y at the correct offset.
// Supports F32 and INT64 (the latter for shape-arithmetic chains).

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"
#include <vector>

namespace nnr {

operator_t* resolver_default_op_Concat(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* concat_source() {
    return R"CUDA(
extern "C" {

#define CONCAT_KERNEL(name, T)                                                  \
__global__                                                                      \
void name(const T* __restrict__ x, T* __restrict__ y,                           \
          int outer, int src_axis, int inner, int dst_axis, int axis_offset)   \
{                                                                               \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                            \
    int total = outer * src_axis * inner;                                       \
    if (idx >= total) return;                                                   \
    int k = idx % inner;    int t = idx / inner;                                \
    int a = t   % src_axis;     t = t   / src_axis;                             \
    int o = t;                                                                  \
    size_t dst_off = ((size_t)o * dst_axis + (a + axis_offset)) * (size_t)inner + k; \
    y[dst_off] = x[idx];                                                        \
}

CONCAT_KERNEL(concat_axis_f32, float)
CONCAT_KERNEL(concat_axis_i64, long long)

} // extern "C"
)CUDA";
}

struct Concat_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int axis = 1, caxis = 0;
    int outer = 1, inner = 1, dst_axis = 0;
    std::vector<int> src_axes;

    bool init() override {
        if (!(inputs.size() >= 1 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_Concat(opset, ctx->attr_pool);
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        axis = (int)attribute(attr_key_t::axis, (int64_t)1);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false;
        device_tag = 0;

        const tensor_t* x0 = inputs[0];
        const tensor_t* y = outputs[0];
        if (x0->type != NNR_DATA_TYPE_FLOAT32 && x0->type != NNR_DATA_TYPE_INT64) return true;

        caxis = axis < 0 ? axis + x0->ndim : axis;
        if (caxis < 0 || caxis >= x0->ndim) return true;

        outer = 1; inner = 1;
        for (int d = 0; d < caxis; ++d)        outer *= x0->dims[d];
        for (int d = caxis + 1; d < x0->ndim; ++d) inner *= x0->dims[d];

        dst_axis = y->dims[caxis];

        src_axes.clear();
        src_axes.reserve(inputs.size());
        for (auto* in : inputs) {
            if (!in || in->type != x0->type) return true;
            if (in->ndim != x0->ndim) return true;
            for (int d = 0; d < x0->ndim; ++d) {
                if (d == caxis) continue;
                if (in->dims[d] != x0->dims[d]) return true;
            }
            src_axes.push_back(in->dims[caxis]);
        }

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        const char* kn = (inputs[0]->type == NNR_DATA_TYPE_INT64)
                         ? "concat_axis_i64" : "concat_axis_f32";
        CUfunction f = be->nvrtc.get("nnr_concat",
                                     concat_source(),
                                     kn,
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_y) { return fallback->exec(); }

        int axis_offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            void* d_x = be->cache->ensure_device(inputs[i]);
            if (!d_x) { return fallback->exec(); }

            int src_axis = src_axes[i];
            int _outer = outer, _inner = inner, _dst_axis = dst_axis;
            void* args[] = {
                &d_x, &d_y, &_outer, &src_axis, &_inner, &_dst_axis, &axis_offset,
            };
            unsigned long long total = (unsigned long long)outer * src_axis * inner;
            unsigned block = 256;
            unsigned grid = (unsigned)((total + block - 1) / block);
            if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args)) {
                return fallback->exec();
            }
            axis_offset += src_axis;
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Concat(int opset, pool_t& pool) {
    return pool_new<Concat_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
