#if defined(NNR_USE_CUDA)

// Shape on CUDA — writes the input's selected dims[start..end) into a device
// int64 buffer via a tiny NVRTC kernel.
//
// The dims values are passed as kernel args, which means cudaLaunchKernel bakes
// them into the captured graph. Replay re-launches with the same args, so the
// device buffer is repopulated correctly. A change in input shape sets
// `shapes_dirty`, which invalidates the captured graph (see plan_run() in
// cuda_backend.cpp:46).

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"
#include <climits>

namespace nnr {

operator_t* resolver_default_op_Shape(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* shape_source() {
    return R"CUDA(
extern "C" __global__
void shape_set_i64(long long* __restrict__ y, int n,
                   long long d0, long long d1, long long d2, long long d3,
                   long long d4, long long d5, long long d6, long long d7)
{
    int i = threadIdx.x;
    if (i >= n) return;
    long long arr[8] = { d0, d1, d2, d3, d4, d5, d6, d7 };
    y[i] = arr[i];
}
)CUDA";
}

struct Shape_cuda : public operator_t {
    operator_t* fallback = nullptr;
    int     start_attr = 0;
    int     end_attr   = INT_MAX;
    bool    has_end    = false;
    int     n_out      = 0;
    int64_t dims_emit[8] = {0,0,0,0,0,0,0,0};

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        layout_mask = LAYOUT_ALL;
        start_attr = (int)attribute(attr_key_t::start, (int64_t)0);
        int64_t e = attribute(attr_key_t::end, (int64_t)INT64_MAX);
        has_end = (e != INT64_MAX);
        end_attr = (int)e;
        // Keep the CPU op around as a hard fallback for non-GPU contexts.
        fallback = resolver_default_op_Shape(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        return fallback->init();
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        int ndim = x->ndim;
        int s = start_attr; if (s < 0) s += ndim; if (s < 0) s = 0; if (s > ndim) s = ndim;
        int e; if (!has_end) e = ndim;
        else { e = end_attr; if (e < 0) e += ndim; if (e < 0) e = 0; if (e > ndim) e = ndim; }
        n_out = (e > s) ? (e - s) : 0;
        int tmp[] = { n_out };
        if (!y->reshape(tmp, NNR_DATA_TYPE_INT64)) return false;
        for (int i = 0; i < 8; ++i) dims_emit[i] = (i < n_out) ? (int64_t)x->dims[s + i] : 0;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (n_out == 0) return true;
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) return fallback->exec();
        CUfunction f = be->nvrtc.get("nnr_shape", shape_source(),
                                     "shape_set_i64",
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) return fallback->exec();
        long long* d_y = (long long*)be->cache->alloc_output(outputs[0]);
        if (!d_y) return fallback->exec();
        int n = n_out;
        long long d0 = dims_emit[0], d1 = dims_emit[1], d2 = dims_emit[2], d3 = dims_emit[3];
        long long d4 = dims_emit[4], d5 = dims_emit[5], d6 = dims_emit[6], d7 = dims_emit[7];
        void* args[] = { &d_y, &n, &d0,&d1,&d2,&d3,&d4,&d5,&d6,&d7 };
        if (!gpu::nvrtc_launch(be->device, f, 1, 1, 1, 8, 1, 1, args)) return fallback->exec();
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Shape(int opset, pool_t& pool) {
    return pool_new<Shape_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
