#if defined(NNR_USE_CUDA)

// Structural CUDA ops: Split, Slice, Pad, Tile, Expand.
// All are device-side data movement via NVRTC kernels. f32 only.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"
#include <vector>

namespace nnr {

operator_t* resolver_default_op_Split (int opset, pool_t& pool);
operator_t* resolver_default_op_Slice (int opset, pool_t& pool);
operator_t* resolver_default_op_Pad   (int opset, pool_t& pool);
operator_t* resolver_default_op_Tile  (int opset, pool_t& pool);
operator_t* resolver_default_op_Expand(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

// --------------------------------------------------------------------
// Shared NVRTC kernels
// --------------------------------------------------------------------

static const char* structural_source() {
    return R"CUDA(
extern "C" {

// Split: copy a contiguous slab of the input along `caxis` into an output.
// Interpreted as (outer, src_axis_slab_size, inner). Caller launches one kernel
// per output with src_offset = sum of preceding split sizes.
__global__ void split_axis_f32(const float* __restrict__ x,
                               float* __restrict__ y,
                               int outer,
                               int src_axis,
                               int inner,
                               int dst_axis,
                               int src_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * dst_axis * inner;
    if (idx >= total) return;

    int k = idx % inner;     int t = idx / inner;
    int a = t   % dst_axis;      t = t   / dst_axis;
    int o = t;

    size_t src_off = ((size_t)o * src_axis + (a + src_offset)) * (size_t)inner + k;
    y[idx] = x[src_off];
}

// Slice with per-axis start/step. Up to 8 dims. Strides packed as args.
#define SLICE_KERNEL(name, T)                                                   \
__global__ void name(const T* __restrict__ x,                                   \
                     T* __restrict__ y,                                         \
                     unsigned long long n,                                      \
                     int ndim,                                                  \
                     int out_dims_0, int out_dims_1, int out_dims_2, int out_dims_3, \
                     int out_dims_4, int out_dims_5, int out_dims_6, int out_dims_7, \
                     int start_0, int start_1, int start_2, int start_3,        \
                     int start_4, int start_5, int start_6, int start_7,        \
                     int step_0, int step_1, int step_2, int step_3,            \
                     int step_4, int step_5, int step_6, int step_7,            \
                     int sstr_0, int sstr_1, int sstr_2, int sstr_3,            \
                     int sstr_4, int sstr_5, int sstr_6, int sstr_7)            \
{                                                                               \
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; \
    if (i >= n) return;                                                         \
    int out_dims[8] = { out_dims_0,out_dims_1,out_dims_2,out_dims_3,out_dims_4,out_dims_5,out_dims_6,out_dims_7 }; \
    int starts[8]   = { start_0,start_1,start_2,start_3,start_4,start_5,start_6,start_7 }; \
    int steps[8]    = { step_0,step_1,step_2,step_3,step_4,step_5,step_6,step_7 }; \
    int sstr[8]     = { sstr_0,sstr_1,sstr_2,sstr_3,sstr_4,sstr_5,sstr_6,sstr_7 }; \
    unsigned long long rem = i;                                                 \
    long long src_off = 0;                                                      \
    for (int d = ndim - 1; d >= 0; --d) {                                       \
        int idx = (int)(rem % (unsigned long long)out_dims[d]);                 \
        rem /= (unsigned long long)out_dims[d];                                 \
        src_off += (long long)(starts[d] + idx * steps[d]) * (long long)sstr[d]; \
    }                                                                           \
    y[i] = x[src_off];                                                          \
}

SLICE_KERNEL(slice_f32, float)
SLICE_KERNEL(slice_i64, long long)
SLICE_KERNEL(slice_i8, signed char)
SLICE_KERNEL(slice_u8, unsigned char)

// Pad (constant-mode only). out = (pad_before + in_dim + pad_after) per axis.
// Element outside the inner range: pad_value. Otherwise: x at shifted index.
__global__ void pad_const_f32(const float* __restrict__ x,
                              float* __restrict__ y,
                              unsigned long long n,
                              int ndim,
                              int out_dims_0, int out_dims_1, int out_dims_2, int out_dims_3,
                              int out_dims_4, int out_dims_5, int out_dims_6, int out_dims_7,
                              int in_dims_0, int in_dims_1, int in_dims_2, int in_dims_3,
                              int in_dims_4, int in_dims_5, int in_dims_6, int in_dims_7,
                              int pad_before_0, int pad_before_1, int pad_before_2, int pad_before_3,
                              int pad_before_4, int pad_before_5, int pad_before_6, int pad_before_7,
                              int sstr_0, int sstr_1, int sstr_2, int sstr_3,
                              int sstr_4, int sstr_5, int sstr_6, int sstr_7,
                              float pad_value)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;

    int out_dims[8] = { out_dims_0,out_dims_1,out_dims_2,out_dims_3,out_dims_4,out_dims_5,out_dims_6,out_dims_7 };
    int in_dims[8]  = { in_dims_0,in_dims_1,in_dims_2,in_dims_3,in_dims_4,in_dims_5,in_dims_6,in_dims_7 };
    int pb[8]       = { pad_before_0,pad_before_1,pad_before_2,pad_before_3,pad_before_4,pad_before_5,pad_before_6,pad_before_7 };
    int sstr[8]     = { sstr_0,sstr_1,sstr_2,sstr_3,sstr_4,sstr_5,sstr_6,sstr_7 };

    unsigned long long rem = i;
    long long src_off = 0;
    int in_range = 1;
    for (int d = ndim - 1; d >= 0; --d) {
        int idx = (int)(rem % (unsigned long long)out_dims[d]);
        rem /= (unsigned long long)out_dims[d];
        int sidx = idx - pb[d];
        if (sidx < 0 || sidx >= in_dims[d]) { in_range = 0; break; }
        src_off += (long long)sidx * (long long)sstr[d];
    }
    y[i] = in_range ? x[src_off] : pad_value;
}

// Tile: repeat input N_i times along axis i. Inner index wraps mod in_dims.
__global__ void tile_f32(const float* __restrict__ x,
                         float* __restrict__ y,
                         unsigned long long n,
                         int ndim,
                         int out_dims_0, int out_dims_1, int out_dims_2, int out_dims_3,
                         int out_dims_4, int out_dims_5, int out_dims_6, int out_dims_7,
                         int in_dims_0, int in_dims_1, int in_dims_2, int in_dims_3,
                         int in_dims_4, int in_dims_5, int in_dims_6, int in_dims_7,
                         int sstr_0, int sstr_1, int sstr_2, int sstr_3,
                         int sstr_4, int sstr_5, int sstr_6, int sstr_7)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;

    int out_dims[8] = { out_dims_0,out_dims_1,out_dims_2,out_dims_3,out_dims_4,out_dims_5,out_dims_6,out_dims_7 };
    int in_dims[8]  = { in_dims_0,in_dims_1,in_dims_2,in_dims_3,in_dims_4,in_dims_5,in_dims_6,in_dims_7 };
    int sstr[8]     = { sstr_0,sstr_1,sstr_2,sstr_3,sstr_4,sstr_5,sstr_6,sstr_7 };

    unsigned long long rem = i;
    long long src_off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int idx = (int)(rem % (unsigned long long)out_dims[d]);
        rem /= (unsigned long long)out_dims[d];
        int sidx = idx % in_dims[d];
        src_off += (long long)sidx * (long long)sstr[d];
    }
    y[i] = x[src_off];
}

// Expand: broadcast — any dim with in_dim==1 uses src idx 0, else mirror.
// Same as tile with in_dims[d] == 1 meaning sstr[d] == 0 in caller.
__global__ void expand_f32(const float* __restrict__ x,
                           float* __restrict__ y,
                           unsigned long long n,
                           int ndim,
                           int out_dims_0, int out_dims_1, int out_dims_2, int out_dims_3,
                           int out_dims_4, int out_dims_5, int out_dims_6, int out_dims_7,
                           int sstr_0, int sstr_1, int sstr_2, int sstr_3,
                           int sstr_4, int sstr_5, int sstr_6, int sstr_7)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;

    int out_dims[8] = { out_dims_0,out_dims_1,out_dims_2,out_dims_3,out_dims_4,out_dims_5,out_dims_6,out_dims_7 };
    int sstr[8]     = { sstr_0,sstr_1,sstr_2,sstr_3,sstr_4,sstr_5,sstr_6,sstr_7 };

    unsigned long long rem = i;
    long long src_off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int idx = (int)(rem % (unsigned long long)out_dims[d]);
        rem /= (unsigned long long)out_dims[d];
        src_off += (long long)idx * (long long)sstr[d];
    }
    y[i] = x[src_off];
}

} // extern "C"
)CUDA";
}

static CUfunction get_struct_kernel(gpu::cuda_backend_t* be, const char* name) {
    return be->nvrtc.get("nnr_structural_f32",
                         structural_source(),
                         name,
                         gpu::nvrtc_arch_option(be->device));
}

static bool launch_1d(gpu::cuda_backend_t* be, CUfunction f, void** args, unsigned long long n) {
    constexpr unsigned BLK = 256;
    unsigned grid = (unsigned)((n + BLK - 1) / BLK);
    return gpu::nvrtc_launch(be->device, f, grid, 1, 1, BLK, 1, 1, args);
}

// --------------------------------------------------------------------
// Split
// --------------------------------------------------------------------

struct Split_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int axis_attr = 0, caxis = 0;
    int outer = 1, inner = 1, src_axis = 0;
    std::vector<int> part_sizes;

    bool init() override {
        if (inputs.size() < 1 || outputs.empty()) return false;
        fallback = resolver_default_op_Split(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        axis_attr = (int)attribute(attr_key_t::axis, (int64_t)0);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return true;

        caxis = axis_attr < 0 ? axis_attr + x->ndim : axis_attr;
        if (caxis < 0 || caxis >= x->ndim) return true;

        outer = 1; inner = 1;
        for (int d = 0; d < caxis; ++d)      outer *= x->dims[d];
        for (int d = caxis+1; d < x->ndim; ++d) inner *= x->dims[d];
        src_axis = x->dims[caxis];

        part_sizes.clear();
        part_sizes.reserve(outputs.size());
        for (auto* y : outputs) {
            if (!y || y->type != NNR_DATA_TYPE_FLOAT32 || y->ndim != x->ndim) return true;
            for (int d = 0; d < x->ndim; ++d) {
                if (d == caxis) continue;
                if (y->dims[d] != x->dims[d]) return true;
            }
            part_sizes.push_back(y->dims[caxis]);
        }
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f = get_struct_kernel(be, "split_axis_f32");
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        if (!d_x) { return fallback->exec(); }

        int src_offset = 0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            float* d_y = (float*)be->cache->alloc_output(outputs[i]);
            if (!d_y) { return fallback->exec(); }
            int dst_axis = part_sizes[i];
            int _outer = outer, _src = src_axis, _inner = inner, _dst = dst_axis, _off = src_offset;
            void* args[] = { &d_x, &d_y, &_outer, &_src, &_inner, &_dst, &_off };
            unsigned long long total = (unsigned long long)outer * dst_axis * inner;
            if (!launch_1d(be, f, args, total)) {
                return fallback->exec();
            }
            be->cache->mark_written(outputs[i]);
            src_offset += dst_axis;
        }
        return true;
    }
};

// --------------------------------------------------------------------
// Slice
// --------------------------------------------------------------------

struct Slice_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int ndim = 0;
    int out_dims[8] = {1,1,1,1,1,1,1,1};
    int starts[8]   = {0,0,0,0,0,0,0,0};
    int steps[8]    = {1,1,1,1,1,1,1,1};
    int sstr[8]     = {0,0,0,0,0,0,0,0};

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;
        fallback = resolver_default_op_Slice(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        const tensor_t* y = outputs[0];
        if ((x->type != NNR_DATA_TYPE_FLOAT32 && x->type != NNR_DATA_TYPE_INT64
             && x->type != NNR_DATA_TYPE_INT8   && x->type != NNR_DATA_TYPE_UINT8)
            || x->ndim > 8 || x->ndim != y->ndim) return true;

        // Read starts/ends/axes/steps from inputs (opset 10+).
        if (inputs.size() < 3) return true;   // attribute form of opset<10 not accelerated
        if (!inputs[1] || !inputs[2]) return true;
        if (inputs[1]->type != NNR_DATA_TYPE_INT64 || inputs[2]->type != NNR_DATA_TYPE_INT64) return true;

        int n_axes = (int)inputs[1]->ndata;
        const int64_t* p_starts = (const int64_t*)inputs[1]->data;
        const int64_t* p_ends   = (const int64_t*)inputs[2]->data;
        const int64_t* p_axes   = (inputs.size() >= 4 && inputs[3] && inputs[3]->data)
                                    ? (const int64_t*)inputs[3]->data : nullptr;
        const int64_t* p_steps  = (inputs.size() >= 5 && inputs[4] && inputs[4]->data)
                                    ? (const int64_t*)inputs[4]->data : nullptr;

        ndim = x->ndim;
        // default: full range, step=1, for every axis
        for (int d = 0; d < ndim; ++d) { out_dims[d] = y->dims[d]; starts[d] = 0; steps[d] = 1; }
        for (int d = ndim; d < 8;   ++d) { out_dims[d] = 1; starts[d] = 0; steps[d] = 1; }

        for (int i = 0; i < n_axes; ++i) {
            int ax = p_axes ? (int)p_axes[i] : i;
            if (ax < 0) ax += ndim;
            if (ax < 0 || ax >= ndim) return true;
            int64_t s = p_starts[i];
            int64_t e = p_ends[i];
            int64_t st = p_steps ? p_steps[i] : 1;
            if (st == 0) return true;
            if (s < 0) s += x->dims[ax];
            if (e < 0) e += x->dims[ax];
            s = (s < 0) ? 0 : (s > x->dims[ax] ? x->dims[ax] : s);
            e = (e < 0) ? 0 : (e > x->dims[ax] ? x->dims[ax] : e);
            starts[ax] = (int)s;
            steps[ax]  = (int)st;
        }
        // src row-major strides
        int src_stride[8]; src_stride[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) src_stride[d] = src_stride[d+1] * x->dims[d+1];
        for (int d = 0; d < ndim; ++d) sstr[d] = src_stride[d];
        for (int d = ndim; d < 8;   ++d) sstr[d] = 0;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        const char* kn;
        switch (inputs[0]->type) {
            case NNR_DATA_TYPE_INT64: kn = "slice_i64"; break;
            case NNR_DATA_TYPE_INT8:  kn = "slice_i8";  break;
            case NNR_DATA_TYPE_UINT8: kn = "slice_u8";  break;
            default:                  kn = "slice_f32"; break;
        }
        CUfunction f = get_struct_kernel(be, kn);
        if (!f) { return fallback->exec(); }

        void* d_x = be->cache->ensure_device(inputs[0]);
        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        void* args[] = {
            &d_x, &d_y, &n, &ndim,
            &out_dims[0], &out_dims[1], &out_dims[2], &out_dims[3],
            &out_dims[4], &out_dims[5], &out_dims[6], &out_dims[7],
            &starts[0], &starts[1], &starts[2], &starts[3],
            &starts[4], &starts[5], &starts[6], &starts[7],
            &steps[0], &steps[1], &steps[2], &steps[3],
            &steps[4], &steps[5], &steps[6], &steps[7],
            &sstr[0], &sstr[1], &sstr[2], &sstr[3],
            &sstr[4], &sstr[5], &sstr[6], &sstr[7],
        };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// --------------------------------------------------------------------
// Pad (constant mode only)
// --------------------------------------------------------------------

struct Pad_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int ndim = 0;
    int out_dims[8] = {1,1,1,1,1,1,1,1};
    int in_dims[8]  = {1,1,1,1,1,1,1,1};
    int pad_before[8] = {0,0,0,0,0,0,0,0};
    int sstr[8] = {0,0,0,0,0,0,0,0};
    float pad_value = 0.f;

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;
        fallback = resolver_default_op_Pad(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        const tensor_t* y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim > 8 || y->ndim != x->ndim) return true;

        // Only accelerate constant mode.
        std::string_view mode = attribute(attr_key_t::mode, "constant");
        if (mode != "constant") return true;

        // opset 11+: inputs[1] = pads (int64), inputs[2] = constant_value, inputs[3] = axes (optional)
        if (inputs.size() < 2 || !inputs[1] || !inputs[1]->data
            || inputs[1]->type != NNR_DATA_TYPE_INT64) return true;
        const int64_t* p = (const int64_t*)inputs[1]->data;
        int n_pads = (int)inputs[1]->ndata;

        pad_value = 0.f;
        if (inputs.size() >= 3 && inputs[2] && inputs[2]->data
            && inputs[2]->type == NNR_DATA_TYPE_FLOAT32 && inputs[2]->ndata == 1) {
            pad_value = *(const float*)inputs[2]->data;
        }

        ndim = x->ndim;
        for (int d = 0; d < ndim; ++d) {
            in_dims[d] = x->dims[d]; out_dims[d] = y->dims[d]; pad_before[d] = 0;
        }
        for (int d = ndim; d < 8; ++d) { in_dims[d] = 1; out_dims[d] = 1; pad_before[d] = 0; }

        if (inputs.size() >= 4 && inputs[3] && inputs[3]->data
            && inputs[3]->type == NNR_DATA_TYPE_INT64) {
            const int64_t* axes = (const int64_t*)inputs[3]->data;
            int n_axes = (int)inputs[3]->ndata;
            if (n_pads != 2 * n_axes) return true;
            for (int i = 0; i < n_axes; ++i) {
                int ax = (int)axes[i]; if (ax < 0) ax += ndim;
                if (ax < 0 || ax >= ndim) return true;
                pad_before[ax] = (int)p[i];
            }
        } else {
            if (n_pads != 2 * ndim) return true;
            for (int d = 0; d < ndim; ++d) pad_before[d] = (int)p[d];
        }

        int src_stride[8]; src_stride[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) src_stride[d] = src_stride[d+1] * x->dims[d+1];
        for (int d = 0; d < ndim; ++d) sstr[d] = src_stride[d];
        for (int d = ndim; d < 8;   ++d) sstr[d] = 0;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f = get_struct_kernel(be, "pad_const_f32");
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        float pv = pad_value;
        void* args[] = {
            &d_x, &d_y, &n, &ndim,
            &out_dims[0], &out_dims[1], &out_dims[2], &out_dims[3],
            &out_dims[4], &out_dims[5], &out_dims[6], &out_dims[7],
            &in_dims[0], &in_dims[1], &in_dims[2], &in_dims[3],
            &in_dims[4], &in_dims[5], &in_dims[6], &in_dims[7],
            &pad_before[0], &pad_before[1], &pad_before[2], &pad_before[3],
            &pad_before[4], &pad_before[5], &pad_before[6], &pad_before[7],
            &sstr[0], &sstr[1], &sstr[2], &sstr[3],
            &sstr[4], &sstr[5], &sstr[6], &sstr[7],
            &pv,
        };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// --------------------------------------------------------------------
// Tile
// --------------------------------------------------------------------

struct Tile_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int ndim = 0;
    int out_dims[8] = {1,1,1,1,1,1,1,1};
    int in_dims[8]  = {1,1,1,1,1,1,1,1};
    int sstr[8]     = {0,0,0,0,0,0,0,0};

    bool init() override {
        if (!(inputs.size() == 2 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_Tile(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        const tensor_t* y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim > 8 || y->ndim != x->ndim) return true;

        ndim = x->ndim;
        for (int d = 0; d < ndim; ++d) { in_dims[d] = x->dims[d]; out_dims[d] = y->dims[d]; }
        for (int d = ndim; d < 8; ++d) { in_dims[d] = 1; out_dims[d] = 1; }
        int src_stride[8]; src_stride[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) src_stride[d] = src_stride[d+1] * x->dims[d+1];
        for (int d = 0; d < ndim; ++d) sstr[d] = src_stride[d];
        for (int d = ndim; d < 8;   ++d) sstr[d] = 0;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_struct_kernel(be, "tile_f32");
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        void* args[] = {
            &d_x, &d_y, &n, &ndim,
            &out_dims[0], &out_dims[1], &out_dims[2], &out_dims[3],
            &out_dims[4], &out_dims[5], &out_dims[6], &out_dims[7],
            &in_dims[0], &in_dims[1], &in_dims[2], &in_dims[3],
            &in_dims[4], &in_dims[5], &in_dims[6], &in_dims[7],
            &sstr[0], &sstr[1], &sstr[2], &sstr[3],
            &sstr[4], &sstr[5], &sstr[6], &sstr[7],
        };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// --------------------------------------------------------------------
// Expand
// --------------------------------------------------------------------

struct Expand_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int ndim = 0;
    int out_dims[8] = {1,1,1,1,1,1,1,1};
    int sstr[8]     = {0,0,0,0,0,0,0,0};

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;
        fallback = resolver_default_op_Expand(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        const tensor_t* y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || y->ndim > 8) return true;

        ndim = y->ndim;
        // Right-align input onto output: prepend 1s to input shape.
        int in_padded[8] = {1,1,1,1,1,1,1,1};
        int off = ndim - x->ndim;
        if (off < 0) return true;
        for (int d = 0; d < x->ndim; ++d) in_padded[off + d] = x->dims[d];

        // Row-major strides of the input (with padded leading 1s → stride 0).
        int in_stride[8] = {0,0,0,0,0,0,0,0};
        int stride = 1;
        for (int d = x->ndim - 1; d >= 0; --d) {
            in_stride[off + d] = (x->dims[d] == 1) ? 0 : stride;
            stride *= x->dims[d];
        }

        for (int d = 0; d < ndim; ++d) {
            out_dims[d] = y->dims[d];
            // Broadcast: input dim must be 1 or match out.
            if (in_padded[d] != 1 && in_padded[d] != y->dims[d]) return true;
            sstr[d] = (in_padded[d] == 1) ? 0 : in_stride[d];
        }
        for (int d = ndim; d < 8; ++d) { out_dims[d] = 1; sstr[d] = 0; }

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_struct_kernel(be, "expand_f32");
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long n = outputs[0]->ndata;
        void* args[] = {
            &d_x, &d_y, &n, &ndim,
            &out_dims[0], &out_dims[1], &out_dims[2], &out_dims[3],
            &out_dims[4], &out_dims[5], &out_dims[6], &out_dims[7],
            &sstr[0], &sstr[1], &sstr[2], &sstr[3],
            &sstr[4], &sstr[5], &sstr[6], &sstr[7],
        };
        if (!launch_1d(be, f, args, n)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Split (int opset, pool_t& pool) { return pool_new<Split_cuda> (pool); }
operator_t* resolver_cuda_op_Slice (int opset, pool_t& pool) { return pool_new<Slice_cuda> (pool); }
operator_t* resolver_cuda_op_Pad   (int opset, pool_t& pool) { return pool_new<Pad_cuda>   (pool); }
operator_t* resolver_cuda_op_Tile  (int opset, pool_t& pool) { return pool_new<Tile_cuda>  (pool); }
operator_t* resolver_cuda_op_Expand(int opset, pool_t& pool) { return pool_new<Expand_cuda>(pool); }

} // namespace nnr

#endif // NNR_USE_CUDA
