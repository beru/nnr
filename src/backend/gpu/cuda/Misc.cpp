#if defined(NNR_USE_CUDA)

// Miscellaneous ops: Trilu, Hardmax, OneHot, LRN, GroupNormalization,
// SpaceToDepth, DepthToSpace. F32 only.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_Trilu             (int opset, pool_t& pool);
operator_t* resolver_default_op_Hardmax           (int opset, pool_t& pool);
operator_t* resolver_default_op_OneHot            (int opset, pool_t& pool);
operator_t* resolver_default_op_LRN               (int opset, pool_t& pool);
operator_t* resolver_default_op_GroupNormalization(int opset, pool_t& pool);
operator_t* resolver_default_op_SpaceToDepth      (int opset, pool_t& pool);
operator_t* resolver_default_op_DepthToSpace      (int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* misc_source() {
    return R"CUDA(
#ifndef INFINITY
#define INFINITY __int_as_float(0x7F800000)
#endif
extern "C" {

// Trilu: zero-out above/below the (k-shifted) main diagonal of last-2 dims.
// batch = product of leading dims; H,W = last two.
__global__ void trilu_f32(const float* __restrict__ x,
                          float* __restrict__ y,
                          unsigned long long n_total,
                          int H, int W, int k, int upper)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    int w = (int)(i % (unsigned long long)W);
    unsigned long long t = i / (unsigned long long)W;
    int h = (int)(t % (unsigned long long)H);

    float v = x[i];
    int diff = w - h;
    // upper: keep w - h >= k; lower: keep w - h <= k
    int keep = upper ? (diff >= k) : (diff <= k);
    y[i] = keep ? v : 0.f;
}

// Hardmax: last-axis; output is one-hot at argmax position.
__global__ void hardmax_lastaxis_f32(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int D, int outer)
{
    extern __shared__ float smem[];
    float* vmem = smem;
    int*   imem = (int*)(smem + blockDim.x);
    const int row = blockIdx.x;
    if (row >= outer) return;
    const float* xr = x + (size_t)row * D;
    float* yr       = y + (size_t)row * D;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float best = -INFINITY;
    int   best_i = 0;
    for (int j = tid; j < D; j += nth) {
        float v = xr[j];
        if (v > best) { best = v; best_i = j; }
    }
    vmem[tid] = best; imem[tid] = best_i;
    __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float a = vmem[tid], b = vmem[tid + off];
            if (b > a) { vmem[tid] = b; imem[tid] = imem[tid + off]; }
        }
        __syncthreads();
    }
    int chosen = imem[0];
    for (int j = tid; j < D; j += nth) yr[j] = (j == chosen) ? 1.f : 0.f;
}

// OneHot (axis = -1): indices shape (outer,), output shape (outer, depth).
__global__ void onehot_f32_lastaxis(const long long* __restrict__ idx,
                                    float* __restrict__ y,
                                    int outer, int depth,
                                    float on_value, float off_value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * depth;
    if (i >= total) return;
    int d = i % depth;
    int o = i / depth;
    long long idx_o = idx[o];
    if (idx_o < 0) idx_o += depth;
    y[i] = (idx_o == (long long)d) ? on_value : off_value;
}

// Local Response Normalization, across-channel.
// y[n,c,h,w] = x[n,c,h,w] / (bias + alpha/size * sum(x[n,c-r..c+r,h,w]^2))^beta
__global__ void lrn_across_f32(const float* __restrict__ x,
                               float* __restrict__ y,
                               int N, int C, int H, int W,
                               int size, float alpha, float beta, float bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int w = idx % W; int t = idx / W;
    int h = t % H; t = t / H;
    int c = t % C; t = t / C;
    int n = t;

    int half = (size - 1) / 2;
    int c0 = c - half; if (c0 < 0) c0 = 0;
    int c1 = c + half; if (c1 >= C) c1 = C - 1;

    float sum = 0.f;
    for (int cc = c0; cc <= c1; ++cc) {
        float v = x[((size_t)n * C + cc) * H * W + (size_t)h * W + w];
        sum += v * v;
    }
    float factor = bias + alpha * sum / (float)size;
    y[idx] = x[idx] / powf(factor, beta);
}

// GroupNormalization: (N, C, inner) split into G groups. Normalize per (n, g).
// scale, bias: shape (C,). eps: small number.
__global__ void group_norm_f32(const float* __restrict__ x,
                               const float* __restrict__ scale,
                               const float* __restrict__ bias,
                               float* __restrict__ y,
                               int N, int C, int inner, int G, float eps)
{
    extern __shared__ float smem[];   // 2 * nth
    // one block per (n, g). channels-per-group = C/G. elements-per-group = (C/G)*inner.
    const int ng  = blockIdx.x;       // 0..N*G
    const int g   = ng % G;
    const int n   = ng / G;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    const int cpg = C / G;
    const int GE = cpg * inner;
    const size_t base = ((size_t)n * C + (size_t)g * cpg) * inner;

    float s = 0.f, ss = 0.f;
    for (int j = tid; j < GE; j += nth) {
        float v = x[base + j];
        s += v; ss += v * v;
    }
    smem[tid] = s; smem[nth + tid] = ss;
    __syncthreads();
    for (int off = nth >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            smem[tid]       += smem[tid + off];
            smem[nth + tid] += smem[nth + tid + off];
        }
        __syncthreads();
    }
    float mean = smem[0]     / (float)GE;
    float mq   = smem[nth]   / (float)GE;
    float var  = mq - mean * mean; if (var < 0.f) var = 0.f;
    float inv  = rsqrtf(var + eps);

    // Apply scale/bias per channel (not per group).
    for (int j = tid; j < GE; j += nth) {
        int local_c = j / inner;
        int c = g * cpg + local_c;
        float v = (x[base + j] - mean) * inv;
        v = v * scale[c] + bias[c];
        y[base + j] = v;
    }
}

// SpaceToDepth (NCHW): input (N, C, H, W) → (N, C*r*r, H/r, W/r).
// Element (n, c', h', w') in output comes from
// (n, c'/(r*r), h'*r + (c' % (r*r))/r, w'*r + c' % r).
// Runs thread per output element.
__global__ void space_to_depth_f32(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int N, int Cin, int H, int W, int r)
{
    int Cout = Cin * r * r;
    int Ho = H / r, Wo = W / r;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Ho * Wo;
    if (idx >= total) return;
    int wo = idx % Wo; int t = idx / Wo;
    int ho = t   % Ho;     t = t   / Ho;
    int co = t   % Cout;   t = t   / Cout;
    int n  = t;

    int c_in  = co / (r * r);
    int rem   = co % (r * r);
    int dh    = rem / r;
    int dw    = rem % r;
    int hi    = ho * r + dh;
    int wi    = wo * r + dw;
    y[idx] = x[((size_t)n * Cin + c_in) * H * W + (size_t)hi * W + wi];
}

// DepthToSpace (NCHW, DCR mode): input (N, C, H, W) → (N, C/(r*r), H*r, W*r).
// Element (n, c_out, h_out, w_out) comes from (n, c_in, h_in, w_in) with
// c_in = c_out + (h_out % r) * r * (C/r/r) + (w_out % r) * (C/r/r)? — impl-dependent.
// Use standard "DCR" (depth-column-row) mode.
__global__ void depth_to_space_f32(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int N, int Cin, int H, int W, int r)
{
    int Cout = Cin / (r * r);
    int Ho = H * r, Wo = W * r;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Ho * Wo;
    if (idx >= total) return;
    int wo = idx % Wo; int t = idx / Wo;
    int ho = t   % Ho;     t = t   / Ho;
    int co = t   % Cout;   t = t   / Cout;
    int n  = t;

    int dh   = ho % r;
    int dw   = wo % r;
    int hi   = ho / r;
    int wi   = wo / r;
    // DCR: c_in = co + (dh * r + dw) * Cout
    int c_in = co + (dh * r + dw) * Cout;
    y[idx] = x[((size_t)n * Cin + c_in) * H * W + (size_t)hi * W + wi];
}

} // extern "C"
)CUDA";
}

static CUfunction get_misc(gpu::cuda_backend_t* be, const char* name) {
    return be->nvrtc.get("nnr_misc_f32", misc_source(), name,
                         gpu::nvrtc_arch_option(be->device));
}

static bool launch_1d(gpu::cuda_backend_t* be, CUfunction f, void** args, unsigned long long n) {
    constexpr unsigned BLK = 256;
    unsigned grid = (unsigned)((n + BLK - 1) / BLK);
    return gpu::nvrtc_launch(be->device, f, grid, 1, 1, BLK, 1, 1, args);
}

// -------- Trilu --------

struct Trilu_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int upper = 1;
    int H=0, W=0;
    unsigned long long n_total = 0;
    int k_off = 0;

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;
        fallback = resolver_default_op_Trilu(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        upper = (int)attribute(attr_key_t::upper, (int64_t)1);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim < 2) return true;
        H = x->dims[x->ndim - 2];
        W = x->dims[x->ndim - 1];
        n_total = x->ndata;
        k_off = 0;
        if (inputs.size() > 1 && inputs[1] && inputs[1]->data && inputs[1]->ndata == 1
            && inputs[1]->type == NNR_DATA_TYPE_INT64) {
            k_off = (int)*(const int64_t*)inputs[1]->data;
        }
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_misc(be, "trilu_f32");
        if (!f) { return fallback->exec(); }
        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }
        int _H = H, _W = W, _k = k_off, _u = upper;
        void* args[] = { &d_x, &d_y, &n_total, &_H, &_W, &_k, &_u };
        if (!launch_1d(be, f, args, n_total)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------- Hardmax --------

struct Hardmax_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int axis = -1, D = 0, outer = 0;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_Hardmax(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        axis = (int)attribute(attr_key_t::axis, (int64_t)-1);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return true;
        int ca = axis < 0 ? axis + x->ndim : axis;
        if (ca != x->ndim - 1) return true;
        D = x->dims[ca];
        outer = 1; for (int d = 0; d < ca; ++d) outer *= x->dims[d];
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_misc(be, "hardmax_lastaxis_f32");
        if (!f) { return fallback->exec(); }
        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }
        unsigned block = 256;
        while ((int)block > D && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = block * (sizeof(float) + sizeof(int));
        int _D = D, _outer = outer;
        void* args[] = { &d_x, &d_y, &_D, &_outer };
        if (!gpu::nvrtc_launch(be->device, f, (unsigned)outer, 1, 1, block, 1, 1, args, shared)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------- OneHot (axis=-1, int64 indices → f32 output) --------

struct OneHot_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int depth = 0;
    float on_val = 1.f, off_val = 0.f;
    int outer = 0;

    bool init() override {
        if (!(inputs.size() == 3 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_OneHot(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* idx = inputs[0];
        const tensor_t* dep = inputs[1];
        const tensor_t* vals = inputs[2];
        tensor_t* y = outputs[0];
        if (idx->type != NNR_DATA_TYPE_INT64) return true;
        if (y->type != NNR_DATA_TYPE_FLOAT32) return true;
        if (vals->type != NNR_DATA_TYPE_FLOAT32 || vals->ndata != 2 || !vals->data) return true;
        if (dep->ndata != 1 || !dep->data) return true;
        if (dep->type == NNR_DATA_TYPE_INT64)      depth = (int)*(const int64_t*)dep->data;
        else if (dep->type == NNR_DATA_TYPE_INT32) depth = *(const int32_t*)dep->data;
        else return true;
        off_val = ((const float*)vals->data)[0];
        on_val  = ((const float*)vals->data)[1];

        int axis_attr = (int)attribute(attr_key_t::axis, (int64_t)-1);
        int ca = axis_attr < 0 ? axis_attr + y->ndim : axis_attr;
        if (ca != y->ndim - 1) return true;   // last-axis only

        outer = (int)idx->ndata;
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_misc(be, "onehot_f32_lastaxis");
        if (!f) { return fallback->exec(); }
        long long* d_idx = (long long*)be->cache->ensure_device(inputs[0]);
        float*     d_y   = (float*)    be->cache->alloc_output(outputs[0]);
        if (!d_idx || !d_y) { return fallback->exec(); }
        int _o = outer, _d = depth; float _on = on_val, _off = off_val;
        void* args[] = { &d_idx, &d_y, &_o, &_d, &_on, &_off };
        if (!launch_1d(be, f, args, (unsigned long long)outer * depth)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------- LRN --------

struct LRN_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int N=0, C=0, H=0, W=0;
    int size = 1;
    float alpha = 0.0001f, beta = 0.75f, bias = 1.f;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_LRN(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        size  = (int)attribute(attr_key_t::size, (int64_t)1);
        alpha = attribute(attr_key_t::alpha, 0.0001f);
        beta  = attribute(attr_key_t::beta, 0.75f);
        bias  = attribute(attr_key_t::bias, 1.f);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim != 4) return true;
        N = x->dims[0]; C = x->dims[1]; H = x->dims[2]; W = x->dims[3];
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_misc(be, "lrn_across_f32");
        if (!f) { return fallback->exec(); }
        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }
        unsigned long long total = (unsigned long long)N * C * H * W;
        int _N = N, _C = C, _H = H, _W = W, _size = size;
        float _a = alpha, _b = beta, _bias = bias;
        void* args[] = { &d_x, &d_y, &_N, &_C, &_H, &_W, &_size, &_a, &_b, &_bias };
        if (!launch_1d(be, f, args, total)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------- GroupNormalization --------

struct GroupNorm_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    float eps = 1e-5f;
    int num_groups = 1;
    int N=0, C=0, inner=0;

    bool init() override {
        if (!(inputs.size() == 3 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_GroupNormalization(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        num_groups = (int)attribute(attr_key_t::num_groups, (int64_t)1);
        eps = attribute(attr_key_t::epsilon, 1e-5f);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim < 2) return true;
        N = x->dims[0]; C = x->dims[1];
        inner = 1; for (int d = 2; d < x->ndim; ++d) inner *= x->dims[d];
        if (num_groups <= 0 || C % num_groups != 0) return true;
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_misc(be, "group_norm_f32");
        if (!f) { return fallback->exec(); }
        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_s = (float*)be->cache->ensure_device(inputs[1]);
        float* d_b = (float*)be->cache->ensure_device(inputs[2]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_s || !d_b || !d_y) { return fallback->exec(); }
        int cpg = C / num_groups;
        int GE = cpg * inner;
        unsigned block = 256;
        while ((int)block > GE && block > 32) block >>= 1;
        if (block < 32) block = 32;
        unsigned shared = 2 * block * sizeof(float);
        int _N = N, _C = C, _inner = inner, _G = num_groups;
        float _eps = eps;
        void* args[] = { &d_x, &d_s, &d_b, &d_y, &_N, &_C, &_inner, &_G, &_eps };
        unsigned grid = (unsigned)(N * num_groups);
        if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args, shared)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------- SpaceToDepth --------

struct SpaceToDepth_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int r = 1, N=0, C=0, H=0, W=0;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_SpaceToDepth(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        r = (int)attribute(attr_key_t::blocksize, (int64_t)1);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim != 4) return true;
        N = x->dims[0]; C = x->dims[1]; H = x->dims[2]; W = x->dims[3];
        if (r <= 0 || H % r != 0 || W % r != 0) return true;
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_misc(be, "space_to_depth_f32");
        if (!f) { return fallback->exec(); }
        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }
        int _N = N, _C = C, _H = H, _W = W, _r = r;
        void* args[] = { &d_x, &d_y, &_N, &_C, &_H, &_W, &_r };
        unsigned long long total = (unsigned long long)outputs[0]->ndata;
        if (!launch_1d(be, f, args, total)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------- DepthToSpace --------

struct DepthToSpace_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int r = 1, N=0, C=0, H=0, W=0;

    bool init() override {
        if (!(inputs.size() == 1 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_DepthToSpace(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        r = (int)attribute(attr_key_t::blocksize, (int64_t)1);
        // DCR mode only (default).
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim != 4) return true;
        N = x->dims[0]; C = x->dims[1]; H = x->dims[2]; W = x->dims[3];
        if (r <= 0 || C % (r * r) != 0) return true;
        std::string_view mode = attribute(attr_key_t::mode, "DCR");
        if (mode != "DCR") return true;
        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        CUfunction f = get_misc(be, "depth_to_space_f32");
        if (!f) { return fallback->exec(); }
        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }
        int _N = N, _C = C, _H = H, _W = W, _r = r;
        void* args[] = { &d_x, &d_y, &_N, &_C, &_H, &_W, &_r };
        unsigned long long total = (unsigned long long)outputs[0]->ndata;
        if (!launch_1d(be, f, args, total)) { return fallback->exec(); }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Trilu             (int opset, pool_t& pool) { return pool_new<Trilu_cuda>       (pool); }
operator_t* resolver_cuda_op_Hardmax           (int opset, pool_t& pool) { return pool_new<Hardmax_cuda>     (pool); }
operator_t* resolver_cuda_op_OneHot            (int opset, pool_t& pool) { return pool_new<OneHot_cuda>      (pool); }
operator_t* resolver_cuda_op_LRN               (int opset, pool_t& pool) { return pool_new<LRN_cuda>         (pool); }
operator_t* resolver_cuda_op_GroupNormalization(int opset, pool_t& pool) { return pool_new<GroupNorm_cuda>   (pool); }
operator_t* resolver_cuda_op_SpaceToDepth      (int opset, pool_t& pool) { return pool_new<SpaceToDepth_cuda>(pool); }
operator_t* resolver_cuda_op_DepthToSpace      (int opset, pool_t& pool) { return pool_new<DepthToSpace_cuda>(pool); }

} // namespace nnr

#endif // NNR_USE_CUDA
