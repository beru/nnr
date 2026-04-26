#if defined(NNR_USE_CUDA)

// ScatterElements + ConvTranspose via NVRTC.
// Both f32 only. Scatter: reduction="none" only (overwrite). ConvTranspose:
// 2D NCHW with atomicAdd-based accumulation.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_ScatterElements(int opset, pool_t& pool);
operator_t* resolver_default_op_ConvTranspose  (int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* scatter_source() {
    return R"CUDA(
extern "C" {

// Copy data[i] → y[i], then for each (o, ii, k), write y[o, idx[o,ii,k], k] = updates[o,ii,k].
// Layout: (outer, axis_dim_data, inner) for y/data; (outer, idx_axis, inner) for idx/updates.
__global__ void scatter_copy_f32(const float* __restrict__ data,
                                 float* __restrict__ y,
                                 unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = data[i];
}

__global__ void scatter_elements_f32(const long long* __restrict__ idx,
                                     const float* __restrict__ updates,
                                     float* __restrict__ y,
                                     int outer, int axis_data, int axis_idx, int inner)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * axis_idx * inner;
    if (id >= total) return;
    int k = id % inner;       int t = id / inner;
    int a = t   % axis_idx;       t = t   / axis_idx;
    int o = t;

    long long ix = idx[id];
    if (ix < 0) ix += axis_data;
    if (ix < 0 || ix >= axis_data) return;
    size_t dst_off = ((size_t)o * axis_data + (size_t)ix) * (size_t)inner + k;
    y[dst_off] = updates[id];
}

// ConvTranspose 2D (input-driven, atomicAdd). NCHW, f32.
__global__ void conv_transpose_f32(const float* __restrict__ x,   // (N, Cin, Hi, Wi)
                                   const float* __restrict__ w,   // (Cin, Cout, kH, kW) — ONNX convention
                                   float* __restrict__ y,         // (N, Cout, Ho, Wo)
                                   int N, int Cin, int Hi, int Wi,
                                   int Cout, int Ho, int Wo,
                                   int kH, int kW, int sH, int sW,
                                   int pT, int pL, int dH, int dW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cin * Hi * Wi;
    if (idx >= total) return;

    int iw = idx % Wi; int t = idx / Wi;
    int ih = t   % Hi;     t = t   / Hi;
    int ci = t   % Cin;    t = t   / Cin;
    int n  = t;

    float v = x[idx];
    for (int kr = 0; kr < kH; ++kr) {
        int oh = ih * sH + kr * dH - pT;
        if (oh < 0 || oh >= Ho) continue;
        for (int kc = 0; kc < kW; ++kc) {
            int ow = iw * sW + kc * dW - pL;
            if (ow < 0 || ow >= Wo) continue;
            for (int co = 0; co < Cout; ++co) {
                float wv = w[((size_t)ci * Cout + co) * kH * kW + (size_t)kr * kW + kc];
                float* p = y + ((size_t)n * Cout + co) * Ho * Wo + (size_t)oh * Wo + ow;
                atomicAdd(p, v * wv);
            }
        }
    }
}

__global__ void zero_f32(float* __restrict__ y, unsigned long long n)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = 0.f;
}

} // extern "C"
)CUDA";
}

static CUfunction get_scat(gpu::cuda_backend_t* be, const char* name) {
    return be->nvrtc.get("nnr_scatter_f32", scatter_source(), name,
                         gpu::nvrtc_arch_option(be->device));
}

static bool launch_1d(gpu::cuda_backend_t* be, CUfunction f, void** args, unsigned long long n) {
    constexpr unsigned BLK = 256;
    unsigned grid = (unsigned)((n + BLK - 1) / BLK);
    return gpu::nvrtc_launch(be->device, f, grid, 1, 1, BLK, 1, 1, args);
}

// -------- ScatterElements --------

struct ScatterElements_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int axis_attr = 0, caxis = 0;
    int outer = 0, axis_data = 0, axis_idx = 0, inner = 0;

    bool init() override {
        if (!(inputs.size() == 3 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_ScatterElements(opset, ctx->attr_pool);
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
        const tensor_t* upd  = inputs[2];
        if (data->type != NNR_DATA_TYPE_FLOAT32 || upd->type != NNR_DATA_TYPE_FLOAT32) return true;
        if (idx->type != NNR_DATA_TYPE_INT64) return true;
        if (data->ndim != idx->ndim || data->ndim != upd->ndim) return true;
        std::string_view red = attribute(attr_key_t::reduction, "none");
        if (red != "none") return true;

        caxis = axis_attr < 0 ? axis_attr + data->ndim : axis_attr;
        if (caxis < 0 || caxis >= data->ndim) return true;

        outer = 1; inner = 1;
        for (int d = 0; d < caxis; ++d)         outer *= data->dims[d];
        for (int d = caxis + 1; d < data->ndim; ++d) inner *= data->dims[d];
        axis_data = data->dims[caxis];
        axis_idx  = idx->dims[caxis];
        for (int d = 0; d < data->ndim; ++d) {
            if (d == caxis) continue;
            if (idx->dims[d] != data->dims[d] || upd->dims[d] != data->dims[d]) return true;
        }
        if (idx->dims[caxis] != upd->dims[caxis]) return true;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f_copy = get_scat(be, "scatter_copy_f32");
        CUfunction f_scat = get_scat(be, "scatter_elements_f32");
        if (!f_copy || !f_scat) { return fallback->exec(); }

        float*     d_data = (float*)    be->cache->ensure_device(inputs[0]);
        long long* d_idx  = (long long*)be->cache->ensure_device(inputs[1]);
        float*     d_upd  = (float*)    be->cache->ensure_device(inputs[2]);
        float*     d_y    = (float*)    be->cache->alloc_output(outputs[0]);
        if (!d_data || !d_idx || !d_upd || !d_y) { return fallback->exec(); }

        unsigned long long n_data = outputs[0]->ndata;
        void* cargs[] = { &d_data, &d_y, &n_data };
        if (!launch_1d(be, f_copy, cargs, n_data)) { return fallback->exec(); }

        unsigned long long n_idx = (unsigned long long)outer * axis_idx * inner;
        int _o = outer, _ad = axis_data, _ai = axis_idx, _in = inner;
        void* sargs[] = { &d_idx, &d_upd, &d_y, &_o, &_ad, &_ai, &_in };
        if (!launch_1d(be, f_scat, sargs, n_idx)) { return fallback->exec(); }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

// -------- ConvTranspose --------

struct ConvTranspose_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int N=0, Cin=0, Hi=0, Wi=0, Cout=0, Ho=0, Wo=0;
    int kH=0, kW=0, sH=1, sW=1, pT=0, pL=0, dH=1, dW=1;
    int group = 1;
    bool has_bias = false;

    bool init() override {
        if (!(inputs.size() >= 2 && outputs.size() == 1)) return false;
        fallback = resolver_default_op_ConvTranspose(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        const tensor_t* y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return true;
        if (x->ndim != 4 || w->ndim != 4 || y->ndim != 4) return true;

        int64_t* ints = nullptr; int n = 0;
        n = attribute(attr_key_t::pads, ints);
        pT = (n >= 2) ? (int)ints[0] : 0;
        pL = (n >= 2) ? (int)ints[1] : 0;
        n = attribute(attr_key_t::strides, ints);
        sH = (n >= 1) ? (int)ints[0] : 1;
        sW = (n >= 2) ? (int)ints[1] : 1;
        n = attribute(attr_key_t::dilations, ints);
        dH = (n >= 1) ? (int)ints[0] : 1;
        dW = (n >= 2) ? (int)ints[1] : 1;
        group = (int)attribute(attr_key_t::group, (int64_t)1);
        if (group != 1) return true;

        N = x->dims[0]; Cin = x->dims[1]; Hi = x->dims[2]; Wi = x->dims[3];
        // ONNX ConvTranspose weight layout: (Cin, Cout_per_group, kH, kW).
        if (w->dims[0] != Cin) return true;
        Cout = w->dims[1];
        kH = w->dims[2]; kW = w->dims[3];
        Ho = y->dims[2]; Wo = y->dims[3];
        if (y->dims[1] != Cout) return true;

        has_bias = (inputs.size() > 2 && inputs[2] && inputs[2]->type == NNR_DATA_TYPE_FLOAT32
                    && (int64_t)inputs[2]->ndata == Cout);

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        CUfunction f_zero = get_scat(be, "zero_f32");
        CUfunction f_conv = get_scat(be, "conv_transpose_f32");
        if (!f_zero || !f_conv) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_w = (float*)be->cache->ensure_device(inputs[1]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_w || !d_y) { return fallback->exec(); }

        // Zero the output (atomicAdd accumulates).
        unsigned long long y_elems = outputs[0]->ndata;
        void* zargs[] = { &d_y, &y_elems };
        if (!launch_1d(be, f_zero, zargs, y_elems)) { return fallback->exec(); }

        unsigned long long x_elems = (unsigned long long)N * Cin * Hi * Wi;
        int _N=N,_Cin=Cin,_Hi=Hi,_Wi=Wi,_Co=Cout,_Ho=Ho,_Wo=Wo,
            _kH=kH,_kW=kW,_sH=sH,_sW=sW,_pT=pT,_pL=pL,_dH=dH,_dW=dW;
        void* args[] = { &d_x, &d_w, &d_y, &_N, &_Cin, &_Hi, &_Wi, &_Co, &_Ho, &_Wo,
                         &_kH, &_kW, &_sH, &_sW, &_pT, &_pL, &_dH, &_dW };
        if (!launch_1d(be, f_conv, args, x_elems)) { return fallback->exec(); }

        // Bias: reuse elementwise bias_nchw kernel for (Cout,) over (N, Cout, Ho*Wo).
        // Call via a tiny inline kernel — simpler to just skip here and fall back
        // if bias is present. (Acceptable: most ConvTranspose in upsample heads has no bias.)
        if (has_bias) {
            // leave accumulate; do NOT add bias here (simpler; fallback handles)
            // or manually add via cublasSaxpy. Skip for simplicity — correctness prefers fallback.
            return fallback->exec();
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_ScatterElements(int opset, pool_t& pool) { return pool_new<ScatterElements_cuda>(pool); }
operator_t* resolver_cuda_op_ConvTranspose  (int opset, pool_t& pool) { return pool_new<ConvTranspose_cuda>  (pool); }

} // namespace nnr

#endif // NNR_USE_CUDA
