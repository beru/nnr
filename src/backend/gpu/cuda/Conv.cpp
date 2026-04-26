// Conv via im2col + WMMA TF32 GEMM (NVRTC). NCHW f32, 2D spatial, group=1.
// im2col packs each output-pixel's receptive field as a column; weight × col
// is a single gemm_device_f32 call per batch element. Bias added via the
// elementwise bias_nchw kernel.
#if defined(NNR_USE_CUDA)
#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "elementwise.h"
#include "attr_key.h"

#include <cfloat>
#include <string_view>

namespace nnr {

operator_t* resolver_default_op_Conv(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* im2col_source() {
    return R"CUDA(
extern "C" __global__
void im2col_f32(const float* __restrict__ x,       // (Cin, Hi, Wi) for one batch
                float* __restrict__ col,            // (Cin*kH*kW, Ho*Wo)
                int Cin, int Hi, int Wi, int Ho, int Wo,
                int kH, int kW, int sH, int sW,
                int pT, int pL, int dH, int dW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cols = Ho * Wo;
    int rows = Cin * kH * kW;
    int total = rows * cols;
    if (idx >= total) return;

    int col_idx = idx % cols;   int row = idx / cols;
    int kc = row % kW;          int t  = row / kW;
    int kr = t   % kH;              t  = t   / kH;
    int ci = t;

    int oh = col_idx / Wo;
    int ow = col_idx % Wo;

    int ih = oh * sH - pT + kr * dH;
    int iw = ow * sW - pL + kc * dW;

    float v = 0.f;
    if (ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
        v = x[((size_t)ci * Hi + ih) * Wi + iw];
    }
    col[idx] = v;
}
)CUDA";
}

// Fused Conv post-op kernel: Y[n,c,i] = activation(Y[n,c,i] + bias[c])
//   act_kind: 0=none, 1=Relu, 2=Clip(lo, hi). A single kernel launch replaces
//   the previous bias_nchw + separate Relu/Clip launches; fuses the post-op
//   with bias writeback. Called from Conv_cuda::exec when fusion_kind != 0
//   or when bias is present.
static const char* conv_postop_source() {
    return R"CUDA(
extern "C" __global__
void conv_postop_f32(float* __restrict__ Y,
                     const float* __restrict__ bias,    // may be null
                     int N, int Cout, int spatial,       // spatial = Ho * Wo
                     int has_bias,
                     int act_kind, float lo, float hi)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)N * Cout * spatial;
    if (idx >= total) return;

    int c = (int)((idx / spatial) % Cout);
    float v = Y[idx];
    if (has_bias) v += bias[c];
    if (act_kind == 1) v = v > 0.f ? v : 0.f;
    else if (act_kind == 2) v = v < lo ? lo : (v > hi ? hi : v);
    Y[idx] = v;
}
)CUDA";
}

// Depthwise 3×3 stride-1 tiled kernel. Each block owns a 16×16 output tile for
// one (n, c) slice. Loads an 18×18 input halo into shared memory once, then
// reads from shared for all 9 kernel taps. Weights (9 floats) live in shared.
//
// Launch: grid=(ceil(Wo/16), ceil(Ho/16), N*C), block=(16, 16).
// Requires sH=sW=1, dH=dW=1, kH=kW=3. Padding-agnostic (zero-fill on load).
static const char* depthwise_3x3s1_source() {
    return R"CUDA(
extern "C" __global__
void depthwise_3x3s1_f32(const float* __restrict__ X,
                         const float* __restrict__ W,
                         float* __restrict__ Y,
                         int N, int C, int Hi, int Wi, int Ho, int Wo,
                         int pT, int pL)
{
    __shared__ float Xs[18][18];
    __shared__ float Ws[9];

    int n = blockIdx.z / C;
    int c = blockIdx.z - n * C;
    int by = blockIdx.y * 16;
    int bx = blockIdx.x * 16;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;

    const float* Xnc = X + ((size_t)n * C + c) * Hi * Wi;

    // 256 threads load 18*18 = 324 elements: 1 per thread + 68 extras.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int lin = tid + i * 256;
        if (lin < 18 * 18) {
            int iy = lin / 18;
            int ix = lin % 18;
            int gy = by + iy - pT;
            int gx = bx + ix - pL;
            float v = 0.f;
            if ((unsigned)gy < (unsigned)Hi && (unsigned)gx < (unsigned)Wi)
                v = Xnc[(size_t)gy * Wi + gx];
            Xs[iy][ix] = v;
        }
    }
    if (tid < 9) Ws[tid] = W[(size_t)c * 9 + tid];
    __syncthreads();

    int oy = by + ty;
    int ox = bx + tx;
    if (oy < Ho && ox < Wo) {
        float acc = 0.f;
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                acc += Xs[ty + ky][tx + kx] * Ws[ky * 3 + kx];
            }
        }
        Y[((size_t)n * C + c) * Ho * Wo + (size_t)oy * Wo + ox] = acc;
    }
}
)CUDA";
}

// Depthwise 2D convolution kernel for group == Cin == Cout, W shape (C, 1, kH, kW).
// One thread per output element. No shared-memory tiling in this first version
// — memory-bound but eliminates the host-roundtrip + CPU fallback cost.
static const char* depthwise_source() {
    return R"CUDA(
extern "C" __global__
void depthwise_f32(const float* __restrict__ X,    // (N, C, Hi, Wi)
                   const float* __restrict__ W,    // (C, 1, kH, kW)
                   float* __restrict__ Y,          // (N, C, Ho, Wo)
                   int N, int C, int Hi, int Wi, int Ho, int Wo,
                   int kH, int kW, int sH, int sW,
                   int pT, int pL, int dH, int dW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Ho * Wo;
    if (idx >= total) return;

    int ow = idx % Wo;
    int t  = idx / Wo;
    int oh = t % Ho;           t  = t / Ho;
    int c  = t % C;            t  = t / C;
    int n  = t;

    const float* Wc = W + (size_t)c * kH * kW;
    const float* Xnc = X + ((size_t)n * C + c) * Hi * Wi;

    float acc = 0.f;
    for (int kr = 0; kr < kH; ++kr) {
        int ih = oh * sH - pT + kr * dH;
        if ((unsigned)ih >= (unsigned)Hi) continue;
        for (int kc = 0; kc < kW; ++kc) {
            int iw = ow * sW - pL + kc * dW;
            if ((unsigned)iw >= (unsigned)Wi) continue;
            acc += Xnc[(size_t)ih * Wi + iw] * Wc[kr * kW + kc];
        }
    }
    Y[((size_t)n * C + c) * Ho * Wo + oh * Wo + ow] = acc;
}
)CUDA";
}

struct Conv_cuda : public operator_t {
    bool prim_valid = false;
    bool is_depthwise = false;
    operator_t* fallback = nullptr;

    int N=0, Cin=0, Hi=0, Wi=0, Cout=0, Ho=0, Wo=0;
    int kH=0, kW=0, sH=1, sW=1, pT=0, pL=0, dH=1, dW=1;
    int group = 1;
    bool has_bias = false;

    // Fused post-op: 0=none, 1=Relu, 2=Clip(lo, hi).
    // Populated in reshape() when fused_op points to a compatible unary op.
    int   act_kind = 0;
    float act_lo = 0.f;
    float act_hi = 0.f;

    // Persistent device scratch for im2col (size = Cin*kH*kW*Ho*Wo floats).
    float* d_col = nullptr;
    size_t d_col_bytes = 0;

    bool init() override {
        fallback = resolver_default_op_Conv(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    // The CPU fallback (including fp16-as-fp32 trampoline) needs host workspace.
    size_t workspace_size() const override { return fallback ? fallback->workspace_size() : 0; }

    // TODO(T1): real CUDA Conv layout_cost — currently delegates to base
    // (estimate_costs → 0 for this op), so layout chain decisions on CUDA
    // models fall back to "no signal." A useful implementation models:
    //   - HBM read bytes:  input_bytes / nhwc_coalesce_util(C, layout)
    //                    + weight_bytes
    //                    + output_bytes
    //   - WMMA tile-util penalty: fp16 wants C aligned to 16, int8 to 32;
    //     misaligned C → tail-WMMA path (~0.6× peak throughput).
    //   - Kernel-launch overhead is layout-invariant — ignore.
    // Return units: device-relative bytes (NOT comparable to CPU
    // layout_cost). See nnr.h:412 ("Backend contract").
    // Comparable layouts on this op: NCHW, NHWC, NHWC_int8_aligned (TBD).
    float layout_cost(memory_layout_t layout, bool input_nhwc) const override {
        return operator_t::layout_cost(layout, input_nhwc);
    }

    bool reshape() override {
        fallback->post_fn = post_fn;
        fallback->fused_op = fused_op;
        if (!fallback->reshape()) return false;

        prim_valid = false; device_tag = 0;

        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        const tensor_t* y = outputs[0];
        if (x->type != NNR_DATA_TYPE_FLOAT32) return true;
        if (x->ndim != 4 || w->ndim != 4 || y->ndim != 4) return true;

        int64_t* ints = nullptr;
        int n = 0;

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

        N = x->dims[0]; Cin = x->dims[1]; Hi = x->dims[2]; Wi = x->dims[3];
        Cout = w->dims[0];
        kH = w->dims[2]; kW = w->dims[3];
        Ho = y->dims[2]; Wo = y->dims[3];
        if (y->dims[1] != Cout) return true;

        // Depthwise: group == Cin == Cout, W shape (Cin, 1, kH, kW).
        // Other grouped convs still fall back to CPU.
        is_depthwise = (group == Cin && group == Cout && w->dims[1] == 1);
        if (group != 1 && !is_depthwise) return true;
        if (group == 1 && w->dims[1] != Cin) return true;

        has_bias = (inputs.size() > 2 && inputs[2] && inputs[2]->data
                    && inputs[2]->type == NNR_DATA_TYPE_FLOAT32
                    && (int64_t)inputs[2]->ndata == Cout);

        // Decode fused unary post-op. fuse_post_ops already marked the consumer
        // as skipped; we're responsible for applying the activation on device.
        act_kind = 0;
        if (fused_op) {
            std::string_view t = fused_op->op_type;
            if (t == "Relu") {
                act_kind = 1;
            } else if (t == "Clip") {
                float lo = -FLT_MAX, hi = FLT_MAX;
                if (fused_op->inputs.size() > 1 && fused_op->inputs[1]
                    && fused_op->inputs[1]->data)
                    lo = *(const float*)fused_op->inputs[1]->data;
                if (fused_op->inputs.size() > 2 && fused_op->inputs[2]
                    && fused_op->inputs[2]->data)
                    hi = *(const float*)fused_op->inputs[2]->data;
                act_kind = 2;
                act_lo = lo;
                act_hi = hi;
            } else {
                // Unsupported fused op — fall back to CPU so correctness wins
                // over perf (post_fn is host-only in this branch).
                return true;
            }
        }

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    // Launch the fused bias + activation kernel over (N, Cout, spatial).
    // No-op when there is no bias and no activation.
    bool launch_postop(gpu::cuda_backend_t* be, float* d_y, int spatial) {
        if (!has_bias && act_kind == 0) return true;
        float* d_b = has_bias ? (float*)be->cache->ensure_device(inputs[2]) : nullptr;
        CUfunction f = be->nvrtc.get("nnr_conv_postop_f32",
                                     conv_postop_source(),
                                     "conv_postop_f32",
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) return false;
        int _N = N, _Cout = Cout, _sp = spatial;
        int _hb = has_bias ? 1 : 0;
        int _ak = act_kind;
        float _lo = act_lo, _hi = act_hi;
        void* args[] = { &d_y, &d_b, &_N, &_Cout, &_sp, &_hb, &_ak, &_lo, &_hi };
        unsigned long long total = (unsigned long long)N * Cout * spatial;
        unsigned block = 256;
        unsigned grid = (unsigned)((total + block - 1) / block);
        return gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args);
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_w = (float*)be->cache->ensure_device(inputs[1]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_w || !d_y) { return fallback->exec(); }

        if (is_depthwise) {
            // Fast path: 3×3 stride-1 tiled kernel (most mobilenet DW layers).
            bool is_3x3s1 = (kH == 3 && kW == 3 && sH == 1 && sW == 1
                          && dH == 1 && dW == 1);
            if (is_3x3s1) {
                CUfunction f = be->nvrtc.get("nnr_depthwise_3x3s1_f32",
                                             depthwise_3x3s1_source(),
                                             "depthwise_3x3s1_f32",
                                             gpu::nvrtc_arch_option(be->device));
                if (!f) { return fallback->exec(); }
                int _N = N, _C = Cin, _Hi = Hi, _Wi = Wi, _Ho = Ho, _Wo = Wo;
                int _pT = pT, _pL = pL;
                void* targs[] = {
                    (void*)&d_x, (void*)&d_w, (void*)&d_y,
                    &_N, &_C, &_Hi, &_Wi, &_Ho, &_Wo, &_pT, &_pL,
                };
                unsigned gx = (unsigned)((Wo + 15) / 16);
                unsigned gy = (unsigned)((Ho + 15) / 16);
                unsigned gz = (unsigned)(N * Cin);
                if (!gpu::nvrtc_launch(be->device, f, gx, gy, gz, 16, 16, 1, targs))
                    return fallback->exec();
            } else {
                CUfunction f_dw = be->nvrtc.get("nnr_depthwise_f32",
                                                depthwise_source(),
                                                "depthwise_f32",
                                                gpu::nvrtc_arch_option(be->device));
                if (!f_dw) { return fallback->exec(); }
                int _N = N, _C = Cin, _Hi = Hi, _Wi = Wi, _Ho = Ho, _Wo = Wo;
                int _kH = kH, _kW = kW, _sH = sH, _sW = sW;
                int _pT = pT, _pL = pL, _dH = dH, _dW = dW;
                void* dargs[] = {
                    (void*)&d_x, (void*)&d_w, (void*)&d_y,
                    &_N, &_C, &_Hi, &_Wi, &_Ho, &_Wo,
                    &_kH, &_kW, &_sH, &_sW, &_pT, &_pL, &_dH, &_dW,
                };
                unsigned long long total = (unsigned long long)N * Cin * Ho * Wo;
                unsigned block = 256;
                unsigned grid = (unsigned)((total + block - 1) / block);
                if (!gpu::nvrtc_launch(be->device, f_dw, grid, 1, 1, block, 1, 1, dargs))
                    return fallback->exec();
            }

            launch_postop(be, d_y, Ho * Wo);
            be->cache->mark_written(outputs[0]);
            return true;
        }

        CUfunction f_im2col = be->nvrtc.get("nnr_im2col_f32",
                                            im2col_source(),
                                            "im2col_f32",
                                            gpu::nvrtc_arch_option(be->device));
        if (!f_im2col) { return fallback->exec(); }

        // Scratch im2col buffer (per-batch slab).
        size_t col_elems = (size_t)Cin * kH * kW * Ho * Wo;
        size_t col_bytes = col_elems * sizeof(float);
        if (col_bytes > d_col_bytes) {
            if (d_col) be->device->free(d_col);
            d_col = (float*)be->device->alloc(col_bytes);
            d_col_bytes = col_bytes;
        }
        if (!d_col) { return fallback->exec(); }

        // Conv dims → GEMM: Y[Cout × M_gemm] = W[Cout × K] · col[K × M_gemm],
        // where M_gemm = Ho*Wo and K = Cin*kH*kW. Both row-major.
        const int M_gemm = Ho * Wo;
        const int K      = Cin * kH * kW;
        const int Ncout  = Cout;

        for (int b = 0; b < N; ++b) {
            const float* d_xb = d_x + (size_t)b * Cin * Hi * Wi;
            void* args[] = {
                (void*)&d_xb, (void*)&d_col,
                &Cin, &Hi, &Wi, &Ho, &Wo,
                &kH, &kW, &sH, &sW, &pT, &pL, &dH, &dW,
            };
            unsigned long long total = (unsigned long long)K * M_gemm;
            unsigned block = 256;
            unsigned grid = (unsigned)((total + block - 1) / block);
            if (!gpu::nvrtc_launch(be->device, f_im2col, grid, 1, 1, block, 1, 1, args)) {
                return fallback->exec();
            }

            // Y_batch[Cout × M_gemm] = W[Cout × K] · col[K × M_gemm]
            float* d_yb = d_y + (size_t)b * Cout * M_gemm;
            if (!gpu::gemm_device_f32(be, d_w, d_col, d_yb, Ncout, M_gemm, K, 0, 0))
                return fallback->exec();
        }

        launch_postop(be, d_y, M_gemm);
        be->cache->mark_written(outputs[0]);
        return true;
    }

    ~Conv_cuda() override {
        // d_col freed by device's memory pool on stream flush; leaving it to
        // backend teardown is safe (async pool releases on sync()).
    }
};

} // namespace

operator_t* resolver_cuda_op_Conv(int opset, pool_t& pool) {
    return pool_new<Conv_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
