#if defined(NNR_USE_CUDA)

// 2D NCHW Resize via NVRTC.
// Supported modes: "nearest" + "linear" (bilinear).
// Supported coordinate transforms: "half_pixel" and "asymmetric".
// Other combinations fall back to CPU.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

namespace nnr {

operator_t* resolver_default_op_Resize(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* resize_source() {
    return R"CUDA(
extern "C" {

// coord_mode: 0=half_pixel, 1=asymmetric, 2=pytorch_half_pixel
__global__ void resize_nearest_f32(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int N, int C, int Hi, int Wi, int Ho, int Wo,
                                   float sH, float sW, int coord_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Ho * Wo;
    if (idx >= total) return;

    int ow = idx % Wo; int t = idx / Wo;
    int oh = t   % Ho;     t = t   / Ho;
    int c  = t   % C;      t = t   / C;
    int n  = t;

    float fh, fw;
    if (coord_mode == 1) {         // asymmetric
        fh = oh * sH; fw = ow * sW;
    } else if (coord_mode == 2) {  // pytorch_half_pixel
        fh = (Ho > 1) ? ((oh + 0.5f) * sH - 0.5f) : 0.f;
        fw = (Wo > 1) ? ((ow + 0.5f) * sW - 0.5f) : 0.f;
    } else {                       // half_pixel
        fh = (oh + 0.5f) * sH - 0.5f;
        fw = (ow + 0.5f) * sW - 0.5f;
    }
    int ih = (int)floorf(fh + 0.5f);
    int iw = (int)floorf(fw + 0.5f);
    if (ih < 0) ih = 0; if (ih >= Hi) ih = Hi - 1;
    if (iw < 0) iw = 0; if (iw >= Wi) iw = Wi - 1;

    y[idx] = x[((size_t)n * C + c) * Hi * Wi + (size_t)ih * Wi + iw];
}

__global__ void resize_bilinear_f32(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int N, int C, int Hi, int Wi, int Ho, int Wo,
                                    float sH, float sW, int coord_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Ho * Wo;
    if (idx >= total) return;

    int ow = idx % Wo; int t = idx / Wo;
    int oh = t   % Ho;     t = t   / Ho;
    int c  = t   % C;      t = t   / C;
    int n  = t;

    float fh, fw;
    if (coord_mode == 1)      { fh = oh * sH;                  fw = ow * sW; }
    else if (coord_mode == 2) { fh = (Ho > 1) ? ((oh + 0.5f) * sH - 0.5f) : 0.f;
                                fw = (Wo > 1) ? ((ow + 0.5f) * sW - 0.5f) : 0.f; }
    else                      { fh = (oh + 0.5f) * sH - 0.5f;  fw = (ow + 0.5f) * sW - 0.5f; }

    int h0 = (int)floorf(fh), w0 = (int)floorf(fw);
    float dh = fh - (float)h0, dw = fw - (float)w0;
    int h1 = h0 + 1, w1 = w0 + 1;
    if (h0 < 0) { h0 = 0; dh = 0.f; } else if (h0 >= Hi - 1) { h0 = Hi - 1; h1 = h0; dh = 0.f; }
    else if (h1 >= Hi) h1 = Hi - 1;
    if (w0 < 0) { w0 = 0; dw = 0.f; } else if (w0 >= Wi - 1) { w0 = Wi - 1; w1 = w0; dw = 0.f; }
    else if (w1 >= Wi) w1 = Wi - 1;

    const float* xc = x + ((size_t)n * C + c) * Hi * Wi;
    float v00 = xc[h0 * Wi + w0];
    float v01 = xc[h0 * Wi + w1];
    float v10 = xc[h1 * Wi + w0];
    float v11 = xc[h1 * Wi + w1];
    float v0 = v00 + (v01 - v00) * dw;
    float v1 = v10 + (v11 - v10) * dw;
    y[idx] = v0 + (v1 - v0) * dh;
}

} // extern "C"
)CUDA";
}

struct Resize_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int mode = 0;        // 0 nearest, 1 bilinear
    int coord_mode = 0;  // 0 half_pixel, 1 asymmetric
    int N=0, C=0, Hi=0, Wi=0, Ho=0, Wo=0;
    float sH = 1.f, sW = 1.f;

    bool init() override {
        if (inputs.size() < 1 || outputs.size() != 1) return false;
        fallback = resolver_default_op_Resize(opset, ctx->attr_pool);
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
        if (x->type != NNR_DATA_TYPE_FLOAT32 || x->ndim != 4 || y->ndim != 4) return true;
        if (x->dims[0] != y->dims[0] || x->dims[1] != y->dims[1]) return true;

        std::string_view m = attribute(attr_key_t::mode, "nearest");
        if (m == "nearest")      mode = 0;
        else if (m == "linear")  mode = 1;
        else return true;

        std::string_view cm = attribute(attr_key_t::coordinate_transformation_mode, "half_pixel");
        if (cm == "half_pixel") coord_mode = 0;
        else if (cm == "asymmetric") coord_mode = 1;
        else if (cm == "pytorch_half_pixel") coord_mode = 2;  // kernel handles out_dim==1
        else return true;

        N = x->dims[0]; C = x->dims[1];
        Hi = x->dims[2]; Wi = x->dims[3];
        Ho = y->dims[2]; Wo = y->dims[3];
        // scale: input/output. Matches ONNX's definition: input_idx = output_idx * (Hi/Ho)
        sH = (Ho > 0) ? ((float)Hi / (float)Ho) : 1.f;
        sW = (Wo > 0) ? ((float)Wi / (float)Wo) : 1.f;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) { return fallback->exec(); }
        const char* kn = (mode == 0) ? "resize_nearest_f32" : "resize_bilinear_f32";
        CUfunction f = be->nvrtc.get("nnr_resize_f32", resize_source(),
                                     kn, gpu::nvrtc_arch_option(be->device));
        if (!f) { return fallback->exec(); }

        float* d_x = (float*)be->cache->ensure_device(inputs[0]);
        float* d_y = (float*)be->cache->alloc_output(outputs[0]);
        if (!d_x || !d_y) { return fallback->exec(); }

        unsigned long long total = (unsigned long long)N * C * Ho * Wo;
        unsigned block = 256;
        unsigned grid = (unsigned)((total + block - 1) / block);
        int _N = N, _C = C, _Hi = Hi, _Wi = Wi, _Ho = Ho, _Wo = Wo, _cm = coord_mode;
        float _sH = sH, _sW = sW;
        void* args[] = { &d_x, &d_y, &_N, &_C, &_Hi, &_Wi, &_Ho, &_Wo, &_sH, &_sW, &_cm };
        if (!gpu::nvrtc_launch(be->device, f, grid, 1, 1, block, 1, 1, args)) {
            return fallback->exec();
        }
        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_Resize(int opset, pool_t& pool) {
    return pool_new<Resize_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
