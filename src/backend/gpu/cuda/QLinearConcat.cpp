#if defined(NNR_USE_CUDA)

// QLinearConcat on CUDA — concatenate N quantized tensors along an axis,
// requantizing each to the output's scale/zp.
//
// ONNX contrib input layout:
//   inputs: (y_scale, y_zp, T1, T1_scale, T1_zp, T2, ..., TN, TN_scale, TN_zp)
// attribute: axis.
//
// Strategy: launch one requantize-and-copy kernel per input tensor. Each
// kernel writes a contiguous slab of Y at a fixed axis-offset. The slabs
// are non-overlapping so the launches can be serialized on a single stream
// without inter-kernel barriers. When (t_scale == y_scale && t_zp == y_zp)
// we skip the rescale math and just cudaMemcpyAsync the slice.
//
// Target: densenet-12-int8 dense-block feature concat (axis=1) on 4D uint8
// tensors. Kernel is layout-agnostic — works on any ndim and any axis as
// long as the (outer × chunk) decomposition holds.

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

#include <cstdint>

namespace nnr {

operator_t* resolver_default_op_QLinearConcat(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static const char* qconcat_source() {
    return R"CUDA(
extern "C" {

// Per-outer-slice requant + copy. grid.x covers positions within one slab.
// y_base + o * y_stride + pos  maps to  t_base + o * t_stride + pos.
#define QCONCAT_BODY(ELT, Y_MIN, Y_MAX)                                        \
    size_t pos  = (size_t)blockIdx.x * blockDim.x + threadIdx.x;               \
    size_t o    = blockIdx.y;                                                  \
    if (pos >= chunk) return;                                                  \
    const ELT* src = T + (size_t)o * chunk + pos;                              \
    ELT*       dst = Y + (size_t)o * y_stride + y_off_in_stride + pos;         \
    float v = rs * (float)((int)*src - t_zp) + (float)y_zp;                    \
    int q = (int)rintf(v);                                                     \
    if (q < (Y_MIN)) q = (Y_MIN);                                              \
    if (q > (Y_MAX)) q = (Y_MAX);                                              \
    *dst = (ELT)q;

__global__ void qconcat_u8(const unsigned char* __restrict__ T,
                           unsigned char* __restrict__ Y,
                           size_t chunk, size_t y_stride, size_t y_off_in_stride,
                           float rs, int t_zp, int y_zp)
{ QCONCAT_BODY(unsigned char, 0, 255) }

__global__ void qconcat_s8(const signed char* __restrict__ T,
                           signed char* __restrict__ Y,
                           size_t chunk, size_t y_stride, size_t y_off_in_stride,
                           float rs, int t_zp, int y_zp)
{ QCONCAT_BODY(signed char, -128, 127) }
#undef QCONCAT_BODY

} // extern "C"
)CUDA";
}

struct QLinearConcat_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;

    int axis = 0;
    int ntens = 0;
    bool is_uint8 = true;
    float y_scale = 0.f;
    int y_zp = 0;

    // Per-input slab info computed at reshape.
    struct slab_t {
        size_t chunk;              // dims-at-axis × inner
        size_t t_off_in_stride;    // unused here (t is contiguous)
        size_t y_off_in_stride;    // offset into the y outer-stride
        float  rs;                 // t_scale / y_scale, 1.0 if identical
        int    t_zp;
        bool   same_qparams;       // skip requant
        int    input_idx;          // tensor(i) index for exec
    };
    std::vector<slab_t> slabs;
    size_t outer_size = 0;
    size_t y_stride = 0;           // inner × output_axis_dim
    const char* kernel_name = nullptr;

    const tensor_t* tensor_at(int i) const { return inputs[2 + i * 3]; }

    bool init() override {
        if (inputs.size() < 5 || outputs.size() != 1) return false;
        if ((inputs.size() - 2) % 3 != 0) return false;
        fallback = resolver_default_op_QLinearConcat(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        axis = (int)attribute(attr_key_t::axis, (int64_t)0);
        ntens = (int)(inputs.size() - 2) / 3;
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;
        slabs.clear();

        tensor_t* Y = outputs[0];
        if (Y->type != NNR_DATA_TYPE_UINT8 && Y->type != NNR_DATA_TYPE_INT8) return true;
        is_uint8 = (Y->type == NNR_DATA_TYPE_UINT8);

        auto is_scalar_f32 = [](const tensor_t* t) {
            return t && t->type == NNR_DATA_TYPE_FLOAT32 && t->ndata == 1 && t->data;
        };
        auto read_zp = [](const tensor_t* t) -> int {
            if (!t || t->ndata == 0 || !t->data) return 0;
            if (t->type == NNR_DATA_TYPE_UINT8) return (int)*(const uint8_t*)t->data;
            if (t->type == NNR_DATA_TYPE_INT8)  return (int)*(const int8_t*)t->data;
            return INT32_MIN;
        };

        if (!is_scalar_f32(inputs[0])) return true;
        y_scale = *(const float*)inputs[0]->data;
        if (y_scale == 0.f) return true;
        y_zp = read_zp(inputs[1]);
        if (y_zp == INT32_MIN) return true;

        int a = axis < 0 ? axis + Y->ndim : axis;
        if (a < 0 || a >= Y->ndim) return true;

        outer_size = 1;
        for (int d = 0; d < a; ++d) outer_size *= (size_t)Y->dims[d];
        size_t inner_size = 1;
        for (int d = a + 1; d < Y->ndim; ++d) inner_size *= (size_t)Y->dims[d];
        y_stride = inner_size * (size_t)Y->dims[a];

        size_t y_off = 0;
        for (int i = 0; i < ntens; ++i) {
            const tensor_t* t = tensor_at(i);
            if (!t || t->type != Y->type) return true;
            if (!is_scalar_f32(inputs[2 + i * 3 + 1])) return true;
            float t_scale = *(const float*)inputs[2 + i * 3 + 1]->data;
            int t_zp = read_zp(inputs[2 + i * 3 + 2]);
            if (t_zp == INT32_MIN) return true;

            size_t chunk = (size_t)t->dims[a] * inner_size;
            slab_t s{};
            s.chunk = chunk;
            s.y_off_in_stride = y_off;
            s.t_zp = t_zp;
            s.rs = (t_scale == y_scale) ? 1.f : (t_scale / y_scale);
            s.same_qparams = (t_scale == y_scale && t_zp == y_zp);
            s.input_idx = i;
            slabs.push_back(s);
            y_off += chunk;
        }

        kernel_name = is_uint8 ? "qconcat_u8" : "qconcat_s8";

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) return fallback->exec();

        CUfunction f = be->nvrtc.get("nnr_qconcat", qconcat_source(),
                                     kernel_name, gpu::nvrtc_arch_option(be->device));
        if (!f) return fallback->exec();

        void* d_y = be->cache->alloc_output(outputs[0]);
        if (!d_y) return fallback->exec();
        size_t elt_size = is_uint8 ? 1 : 1;   // always 1 byte; kept for clarity
        (void)elt_size;

        for (auto& s : slabs) {
            void* d_t = be->cache->ensure_device(tensor_at(s.input_idx));
            if (!d_t) return fallback->exec();

            size_t _chunk  = s.chunk;
            size_t _ystr   = y_stride;
            size_t _yoff   = s.y_off_in_stride;
            float  _rs     = s.same_qparams ? 1.f : s.rs;
            int    _tzp    = s.same_qparams ? y_zp : s.t_zp;
            int    _yzp    = y_zp;
            void* args[] = { &d_t, &d_y, &_chunk, &_ystr, &_yoff,
                             &_rs, &_tzp, &_yzp };
            unsigned block = 256;
            unsigned grid_x = (unsigned)((_chunk + block - 1) / block);
            unsigned grid_y = (unsigned)outer_size;
            if (!gpu::nvrtc_launch(be->device, f, grid_x, grid_y, 1,
                                   block, 1, 1, args))
                return fallback->exec();
        }

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_QLinearConcat(int opset, pool_t& pool) {
    return pool_new<QLinearConcat_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
