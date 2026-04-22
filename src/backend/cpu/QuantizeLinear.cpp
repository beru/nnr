#include "nnr.h"
#include "nnrconf.h"
#include "util.h"
#include "thread_pool.h"
#include <cmath>
#include <algorithm>
#ifdef NNR_ARCH_X64
#include <immintrin.h>
#endif

namespace nnr {

namespace {

struct QuantizeLinear_operator : public operator_t {
    int axis = 1;
    int block_size = 0;
    data_type_t output_dtype = NNR_DATA_TYPE_UNDEFINED;

    bool init() override {
        if (inputs.size() < 2 || outputs.size() != 1)
            return false;
        axis = attribute(attr_key_t::axis, (int32_t)1);
        block_size = attribute(attr_key_t::block_size, (int32_t)0);
        output_dtype = (data_type_t)attribute(attr_key_t::output_dtype, (int32_t)NNR_DATA_TYPE_UNDEFINED);
        layout_mask = LAYOUT_ALL;  // element-wise, layout-agnostic
        return true;
    }

    bool reshape() override {
        // Output type priority: zero_point (if present) > output_dtype attr > uint8.
        data_type_t out_type = NNR_DATA_TYPE_UINT8;
        if (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0) {
            out_type = inputs[2]->type;
        } else if (output_dtype != NNR_DATA_TYPE_UNDEFINED) {
            out_type = output_dtype;
        }
        return outputs[0]->reshape_identity(inputs[0], out_type);
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        const tensor_t* y_scale = inputs[1];
        const tensor_t* y_zero = (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0) ? inputs[2] : nullptr;
        tensor_t* y = outputs[0];

        int caxis = axis;
        if (caxis < 0) caxis += x->ndim;

        // Float8 output types
        if (y->type == NNR_DATA_TYPE_FLOAT8E4M3FN || y->type == NNR_DATA_TYPE_FLOAT8E4M3FNUZ
         || y->type == NNR_DATA_TYPE_FLOAT8E5M2 || y->type == NNR_DATA_TYPE_FLOAT8E5M2FNUZ) {
            // decode zero_point from float8 to float32
            float fzero = 0.0f;
            if (y_zero) {
                uint8_t zb = ((const uint8_t*)y_zero->data)[0];
                switch (y->type) {
                case NNR_DATA_TYPE_FLOAT8E4M3FN:   fzero = float8e4m3fn_to_float32(zb); break;
                case NNR_DATA_TYPE_FLOAT8E4M3FNUZ: fzero = float8e4m3fnuz_to_float32(zb); break;
                case NNR_DATA_TYPE_FLOAT8E5M2:     fzero = float8e5m2_to_float32(zb); break;
                case NNR_DATA_TYPE_FLOAT8E5M2FNUZ: fzero = float8e5m2fnuz_to_float32(zb); break;
                default: break;
                }
            }
            float scale = ((const float*)y_scale->data)[0];
            const float* px = (const float*)x->data;
            uint8_t* py = (uint8_t*)y->data;
            if (y->type == NNR_DATA_TYPE_FLOAT8E4M3FN)
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = float32_to_float8e4m3fn(px[i] / scale + fzero, true);
            else if (y->type == NNR_DATA_TYPE_FLOAT8E4M3FNUZ)
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = float32_to_float8e4m3fnuz(px[i] / scale + fzero, true);
            else if (y->type == NNR_DATA_TYPE_FLOAT8E5M2)
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = float32_to_float8e5m2(px[i] / scale + fzero, true);
            else // FLOAT8E5M2FNUZ
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = float32_to_float8e5m2fnuz(px[i] / scale + fzero, true);
            return true;
        }

        if (y->type == NNR_DATA_TYPE_FLOAT4E2M1) {
            const float* px = (const float*)x->data;
            uint8_t* py = (uint8_t*)y->data;
            if (y_scale->ndata == 1) {
                float scale = ((const float*)y_scale->data)[0];
                float fzero = y_zero ? float4e2m1_to_float32(((const uint8_t*)y_zero->data)[0]) : 0.0f;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = float32_to_float4e2m1(px[i] / scale + fzero);
            } else {
                int axis_size = x->dims[caxis];
                int outer = 1, inner = 1;
                for (int i = 0; i < caxis; ++i) outer *= x->dims[i];
                for (int i = caxis + 1; i < x->ndim; ++i) inner *= x->dims[i];
                const float* scales = (const float*)y_scale->data;
                const uint8_t* zeros = y_zero ? (const uint8_t*)y_zero->data : nullptr;
                for (int o = 0; o < outer; ++o)
                    for (int a = 0; a < axis_size; ++a) {
                        float fzero = zeros ? float4e2m1_to_float32(zeros[a]) : 0.0f;
                        for (int i = 0; i < inner; ++i) {
                            int idx = (o * axis_size + a) * inner + i;
                            py[idx] = float32_to_float4e2m1(px[idx] / scales[a] + fzero);
                        }
                    }
            }
            return true;
        }

        if (y_scale->ndata == 1) {
            float scale = ((const float*)y_scale->data)[0];
            const float* px = (const float*)x->data;

            if (y->type == NNR_DATA_TYPE_UINT4) {
                uint8_t zero = y_zero ? ((const uint8_t*)y_zero->data)[0] : 0;
                uint8_t* py = (uint8_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = (uint8_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, 0, 15);
            } else if (y->type == NNR_DATA_TYPE_INT4) {
                int8_t zero = y_zero ? ((const int8_t*)y_zero->data)[0] : 0;
                int8_t* py = (int8_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = (int8_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, -8, 7);
            } else if (y->type == NNR_DATA_TYPE_UINT2) {
                uint8_t zero = y_zero ? ((const uint8_t*)y_zero->data)[0] : 0;
                uint8_t* py = (uint8_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = (uint8_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, 0, 3);
            } else if (y->type == NNR_DATA_TYPE_INT2) {
                int8_t zero = y_zero ? ((const int8_t*)y_zero->data)[0] : 0;
                int8_t* py = (int8_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = (int8_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, -2, 1);
            } else if (y->type == NNR_DATA_TYPE_UINT8) {
                uint8_t zero = y_zero ? ((const uint8_t*)y_zero->data)[0] : 0;
                uint8_t* py = (uint8_t*)y->data;
                size_t total = x->ndata;
#ifdef NNR_ARCH_X64
                if (has_avx512() && total >= 64) {
                    float inv_scale = 1.0f / scale;
                    // ORT-style cost-based threading: float→uint8 = 4B in, 1B out, 3 compute
                    int nt = nnr::elementwise_threads(total, 4, 1, 3);
                    constexpr size_t BLOCK = 4096;
                    int nblocks = (int)((total + BLOCK - 1) / BLOCK);
                    nnr::for_dynamic(0, nblocks, nt, [&](int /*tid*/, int blk) {
                        size_t base = (size_t)blk * BLOCK;
                        size_t end = std::min(base + BLOCK, total);
                        __m512 vis = _mm512_set1_ps(inv_scale);
                        __m512 vzp = _mm512_set1_ps((float)zero);
                        __m512 vmin = _mm512_setzero_ps();
                        __m512 vmax = _mm512_set1_ps(255.0f);
                        size_t i = base;
                        for (; i + 16 <= end; i += 16) {
                            __m512 v = _mm512_loadu_ps(px + i);
                            v = _mm512_add_ps(_mm512_roundscale_ps(
                                _mm512_mul_ps(v, vis), _MM_FROUND_TO_NEAREST_INT), vzp);
                            v = _mm512_max_ps(_mm512_min_ps(v, vmax), vmin);
                            _mm_storeu_si128((__m128i*)(py + i),
                                _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v)));
                        }
                        for (; i < end; i++)
                            py[i] = (uint8_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, 0, 255);
                    });
                } else
#endif
                for (size_t i = 0; i < total; ++i)
                    py[i] = (uint8_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, 0, 255);
            } else if (y->type == NNR_DATA_TYPE_INT8) {
                int8_t zero = y_zero ? ((const int8_t*)y_zero->data)[0] : 0;
                int8_t* py = (int8_t*)y->data;
                size_t total = x->ndata;
#ifdef NNR_ARCH_X64
                if (has_avx512() && total >= 64) {
                    float inv_scale = 1.0f / scale;
                    int nt = nnr::elementwise_threads(total, 4, 1, 3);
                    constexpr size_t BLOCK = 4096;
                    int nblocks = (int)((total + BLOCK - 1) / BLOCK);
                    nnr::for_dynamic(0, nblocks, nt, [&](int /*tid*/, int blk) {
                        size_t base = (size_t)blk * BLOCK;
                        size_t end = std::min(base + BLOCK, total);
                        __m512 vis = _mm512_set1_ps(inv_scale);
                        __m512 vzp = _mm512_set1_ps((float)zero);
                        __m512 vmin = _mm512_set1_ps(-128.0f);
                        __m512 vmax = _mm512_set1_ps(127.0f);
                        size_t i = base;
                        for (; i + 16 <= end; i += 16) {
                            __m512 v = _mm512_loadu_ps(px + i);
                            v = _mm512_add_ps(_mm512_roundscale_ps(
                                _mm512_mul_ps(v, vis), _MM_FROUND_TO_NEAREST_INT), vzp);
                            v = _mm512_max_ps(_mm512_min_ps(v, vmax), vmin);
                            _mm_storeu_si128((__m128i*)(py + i),
                                _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v)));
                        }
                        for (; i < end; i++)
                            py[i] = (int8_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, -128, 127);
                    });
                } else
#endif
                for (size_t i = 0; i < total; ++i)
                    py[i] = (int8_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, -128, 127);
            } else if (y->type == NNR_DATA_TYPE_UINT16) {
                uint16_t zero = y_zero ? ((const uint16_t*)y_zero->data)[0] : 0;
                uint16_t* py = (uint16_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = (uint16_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, 0, 65535);
            } else if (y->type == NNR_DATA_TYPE_INT16) {
                int16_t zero = y_zero ? ((const int16_t*)y_zero->data)[0] : 0;
                int16_t* py = (int16_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = (int16_t)std::clamp((int)std::nearbyint(px[i] / scale) + zero, -32768, 32767);
            }
            return true;
        }

        // Blocked quantization: y_scale has the same ndim as x, with dims[axis]
        // reduced by block_size. Scales/zero-points are broadcast over each
        // block of block_size contiguous elements along `axis`.
        if (block_size > 0 && y_scale->ndim == x->ndim
            && y_scale->dims[caxis] * block_size == x->dims[caxis]) {
            const float* scales = (const float*)y_scale->data;
            const float* px = (const float*)x->data;
            const int ndim = x->ndim;
            int scale_strides[MAX_NDIM];
            scale_strides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; --d)
                scale_strides[d] = scale_strides[d + 1] * y_scale->dims[d + 1];

            int x_idx[MAX_NDIM] = {0};
            auto advance = [&]() {
                for (int d = ndim - 1; d >= 0; --d) {
                    if (++x_idx[d] < x->dims[d]) return;
                    x_idx[d] = 0;
                }
            };
            auto scale_idx = [&]() {
                int s = 0;
                for (int d = 0; d < ndim; ++d) {
                    int c = (d == caxis) ? (x_idx[d] / block_size) : x_idx[d];
                    s += c * scale_strides[d];
                }
                return s;
            };

            if (y->type == NNR_DATA_TYPE_UINT8) {
                const uint8_t* zeros = y_zero ? (const uint8_t*)y_zero->data : nullptr;
                uint8_t* py = (uint8_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i, advance()) {
                    int si = scale_idx();
                    uint8_t z = zeros ? zeros[si] : 0;
                    py[i] = (uint8_t)std::clamp((int)std::nearbyint(px[i] / scales[si]) + z, 0, 255);
                }
            } else if (y->type == NNR_DATA_TYPE_INT8) {
                const int8_t* zeros = y_zero ? (const int8_t*)y_zero->data : nullptr;
                int8_t* py = (int8_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i, advance()) {
                    int si = scale_idx();
                    int8_t z = zeros ? zeros[si] : 0;
                    py[i] = (int8_t)std::clamp((int)std::nearbyint(px[i] / scales[si]) + z, -128, 127);
                }
            } else if (y->type == NNR_DATA_TYPE_UINT16) {
                const uint16_t* zeros = y_zero ? (const uint16_t*)y_zero->data : nullptr;
                uint16_t* py = (uint16_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i, advance()) {
                    int si = scale_idx();
                    uint16_t z = zeros ? zeros[si] : 0;
                    py[i] = (uint16_t)std::clamp((int)std::nearbyint(px[i] / scales[si]) + z, 0, 65535);
                }
            } else if (y->type == NNR_DATA_TYPE_INT16) {
                const int16_t* zeros = y_zero ? (const int16_t*)y_zero->data : nullptr;
                int16_t* py = (int16_t*)y->data;
                for (size_t i = 0; i < x->ndata; ++i, advance()) {
                    int si = scale_idx();
                    int16_t z = zeros ? zeros[si] : 0;
                    py[i] = (int16_t)std::clamp((int)std::nearbyint(px[i] / scales[si]) + z, -32768, 32767);
                }
            }
            return true;
        }

        // Per-axis
        int axis_size = x->dims[caxis];
        int outer = 1, inner = 1;
        for (int i = 0; i < caxis; ++i) outer *= x->dims[i];
        for (int i = caxis + 1; i < x->ndim; ++i) inner *= x->dims[i];

        const float* scales = (const float*)y_scale->data;
        const float* px = (const float*)x->data;

        if (y->type == NNR_DATA_TYPE_UINT4) {
            const uint8_t* zeros = y_zero ? (const uint8_t*)y_zero->data : nullptr;
            uint8_t* py = (uint8_t*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        uint8_t z = zeros ? zeros[a] : 0;
                        py[idx] = (uint8_t)std::clamp((int)std::nearbyint(px[idx] / scales[a]) + z, 0, 15);
                    }
        } else if (y->type == NNR_DATA_TYPE_INT4) {
            const int8_t* zeros = y_zero ? (const int8_t*)y_zero->data : nullptr;
            int8_t* py = (int8_t*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        int8_t z = zeros ? zeros[a] : 0;
                        py[idx] = (int8_t)std::clamp((int)std::nearbyint(px[idx] / scales[a]) + z, -8, 7);
                    }
        } else if (y->type == NNR_DATA_TYPE_UINT2) {
            const uint8_t* zeros = y_zero ? (const uint8_t*)y_zero->data : nullptr;
            uint8_t* py = (uint8_t*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        uint8_t z = zeros ? zeros[a] : 0;
                        py[idx] = (uint8_t)std::clamp((int)std::nearbyint(px[idx] / scales[a]) + z, 0, 3);
                    }
        } else if (y->type == NNR_DATA_TYPE_INT2) {
            const int8_t* zeros = y_zero ? (const int8_t*)y_zero->data : nullptr;
            int8_t* py = (int8_t*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        int8_t z = zeros ? zeros[a] : 0;
                        py[idx] = (int8_t)std::clamp((int)std::nearbyint(px[idx] / scales[a]) + z, -2, 1);
                    }
        } else if (y->type == NNR_DATA_TYPE_UINT8) {
            const uint8_t* zeros = y_zero ? (const uint8_t*)y_zero->data : nullptr;
            uint8_t* py = (uint8_t*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        uint8_t z = zeros ? zeros[a] : 0;
                        py[idx] = (uint8_t)std::clamp((int)std::nearbyint(px[idx] / scales[a]) + z, 0, 255);
                    }
        } else if (y->type == NNR_DATA_TYPE_INT8) {
            const int8_t* zeros = y_zero ? (const int8_t*)y_zero->data : nullptr;
            int8_t* py = (int8_t*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        int8_t z = zeros ? zeros[a] : 0;
                        py[idx] = (int8_t)std::clamp((int)std::nearbyint(px[idx] / scales[a]) + z, -128, 127);
                    }
        } else if (y->type == NNR_DATA_TYPE_UINT16) {
            const uint16_t* zeros = y_zero ? (const uint16_t*)y_zero->data : nullptr;
            uint16_t* py = (uint16_t*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        uint16_t z = zeros ? zeros[a] : 0;
                        py[idx] = (uint16_t)std::clamp((int)std::nearbyint(px[idx] / scales[a]) + z, 0, 65535);
                    }
        } else if (y->type == NNR_DATA_TYPE_INT16) {
            const int16_t* zeros = y_zero ? (const int16_t*)y_zero->data : nullptr;
            int16_t* py = (int16_t*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        int16_t z = zeros ? zeros[a] : 0;
                        py[idx] = (int16_t)std::clamp((int)std::nearbyint(px[idx] / scales[a]) + z, -32768, 32767);
                    }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=dynamic layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_QuantizeLinear(int opset, pool_t& pool)
{
    return pool_new<QuantizeLinear_operator>(pool);
}

} // namespace nnr
