#include "nnr.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include <immintrin.h>
#endif
#include "nnrconf.h"
#include "util.h"

namespace nnr {

namespace {

struct DequantizeLinear_operator : public operator_t {
    int axis = 1;
    int block_size = 0;

    bool init() override {
        if (inputs.size() < 2 || outputs.size() != 1)
            return false;
        axis = attribute(attr_key_t::axis, (int32_t)1);
        block_size = attribute(attr_key_t::block_size, (int32_t)0);
        layout_mask = LAYOUT_ALL;  // element-wise, layout-agnostic
        return true;
    }

    bool reshape() override {
        return outputs[0]->reshape_identity(inputs[0], inputs[1]->type);
    }

    // Decode one float4/float8 element to float32 based on type
    static float decode_float8(data_type_t t, uint8_t v) {
        switch (t) {
        case NNR_DATA_TYPE_FLOAT8E4M3FN:   return float8e4m3fn_to_float32(v);
        case NNR_DATA_TYPE_FLOAT8E4M3FNUZ: return float8e4m3fnuz_to_float32(v);
        case NNR_DATA_TYPE_FLOAT8E5M2:     return float8e5m2_to_float32(v);
        case NNR_DATA_TYPE_FLOAT8E5M2FNUZ: return float8e5m2fnuz_to_float32(v);
        case NNR_DATA_TYPE_FLOAT4E2M1:     return float4e2m1_to_float32(v);
        case NNR_DATA_TYPE_FLOAT8E8M0:     return float8e8m0_to_float32(v);
        default: return 0.0f;
        }
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        const tensor_t* x_scale = inputs[1];
        const tensor_t* x_zero = (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0) ? inputs[2] : nullptr;
        tensor_t* y = outputs[0];

        int caxis = axis;
        if (caxis < 0) caxis += x->ndim;

        // Float4/Float8 input: decode through float32, write to float32 or float16 output
        if (x->type == NNR_DATA_TYPE_FLOAT8E4M3FN || x->type == NNR_DATA_TYPE_FLOAT8E4M3FNUZ
         || x->type == NNR_DATA_TYPE_FLOAT8E5M2 || x->type == NNR_DATA_TYPE_FLOAT8E5M2FNUZ
         || x->type == NNR_DATA_TYPE_FLOAT4E2M1 || x->type == NNR_DATA_TYPE_FLOAT8E8M0) {
            const bool out_f16 = (y->type == NNR_DATA_TYPE_FLOAT16);
            const uint8_t* px = (const uint8_t*)x->data;
            auto read_scale = [&](size_t i) -> float {
                if (x_scale->type == NNR_DATA_TYPE_FLOAT16)
                    return float16_to_float32(((const uint16_t*)x_scale->data)[i]);
                return ((const float*)x_scale->data)[i];
            };
            auto write_out = [&](size_t i, float v) {
                if (out_f16) ((uint16_t*)y->data)[i] = float32_to_float16(v);
                else ((float*)y->data)[i] = v;
            };
            if (x_scale->ndata == 1) {
                float scale = read_scale(0);
                float zero = x_zero ? decode_float8(x->type, ((const uint8_t*)x_zero->data)[0]) : 0.0f;
                for (size_t i = 0; i < x->ndata; ++i)
                    write_out(i, (decode_float8(x->type, px[i]) - zero) * scale);
            } else {
                int axis_size = x->dims[caxis];
                int outer = 1, inner = 1;
                for (int i = 0; i < caxis; ++i) outer *= x->dims[i];
                for (int i = caxis + 1; i < x->ndim; ++i) inner *= x->dims[i];
                const uint8_t* zeros = x_zero ? (const uint8_t*)x_zero->data : nullptr;
                for (int o = 0; o < outer; ++o)
                    for (int a = 0; a < axis_size; ++a) {
                        float scale = read_scale(a);
                        float zero = zeros ? decode_float8(x->type, zeros[a]) : 0.0f;
                        for (int i = 0; i < inner; ++i) {
                            int idx = (o * axis_size + a) * inner + i;
                            write_out(idx, (decode_float8(x->type, px[idx]) - zero) * scale);
                        }
                    }
            }
            return true;
        }

        if (x_scale->ndata == 1) {
            float scale = ((const float*)x_scale->data)[0];

            if (x->type == NNR_DATA_TYPE_INT4 || x->type == NNR_DATA_TYPE_INT2) {
                const int8_t* px = (const int8_t*)x->data;
                int8_t zero = (x_zero) ? ((const int8_t*)x_zero->data)[0] : 0;
                float* py = (float*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = ((float)px[i] - (float)zero) * scale;
            } else if (x->type == NNR_DATA_TYPE_UINT4 || x->type == NNR_DATA_TYPE_UINT2) {
                const uint8_t* px = (const uint8_t*)x->data;
                uint8_t zero = (x_zero) ? ((const uint8_t*)x_zero->data)[0] : 0;
                float* py = (float*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = ((float)px[i] - (float)zero) * scale;
            } else if (x->type == NNR_DATA_TYPE_INT8) {
                const int8_t* px = (const int8_t*)x->data;
                int8_t zero = (x_zero) ? ((const int8_t*)x_zero->data)[0] : 0;
                float* py = (float*)y->data;
                size_t total = x->ndata;
#ifdef NNR_ARCH_X64
                if (has_avx512() && total >= 64) {
                    // ORT-style cost-based threading: int8→float = 1B in, 4B out, 2 compute
                    int nt = nnr::elementwise_threads(total, 1, 4, 2);
                    constexpr size_t BLOCK = 4096;
                    int nblocks = (int)((total + BLOCK - 1) / BLOCK);
                    nnr::for_dynamic(0, nblocks, nt, [&](int /*tid*/, int blk) {
                        size_t base = (size_t)blk * BLOCK;
                        size_t end = std::min(base + BLOCK, total);
                        __m512 vs = _mm512_set1_ps(scale);
                        __m512 vz = _mm512_set1_ps((float)zero);
                        size_t i = base;
                        for (; i + 16 <= end; i += 16) {
                            __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                                _mm_loadu_si128((const __m128i*)(px + i))));
                            _mm512_storeu_ps(py + i, _mm512_mul_ps(_mm512_sub_ps(v, vz), vs));
                        }
                        for (; i < end; i++)
                            py[i] = ((float)px[i] - (float)zero) * scale;
                    });
                } else
#endif
                for (size_t i = 0; i < total; ++i)
                    py[i] = ((float)px[i] - (float)zero) * scale;
            } else if (x->type == NNR_DATA_TYPE_UINT8) {
                const uint8_t* px = (const uint8_t*)x->data;
                uint8_t zero = (x_zero) ? ((const uint8_t*)x_zero->data)[0] : 0;
                float* py = (float*)y->data;
                size_t total = x->ndata;
#ifdef NNR_ARCH_X64
                if (has_avx512() && total >= 64) {
                    int nt = nnr::elementwise_threads(total, 1, 4, 2);
                    constexpr size_t BLOCK = 4096;
                    int nblocks = (int)((total + BLOCK - 1) / BLOCK);
                    nnr::for_dynamic(0, nblocks, nt, [&](int /*tid*/, int blk) {
                        size_t base = (size_t)blk * BLOCK;
                        size_t end = std::min(base + BLOCK, total);
                        __m512 vs = _mm512_set1_ps(scale);
                        __m512 vz = _mm512_set1_ps((float)zero);
                        size_t i = base;
                        for (; i + 16 <= end; i += 16) {
                            __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                _mm_loadu_si128((const __m128i*)(px + i))));
                            _mm512_storeu_ps(py + i, _mm512_mul_ps(_mm512_sub_ps(v, vz), vs));
                        }
                        for (; i < end; i++)
                            py[i] = ((float)px[i] - (float)zero) * scale;
                    });
                } else
#endif
                for (size_t i = 0; i < total; ++i)
                    py[i] = ((float)px[i] - (float)zero) * scale;
            } else if (x->type == NNR_DATA_TYPE_INT16) {
                const int16_t* px = (const int16_t*)x->data;
                int16_t zero = (x_zero) ? ((const int16_t*)x_zero->data)[0] : 0;
                float* py = (float*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = ((float)px[i] - (float)zero) * scale;
            } else if (x->type == NNR_DATA_TYPE_UINT16) {
                const uint16_t* px = (const uint16_t*)x->data;
                uint16_t zero = (x_zero) ? ((const uint16_t*)x_zero->data)[0] : 0;
                float* py = (float*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = ((float)px[i] - (float)zero) * scale;
            } else if (x->type == NNR_DATA_TYPE_INT32) {
                const int32_t* px = (const int32_t*)x->data;
                int32_t zero = (x_zero) ? ((const int32_t*)x_zero->data)[0] : 0;
                float* py = (float*)y->data;
                for (size_t i = 0; i < x->ndata; ++i)
                    py[i] = ((float)px[i] - (float)zero) * scale;
            }
            return true;
        }

        // Blocked dequantization: x_scale has the same ndim as x, with dims[axis]
        // reduced by block_size. Scales/zero-points broadcast over each block.
        if (block_size > 0 && x_scale->ndim == x->ndim
            && x_scale->dims[caxis] * block_size == x->dims[caxis]) {
            const float* scales = (const float*)x_scale->data;
            float* py = (float*)y->data;
            const int ndim = x->ndim;
            int scale_strides[MAX_NDIM];
            scale_strides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; --d)
                scale_strides[d] = scale_strides[d + 1] * x_scale->dims[d + 1];

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

            auto run = [&](auto tag) {
                using T = decltype(tag);
                const T* px = (const T*)x->data;
                const T* zeros = x_zero ? (const T*)x_zero->data : nullptr;
                for (size_t i = 0; i < x->ndata; ++i, advance()) {
                    int si = scale_idx();
                    T z = zeros ? zeros[si] : T(0);
                    py[i] = ((float)px[i] - (float)z) * scales[si];
                }
            };

            switch (x->type) {
            case NNR_DATA_TYPE_UINT8:  run((uint8_t)0); return true;
            case NNR_DATA_TYPE_INT8:   run((int8_t)0);  return true;
            case NNR_DATA_TYPE_UINT16: run((uint16_t)0); return true;
            case NNR_DATA_TYPE_INT16:  run((int16_t)0);  return true;
            case NNR_DATA_TYPE_INT32:  run((int32_t)0);  return true;
            default: break;
            }
        }

        int axis_size = x->dims[caxis];
        int outer = 1, inner = 1;
        for (int i = 0; i < caxis; ++i) outer *= x->dims[i];
        for (int i = caxis + 1; i < x->ndim; ++i) inner *= x->dims[i];

        const float* scales = (const float*)x_scale->data;

        if (x->type == NNR_DATA_TYPE_UINT4 || x->type == NNR_DATA_TYPE_UINT2) {
            const uint8_t* px = (const uint8_t*)x->data;
            const uint8_t* zeros = x_zero ? (const uint8_t*)x_zero->data : nullptr;
            float* py = (float*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        uint8_t z = zeros ? zeros[a] : 0;
                        py[idx] = ((float)px[idx] - (float)z) * scales[a];
                    }
        } else if (x->type == NNR_DATA_TYPE_INT4 || x->type == NNR_DATA_TYPE_INT2) {
            const int8_t* px = (const int8_t*)x->data;
            const int8_t* zeros = x_zero ? (const int8_t*)x_zero->data : nullptr;
            float* py = (float*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        int8_t z = zeros ? zeros[a] : 0;
                        py[idx] = ((float)px[idx] - (float)z) * scales[a];
                    }
        } else if (x->type == NNR_DATA_TYPE_UINT8) {
            const uint8_t* px = (const uint8_t*)x->data;
            const uint8_t* zeros = x_zero ? (const uint8_t*)x_zero->data : nullptr;
            float* py = (float*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        uint8_t z = zeros ? zeros[a] : 0;
                        py[idx] = ((float)px[idx] - (float)z) * scales[a];
                    }
        } else if (x->type == NNR_DATA_TYPE_INT8) {
            const int8_t* px = (const int8_t*)x->data;
            const int8_t* zeros = x_zero ? (const int8_t*)x_zero->data : nullptr;
            float* py = (float*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        int8_t z = zeros ? zeros[a] : 0;
                        py[idx] = ((float)px[idx] - (float)z) * scales[a];
                    }
        } else if (x->type == NNR_DATA_TYPE_INT16) {
            const int16_t* px = (const int16_t*)x->data;
            const int16_t* zeros = x_zero ? (const int16_t*)x_zero->data : nullptr;
            float* py = (float*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        int16_t z = zeros ? zeros[a] : 0;
                        py[idx] = ((float)px[idx] - (float)z) * scales[a];
                    }
        } else if (x->type == NNR_DATA_TYPE_UINT16) {
            const uint16_t* px = (const uint16_t*)x->data;
            const uint16_t* zeros = x_zero ? (const uint16_t*)x_zero->data : nullptr;
            float* py = (float*)y->data;
            for (int o = 0; o < outer; ++o)
                for (int a = 0; a < axis_size; ++a)
                    for (int i = 0; i < inner; ++i) {
                        int idx = (o * axis_size + a) * inner + i;
                        uint16_t z = zeros ? zeros[a] : 0;
                        py[idx] = ((float)px[idx] - (float)z) * scales[a];
                    }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=dynamic layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_DequantizeLinear(int opset, pool_t& pool)
{
    return pool_new<DequantizeLinear_operator>(pool);
}

} // namespace nnr
