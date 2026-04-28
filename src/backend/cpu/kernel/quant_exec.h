#pragma once
// Quantized execution utilities: bulk dequantize/quantize with SIMD,
// and generic wrappers for operators to process quantized tensors.
//
// Usage in an operator:
//   if (inputs[0]->is_quantized())
//       return exec_quantized_unary(this, [](float x) { return std::max(0.0f, x); });

#include "nnr.h"
#include "aligned_alloc.h"
#include <cmath>
#include <algorithm>

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#endif

namespace nnr {

// ── Bulk dequantize: int8/uint8/int16 → float32 ────────────────────────────

inline void dequantize_to_f32(float* __restrict dst, const void* __restrict src,
    data_type_t type, size_t n, float scale, int32_t zp)
{
    if (type == NNR_DATA_TYPE_UINT8) {
        const uint8_t* p = (const uint8_t*)src;
#ifdef NNR_ARCH_X64
        if (has_avx512()) {
            __m512 vs = _mm512_set1_ps(scale);
            __m512 vzp = _mm512_set1_ps((float)zp);
            size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                __m128i bytes = _mm_loadu_si128((const __m128i*)(p + i));
                __m512i words = _mm512_cvtepu8_epi32(bytes);
                __m512 fv = _mm512_cvtepi32_ps(words);
                fv = _mm512_mul_ps(_mm512_sub_ps(fv, vzp), vs);
                _mm512_storeu_ps(dst + i, fv);
            }
            for (; i < n; i++)
                dst[i] = ((float)p[i] - zp) * scale;
        } else if (detect_isa() >= isa_t::avx2) {
            __m256 vs = _mm256_set1_ps(scale);
            __m256 vzp = _mm256_set1_ps((float)zp);
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m128i bytes = _mm_loadl_epi64((const __m128i*)(p + i));
                __m256i words = _mm256_cvtepu8_epi32(bytes);
                __m256 fv = _mm256_cvtepi32_ps(words);
                fv = _mm256_mul_ps(_mm256_sub_ps(fv, vzp), vs);
                _mm256_storeu_ps(dst + i, fv);
            }
            for (; i < n; i++)
                dst[i] = ((float)p[i] - zp) * scale;
        } else {
            for (size_t i = 0; i < n; i++)
                dst[i] = ((float)p[i] - zp) * scale;
        }
#else
        for (size_t i = 0; i < n; i++)
            dst[i] = ((float)p[i] - zp) * scale;
#endif
    } else if (type == NNR_DATA_TYPE_INT8) {
        const int8_t* p = (const int8_t*)src;
#ifdef NNR_ARCH_X64
        if (has_avx512()) {
            __m512 vs = _mm512_set1_ps(scale);
            __m512 vzp = _mm512_set1_ps((float)zp);
            size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                __m128i bytes = _mm_loadu_si128((const __m128i*)(p + i));
                __m512i words = _mm512_cvtepi8_epi32(bytes);
                __m512 fv = _mm512_cvtepi32_ps(words);
                fv = _mm512_mul_ps(_mm512_sub_ps(fv, vzp), vs);
                _mm512_storeu_ps(dst + i, fv);
            }
            for (; i < n; i++)
                dst[i] = ((float)p[i] - zp) * scale;
        } else if (detect_isa() >= isa_t::avx2) {
            __m256 vs = _mm256_set1_ps(scale);
            __m256 vzp = _mm256_set1_ps((float)zp);
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m128i bytes = _mm_loadl_epi64((const __m128i*)(p + i));
                __m256i words = _mm256_cvtepi8_epi32(bytes);
                __m256 fv = _mm256_cvtepi32_ps(words);
                fv = _mm256_mul_ps(_mm256_sub_ps(fv, vzp), vs);
                _mm256_storeu_ps(dst + i, fv);
            }
            for (; i < n; i++)
                dst[i] = ((float)p[i] - zp) * scale;
        } else {
            for (size_t i = 0; i < n; i++)
                dst[i] = ((float)p[i] - zp) * scale;
        }
#else
        for (size_t i = 0; i < n; i++)
            dst[i] = ((float)p[i] - zp) * scale;
#endif
    } else if (type == NNR_DATA_TYPE_INT16) {
        const int16_t* p = (const int16_t*)src;
        for (size_t i = 0; i < n; i++)
            dst[i] = ((float)p[i] - zp) * scale;
    } else if (type == NNR_DATA_TYPE_UINT16) {
        const uint16_t* p = (const uint16_t*)src;
        for (size_t i = 0; i < n; i++)
            dst[i] = ((float)p[i] - zp) * scale;
    }
}

// ── Bulk quantize: float32 → int8/uint8/int16 ──────────────────────────────

inline void quantize_from_f32(void* __restrict dst, data_type_t type,
    const float* __restrict src, size_t n, float scale, int32_t zp)
{
    float inv_scale = 1.0f / scale;
    if (type == NNR_DATA_TYPE_UINT8) {
        uint8_t* p = (uint8_t*)dst;
#ifdef NNR_ARCH_X64
        if (has_avx512()) {
            __m512 vis = _mm512_set1_ps(inv_scale);
            __m512 vzp = _mm512_set1_ps((float)zp);
            __m512 vmin = _mm512_set1_ps(0.0f);
            __m512 vmax = _mm512_set1_ps(255.0f);
            size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                __m512 fv = _mm512_loadu_ps(src + i);
                fv = _mm512_add_ps(_mm512_roundscale_ps(
                    _mm512_mul_ps(fv, vis), _MM_FROUND_TO_NEAREST_INT), vzp);
                fv = _mm512_max_ps(_mm512_min_ps(fv, vmax), vmin);
                __m512i iv = _mm512_cvtps_epi32(fv);
                __m128i packed = _mm512_cvtepi32_epi8(iv);
                _mm_storeu_si128((__m128i*)(p + i), packed);
            }
            for (; i < n; i++) {
                int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
                p[i] = (uint8_t)std::clamp(v, 0, 255);
            }
        } else if (detect_isa() >= isa_t::avx2) {
            __m256 vis = _mm256_set1_ps(inv_scale);
            __m256 vzp = _mm256_set1_ps((float)zp);
            __m256 vmin = _mm256_set1_ps(0.0f);
            __m256 vmax = _mm256_set1_ps(255.0f);
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 fv = _mm256_loadu_ps(src + i);
                fv = _mm256_add_ps(_mm256_round_ps(
                    _mm256_mul_ps(fv, vis), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), vzp);
                fv = _mm256_max_ps(_mm256_min_ps(fv, vmax), vmin);
                __m256i iv = _mm256_cvtps_epi32(fv);
                __m128i lo = _mm256_castsi256_si128(iv);
                __m128i hi = _mm256_extracti128_si256(iv, 1);
                __m128i u16 = _mm_packus_epi32(lo, hi);
                __m128i u8  = _mm_packus_epi16(u16, u16);
                _mm_storel_epi64((__m128i*)(p + i), u8);
            }
            for (; i < n; i++) {
                int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
                p[i] = (uint8_t)std::clamp(v, 0, 255);
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
                p[i] = (uint8_t)std::clamp(v, 0, 255);
            }
        }
#else
        for (size_t i = 0; i < n; i++) {
            int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
            p[i] = (uint8_t)std::clamp(v, 0, 255);
        }
#endif
    } else if (type == NNR_DATA_TYPE_INT8) {
        int8_t* p = (int8_t*)dst;
#ifdef NNR_ARCH_X64
        if (has_avx512()) {
            __m512 vis = _mm512_set1_ps(inv_scale);
            __m512 vzp = _mm512_set1_ps((float)zp);
            __m512 vmin = _mm512_set1_ps(-128.0f);
            __m512 vmax = _mm512_set1_ps(127.0f);
            size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                __m512 fv = _mm512_loadu_ps(src + i);
                fv = _mm512_add_ps(_mm512_roundscale_ps(
                    _mm512_mul_ps(fv, vis), _MM_FROUND_TO_NEAREST_INT), vzp);
                fv = _mm512_max_ps(_mm512_min_ps(fv, vmax), vmin);
                __m512i iv = _mm512_cvtps_epi32(fv);
                __m128i packed = _mm512_cvtepi32_epi8(iv);
                _mm_storeu_si128((__m128i*)(p + i), packed);
            }
            for (; i < n; i++) {
                int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
                p[i] = (int8_t)std::clamp(v, -128, 127);
            }
        } else if (detect_isa() >= isa_t::avx2) {
            __m256 vis = _mm256_set1_ps(inv_scale);
            __m256 vzp = _mm256_set1_ps((float)zp);
            __m256 vmin = _mm256_set1_ps(-128.0f);
            __m256 vmax = _mm256_set1_ps(127.0f);
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 fv = _mm256_loadu_ps(src + i);
                fv = _mm256_add_ps(_mm256_round_ps(
                    _mm256_mul_ps(fv, vis), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), vzp);
                fv = _mm256_max_ps(_mm256_min_ps(fv, vmax), vmin);
                __m256i iv = _mm256_cvtps_epi32(fv);
                __m128i lo = _mm256_castsi256_si128(iv);
                __m128i hi = _mm256_extracti128_si256(iv, 1);
                __m128i s16 = _mm_packs_epi32(lo, hi);    // signed sat
                __m128i s8  = _mm_packs_epi16(s16, s16);
                _mm_storel_epi64((__m128i*)(p + i), s8);
            }
            for (; i < n; i++) {
                int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
                p[i] = (int8_t)std::clamp(v, -128, 127);
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
                p[i] = (int8_t)std::clamp(v, -128, 127);
            }
        }
#else
        for (size_t i = 0; i < n; i++) {
            int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
            p[i] = (int8_t)std::clamp(v, -128, 127);
        }
#endif
    } else if (type == NNR_DATA_TYPE_INT16) {
        int16_t* p = (int16_t*)dst;
        for (size_t i = 0; i < n; i++) {
            int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
            p[i] = (int16_t)std::clamp(v, -32768, 32767);
        }
    } else if (type == NNR_DATA_TYPE_UINT16) {
        uint16_t* p = (uint16_t*)dst;
        for (size_t i = 0; i < n; i++) {
            int32_t v = (int32_t)std::nearbyint(src[i] * inv_scale) + zp;
            p[i] = (uint16_t)std::clamp(v, 0, 65535);
        }
    }
}

// ── Generic quantized unary op: dequantize → fn → quantize ─────────────────
// Usage: exec_quantized_unary(this, [](float x) { return std::max(0.0f, x); });

template <typename Fn>
inline bool exec_quantized_unary(operator_t* op, Fn&& fn) {
    const tensor_t* x = op->inputs[0];
    tensor_t* y = op->outputs[0];
    if (!x->is_quantized()) return false;
    if (!x->data || !y->data || x->ndata == 0) return false;

    size_t n = x->ndata;
    float* tmp = (float*)op->ctx->workspace;

    dequantize_to_f32(tmp, x->data, x->type, n, x->quant_scale, x->quant_zero_point);

    for (size_t i = 0; i < n; i++)
        tmp[i] = fn(tmp[i]);

    quantize_from_f32(y->data, y->type, tmp, n, y->quant_scale, y->quant_zero_point);
    return true;
}

// ── Generic quantized binary op: dequantize both → fn → quantize ────────────

template <typename Fn>
inline bool exec_quantized_binary(operator_t* op, Fn&& fn) {
    const tensor_t* a = op->inputs[0];
    const tensor_t* b = op->inputs[1];
    tensor_t* y = op->outputs[0];
    if (!a->is_quantized() || !b->is_quantized()) return false;

    size_t na = a->ndata, nb = b->ndata, ny = y->ndata;
    size_t buf_size = (na + nb) * sizeof(float);
    float* tmp = (float*)op->ctx->workspace;
    float* a_f32 = tmp;
    float* b_f32 = tmp + na;

    dequantize_to_f32(a_f32, a->data, a->type, na, a->quant_scale, a->quant_zero_point);
    dequantize_to_f32(b_f32, b->data, b->type, nb, b->quant_scale, b->quant_zero_point);

    // Compute element-wise with broadcast
    // For now, simple non-broadcast case (same shape)
    for (size_t i = 0; i < ny; i++) {
        float av = a_f32[i % na];
        float bv = b_f32[i % nb];
        a_f32[i] = fn(av, bv);  // reuse a_f32 as output buffer
    }

    quantize_from_f32(y->data, y->type, a_f32, ny, y->quant_scale, y->quant_zero_point);
    return true;
}

// ── Quantized-as-float: dequantize input, run float exec, quantize output ───
// For operators that change shape (Reduce*, Pool, etc.) where the float exec
// path already has SIMD optimization. Caller provides exec_float function.
// Usage: exec_quantized_via_float(this, [this]() { return exec<float>(); });

template <typename ExecFloat>
inline bool exec_quantized_via_float(operator_t* op, ExecFloat&& exec_float) {
    const tensor_t* x = op->inputs[0];
    tensor_t* y = op->outputs[0];
    if (!x->is_quantized()) return false;

    // Save original input/output state
    void* orig_x_data = x->data;
    data_type_t orig_x_type = x->type;
    void* orig_y_data = y->data;
    data_type_t orig_y_type = y->type;

    // Allocate float buffers
    size_t nx = x->ndata;
    size_t ny = y->ndata;
    float* x_f32 = (float*)nnr_aligned_alloc(nx * sizeof(float), 64);
    float* y_f32 = (float*)nnr_aligned_alloc(ny * sizeof(float), 64);

    // Dequantize input
    dequantize_to_f32(x_f32, x->data, x->type, nx, x->quant_scale, x->quant_zero_point);

    // Temporarily swap tensor data/type to float
    const_cast<tensor_t*>(x)->data = x_f32;
    const_cast<tensor_t*>(x)->type = NNR_DATA_TYPE_FLOAT32;
    y->data = y_f32;
    y->type = NNR_DATA_TYPE_FLOAT32;

    // Run float exec
    bool ok = exec_float();

    // Restore original types
    const_cast<tensor_t*>(x)->data = orig_x_data;
    const_cast<tensor_t*>(x)->type = orig_x_type;
    y->type = orig_y_type;
    y->data = orig_y_data;

    // Quantize float output back to quantized type
    if (ok && y->is_quantized())
        quantize_from_f32(y->data, y->type, y_f32, ny, y->quant_scale, y->quant_zero_point);

    nnr_aligned_free(x_f32);
    nnr_aligned_free(y_f32);
    return ok;
}

} // namespace nnr
