#include "nnr.h"
#include "util.h"
#include "thread_pool.h"
#include <cmath>
#include <algorithm>
#ifdef NNR_ARCH_X64
#include <immintrin.h>
#endif

namespace nnr {

namespace {

// QLinearMul: quantized element-wise multiplication.
// Inputs: (A, A_scale, A_zp, B, B_scale, B_zp, C_scale, C_zp)
// Output: C  where C = Quantize((A - A_zp) * A_scale * (B - B_zp) * B_scale, C_scale, C_zp)
struct QLinearMul_operator : public operator_t {
    bool init() override {
        // Element-wise quantized multiply. SAME_SHAPE / scalar cases walk the
        // flat buffer so they're layout-agnostic. PER_CHANNEL has an explicit
        // NHWC branch below. GENERAL broadcasts fall back to the index-stride
        // path which is also layout-agnostic. So we can support both layouts.
        layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
        return inputs.size() == 8 && outputs.size() == 1;
    }

    bool reshape() override {
        return outputs[0]->reshape_multi_broadcast(inputs[0], inputs[3], inputs[0]->type);
    }

    template <typename T>
    bool exec_typed() {
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[3];
        tensor_t* y = outputs[0];
        float a_scale = *(float*)inputs[1]->data;
        float b_scale = *(float*)inputs[4]->data;
        float y_scale = *(float*)inputs[6]->data;

        int32_t a_zp = 0, b_zp = 0, y_zp = 0;
        if (inputs[2]->ndata > 0) a_zp = (int32_t)((T*)inputs[2]->data)[0];
        if (inputs[5]->ndata > 0) b_zp = (int32_t)((T*)inputs[5]->data)[0];
        if (inputs[7]->ndata > 0) y_zp = (int32_t)((T*)inputs[7]->data)[0];

        // combined_scale = a_scale * b_scale / y_scale
        float cs = a_scale * b_scale / y_scale;

        int clamp_min, clamp_max;
        if constexpr (std::is_same_v<T, uint8_t>) { clamp_min = 0; clamp_max = 255; }
        else { clamp_min = -128; clamp_max = 127; }

        const T* pa = (const T*)a->data;
        const T* pb = (const T*)b->data;
        T* py = (T*)y->data;
        size_t n = y->ndata;

        auto kind = classify_broadcast(a, b, y);
        if (kind == broadcast_kind::SAME_SHAPE) {
#ifdef NNR_ARCH_X64
            if (has_avx512()) {
                __m512 vcs = _mm512_set1_ps(cs);
                __m512 vazp = _mm512_set1_ps((float)a_zp);
                __m512 vbzp = _mm512_set1_ps((float)b_zp);
                __m512 vyzp = _mm512_set1_ps((float)y_zp);
                __m512 vmin = _mm512_set1_ps((float)clamp_min);
                __m512 vmax = _mm512_set1_ps((float)clamp_max);
                size_t i = 0;
                for (; i + 16 <= n; i += 16) {
                    __m512 fa, fb;
                    if constexpr (std::is_same_v<T, uint8_t>) {
                        fa = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(pa + i))));
                        fb = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(pb + i))));
                    } else {
                        fa = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(pa + i))));
                        fb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(pb + i))));
                    }
                    __m512 va = _mm512_sub_ps(fa, vazp);
                    __m512 vb = _mm512_sub_ps(fb, vbzp);
                    __m512 v = _mm512_fmadd_ps(_mm512_mul_ps(va, vb), vcs, vyzp);
                    v = _mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT);
                    v = _mm512_max_ps(_mm512_min_ps(v, vmax), vmin);
                    _mm_storeu_si128((__m128i*)(py + i),
                        _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v)));
                }
                for (; i < n; i++) {
                    float val = cs * (float)((int32_t)pa[i] - a_zp) * (float)((int32_t)pb[i] - b_zp) + (float)y_zp;
                    py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                }
                return true;
            }
#endif
            for (size_t i = 0; i < n; i++) {
                float val = cs * (float)((int32_t)pa[i] - a_zp) * (float)((int32_t)pb[i] - b_zp) + (float)y_zp;
                py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
            }
        } else if (kind == broadcast_kind::B_SCALAR) {
            float bval = cs * (float)((int32_t)pb[0] - b_zp);
            for (size_t i = 0; i < n; i++) {
                float val = bval * (float)((int32_t)pa[i] - a_zp) + (float)y_zp;
                py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
            }
        } else if (kind == broadcast_kind::A_SCALAR) {
            float aval = cs * (float)((int32_t)pa[0] - a_zp);
            for (size_t i = 0; i < n; i++) {
                float val = aval * (float)((int32_t)pb[i] - b_zp) + (float)y_zp;
                py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
            }
        } else if (kind == broadcast_kind::PER_CHANNEL) {
            // B is per-channel [1,C,1,1] or [C], A is full [N,C,H,W].
            // Swap if A is the per-channel source.
            const T* p_full = pa;
            const T* p_ch = pb;
            int32_t full_zp = a_zp, ch_zp = b_zp;
            if (is_per_channel_4d(a, y)) {
                p_full = pb; p_ch = pa;
                full_zp = b_zp; ch_zp = a_zp;
            }
            int N = y->dims[0], C = y->dims[1], HW = y->dims[2] * y->dims[3];
            int NC = N * C;
            bool par = (int64_t)NC * HW > (1 << 16);

            // NHWC branch: channels are the innermost stride-1 dim; per-channel
            // factor is applied with `c = i % C` inside the contiguous inner loop.
            // Pre-compute ch_val[c] once so the hot loop is a simple FMA chain.
            if (y->format == memory_layout_t::NHWC) {
                int NHW = N * HW;
                arena_scope_t scope(ctx->arena);
                float* ch_val = scope.alloc_arr<float>(C);
                for (int c = 0; c < C; c++)
                    ch_val[c] = cs * (float)((int32_t)p_ch[c] - ch_zp);
                bool par_nhwc = (int64_t)NHW * C > (1 << 16);
                nnr::for_dynamic(0, NHW, par_nhwc, [&](int, int pos) {
                    const T* src = p_full + (size_t)pos * C;
                    T* dst = py + (size_t)pos * C;
                    int c = 0;
#ifdef NNR_ARCH_X64
                    if (has_avx512()) {
                        __m512 vfzp = _mm512_set1_ps((float)full_zp);
                        __m512 vyzp = _mm512_set1_ps((float)y_zp);
                        __m512 vmin = _mm512_set1_ps((float)clamp_min);
                        __m512 vmax = _mm512_set1_ps((float)clamp_max);
                        for (; c + 16 <= C; c += 16) {
                            __m512 fa;
                            if constexpr (std::is_same_v<T, uint8_t>)
                                fa = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                    _mm_loadu_si128((const __m128i*)(src + c))));
                            else
                                fa = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                                    _mm_loadu_si128((const __m128i*)(src + c))));
                            __m512 vcv = _mm512_loadu_ps(ch_val + c);
                            __m512 v = _mm512_fmadd_ps(
                                _mm512_sub_ps(fa, vfzp), vcv, vyzp);
                            v = _mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT);
                            v = _mm512_max_ps(_mm512_min_ps(v, vmax), vmin);
                            _mm_storeu_si128((__m128i*)(dst + c),
                                _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v)));
                        }
                    }
#endif
                    for (; c < C; c++) {
                        float val = ch_val[c] * (float)((int32_t)src[c] - full_zp)
                                  + (float)y_zp;
                        dst[c] = (T)std::clamp(
                            (int32_t)std::nearbyint(val), clamp_min, clamp_max);
                    }
                });
                return true;
            }

#ifdef NNR_ARCH_X64
            if (has_avx512()) {
                nnr::for_dynamic(0, NC, par, [&](int, int nc) {
                    int c = nc % C;
                    float bval = cs * (float)((int32_t)p_ch[c] - ch_zp);
                    __m512 vbval = _mm512_set1_ps(bval);
                    __m512 vfzp = _mm512_set1_ps((float)full_zp);
                    __m512 vyzp = _mm512_set1_ps((float)y_zp);
                    __m512 vmin = _mm512_set1_ps((float)clamp_min);
                    __m512 vmax = _mm512_set1_ps((float)clamp_max);
                    const T* src = p_full + (size_t)nc * HW;
                    T* dst = py + (size_t)nc * HW;
                    int i = 0;
                    for (; i + 16 <= HW; i += 16) {
                        __m512 fa;
                        if constexpr (std::is_same_v<T, uint8_t>)
                            fa = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(src + i))));
                        else
                            fa = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(src + i))));
                        __m512 v = _mm512_fmadd_ps(_mm512_sub_ps(fa, vfzp), vbval, vyzp);
                        v = _mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT);
                        v = _mm512_max_ps(_mm512_min_ps(v, vmax), vmin);
                        _mm_storeu_si128((__m128i*)(dst + i),
                            _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v)));
                    }
                    for (; i < HW; i++) {
                        float val = bval * (float)((int32_t)src[i] - full_zp) + (float)y_zp;
                        dst[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                    }
                });
                return true;
            }
#endif
            nnr::for_dynamic(0, NC, par, [&](int, int nc) {
                int c = nc % C;
                float bval = cs * (float)((int32_t)p_ch[c] - ch_zp);
                const T* src = p_full + (size_t)nc * HW;
                T* dst = py + (size_t)nc * HW;
                for (int i = 0; i < HW; i++) {
                    float val = bval * (float)((int32_t)src[i] - full_zp) + (float)y_zp;
                    dst[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                }
            });
        } else {
            int a_bstr[MAX_NDIM], b_bstr[MAX_NDIM];
            compute_broadcast_strides(a, y, a_bstr);
            compute_broadcast_strides(b, y, b_bstr);
            int ndim = y->ndim;
            int idx[MAX_NDIM] = {};
            for (size_t i = 0; i < n; i++) {
                int ai = 0, bi = 0;
                for (int d = 0; d < ndim; d++) {
                    ai += idx[d] * a_bstr[d];
                    bi += idx[d] * b_bstr[d];
                }
                float val = cs * (float)((int32_t)pa[ai] - a_zp) * (float)((int32_t)pb[bi] - b_zp) + (float)y_zp;
                py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                for (int d = ndim - 1; d >= 0; d--) {
                    if (++idx[d] < y->dims[d]) break;
                    idx[d] = 0;
                }
            }
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        if (type == NNR_DATA_TYPE_UINT8) return exec_typed<uint8_t>();
        if (type == NNR_DATA_TYPE_INT8) return exec_typed<int8_t>();
        return false;
    }
};

} // namespace

// @nnr-meta-op mt=dynamic layout=[NCHW,NHWC]
operator_t* resolver_default_op_QLinearMul(int opset, pool_t& pool) {
    return pool_new<QLinearMul_operator>(pool);
}

} // namespace nnr
