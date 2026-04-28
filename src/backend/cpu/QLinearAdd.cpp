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

// QLinearAdd: quantized element-wise addition.
// Inputs: (A, A_scale, A_zp, B, B_scale, B_zp, C_scale, C_zp)
// Output: C  where C = Quantize((A - A_zp) * A_scale + (B - B_zp) * B_scale, C_scale, C_zp)
struct QLinearAdd_operator : public operator_t {
    bool init() override {
        // SAME_SHAPE / scalar cases walk the flat buffer (layout-agnostic).
        // PER_CHANNEL has an explicit NHWC branch below. GENERAL uses the
        // index-stride path which is also layout-agnostic.
        // Keep LAYOUT_ALL so assign_layouts' inter-chain propagation can walk
        // through this op freely.
        layout_mask = LAYOUT_ALL;
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

        float sa = a_scale / y_scale;
        float sb = b_scale / y_scale;

        int clamp_min, clamp_max;
        if constexpr (std::is_same_v<T, uint8_t>) { clamp_min = 0; clamp_max = 255; }
        else { clamp_min = -128; clamp_max = 127; }

        const T* pa = (const T*)a->data;
        const T* pb = (const T*)b->data;
        T* py = (T*)y->data;
        size_t n = y->ndata;

        auto kind = classify_broadcast(a, b, y);
        if (kind == broadcast_kind::SAME_SHAPE) {
            // Algebra: (a-a_zp)*sa + (b-b_zp)*sb + y_zp
            //       = a*sa + b*sb + [y_zp - a_zp*sa - b_zp*sb]
            // Hoist the bracketed constant out of the loop; inner body
            // becomes two FMAs. cvtps_epi32 uses MXCSR rounding
            // (round-to-nearest-even by default, matches nearbyint).
            // Saturating pack replaces the FP min/max + plain cvt.
            const float fixed = (float)y_zp
                - (sa * (float)a_zp + sb * (float)b_zp);
            // Cost-based threading (ORT-matched): 2B in + 1B out + ~4c compute.
            constexpr size_t BLOCK = 65536;
            int nblocks = (int)((n + BLOCK - 1) / BLOCK);
            int nt = nnr::elementwise_threads(n, 2, 1, 4);
            auto chunk_fn = [&](size_t i0, size_t i1) {
#ifdef NNR_ARCH_X64
                if (has_avx512()) {
                    const __m512 vsa = _mm512_set1_ps(sa);
                    const __m512 vsb = _mm512_set1_ps(sb);
                    const __m512 vfixed = _mm512_set1_ps(fixed);
                    size_t i = i0;
                    for (; i + 16 <= i1; i += 16) {
                        __m512 fa, fb;
                        if constexpr (std::is_same_v<T, uint8_t>) {
                            fa = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(pa + i))));
                            fb = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(pb + i))));
                        } else {
                            fa = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(pa + i))));
                            fb = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(pb + i))));
                        }
                        __m512 v = _mm512_fmadd_ps(fb, vsb, vfixed);
                        v = _mm512_fmadd_ps(fa, vsa, v);
                        __m512i vi = _mm512_cvtps_epi32(v);
                        __m128i out;
                        if constexpr (std::is_same_v<T, uint8_t>) {
                            // cvtusepi32_epi8 treats src as unsigned; clamp negatives to 0 first.
                            vi = _mm512_max_epi32(vi, _mm512_setzero_si512());
                            out = _mm512_cvtusepi32_epi8(vi);
                        } else {
                            out = _mm512_cvtsepi32_epi8(vi);
                        }
                        _mm_storeu_si128((__m128i*)(py + i), out);
                    }
                    for (; i < i1; i++) {
                        float val = sa * ((int32_t)pa[i] - a_zp) + sb * ((int32_t)pb[i] - b_zp) + (float)y_zp;
                        py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                    }
                    return;
                }
                if (detect_isa() >= isa_t::avx2) {
                    const __m256 vsa = _mm256_set1_ps(sa);
                    const __m256 vsb = _mm256_set1_ps(sb);
                    const __m256 vfixed = _mm256_set1_ps(fixed);
                    size_t i = i0;
                    for (; i + 8 <= i1; i += 8) {
                        __m256 fa, fb;
                        if constexpr (std::is_same_v<T, uint8_t>) {
                            fa = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(pa + i))));
                            fb = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(pb + i))));
                        } else {
                            fa = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(pa + i))));
                            fb = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(pb + i))));
                        }
                        __m256 v = _mm256_fmadd_ps(fb, vsb, vfixed);
                        v = _mm256_fmadd_ps(fa, vsa, v);
                        __m256i vi = _mm256_cvtps_epi32(v);
                        __m128i lo = _mm256_castsi256_si128(vi);
                        __m128i hi = _mm256_extracti128_si256(vi, 1);
                        __m128i out;
                        if constexpr (std::is_same_v<T, uint8_t>) {
                            __m128i u16 = _mm_packus_epi32(lo, hi);
                            out = _mm_packus_epi16(u16, u16);
                        } else {
                            __m128i s16 = _mm_packs_epi32(lo, hi);
                            out = _mm_packs_epi16(s16, s16);
                        }
                        _mm_storel_epi64((__m128i*)(py + i), out);
                    }
                    for (; i < i1; i++) {
                        float val = sa * ((int32_t)pa[i] - a_zp) + sb * ((int32_t)pb[i] - b_zp) + (float)y_zp;
                        py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                    }
                    return;
                }
#endif
                for (size_t i = i0; i < i1; i++) {
                    float val = sa * ((int32_t)pa[i] - a_zp) + sb * ((int32_t)pb[i] - b_zp) + (float)y_zp;
                    py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                }
            };
            if (nt <= 1) {
                chunk_fn(0, n);
            } else {
                nnr::for_dynamic(0, nblocks, nt, [&](int, int blk) {
                    size_t start = (size_t)blk * BLOCK;
                    size_t end = std::min(start + BLOCK, n);
                    chunk_fn(start, end);
                });
            }
            return true;
        } else if (kind == broadcast_kind::B_SCALAR) {
            float bval = sb * ((int32_t)pb[0] - b_zp);
            for (size_t i = 0; i < n; i++) {
                float val = sa * ((int32_t)pa[i] - a_zp) + bval + (float)y_zp;
                py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
            }
        } else if (kind == broadcast_kind::A_SCALAR) {
            float aval = sa * ((int32_t)pa[0] - a_zp);
            for (size_t i = 0; i < n; i++) {
                float val = aval + sb * ((int32_t)pb[i] - b_zp) + (float)y_zp;
                py[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
            }
        } else if (kind == broadcast_kind::PER_CHANNEL) {
            // B is per-channel [1,C,1,1] or [C], A is full [N,C,H,W].
            // Swap if A is the per-channel source.
            const T* p_full = pa;
            const T* p_ch = pb;
            float s_full = sa, s_ch = sb;
            int32_t full_zp = a_zp, ch_zp = b_zp;
            if (is_per_channel_4d(a, y)) {
                p_full = pb; p_ch = pa;
                s_full = sb; s_ch = sa;
                full_zp = b_zp; ch_zp = a_zp;
            }
            int N = y->dims[0], C = y->dims[1], HW = y->dims[2] * y->dims[3];
            int NC = N * C;
            bool par = (int64_t)NC * HW > (1 << 16);

            // NHWC branch: channels are innermost stride-1 dim. Precompute the
            // per-channel bias `ch_off[c] = s_ch * (p_ch[c] - ch_zp) + y_zp`
            // once, then the hot loop is one FMA chain across the contiguous
            // channel run for each (n,h,w) position.
            if (y->format == memory_layout_t::NHWC) {
                int NHW = N * HW;
                arena_scope_t scope(ctx->arena);
                float* ch_off = scope.alloc_arr<float>(C);
                for (int c = 0; c < C; c++)
                    ch_off[c] = s_ch * (float)((int32_t)p_ch[c] - ch_zp)
                              + (float)y_zp;
                bool par_nhwc = (int64_t)NHW * C > (1 << 16);
                nnr::for_dynamic(0, NHW, par_nhwc, [&](int, int pos) {
                    const T* src = p_full + (size_t)pos * C;
                    T* dst = py + (size_t)pos * C;
                    int c = 0;
#ifdef NNR_ARCH_X64
                    if (has_avx512()) {
                        __m512 vsf = _mm512_set1_ps(s_full);
                        __m512 vfzp = _mm512_set1_ps((float)full_zp);
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
                            __m512 voff = _mm512_loadu_ps(ch_off + c);
                            __m512 v = _mm512_fmadd_ps(
                                _mm512_sub_ps(fa, vfzp), vsf, voff);
                            v = _mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT);
                            v = _mm512_max_ps(_mm512_min_ps(v, vmax), vmin);
                            _mm_storeu_si128((__m128i*)(dst + c),
                                _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v)));
                        }
                    } else if (detect_isa() >= isa_t::avx2) {
                        __m256 vsf = _mm256_set1_ps(s_full);
                        __m256 vfzp = _mm256_set1_ps((float)full_zp);
                        __m256 vmin = _mm256_set1_ps((float)clamp_min);
                        __m256 vmax = _mm256_set1_ps((float)clamp_max);
                        for (; c + 8 <= C; c += 8) {
                            __m256 fa;
                            if constexpr (std::is_same_v<T, uint8_t>)
                                fa = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                                    _mm_loadl_epi64((const __m128i*)(src + c))));
                            else
                                fa = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                    _mm_loadl_epi64((const __m128i*)(src + c))));
                            __m256 voff = _mm256_loadu_ps(ch_off + c);
                            __m256 v = _mm256_fmadd_ps(
                                _mm256_sub_ps(fa, vfzp), vsf, voff);
                            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                            v = _mm256_max_ps(_mm256_min_ps(v, vmax), vmin);
                            __m256i vi = _mm256_cvtps_epi32(v);
                            __m128i lo = _mm256_castsi256_si128(vi);
                            __m128i hi = _mm256_extracti128_si256(vi, 1);
                            __m128i out;
                            if constexpr (std::is_same_v<T, uint8_t>) {
                                __m128i u16 = _mm_packus_epi32(lo, hi);
                                out = _mm_packus_epi16(u16, u16);
                            } else {
                                __m128i s16 = _mm_packs_epi32(lo, hi);
                                out = _mm_packs_epi16(s16, s16);
                            }
                            _mm_storel_epi64((__m128i*)(dst + c), out);
                        }
                    }
#endif
                    for (; c < C; c++) {
                        float val = s_full * (float)((int32_t)src[c] - full_zp)
                                  + ch_off[c];
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
                    float ch_val = s_ch * (float)((int32_t)p_ch[c] - ch_zp);
                    __m512 vs_full = _mm512_set1_ps(s_full);
                    __m512 vfzp = _mm512_set1_ps((float)full_zp);
                    __m512 voff = _mm512_set1_ps(ch_val + (float)y_zp);
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
                        __m512 v = _mm512_fmadd_ps(_mm512_sub_ps(fa, vfzp), vs_full, voff);
                        v = _mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT);
                        v = _mm512_max_ps(_mm512_min_ps(v, vmax), vmin);
                        _mm_storeu_si128((__m128i*)(dst + i),
                            _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v)));
                    }
                    for (; i < HW; i++) {
                        float val = s_full * ((int32_t)src[i] - full_zp) + ch_val + (float)y_zp;
                        dst[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                    }
                });
                return true;
            }
            if (detect_isa() >= isa_t::avx2) {
                nnr::for_dynamic(0, NC, par, [&](int, int nc) {
                    int c = nc % C;
                    float ch_val = s_ch * (float)((int32_t)p_ch[c] - ch_zp);
                    __m256 vs_full = _mm256_set1_ps(s_full);
                    __m256 vfzp = _mm256_set1_ps((float)full_zp);
                    __m256 voff = _mm256_set1_ps(ch_val + (float)y_zp);
                    __m256 vmin = _mm256_set1_ps((float)clamp_min);
                    __m256 vmax = _mm256_set1_ps((float)clamp_max);
                    const T* src = p_full + (size_t)nc * HW;
                    T* dst = py + (size_t)nc * HW;
                    int i = 0;
                    for (; i + 8 <= HW; i += 8) {
                        __m256 fa;
                        if constexpr (std::is_same_v<T, uint8_t>)
                            fa = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(src + i))));
                        else
                            fa = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(src + i))));
                        __m256 v = _mm256_fmadd_ps(_mm256_sub_ps(fa, vfzp), vs_full, voff);
                        v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        v = _mm256_max_ps(_mm256_min_ps(v, vmax), vmin);
                        __m256i vi = _mm256_cvtps_epi32(v);
                        __m128i lo = _mm256_castsi256_si128(vi);
                        __m128i hi = _mm256_extracti128_si256(vi, 1);
                        __m128i out;
                        if constexpr (std::is_same_v<T, uint8_t>) {
                            __m128i u16 = _mm_packus_epi32(lo, hi);
                            out = _mm_packus_epi16(u16, u16);
                        } else {
                            __m128i s16 = _mm_packs_epi32(lo, hi);
                            out = _mm_packs_epi16(s16, s16);
                        }
                        _mm_storel_epi64((__m128i*)(dst + i), out);
                    }
                    for (; i < HW; i++) {
                        float val = s_full * ((int32_t)src[i] - full_zp) + ch_val + (float)y_zp;
                        dst[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                    }
                });
                return true;
            }
#endif
            nnr::for_dynamic(0, NC, par, [&](int, int nc) {
                int c = nc % C;
                float ch_val = s_ch * (float)((int32_t)p_ch[c] - ch_zp);
                const T* src = p_full + (size_t)nc * HW;
                T* dst = py + (size_t)nc * HW;
                for (int i = 0; i < HW; i++) {
                    float val = s_full * ((int32_t)src[i] - full_zp) + ch_val + (float)y_zp;
                    dst[i] = (T)std::clamp((int32_t)std::nearbyint(val), clamp_min, clamp_max);
                }
            });
        } else {
            // General broadcast
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
                float val = sa * ((int32_t)pa[ai] - a_zp) + sb * ((int32_t)pb[bi] - b_zp) + (float)y_zp;
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

// @nnr-meta-op mt=dynamic layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8]
operator_t* resolver_default_op_QLinearAdd(int opset, pool_t& pool) {
    return pool_new<QLinearAdd_operator>(pool);
}

} // namespace nnr
