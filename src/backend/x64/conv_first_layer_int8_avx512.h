#pragma once
// Direct int8 first-layer Conv (IC <= 4, e.g. RGB) with AVX-512 VNNI.
// Skips im2col. Mirrors conv_first_layer_avx512.h structure but for
// uint8 input / int8 weights / uint8 output via VPDPBUSD.
//
// Algorithm:
//  - Pre-pad input as interleaved [pH][pW][4] uint8. Pad IC to 4 with zeros
//    so unused channel lanes contribute nothing to VPDPBUSD (zero weight).
//    Border fill = (x_zp, x_zp, x_zp, 0) for IC=3 — one dword per pixel.
//  - Loop: outer = output rows (parallel), inner = tile of 14 output pixels.
//    Per (kh, kw): one VPDPBUSD per tile slot. 16 OCs per ZMM accumulator.
//  - After row finished, transpose 16x16 int32 tile (ow-major → oc-major)
//    via existing transpose_16x16_avx512 helper, requantize each 16-ow
//    row (per-OC scale + bias + zp compensation), store uint8 to NCHW y.
//
// Eligibility is decided at reshape time. Kernel itself asserts IC<=4.

#ifdef NNR_ARCH_X64

#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include "thread_pool.h"
#include "cpu_features.h"
#include "backend/x64/layout_x64.h"  // transpose_16x16_avx512

namespace nnr {

// ---------- weight packing ----------
// Packed layout: [KH][KW][OC_blocks][16 lanes][4 bytes] = 64 bytes per (kh,kw,ob).
// dst[((kh*KW+kw)*OC_blocks + ob) * 64 + lane*4 + ic] = w[ob*16+lane, ic, kh, kw]
// Unused (ic >= IC) lanes are zero.
// @nnr-meta isa=scalar dtype=[int8,uint8]
inline size_t pack_weights_first_layer_int8_size(int OC, int IC, int KH, int KW)
{
    (void)IC;
    return (size_t)KH * KW * ((OC + 15) / 16) * 64;
}

// @nnr-meta isa=scalar dtype=[int8,uint8]
inline void pack_weights_first_layer_int8(
    int8_t* __restrict dst,
    const int8_t* __restrict src,  // [OC, IC, KH, KW]
    int OC, int IC, int KH, int KW)
{
    const int OC_blocks = (OC + 15) / 16;
    const int kSpatial = KH * KW;
    memset(dst, 0, pack_weights_first_layer_int8_size(OC, IC, KH, KW));
    for (int oc = 0; oc < OC; oc++) {
        int ob = oc / 16, lane = oc % 16;
        for (int ic = 0; ic < IC; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    size_t didx = ((size_t)(kh * KW + kw) * OC_blocks + ob) * 64
                                  + lane * 4 + ic;
                    dst[didx] = src[((size_t)oc * IC + ic) * kSpatial + kh * KW + kw];
                }
            }
        }
    }
}

// Per-OC weight sum (int32). Needed for x_zp compensation.
// @nnr-meta isa=scalar dtype=int8
inline void compute_weight_sums_first_layer_int8(
    int32_t* __restrict w_sum,
    const int8_t* __restrict src,
    int OC, int IC, int KH, int KW)
{
    const int CHW = IC * KH * KW;
    for (int oc = 0; oc < OC; oc++) {
        int32_t s = 0;
        const int8_t* row = src + (size_t)oc * CHW;
        for (int k = 0; k < CHW; k++) s += (int32_t)row[k];
        w_sum[oc] = s;
    }
}

// ---------- main kernel ----------
// Returns false if AVX-512 VNNI is unavailable (caller should fall back).
// output_nhwc = false: y laid out as NCHW [OC, oH, oW]
// output_nhwc = true : y laid out as NHWC [oH, oW, OC] (OC contiguous)
// @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NCHW special=FirstLayer fusion=qdq
inline bool conv_first_layer_int8_avx512(
    uint8_t* __restrict y,              // NCHW or NHWC per output_nhwc
    const uint8_t* __restrict x,        // [IC, iH, iW] NCHW uint8
    const int8_t* __restrict w_packed,  // pack_weights_first_layer_int8 output
    const int32_t* __restrict w_row_sum,// [OC] per-OC weight sum
    const int32_t* __restrict bias,     // [OC] int32 bias, may be nullptr
    const float* __restrict w_scale,    // per-OC weight scale (w_scale_count>=1)
    int w_scale_count,                  // 1 (per-tensor) or OC (per-channel)
    float x_scale,
    float inv_y_scale,
    int32_t x_zp,
    int32_t y_zp,
    int32_t qmin,
    int32_t qmax,
    int IC, int iH, int iW,
    int OC, int oH, int oW,
    int KH, int KW,
    int sH, int sW,
    int padH, int padW,
    bool output_nhwc = false)
{
    if (!has_avx512() || !cpu_features().avx512vnni) return false;
    if (IC <= 0 || IC > 4) return false;

    const int OC_blocks = (OC + 15) / 16;
    const int spatial = oH * oW;
    constexpr int WT = 14;

    const int pH = iH + 2 * padH;
    const int pW = iW + 2 * padW;

    // Pre-padded input, interleaved 4 ICs per pixel (dword load per pixel).
    // Layout: [pH * pW * 4] bytes. Lives at front of NNR_POOL_SCRATCH(0);
    // per-thread row scratch follows immediately after (wasted space on
    // non-zero threads, but matches conv_first_layer_avx512 pattern).
    const size_t prepad_bytes = (size_t)pH * pW * 4;
    // Per-thread row buf holds int32 accumulators [ob][ow][16 lanes].
    const size_t row_bytes = (size_t)OC_blocks * oW * 16 * sizeof(int32_t);
    NNR_POOL_ENSURE_SCRATCH(prepad_bytes + row_bytes);

    // Fill prepadded buffer on main thread.
    {
        uint8_t* prepad_w = (uint8_t*)NNR_POOL_SCRATCH(0);
        // Border fill: (x_zp repeated in real IC lanes, 0 in padded lanes).
        uint8_t fill[4] = {0, 0, 0, 0};
        for (int ic = 0; ic < IC; ic++) fill[ic] = (uint8_t)x_zp;
        uint32_t fill_dw;
        memcpy(&fill_dw, fill, 4);
        uint32_t* pd = (uint32_t*)prepad_w;
        size_t n = (size_t)pH * pW;
        for (size_t i = 0; i < n; i++) pd[i] = fill_dw;

        // Copy interior pixels. Interior lanes: real ICs from x, padded lanes 0.
        for (int ih = 0; ih < iH; ih++) {
            for (int iw = 0; iw < iW; iw++) {
                uint8_t* dstp = prepad_w + ((size_t)(ih + padH) * pW + (iw + padW)) * 4;
                for (int ic = 0; ic < IC; ic++)
                    dstp[ic] = x[((size_t)ic * iH + ih) * iW + iw];
                for (int ic = IC; ic < 4; ic++) dstp[ic] = 0;
            }
        }
    }
    const uint8_t* prepad = (const uint8_t*)NNR_POOL_SCRATCH(0);

    nnr::for_dynamic(0, oH, oH >= 4, [&](int tid, int oh) {
        int32_t* buf = (int32_t*)((uint8_t*)NNR_POOL_SCRATCH(tid) + prepad_bytes);

        // ---- Conv pass: fill buf[ob][ow][16 lanes] for this row ----
        for (int ob = 0; ob < OC_blocks; ob++) {
            int32_t* out_ob = buf + (size_t)ob * oW * 16;

            int ow = 0;
            for (; ow + WT <= oW; ow += WT) {
                __m512i a0  = _mm512_setzero_si512();
                __m512i a1  = _mm512_setzero_si512();
                __m512i a2  = _mm512_setzero_si512();
                __m512i a3  = _mm512_setzero_si512();
                __m512i a4  = _mm512_setzero_si512();
                __m512i a5  = _mm512_setzero_si512();
                __m512i a6  = _mm512_setzero_si512();
                __m512i a7  = _mm512_setzero_si512();
                __m512i a8  = _mm512_setzero_si512();
                __m512i a9  = _mm512_setzero_si512();
                __m512i a10 = _mm512_setzero_si512();
                __m512i a11 = _mm512_setzero_si512();
                __m512i a12 = _mm512_setzero_si512();
                __m512i a13 = _mm512_setzero_si512();

                for (int kh = 0; kh < KH; kh++) {
                    int ih_p = oh * sH + kh;
                    const uint8_t* x_row = prepad + (size_t)ih_p * pW * 4;
                    for (int kw = 0; kw < KW; kw++) {
                        const __m512i wv = _mm512_loadu_si512(
                            w_packed + ((size_t)(kh * KW + kw) * OC_blocks + ob) * 64);

                        #define NNR_DP(T) { \
                            int32_t xv; \
                            memcpy(&xv, x_row + ((ow + (T)) * sW + kw) * 4, 4); \
                            a##T = _mm512_dpbusd_epi32(a##T, \
                                _mm512_set1_epi32(xv), wv); \
                        }
                        NNR_DP(0)  NNR_DP(1)  NNR_DP(2)  NNR_DP(3)
                        NNR_DP(4)  NNR_DP(5)  NNR_DP(6)  NNR_DP(7)
                        NNR_DP(8)  NNR_DP(9)  NNR_DP(10) NNR_DP(11)
                        NNR_DP(12) NNR_DP(13)
                        #undef NNR_DP
                    }
                }

                _mm512_storeu_si512(out_ob + (size_t)(ow +  0) * 16, a0);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  1) * 16, a1);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  2) * 16, a2);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  3) * 16, a3);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  4) * 16, a4);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  5) * 16, a5);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  6) * 16, a6);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  7) * 16, a7);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  8) * 16, a8);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  9) * 16, a9);
                _mm512_storeu_si512(out_ob + (size_t)(ow + 10) * 16, a10);
                _mm512_storeu_si512(out_ob + (size_t)(ow + 11) * 16, a11);
                _mm512_storeu_si512(out_ob + (size_t)(ow + 12) * 16, a12);
                _mm512_storeu_si512(out_ob + (size_t)(ow + 13) * 16, a13);
            }

            // Remainder ow (< WT)
            for (; ow < oW; ow++) {
                __m512i a = _mm512_setzero_si512();
                for (int kh = 0; kh < KH; kh++) {
                    int ih_p = oh * sH + kh;
                    const uint8_t* x_row = prepad + (size_t)ih_p * pW * 4;
                    for (int kw = 0; kw < KW; kw++) {
                        const __m512i wv = _mm512_loadu_si512(
                            w_packed + ((size_t)(kh * KW + kw) * OC_blocks + ob) * 64);
                        int32_t xv;
                        memcpy(&xv, x_row + (ow * sW + kw) * 4, 4);
                        a = _mm512_dpbusd_epi32(a, _mm512_set1_epi32(xv), wv);
                    }
                }
                _mm512_storeu_si512(out_ob + (size_t)ow * 16, a);
            }
        }

        // ---- Requantize + store ----
        const __m512 vis   = _mm512_set1_ps(inv_y_scale);
        const __m512 vyzp  = _mm512_set1_ps((float)y_zp);
        const __m512 vqmin = _mm512_set1_ps((float)qmin);
        const __m512 vqmax = _mm512_set1_ps((float)qmax);

        for (int ob = 0; ob < OC_blocks; ob++) {
            int oc0 = ob * 16;
            int oc_cnt = std::min(16, OC - oc0);

            // Per-OC combined scale, bias, zp_comp (16 lanes, pad unused with 0/1).
            alignas(64) float cs_arr[16] = {};
            alignas(64) float bs_arr[16] = {};
            alignas(64) int32_t zc_arr[16] = {};
            for (int l = 0; l < oc_cnt; l++) {
                float ws = (w_scale_count > 1) ? w_scale[oc0 + l] : w_scale[0];
                float cs = x_scale * ws;
                cs_arr[l] = cs;
                bs_arr[l] = bias ? (float)bias[oc0 + l] * cs : 0.0f;
                zc_arr[l] = x_zp * w_row_sum[oc0 + l];
            }
            __m512  vcs  = _mm512_load_ps(cs_arr);
            __m512  vbs  = _mm512_load_ps(bs_arr);
            __m512i vzc  = _mm512_load_si512((const __m512i*)zc_arr);

            const int32_t* src_ob = buf + (size_t)ob * oW * 16;

            if (output_nhwc) {
                // NHWC: per-(ob,ow) requantize 16 OC lanes and store 16 uint8
                // contiguously at y[(oh*oW + ow) * OC + oc0]. No transpose.
                // Mask the high lanes when oc_cnt < 16 (last block, OC % 16 != 0).
                __mmask16 kstore = (oc_cnt == 16)
                    ? (__mmask16)0xFFFF
                    : (__mmask16)((1u << oc_cnt) - 1u);
                for (int ow2 = 0; ow2 < oW; ow2++) {
                    __m512i raw = _mm512_loadu_si512(
                        (const __m512i*)(src_ob + (size_t)ow2 * 16));
                    __m512i comp = _mm512_sub_epi32(raw, vzc);
                    __m512  fv   = _mm512_cvtepi32_ps(comp);
                    fv = _mm512_fmadd_ps(fv, vcs, vbs);
                    fv = _mm512_fmadd_ps(fv, vis, vyzp);
                    fv = _mm512_max_ps(_mm512_min_ps(fv, vqmax), vqmin);
                    __m512i qi = _mm512_cvtps_epi32(fv);
                    __m128i b  = _mm512_cvtepi32_epi8(qi);
                    _mm_mask_storeu_epi8(
                        y + ((size_t)oh * oW + ow2) * OC + oc0,
                        kstore, b);
                }
                continue;
            }

            // NCHW: transpose 16x16 int32 tile (ow-major → oc-major),
            // per-OC vector requantize, then 16-byte store to y[oc][oh, ow..ow+15].
            int ow2 = 0;
            for (; ow2 + 16 <= oW; ow2 += 16) {
                alignas(64) int32_t tbuf[16 * 16];
                transpose_16x16_avx512(
                    reinterpret_cast<const float*>(src_ob + (size_t)ow2 * 16),
                    /*src_stride=*/16,
                    reinterpret_cast<float*>(tbuf),
                    /*dst_stride=*/16);

                for (int l = 0; l < oc_cnt; l++) {
                    __m512i raw = _mm512_load_si512((const __m512i*)(tbuf + l * 16));
                    __m512i comp = _mm512_sub_epi32(raw, _mm512_set1_epi32(zc_arr[l]));
                    __m512  fv = _mm512_cvtepi32_ps(comp);
                    fv = _mm512_fmadd_ps(fv, _mm512_set1_ps(cs_arr[l]),
                                         _mm512_set1_ps(bs_arr[l]));
                    fv = _mm512_fmadd_ps(fv, vis, vyzp);
                    fv = _mm512_max_ps(_mm512_min_ps(fv, vqmax), vqmin);
                    __m512i qi = _mm512_cvtps_epi32(fv);
                    __m128i b  = _mm512_cvtepi32_epi8(qi);
                    _mm_storeu_si128(
                        (__m128i*)(y + (size_t)(oc0 + l) * spatial + oh * oW + ow2),
                        b);
                }
            }
            (void)vcs; (void)vbs; (void)vzc;
            for (; ow2 < oW; ow2++) {
                const int32_t* src_px = src_ob + (size_t)ow2 * 16;
                for (int l = 0; l < oc_cnt; l++) {
                    int32_t raw = src_px[l] - zc_arr[l];
                    float f = (float)raw * cs_arr[l] + bs_arr[l];
                    int32_t q = (int32_t)std::nearbyint(f * inv_y_scale) + y_zp;
                    q = std::max(qmin, std::min(qmax, q));
                    y[(size_t)(oc0 + l) * spatial + oh * oW + ow2] = (uint8_t)q;
                }
            }
        }
    });

    return true;
}

} // namespace nnr

#endif // NNR_ARCH_X64
