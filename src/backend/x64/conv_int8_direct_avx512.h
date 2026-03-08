#pragma once
// Direct int8 convolution with AVX-512 VNNI (VPDPBUSD) — no im2col buffer.
//
// Vectorizes over output channels (16 per ZMM register), tiles 14 output width positions.
// Uses VPDPBUSD: uint8 input (src1, unsigned) × int8 weight (src2, signed) → int32 accumulator.
//
// Weight layout: [KH*KW][IC/4][OC/16] panels of 64 bytes (16 OC × 4 IC in VNNI order).
// Input: uint8 NCHW, pre-padded with x_zp to eliminate boundary checks.
// Output: int32 [OC, oH*oW] raw dot products. Caller applies zero-point compensation.
//
// Raw output: y[oc, s] = sum_{ic,kh,kw} x_uint8[ic, ih, iw] * w_int8[oc, ic, kh, kw]
// Caller computes: (y[oc,s] - x_zp * w_sum[oc]) * combined_scale + bias → requantize.

#ifdef NNR_ARCH_X64

#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include "cpu_features.h"
#include "thread_pool.h"

namespace nnr {

// Packed weight size in bytes.
// @nnr-meta isa=scalar dtype=[int8,uint8]
inline size_t pack_weights_int8_direct_size(int OC, int IC, int KH, int KW)
{
    int OC16 = (OC + 15) / 16;
    int IC4 = (IC + 3) / 4;
    return (size_t)KH * KW * IC4 * OC16 * 64;
}

// Repack weights from OIHW int8 to VNNI direct format.
// For each kernel position (kh,kw), IC group of 4, OC block of 16:
//   64 bytes where byte[lane*4 + ic_off] = w[oc_block*16+lane, ic_group*4+ic_off, kh, kw].
// @nnr-meta isa=scalar dtype=[int8,uint8]
inline void pack_weights_int8_direct(
    int8_t* __restrict dst,
    const int8_t* __restrict src,  // [OC, IC, KH, KW]
    int OC, int IC, int KH, int KW)
{
    int OC16 = (OC + 15) / 16;
    int IC4 = (IC + 3) / 4;
    int kSpatial = KH * KW;
    memset(dst, 0, pack_weights_int8_direct_size(OC, IC, KH, KW));

    for (int oc = 0; oc < OC; oc++) {
        int ob = oc / 16, lane = oc % 16;
        for (int ic = 0; ic < IC; ic++) {
            int ig = ic / 4, ic_off = ic % 4;
            for (int ks = 0; ks < kSpatial; ks++) {
                size_t didx = ((size_t)ks * IC4 + ig) * OC16 * 64 + ob * 64 + lane * 4 + ic_off;
                dst[didx] = src[((size_t)oc * IC + ic) * kSpatial + ks];
            }
        }
    }
}

// Pre-compute per-OC weight sums: w_sum[oc] = sum over (ic,kh,kw) of w[oc,ic,kh,kw].
// @nnr-meta isa=scalar dtype=int8
inline void compute_weight_sums_int8_direct(
    int32_t* __restrict w_sum,
    const int8_t* __restrict src,  // [OC, IC, KH, KW]
    int OC, int IC, int KH, int KW)
{
    int CHW = IC * KH * KW;
    for (int oc = 0; oc < OC; oc++) {
        int32_t sum = 0;
        const int8_t* row = src + (size_t)oc * CHW;
        for (int k = 0; k < CHW; k++)
            sum += (int32_t)row[k];
        w_sum[oc] = sum;
    }
}

// Direct int8 conv kernel.
// x_padded: uint8 [IC_padded, pH, pW] pre-padded with x_zp (IC_padded = round_up(IC, 4)).
// w_packed: VNNI-packed weights from pack_weights_int8_direct().
// y_i32: int32 [OC, oH*oW] output (raw dot products, no zero-point compensation).
// @nnr-meta isa=[AVX512,AVX512_VNNI] dtype=[int8,uint8] layout=NCHW special=Direct
inline bool conv_int8_direct_avx512(
    int32_t* __restrict y_i32,
    const uint8_t* __restrict x_padded,
    const int8_t* __restrict w_packed,
    int IC, int pH, int pW,       // IC (original), padded height/width
    int OC, int oH, int oW,
    int KH, int KW,
    int sH, int sW)
{
    if (!has_avx512() || !cpu_features().avx512vnni) return false;

    const int OC16 = (OC + 15) / 16;
    const int IC4 = (IC + 3) / 4;
    const int spatial = oH * oW;
    const size_t plane_stride = (size_t)pH * pW;
    constexpr int WT = 14;  // output width tile (14 ZMM accumulators)

    nnr::for_static(0, oH, oH >= 4, [&](int oh) {
        const int out_stride = OC16 * 16;
        std::vector<int32_t> buf((size_t)oW * out_stride);

        for (int ob = 0; ob < OC16; ob++) {
            int32_t* out_ob = buf.data() + ob * 16;

            // --- Tiled main loop: 14 output positions at a time ---
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
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int ks = kh * KW + kw;
                        for (int ig = 0; ig < IC4; ig++) {
                            const __m512i wv = _mm512_loadu_si512(
                                w_packed + ((size_t)ks * IC4 + ig) * OC16 * 64 + ob * 64);

                            // Channel plane pointers for this IC group, at row ih
                            const uint8_t* p0 = x_padded + (size_t)(ig * 4 + 0) * plane_stride + ih * pW;
                            const uint8_t* p1 = p0 + plane_stride;
                            const uint8_t* p2 = p0 + 2 * plane_stride;
                            const uint8_t* p3 = p0 + 3 * plane_stride;

                            #define NNR_DPBUSD_T(T) { \
                                const int iw = (ow + (T)) * sW + kw; \
                                const uint32_t xv = (uint32_t)p0[iw] \
                                    | ((uint32_t)p1[iw] << 8) \
                                    | ((uint32_t)p2[iw] << 16) \
                                    | ((uint32_t)p3[iw] << 24); \
                                a##T = _mm512_dpbusd_epi32(a##T, \
                                    _mm512_set1_epi32((int32_t)xv), wv); \
                            }
                            NNR_DPBUSD_T(0)  NNR_DPBUSD_T(1)
                            NNR_DPBUSD_T(2)  NNR_DPBUSD_T(3)
                            NNR_DPBUSD_T(4)  NNR_DPBUSD_T(5)
                            NNR_DPBUSD_T(6)  NNR_DPBUSD_T(7)
                            NNR_DPBUSD_T(8)  NNR_DPBUSD_T(9)
                            NNR_DPBUSD_T(10) NNR_DPBUSD_T(11)
                            NNR_DPBUSD_T(12) NNR_DPBUSD_T(13)
                            #undef NNR_DPBUSD_T
                        }
                    }
                }

                _mm512_storeu_si512(out_ob + (size_t)(ow +  0) * out_stride, a0);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  1) * out_stride, a1);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  2) * out_stride, a2);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  3) * out_stride, a3);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  4) * out_stride, a4);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  5) * out_stride, a5);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  6) * out_stride, a6);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  7) * out_stride, a7);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  8) * out_stride, a8);
                _mm512_storeu_si512(out_ob + (size_t)(ow +  9) * out_stride, a9);
                _mm512_storeu_si512(out_ob + (size_t)(ow + 10) * out_stride, a10);
                _mm512_storeu_si512(out_ob + (size_t)(ow + 11) * out_stride, a11);
                _mm512_storeu_si512(out_ob + (size_t)(ow + 12) * out_stride, a12);
                _mm512_storeu_si512(out_ob + (size_t)(ow + 13) * out_stride, a13);
            }

            // --- Remainder pixels (< 14) ---
            for (; ow < oW; ow++) {
                __m512i a = _mm512_setzero_si512();
                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * sH + kh;
                    for (int kw = 0; kw < KW; kw++) {
                        const int ks = kh * KW + kw;
                        const int iw = ow * sW + kw;
                        for (int ig = 0; ig < IC4; ig++) {
                            const __m512i wv = _mm512_loadu_si512(
                                w_packed + ((size_t)ks * IC4 + ig) * OC16 * 64 + ob * 64);
                            const uint8_t* p0 = x_padded + (size_t)(ig * 4) * plane_stride + ih * pW + iw;
                            const uint32_t xv = (uint32_t)p0[0]
                                | ((uint32_t)p0[plane_stride] << 8)
                                | ((uint32_t)p0[2 * plane_stride] << 16)
                                | ((uint32_t)p0[3 * plane_stride] << 24);
                            a = _mm512_dpbusd_epi32(a,
                                _mm512_set1_epi32((int32_t)xv), wv);
                        }
                    }
                }
                _mm512_storeu_si512(out_ob + (size_t)ow * out_stride, a);
            }
        }

        // Transpose: interleaved [oW][OC16*16] → NCHW [OC][spatial]
        for (int oc = 0; oc < OC; oc++) {
            int32_t* yrow = y_i32 + (size_t)oc * spatial + oh * oW;
            const int ob = oc / 16, lane = oc % 16;
            for (int ow2 = 0; ow2 < oW; ow2++)
                yrow[ow2] = buf[(size_t)ow2 * out_stride + ob * 16 + lane];
        }
    });

    return true;
}

} // namespace nnr

#endif
