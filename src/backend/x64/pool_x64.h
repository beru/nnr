#pragma once
// AVX2/AVX-512 SIMD pooling kernels.

#include <immintrin.h>
#include <algorithm>
#include <cfloat>
#include "cpu_features.h"
#include "thread_pool.h"

namespace nnr {

// Forward decl for the 3x3/s2/p0 AVX-512 fast-path used from the generic NCHW dispatcher.
inline void maxpool_2d_nchw_3x3s2p0_x64(const float* input, float* output,
    int NC, int iH, int iW, int oH, int oW);

// Forward decl: 3x3/s2/p1 NCHW AVX-512 (ssd-12 pool1: [64,600,600]→[300,300]).
inline void maxpool_2d_nchw_3x3s2p1_x64(const float* input, float* output,
    int NC, int iH, int iW, int oH, int oW);

// Forward decl: 3x3/s1/p1 NCHW AVX-512 (googlenet inception pools).
inline void maxpool_2d_nchw_3x3s1p1_x64(const float* input, float* output,
    int NC, int iH, int iW);

// SIMD NCHW maxpool for float, no indices.
// Handles common cases: stride-1 (any kW), 2x2/s2, generic stride-2.
// @nnr-meta isa=[AVX512,AVX2] dtype=fp32 layout=NCHW
inline void maxpool_2d_float_simd(const float* input, float* output,
    int NC, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    // Fast path: 3x3 / s=2 / p=0 (adv_inception_v3 NCHW reduction MaxPools).
    // 8 outputs/iter, 3 parallel row-max chains, vcompressps to pack even lanes.
    // ceil_mode can produce an output whose rightmost/bottom window extends
    // past the input; guard against that so we only take the fast path when
    // every output cell's full 3x3 window fits within iH/iW.
    if (kH == 3 && kW == 3 && sH == 2 && sW == 2 && pH == 0 && pW == 0
        && has_avx512() && iH >= 3 && iW >= 3
        && (oH - 1) * 2 + 3 <= iH && (oW - 1) * 2 + 3 <= iW) {
        maxpool_2d_nchw_3x3s2p0_x64(input, output, NC, iH, iW, oH, oW);
        return;
    }
    // Fast path: 3x3 / s=2 / p=1 (ssd-12 pool1: [64,600,600]→[300,300]).
    if (kH == 3 && kW == 3 && sH == 2 && sW == 2 && pH == 1 && pW == 1
        && has_avx512() && iH >= 3 && iW >= 3
        && (oH - 1) * 2 + 2 <= iH && (oW - 1) * 2 + 2 <= iW) {
        maxpool_2d_nchw_3x3s2p1_x64(input, output, NC, iH, iW, oH, oW);
        return;
    }
    // Fast path: 3x3 / s=1 / p=1 (googlenet inception pools).
    if (kH == 3 && kW == 3 && sH == 1 && sW == 1 && pH == 1 && pW == 1
        && has_avx512() && iH >= 3 && iW >= 3 && oH == iH && oW == iW) {
        maxpool_2d_nchw_3x3s1p1_x64(input, output, NC, iH, iW);
        return;
    }

    // Safe oW range where full kernel fits horizontally (no padding clipping)
    int ow_safe_lo = pW > 0 ? (pW + sW - 1) / sW : 0;
    int ow_safe_hi = std::min(oW, (iW + pW - kW) / sW + 1);

    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const float* inp = input + (size_t)nc * iH * iW;
        float* out = output + (size_t)nc * oH * oW;

        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            float* dst = out + oh * oW;

            // Left edge (scalar)
            for (int ow = 0; ow < ow_safe_lo && ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float maxv = -FLT_MAX;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw)
                        maxv = std::max(maxv, inp[(ih0 + kh) * iW + (iw0 + kw)]);
                dst[ow] = maxv;
            }

            // Interior (SIMD)
            int ow = ow_safe_lo;

            if (sW == 1 && detect_isa() >= isa_t::avx2) {
                // Stride-1: shift+max across 8 oW positions
                for (; ow + 8 <= ow_safe_hi; ow += 8) {
                    int iw0 = ow - pW;
                    __m256 vmax = _mm256_set1_ps(-FLT_MAX);
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const float* row = inp + (ih0 + kh) * iW + iw0;
                        __m256 row_max = _mm256_loadu_ps(row);
                        for (int kw = 1; kw < kW; ++kw)
                            row_max = _mm256_max_ps(row_max, _mm256_loadu_ps(row + kw));
                        vmax = _mm256_max_ps(vmax, row_max);
                    }
                    _mm256_storeu_ps(dst + ow, vmax);
                }
            } else if (sW == 2 && kW == 2 && detect_isa() >= isa_t::avx2) {
                // Specialized 2x2/s2: load 8 elements = 4 pairs, max pairs
                __m256i perm = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
                for (; ow + 4 <= ow_safe_hi; ow += 4) {
                    int iw0 = ow * 2 - pW;
                    __m256 vmax = _mm256_set1_ps(-FLT_MAX);
                    for (int kh = kh0; kh < kh1; ++kh) {
                        __m256 v = _mm256_loadu_ps(inp + (ih0 + kh) * iW + iw0);
                        __m256 shifted = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
                        vmax = _mm256_max_ps(vmax, _mm256_max_ps(v, shifted));
                    }
                    __m256 result = _mm256_permutevar8x32_ps(vmax, perm);
                    _mm_storeu_ps(dst + ow, _mm256_castps256_ps128(result));
                }
            } else if (sW == 2 && detect_isa() >= isa_t::avx2) {
                // Generic stride-2: shift+max, then pick elements 0,2,4,6
                int ow_s2_end = std::min(ow_safe_hi, (iW + pW - kW - 6) / 2 + 1);
                __m256i perm = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
                for (; ow + 4 <= ow_s2_end; ow += 4) {
                    int iw0 = ow * 2 - pW;
                    __m256 vmax = _mm256_set1_ps(-FLT_MAX);
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const float* row = inp + (ih0 + kh) * iW + iw0;
                        __m256 row_max = _mm256_loadu_ps(row);
                        for (int kw = 1; kw < kW; ++kw)
                            row_max = _mm256_max_ps(row_max, _mm256_loadu_ps(row + kw));
                        vmax = _mm256_max_ps(vmax, row_max);
                    }
                    __m256 result = _mm256_permutevar8x32_ps(vmax, perm);
                    _mm_storeu_ps(dst + ow, _mm256_castps256_ps128(result));
                }
            }

            // Remainder + right edge (scalar)
            for (; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float maxv = -FLT_MAX;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw)
                        maxv = std::max(maxv, inp[(ih0 + kh) * iW + (iw0 + kw)]);
                dst[ow] = maxv;
            }
        }
    });
}

// NCHW avgpool SIMD for float.
// Splits oW into safe middle (no column clipping) + scalar edges.
// Middle region vectorizes 16 outputs at a time when sW==1. For sW>1, the
// middle SIMD is skipped and the scalar edge loop handles all columns.
// @nnr-meta isa=[AVX512,AVX2] dtype=fp32 layout=NCHW
inline void avgpool_2d_float_simd(const float* input, float* output,
    int NC, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    bool count_include_pad)
{
    int ow_safe_lo = pW > 0 ? (pW + sW - 1) / sW : 0;
    int ow_safe_hi = std::min(oW, (iW + pW - kW) / sW + 1);

    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const float* inp = input + (size_t)nc * iH * iW;
        float* out = output + (size_t)nc * oH * oW;

        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            int valid_h = kh1 - kh0;
            float* dst = out + oh * oW;

            // Left edge (scalar: column window is clipped)
            for (int ow = 0; ow < ow_safe_lo && ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float sum = 0;
                int cnt = 0;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        sum += inp[(ih0 + kh) * iW + (iw0 + kw)];
                        ++cnt;
                    }
                int div = count_include_pad ? (kH * kW) : (cnt > 0 ? cnt : 1);
                dst[ow] = sum / (float)div;
            }

            int ow = ow_safe_lo;

            // Interior SIMD — only for sW==1 where loads are contiguous.
            if (sW == 1 && has_avx512()) {
                // In the safe middle, valid_w == kW for every lane, so divisor
                // is constant per row: kH*kW (cip) or valid_h*kW (non-cip).
                float div = count_include_pad ? (float)(kH * kW) : (float)(valid_h * kW);
                __m512 vinv = _mm512_set1_ps(div > 0 ? 1.0f / div : 0.0f);
                // Full 16-wide vectors
                for (; ow + 16 <= ow_safe_hi; ow += 16) {
                    int iw0 = ow - pW;
                    __m512 vsum = _mm512_setzero_ps();
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const float* row = inp + (ih0 + kh) * iW + iw0;
                        __m512 row_sum = _mm512_loadu_ps(row);
                        for (int kw = 1; kw < kW; ++kw)
                            row_sum = _mm512_add_ps(row_sum, _mm512_loadu_ps(row + kw));
                        vsum = _mm512_add_ps(vsum, row_sum);
                    }
                    _mm512_storeu_ps(dst + ow, _mm512_mul_ps(vsum, vinv));
                }
                // Partial tail vector (safe region remainder < 16)
                int rem = ow_safe_hi - ow;
                if (rem > 0) {
                    __mmask16 mask = (__mmask16)((1u << rem) - 1);
                    int iw0 = ow - pW;
                    __m512 vsum = _mm512_setzero_ps();
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const float* row = inp + (ih0 + kh) * iW + iw0;
                        __m512 row_sum = _mm512_maskz_loadu_ps(mask, row);
                        for (int kw = 1; kw < kW; ++kw)
                            row_sum = _mm512_mask_add_ps(row_sum, mask, row_sum,
                                _mm512_maskz_loadu_ps(mask, row + kw));
                        vsum = _mm512_mask_add_ps(vsum, mask, vsum, row_sum);
                    }
                    _mm512_mask_storeu_ps(dst + ow, mask, _mm512_mul_ps(vsum, vinv));
                    ow = ow_safe_hi;
                }
            }

            // Remainder + right edge (scalar)
            for (; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                float sum = 0;
                int cnt = 0;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        sum += inp[(ih0 + kh) * iW + (iw0 + kw)];
                        ++cnt;
                    }
                int div = count_include_pad ? (kH * kW) : (cnt > 0 ? cnt : 1);
                dst[ow] = sum / (float)div;
            }
        }
    });
}

// NHWC maxpool with AVX2/AVX-512 SIMD over channels.
// @nnr-meta isa=[AVX512,AVX2] dtype=fp32 layout=NHWC
inline void maxpool_2d_nhwc_x64(const float* input, float* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    nnr::for_static(0, N * oH, N * oH > 4, [&](int noh) {
        int n = noh / oH, oh = noh % oH;
        const float* xn = input + (size_t)n * iH * iW * C;
        float* yn = output + (size_t)n * oH * oW * C;
        int ih0 = oh * sH - pH;
        int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
        for (int ow = 0; ow < oW; ow++) {
            int iw0 = ow * sW - pW;
            int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
            float* out = yn + (oh * oW + ow) * C;
            int c = 0;
            if (has_avx512()) {
                for (; c + 16 <= C; c += 16) {
                    __m512 vmax = _mm512_set1_ps(std::numeric_limits<float>::lowest());
                    for (int kh = kh0; kh < kh1; kh++)
                        for (int kw = kw0; kw < kw1; kw++)
                            vmax = _mm512_max_ps(vmax,
                                _mm512_loadu_ps(xn + ((ih0 + kh) * iW + (iw0 + kw)) * C + c));
                    _mm512_storeu_ps(out + c, vmax);
                }
            } else if (detect_isa() >= isa_t::avx2) {
                for (; c + 8 <= C; c += 8) {
                    __m256 vmax = _mm256_set1_ps(-FLT_MAX);
                    for (int kh = kh0; kh < kh1; kh++)
                        for (int kw = kw0; kw < kw1; kw++)
                            vmax = _mm256_max_ps(vmax,
                                _mm256_loadu_ps(xn + ((ih0 + kh) * iW + (iw0 + kw)) * C + c));
                    _mm256_storeu_ps(out + c, vmax);
                }
            }
            for (; c < C; c++) {
                float maxv = std::numeric_limits<float>::lowest();
                for (int kh = kh0; kh < kh1; kh++)
                    for (int kw = kw0; kw < kw1; kw++) {
                        float v = xn[((ih0 + kh) * iW + (iw0 + kw)) * C + c];
                        if (v > maxv) maxv = v;
                    }
                out[c] = maxv;
            }
        }
    });
}

// NCHW maxpool for uint8, no indices.
// Same structure as float_simd but operates on bytes — 64 outputs per _mm512_max_epu8.
// @nnr-meta isa=AVX512 dtype=uint8 layout=NCHW
inline void maxpool_2d_uint8_simd(const uint8_t* input, uint8_t* output,
    int NC, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    int ow_safe_lo = pW > 0 ? (pW + sW - 1) / sW : 0;
    int ow_safe_hi = std::min(oW, (iW + pW - kW) / sW + 1);

    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const uint8_t* inp = input + (size_t)nc * iH * iW;
        uint8_t* out = output + (size_t)nc * oH * oW;

        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            uint8_t* dst = out + oh * oW;

            // Left edge (scalar)
            for (int ow = 0; ow < ow_safe_lo && ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                uint8_t maxv = 0;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        uint8_t v = inp[(ih0 + kh) * iW + (iw0 + kw)];
                        if (v > maxv) maxv = v;
                    }
                dst[ow] = maxv;
            }

            // Interior (SIMD)
            int ow = ow_safe_lo;

            if (sW == 1) {
                // Stride-1: vectorize over oW, 64 outputs at a time
                for (; ow + 64 <= ow_safe_hi; ow += 64) {
                    int iw0 = ow - pW;
                    __m512i vmax = _mm512_setzero_si512();
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const uint8_t* row = inp + (ih0 + kh) * iW + iw0;
                        __m512i row_max = _mm512_loadu_si512(row);
                        for (int kw = 1; kw < kW; ++kw)
                            row_max = _mm512_max_epu8(row_max, _mm512_loadu_si512(row + kw));
                        vmax = _mm512_max_epu8(vmax, row_max);
                    }
                    _mm512_storeu_si512(dst + ow, vmax);
                }
            } else if (sW == 2 && kW == 2) {
                // 2x2/s2: load 128 bytes (64 pairs), max adjacent bytes, compress
                for (; ow + 32 <= ow_safe_hi; ow += 32) {
                    int iw0 = ow * 2 - pW;
                    __m512i vmax = _mm512_setzero_si512();
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const uint8_t* row = inp + (ih0 + kh) * iW + iw0;
                        // Load 64 bytes = 32 pairs of adjacent elements
                        __m512i a = _mm512_loadu_si512(row);
                        // Shift right by 1 byte within each 16-bit lane
                        __m512i b = _mm512_srli_epi16(a, 8);
                        // a has [e0,e1,e2,e3,...], b has [e1,0,e3,0,...]
                        // max(a,b) gives [max(e0,e1),?,max(e2,e3),?,...]
                        __m512i pairmax = _mm512_max_epu8(a, b);
                        vmax = _mm512_max_epu8(vmax, pairmax);
                    }
                    // Extract even-indexed bytes (the pair maxima) → 32 outputs
                    // Use vpmovwb: truncate 16-bit to 8-bit (keeps low byte of each word)
                    __m256i packed = _mm512_cvtepi16_epi8(
                        _mm512_and_si512(vmax, _mm512_set1_epi16(0x00FF)));
                    _mm256_storeu_si256((__m256i*)(dst + ow), packed);
                }
            } else if (sW == 2) {
                // Generic stride-2 with any kW
                for (; ow + 32 <= ow_safe_hi; ow += 32) {
                    int iw0 = ow * 2 - pW;
                    __m512i vmax = _mm512_setzero_si512();
                    for (int kh = kh0; kh < kh1; ++kh) {
                        const uint8_t* row = inp + (ih0 + kh) * iW + iw0;
                        __m512i row_max = _mm512_loadu_si512(row);
                        for (int kw = 1; kw < kW; ++kw)
                            row_max = _mm512_max_epu8(row_max, _mm512_loadu_si512(row + kw));
                        vmax = _mm512_max_epu8(vmax, row_max);
                    }
                    __m256i packed = _mm512_cvtepi16_epi8(
                        _mm512_and_si512(vmax, _mm512_set1_epi16(0x00FF)));
                    _mm256_storeu_si256((__m256i*)(dst + ow), packed);
                }
            }

            // Remainder + right edge (scalar)
            for (; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                uint8_t maxv = 0;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        uint8_t v = inp[(ih0 + kh) * iW + (iw0 + kw)];
                        if (v > maxv) maxv = v;
                    }
                dst[ow] = maxv;
            }
        }
    });
}

// NHWC maxpool for uint8 with AVX-512 SIMD over channels.
// @nnr-meta isa=[AVX512,AVX2] dtype=uint8 layout=NHWC
inline void maxpool_2d_uint8_nhwc_x64(const uint8_t* input, uint8_t* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    nnr::for_static(0, N * oH, N * oH > 4, [&](int noh) {
        int n = noh / oH, oh = noh % oH;
        const uint8_t* xn = input + (size_t)n * iH * iW * C;
        uint8_t* yn = output + (size_t)n * oH * oW * C;
        int ih0 = oh * sH - pH;
        int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
        for (int ow = 0; ow < oW; ow++) {
            int iw0 = ow * sW - pW;
            int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
            uint8_t* out = yn + (oh * oW + ow) * C;
            int c = 0;
            for (; c + 64 <= C; c += 64) {
                __m512i vmax = _mm512_setzero_si512();
                for (int kh = kh0; kh < kh1; kh++)
                    for (int kw = kw0; kw < kw1; kw++)
                        vmax = _mm512_max_epu8(vmax,
                            _mm512_loadu_si512(xn + ((ih0 + kh) * iW + (iw0 + kw)) * C + c));
                _mm512_storeu_si512(out + c, vmax);
            }
            for (; c + 32 <= C; c += 32) {
                __m256i vmax = _mm256_setzero_si256();
                for (int kh = kh0; kh < kh1; kh++)
                    for (int kw = kw0; kw < kw1; kw++)
                        vmax = _mm256_max_epu8(vmax,
                            _mm256_loadu_si256((const __m256i*)(xn + ((ih0 + kh) * iW + (iw0 + kw)) * C + c)));
                _mm256_storeu_si256((__m256i*)(out + c), vmax);
            }
            for (; c < C; c++) {
                uint8_t maxv = 0;
                for (int kh = kh0; kh < kh1; kh++)
                    for (int kw = kw0; kw < kw1; kw++) {
                        uint8_t v = xn[((ih0 + kh) * iW + (iw0 + kw)) * C + c];
                        if (v > maxv) maxv = v;
                    }
                out[c] = maxv;
            }
        }
    });
}

// NHWC avgpool with AVX-512 SIMD over channels.
// @nnr-meta isa=AVX512 dtype=fp32 layout=NHWC
inline void avgpool_2d_nhwc_x64(const float* input, float* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    bool count_include_pad = false)
{
    nnr::for_static(0, N * oH, N * oH > 4, [&](int noh) {
        int n = noh / oH, oh = noh % oH;
        const float* xn = input + (size_t)n * iH * iW * C;
        float* yn = output + (size_t)n * oH * oW * C;
        int ih0 = oh * sH - pH;
        for (int ow = 0; ow < oW; ow++) {
            int iw0 = ow * sW - pW;
            float* out = yn + (oh * oW + ow) * C;
            int valid = 0;
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih0 + kh;
                if (ih < 0 || ih >= iH) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw0 + kw;
                    if (iw < 0 || iw >= iW) continue;
                    valid++;
                }
            }
            float divisor = count_include_pad ? (float)(kH * kW) : (valid > 0 ? (float)valid : 1.0f);
            float inv = 1.0f / divisor;
            int c = 0;
            if (has_avx512()) {
                __m512 vinv = _mm512_set1_ps(inv);
                for (; c + 16 <= C; c += 16) {
                    __m512 vsum = _mm512_setzero_ps();
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= iH) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = iw0 + kw;
                            if (iw < 0 || iw >= iW) continue;
                            vsum = _mm512_add_ps(vsum,
                                _mm512_loadu_ps(xn + (ih * iW + iw) * C + c));
                        }
                    }
                    _mm512_storeu_ps(out + c, _mm512_mul_ps(vsum, vinv));
                }
            }
            for (; c < C; c++) {
                float sum = 0;
                for (int kh = 0; kh < kH; kh++) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= iH) continue;
                    for (int kw = 0; kw < kW; kw++) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= iW) continue;
                        sum += xn[(ih * iW + iw) * C + c];
                    }
                }
                out[c] = sum * inv;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHWc (BLOCKED_16) MaxPool — AVX-512.
// Each [H,W,16] block is processed with ZMM-wide max across spatial window.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=NCHWc
inline void maxpool_2d_nchwc_x64(const float* input, float* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    constexpr int block = 16;
    const int Cb = C / block;
    const size_t in_spatial  = (size_t)iH * iW * block;
    const size_t out_spatial = (size_t)oH * oW * block;

    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const float* inp = input + (size_t)ncb * in_spatial;
        float* out = output + (size_t)ncb * out_spatial;
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                __m512 vmax = _mm512_set1_ps(-FLT_MAX);
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw)
                        vmax = _mm512_max_ps(vmax,
                            _mm512_loadu_ps(&inp[((ih0 + kh) * iW + (iw0 + kw)) * block]));
                _mm512_storeu_ps(&out[(oh * oW + ow) * block], vmax);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHWc (BLOCKED_16) AveragePool — AVX-512.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=NCHWc
inline void avgpool_2d_nchwc_x64(const float* input, float* output,
    int N, int C, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    bool count_include_pad)
{
    constexpr int block = 16;
    const int Cb = C / block;
    const size_t in_spatial  = (size_t)iH * iW * block;
    const size_t out_spatial = (size_t)oH * oW * block;

    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const float* inp = input + (size_t)ncb * in_spatial;
        float* out = output + (size_t)ncb * out_spatial;
        for (int oh = 0; oh < oH; ++oh) {
            int ih0 = oh * sH - pH;
            int kh0 = std::max(0, -ih0), kh1 = std::min(kH, iH - ih0);
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                int kw0 = std::max(0, -iw0), kw1 = std::min(kW, iW - iw0);
                __m512 vsum = _mm512_setzero_ps();
                int valid = 0;
                for (int kh = kh0; kh < kh1; ++kh)
                    for (int kw = kw0; kw < kw1; ++kw) {
                        vsum = _mm512_add_ps(vsum,
                            _mm512_loadu_ps(&inp[((ih0 + kh) * iW + (iw0 + kw)) * block]));
                        ++valid;
                    }
                float div = count_include_pad ? (float)(kH * kW) : (valid > 0 ? (float)valid : 1.0f);
                __m512 vinv = _mm512_set1_ps(1.0f / div);
                _mm512_storeu_ps(&out[(oh * oW + ow) * block], _mm512_mul_ps(vsum, vinv));
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHW AveragePool 3x3 / stride=1 / pad=1 / count_include_pad=1.
// Specialization for adv_inception_v3's 9 inception-branch AvgPool nodes
// whose inputs stay in NCHW format (the upstream Concat is not BLOCKED_16).
//
// Fixed divisor = 9. Within each output row, 16 contiguous ow values share
// input rows, so one SIMD vector load at offset iw covers 16 outputs with
// the "center" column in place; the "left" and "right" columns are the
// same row loaded at iw-1 and iw+1 (shifted by one float).
//
// Uses three parallel row accumulators so the add dependency chain per
// output-vector is ~3 add latencies deep (vs the generic kernel's 6-7).
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW
inline void avgpool_2d_nchw_3x3s1p1_cip_x64(const float* input, float* output,
    int NC, int iH, int iW)
{
    const int oH = iH, oW = iW;  // SAME padding, stride 1
    const __m512 vinv = _mm512_set1_ps(1.0f / 9.0f);

    // Vectorization only pays off once the row is wide enough for at least
    // one full 16-wide interior column block. 35/17/8 all qualify only at
    // iW >= 16 for the 16-wide path; at iW < 16 we fall through to masked
    // loads for the whole row.
    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const float* inp = input + (size_t)nc * iH * iW;
        float* out       = output + (size_t)nc * oH * oW;

        // Helper: 3-tap horizontal sum of a row at positions [iw0, iw0+16),
        // with the left and right taps shifted by ±1 (unaligned loads). The
        // 16 output positions are [iw0, iw0+16); taps are iw0-1, iw0, iw0+1.
        // Caller must ensure iw0-1 >= 0 and iw0+16 <= iW.
        auto row3 = [&](const float* row, int iw0) -> __m512 {
            __m512 a = _mm512_loadu_ps(row + iw0 - 1);
            __m512 b = _mm512_loadu_ps(row + iw0);
            __m512 c = _mm512_loadu_ps(row + iw0 + 1);
            return _mm512_add_ps(a, _mm512_add_ps(b, c));
        };
        // Same but with a mask (for the tail where iw0+16 > iW).
        auto row3_m = [&](const float* row, int iw0, __mmask16 m) -> __m512 {
            __m512 a = _mm512_maskz_loadu_ps(m, row + iw0 - 1);
            __m512 b = _mm512_maskz_loadu_ps(m, row + iw0);
            __m512 c = _mm512_maskz_loadu_ps(m, row + iw0 + 1);
            return _mm512_add_ps(a, _mm512_add_ps(b, c));
        };

        // --- Interior rows: oh ∈ [1, oH-1) ---
        // The leftmost output (ow=0) has iw-1 = -1 (padding, zero); the
        // rightmost (ow=oW-1) has iw+1 = oW (padding, zero). We handle those
        // two outputs scalar-ish; everything else is clean 16-wide SIMD.
        for (int oh = 1; oh + 1 < oH; ++oh) {
            const int ih1 = oh;
            const float* rA = inp + (size_t)(ih1 - 1) * iW;
            const float* rB = inp + (size_t)(ih1    ) * iW;
            const float* rC = inp + (size_t)(ih1 + 1) * iW;
            float* dst = out + (size_t)oh * oW;

            // ow = 0: iw taps {-1, 0, 1} → only {0, 1} are real.
            {
                float s = rA[0] + rA[1] + rB[0] + rB[1] + rC[0] + rC[1];
                dst[0] = s * (1.0f / 9.0f);
            }

            // Interior ow ∈ [1, oW-1): 16-wide blocks. iw0 starts at 1
            // (ow-1+1=ow center; we load at offset iw0-1=0 which is in bounds).
            int ow = 1;
            // Process full 16-output blocks while iw0+16 ≤ iW, i.e. ow+16 ≤ iW.
            // Equivalently ow ≤ iW-16.
            int ow_simd_end = oW - 1;           // last ow we may process (exclusive)
            int ow_full_end = (iW - 16) + 1;    // last ow such that ow+16 ≤ iW
            if (ow_full_end > ow_simd_end) ow_full_end = ow_simd_end;
            for (; ow + 16 <= ow_full_end; ow += 16) {
                __m512 r0 = row3(rA, ow);
                __m512 r1 = row3(rB, ow);
                __m512 r2 = row3(rC, ow);
                __m512 sum = _mm512_add_ps(r0, _mm512_add_ps(r1, r2));
                _mm512_storeu_ps(dst + ow, _mm512_mul_ps(sum, vinv));
            }
            // Partial block (masked) for the final 1..15 interior outputs.
            if (ow < ow_simd_end) {
                int rem = ow_simd_end - ow;                 // ≥ 1, ≤ 15
                // Guard against iw0+16 > iW: the masked load reads up to
                // iw0+16 = ow+16 but the tap-offset +1 pushes it to ow+17.
                // We must cap rem so that ow+rem+1 ≤ iW.
                int max_rem = iW - ow - 1;                  // ≥ 1
                if (rem > max_rem) rem = max_rem;
                if (rem > 0) {
                    __mmask16 m = (__mmask16)((1u << rem) - 1);
                    __m512 r0 = row3_m(rA, ow, m);
                    __m512 r1 = row3_m(rB, ow, m);
                    __m512 r2 = row3_m(rC, ow, m);
                    __m512 sum = _mm512_add_ps(r0, _mm512_add_ps(r1, r2));
                    _mm512_mask_storeu_ps(dst + ow, m, _mm512_mul_ps(sum, vinv));
                    ow += rem;
                }
            }
            // Any remaining interior ow values (edge case: iW < 17) handled scalar.
            for (; ow < ow_simd_end; ++ow) {
                float s = rA[ow-1] + rA[ow] + rA[ow+1]
                        + rB[ow-1] + rB[ow] + rB[ow+1]
                        + rC[ow-1] + rC[ow] + rC[ow+1];
                dst[ow] = s * (1.0f / 9.0f);
            }
            // ow = oW-1: iw taps {oW-2, oW-1, oW} → only {oW-2, oW-1} are real.
            {
                int owR = oW - 1;
                float s = rA[owR-1] + rA[owR] + rB[owR-1] + rB[owR] + rC[owR-1] + rC[owR];
                dst[owR] = s * (1.0f / 9.0f);
            }
        }

        // --- Top row (oh=0) and bottom row (oh=oH-1): scalar ---
        // These are just 2*oW outputs; the border cost is negligible.
        auto border_row = [&](int oh) {
            int ih_c = oh;
            const float* rP = (ih_c - 1 >= 0) ? inp + (size_t)(ih_c - 1) * iW : nullptr;
            const float* rM = inp + (size_t)ih_c * iW;
            const float* rN = (ih_c + 1 < iH) ? inp + (size_t)(ih_c + 1) * iW : nullptr;
            float* dst = out + (size_t)oh * oW;
            for (int ow = 0; ow < oW; ++ow) {
                float s = 0;
                int iwL = ow - 1, iwR = ow + 1;
                if (iwL >= 0) { if (rP) s += rP[iwL]; s += rM[iwL]; if (rN) s += rN[iwL]; }
                { if (rP) s += rP[ow]; s += rM[ow]; if (rN) s += rN[ow]; }
                if (iwR < iW) { if (rP) s += rP[iwR]; s += rM[iwR]; if (rN) s += rN[iwR]; }
                dst[ow] = s * (1.0f / 9.0f);
            }
        };
        if (oH > 0) border_row(0);
        if (oH > 1) border_row(oH - 1);
    });
}

// ---------------------------------------------------------------------------
// NCHWc (BLOCKED_16) AveragePool 3x3 / stride=1 / pad=1 / count_include_pad=1.
// Specialization for adv_inception_v3's 9 inception-branch AvgPool nodes.
//
// Fixed divisor = 9 (count_include_pad counts the padded border positions).
// Border outputs get the padded positions contributing zero to the sum.
//
// Uses three row accumulators in the interior fast-path so the sum dependency
// chain is ~5 adds deep instead of 9 (the generic kernel's single-accumulator
// chain was the dominant bottleneck).
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=NCHWc
inline void avgpool_2d_nchwc_3x3s1p1_x64(const float* input, float* output,
    int N, int C, int iH, int iW)
{
    constexpr int block = 16;
    const int Cb = C / block;
    const int oH = iH, oW = iW;  // SAME padding, stride 1
    const size_t in_spatial  = (size_t)iH * iW * block;
    const size_t out_spatial = (size_t)oH * oW * block;
    const __m512 vinv = _mm512_set1_ps(1.0f / 9.0f);

    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const float* inp = input + (size_t)ncb * in_spatial;
        float* out = output + (size_t)ncb * out_spatial;

        auto row_sum = [&](int ih, int iw) -> __m512 {
            // Sum of 3 horizontal input vectors centered at (ih, iw). Padded
            // (out-of-bounds) positions contribute zero. sW=1, pW=1.
            __m512 a = (iw - 1 >= 0)
                ? _mm512_loadu_ps(&inp[(ih * iW + (iw - 1)) * block])
                : _mm512_setzero_ps();
            __m512 b = _mm512_loadu_ps(&inp[(ih * iW + iw) * block]);
            __m512 c = (iw + 1 < iW)
                ? _mm512_loadu_ps(&inp[(ih * iW + (iw + 1)) * block])
                : _mm512_setzero_ps();
            return _mm512_add_ps(a, _mm512_add_ps(b, c));
        };

        auto emit = [&](int oh, int ow, __m512 r0, __m512 r1, __m512 r2) {
            __m512 sum = _mm512_add_ps(r0, _mm512_add_ps(r1, r2));
            _mm512_storeu_ps(&out[(oh * oW + ow) * block],
                             _mm512_mul_ps(sum, vinv));
        };

        const __m512 vzero = _mm512_setzero_ps();

        // Top row (oh=0): rows -1, 0, 1; row -1 is padding → zero.
        if (oH > 0) {
            int oh = 0;
            for (int ow = 0; ow < oW; ++ow)
                emit(oh, ow, vzero, row_sum(0, ow),
                     (iH > 1 ? row_sum(1, ow) : vzero));
        }

        // Interior rows: oh ∈ [1, oH-1). Full 3x3 window, no row padding.
        // Interior columns ow ∈ [1, oW-1) have no column padding either —
        // that's the hot path. Borders (ow=0, ow=oW-1) fall through row_sum's
        // cheap runtime check.
        for (int oh = 1; oh + 1 < oH; ++oh) {
            const int ih1 = oh;  // center row
            const float* rowA = inp + (size_t)(ih1 - 1) * iW * block;
            const float* rowB = inp + (size_t)ih1       * iW * block;
            const float* rowC = inp + (size_t)(ih1 + 1) * iW * block;

            // ow=0 border (iw = -1, 0, 1): iw=-1 is padding.
            if (oW > 0) {
                __m512 r0 = _mm512_add_ps(
                    _mm512_loadu_ps(&rowA[0 * block]),
                    _mm512_loadu_ps(&rowA[1 * block]));
                __m512 r1 = _mm512_add_ps(
                    _mm512_loadu_ps(&rowB[0 * block]),
                    _mm512_loadu_ps(&rowB[1 * block]));
                __m512 r2 = _mm512_add_ps(
                    _mm512_loadu_ps(&rowC[0 * block]),
                    _mm512_loadu_ps(&rowC[1 * block]));
                emit(oh, 0, r0, r1, r2);
            }

            // Interior: ow ∈ [1, oW-1), no bounds checks. Three parallel
            // row accumulators break the serial dep chain.
            for (int ow = 1; ow + 1 < oW; ++ow) {
                const int iw0 = ow - 1;  // sW=1, pW=1 → iw0 >= 0
                __m512 r0 = _mm512_add_ps(
                    _mm512_loadu_ps(&rowA[(iw0 + 0) * block]),
                    _mm512_add_ps(
                        _mm512_loadu_ps(&rowA[(iw0 + 1) * block]),
                        _mm512_loadu_ps(&rowA[(iw0 + 2) * block])));
                __m512 r1 = _mm512_add_ps(
                    _mm512_loadu_ps(&rowB[(iw0 + 0) * block]),
                    _mm512_add_ps(
                        _mm512_loadu_ps(&rowB[(iw0 + 1) * block]),
                        _mm512_loadu_ps(&rowB[(iw0 + 2) * block])));
                __m512 r2 = _mm512_add_ps(
                    _mm512_loadu_ps(&rowC[(iw0 + 0) * block]),
                    _mm512_add_ps(
                        _mm512_loadu_ps(&rowC[(iw0 + 1) * block]),
                        _mm512_loadu_ps(&rowC[(iw0 + 2) * block])));
                emit(oh, ow, r0, r1, r2);
            }

            // ow=oW-1 border (iw = oW-2, oW-1, oW): last iw is padding.
            if (oW >= 2) {
                const int iw0 = oW - 2;
                __m512 r0 = _mm512_add_ps(
                    _mm512_loadu_ps(&rowA[(iw0 + 0) * block]),
                    _mm512_loadu_ps(&rowA[(iw0 + 1) * block]));
                __m512 r1 = _mm512_add_ps(
                    _mm512_loadu_ps(&rowB[(iw0 + 0) * block]),
                    _mm512_loadu_ps(&rowB[(iw0 + 1) * block]));
                __m512 r2 = _mm512_add_ps(
                    _mm512_loadu_ps(&rowC[(iw0 + 0) * block]),
                    _mm512_loadu_ps(&rowC[(iw0 + 1) * block]));
                emit(oh, oW - 1, r0, r1, r2);
            }
        }

        // Bottom row (oh = oH-1): rows oH-2, oH-1, oH; row oH is padding.
        if (oH >= 2) {
            int oh = oH - 1;
            for (int ow = 0; ow < oW; ++ow)
                emit(oh, ow, row_sum(oH - 2, ow), row_sum(oH - 1, ow), vzero);
        }
    });
}

// ---------------------------------------------------------------------------
// NCHWc (BLOCKED_16) MaxPool 3x3 / stride=2 / pad=0.
// Specialization for adv_inception_v3's 4 reduction MaxPool nodes.
//
// No padding → no bounds checks in the interior. Stride 2 means each output
// row touches 3 input rows and each output col touches 3 input cols starting
// at (2*oh, 2*ow). Full unroll + tree reduction over the 9 positions.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=NCHWc
inline void maxpool_2d_nchwc_3x3s2p0_x64(const float* input, float* output,
    int N, int C, int iH, int iW, int oH, int oW)
{
    constexpr int block = 16;
    const int Cb = C / block;
    const size_t in_spatial  = (size_t)iH * iW * block;
    const size_t out_spatial = (size_t)oH * oW * block;

    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const float* inp = input + (size_t)ncb * in_spatial;
        float* out = output + (size_t)ncb * out_spatial;

        for (int oh = 0; oh < oH; ++oh) {
            const int ih0 = oh * 2;
            const float* r0 = inp + (size_t)(ih0 + 0) * iW * block;
            const float* r1 = inp + (size_t)(ih0 + 1) * iW * block;
            const float* r2 = inp + (size_t)(ih0 + 2) * iW * block;
            for (int ow = 0; ow < oW; ++ow) {
                const int iw0 = ow * 2;
                // Three parallel row-max reductions.
                __m512 m0 = _mm512_max_ps(
                    _mm512_loadu_ps(&r0[(iw0 + 0) * block]),
                    _mm512_max_ps(
                        _mm512_loadu_ps(&r0[(iw0 + 1) * block]),
                        _mm512_loadu_ps(&r0[(iw0 + 2) * block])));
                __m512 m1 = _mm512_max_ps(
                    _mm512_loadu_ps(&r1[(iw0 + 0) * block]),
                    _mm512_max_ps(
                        _mm512_loadu_ps(&r1[(iw0 + 1) * block]),
                        _mm512_loadu_ps(&r1[(iw0 + 2) * block])));
                __m512 m2 = _mm512_max_ps(
                    _mm512_loadu_ps(&r2[(iw0 + 0) * block]),
                    _mm512_max_ps(
                        _mm512_loadu_ps(&r2[(iw0 + 1) * block]),
                        _mm512_loadu_ps(&r2[(iw0 + 2) * block])));
                __m512 vmax = _mm512_max_ps(m0, _mm512_max_ps(m1, m2));
                _mm512_storeu_ps(&out[(oh * oW + ow) * block], vmax);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHW MaxPool 3x3 / stride=2 / pad=0 — AVX-512.
// Specialization for adv_inception_v3's 4 NCHW MaxPool reduction nodes
// (73→35, 71→35, 35→17, 35→17). The generic NCHW SIMD path uses an
// AVX2 stride-2 shuffle at 4 outputs/iter; this does 8 outputs/iter with
// three parallel row-max chains and compresses even lanes via vcompressps.
//
// Each iteration loads 3 unaligned 16-wide vectors per row at offsets
// +0/+1/+2, computes a 16-lane stride-1 3-wide row-max, reduces the 3
// rows, then compresses lanes 0,2,...,14 → 8 stride-2 outputs.
//
// Safe range: iw0+17 < iW (the +2 shifted load reads [iw0+2..iw0+17]).
// Tail is scalar. No padding → no bounds checks on interior rows.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW
inline void maxpool_2d_nchw_3x3s2p0_x64(const float* input, float* output,
    int NC, int iH, int iW, int oH, int oW)
{
    // Mask 0x5555 = 0b0101010101010101 selects even lanes (0,2,4,...,14).
    // maskz_compress packs those 8 lanes into the low half of the result.
    const __mmask16 even_mask = 0x5555;

    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const float* inp = input + (size_t)nc * iH * iW;
        float* out       = output + (size_t)nc * oH * oW;

        // Largest ow such that the +2-shifted 16-wide load stays in bounds:
        //   iw0 + 17 <= iW - 1  =>  2*ow + 17 <= iW - 1  =>  ow <= (iW-18)/2.
        // +1 to make it an exclusive upper bound usable by `ow+8 <= ow_vec_end`.
        int ow_vec_end = (iW >= 18) ? ((iW - 18) / 2 + 1) : 0;
        if (ow_vec_end > oW) ow_vec_end = oW;

        for (int oh = 0; oh < oH; ++oh) {
            const int ih0 = oh * 2;
            const float* r0 = inp + (size_t)(ih0 + 0) * iW;
            const float* r1 = inp + (size_t)(ih0 + 1) * iW;
            const float* r2 = inp + (size_t)(ih0 + 2) * iW;
            float* dst = out + (size_t)oh * oW;

            int ow = 0;
            for (; ow + 8 <= ow_vec_end; ow += 8) {
                const int iw0 = ow * 2;
                // Three parallel row-max chains (break the dep chain so the
                // two vmax units on Zen 4 can co-issue).
                __m512 a = _mm512_max_ps(
                    _mm512_loadu_ps(r0 + iw0),
                    _mm512_max_ps(
                        _mm512_loadu_ps(r0 + iw0 + 1),
                        _mm512_loadu_ps(r0 + iw0 + 2)));
                __m512 b = _mm512_max_ps(
                    _mm512_loadu_ps(r1 + iw0),
                    _mm512_max_ps(
                        _mm512_loadu_ps(r1 + iw0 + 1),
                        _mm512_loadu_ps(r1 + iw0 + 2)));
                __m512 c = _mm512_max_ps(
                    _mm512_loadu_ps(r2 + iw0),
                    _mm512_max_ps(
                        _mm512_loadu_ps(r2 + iw0 + 1),
                        _mm512_loadu_ps(r2 + iw0 + 2)));
                __m512 vmax = _mm512_max_ps(a, _mm512_max_ps(b, c));
                // Pack even lanes into the low 8 positions.
                __m512 packed = _mm512_maskz_compress_ps(even_mask, vmax);
                _mm256_storeu_ps(dst + ow, _mm512_castps512_ps256(packed));
            }
            // Scalar tail (interior + right edge).
            for (; ow < oW; ++ow) {
                const int iw0 = ow * 2;
                float mx = r0[iw0];
                float v;
                v = r0[iw0 + 1]; if (v > mx) mx = v;
                v = r0[iw0 + 2]; if (v > mx) mx = v;
                v = r1[iw0    ]; if (v > mx) mx = v;
                v = r1[iw0 + 1]; if (v > mx) mx = v;
                v = r1[iw0 + 2]; if (v > mx) mx = v;
                v = r2[iw0    ]; if (v > mx) mx = v;
                v = r2[iw0 + 1]; if (v > mx) mx = v;
                v = r2[iw0 + 2]; if (v > mx) mx = v;
                dst[ow] = mx;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHW MaxPool 3x3 / stride=2 / pad=1 — AVX-512.
// ssd-12 pool1 ([64,600,600]→[300,300]). Same 8-outputs/iter structure as the
// p=0 variant, but iw0 = 2*ow - 1, so the left (ow=0) and top (oh=0) output
// cells overlap the padded border and are handled scalar. The interior vector
// path computes 8 stride-2 outputs per iteration exactly like the p=0 kernel.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW
inline void maxpool_2d_nchw_3x3s2p1_x64(const float* input, float* output,
    int NC, int iH, int iW, int oH, int oW)
{
    const __mmask16 even_mask = 0x5555;

    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const float* inp = input + (size_t)nc * iH * iW;
        float* out       = output + (size_t)nc * oH * oW;

        // Interior safe range for ow: iw0 = 2*ow-1; +2-shifted 16-wide load
        // touches [iw0..iw0+17], need iw0+17 <= iW-1, so ow <= (iW-18)/2 + 0,
        // but since iw0=2*ow-1 we need 2*ow+16 <= iW-1 → ow <= (iW-17)/2.
        int ow_vec_end = (iW >= 19) ? ((iW - 17) / 2 + 1) : 0;
        if (ow_vec_end > oW) ow_vec_end = oW;

        auto max3_pooled = [&](const float* r, int iw0) -> __m512 {
            return _mm512_max_ps(
                _mm512_loadu_ps(r + iw0),
                _mm512_max_ps(
                    _mm512_loadu_ps(r + iw0 + 1),
                    _mm512_loadu_ps(r + iw0 + 2)));
        };

        for (int oh = 0; oh < oH; ++oh) {
            const int ih0 = oh * 2 - 1;
            const bool top_pad    = (oh == 0);
            const bool bottom_pad = (ih0 + 2 >= iH);

            const float* r0 = top_pad    ? nullptr : inp + (size_t)(ih0 + 0) * iW;
            const float* r1 =                      inp + (size_t)(ih0 + 1) * iW;
            const float* r2 = bottom_pad ? nullptr : inp + (size_t)(ih0 + 2) * iW;
            float* dst = out + (size_t)oh * oW;

            // ow=0 is always scalar (iw starts at -1).
            {
                float mx = -FLT_MAX;
                const int iw0 = -1;
                for (int kh = 0; kh < 3; ++kh) {
                    int ih = ih0 + kh;
                    if ((unsigned)ih >= (unsigned)iH) continue;
                    const float* r = inp + (size_t)ih * iW;
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = iw0 + kw;
                        if ((unsigned)iw >= (unsigned)iW) continue;
                        float v = r[iw];
                        if (v > mx) mx = v;
                    }
                }
                dst[0] = mx;
            }

            int ow = 1;
            if (!top_pad && !bottom_pad) {
                // Full 3-row interior: 8 outputs per iteration, three parallel row-max chains.
                for (; ow + 8 <= ow_vec_end; ow += 8) {
                    const int iw0 = ow * 2 - 1;
                    __m512 a = max3_pooled(r0, iw0);
                    __m512 b = max3_pooled(r1, iw0);
                    __m512 c = max3_pooled(r2, iw0);
                    __m512 vmax = _mm512_max_ps(a, _mm512_max_ps(b, c));
                    __m512 packed = _mm512_maskz_compress_ps(even_mask, vmax);
                    _mm256_storeu_ps(dst + ow, _mm512_castps512_ps256(packed));
                }
            } else {
                // 2-row border (top or bottom) fast path.
                const float* rA = top_pad ? r1 : r0;
                const float* rB = top_pad ? r2 : r1;
                if (rB == nullptr) rB = rA;  // oH==1 edge case — shouldn't happen with pool1.
                for (; ow + 8 <= ow_vec_end; ow += 8) {
                    const int iw0 = ow * 2 - 1;
                    __m512 a = max3_pooled(rA, iw0);
                    __m512 b = max3_pooled(rB, iw0);
                    __m512 vmax = _mm512_max_ps(a, b);
                    __m512 packed = _mm512_maskz_compress_ps(even_mask, vmax);
                    _mm256_storeu_ps(dst + ow, _mm512_castps512_ps256(packed));
                }
            }

            // Scalar tail (interior remainder + right edge where iw0+2 may hit iW-1 or pad).
            for (; ow < oW; ++ow) {
                const int iw0 = ow * 2 - 1;
                float mx = -FLT_MAX;
                for (int kh = 0; kh < 3; ++kh) {
                    int ih = ih0 + kh;
                    if ((unsigned)ih >= (unsigned)iH) continue;
                    const float* r = inp + (size_t)ih * iW;
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = iw0 + kw;
                        if ((unsigned)iw >= (unsigned)iW) continue;
                        float v = r[iw];
                        if (v > mx) mx = v;
                    }
                }
                dst[ow] = mx;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// NCHW MaxPool 3x3 / stride=1 / pad=1 — AVX-512.
// googlenet inception-branch pool nodes (14×14, 28×28, 7×7, etc). Three
// parallel row-max chains over 16-wide lanes, shifted-load trick for the
// horizontal 3-tap; border rows zero-masked inputs by collapsing to 2-row.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW
inline void maxpool_2d_nchw_3x3s1p1_x64(const float* input, float* output,
    int NC, int iH, int iW)
{
    const int oH = iH, oW = iW;

    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const float* inp = input + (size_t)nc * iH * iW;
        float* out       = output + (size_t)nc * oH * oW;

        auto row3 = [](const float* row, int iw0) -> __m512 {
            __m512 a = _mm512_loadu_ps(row + iw0 - 1);
            __m512 b = _mm512_loadu_ps(row + iw0);
            __m512 c = _mm512_loadu_ps(row + iw0 + 1);
            return _mm512_max_ps(a, _mm512_max_ps(b, c));
        };
        auto row3_m = [](const float* row, int iw0, __mmask16 m) -> __m512 {
            const __m512 neg_inf = _mm512_set1_ps(-FLT_MAX);
            __m512 a = _mm512_mask_loadu_ps(neg_inf, m, row + iw0 - 1);
            __m512 b = _mm512_mask_loadu_ps(neg_inf, m, row + iw0);
            __m512 c = _mm512_mask_loadu_ps(neg_inf, m, row + iw0 + 1);
            return _mm512_max_ps(a, _mm512_max_ps(b, c));
        };

        auto scalar_cell = [&](int oh, int ow) -> float {
            const int ih0 = oh - 1;
            const int iw0 = ow - 1;
            float mx = -FLT_MAX;
            for (int kh = 0; kh < 3; ++kh) {
                int ih = ih0 + kh;
                if ((unsigned)ih >= (unsigned)iH) continue;
                const float* r = inp + (size_t)ih * iW;
                for (int kw = 0; kw < 3; ++kw) {
                    int iw = iw0 + kw;
                    if ((unsigned)iw >= (unsigned)iW) continue;
                    float v = r[iw];
                    if (v > mx) mx = v;
                }
            }
            return mx;
        };

        // Interior rows (oh ∈ [1, oH-1)).
        for (int oh = 1; oh + 1 < oH; ++oh) {
            const float* rA = inp + (size_t)(oh - 1) * iW;
            const float* rB = inp + (size_t)(oh    ) * iW;
            const float* rC = inp + (size_t)(oh + 1) * iW;
            float* dst = out + (size_t)oh * oW;

            // ow = 0 scalar (iw0 = -1 → the left tap is padded).
            dst[0] = scalar_cell(oh, 0);

            // Interior ow ∈ [1, oW-1): iw0 = ow. Full 16-wide blocks need
            // iw0+16 ≤ iW, i.e. ow+16 ≤ iW → ow ≤ iW-16.
            int ow = 1;
            int ow_simd_end = oW - 1;            // last interior ow (exclusive)
            int ow_full_end = iW - 16 + 1;       // ow full-block end
            if (ow_full_end > ow_simd_end) ow_full_end = ow_simd_end;
            for (; ow + 16 <= ow_full_end; ow += 16) {
                __m512 r0 = row3(rA, ow);
                __m512 r1 = row3(rB, ow);
                __m512 r2 = row3(rC, ow);
                _mm512_storeu_ps(dst + ow, _mm512_max_ps(r0, _mm512_max_ps(r1, r2)));
            }
            // Masked partial block.
            if (ow < ow_simd_end) {
                int rem = ow_simd_end - ow;
                int max_rem = iW - ow - 1;
                if (rem > max_rem) rem = max_rem;
                if (rem > 0) {
                    __mmask16 m = (__mmask16)((1u << rem) - 1);
                    __m512 r0 = row3_m(rA, ow, m);
                    __m512 r1 = row3_m(rB, ow, m);
                    __m512 r2 = row3_m(rC, ow, m);
                    _mm512_mask_storeu_ps(dst + ow, m,
                        _mm512_max_ps(r0, _mm512_max_ps(r1, r2)));
                    ow += rem;
                }
            }
            // Remaining interior scalar (for very narrow iW).
            for (; ow < ow_simd_end; ++ow) dst[ow] = scalar_cell(oh, ow);
            // ow = oW-1 scalar (right border, iw+1 = oW → padded).
            dst[oW - 1] = scalar_cell(oh, oW - 1);
        }

        // Top/bottom border rows — small, scalar fine.
        auto border_row = [&](int oh) {
            float* dst = out + (size_t)oh * oW;
            for (int ow = 0; ow < oW; ++ow) dst[ow] = scalar_cell(oh, ow);
        };
        if (oH > 0) border_row(0);
        if (oH > 1) border_row(oH - 1);
    });
}

// ---------------------------------------------------------------------------
// NCHW AveragePool 3x3 / stride=1 / pad=1 / count_include_pad=1 — prefix-sum.
// Refinement of `avgpool_2d_nchw_3x3s1p1_cip_x64`.
//
// Per-row horizontal prefix-sum reuses each input load across 3 output rows:
//   hsum[ih][iw] = in[ih][iw-1] + in[ih][iw] + in[ih][iw+1]   (pad=0 at edges)
//   out[oh][ow]  = (hsum[oh-1][ow] + hsum[oh][ow] + hsum[oh+1][ow]) / 9
//
// Loads per 16 outputs drop from 9 (3 taps × 3 rows) to ~4 (3 hsum loads +
// 1/3 of a compute pass amortized across 3 output rows). Zen 4 zmm load
// throughput is the dominant term here.
//
// Rolling 3-row scratch on stack — avoids O(iH*iW) scratch and avoids
// any heap allocation. iW capped at MAX_W; large-iW cases fall back to
// the 3-parallel-row kernel via dispatch in AveragePool.cpp.
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=NCHW
inline void avgpool_2d_nchw_3x3s1p1_cip_x64_psum(const float* input, float* output,
    int NC, int iH, int iW)
{
    // 2-pass per channel: first compute hsum[iH][iW] into scratch, then emit
    // all output rows from scratch. Decoupling the write pass from the read
    // pass avoids store-to-load forwarding stalls that plagued the rolling
    // variant. Caller gates iH,iW <= MAX_DIM so scratch fits in L1.
    constexpr int MAX_DIM = 64;
    const int oH = iH, oW = iW;
    const __m512 vinv = _mm512_set1_ps(1.0f / 9.0f);

    // hsum[iw] = row[iw-1] + row[iw] + row[iw+1], with zero pad at iw=-1, iW.
    auto compute_hsum = [iW](const float* row, float* hrow) {
        // Border iw=0: row[-1]=0 + row[0] + row[1].
        if (iW == 1) { hrow[0] = row[0]; return; }
        hrow[0] = row[0] + row[1];
        int iw = 1;
        const int end = iW - 1;
        for (; iw + 16 <= end; iw += 16) {
            __m512 a = _mm512_loadu_ps(row + iw - 1);
            __m512 b = _mm512_loadu_ps(row + iw    );
            __m512 c = _mm512_loadu_ps(row + iw + 1);
            _mm512_storeu_ps(hrow + iw, _mm512_add_ps(a, _mm512_add_ps(b, c)));
        }
        // Masked tail (interior positions still fully 3-wide).
        int rem = end - iw;
        if (rem > 0) {
            __mmask16 m = (__mmask16)((1u << rem) - 1);
            __m512 a = _mm512_maskz_loadu_ps(m, row + iw - 1);
            __m512 b = _mm512_maskz_loadu_ps(m, row + iw    );
            __m512 c = _mm512_maskz_loadu_ps(m, row + iw + 1);
            _mm512_mask_storeu_ps(hrow + iw, m,
                _mm512_add_ps(a, _mm512_add_ps(b, c)));
        }
        // Border iw=iW-1: row[iW-2] + row[iW-1] + row[iW]=0.
        hrow[iW - 1] = row[iW - 2] + row[iW - 1];
    };

    // 3-sum across 3 hsum rows, scaled by 1/9.
    auto emit_row_3 = [&](const float* hT, const float* hM, const float* hB,
                          float* dst) {
        int ow = 0;
        for (; ow + 16 <= oW; ow += 16) {
            __m512 s = _mm512_add_ps(_mm512_loadu_ps(hT + ow),
                       _mm512_add_ps(_mm512_loadu_ps(hM + ow),
                                     _mm512_loadu_ps(hB + ow)));
            _mm512_storeu_ps(dst + ow, _mm512_mul_ps(s, vinv));
        }
        int rem = oW - ow;
        if (rem > 0) {
            __mmask16 m = (__mmask16)((1u << rem) - 1);
            __m512 s = _mm512_add_ps(_mm512_maskz_loadu_ps(m, hT + ow),
                       _mm512_add_ps(_mm512_maskz_loadu_ps(m, hM + ow),
                                     _mm512_maskz_loadu_ps(m, hB + ow)));
            _mm512_mask_storeu_ps(dst + ow, m, _mm512_mul_ps(s, vinv));
        }
    };
    // 2-sum for border rows (padded row contributes zero).
    auto emit_row_2 = [&](const float* hM, const float* hB, float* dst) {
        int ow = 0;
        for (; ow + 16 <= oW; ow += 16) {
            __m512 s = _mm512_add_ps(_mm512_loadu_ps(hM + ow),
                                     _mm512_loadu_ps(hB + ow));
            _mm512_storeu_ps(dst + ow, _mm512_mul_ps(s, vinv));
        }
        int rem = oW - ow;
        if (rem > 0) {
            __mmask16 m = (__mmask16)((1u << rem) - 1);
            __m512 s = _mm512_add_ps(_mm512_maskz_loadu_ps(m, hM + ow),
                                     _mm512_maskz_loadu_ps(m, hB + ow));
            _mm512_mask_storeu_ps(dst + ow, m, _mm512_mul_ps(s, vinv));
        }
    };

    nnr::for_static(0, NC, NC > 4 && oH * oW > 64, [&](int nc) {
        const float* inp = input + (size_t)nc * iH * iW;
        float* out       = output + (size_t)nc * oH * oW;

        // Pass 1: compute hsum for all iH rows into a contiguous scratch.
        // Scratch is stack-allocated, bounded by MAX_DIM*MAX_DIM (16 KB).
        alignas(64) float hsum_buf[MAX_DIM * MAX_DIM];
        for (int ih = 0; ih < iH; ++ih) {
            compute_hsum(inp + (size_t)ih * iW, hsum_buf + (size_t)ih * iW);
        }

        // Pass 2: emit output rows from the hsum scratch.
        // Top row (oh=0): padded top contributes zero; sum of hsum[0] + hsum[1]
        // (or just hsum[0] if iH==1).
        if (iH == 1) {
            int ow = 0;
            for (; ow + 16 <= oW; ow += 16) {
                __m512 s = _mm512_loadu_ps(hsum_buf + ow);
                _mm512_storeu_ps(out + ow, _mm512_mul_ps(s, vinv));
            }
            int rem = oW - ow;
            if (rem > 0) {
                __mmask16 m = (__mmask16)((1u << rem) - 1);
                __m512 s = _mm512_maskz_loadu_ps(m, hsum_buf + ow);
                _mm512_mask_storeu_ps(out + ow, m, _mm512_mul_ps(s, vinv));
            }
            return;
        }

        emit_row_2(hsum_buf + 0 * iW, hsum_buf + 1 * iW, out + 0 * oW);
        // Interior rows oh ∈ [1, oH-1): full 3-sum.
        for (int oh = 1; oh + 1 < oH; ++oh) {
            emit_row_3(hsum_buf + (size_t)(oh - 1) * iW,
                       hsum_buf + (size_t)oh       * iW,
                       hsum_buf + (size_t)(oh + 1) * iW,
                       out + (size_t)oh * oW);
        }
        // Bottom row (oh = oH-1): padded bottom contributes zero.
        if (oH >= 2) {
            emit_row_2(hsum_buf + (size_t)(oH - 2) * iW,
                       hsum_buf + (size_t)(oH - 1) * iW,
                       out + (size_t)(oH - 1) * oW);
        }
    });
}

// ---------------------------------------------------------------------------
// NCHWc (BLOCKED_16) GlobalAveragePool — AVX-512.
// input:  [N, Cb, H, W, 16]
// output: [N, Cb, 1, 1, 16]
// ---------------------------------------------------------------------------
// @nnr-meta isa=AVX512 dtype=fp32 layout=BLOCKED_16 special=NCHWc
inline void global_avgpool_nchwc_x64(const float* input, float* output,
    int N, int C, int spatial)
{
    constexpr int block = 16;
    const int Cb = C / block;
    const int NCb = N * Cb;
    nnr::for_static(0, NCb, NCb > 1, [&](int ncb) {
        const float* inp = input + (size_t)ncb * spatial * block;
        float* out = output + (size_t)ncb * block;
        __m512 vsum = _mm512_setzero_ps();
        for (int s = 0; s < spatial; ++s)
            vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(&inp[s * block]));
        __m512 vinv = _mm512_set1_ps(1.0f / (float)spatial);
        _mm512_storeu_ps(out, _mm512_mul_ps(vsum, vinv));
    });
}

} // namespace nnr
