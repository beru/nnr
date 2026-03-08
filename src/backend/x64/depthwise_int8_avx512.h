#pragma once
// AVX-512 int8 depthwise convolution (NHWC layout).
// Vectorizes across channels: 16 channels per zmm register.
// Uses indirection buffer to handle padding without branches in the inner loop.
// Processes 6 output pixels per iteration to amortize filter loads.

#ifdef NNR_ARCH_X64
#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include "thread_pool.h"

namespace nnr {
namespace int8 {

// Repack depthwise weights from ONNX [OC, 1, kH, kW] to [kH*kW, OC_padded].
// OC_padded is rounded up to 16 for AVX-512 alignment.
// @nnr-meta isa=scalar dtype=int8 layout=NHWC special=DW
inline void repack_depthwise_weights(
    int8_t* dst, const int8_t* src,
    int OC, int kH, int kW)
{
    int OC_pad = (OC + 15) & ~15;
    int kHW = kH * kW;
    memset(dst, 0, (size_t)kHW * OC_pad);
    for (int oc = 0; oc < OC; oc++)
        for (int k = 0; k < kHW; k++)
            dst[k * OC_pad + oc] = src[oc * kHW + k];
}

// @nnr-meta isa=scalar
inline size_t repack_depthwise_weights_size(int OC, int kH, int kW) {
    return (size_t)kH * kW * ((OC + 15) & ~15);
}

// Build indirection buffer for depthwise conv.
// ind[output_pixel * kHW + k] points to the input row for that (pixel, kernel_tap).
// Padding pixels point to zero_buf (filled with x_zp).
// Assumes dilation=1.
// @nnr-meta isa=scalar layout=NHWC special=DW
inline void build_depthwise_indirection(
    const uint8_t** ind,
    const uint8_t* x_nhwc,
    const uint8_t* zero_buf,
    int oH, int oW, int iH, int iW, int C,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int dH, int dW)
{
    int kHW = kH * kW;
    for (int oh = 0; oh < oH; oh++) {
        for (int ow = 0; ow < oW; ow++) {
            int out_idx = oh * oW + ow;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * sH - pH + kh * dH;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * sW - pW + kw * dW;
                    int k = kh * kW + kw;
                    if ((unsigned)ih < (unsigned)iH && (unsigned)iw < (unsigned)iW)
                        ind[out_idx * kHW + k] = x_nhwc + ((size_t)ih * iW + iw) * C;
                    else
                        ind[out_idx * kHW + k] = zero_buf;
                }
            }
        }
    }
}

// Int8 depthwise conv, NHWC layout, indirection-based.
// Processes 6 output pixels per iteration to amortize filter loads across pixels.
// filter: [kH*kW, C_pad] repacked weights (from repack_depthwise_weights).
// indirection: [output_count * kernel_size] pointers to input rows.
// combined_scale: [C] — x_scale * w_scale[c].
// bias_f: [C] — (float)bias[c] * combined_scale[c], or nullptr.
static constexpr int DW_PIXEL_BLOCK = 6;

// @nnr-meta isa=AVX512 dtype=[int8,uint8] layout=NHWC special=DW
inline void depthwise_int8_nhwc_avx512(
    uint8_t* y_nhwc,
    const uint8_t* const* indirection,
    const int8_t* filter,
    const float* combined_scale,
    const float* bias_f,
    int C, int output_count, int kernel_size,
    int x_zp, int w_zp,
    float inv_y_scale, float y_zp_f, float qmin, float qmax)
{
    int C_pad = (C + 15) & ~15;
    __m512 v_inv_y = _mm512_set1_ps(inv_y_scale);
    __m512 v_y_zp  = _mm512_set1_ps(y_zp_f);
    __m512 v_qmin  = _mm512_set1_ps(qmin);
    __m512 v_qmax  = _mm512_set1_ps(qmax);
    __m512i v_x_zp = _mm512_set1_epi32(x_zp);
    __m512i v_w_zp = _mm512_set1_epi32(w_zp);

    int op = 0;
    for (; op + DW_PIXEL_BLOCK <= output_count; op += DW_PIXEL_BLOCK) {
        const uint8_t* ptrs[DW_PIXEL_BLOCK][25]; // max kernel_size = 5×5 = 25
        for (int p = 0; p < DW_PIXEL_BLOCK; p++)
            for (int k = 0; k < kernel_size; k++)
                ptrs[p][k] = indirection[(op + p) * kernel_size + k];

        for (int c = 0; c + 16 <= C; c += 16) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i acc4 = _mm512_setzero_si512();
            __m512i acc5 = _mm512_setzero_si512();

            for (int k = 0; k < kernel_size; k++) {
                __m128i fv8 = _mm_loadu_si128((const __m128i*)(filter + k * C_pad + c));
                __m512i fv = _mm512_sub_epi32(_mm512_cvtepi8_epi32(fv8), v_w_zp);

                #define DW_PIXEL_MAC(IDX, ACC) { \
                    __m128i xv8 = _mm_loadu_si128((const __m128i*)(ptrs[IDX][k] + c)); \
                    __m512i xv = _mm512_sub_epi32(_mm512_cvtepu8_epi32(xv8), v_x_zp); \
                    ACC = _mm512_add_epi32(ACC, _mm512_mullo_epi32(xv, fv)); \
                }
                DW_PIXEL_MAC(0, acc0);
                DW_PIXEL_MAC(1, acc1);
                DW_PIXEL_MAC(2, acc2);
                DW_PIXEL_MAC(3, acc3);
                DW_PIXEL_MAC(4, acc4);
                DW_PIXEL_MAC(5, acc5);
                #undef DW_PIXEL_MAC
            }

            __m512 vcs = _mm512_loadu_ps(combined_scale + c);
            __m512 vbias = bias_f ? _mm512_loadu_ps(bias_f + c) : _mm512_setzero_ps();

            #define DW_REQUANTIZE_STORE(ACC, P_IDX) { \
                __m512 facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ACC), vcs, vbias); \
                __m512 q = _mm512_mul_ps(facc, v_inv_y); \
                q = _mm512_roundscale_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); \
                q = _mm512_add_ps(q, v_y_zp); \
                q = _mm512_max_ps(_mm512_min_ps(q, v_qmax), v_qmin); \
                _mm_storeu_si128((__m128i*)(y_nhwc + (size_t)(op + P_IDX) * C + c), \
                    _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(q))); \
            }
            DW_REQUANTIZE_STORE(acc0, 0);
            DW_REQUANTIZE_STORE(acc1, 1);
            DW_REQUANTIZE_STORE(acc2, 2);
            DW_REQUANTIZE_STORE(acc3, 3);
            DW_REQUANTIZE_STORE(acc4, 4);
            DW_REQUANTIZE_STORE(acc5, 5);
            #undef DW_REQUANTIZE_STORE
        }
    }

    // Tail: remaining pixels (< DW_PIXEL_BLOCK), 1 at a time
    for (; op < output_count; op++) {
        for (int c = 0; c + 16 <= C; c += 16) {
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < kernel_size; k++) {
                __m128i fv8 = _mm_loadu_si128((const __m128i*)(filter + k * C_pad + c));
                __m512i fv = _mm512_sub_epi32(_mm512_cvtepi8_epi32(fv8), v_w_zp);
                __m128i xv8 = _mm_loadu_si128((const __m128i*)(indirection[op * kernel_size + k] + c));
                __m512i xv = _mm512_sub_epi32(_mm512_cvtepu8_epi32(xv8), v_x_zp);
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(xv, fv));
            }
            __m512 vcs = _mm512_loadu_ps(combined_scale + c);
            __m512 vbias = bias_f ? _mm512_loadu_ps(bias_f + c) : _mm512_setzero_ps();
            __m512 facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), vcs, vbias);
            __m512 q = _mm512_mul_ps(facc, v_inv_y);
            q = _mm512_roundscale_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            q = _mm512_add_ps(q, v_y_zp);
            q = _mm512_max_ps(_mm512_min_ps(q, v_qmax), v_qmin);
            _mm_storeu_si128((__m128i*)(y_nhwc + (size_t)op * C + c),
                _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(q)));
        }
    }
}

// Threaded wrapper: splits output pixels across threads by row.
// @nnr-meta isa=AVX512 dtype=[int8,uint8] layout=NHWC special=DW
inline void depthwise_int8_nhwc_avx512_mt(
    uint8_t* y_nhwc,
    const uint8_t* const* indirection,
    const int8_t* filter,
    const float* combined_scale,
    const float* bias_f,
    int C, int oH, int oW, int kernel_size,
    int x_zp, int w_zp,
    float inv_y_scale, float y_zp_f, float qmin, float qmax)
{
    int output_count = oH * oW;
    bool par = oH >= 4 && C >= 32;
    nnr::for_static(0, oH, par, [&](int oh) {
        int op_start = oh * oW;
        int op_count = oW;
        depthwise_int8_nhwc_avx512(
            y_nhwc + (size_t)op_start * C,
            indirection + (size_t)op_start * kernel_size,
            filter, combined_scale, bias_f,
            C, op_count, kernel_size,
            x_zp, w_zp,
            inv_y_scale, y_zp_f, qmin, qmax);
    });
}

// ── Direct-addressing depthwise kernel (no indirection buffer) ────────────────
// Computes input addresses inline from output coordinates.
// Eliminates 903KB indirection buffer and its random pointer chasing.
// Interior pixels (no padding) use a branch-free fast path.

// @nnr-meta isa=AVX512 dtype=[int8,uint8] layout=NHWC special=[DW,Direct]
inline void depthwise_int8_nhwc_avx512_direct_row(
    uint8_t* y_row,                    // output: oW * C bytes
    const uint8_t* x_nhwc,            // full input tensor (NHWC)
    const int8_t* filter,              // [kH*kW, C_pad] repacked
    const float* combined_scale,
    const float* bias_f,
    const uint8_t* zero_buf,           // C bytes of x_zp for padding
    int C, int C_pad, int oW, int iH, int iW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int oh,                            // current output row
    __m512i v_x_zp, __m512i v_w_zp,
    __m512 v_inv_y, __m512 v_y_zp, __m512 v_qmin, __m512 v_qmax)
{
    const int kHW = kH * kW;

    // Determine interior column range (no horizontal padding needed)
    int ow_inner_start = (pW + sW - 1) / sW;  // first ow where all kw are in-bounds
    int ow_inner_end   = std::min(oW, (iW + pW - kW) / sW + 1);  // last+1 ow fully in-bounds
    if (ow_inner_end < ow_inner_start) ow_inner_end = ow_inner_start;

    // Check which kernel rows are vertically in-bounds for this output row
    int ih_base = oh * sH - pH;

    auto process_pixel_interior = [&](int ow, uint8_t* py) {
        int iw_base = ow * sW - pW;
        for (int c = 0; c + 16 <= C; c += 16) {
            __m512i acc = _mm512_setzero_si512();
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih_base + kh;
                const uint8_t* row_ptr = x_nhwc + (size_t)ih * iW * C;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw_base + kw;
                    int k = kh * kW + kw;
                    __m128i fv8 = _mm_loadu_si128((const __m128i*)(filter + k * C_pad + c));
                    __m512i fv = _mm512_sub_epi32(_mm512_cvtepi8_epi32(fv8), v_w_zp);
                    __m128i xv8 = _mm_loadu_si128((const __m128i*)(row_ptr + (size_t)iw * C + c));
                    __m512i xv = _mm512_sub_epi32(_mm512_cvtepu8_epi32(xv8), v_x_zp);
                    acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(xv, fv));
                }
            }
            __m512 vcs = _mm512_loadu_ps(combined_scale + c);
            __m512 vbias = bias_f ? _mm512_loadu_ps(bias_f + c) : _mm512_setzero_ps();
            __m512 facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), vcs, vbias);
            __m512 q = _mm512_mul_ps(facc, v_inv_y);
            q = _mm512_roundscale_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            q = _mm512_add_ps(q, v_y_zp);
            q = _mm512_max_ps(_mm512_min_ps(q, v_qmax), v_qmin);
            _mm_storeu_si128((__m128i*)(py + c),
                _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(q)));
        }
    };

    auto process_pixel_border = [&](int ow, uint8_t* py) {
        int iw_base = ow * sW - pW;
        for (int c = 0; c + 16 <= C; c += 16) {
            __m512i acc = _mm512_setzero_si512();
            for (int kh = 0; kh < kH; kh++) {
                int ih = ih_base + kh;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = iw_base + kw;
                    int k = kh * kW + kw;
                    __m128i fv8 = _mm_loadu_si128((const __m128i*)(filter + k * C_pad + c));
                    __m512i fv = _mm512_sub_epi32(_mm512_cvtepi8_epi32(fv8), v_w_zp);
                    const uint8_t* src;
                    if ((unsigned)ih < (unsigned)iH && (unsigned)iw < (unsigned)iW)
                        src = x_nhwc + ((size_t)ih * iW + iw) * C + c;
                    else
                        src = zero_buf + c;
                    __m128i xv8 = _mm_loadu_si128((const __m128i*)src);
                    __m512i xv = _mm512_sub_epi32(_mm512_cvtepu8_epi32(xv8), v_x_zp);
                    acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(xv, fv));
                }
            }
            __m512 vcs = _mm512_loadu_ps(combined_scale + c);
            __m512 vbias = bias_f ? _mm512_loadu_ps(bias_f + c) : _mm512_setzero_ps();
            __m512 facc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), vcs, vbias);
            __m512 q = _mm512_mul_ps(facc, v_inv_y);
            q = _mm512_roundscale_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            q = _mm512_add_ps(q, v_y_zp);
            q = _mm512_max_ps(_mm512_min_ps(q, v_qmax), v_qmin);
            _mm_storeu_si128((__m128i*)(py + c),
                _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(q)));
        }
    };

    // Check if this row has any vertical padding
    bool has_vpad = (ih_base < 0) || (ih_base + kH > iH);

    if (has_vpad) {
        // Border row: all pixels need bounds checking
        for (int ow = 0; ow < oW; ow++)
            process_pixel_border(ow, y_row + (size_t)ow * C);
    } else {
        // Left border
        for (int ow = 0; ow < ow_inner_start; ow++)
            process_pixel_border(ow, y_row + (size_t)ow * C);

        // Interior: no bounds checks needed
        for (int ow = ow_inner_start; ow < ow_inner_end; ow++)
            process_pixel_interior(ow, y_row + (size_t)ow * C);

        // Right border
        for (int ow = ow_inner_end; ow < oW; ow++)
            process_pixel_border(ow, y_row + (size_t)ow * C);
    }
}

// Threaded direct-addressing depthwise.
// No indirection buffer needed — saves allocation + build + cache pressure.
// @nnr-meta isa=AVX512 dtype=[int8,uint8] layout=NHWC special=[DW,Direct]
inline void depthwise_int8_nhwc_avx512_direct(
    uint8_t* y_nhwc,
    const uint8_t* x_nhwc,
    const int8_t* filter,
    const float* combined_scale,
    const float* bias_f,
    const uint8_t* zero_buf,
    int C, int oH, int oW, int iH, int iW,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int x_zp, int w_zp,
    float inv_y_scale, float y_zp_f, float qmin, float qmax)
{
    int C_pad = (C + 15) & ~15;
    __m512i v_x_zp = _mm512_set1_epi32(x_zp);
    __m512i v_w_zp = _mm512_set1_epi32(w_zp);
    __m512 v_inv_y = _mm512_set1_ps(inv_y_scale);
    __m512 v_y_zp  = _mm512_set1_ps(y_zp_f);
    __m512 v_qmin  = _mm512_set1_ps(qmin);
    __m512 v_qmax  = _mm512_set1_ps(qmax);

    // Threading heuristic: skip threading when per-thread work is too small.
    // For C=32, oW=112: ~10µs per row. With 24 threads and 5 rows each,
    // threading overhead (5-20µs) dominates the 2µs of compute per thread.
    int64_t ops_per_row = (int64_t)oW * C * kH * kW;
    bool par = oH >= 4 && ops_per_row >= (1 << 14);  // ~16K ops/row minimum

    nnr::for_static(0, oH, par, [&](int oh) {
        depthwise_int8_nhwc_avx512_direct_row(
            y_nhwc + (size_t)oh * oW * C,
            x_nhwc, filter, combined_scale, bias_f, zero_buf,
            C, C_pad, oW, iH, iW,
            kH, kW, sH, sW, pH, pW, oh,
            v_x_zp, v_w_zp, v_inv_y, v_y_zp, v_qmin, v_qmax);
    });
}

} // namespace int8
} // namespace nnr

#endif // NNR_ARCH_X64
