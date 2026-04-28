#include "nnr.h"
#include "tensor_view.h"
#include "util.h"
#include "thread_pool.h"
#include "cpu_features.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/ops_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/vec_ops_neon.h"
#endif

namespace nnr {

namespace {

struct BatchNormalization_operator : public operator_t {
    float epsilon;
    float momentum;
    int training_mode = 0;

    bool init() override {
        if (!(inputs.size() == 5 && outputs.size() >= 1)) {
            return false;
        }
        epsilon = attribute(attr_key_t::epsilon, 1e-05f);
        momentum = attribute(attr_key_t::momentum, 0.9f);
        training_mode = attribute(attr_key_t::training_mode, 0);
        return true;
    }

    bool reshape() override {
        // Preserve a quantized output type set by the BN→Q fold
        // (fold_bn_qdq.cpp re-types y to UINT8/INT8 and rewrites scale/bias
        // into combined A[c]/B[c]). reshape_identity() with the default
        // type would clobber it back to the fp32 input type; downstream
        // QLinearMul/QLinearAdd would then see fp32 and bail.
        const data_type_t y_type = outputs[0]->type;
        const bool fold_typed = (y_type == NNR_DATA_TYPE_UINT8 || y_type == NNR_DATA_TYPE_INT8);
        const data_type_t out_type = fold_typed ? y_type : inputs[0]->type;
        if (!outputs[0]->reshape_identity(inputs[0], out_type))
            return false;
        // NHWC support for 4D inputs in inference mode.
        // uint8 is the fused DQ→BN→Q path (scale/bias already rewritten to
        // combined A[c]/B[c] by fold_bn_qdq); it supports NHWC too so the
        // dense-block NHWC chain can propagate across the fold.
        if (!training_mode && inputs[0]->ndim == 4
            && (inputs[0]->type == NNR_DATA_TYPE_FLOAT32
                || inputs[0]->type == NNR_DATA_TYPE_UINT8))
            layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
        // Training mode outputs: running_mean, running_var, saved_mean, saved_var
        if (outputs.size() > 1 && outputs[1]) {
            small_vector<int> cdims(1);
            cdims[0] = inputs[0]->dims[1]; // C
            for (size_t i = 1; i < outputs.size() && i <= 4; ++i) {
                if (outputs[i]) {
                    if (!outputs[i]->reshape(cdims, inputs[0]->type))
                        return false;
                }
            }
        }
        return true;
    }

    bool supports_strided_output(memory_layout_t format) const override {
        if (format != memory_layout_t::NHWC) return false;
        if (training_mode) return false;
        if (outputs.empty() || !outputs[0]) return false;
        if (outputs[0]->ndim != 4) return false;
        // M5 wired LDC into the fp32 NHWC inference path only. uint8
        // (fold_bn_qdq fast path) and fp32→uint8 paths still write contiguous
        // — opt them in when the kernels learn LDC.
        return outputs[0]->type == NNR_DATA_TYPE_FLOAT32;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* scale = inputs[1];
        const tensor_t* b = inputs[2];
        const tensor_t* input_mean = inputs[3];
        const tensor_t* input_var = inputs[4];
        tensor_t* y = outputs[0];
        const T* px = (const T*)x->data;
        const T* pscale = (const T*)scale->data;
        const T* pb = (const T*)b->data;
        T* py = (T*)y->data;
        int N = x->dims[0];
        int C = x->dims[1];
        int NC = N * C;
        int channel = std::reduce(x->dims + 2, x->dims + x->ndim, 1, std::multiplies<>{});

        if (training_mode && outputs.size() >= 3) {
            // Training mode: compute batch statistics
            arena_scope_t scope(ctx->arena);
            double* batch_mean = scope.alloc_arr<double>(C);
            double* batch_var  = scope.alloc_arr<double>(C);
            memset(batch_mean, 0, C * sizeof(double));
            memset(batch_var,  0, C * sizeof(double));
            int count = N * channel;

            // Compute mean
            for (int j = 0; j < NC; ++j) {
                int o = j * channel;
                int c = j % C;
                for (int i = 0; i < channel; ++i)
                    batch_mean[c] += (double)px[o + i];
            }
            for (int c = 0; c < C; ++c)
                batch_mean[c] /= count;

            // Compute variance
            for (int j = 0; j < NC; ++j) {
                int o = j * channel;
                int c = j % C;
                for (int i = 0; i < channel; ++i) {
                    double d = (double)px[o + i] - batch_mean[c];
                    batch_var[c] += d * d;
                }
            }
            for (int c = 0; c < C; ++c)
                batch_var[c] /= count;

            // Normalize
            for (int j = 0; j < NC; ++j) {
                int o = j * channel;
                int c = j % C;
                double denom = sqrt(batch_var[c] + epsilon);
                for (int i = 0; i < channel; ++i)
                    py[o + i] = (T)((double)pscale[c] * (((double)px[o + i] - batch_mean[c]) / denom) + (double)pb[c]);
            }

            // Output running_mean and running_var
            const T* pmean = (const T*)input_mean->data;
            const T* pvar = (const T*)input_var->data;
            if (outputs.size() > 1 && outputs[1] && outputs[1]->ndata > 0) {
                T* p = (T*)outputs[1]->data;
                for (int c = 0; c < C; ++c)
                    p[c] = (T)((double)pmean[c] * momentum + batch_mean[c] * (1.0 - momentum));
            }
            if (outputs.size() > 2 && outputs[2] && outputs[2]->ndata > 0) {
                T* p = (T*)outputs[2]->data;
                for (int c = 0; c < C; ++c)
                    p[c] = (T)((double)pvar[c] * momentum + batch_var[c] * (1.0 - momentum));
            }
        } else {
            const T* pmean = (const T*)input_mean->data;
            const T* pvar = (const T*)input_var->data;
            // Precompute per-channel alpha and beta: y = alpha * x + beta
            // where alpha = scale / sqrt(var + eps), beta = bias - alpha * mean
            arena_scope_t scope(ctx->arena);
            float* ch_alpha = scope.alloc_arr<float>(C);
            float* ch_beta  = scope.alloc_arr<float>(C);
            for (int c = 0; c < C; ++c) {
                float a = (float)(pscale[c] / sqrt((double)pvar[c] + epsilon));
                ch_alpha[c] = a;
                ch_beta[c]  = (float)pb[c] - a * (float)pmean[c];
            }
            if (y->format == memory_layout_t::NHWC && x->ndim == 4) {
                // NHWC: data is [N, H, W, C], channel is innermost.
                // Strided dst (Concat alias): output's spatial-major stride is
                // ldc = parent C count, not local C.
                const int spatial = N * channel; // N * H * W
                const int ldc = make_addr(y).elem_stride<T>(3);
                if constexpr (std::is_same_v<T, float>) {
#ifdef NNR_ARCH_X64
                    if (has_avx512()) {
                        for (int s = 0; s < spatial; ++s)
                            channel_affine_avx512((float*)py + (size_t)s * ldc,
                                (const float*)px + (size_t)s * C, ch_alpha, ch_beta, C);
                    } else {
                        for (int s = 0; s < spatial; ++s)
                            channel_affine_avx2((float*)py + (size_t)s * ldc,
                                (const float*)px + (size_t)s * C, ch_alpha, ch_beta, C);
                    }
#elifdef NNR_ARCH_ARM64
                    {
                        for (int s = 0; s < spatial; ++s)
                            neon::affine_channel((const float*)px + (size_t)s * C, (float*)py + (size_t)s * ldc, ch_alpha, ch_beta, C);
                    }
#else
                    {
                        for (int s = 0; s < spatial; ++s) {
                            const float* src = (const float*)px + (size_t)s * C;
                            float* dst = (float*)py + (size_t)s * ldc;
                            for (int c = 0; c < C; ++c)
                                dst[c] = ch_alpha[c] * src[c] + ch_beta[c];
                        }
                    }
#endif
                } else {
                    for (int s = 0; s < spatial; ++s) {
                        const T* src = px + (size_t)s * C;
                        T* dst = py + (size_t)s * ldc;
                        for (int c = 0; c < C; ++c)
                            dst[c] = (T)(ch_alpha[c] * (float)src[c] + ch_beta[c]);
                    }
                }
            } else {
                // Single-pass BN+Relu fusion: when post_fn is a leaf Relu,
                // fold max(0, ...) into the affine kernel. Saves one sweep
                // over dst.
                const bool fuse_relu = post_fn != nullptr
                    && fused_op != nullptr
                    && fused_op->op_type == "Relu"
                    && fused_op->post_fn == nullptr
                    && std::is_same_v<T, float>;
                nnr::for_cost(0, NC, channel, [&](int j) {
                    int o = j * channel;
                    int jc = j % C;
                    float a = ch_alpha[jc], b = ch_beta[jc];
                    const T* src = px + o;
                    T* dst = py + o;
#ifdef NNR_ARCH_X64
                    if constexpr (std::is_same_v<T, float>) {
                        if (fuse_relu) {
                            if (has_avx512())
                                affine_relu_avx512((float*)dst, (const float*)src, channel, a, b);
                            else
                                affine_relu_avx2  ((float*)dst, (const float*)src, channel, a, b);
                        } else {
                            if (has_avx512())
                                affine_avx512((float*)dst, (const float*)src, channel, a, b);
                            else
                                affine_avx2  ((float*)dst, (const float*)src, channel, a, b);
                        }
                    } else
#elifdef NNR_ARCH_ARM64
                    if constexpr (std::is_same_v<T, float>) {
                        neon::affine_broadcast(src, dst, channel, a, b);
                        if (fuse_relu) {
                            float* fdst = (float*)dst;
                            for (int i = 0; i < channel; ++i)
                                if (fdst[i] < 0.0f) fdst[i] = 0.0f;
                        }
                    } else
#endif
                    for (int i = 0; i < channel; ++i) {
                        float v = a * (float)src[i] + b;
                        if (fuse_relu && v < 0.0f) v = 0.0f;
                        dst[i] = (T)v;
                    }
                });
                if (post_fn && !fuse_relu) {
                    post_fn((float*)py, NC, channel, channel,
                            fused_op, nullptr, 0);
                }
            }
        }
        return true;
    }

    scroll_info_t scroll_info() const override {
        // Only inference mode is scrollable (training computes global stats)
        if (training_mode) return {};
        if (inputs[0]->ndim < 3) return {};
        return { .scrollable = true };
    }

    bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) override
    {
        if (training_mode) return false;
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        if (x->ndim < 3) return false;
        const float* px = (const float*)x->data;
        float* py = (float*)y->data;
        const float* pscale = (const float*)inputs[1]->data;
        const float* pb = (const float*)inputs[2]->data;
        const float* pmean = (const float*)inputs[3]->data;
        const float* pvar = (const float*)inputs[4]->data;
        int C = x->dims[1];
        int iH = x->dims[x->ndim - 2];
        int W = x->dims[x->ndim - 1];
        int oH = y->dims[y->ndim - 2];
        int N = x->dims[0];
        int clamp_H = ring_out.orig_H > 0 ? ring_out.orig_H : oH;
        int out_end = std::min(out_row_start + out_rows, clamp_H);
        int row_count = out_end - out_row_start;
        if (row_count <= 0) return true;

        int NC = N * C;
        int elem_count = row_count * W;
        auto bn_strip = [](const float* src, float* dst, int n, float a, float b) {
#ifdef NNR_ARCH_X64
            if (has_avx512()) affine_avx512(dst, src, n, a, b);
            else              affine_avx2  (dst, src, n, a, b);
#elifdef NNR_ARCH_ARM64
            neon::affine_broadcast(src, dst, n, a, b);
#else
            for (int i = 0; i < n; ++i)
                dst[i] = a * src[i] + b;
#endif
        };
        nnr::for_cost(0, NC, elem_count, [&](int nc) {
            int c = nc % C;
            float a = pscale[c] / sqrtf(pvar[c] + epsilon);
            float b = pb[c] - a * pmean[c];
            const float* src = px + (size_t)nc * iH * W + (size_t)out_row_start * W;
            float* dst = py + (size_t)nc * oH * W + (size_t)out_row_start * W;
            bn_strip(src, dst, elem_count, a, b);
        });
        return true;
    }

    // Fused DQ→BN→Q path: uint8 input, precomputed scale=A[c], bias=B[c],
    // y = clamp(round(x * A[c] + B[c]), 0, 255)
    bool exec_uint8() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const float* A = (const float*)inputs[1]->data;  // combined scale
        const float* B = (const float*)inputs[2]->data;  // combined offset
        const uint8_t* px = (const uint8_t*)x->data;
        uint8_t* py = (uint8_t*)y->data;
        int N = x->dims[0];
        int C = x->dims[1];
        int HW = 1;
        for (int d = 2; d < x->ndim; d++) HW *= x->dims[d];

        // NHWC: channel is innermost. For each of N*H*W pixels, apply
        // per-channel A[c]/B[c] to a contiguous C-vector.
        if (x->format == memory_layout_t::NHWC && x->ndim == 4) {
            int NHW = N * HW;
            nnr::for_cost(0, NHW, C, [&](int p) {
                const uint8_t* src = px + (size_t)p * (size_t)C;
                uint8_t* dst = py + (size_t)p * (size_t)C;
#ifdef NNR_ARCH_X64
                if (has_avx512()) channel_affine_u8_avx512(dst, src, A, B, C);
                else              channel_affine_u8_avx2  (dst, src, A, B, C);
#else
                for (int c = 0; c < C; c++) {
                    float v = A[c] * (float)src[c] + B[c];
                    v = std::max(0.0f, std::min(255.0f, v));
                    dst[c] = (uint8_t)std::lrintf(v);
                }
#endif
            });
            y->format = memory_layout_t::NHWC;
            return true;
        }

        // NCHW: for each (n,c), process HW elements with same A[c]/B[c]
        int NC = N * C;
        nnr::for_cost(0, NC, HW, [&](int nc) {
            int c = nc % C;
            float a = A[c], b = B[c];
            const uint8_t* src = px + (size_t)nc * HW;
            uint8_t* dst = py + (size_t)nc * HW;
            int i = 0;
#ifdef NNR_ARCH_X64
            if (has_avx512()) {
                __m512 va = _mm512_set1_ps(a);
                __m512 vb = _mm512_set1_ps(b);
                __m512 v0 = _mm512_setzero_ps();
                __m512 v255 = _mm512_set1_ps(255.0f);
                for (; i + 16 <= HW; i += 16) {
                    __m128i xu8 = _mm_loadu_si128((const __m128i*)(src + i));
                    __m512i xi32 = _mm512_cvtepu8_epi32(xu8);
                    __m512 xf = _mm512_cvtepi32_ps(xi32);
                    xf = _mm512_fmadd_ps(xf, va, vb);
                    xf = _mm512_max_ps(_mm512_min_ps(xf, v255), v0);
                    __m512i yi32 = _mm512_cvtps_epi32(xf);
                    _mm_storeu_si128((__m128i*)(dst + i),
                        _mm512_cvtusepi32_epi8(yi32));
                }
            } else {
                __m256 va = _mm256_set1_ps(a);
                __m256 vb = _mm256_set1_ps(b);
                __m256 v0 = _mm256_setzero_ps();
                __m256 v255 = _mm256_set1_ps(255.0f);
                for (; i + 8 <= HW; i += 8) {
                    __m128i xu8 = _mm_loadl_epi64((const __m128i*)(src + i));
                    __m256i xi32 = _mm256_cvtepu8_epi32(xu8);
                    __m256 xf = _mm256_cvtepi32_ps(xi32);
                    xf = _mm256_fmadd_ps(xf, va, vb);
                    xf = _mm256_max_ps(_mm256_min_ps(xf, v255), v0);
                    __m256i yi32 = _mm256_cvtps_epi32(xf);
                    __m128i lo = _mm256_castsi256_si128(yi32);
                    __m128i hi = _mm256_extracti128_si256(yi32, 1);
                    __m128i u16 = _mm_packus_epi32(lo, hi);
                    __m128i u8  = _mm_packus_epi16(u16, u16);
                    _mm_storel_epi64((__m128i*)(dst + i), u8);
                }
            }
#endif
            for (; i < HW; i++) {
                float v = (float)src[i] * a + b;
                v = std::max(0.0f, std::min(255.0f, v));
                dst[i] = (uint8_t)std::roundf(v);
            }
        });
        return true;
    }

    // Fused BN→Q path: fp32 input → uint8 output
    // scale[c] = gamma/(sqrt(var+eps)*q_scale), bias[c] = (beta-gamma*mean/sqrt(var+eps))/q_scale + q_zp
    bool exec_fp32_to_uint8() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const float* A = (const float*)inputs[1]->data;
        const float* B = (const float*)inputs[2]->data;
        const float* px = (const float*)x->data;
        uint8_t* py = (uint8_t*)y->data;
        int N = x->dims[0];
        int C = x->dims[1];
        int HW = 1;
        for (int d = 2; d < x->ndim; d++) HW *= x->dims[d];

        // NHWC: channel is innermost. Per-pixel apply A[c]/B[c] with saturation.
        if (x->format == memory_layout_t::NHWC && x->ndim == 4) {
            int NHW = N * HW;
            nnr::for_cost(0, NHW, C, [&](int p) {
                const float* src = px + (size_t)p * (size_t)C;
                uint8_t* dst = py + (size_t)p * (size_t)C;
#ifdef NNR_ARCH_X64
                if (has_avx512()) channel_affine_f32_u8_avx512(dst, src, A, B, C);
                else              channel_affine_f32_u8_avx2  (dst, src, A, B, C);
#else
                for (int c = 0; c < C; c++) {
                    float v = src[c] * A[c] + B[c];
                    v = std::max(0.0f, std::min(255.0f, v));
                    dst[c] = (uint8_t)std::lrintf(v);
                }
#endif
            });
            y->format = memory_layout_t::NHWC;
            return true;
        }

        int NC = N * C;
        nnr::for_cost(0, NC, HW, [&](int nc) {
            int c = nc % C;
            float a = A[c], b = B[c];
            const float* src = px + (size_t)nc * HW;
            uint8_t* dst = py + (size_t)nc * HW;
            int i = 0;
#ifdef NNR_ARCH_X64
            if (has_avx512()) {
                __m512 va = _mm512_set1_ps(a);
                __m512 vb = _mm512_set1_ps(b);
                __m512 v0 = _mm512_setzero_ps();
                __m512 v255 = _mm512_set1_ps(255.0f);
                for (; i + 16 <= HW; i += 16) {
                    __m512 xf = _mm512_loadu_ps(src + i);
                    xf = _mm512_fmadd_ps(xf, va, vb);
                    xf = _mm512_max_ps(_mm512_min_ps(xf, v255), v0);
                    __m512i yi32 = _mm512_cvtps_epi32(xf);
                    _mm_storeu_si128((__m128i*)(dst + i),
                        _mm512_cvtusepi32_epi8(yi32));
                }
            } else {
                __m256 va = _mm256_set1_ps(a);
                __m256 vb = _mm256_set1_ps(b);
                __m256 v0 = _mm256_setzero_ps();
                __m256 v255 = _mm256_set1_ps(255.0f);
                for (; i + 8 <= HW; i += 8) {
                    __m256 xf = _mm256_loadu_ps(src + i);
                    xf = _mm256_fmadd_ps(xf, va, vb);
                    xf = _mm256_max_ps(_mm256_min_ps(xf, v255), v0);
                    __m256i yi32 = _mm256_cvtps_epi32(xf);
                    __m128i lo = _mm256_castsi256_si128(yi32);
                    __m128i hi = _mm256_extracti128_si256(yi32, 1);
                    __m128i u16 = _mm_packus_epi32(lo, hi);
                    __m128i u8  = _mm_packus_epi16(u16, u16);
                    _mm_storel_epi64((__m128i*)(dst + i), u8);
                }
            }
#endif
            for (; i < HW; i++) {
                float v = src[i] * a + b;
                v = std::max(0.0f, std::min(255.0f, v));
                dst[i] = (uint8_t)std::roundf(v);
            }
        });
        return true;
    }

    bool exec() override {
        if (inputs[0]->type == NNR_DATA_TYPE_UINT8)
            return exec_uint8();
        if (outputs[0]->type == NNR_DATA_TYPE_UINT8)
            return exec_fp32_to_uint8();
        return typed_exec<BatchNormalization_operator,
            opset_t<14, float16_t, float, double, bfloat16_t>,
            opset_t<7, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=cost layout=[NCHW,NHWC] scroll=yes
operator_t* resolver_default_op_BatchNormalization(int opset, pool_t& pool) { return pool_new<BatchNormalization_operator>(pool); }

} // namespace nnr
