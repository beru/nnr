#pragma once
// NCHW convolution execution paths, included from Conv.cpp.
// Requires: Conv_operator members (w_packed_nchw, w_winograd, w_first_layer, strides, etc.)

// NCHW 1×1 Conv: direct GEMM, no im2col
template <typename T>
bool exec_conv_nchw_1x1(T* bias) {
    tensor_t* y = outputs[0];
    const tensor_t* x = inputs[0];
    const tensor_t* w = inputs[1];
    const int M = w->dims[0], kC = w->dims[1];
    const int iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oH = y->dims[2], oW = y->dims[3];
    const int MM = M / group, CC = iC / group;
    const int spatial = oH * oW;
    const int CHW = kC;  // 1×1: CHW = kC * 1 * 1
    T* xd = (T*)x->data, *yd = (T*)y->data, *wd = (T*)w->data;

    for (int ng = 0; ng < oN * group; ++ng) {
        int n = ng / group, g = ng % group;
        T* yn1x1 = yd + ((size_t)n * M + g * MM) * spatial;
        int yn1x1_off = (int)((n * M + g * MM) * spatial);
        if constexpr (std::is_same_v<T, float>) {
            gemm_post_t post((const float*)bias, g * MM, (const float*)yn1x1, yn1x1_off, this);
            if (!w_packed_nchw.empty() && spatial >= 16) {
                dgemm_packed_a(MM, spatial, CC,
                    w_packed_nchw.data() + (size_t)g * pack_a_size(MM, CC),
                    xd + ((size_t)n * iC + g * CC) * iH * iW,
                    yn1x1, post);
            } else {
                dgemm_generic(MM, spatial, CC,
                    wd + (size_t)g * MM * CC,
                    xd + ((size_t)n * iC + g * CC) * iH * iW,
                    yn1x1, post);
            }
        } else {
            dgemm_generic(MM, spatial, CC,
                wd + (size_t)g * MM * CC,
                xd + ((size_t)n * iC + g * CC) * iH * iW,
                yn1x1);
            if (bias) {
                for (int m = 0; m < MM; ++m) {
                    T bv = bias[g * MM + m];
                    T* row = yd + ((size_t)n * M + g * MM + m) * spatial;
                    for (int j = 0; j < spatial; ++j)
                        row[j] += bv;
                }
            }
        }
    }
    return true;
}

// NCHW Winograd F(4×4, 3×3) path
bool exec_conv_winograd(float* bias) {
    tensor_t* y = outputs[0];
    const tensor_t* x = inputs[0];
    const int iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
    const int M = inputs[1]->dims[0];
    const int oN = y->dims[0], oH = y->dims[2], oW = y->dims[3];
    const int pH = cpads[0], pW = cpads[1];
    float* ws = (float*)ctx->workspace;
    const float* wpk = w_winograd_packed.empty() ? nullptr : w_winograd_packed.data();
    winograd_conv2d((float*)y->data, (float*)x->data, w_winograd.data(), wpk, bias,
        oN, iC, iH, iW, M, oH, oW, pH, pW, ws, wino_group, post_fn, fused_op);
    return true;
}

// First-layer direct conv: skip im2col for small-IC Conv (e.g., RGB)
bool exec_conv_first_layer(float* bias) {
    tensor_t* y = outputs[0];
    const tensor_t* x = inputs[0];
    const tensor_t* w = inputs[1];
    const int M = w->dims[0], kH = w->dims[2], kW = w->dims[3];
    const int iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oH = y->dims[2], oW = y->dims[3];
    const int spatial = oH * oW;
    const int sH = strides[0], sW = strides[1];
    const int pH = cpads[0], pW = cpads[1];
    const float* xd = (const float*)x->data;
    float* yd = (float*)y->data;

    for (int n = 0; n < oN; n++) {
        conv_first_layer_avx512(
            yd + (size_t)n * M * spatial,
            xd + (size_t)n * iC * iH * iW,
            w_first_layer.data(), (const float*)bias,
            iC, iH, iW, M, oH, oW, kH, kW, sH, sW, pH, pW,
            post_fn, fused_op);
    }
    return true;
}

// NCHW im2col + GEMM (tiled path)
template <typename T>
bool exec_conv_im2col_tiled(T* bias) {
    tensor_t* y = outputs[0];
    const tensor_t* x = inputs[0];
    const tensor_t* w = inputs[1];
    const int M = w->dims[0], kC = w->dims[1];
    const int kH = w->dims[2], kW = w->dims[3];
    const int kHW = kH * kW, CHW = kC * kHW;
    const int iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oH = y->dims[2], oW = y->dims[3];
    const int MM = M / group, CC = iC / group;
    const int spatial = oH * oW;
    const int sH = strides[0], sW = strides[1];
    const int dH = dilations[0], dW = dilations[1];
    const int pH = cpads[0], pW = cpads[1];
    float* xd = (float*)x->data, *yd = (float*)y->data, *wd = (float*)w->data;

    const int tile_h = im2col_tile_h();
    float* col = (float*)ctx->workspace;
    float* tmp = col + (size_t)CHW * tile_h * oW;
    for (int n = 0; n < oN; ++n) {
        for (int g = 0; g < group; ++g) {
            const float* xn = xd + ((size_t)n * iC + g * CC) * iH * iW;
            float* yn = yd + ((size_t)n * M + g * MM) * spatial;
            const float* wg = !w_packed_nchw.empty()
                ? w_packed_nchw.data() + (size_t)g * pack_a_size(MM, CHW)
                : wd + (size_t)g * MM * CHW;
            for (int oh0 = 0; oh0 < oH; oh0 += tile_h) {
                int th = std::min(tile_h, oH - oh0);
                int tile_sp = th * oW;
                { NNR_PROFILE_SCOPE("im2col_tiled");
                im2col_tiled(col, xn, kC, iH, iW, kH, kW, oW,
                    sH, sW, pH, pW, dH, dW, oh0, th); }
                if (!w_packed_nchw.empty())
                    dgemm_packed_a(MM, tile_sp, CHW, wg, col, tmp);
                else
                    dgemm_generic(MM, tile_sp, CHW, wg, col, tmp);
                // Scatter: tmp[m * tile_sp] → yn[m * spatial + oh0*oW]
                { NNR_PROFILE_SCOPE("scatter");
                for (int m = 0; m < MM; ++m)
                    memcpy(yn + (size_t)m * spatial + oh0 * oW,
                        tmp + (size_t)m * tile_sp,
                        tile_sp * sizeof(float)); }
            }
            // Apply bias + post-op after all tiles
            if (bias || post_fn) {
                int yn_off = (int)((n * M + g * MM) * spatial);
                if (post_fn) {
                    post_fn(yn, MM, spatial, spatial, fused_op,
                            bias ? (const float*)bias + g * MM : nullptr, yn_off);
                } else {
                    for (int m = 0; m < MM; ++m) {
                        float bv = bias ? bias[g * MM + m] : 0.0f;
                        if (bv != 0.0f) {
                            float* row = yn + (size_t)m * spatial;
                            for (int j = 0; j < spatial; ++j)
                                row[j] += bv;
                        }
                    }
                }
            }
        }
    }
    return true;
}

// NCHW im2col + GEMM (full, non-tiled path)
template <typename T>
bool exec_conv_im2col(T* bias) {
    tensor_t* y = outputs[0];
    const tensor_t* x = inputs[0];
    const tensor_t* w = inputs[1];
    const int M = w->dims[0], kC = w->dims[1];
    const int kH = w->dims[2], kW = w->dims[3];
    const int kHW = kH * kW, CHW = kC * kHW;
    const int iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oH = y->dims[2], oW = y->dims[3];
    const int MM = M / group, CC = iC / group;
    const int spatial = oH * oW;
    const int sH = strides[0], sW = strides[1];
    const int dH = dilations[0], dW = dilations[1];
    const int pH = cpads[0], pW = cpads[1];
    T* xd = (T*)x->data, *yd = (T*)y->data, *wd = (T*)w->data;
    T* col = (T*)ctx->workspace;

    for (int n = 0; n < oN; ++n) {
        for (int g = 0; g < group; ++g) {
            const T* xn = xd + ((size_t)n * iC + g * CC) * iH * iW;
            { NNR_PROFILE_SCOPE("im2col");
            im2col(col, xn, kC, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW); }

            T* yn = yd + ((size_t)n * M + g * MM) * spatial;
            int yn_off = (int)((n * M + g * MM) * spatial);
            if constexpr (std::is_same_v<T, float>) {
                gemm_post_t post2((const float*)bias, g * MM, (const float*)yn, yn_off, this);
                if (!w_packed_nchw.empty() && spatial >= 16) {
                    dgemm_packed_a(MM, spatial, CHW,
                        w_packed_nchw.data() + (size_t)g * pack_a_size(MM, CHW), col, yn, post2);
                } else {
                    dgemm_generic(MM, spatial, CHW,
                        wd + (size_t)g * MM * CHW, col, yn, post2);
                }
            } else {
                dgemm_generic(MM, spatial, CHW,
                    wd + (size_t)g * MM * CHW, col, yn);
                if (bias) {
                    for (int m = 0; m < MM; ++m) {
                        T bv = bias[g * MM + m];
                        T* row = yn + (size_t)m * spatial;
                        for (int j = 0; j < spatial; ++j)
                            row[j] += bv;
                    }
                }
            }
        }
    }
    return true;
}
