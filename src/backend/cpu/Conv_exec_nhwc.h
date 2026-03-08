#pragma once
// NHWC convolution execution paths, included from Conv.cpp.
// Requires: Conv_operator members (w_nhwc, w_gemm_nhwc, w_packed, strides, etc.)

// NHWC 1×1 Conv: transposed weights, optional packed B
bool exec_conv_nhwc_1x1(float* bias) {
    tensor_t* y = outputs[0];
    const tensor_t* x = inputs[0];
    const int M = inputs[1]->dims[0];
    const int iC = x->dims[1];
    const int oN = y->dims[0], oH = y->dims[2], oW = y->dims[3];
    const int spatial = oH * oW;
    float* xd = (float*)x->data;
    float* yd = (float*)y->data;

    // Lazy pack: only pack when NHWC path is actually entered
    if (w_packed.empty() && M >= 64) {
        w_packed.resize(pack_b_size(iC, M));
        pack_b(w_packed.data(), w_nhwc.data(), iC, M);
    }
    for (int n = 0; n < oN; ++n) {
        float* Y_n = yd + (size_t)n * M * spatial;
        const float* X_n;
        if (x->format == memory_layout_t::NHWC) {
            X_n = xd + (size_t)n * spatial * iC;
        } else {
            float* ws = (float*)ctx->workspace;
            const float* X_nchw = xd + (size_t)n * iC * spatial;
            // SIMD block-transpose [iC × spatial] → [spatial × iC]
            nchw_to_nhwc(ws, X_nchw, 1, iC, 1, spatial);
            X_n = ws;
        }
        int y_off = (int)((size_t)n * M * spatial);
        if (!w_packed.empty()) {
            gemm_post_nhwc_t nhwc_post;
            nhwc_post.bias = (const float*)bias;
            nhwc_post.c_base = Y_n;
            nhwc_post.c_base_offset = y_off;
            nhwc_post.post_fn = post_fn;
            nhwc_post.fused_op = fused_op;
            nhwc_post.classify();
            dgemm_nhwc(spatial, M, iC, X_n, w_packed.data(), Y_n, nhwc_post);
        } else {
            gemm_post_nhwc_t nhwc_post2;
            nhwc_post2.bias = (const float*)bias;
            nhwc_post2.c_base = Y_n;
            nhwc_post2.c_base_offset = y_off;
            nhwc_post2.post_fn = post_fn;
            nhwc_post2.fused_op = fused_op;
            nhwc_post2.classify();
            dgemm_generic(spatial, M, iC, X_n, w_nhwc.data(), Y_n, nhwc_post2);
        }
    }
    y->format = memory_layout_t::NHWC;
    return true;
}

// NHWC general Conv: im2col (NHWC order) + GEMM, with optional Winograd
bool exec_conv_nhwc_general(float* bias) {
    tensor_t* y = outputs[0];
    const tensor_t* x = inputs[0];
    const tensor_t* w = inputs[1];
    const int M = w->dims[0], kC = w->dims[1];
    const int kH = w->dims[2], kW = w->dims[3];
    const int kHW = kH * kW;
    const int iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oH = y->dims[2], oW = y->dims[3];
    const int MM = M / group;
    const int spatial = oH * oW;
    const int sH = strides[0], sW = strides[1];
    const int dH = dilations[0], dW = dilations[1];
    const int pH = cpads[0], pW = cpads[1];
    const int K = kC * kHW;  // im2col column width
    float* xd = (float*)x->data;
    float* yd = (float*)y->data;

    // Winograd NHWC path
    const int wino_tiles_nhwc = ((oH + 3) / 4) * ((oW + 3) / 4);
    if (!w_winograd.empty() && group == 1 && wino_tiles_nhwc >= 16) {
        float* ws = (float*)ctx->workspace;
        bool input_nhwc = (x->format == memory_layout_t::NHWC);
        const float* wpb = w_winograd_packed_nhwc.empty() ? nullptr : w_winograd_packed_nhwc.data();
        winograd_conv2d_nhwc(yd, xd, w_winograd_nhwc.data(), wpb, bias,
            oN, iC, iH, iW, M, oH, oW, pH, pW, ws, input_nhwc, wino_group, post_fn, fused_op);
        y->format = memory_layout_t::NHWC;
        return true;
    }

    // Lazy pack: only pack when NHWC path is actually entered
    if (w_packed.empty() && MM >= 64) {
        size_t psz = pack_b_size(K, MM);
        w_packed.resize(psz * group);
        for (int g = 0; g < group; g++)
            pack_b(w_packed.data() + g * psz,
                w_gemm_nhwc.data() + (size_t)g * K * MM, K, MM);
    }
    const int nhwc_tile_h = im2col_tile_h();
    float* ws = (float*)ctx->workspace;
    int tile_spatial = nhwc_tile_h * oW;
    // workspace layout: [im2col buffer] [scatter buffer if groups>1] [reorder buffer]
    float* col = ws;
    float* tmp = (group > 1) ? col + (size_t)tile_spatial * K : nullptr;
    float* reorder_buf = (group > 1) ? tmp + (size_t)tile_spatial * MM
        : col + (size_t)tile_spatial * K;

    for (int n = 0; n < oN; ++n) {
        // At NCHW boundary: reorder once per batch, then use efficient im2col_nhwc
        const float* xn_nhwc;
        if (x->format == memory_layout_t::NHWC) {
            xn_nhwc = xd + (size_t)n * iC * iH * iW;
        } else {
            nchw_to_nhwc(reorder_buf, xd + (size_t)n * iC * iH * iW, 1, iC, iH, iW);
            xn_nhwc = reorder_buf;
        }

        if (group == 1 && nhwc_tile_h < oH) {
            // Tiled NHWC: NHWC output is [spatial × M], contiguous in spatial
            float* yn = yd + (size_t)n * spatial * M;
            for (int oh0 = 0; oh0 < oH; oh0 += nhwc_tile_h) {
                int th = std::min(nhwc_tile_h, oH - oh0);
                int tile_sp = th * oW;
                im2col_nhwc_tiled(col, xn_nhwc, iC, kC, iH, iW, kH, kW, oW,
                    sH, sW, pH, pW, dH, dW, 0, oh0, th);
                float* yn_tile = yn + (size_t)oh0 * oW * M;
                gemm_post_nhwc_t nhwc_post;
                nhwc_post.bias = (const float*)bias;
                nhwc_post.c_base = yn_tile;
                nhwc_post.c_base_offset = (int)((size_t)n * spatial * M + oh0 * oW * M);
                nhwc_post.post_fn = post_fn;
                nhwc_post.fused_op = fused_op;
                nhwc_post.classify();
                if (!w_packed.empty())
                    dgemm_nhwc(tile_sp, MM, K, col, w_packed.data(), yn_tile, nhwc_post);
                else
                    dgemm_generic(tile_sp, MM, K, col, w_gemm_nhwc.data(), yn_tile, nhwc_post);
            }
        } else {
            // Non-tiled path (small buffer or groups > 1)
            for (int g = 0; g < group; ++g) {
                im2col_nhwc(col, xn_nhwc, iC, kC, iH, iW, kH, kW, oH, oW,
                    sH, sW, pH, pW, dH, dW, g * kC);

                // GEMM: col[spatial x K] x W[K x MM] -> out[spatial x MM]
                if (group == 1) {
                    float* yn = yd + (size_t)n * spatial * M;
                    if (!w_packed.empty()) {
                        gemm_post_nhwc_t nhwc_post;
                        nhwc_post.bias = (const float*)bias;
                        nhwc_post.c_base = yn;
                        nhwc_post.c_base_offset = (int)((size_t)n * spatial * M);
                        nhwc_post.post_fn = post_fn;
                        nhwc_post.fused_op = fused_op;
                        nhwc_post.classify();
                        dgemm_nhwc(spatial, MM, K, col, w_packed.data(), yn, nhwc_post);
                    } else {
                        gemm_post_nhwc_t nhwc_post;
                        nhwc_post.bias = (const float*)bias;
                        nhwc_post.c_base = yn;
                        nhwc_post.c_base_offset = (int)((size_t)n * spatial * M);
                        nhwc_post.post_fn = post_fn;
                        nhwc_post.fused_op = fused_op;
                        nhwc_post.classify();
                        dgemm_generic(spatial, MM, K, col, w_gemm_nhwc.data(), yn, nhwc_post);
                    }
                } else {
                    // Groups > 1: GEMM into temp, then scatter to interleaved output
                    size_t psz = pack_b_size(K, MM);
                    if (!w_packed.empty())
                        dgemm_nhwc(spatial, MM, K, col, w_packed.data() + g * psz, tmp);
                    else
                        dgemm_generic(spatial, MM, K, col,
                            w_gemm_nhwc.data() + (size_t)g * K * MM, tmp);
                    float* yn = yd + (size_t)n * spatial * M;
                    for (int s = 0; s < spatial; ++s)
                        memcpy(yn + s * M + g * MM, tmp + s * MM, MM * sizeof(float));
                }
            }

            // Bias already fused for group==1 (packed and unpacked paths)
            if (group != 1 && bias)
                nhwc_bias_add(yd + (size_t)n * spatial * M, (const float*)bias, spatial, M);
        }
    }

    // Post-op: skip if already fused into GEMM for group==1
    if (group != 1 && post_fn)
        post_fn(yd, 1, (int)((size_t)oN * spatial * M), (int)((size_t)oN * spatial * M), fused_op, nullptr, 0);

    y->format = memory_layout_t::NHWC;
    return true;
}
