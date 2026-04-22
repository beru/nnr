// -*- C++ -*-
// Conv_operator exec entry points — extracted from Conv.cpp.
//
// This file is NOT a standalone TU. It is #included inside the
// Conv_operator class body in Conv.cpp. References class members
// directly and relies on Conv_exec.h (the 4D dispatcher) already
// being included earlier in the class body.
//
// Contains:
//   exec_strip()       — scrolling (ring-buffer) exec variant
//   exec_f16_as_f32()  — FP16 I/O trampoline (convert -> FP32 conv -> convert back)
//   exec_bf16_as_f32() — BF16 I/O trampoline (same structure as f16)
//   exec() override    — public virtual entry; dispatches on input dtype
//
// Hot path: the virtual exec() override is the public entry for every
// Conv call; exec_strip is called per-strip from scroll_info's returned
// function when the conv participates in a scroll chain.


    bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) override
    {
        if (inputs[0]->ndim != 4) return false;
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        float* bias = (inputs.size() > 2 && inputs[2]) ? (float*)inputs[2]->data : nullptr;
        const int M = w->dims[0], kC = w->dims[1];
        const int kH = w->dims[2], kW = w->dims[3];
        const int kHW = kH * kW, CHW = kC * kHW;
        const int iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
        const int oN = y->dims[0], oC = y->dims[1], oH = y->dims[2], oW = y->dims[3];
        const int MM = M / group, CC = iC / group;
        const int sH = strides[0], sW = strides[1];
        const int dH = dilations[0], dW = dilations[1];
        const int pH = cpads[0], pW = cpads[1];
        const float* xd = (const float*)x->data;
        float* yd = (float*)y->data;
        const float* wd = (const float*)w->data;

        // Ring buffer: dims[2] may be ring_H (for stride), use orig_H for bounds
        const int iH_pad = ring_in.orig_H > 0 ? ring_in.orig_H : iH;
        const int oH_clamp = ring_out.orig_H > 0 ? ring_out.orig_H : oH;

        // Clamp output rows to valid range
        int out_end = std::min(out_row_start + out_rows, oH_clamp);
        int actual_out_rows = out_end - out_row_start;
        if (actual_out_rows <= 0) return true;

        int spatial_strip = actual_out_rows * oW;

        // Depthwise path: parallelized over N*C with AVX-512
        if (group == iC && kC == 1) {
            auto dw_kernel = [&](int nc) {
                int n = nc / oC, c = nc % oC;
                int ic = (int)((size_t)c * iC / oC);
                const float* xc = xd + ((size_t)n * iC + ic) * iH * iW;
                const float* wc = wd + (size_t)c * kH * kW;
                float* yc = yd + ((size_t)n * oC + c) * oH * oW + (size_t)out_row_start * oW;
                float bv = bias ? bias[c] : 0.0f;
                int toff = (int)(((size_t)n * oC + c) * oH * oW + (size_t)out_row_start * oW);
                depthwise_strip_channel(xc, wc, yc, bv, kH, kW,
                    iH_pad, iW, oW, sH, sW, dH, dW, pH, pW,
                    out_row_start, out_end, spatial_strip, toff);
            };
            nnr::for_static(0, oN * oC, oN * oC > 4, dw_kernel);
            return true;
        }

        // NCHWc Winograd strip path: BLOCKED_16 layout, register-tiled matmul.
        // Handles both chain-interior (input already NCHWc) and chain-boundary
        // (input NCHW, converted to NCHWc into workspace — pre-sized by reshape).
#ifdef NNR_ARCH_X64
        if (!w_wino_nchwc.empty() && y->format == NATIVE_BLOCKED_FMT && group == 1) {
            constexpr int block = NATIVE_BLOCK;
            const int ICb = iC / block;

            const float* in_nchwc;
            size_t in_bytes = 0;
            if (x->format == NATIVE_BLOCKED_FMT) {
                in_nchwc = xd;
            } else {
                float* ws_in = (float*)ctx->workspace;
                nchw_to_nchwc(ws_in, xd, oN, iC, iH, iW, block);
                in_nchwc = ws_in;
                in_bytes = (size_t)oN * ICb * iH * iW * block * sizeof(float);
            }
            void* wino_ws = (uint8_t*)ctx->workspace + in_bytes;

            const int OCb = M / block;
            for (int n = 0; n < oN; ++n) {
                const float* xn = in_nchwc + (size_t)n * ICb * iH * iW * block;
                float* yn = yd + (size_t)n * OCb * oH * oW * block;
                winograd_conv2d_nchwc_avx512(
                    yn, xn, w_wino_nchwc.data(), bias_nchwc.data(),
                    1, iC, iH, iW, M, oH, oW, pH, pW, wino_ws,
                    out_row_start, out_end, oH_clamp);
            }
            // Note: post_fn (fused Relu/Clip) is not applied here.
            // The fused ops run as separate chain members in scroll mode
            // because fuse_post_ops marks them as non-skippable when
            // exec_strip is the active path.
            return true;
        }
#endif

        // NCHW Winograd strip path: restrict to tile rows covering the output strip
        {
            const int wino_tiles = ((oH_clamp + 3) / 4) * ((oW + 3) / 4);
            if (!w_winograd.empty() && group == 1 && wino_tiles >= WINOGRAD_MIN_TILES) {
                float* ws = (float*)ctx->workspace;
                const float* wpk = w_winograd_packed.empty() ? nullptr : w_winograd_packed.data();
                for (int n = 0; n < oN; ++n) {
                    const float* xn = xd + (size_t)n * iC * iH * iW;
                    float* yn = yd + (size_t)n * M * oH * oW;
                    winograd_conv2d(yn, xn, w_winograd.data(), wpk, bias,
                        1, iC, iH, iW, M, oH, oW, pH, pW, ws, wino_group,
                        post_fn, fused_op,
                        out_row_start, out_end, iH_pad, oH_clamp);
                }
                return true;
            }
        }

        // Non-depthwise paths (1x1 and im2col+GEMM): sequential over N*groups
        for (int n = 0; n < oN; ++n) {
            for (int g = 0; g < group; ++g) {
                const float* xn = xd + ((size_t)n * iC + g * CC) * iH * iW;
                float* yn = yd + ((size_t)n * M + g * MM) * oH * oW + (size_t)out_row_start * oW;
                int yn_off = (int)(((size_t)n * M + g * MM) * oH * oW + (size_t)out_row_start * oW);

                // 1x1 conv shortcut
                if (kH == 1 && kW == 1 && sH == 1 && sW == 1
                    && dH == 1 && dW == 1
                    && cpads[0] == 0 && cpads[1] == 0
                    && cpads[kernels.size()] == 0 && cpads[kernels.size() + 1] == 0) {
                    const float* xstrip = xn + (size_t)out_row_start * oW;
                    dgemm_generic(MM, spatial_strip, CC,
                        wd + (size_t)g * MM * CC,
                        xstrip,
                        yn,
                        gemm_post_t((const float*)bias, g * MM, yn, yn_off, this));
                    continue;
                }

                // im2col for the strip + GEMM
                float* col = (float*)ctx->workspace;
                for (int c = 0; c < kC; ++c) {
                    const float* xc = xn + (size_t)c * iH * iW;
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int k = (c * kH + kh) * kW + kw;
                            float* dst = col + (size_t)k * spatial_strip;
                            for (int oh = out_row_start; oh < out_end; ++oh) {
                                int ih = oh * sH - pH + kh * dH;
                                float* drow = dst + (oh - out_row_start) * oW;
                                if (ih < 0 || ih >= iH_pad) {
                                    memset(drow, 0, oW * sizeof(float));
                                    continue;
                                }
                                const float* srow = xc + ih * iW;
                                int iw_base = kw * dW - pW;
                                if (sW == 1) {
                                    int w0 = std::max(0, -iw_base);
                                    int w1 = std::min(oW, iW - iw_base);
                                    if (w0 > 0) memset(drow, 0, w0 * sizeof(float));
                                    if (w1 > w0) memcpy(drow + w0, srow + iw_base + w0, (w1 - w0) * sizeof(float));
                                    if (w1 < oW) memset(drow + w1, 0, (oW - w1) * sizeof(float));
                                } else {
                                    for (int ow = 0; ow < oW; ++ow) {
                                        int iw = ow * sW + iw_base;
                                        drow[ow] = (iw >= 0 && iw < iW) ? srow[iw] : 0.0f;
                                    }
                                }
                            }
                        }
                    }
                }
                dgemm_generic(MM, spatial_strip, CHW,
                    wd + (size_t)g * MM * CHW, col, yn,
                    gemm_post_t((const float*)bias, g * MM, yn, yn_off, this));
            }
        }
        return true;
    }

#ifdef NNR_ARCH_ARM64
    // Native FP16 NCHW direct conv (ARM64 + has_neon_fp16). Reads FP16 input
    // and weights directly, widens into FP32 accumulator inside the kernel,
    // then adds bias / fused post-op in FP32 before narrowing to FP16 output.
    // Eligibility is decided at reshape (w_fp16_direct populated); this just
    // checks the packed vector and hands off to the kernel.  Returns false on
    // any runtime reason not to use the path so the caller can fall through
    // to the convert-to-FP32 fallback.
    bool exec_fp16_direct_neon() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        // The NCHW kernel writes NCHW-ordered bytes.  If the layout pass
        // promoted y to NHWC, returning true here would plant NCHW bytes
        // under an NHWC-labeled tensor and corrupt the next op's read.
        if (y->format == memory_layout_t::NHWC) return false;
        if (x->format == memory_layout_t::NHWC) return false;
        const int nk = (int)kernels.size();
        const int iN = x->dims[0], iC = x->dims[1];
        const int iH = x->dims[2], iW = x->dims[3];
        const int M  = w->dims[0];
        const int kH = w->dims[2], kW = w->dims[3];
        const int oH = y->dims[2], oW = y->dims[3];
        const int sH = strides[0], sW = strides[1];
        const int padh_b = cpads[0];
        const int padw_b = cpads[1];
        const int padh_e = cpads[nk];
        const int padw_e = cpads[nk + 1];
        const int pH = iH + padh_b + padh_e;
        const int pW = iW + padw_b + padw_e;
        const size_t pad_bytes = (size_t)iC * pH * pW * sizeof(uint16_t);
        const size_t y_bytes   = (size_t)iN * M * oH * oW * sizeof(float);
        const size_t need = ((pad_bytes + 63) & ~(size_t)63) + y_bytes;
        if (need > ctx->workspace_size)
            return false;

        uint8_t* ws = (uint8_t*)ctx->workspace;
        uint16_t* x_pad = (uint16_t*)ws;
        float* y_f32   = (float*)(ws + ((pad_bytes + 63) & ~(size_t)63));

        for (int n = 0; n < iN; n++) {
            std::memset(x_pad, 0, pad_bytes);
            const uint16_t* xn = (const uint16_t*)x->data
                + (size_t)n * iC * iH * iW;
            for (int c = 0; c < iC; c++) {
                const uint16_t* src = xn + (size_t)c * iH * iW;
                uint16_t* dst = x_pad + (size_t)c * pH * pW
                    + (size_t)padh_b * pW + padw_b;
                for (int h = 0; h < iH; h++)
                    std::memcpy(dst + (size_t)h * pW,
                                src + (size_t)h * iW,
                                (size_t)iW * sizeof(uint16_t));
            }
            float* yn = y_f32 + (size_t)n * M * oH * oW;
            if (!nnr::fp16::neon::conv_fp16_direct_neon(
                yn, x_pad, w_fp16_direct.data(),
                iC, pH, pW, M, oH, oW, kH, kW, sH, sW))
                return false;
        }

        // Bias + fused post-op in FP32, then narrow to FP16.
        const int spatial = oH * oW;
        if (post_fn) {
            for (int n = 0; n < iN; n++) {
                int off = (int)((size_t)n * M * spatial);
                post_fn(y_f32 + off, M, spatial, spatial, fused_op,
                        bias_f32, off);
            }
        } else if (bias_f32) {
            for (int n = 0; n < iN; n++)
                for (int c = 0; c < M; c++) {
                    float* ch = y_f32 + ((size_t)n * M + c) * spatial;
                    float bc = bias_f32[c];
                    for (int i = 0; i < spatial; i++) ch[i] += bc;
                }
        }
        convert_f32_to_f16((float16_t*)y->data, y_f32, y->ndata);
        return true;
    }

    // Native FP16 NHWC depthwise conv (ARM64 + has_neon_fp16). Uses
    // `depthwise_fp16_nhwc_neon` on repacked FP16 weights; widens to FP32
    // inside the kernel, adds FP32 bias, narrows to FP16 on store.  Returns
    // false for any shape this path can't handle (including post_fn or a
    // non-NHWC input layout) so the caller falls back.
    bool exec_fp16_dw_nhwc_neon() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        if (post_fn && fused_tensor) return false;
        if (x->format != memory_layout_t::NHWC) return false;
        if (y->format != memory_layout_t::NHWC) return false;
        const int iN = x->dims[0], iC = x->dims[1];
        const int iH = x->dims[2], iW = x->dims[3];
        const int oH = y->dims[2], oW = y->dims[3];
        const int kH = inputs[1]->dims[2], kW = inputs[1]->dims[3];
        const int sH = strides[0], sW = strides[1];
        const int dH = dilations[0], dW = dilations[1];
        const int pH_b = cpads[0], pW_b = cpads[1];

        const size_t y_bytes = (size_t)oH * oW * iC * sizeof(float);
        if (y_bytes > ctx->workspace_size)
            return false;
        float* y_f32 = (float*)ctx->workspace;

        for (int n = 0; n < iN; n++) {
            const uint16_t* xn = (const uint16_t*)x->data
                + (size_t)n * iH * iW * iC;
            // When post_fn is fused, let it handle bias — the kernel runs
            // bias-less so post_fn's bias argument adds it once.
            const float* bias_for_kernel = post_fn ? nullptr : bias_f32;
            if (!nnr::fp16::neon::depthwise_fp16_nhwc_neon(
                y_f32, xn, w_fp16_dw_nhwc.data(), bias_for_kernel,
                iC, iH, iW, oH, oW, kH, kW, sH, sW, dH, dW, pH_b, pW_b))
                return false;

            uint16_t* yn_f16 = (uint16_t*)y->data + (size_t)n * oH * oW * iC;
            const int spatial = oH * oW;
            if (post_fn) {
                int n_off = (int)((size_t)n * spatial * iC);
                for (int p = 0; p < spatial; p++) {
                    post_fn(y_f32 + (size_t)p * iC, iC, 1, 1,
                            fused_op, bias_f32, n_off + p * iC);
                }
            }
            convert_f32_to_f16((float16_t*)yn_f16, y_f32, (size_t)spatial * iC);
        }
        y->format = memory_layout_t::NHWC;
        return true;
    }

    // Native FP16 NHWC direct conv (ARM64 + has_neon_fp16). Mirrors the NCHW
    // variant but reads/writes NHWC-layout tensors.  Eligibility decided at
    // reshape (w_fp16_nhwc_direct populated).  Returns false for shapes this
    // path can't handle (including any post_fn), so the caller falls back to
    // the convert-to-FP32 path that handles NHWC via the existing FP32 Conv.
    bool exec_fp16_nhwc_direct_neon() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        // Binary post-ops (Add with an external tensor) assume NCHW indexing
        // into the external operand — NHWC would read garbage.
        if (post_fn && fused_tensor) return false;
        // This path writes NHWC-ordered bytes; require y to be tagged NHWC.
        if (y->format != memory_layout_t::NHWC) return false;
        // x may be NCHW (first Conv in a chain, graph input) — handled below
        // by a scalar NCHW→NHWC transpose into the padded workspace.
        const bool x_is_nhwc = (x->format == memory_layout_t::NHWC);
        const int nk = (int)kernels.size();
        const int iN = x->dims[0], iC = x->dims[1];
        const int iH = x->dims[2], iW = x->dims[3];
        const int M  = w->dims[0];
        const int kH = w->dims[2], kW = w->dims[3];
        const int oH = y->dims[2], oW = y->dims[3];
        const int sH = strides[0], sW = strides[1];
        const int padh_b = cpads[0];
        const int padw_b = cpads[1];
        const int padh_e = cpads[nk];
        const int padw_e = cpads[nk + 1];
        const int pH = iH + padh_b + padh_e;
        const int pW = iW + padw_b + padw_e;
        const size_t pad_bytes = (size_t)pH * pW * iC * sizeof(uint16_t);
        const size_t y_bytes   = (size_t)oH * oW * M * sizeof(float);
        const size_t need = ((pad_bytes + 63) & ~(size_t)63) + y_bytes;
        if (need > ctx->workspace_size)
            return false;

        uint8_t* ws = (uint8_t*)ctx->workspace;
        uint16_t* x_pad = (uint16_t*)ws;
        float* y_f32   = (float*)(ws + ((pad_bytes + 63) & ~(size_t)63));

        for (int n = 0; n < iN; n++) {
            std::memset(x_pad, 0, pad_bytes);
            const uint16_t* xn_data = (const uint16_t*)x->data;
            if (x_is_nhwc) {
                const uint16_t* xn = xn_data + (size_t)n * iH * iW * iC;
                // NHWC → NHWC: row-by-row memcpy into padded position.
                for (int h = 0; h < iH; h++) {
                    std::memcpy(
                        x_pad + ((size_t)(padh_b + h) * pW + padw_b) * iC,
                        xn + (size_t)h * iW * iC,
                        (size_t)iW * iC * sizeof(uint16_t));
                }
            } else {
                // NCHW → NHWC: scatter-copy, transposing the C and (H,W) dims.
                const uint16_t* xn = xn_data + (size_t)n * iC * iH * iW;
                for (int c = 0; c < iC; c++) {
                    const uint16_t* xc = xn + (size_t)c * iH * iW;
                    for (int h = 0; h < iH; h++) {
                        uint16_t* dst_row = x_pad
                            + ((size_t)(padh_b + h) * pW + padw_b) * iC + c;
                        const uint16_t* src_row = xc + (size_t)h * iW;
                        for (int ww = 0; ww < iW; ww++)
                            dst_row[(size_t)ww * iC] = src_row[ww];
                    }
                }
            }
            if (!nnr::fp16::neon::conv_fp16_direct_nhwc_neon(
                y_f32, x_pad, w_fp16_nhwc_direct.data(),
                iC, pH, pW, M, oH, oW, kH, kW, sH, sW))
                return false;

            // Post-op / bias + narrow to FP16.  NHWC layout: each output pixel
            // stores OC values contiguous along the C axis, so per-pixel calls
            // to post_fn with rows=M, cols=1, stride=1 let the fused activation
            // operate on one pixel at a time (bias[r] indexes the channel).
            uint16_t* yn_f16 = (uint16_t*)y->data + (size_t)n * oH * oW * M;
            const int spatial = oH * oW;
            if (post_fn) {
                int n_off = (int)((size_t)n * spatial * M);
                for (int p = 0; p < spatial; p++) {
                    post_fn(y_f32 + (size_t)p * M, M, 1, 1,
                            fused_op, bias_f32, n_off + p * M);
                }
            } else if (bias_f32) {
                for (int p = 0; p < spatial; p++) {
                    float* y_row = y_f32 + (size_t)p * M;
                    for (int c = 0; c < M; c++) y_row[c] += bias_f32[c];
                }
            }
            convert_f32_to_f16((float16_t*)yn_f16, y_f32, (size_t)spatial * M);
        }
        y->format = memory_layout_t::NHWC;
        return true;
    }
#endif

    // FP16 I/O with FP32 compute: convert input, run im2col + float GEMM, convert output.
    bool exec_f16_as_f32() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        const int ndim = x->ndim;

        if (ndim != 3 && ndim != 4)
            return false;  // 1D (ndim=3) and 2D (ndim=4) spatial supported

#ifdef NNR_ARCH_ARM64
        // NHWC output paths: try these first when y is tagged NHWC.  The
        // depthwise path requires x NHWC (no cross-channel reuse makes an
        // input transpose lose); the direct path handles NCHW input via
        // scalar transpose into its padded workspace. NHWC only for 2D.
        if (ndim == 4 && y->format == memory_layout_t::NHWC
            && !w_fp16_dw_nhwc.empty()
            && exec_fp16_dw_nhwc_neon())
            return true;
        if (ndim == 4 && y->format == memory_layout_t::NHWC
            && !w_fp16_nhwc_direct.empty()
            && exec_fp16_nhwc_direct_neon())
            return true;
        if (ndim == 4 && !w_fp16_direct.empty() && exec_fp16_direct_neon())
            return true;
#endif

        // 1D conv (ndim=3): treat as 2D with H=1, kH=1, sH=1, pH=0, dH=1.
        const int M = w->dims[0], kC = w->dims[1];
        const int kH = (ndim == 4) ? w->dims[2] : 1;
        const int kW = w->dims[w->ndim - 1];
        const int kHW = kH * kW, CHW = kC * kHW;
        const int iC = x->dims[1];
        const int iH = (ndim == 4) ? x->dims[2] : 1;
        const int iW = x->dims[ndim - 1];
        const int oN = y->dims[0];
        const int oH = (ndim == 4) ? y->dims[2] : 1;
        const int oW = y->dims[ndim - 1];
        const int MM = M / group, CC = iC / group;
        const int spatial = oH * oW;
        const int sH = (ndim == 4) ? strides[0] : 1;
        const int sW = strides[ndim == 4 ? 1 : 0];
        const int dH = (ndim == 4) ? dilations[0] : 1;
        const int dW = dilations[ndim == 4 ? 1 : 0];
        const int pH = (ndim == 4) ? cpads[0] : 0;
        const int pW = cpads[ndim == 4 ? 1 : 0];

        // Workspace layout: [X_f32 | Y_f32 | im2col]
        float* ws = (float*)ctx->workspace;
        float* x_f32 = ws;
        float* y_f32 = x_f32 + x->ndata;

        convert_f16_to_f32(x_f32, (const float16_t*)x->data, x->ndata);

        const float* wd = w_f32;  // Pre-converted in reshape

        // Depthwise conv: group == iC, kC == 1
        if (group == iC && kC == 1) {
            for (int nc = 0; nc < oN * M; ++nc) {
                int n = nc / M, c = nc % M;
                int ic = (int)((size_t)c * iC / M);
                const float* xc = x_f32 + ((size_t)n * iC + ic) * iH * iW;
                const float* wc = wd + (size_t)c * kH * kW;
                float* yc = y_f32 + ((size_t)n * M + c) * spatial;
                float bv = (!post_fn && bias_f32) ? bias_f32[c] : 0.0f;
                for (int oh = 0; oh < oH; ++oh) {
                    for (int ow = 0; ow < oW; ++ow) {
                        float sum = 0.0f;
                        for (int kh = 0; kh < kH; ++kh) {
                            int ih = oh * sH - pH + kh * dH;
                            if (ih < 0 || ih >= iH) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                int iw = ow * sW - pW + kw * dW;
                                if (iw < 0 || iw >= iW) continue;
                                sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                            }
                        }
                        yc[oh * oW + ow] = sum + bv;
                    }
                }
            }
            // Apply fused post-op (handles bias + activation in one pass)
            if (post_fn) {
                for (int n = 0; n < oN; ++n) {
                    int off = (int)(n * M * spatial);
                    post_fn(y_f32 + off, M, spatial, spatial, fused_op,
                            bias_f32, off);
                }
            }
            convert_f32_to_f16((float16_t*)y->data, y_f32, y->ndata);
            return true;
        }

        // Regular conv: im2col + GEMM
        float* col = y_f32 + y->ndata;

        // 1x1 conv shortcut (no im2col)
        bool is1x1 = (kH == 1 && kW == 1 && sH == 1 && sW == 1
            && dH == 1 && dW == 1
            && cpads[0] == 0 && cpads[1] == 0
            && cpads[kernels.size()] == 0 && cpads[kernels.size() + 1] == 0);

        for (int n = 0; n < oN; ++n) {
            for (int g = 0; g < group; ++g) {
                const float* xn = x_f32 + ((size_t)n * iC + g * CC) * iH * iW;
                float* yn = y_f32 + ((size_t)n * M + g * MM) * spatial;
                int yn_off = (int)((n * M + g * MM) * spatial);

                if (is1x1) {
                    dgemm_generic(MM, spatial, CC,
                        wd + (size_t)g * MM * CC, xn, yn,
                        gemm_post_t(bias_f32, g * MM, yn, yn_off, this));
                    continue;
                }

                // im2col
                im2col(col, xn, kC, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);

                dgemm_generic(MM, spatial, CHW,
                    wd + (size_t)g * MM * CHW, col, yn,
                    gemm_post_t(bias_f32, g * MM, yn, yn_off, this));
            }
        }

        convert_f32_to_f16((float16_t*)y->data, y_f32, y->ndata);
        return true;
    }

    // BF16 I/O with FP32 compute: same structure as exec_f16_as_f32 but with BF16 conversion.
    bool exec_bf16_as_f32() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        const int ndim = x->ndim;

        if (ndim != 4)
            return false;

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

        float* ws = (float*)ctx->workspace;
        float* x_f32 = ws;
        float* y_f32 = x_f32 + x->ndata;

        convert_bf16_to_f32(x_f32, (const bfloat16_t*)x->data, x->ndata);

        const float* wd = w_f32;  // Pre-converted in reshape

        // Depthwise conv
        if (group == iC && kC == 1) {
            for (int nc = 0; nc < oN * M; ++nc) {
                int n = nc / M, c = nc % M;
                int ic = (int)((size_t)c * iC / M);
                const float* xc = x_f32 + ((size_t)n * iC + ic) * iH * iW;
                const float* wc = wd + (size_t)c * kH * kW;
                float* yc = y_f32 + ((size_t)n * M + c) * spatial;
                float bv = (!post_fn && bias_f32) ? bias_f32[c] : 0.0f;
                for (int oh = 0; oh < oH; ++oh)
                    for (int ow = 0; ow < oW; ++ow) {
                        float sum = 0.0f;
                        for (int kh = 0; kh < kH; ++kh) {
                            int ih = oh * sH - pH + kh * dH;
                            if (ih < 0 || ih >= iH) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                int iw = ow * sW - pW + kw * dW;
                                if (iw < 0 || iw >= iW) continue;
                                sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                            }
                        }
                        yc[oh * oW + ow] = sum + bv;
                    }
            }
            if (post_fn) {
                for (int n = 0; n < oN; ++n) {
                    int off = (int)(n * M * spatial);
                    post_fn(y_f32 + off, M, spatial, spatial, fused_op, bias_f32, off);
                }
            }
            convert_f32_to_bf16((bfloat16_t*)y->data, y_f32, y->ndata);
            return true;
        }

        // Regular conv: im2col + GEMM
        float* col = y_f32 + y->ndata;
        bool is1x1 = (kH == 1 && kW == 1 && sH == 1 && sW == 1
            && dH == 1 && dW == 1
            && cpads[0] == 0 && cpads[1] == 0
            && cpads[kernels.size()] == 0 && cpads[kernels.size() + 1] == 0);

        for (int n = 0; n < oN; ++n) {
            for (int g = 0; g < group; ++g) {
                const float* xn = x_f32 + ((size_t)n * iC + g * CC) * iH * iW;
                float* yn = y_f32 + ((size_t)n * M + g * MM) * spatial;
                int yn_off = (int)((n * M + g * MM) * spatial);

                if (is1x1) {
                    dgemm_generic(MM, spatial, CC,
                        wd + (size_t)g * MM * CC, xn, yn,
                        gemm_post_t(bias_f32, g * MM, yn, yn_off, this));
                    continue;
                }

                im2col(col, xn, kC, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);
                dgemm_generic(MM, spatial, CHW,
                    wd + (size_t)g * MM * CHW, col, yn,
                    gemm_post_t(bias_f32, g * MM, yn, yn_off, this));
            }
        }

        convert_f32_to_bf16((bfloat16_t*)y->data, y_f32, y->ndata);
        return true;
    }

    bool exec() override {
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT16)
            return exec_f16_as_f32();
        if (inputs[0]->type == NNR_DATA_TYPE_BFLOAT16)
            return exec_bf16_as_f32();
        if (!typed_exec<Conv_operator, float, double>(this, inputs[0]->type))
            return false;
        return true;
    }
