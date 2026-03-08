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

    // FP16 I/O with FP32 compute: convert input, run im2col + float GEMM, convert output.
    bool exec_f16_as_f32() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        const int ndim = x->ndim;

        if (ndim != 4)
            return false;  // Only 4D supported for FP16 fast path

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
