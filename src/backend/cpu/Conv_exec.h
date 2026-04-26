// -*- C++ -*-
// Conv_operator 4D exec() helpers and dispatcher — extracted from Conv.cpp.
//
// This file is NOT a standalone TU. It is #included inside the
// Conv_operator class body in Conv.cpp. It references class members
// directly (group, kernels, cpads, strides, dilations, w_*, ws_*,
//  layout_mask, inputs, outputs, bias_*, use_last_layer, etc.) and
// relies on all the kernel headers included from Conv.cpp being in
// scope.
//
// Contains:
//   struct conv4d_vars_t<T>        — per-call locals bundle
//   init_conv4d_vars<T>            — populate locals from inputs/outputs
//   exec_depthwise_maybe<T>        — NHWC/NCHW depthwise + NCHWc DW
//   exec_nchwc_blocked<T>          — NCHWc (BLOCKED_16) general path
//   exec_nhwc_1x1<T>               — NHWC 1x1 GEMM
//   exec_last_layer<T>             — small-OC last-layer direct
//   exec_nhwc_general<T>           — NHWC general conv (fused im2col + GEMM)
//   exec_1x1_nchw<T>               — NCHW 1x1 GEMM
//   exec_winograd_2d<T>            — Winograd F(4x4, 3x3)
//   exec_first_layer<T>            — small-IC first-layer direct
//   exec_im2col_gemm<T>            — NCHW generic im2col + GEMM
//   exec_nd_generic<T>             — 3D/5D+ fallback
//   exec<T>                        — 4D/ND dispatcher
//
// Hot path: every exec<T> call lands here.

    // 4D exec helpers ---------------------------------------------------
    // Bundle of per-call locals so each branch helper can be written with
    // the original identifier names. Populated once by init_conv4d_vars()
    // at the top of exec(), then passed by const reference to each helper.
    //
    // Conv_operator is NOT a template struct, but exec() and all these
    // helpers are member function templates parameterized on element type
    // T (float / float16_t / bfloat16_t), so conv4d_vars_t is also a
    // template to hold T-typed data pointers.
    template <typename T>
    struct conv4d_vars_t {
        tensor_t* y;
        const tensor_t* x;
        const tensor_t* w;
        T* bias;
        int M;
        int kC;
        int kH;
        int kW;
        int kHW;
        int CHW;
        int iC;
        int iH;
        int iW;
        int oN;
        int oH;
        int oW;
        int MM;
        int CC;
        int spatial;
        T* xd;
        T* yd;
        T* wd;
    };

    template <typename T>
    void init_conv4d_vars(conv4d_vars_t<T>& v) {
        v.y = outputs[0];
        v.x = inputs[0];
        v.w = inputs[1];
        v.bias = (inputs.size() > 2) ? (T*)inputs[2]->data : nullptr;
        v.M = v.w->dims[0]; v.kC = v.w->dims[1];
        v.kH = v.w->dims[2]; v.kW = v.w->dims[3];
        v.kHW = v.kH * v.kW; v.CHW = v.kC * v.kHW;
        v.iC = v.x->dims[1]; v.iH = v.x->dims[2]; v.iW = v.x->dims[3];
        v.oN = v.y->dims[0]; v.oH = v.y->dims[2]; v.oW = v.y->dims[3];
        v.MM = v.M / group; v.CC = v.iC / group;
        v.spatial = v.oH * v.oW;
        v.xd = (T*)v.x->data;
        v.yd = (T*)v.y->data;
        v.wd = (T*)v.w->data;
    }

    template <typename T>
    bool exec_depthwise_maybe(const conv4d_vars_t<T>& v) {
        auto& y = v.y;
        auto& x = v.x;
        auto& w = v.w;
        auto& bias = v.bias;
        auto& M = v.M;
        auto& kC = v.kC;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& kHW = v.kHW;
        auto& CHW = v.CHW;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& spatial = v.spatial;
        auto& xd = v.xd;
        auto& yd = v.yd;
        auto& wd = v.wd;
        (void)y;(void)x;(void)w;(void)bias;

            if (group == iC && kC == 1) {
                // NCHWc depthwise path — uses the native blocked layout
                // (BLOCKED_16 on x64, BLOCKED_8 on ARM once M2 lands).
                if constexpr (std::is_same_v<T, float>) {
                    if (y->format == NATIVE_BLOCKED_FMT
                        && (layout_mask & LAYOUT_NATIVE_BLOCKED)) {
                        // Lazy-pack NCHWc DW weights on first entry
                        // Weights: [C, 1, kH, kW] → [Cb, kH, kW, NATIVE_BLOCK]
                        if (w_dw_nchwc.empty()) {
                            constexpr int block = NATIVE_BLOCK;
                            const int C = w->dims[0];
                            const int Cb = C / block;
                            w_dw_nchwc.resize((size_t)Cb * kH * kW * block);
                            const float* src = (const float*)w->data;
                            for (int cb = 0; cb < Cb; cb++)
                                for (int kh_ = 0; kh_ < kH; kh_++)
                                    for (int kw_ = 0; kw_ < kW; kw_++)
                                        for (int c = 0; c < block; c++)
                                            w_dw_nchwc[((size_t)cb * kH * kW + kh_ * kW + kw_) * block + c]
                                                = src[(cb * block + c) * kH * kW + kh_ * kW + kw_];
                            bias_dw_nchwc.resize(nchwc_padded_channels(C, block));
                            pack_bias_nchwc(bias_dw_nchwc.data(), bias, C, block);
                        }
                        return exec_depthwise_2d_nchwc(y, x,
                            w_dw_nchwc.data(), bias_dw_nchwc.data());
                    }
                }
                if constexpr (std::is_same_v<T, float>) {
                    if (!w_dw_nhwc.empty() && y->format == memory_layout_t::NHWC)
                        return exec_depthwise_2d_nhwc(y, x, w_dw_nhwc.data(), bias);
                }
                // Skip NCHW depthwise if NHWC general path should handle it
                // (false depthwise: group==iC && kC==1 but M != iC)
                if constexpr (std::is_same_v<T, float>) {
                    if (!w_gemm_nhwc.empty() && y->format == memory_layout_t::NHWC) {
                        // Fall through to NHWC general path below
                    } else {
                        return exec_depthwise_2d<T>(y, x, w, bias);
                    }
                } else {
                    return exec_depthwise_2d<T>(y, x, w, bias);
                }
            }

        return false; // no branch matched
    }

    template <typename T>
    bool exec_nchwc_blocked(const conv4d_vars_t<T>& v) {
        auto& y = v.y;
        auto& x = v.x;
        auto& w = v.w;
        auto& bias = v.bias;
        auto& M = v.M;
        auto& kC = v.kC;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& kHW = v.kHW;
        auto& CHW = v.CHW;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& spatial = v.spatial;
        auto& xd = v.xd;
        auto& yd = v.yd;
        auto& wd = v.wd;
        (void)y;(void)x;(void)w;(void)bias;

            // NCHWc Conv path: used when layout pass assigns the native blocked
            // format (BLOCKED_16 on x64, BLOCKED_8 on ARM).
            // Handles both 1x1 (pointwise) and general (KxK) convolutions.
            // Weights are lazy-packed on first entry to avoid memory overhead for
            // Convs that the eligibility gate rejects.
#ifdef NNR_ARCH_X64
            if constexpr (std::is_same_v<T, float>) {
                // Enter NCHWc path when either:
                //   (a) output tensor is blocked (chain interior / aligned OC), or
                //   (b) input tensor is blocked AND Conv advertises the layout —
                //       "terminal blocked consumer" that writes NCHW output.
                //       Used for OC-tail pred-heads in SSD-12 (OC ∈ {24, 324, 486})
                //       so the big backbone-exit reorder can be skipped.
                const bool y_is_blocked = (y->format == NATIVE_BLOCKED_FMT);
                const bool x_is_blocked = (x->format == NATIVE_BLOCKED_FMT);
                if ((y_is_blocked || x_is_blocked)
                    && (layout_mask & LAYOUT_NATIVE_BLOCKED)) {
                    constexpr int block = NATIVE_BLOCK;
                    const bool is1x1_exec = (kH == 1 && kW == 1
                        && strides.size() >= 2 && strides[0] == 1 && strides[1] == 1
                        && cpads[0] == 0 && cpads[1] == 0);

                    // Winograd F(4,3) NCHWc eligibility. The kernel is inherently
                    // s=1/d=1/g=1 and requires iC, M multiples of `block`, AND
                    // writes BLOCKED_16 output directly — so skip in the terminal
                    // case (y is NCHW) to avoid output-format mismatch.
                    const bool is_wino_exec = !is1x1_exec
                        && y_is_blocked
                        && kH == 3 && kW == 3
                        && kC == iC
                        && strides.size() >= 2 && strides[0] == 1 && strides[1] == 1
                        && (dilations.empty() || (dilations[0] == 1 && dilations[1] == 1))
                        && (iC % block == 0) && (M % block == 0)
                        && ((oH + 3) / 4) * ((oW + 3) / 4) >= WINOGRAD_MIN_TILES;

                    // Lazy-pack NCHWc weights on first entry
                    if (is_wino_exec) {
                        if (w_wino_nchwc.empty()) {
                            const int OCb = M / block;
                            w_wino_nchwc.resize((size_t)36 * OCb * iC * block);
                            nnr::winograd_nchwc_weight_transform(
                                w_wino_nchwc.data(), (const float*)w->data, M, iC);
                        }
                        if (bias_nchwc.empty()) {
                            bias_nchwc.resize(nchwc_padded_channels(M, block));
                            const float* bias_src = (inputs.size() > 2) ? (const float*)inputs[2]->data : nullptr;
                            pack_bias_nchwc(bias_nchwc.data(), bias_src, M, block);
                        }
                    } else if (w_nchwc.empty()) {
                        const int OC = w->dims[0];
                        const int OCb = (OC + block - 1) / block;
                        const int ICb = (iC + block - 1) / block;
                        if (is1x1_exec) {
                            w_nchwc.resize((size_t)OCb * iC * block);
                            pack_weight_nchwc_1x1(w_nchwc.data(), (const float*)w->data, OC, iC, block);
                        } else {
                            // IC-blocked layout: [OCb, ICb, KH, KW, 16ic, 16oc]
                            w_nchwc.resize((size_t)OCb * ICb * kH * kW * block * block);
                            pack_weight_nchwc_blocked(w_nchwc.data(), (const float*)w->data, OC, iC, kH, kW, block);
                        }
                        bias_nchwc.resize(nchwc_padded_channels(OC, block));
                        const float* bias_src = (inputs.size() > 2) ? (const float*)inputs[2]->data : nullptr;
                        pack_bias_nchwc(bias_nchwc.data(), bias_src, OC, block);
                    }

                    // Winograd NCHWc path: handles pads internally (no pre-pad).
                    // Input is reordered (boundary) or used in place (interior);
                    // V/M scratch lives at the tail of ctx->workspace. post_fn
                    // applied after the output transform (same hook as the
                    // direct-conv path below).
                    if (is_wino_exec) {
                        const int ICb = iC / block;
                        float* in_nchwc_w;
                        size_t in_bytes = 0;
                        if (x->format == NATIVE_BLOCKED_FMT) {
                            in_nchwc_w = (float*)x->data;
                        } else {
                            in_nchwc_w = (float*)ctx->workspace;
                            nchw_to_nchwc(in_nchwc_w, (const float*)x->data, oN, iC, iH, iW, block);
                            in_bytes = (size_t)oN * ICb * iH * iW * block * sizeof(float);
                        }
                        void* wino_ws = (uint8_t*)ctx->workspace + in_bytes;
                        float* out_nchwc_w = (float*)y->data;

                        // One-shot activation trace (feedback_path_activation.md).
                        static bool s_wino_nchwc_traced = false;
                        if (!s_wino_nchwc_traced) {
                            s_wino_nchwc_traced = true;
                            fprintf(stderr, "[wino_nchwc] active (iC=%d M=%d %dx%d->%dx%d)\n",
                                iC, M, iH, iW, oH, oW);
                        }
                        nnr::winograd_conv2d_nchwc_avx512(
                            out_nchwc_w, in_nchwc_w,
                            w_wino_nchwc.data(), bias_nchwc.data(),
                            oN, iC, iH, iW, M, oH, oW,
                            cpads[0], cpads[1],
                            wino_ws);

                        if (post_fn) {
                            const int OCb = (M + block - 1) / block;
                            const int total = oN * OCb * oH * oW * block;
                            post_fn(out_nchwc_w, 1, total, total, fused_op, nullptr, 0);
                        }
                        y->format = y->declared_layout;
                        return true;
                    }

                    // Pre-pad input to eliminate bounds-checking branches.
                    // The padded input lets the kernel run with pad=0 — all output
                    // positions use the branch-free safe path. Critical for small
                    // spatial sizes (7×7: ~27-49% of pixels are edge pixels).
                    const int nk = (int)kernels.size();
                    const int padT = cpads[0], padL = cpads[1];
                    const int padB = cpads[nk], padR = cpads[nk + 1];
                    const bool needs_pad = !is1x1_exec
                        && (padT > 0 || padL > 0 || padB > 0 || padR > 0);
                    float* in_nchwc;
                    int eff_iH = iH, eff_iW = iW;

                    if (needs_pad) {
                        const int pH = iH + padT + padB;
                        const int pW = iW + padL + padR;
                        in_nchwc = (float*)ctx->workspace;
                        if (x->format == NATIVE_BLOCKED_FMT) {
                            // Chain interior: copy NCHWc to padded workspace
                            nchwc_pad(in_nchwc, (const float*)x->data,
                                oN, iC, iH, iW, padT, padL, padB, padR, block);
                        } else {
                            // Chain boundary: reorder NCHW to padded NCHWc
                            nchw_to_nchwc_padded(in_nchwc, (const float*)x->data,
                                oN, iC, iH, iW, padT, padL, padB, padR, block);
                        }
                        eff_iH = pH;
                        eff_iW = pW;
                    } else {
                        if (x->format == NATIVE_BLOCKED_FMT) {
                            in_nchwc = (float*)x->data;
                        } else {
                            in_nchwc = (float*)ctx->workspace;
                            nchw_to_nchwc(in_nchwc, (const float*)x->data, oN, iC, iH, iW, block);
                        }
                    }

                    // Output buffer selection:
                    //   y_is_blocked: kernel writes directly to y->data
                    //     (y is sized for OCb*block — safe when OC%block==0).
                    //   terminal:     kernel writes padded BLOCKED_16 to the
                    //     tail of ctx->workspace, then nchwc_to_nchw copies
                    //     into y->data (NCHW, sized for logical OC).
                    const int OCb = (M + block - 1) / block;
                    float* out_nchwc;
                    if (y_is_blocked) {
                        out_nchwc = (float*)y->data;
                    } else {
                        // Offset past the input-padded region (if present).
                        size_t in_off = 0;
                        if (needs_pad) {
                            const int ICb = (iC + block - 1) / block;
                            in_off = (size_t)oN * ICb * eff_iH * eff_iW * block * sizeof(float);
                        } else if (!x_is_blocked) {
                            const int ICb = (iC + block - 1) / block;
                            in_off = (size_t)oN * ICb * iH * iW * block * sizeof(float);
                        }
                        out_nchwc = (float*)((uint8_t*)ctx->workspace + in_off);
                    }

                    if (is1x1_exec) {
                        conv1x1_nchwc_avx512(out_nchwc, in_nchwc,
                            w_nchwc.data(), bias_nchwc.data(),
                            oN, iC, M, oH, oW);
                    } else {
                        conv_nchwc_avx512(out_nchwc, in_nchwc,
                            w_nchwc.data(), bias_nchwc.data(),
                            oN, iC, M,
                            eff_iH, eff_iW, oH, oW, kH, kW,
                            strides[0], strides[1],
                            needs_pad ? 0 : cpads[0],
                            needs_pad ? 0 : cpads[1]);
                    }

                    // Apply fused post-op (Relu, Clip, Add, etc.) on BLOCKED_16 output.
                    // Element-wise ops work correctly on flat BLOCKED_16 data because
                    // all tensors share the same element ordering at the same spatial dims.
                    if (post_fn) {
                        const int total = oN * OCb * oH * oW * block;
                        post_fn(out_nchwc, 1, total, total, fused_op, nullptr, 0);
                    }

                    if (!y_is_blocked) {
                        // Terminal Conv: transpose blocked-padded output → NCHW.
                        nchwc_to_nchw((float*)y->data, out_nchwc, oN, M, oH, oW, block);
                    }
                    y->format = y->declared_layout;
                    return true;
                }
            }
#elif defined(NNR_ARCH_ARM64)
            // ARM NCHW8c path — M1 only lands 1×1 Conv.
            // General K×K and depthwise come in M2/M3 of the NCHWc plan.
            if constexpr (std::is_same_v<T, float>) {
                if (y->format == NATIVE_BLOCKED_FMT
                    && (layout_mask & LAYOUT_NATIVE_BLOCKED)) {
                    constexpr int block = NATIVE_BLOCK;  // 8 on ARM
                    const bool is1x1_exec = (kH == 1 && kW == 1
                        && strides.size() >= 2 && strides[0] == 1 && strides[1] == 1
                        && cpads[0] == 0 && cpads[1] == 0);

                    // M1 only supports 1×1. Fall through to NHWC/NCHW on non-1×1.
                    if (!is1x1_exec) return false;

                    // Lazy-pack NCHW8c 1×1 weights on first entry.
                    if (w_nchwc.empty()) {
                        const int OC = w->dims[0];
                        const int OCb = (OC + block - 1) / block;
                        w_nchwc.resize((size_t)OCb * iC * block);
                        pack_weight_nchwc_1x1(w_nchwc.data(), (const float*)w->data, OC, iC, block);
                        bias_nchwc.resize(nchwc_padded_channels(OC, block));
                        const float* bias_src = (inputs.size() > 2) ? (const float*)inputs[2]->data : nullptr;
                        pack_bias_nchwc(bias_nchwc.data(), bias_src, OC, block);
                    }

                    // Input reorder: NCHW -> NCHW8c at chain boundary, or direct
                    // pointer in chain interior when upstream already produced
                    // BLOCKED_8.
                    float* in_nchwc;
                    if (x->format == NATIVE_BLOCKED_FMT) {
                        in_nchwc = (float*)x->data;
                    } else {
                        in_nchwc = (float*)ctx->workspace;
                        nchw_to_nchwc(in_nchwc, (const float*)x->data, oN, iC, iH, iW, block);
                    }

                    float* out_nchwc = (float*)y->data;

                    conv1x1_nchwc_neon(out_nchwc, in_nchwc,
                        w_nchwc.data(), bias_nchwc.data(),
                        oN, iC, M, oH, oW);

                    // Apply fused post-op on BLOCKED_8 flat output.
                    if (post_fn) {
                        const int OCb = (M + block - 1) / block;
                        const int total = oN * OCb * oH * oW * block;
                        post_fn(out_nchwc, 1, total, total, fused_op, nullptr, 0);
                    }

                    y->format = y->declared_layout;
                    return true;
                }
            }
#endif

        return false; // no branch matched
    }

    template <typename T>
    bool exec_nhwc_1x1(const conv4d_vars_t<T>& v) {
        auto& y = v.y;
        auto& x = v.x;
        auto& w = v.w;
        auto& bias = v.bias;
        auto& M = v.M;
        auto& kC = v.kC;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& kHW = v.kHW;
        auto& CHW = v.CHW;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& spatial = v.spatial;
        auto& xd = v.xd;
        auto& yd = v.yd;
        auto& wd = v.wd;
        (void)y;(void)x;(void)w;(void)bias;

            // NHWC 1×1 Conv path: used when layout pass assigns NHWC format
            if constexpr (std::is_same_v<T, float>) {
                if (!w_nhwc.empty() && y->format == memory_layout_t::NHWC) {
                    // Strided dst (NHWC channel-axis Concat alias): output's
                    // per-spatial-position C stride equals the Concat parent's
                    // full C count, not this Conv's local M. The AVX-512 path
                    // (`dgemm_packed_supports_ldc()`) honors ldc directly via
                    // packed-B; AVX-2 / NEON fall back to dgemm_generic_ldc.
                    const auto Y = make_addr(y);
                    const int ldc = Y.elem_stride<float>(3);
                    const bool strided = (ldc != M);
                    const bool simd_ldc = dgemm_packed_supports_ldc();
                    // Lazy pack: only pack when NHWC path is actually entered.
                    // When strided AND no SIMD LDC: skip pack (we'll use
                    // dgemm_generic_ldc with unpacked B).
                    if ((!strided || simd_ldc) && w_packed.empty() && M >= 64) {
                        w_packed.resize(pack_b_size(iC, M));
                        pack_b(w_packed.data(), w_nhwc.data(), iC, M);
                    }
                    for (int n = 0; n < oN; ++n) {
                        float* Y_n = Y.at<float>(n, 0, 0, 0);
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
                        int y_off = (int)((size_t)n * (size_t)spatial * ldc);
                        gemm_post_nhwc_t nhwc_post;
                        nhwc_post.bias = (const float*)bias;
                        nhwc_post.c_base = Y_n;
                        nhwc_post.c_base_offset = y_off;
                        nhwc_post.post_fn = post_fn;
                        nhwc_post.fused_op = fused_op;
                        nhwc_post.classify();
                        if (!w_packed.empty() && (!strided || simd_ldc)) {
                            dgemm_nhwc(spatial, M, iC, X_n, w_packed.data(), Y_n,
                                       nhwc_post, strided ? ldc : 0);
                        } else {
                            dgemm_generic_ldc(spatial, M, iC, X_n, w_nhwc.data(),
                                              Y_n, ldc, nhwc_post);
                        }
                    }
                    y->format = y->declared_layout;
                    return true;
                }
            }

        return false; // no branch matched
    }

    template <typename T>
    bool exec_last_layer(const conv4d_vars_t<T>& v) {
        auto& y = v.y;
        auto& x = v.x;
        auto& w = v.w;
        auto& bias = v.bias;
        auto& M = v.M;
        auto& kC = v.kC;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& kHW = v.kHW;
        auto& CHW = v.CHW;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& spatial = v.spatial;
        auto& xd = v.xd;
        auto& yd = v.yd;
        auto& wd = v.wd;
        (void)y;(void)x;(void)w;(void)bias;

            // Last-layer direct conv: small-OC Conv (e.g., RGB output).
            // Vectorizes over output width instead of output channels.
            // Intercepts before NHWC/NCHW GEMM paths for maximum benefit.
#if defined(NNR_ARCH_X64) || defined(NNR_ARCH_ARM64)
            if constexpr (std::is_same_v<T, float>) {
                if (use_last_layer && w_winograd.empty()) {
                    const int nk = (int)kernels.size();
                    const int padT = cpads[0], padL = cpads[1];
                    const int padB = cpads[nk], padR = cpads[nk + 1];
#ifdef NNR_ARCH_X64
                    const int ppW_extra = 15;
#else
                    const int ppW_extra = 3;
#endif
                    const int ppH = iH + padT + padB;
                    const int ppW = iW + padL + padR + ppW_extra;
                    float* ws = (float*)ctx->workspace;

                    for (int n = 0; n < oN; ++n) {
                        // Pre-pad input into workspace (handles NHWC or NCHW)
                        if (x->format == memory_layout_t::NHWC) {
                            conv_last_layer_prepad_nhwc(ws,
                                xd + (size_t)n * iC * iH * iW,
                                iC, iH, iW, ppH, ppW, padT, padL);
                        } else {
                            conv_last_layer_prepad(ws,
                                xd + (size_t)n * iC * iH * iW,
                                iC, iH, iW, ppH, ppW, padT, padL);
                        }
#ifdef NNR_ARCH_X64
                        conv_last_layer_avx512(
#else
                        conv_last_layer_neon(
#endif
                            yd + (size_t)n * M * spatial,
                            wd, (const float*)bias,
                            iC, M, oH, oW, kH, kW,
                            ppH, ppW, ws,
                            post_fn, fused_op);
                    }
                    return true;
                }
            }
#endif

        return false; // no branch matched
    }

    template <typename T>
    bool exec_nhwc_general(const conv4d_vars_t<T>& v) {
        auto& y = v.y;
        auto& x = v.x;
        auto& w = v.w;
        auto& bias = v.bias;
        auto& M = v.M;
        auto& kC = v.kC;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& kHW = v.kHW;
        auto& CHW = v.CHW;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& spatial = v.spatial;
        auto& xd = v.xd;
        auto& yd = v.yd;
        auto& wd = v.wd;
        (void)y;(void)x;(void)w;(void)bias;

            // NHWC general Conv path: im2col (NHWC order) + GEMM
            if constexpr (std::is_same_v<T, float>) {
                if (!w_gemm_nhwc.empty() && y->format == memory_layout_t::NHWC) {
                    const int sH = strides[0], sW = strides[1];
                    const int dH = dilations[0], dW = dilations[1];
                    const int pH = cpads[0], pW = cpads[1];
                    const int K = kC * kHW;  // im2col column width
                    const int wino_tiles_nhwc = ((oH + 3) / 4) * ((oW + 3) / 4);
#ifdef NNR_ARCH_X64
                    if (!w_winograd.empty() && group == 1 && wino_tiles_nhwc >= 16) {
                        float* ws = (float*)ctx->workspace;
                        bool input_nhwc = (x->format == memory_layout_t::NHWC);
                        const float* wpb = w_winograd_packed_nhwc.empty() ? nullptr : w_winograd_packed_nhwc.data();
                        winograd_conv2d_nhwc(yd, xd, w_winograd_nhwc.data(), wpb, bias,
                            oN, iC, iH, iW, M, oH, oW, pH, pW, ws, input_nhwc, wino_group, post_fn, fused_op);
                        y->format = y->declared_layout;
                        return true;
                    }
#endif
                    // Strided dst (NHWC channel-axis Concat alias): output's
                    // per-spatial-position C stride equals the parent's full C
                    // count, not this Conv's local M. AVX-512 path
                    // (`dgemm_packed_supports_ldc()`) honors ldc directly via
                    // packed-B; AVX-2 / NEON fall back to dgemm_generic_ldc.
                    const auto Y = make_addr(y);
                    const int ldc = (group == 1) ? Y.elem_stride<float>(3) : M;
                    const bool strided = (ldc != M);
                    const bool simd_ldc = dgemm_packed_supports_ldc();
                    // Lazy pack: only pack when NHWC path is actually entered.
                    // When strided AND no SIMD LDC support: skip pack since the
                    // dgemm_generic_ldc fallback uses the unpacked weights.
                    if ((!strided || simd_ldc) && w_packed.empty() && MM >= 64) {
                        size_t psz = pack_b_size(K, MM);
                        w_packed.resize(psz * group);
                        for (int g = 0; g < group; g++)
                            pack_b(w_packed.data() + g * psz,
                                w_gemm_nhwc.data() + (size_t)g * K * MM, K, MM);
                    }
#ifdef NNR_USE_XBYAK_AARCH64
                    // JIT 6×16 packed weights: pack on first entry (group==1 only).
                    // Skip on strided dst — JIT path assumes contiguous output.
                    if (!strided && w_packed_jit.empty() && group == 1 && MM >= 16) {
                        w_packed_jit.resize(neon_jit::pack_b_jit_size(K, MM));
                        neon_jit::pack_b_jit(w_packed_jit.data(),
                            w_gemm_nhwc.data(), K, MM);
                    }
#endif
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
                            // Tiled NHWC: NHWC output is [spatial × ldc], contiguous in spatial
                            float* yn = Y.at<float>(n, 0, 0, 0);
                            for (int oh0 = 0; oh0 < oH; oh0 += nhwc_tile_h) {
                                int th = std::min(nhwc_tile_h, oH - oh0);
                                int tile_sp = th * oW;
                                im2col_nhwc_tiled(col, xn_nhwc, iC, kC, iH, iW, kH, kW, oW,
                                    sH, sW, pH, pW, dH, dW, 0, oh0, th);
                                float* yn_tile = Y.at<float>(n, 0, oh0, 0);
                                gemm_post_nhwc_t nhwc_post;
                                nhwc_post.bias = (const float*)bias;
                                nhwc_post.c_base = yn_tile;
                                nhwc_post.c_base_offset = (int)((size_t)n * spatial * ldc + (size_t)oh0 * oW * ldc);
                                nhwc_post.post_fn = post_fn;
                                nhwc_post.fused_op = fused_op;
                                nhwc_post.classify();
#ifdef NNR_USE_XBYAK_AARCH64
                                // JIT 6×16 path: profitable for large GEMMs where
                                // the bigger tile outweighs unfused post-op overhead.
                                if (!strided && !w_packed_jit.empty()
                                    && (int64_t)tile_sp * MM * K > (1 << 22)) {
                                    neon_jit::dgemm_jit(tile_sp, MM, K,
                                        col, K, w_packed_jit.data(),
                                        yn_tile, MM);
                                    // Apply post-op (bias + activation) per row
                                    for (int s = 0; s < tile_sp; s++)
                                        nhwc_post.apply(0, yn_tile + s * MM, MM);
                                } else
#endif
                                if (!w_packed.empty() && (!strided || simd_ldc))
                                    dgemm_nhwc(tile_sp, MM, K, col, w_packed.data(),
                                               yn_tile, nhwc_post, strided ? ldc : 0);
                                else
                                    dgemm_generic_ldc(tile_sp, MM, K, col,
                                                      w_gemm_nhwc.data(), yn_tile, ldc, nhwc_post);
                            }
                        } else {
                            // Non-tiled path (small buffer or groups > 1)
                            for (int g = 0; g < group; ++g) {
                                im2col_nhwc(col, xn_nhwc, iC, kC, iH, iW, kH, kW, oH, oW,
                                    sH, sW, pH, pW, dH, dW, g * kC);

                                // GEMM: col[spatial x K] x W[K x MM] -> out[spatial x MM]
                                if (group == 1) {
                                    float* yn = Y.at<float>(n, 0, 0, 0);
                                    gemm_post_nhwc_t nhwc_post;
                                    nhwc_post.bias = (const float*)bias;
                                    nhwc_post.c_base = yn;
                                    nhwc_post.c_base_offset = (int)((size_t)n * spatial * ldc);
                                    nhwc_post.post_fn = post_fn;
                                    nhwc_post.fused_op = fused_op;
                                    nhwc_post.classify();
                                    if (!w_packed.empty() && (!strided || simd_ldc)) {
                                        dgemm_nhwc(spatial, MM, K, col, w_packed.data(),
                                                   yn, nhwc_post, strided ? ldc : 0);
                                    } else {
                                        dgemm_generic_ldc(spatial, MM, K, col,
                                                          w_gemm_nhwc.data(), yn, ldc, nhwc_post);
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

                    y->format = y->declared_layout;
                    return true;
                }
            }

        return false; // no branch matched
    }

    template <typename T>
    bool exec_1x1_nchw(const conv4d_vars_t<T>& v) {
        auto& y = v.y;
        auto& x = v.x;
        auto& w = v.w;
        auto& bias = v.bias;
        auto& M = v.M;
        auto& kC = v.kC;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& kHW = v.kHW;
        auto& CHW = v.CHW;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& spatial = v.spatial;
        auto& xd = v.xd;
        auto& yd = v.yd;
        auto& wd = v.wd;
        (void)y;(void)x;(void)w;(void)bias;

            // 1×1 conv shortcut: direct GEMM, no im2col (NCHW path)
            if (kH == 1 && kW == 1 && strides.size() >= 2
                && strides[0] == 1 && strides[1] == 1
                && dilations[0] == 1 && dilations[1] == 1
                && cpads[0] == 0 && cpads[1] == 0
                && cpads[kernels.size()] == 0
                && cpads[kernels.size() + 1] == 0) {
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

        return false; // no branch matched
    }

    template <typename T>
    bool exec_winograd_2d(const conv4d_vars_t<T>& v) {
        auto& y = v.y;
        auto& x = v.x;
        auto& w = v.w;
        auto& bias = v.bias;
        auto& M = v.M;
        auto& kC = v.kC;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& kHW = v.kHW;
        auto& CHW = v.CHW;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& spatial = v.spatial;
        auto& xd = v.xd;
        auto& yd = v.yd;
        auto& wd = v.wd;
        (void)y;(void)x;(void)w;(void)bias;

            // Winograd F(4×4, 3×3) path with batched GEMM kernel.
            // Only profitable when spatial size is large enough for transform overhead
            // to be amortized: need enough tiles (oH/4 * oW/4 >= 16).
            if constexpr (std::is_same_v<T, float>) {
                const int wino_tiles = ((oH + 3) / 4) * ((oW + 3) / 4);
                if (!w_winograd.empty() && wino_tiles >= WINOGRAD_MIN_TILES) {
                    const int pH = cpads[0], pW = cpads[1];
                    float* ws = (float*)ctx->workspace;
                    const float* wpk = w_winograd_packed.empty() ? nullptr : w_winograd_packed.data();
                    winograd_conv2d(yd, xd, w_winograd.data(), wpk, bias,
                        oN, iC, iH, iW, M, oH, oW, pH, pW, ws, wino_group, post_fn, fused_op);
                    return true;
                }
            }

        return false; // no branch matched
    }

    template <typename T>
    bool exec_first_layer(const conv4d_vars_t<T>& v) {
        auto& y = v.y;
        auto& x = v.x;
        auto& w = v.w;
        auto& bias = v.bias;
        auto& M = v.M;
        auto& kC = v.kC;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& kHW = v.kHW;
        auto& CHW = v.CHW;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& spatial = v.spatial;
        auto& xd = v.xd;
        auto& yd = v.yd;
        auto& wd = v.wd;
        (void)y;(void)x;(void)w;(void)bias;

            // First-layer direct conv: skip im2col for small-IC Conv (e.g., RGB)
#if defined(NNR_ARCH_X64) || defined(NNR_ARCH_ARM64)
            if constexpr (std::is_same_v<T, float>) {
                if (!w_first_layer.empty()) {
                    const int sH = strides[0], sW = strides[1];
                    const int pH = cpads[0], pW = cpads[1];
                    for (int n = 0; n < oN; n++) {
#ifdef NNR_ARCH_X64
                        conv_first_layer_avx512(
#else
                        conv_first_layer_neon(
#endif
                            yd + (size_t)n * M * spatial,
                            xd + (size_t)n * iC * iH * iW,
                            w_first_layer.data(), (const float*)bias,
                            iC, iH, iW, M, oH, oW, kH, kW, sH, sW, pH, pW,
                            post_fn, fused_op);
                    }
                    return true;
                }
            }
#endif

        return false; // no branch matched
    }

    // Direct NCHW conv: pre-pad input, direct micro-kernel (no im2col).
    // Eligibility: float, stride=1, dilation=1, groups=1, kH*kW > 1, packed_a available.
    template <typename T>
    bool exec_direct_conv(const conv4d_vars_t<T>& v) {
        if constexpr (std::is_same_v<T, float>) {
            if (w_packed_nchw.empty()) return false;
            if (group != 1) return false;
            if (v.kH == 1 && v.kW == 1) return false;
            if (strides.size() < 2 || strides[0] != 1 || strides[1] != 1) return false;
            if (dilations[0] != 1 || dilations[1] != 1) return false;
            // Only beneficial when im2col expansion is large (kH*kW >= 9) and output
            // rows are wide enough for efficient NR=16 vectorization (oW >= 32).
            // Small oW forces per-row processing with poor weight reuse vs GEMM.
            if (v.kH * v.kW < 9) return false;
            if (v.oW < 32) return false;

            const int K = v.CHW;
            const int padT = cpads[0], padL = cpads[1];
            const int padB = cpads[kernels.size()], padR = cpads[kernels.size() + 1];
            const int padH = v.iH + padT + padB;
            const int padW = v.iW + padL + padR;

            // Use existing workspace for padded input + offset table
            const size_t pad_bytes = (size_t)v.kC * padH * padW * sizeof(float);
            float* padded = (float*)ctx->workspace;
            int* offsets = (int*)((char*)ctx->workspace + pad_bytes);

            build_conv_offsets(offsets, v.kC, v.kH, v.kW, padH, padW);

            for (int n = 0; n < v.oN; n++) {
                const float* xn = v.xd + (size_t)n * v.iC * v.iH * v.iW;
                float* yn = v.yd + (size_t)n * v.M * v.spatial;

                prepad_nchw(padded, xn, v.kC, v.iH, v.iW, padT, padB, padL, padR);

                int yn_off = (int)((size_t)n * v.M * v.spatial);
                gemm_post_t post(v.bias, 0, yn, yn_off, (operator_t*)this);
                if (!conv_direct(v.MM, v.oH, v.oW, K, w_packed_nchw.data(), padded,
                        offsets, padW, 1, yn, post)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    template <typename T>
    bool exec_im2col_gemm(const conv4d_vars_t<T>& v) {
        auto& y = v.y;
        auto& x = v.x;
        auto& w = v.w;
        auto& bias = v.bias;
        auto& M = v.M;
        auto& kC = v.kC;
        auto& kH = v.kH;
        auto& kW = v.kW;
        auto& kHW = v.kHW;
        auto& CHW = v.CHW;
        auto& iC = v.iC;
        auto& iH = v.iH;
        auto& iW = v.iW;
        auto& oN = v.oN;
        auto& oH = v.oH;
        auto& oW = v.oW;
        auto& MM = v.MM;
        auto& CC = v.CC;
        auto& spatial = v.spatial;
        auto& xd = v.xd;
        auto& yd = v.yd;
        auto& wd = v.wd;
        (void)y;(void)x;(void)w;(void)bias;

            // im2col [CHW × spatial] + GEMM: y = W × col
            T* col = (T*)ctx->workspace;
            const int sH = strides[0], sW = strides[1];
            const int dH = dilations[0], dW = dilations[1];
            const int pH = cpads[0], pW = cpads[1];

            // Spatially-tiled im2col: when the full im2col buffer exceeds the
            // cache threshold, process tile_h output rows at a time. GEMM writes
            // to a temp buffer which is scattered to the strided NCHW output.
            const int tile_h = im2col_tile_h();
            if constexpr (std::is_same_v<T, float>) {
                if (tile_h < oH) {
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
            }

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

        return false; // no branch matched
    }

    template <typename T>
    bool exec_nd_generic() {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        T* bias = (inputs.size() > 2) ? (T*)inputs[2]->data : nullptr;
        const int ndim = x->ndim;
        const int M = w->dims[0], kC = w->dims[1];
        (void)M; (void)kC; (void)w; (void)y; (void)bias;
        // Generic N-D fallback
        const T* px = (const T*)x->data;
        T* py = (T*)y->data;
        const T* pw = (const T*)w->data;
        small_vector<int> i_dim(ndim), o_dim(ndim);
        small_vector<int> w_dim(ndim), b_dim(ndim);
        do {
            b_dim[0] = o_dim[0];
            for (int i = 2; i < ndim; ++i)
                b_dim[i] = o_dim[i] * strides[i - 2] - cpads[i - 2];
            T sum = 0;
            std::fill(w_dim.begin(), w_dim.end(), 0);
            w_dim[0] = o_dim[1];
            do {
                if (w_dim[1] == 1) break;
                i_dim[0] = b_dim[0];
                for (int i = 2; i < ndim; ++i)
                    i_dim[i] = b_dim[i] + w_dim[i] * dilations[i - 2];
                for (int ch = 0; ch < kC; ++ch) {
                    i_dim[1] = (o_dim[1] * group / M) * kC + ch;
                    w_dim[1] = ch;
                    T v = 0;
                    int i;
                    for (i = 0; i < ndim; ++i)
                        if (i_dim[i] < 0 || i_dim[i] >= x->dims[i]) break;
                    if (i >= ndim)
                        v = px[dim_offset(i_dim, x->dim_span())];
                    T wt = 0;
                    for (i = 0; i < ndim; ++i)
                        if (w_dim[i] < 0 || w_dim[i] >= w->dims[i]) break;
                    if (i >= ndim)
                        wt = pw[dim_offset(w_dim, w->dim_span())];
                    sum += v * wt;
                }
                w_dim[1] = 0;
            } while (dim_next(w_dim, w->dim_span()));
            if (!post_fn && bias) sum += bias[o_dim[1]];
            py[dim_offset(o_dim, y->dim_span())] = sum;
        } while (dim_next(o_dim, y->dim_span()));
        // Apply fused post-op (handles bias + activation in one pass)
        if constexpr (std::is_same_v<T, float>) {
            if (post_fn) {
                // Apply per-batch: post_fn processes C channels × spatial per batch
                int outer = y->dims[0];
                int C = y->dims[1];
                int spatial = (int)(y->ndata / (outer * C));
                for (int n = 0; n < outer; ++n) {
                    int off = (int)(n * C * spatial);
                    post_fn((float*)py + off, C, spatial, spatial, fused_op,
                            bias ? (const float*)bias : nullptr, off);
                }
            }
        }
        return true;

        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const int ndim = x->ndim;

        if (ndim == 4) {
            conv4d_vars_t<T> v;
            init_conv4d_vars<T>(v);

            if (v.iC == group && v.kC == 1) {
                if (exec_depthwise_maybe<T>(v)) return true;
                // false-depthwise fall-through to NHWC general path below
            }
            // First-layer direct conv runs ahead of NHWC/NCHWc paths:
            // small-IC stems (RGB inputs) waste lanes in general GEMM/blocked
            // layouts because C_in is padded up (e.g. C=3 → 16 in NCHWc).
            // conv_first_layer_avx512 is the specialized path for this shape.
            if (exec_first_layer<T>(v))   return true;
            if (exec_nchwc_blocked<T>(v)) return true;
            if (exec_nhwc_1x1<T>(v))      return true;
            if (exec_last_layer<T>(v))    return true;
            if (exec_nhwc_general<T>(v))  return true;
            if (exec_1x1_nchw<T>(v))      return true;
            if (exec_winograd_2d<T>(v))   return true;
            if (exec_direct_conv<T>(v))   return true;
            return exec_im2col_gemm<T>(v);
        }

        return exec_nd_generic<T>();
    }
