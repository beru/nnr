// -*- C++ -*-
// Conv_operator::reshape() — extracted from Conv.cpp for readability.
//
// This file is NOT a standalone TU. It is #included inside the
// Conv_operator class body in Conv.cpp. It references class members
// (auto_pad, group, kernels, dilations, pads, strides, cpads,
//  ws_CHW, ws_spatial, ws_kHW, ws_oW, ws_nhwc_reorder, ws_nchwc_reorder,
//  w_nhwc, w_dw_nhwc, w_gemm_nhwc, w_packed, w_packed_nchw,
//  w_winograd, w_winograd_packed, w_winograd_nhwc, w_winograd_packed_nhwc,
//  ws_winograd, wino_group, w_nchwc, bias_nchwc, w_dw_nchwc, bias_dw_nchwc,
//  w_first_layer, use_last_layer, ws_last_layer, w_f32, bias_f32,
//  layout_mask, inputs, outputs) and the anonymous-namespace enum
//  auto_pad_t directly — they must all be in scope at the include site.
//
// Cold path: runs once per tensor-shape change, never per inference.

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[1];
        const int ndim = x->ndim;
        const int nk = (int)kernels.size();
        small_vector<int> dims(ndim);

        compute_conv_output_shape(
            (int)auto_pad,
            x->dims, ndim,
            w->dims[0],
            kernels.data(), nk,
            dilations.data(),
            strides.data(),
            pads.data(),
            cpads,
            dims.data());
        // Cache dimensions for workspace_size()
        ws_CHW = 0; ws_spatial = 0; ws_kHW = 0; ws_oW = 0;
        ws_nhwc_reorder = 0;
        // 1D Conv (ndim==3): size im2col workspace as 2D with H=1. No NHWC/NCHWc
        // eligibility — 1D models (e.g. Whisper encoder front-end) run the
        // NCHW/FP16 scalar im2col+GEMM path.
        if (ndim == 3) {
            const int kC = w->dims[1];
            const int kL = w->dims[2];
            const int CHW = kC * kL;
            const int oL = dims[2];
            const int iC = x->dims[1];
            const bool depthwise = (group == iC && kC == 1);
            const bool is1x1 = (kL == 1
                && strides.size() >= 1 && strides[0] == 1
                && dilations[0] == 1
                && cpads[0] == 0 && cpads[kernels.size()] == 0);
            const bool true_dw = depthwise && (w->dims[0] == iC);
            if (!is1x1 && (!depthwise || !true_dw)) {
                ws_CHW = CHW;
                ws_spatial = oL;
                ws_kHW = kL;
                ws_oW = oL;
            }
        }
        if (ndim == 4) {
            const int kC = w->dims[1], kH = w->dims[2], kW = w->dims[3];
            const int kHW = kH * kW, CHW = kC * kHW;
            const int iC = x->dims[1];
            const int oH = dims[2], oW = dims[3];
            const int MM = w->dims[0] / group;
            const bool depthwise = (group == iC && kC == 1);
            const bool is1x1 = (kH == 1 && kW == 1
                && strides.size() >= 2 && strides[0] == 1 && strides[1] == 1
                && dilations[0] == 1 && dilations[1] == 1
                && cpads[0] == 0 && cpads[1] == 0
                && cpads[kernels.size()] == 0 && cpads[kernels.size() + 1] == 0);
            const bool true_dw = depthwise && (w->dims[0] == iC); // M == iC
            if (!is1x1 && (!depthwise || !true_dw)) {
                ws_CHW = CHW;
                ws_spatial = oH * oW;
                ws_kHW = kHW;
                ws_oW = oW;
            }
            // NHWC 1×1 Conv: group=1, float, pre-transpose weights
            if (is1x1 && !depthwise && group == 1 && x->type == NNR_DATA_TYPE_FLOAT32) {
                layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
                const int M = w->dims[0];
#ifdef NNR_ARCH_X64
                // NCHWc 1x1 conv: kernel + weight pack already exist in
                // exec_nchwc_blocked (conv1x1_nchwc_avx512 / pack_weight_nchwc_1x1).
                // Advertise the layout so chains containing 1x1 Convs can stay in
                // BLOCKED_16 — and so their Concat outputs become alias-eligible
                // in memory_planner (Phase 4.5 of assign_blocked_layouts).
                if (iC % NATIVE_BLOCK == 0 && M % NATIVE_BLOCK == 0
                    && iC >= NATIVE_BLOCK && M >= NATIVE_BLOCK)
                    layout_mask |= LAYOUT_NATIVE_BLOCKED;
#endif
                w_nhwc.resize(M * iC);
                transpose_weights(w_nhwc.data(), (const float*)w->data, M, iC);
                w_dw_nhwc.clear();
                w_packed.clear(); // packed lazily in exec() when NHWC is actually assigned
                ws_nhwc_reorder = (size_t)x->dims[0] * iC * oH * oW * sizeof(float);
            } else if (true_dw && x->type == NNR_DATA_TYPE_FLOAT32) {
                // NHWC depthwise: repack weights [C, 1, kH, kW] → [kH, kW, C]
                layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
                // NCHWc depthwise (BLOCKED_16 on x64, BLOCKED_8 on ARM64):
                // C must be multiple of NATIVE_BLOCK, no dilation.
                // Both arches ship a per-block pixel kernel (see
                // depthwise_nchwc_pixel_avx512 / depthwise_nchwc_pixel_neon).
                if (iC % NATIVE_BLOCK == 0 && iC >= NATIVE_BLOCK
                    && dilations[0] == 1 && dilations[1] == 1)
                    layout_mask |= LAYOUT_NATIVE_BLOCKED;
                w_nhwc.clear();
                w_packed.clear();
                const int C = w->dims[0];
                w_dw_nhwc.resize((size_t)kH * kW * C);
                const float* src = (const float*)w->data;
                for (int c = 0; c < C; c++)
                    for (int kh_ = 0; kh_ < kH; kh_++)
                        for (int kw_ = 0; kw_ < kW; kw_++)
                            w_dw_nhwc[(kh_ * kW + kw_) * C + c] = src[c * kH * kW + kh_ * kW + kw_];
                ws_nhwc_reorder = (size_t)x->dims[0] * iC * x->dims[2] * x->dims[3] * sizeof(float);
                // Clear NCHWc DW weights (lazy-packed on first exec entry)
                w_dw_nchwc.clear();
                bias_dw_nchwc.clear();
            } else if (x->type == NNR_DATA_TYPE_FLOAT32) {
                // General NHWC conv: repack weights from NCHW (c,kh,kw) to NHWC (kh,kw,c) ordering
                layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
#ifdef NNR_ARCH_X64
                // NCHWc general conv: IC and OC must be multiples of NATIVE_BLOCK, no dilation
                if (group == 1 && iC % NATIVE_BLOCK == 0 && w->dims[0] % NATIVE_BLOCK == 0
                    && iC >= NATIVE_BLOCK && w->dims[0] >= NATIVE_BLOCK
                    && dilations[0] == 1 && dilations[1] == 1)
                    layout_mask |= LAYOUT_NATIVE_BLOCKED;
                // (ARM M3: add NEON NCHWc general K×K kernel here.)
#endif
                w_nhwc.clear();
                w_dw_nhwc.clear();
                const int K = kC * kH * kW;
                const int M = w->dims[0];
                w_gemm_nhwc.resize((size_t)M * K);
                const float* src = (const float*)w->data;
                for (int g = 0; g < group; g++) {
                    for (int m = 0; m < MM; m++) {
                        int m_abs = g * MM + m;
                        for (int kh_ = 0; kh_ < kH; kh_++)
                            for (int kw_ = 0; kw_ < kW; kw_++)
                                for (int c = 0; c < kC; c++) {
                                    int k_nhwc = (kh_ * kW + kw_) * kC + c;
                                    int k_nchw = (c * kH + kh_) * kW + kw_;
                                    w_gemm_nhwc[(size_t)g * K * MM + k_nhwc * MM + m] =
                                        src[(size_t)m_abs * K + k_nchw];
                                }
                    }
                }
                w_packed.clear(); // packed lazily in exec() when NHWC is actually assigned
                ws_nhwc_reorder = (size_t)x->dims[0] * iC * x->dims[2] * x->dims[3] * sizeof(float);
            } else if (x->type == NNR_DATA_TYPE_FLOAT16) {
                // FP16 on ARM64: advertise NHWC so the layout optimizer can
                // promote FP16 chains to NHWC, which unlocks the native
                // `exec_fp16_{dw,}nhwc_direct_neon` paths.  The FP16-specific
                // NHWC packs (`w_fp16_nhwc_direct`, `w_fp16_dw_nhwc`) are
                // populated separately in the NNR_ARCH_ARM64 block below;
                // the FP32 packs (`w_nhwc` etc.) are unused on this path.
#ifdef NNR_ARCH_ARM64
                layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
#else
                layout_mask = LAYOUT_NCHW;
#endif
                w_nhwc.clear();
                w_dw_nhwc.clear();
                w_gemm_nhwc.clear();
                w_packed.clear();
            } else {
                layout_mask = LAYOUT_NCHW;
                w_nhwc.clear();
                w_dw_nhwc.clear();
                w_gemm_nhwc.clear();
                w_packed.clear();
            }
            // Winograd F(4×4, 3×3): pre-transform filters when eligible.
            if (x->type == NNR_DATA_TYPE_FLOAT32 && !depthwise
                && kH == 3 && kW == 3
                && strides.size() >= 2 && strides[0] == 1 && strides[1] == 1
                && dilations[0] == 1 && dilations[1] == 1) {
                const int M_total = w->dims[0];
                w_winograd.resize((size_t)36 * M_total * kC);
                const float* wd = (const float*)w->data;
                for (int m = 0; m < M_total; ++m) {
                    for (int c = 0; c < kC; ++c) {
                        float u[36];
                        winograd_filter_transform(u, wd + ((size_t)m * kC + c) * 9);
                        for (int pos = 0; pos < 36; ++pos)
                            w_winograd[(size_t)pos * M_total * kC + m * kC + c] = u[pos];
                    }
                }
                const int tH = (oH + 3) / 4, tW = (oW + 3) / 4;
                const int num_tiles = tH * tW;
                // Pre-pack each of the 36 weight matrices for dgemm_packed_a
                size_t pa_sz = pack_a_size(M_total, kC);
                if (pa_sz > 0) {
                    w_winograd_packed.resize(36 * pa_sz);
                    for (int pos = 0; pos < 36; ++pos)
                        pack_a(w_winograd_packed.data() + pos * pa_sz,
                            w_winograd.data() + (size_t)pos * M_total * kC, M_total, kC);
                } else {
                    w_winograd_packed.clear();
                }
                // NHWC: transpose weights to [36][kC][M] and pre-pack as B-panels
                w_winograd_nhwc.resize((size_t)36 * kC * M_total);
                for (int pos = 0; pos < 36; ++pos)
                    for (int c = 0; c < kC; ++c)
                        for (int m = 0; m < M_total; ++m)
                            w_winograd_nhwc[(size_t)pos * kC * M_total + c * M_total + m]
                                = w_winograd[(size_t)pos * M_total * kC + m * kC + c];
                size_t pb_sz = pack_b_size(kC, M_total);
                if (pb_sz > 0) {
                    w_winograd_packed_nhwc.resize(36 * pb_sz);
                    for (int pos = 0; pos < 36; ++pos)
                        pack_b(w_winograd_packed_nhwc.data() + pos * pb_sz,
                            w_winograd_nhwc.data() + (size_t)pos * kC * M_total, kC, M_total);
                } else {
                    w_winograd_packed_nhwc.clear();
                }
                // Workspace: V[36][kC][gs] + M_buf[36][M][gs] for one tile group.
                // Process all tiles in a single group to minimize GEMM call count (36 per group).
                // Cap workspace at half an L3 domain to prevent excessive allocation.
                const int WS_MAX = (int)cpu_features().l3_kb_per_domain * 1024 / 2;
                int max_gs = WS_MAX / (int)(36 * (kC + M_total) * sizeof(float));
                max_gs = std::max(max_gs, 16);
                wino_group = std::min(max_gs, num_tiles);
                ws_winograd = 36 * (kC + M_total) * wino_group;
            } else {
                w_winograd.clear();
                w_winograd_packed.clear();
                w_winograd_nhwc.clear();
                w_winograd_packed_nhwc.clear();
                ws_winograd = 0;
            }
            // NCHWc (blocked layout) eligibility check.
            // Mark eligible Convs with LAYOUT_NATIVE_BLOCKED so the graph
            // optimizer's structural gate can consider them for NCHWc chains.
            // Weight packing is deferred to exec() (lazy-pack on first entry)
            // to avoid allocating duplicate weights for Convs that the gate
            // ultimately rejects.
            //
            // Block width is NATIVE_BLOCK (16 on x64 AVX-512, 8 on ARM NEON).
            // On ARM this is M1 of the NCHWc plan: only 1×1 Conv advertises
            // BLOCKED_8; general K×K and depthwise land in later milestones.
#ifdef NNR_ARCH_X64
            // NCHWc (LAYOUT_NATIVE_BLOCKED) eligibility:
            //   - IC ≥ NATIVE_BLOCK. IC-tail (iC % NATIVE_BLOCK != 0) is
            //     accepted for K×K (K > 1) Convs only when the partial last
            //     block is at least half-full (ic_tail >= block/2). The kernel
            //     iterates ICb = ceil(IC/block) and FMAs over zero-padded lanes
            //     contribute 0, but those wasted lanes still consume issue slots.
            //     1×1 Convs are bandwidth-bound, so any IC-tail waste dominates
            //     (mobilenetv2's iC=24 1×1 expands regress +9% with IC-tail).
            //     K×K Convs amortize waste over kH*kW reuse — googlenet inception
            //     branch3.1 (iC=24, tail=8=block/2, 3×3) wins -16% by joining the
            //     chain that unblocks two Concat aliases. But densenet-/efficientnet-
            //     style chains with smaller tails (e.g., tail=2-6) over-promote
            //     into long BLOCKED chains where boundary reorders + partial-block
            //     waste exceed the alias savings (densenetblur121d +38%, densenet-
            //     12-int8 +76% under unconditional K×K IC-tail). The block/2
            //     threshold caps wasted SIMD lanes at <=50% of the last block.
            //   - OC no longer required to be a multiple of NATIVE_BLOCK.
            //     OC-tail Convs become "terminal blocked consumers": they
            //     accept BLOCKED_16 input (avoiding the large input reorder)
            //     but produce NCHW output via an OCb-sized workspace buffer
            //     plus a small nchwc_to_nchw transpose. This is what unlocks
            //     the SSD-12 pred-head chain (OC ∈ {24, 324, 486}).
            const int ic_tail = iC % NATIVE_BLOCK;
            const bool ic_tail_ok = (ic_tail == 0)
                || ((kH > 1 || kW > 1) && ic_tail >= NATIVE_BLOCK / 2);
            if (!depthwise && group == 1 && x->type == NNR_DATA_TYPE_FLOAT32
                && iC >= NATIVE_BLOCK && w->dims[0] >= 1
                && ic_tail_ok
                && dilations[0] == 1 && dilations[1] == 1) {
                layout_mask |= LAYOUT_NATIVE_BLOCKED;
                // Workspace for NCHWc convolution at chain boundary.
                constexpr int block = NATIVE_BLOCK;
                const int OC = w->dims[0];
                const int OCb = (OC + block - 1) / block;
                const bool oc_tail = (OC % block != 0);
                // Output blocked workspace (used when y->format != BLOCKED_16,
                // i.e., the terminal-consumer case): OCb*block output channels
                // × oH × oW per batch. Only needed when OC is non-aligned or
                // when the Conv is picked as a chain exit.
                size_t out_blk = (size_t)x->dims[0] * OCb * oH * oW * block * sizeof(float);
                if (is1x1) {
                    // GEMM path: need NCHW output (OC × HW) + NCHW input (IC × HW)
                    // per batch (processed one batch at a time).
                    ws_nchwc_reorder = ((size_t)iC + OC) * oH * oW * sizeof(float);
                    if (oc_tail) ws_nchwc_reorder = std::max(ws_nchwc_reorder, out_blk);
                    // 1×1 stride>1 falls through to the general K×K NCHWc strip
                    // path (the 1×1 fast path requires sH=1). When the input is
                    // NCHW (boundary case), exec_strip reorders into NCHWc
                    // workspace sized [N][ICb][iH][iW][block].
                    if (strides[0] != 1 || strides[1] != 1) {
                        const int ICb = (iC + block - 1) / block;
                        size_t in_blk = (size_t)x->dims[0] * ICb
                                      * x->dims[2] * x->dims[3] * block * sizeof(float);
                        ws_nchwc_reorder = std::max(ws_nchwc_reorder, in_blk);
                    }
                } else {
                    // Pre-padding eliminates bounds-checking branches in the kernel's
                    // inner loop — critical for small spatial (7×7: ~27-49% edge pixels).
                    const int ICb = (iC + block - 1) / block;
                    const int nk2 = (int)kernels.size();
                    const int pH = x->dims[2] + cpads[0] + cpads[nk2];
                    const int pW = x->dims[3] + cpads[1] + cpads[nk2 + 1];
                    const size_t in_pad = (size_t)x->dims[0] * ICb * pH * pW * block * sizeof(float);
                    ws_nchwc_reorder = in_pad;
                    // Terminal consumer needs room for input-padded + blocked output.
                    if (oc_tail) ws_nchwc_reorder = in_pad + out_blk;

                    // Winograd F(4,3) NCHWc additionally needs V[36][ICb][num_tiles][block]
                    // + M[36][OCb][num_tiles][block] buffers. Input goes unpadded
                    // (kernel handles edges internally). Take the max so the same
                    // workspace covers either NCHWc kernel path at exec() time.
                    // Winograd only fires when OC aligns to block, so skip when oc_tail.
                    const bool wino_eligible = kH == 3 && kW == 3
                        && strides[0] == 1 && strides[1] == 1
                        && !oc_tail && (OC % block == 0);
                    if (wino_eligible) {
                        const int M_total = w->dims[0];
                        // Unpadded NCHWc input (chain boundary case).
                        size_t wino_in = (size_t)x->dims[0] * ICb * x->dims[2] * x->dims[3] * block * sizeof(float);
                        // V+M scratch for the Winograd transforms.
                        size_t wino_vm = nnr::winograd_nchwc_workspace_size(iC, M_total, oH, oW);
                        if (wino_in + wino_vm > ws_nchwc_reorder)
                            ws_nchwc_reorder = wino_in + wino_vm;
                    }
                }
            }
            w_nchwc.clear();
            bias_nchwc.clear();
            w_wino_nchwc.clear();
#elif defined(NNR_ARCH_ARM64)
            // ARM M1: 1×1 Conv only. General K×K and depthwise come in M2/M3.
            if (is1x1 && !depthwise && group == 1 && x->type == NNR_DATA_TYPE_FLOAT32
                && iC % NATIVE_BLOCK == 0 && w->dims[0] % NATIVE_BLOCK == 0
                && iC >= NATIVE_BLOCK && w->dims[0] >= NATIVE_BLOCK
                && dilations[0] == 1 && dilations[1] == 1) {
                layout_mask |= LAYOUT_NATIVE_BLOCKED;
                // 1×1 path: input reordered NCHW→NCHW8c at chain boundary,
                // output written directly into BLOCKED_8 layout by the kernel.
                constexpr int block = NATIVE_BLOCK;
                const int ICb = (iC + block - 1) / block;
                ws_nchwc_reorder = (size_t)x->dims[0] * ICb * oH * oW * block * sizeof(float);
            }
            w_nchwc.clear();
            bias_nchwc.clear();
#endif

            // Pre-pack NCHW weights for the tiled GEMM path (A-panel format).
            // Conditions: MM > 1 (not GEMV). Always pre-pack regardless of CHW —
            // the pack cost is paid once at prepare time, and dgemm_packed_a avoids
            // per-tile A-packing overhead that dgemm_generic would incur.
            // The spatial >= 16 check (not small-M) is deferred to exec().
            if (x->type == NNR_DATA_TYPE_FLOAT32 && !depthwise && MM > 1) {
                size_t per_group = pack_a_size(MM, CHW);
                w_packed_nchw.resize(per_group * group);
                for (int g = 0; g < group; g++)
                    pack_a(w_packed_nchw.data() + g * per_group,
                        (const float*)w->data + (size_t)g * MM * CHW, MM, CHW);
            } else {
                w_packed_nchw.clear();
            }

            // First-layer direct conv: pack weights for small-IC Conv (e.g., RGB input).
            // Skips im2col entirely, saving 7+MB buffer for 7×7 stride-2 layers.
#ifdef NNR_ARCH_X64
            // Format: [IC, KH, KW, OC/16, 16] — vectorize over 16 OC.
            // First/last-layer direct kernels are AVX-512-only (16-wide ZMM tiles).
            if (has_avx512()
                && x->type == NNR_DATA_TYPE_FLOAT32 && !depthwise && group == 1
                && iC <= 4 && kH >= 3 && MM >= 16) {
                size_t psz = pack_weights_first_layer_size(MM, iC, kH, kW);
                w_first_layer.resize(psz);
                pack_weights_first_layer(w_first_layer.data(),
                    (const float*)w->data, MM, iC, kH, kW);
            } else {
                w_first_layer.clear();
            }

            // Last-layer direct conv: small-OC Conv (e.g., RGB output).
            // Vectorizes over output width (16 pixels per ZMM), skipping im2col.
            // Conditions: OC <= 16, stride == 1, group == 1, not first-layer.
            if (has_avx512()
                && x->type == NNR_DATA_TYPE_FLOAT32 && !depthwise && group == 1
                && w_first_layer.empty()
                && MM <= 16 && kH >= 3
                && strides.size() >= 2 && strides[0] == 1 && strides[1] == 1
                && dilations[0] == 1 && dilations[1] == 1) {
                use_last_layer = true;
                const int nk2 = (int)kernels.size();
                ws_last_layer = conv_last_layer_workspace(
                    iC, x->dims[2], x->dims[3],
                    cpads[0], cpads[nk2], cpads[1], cpads[nk2 + 1]);
            } else {
                use_last_layer = false;
                ws_last_layer = 0;
            }
#elif defined(NNR_ARCH_ARM64)
            // Format: [IC, KH, KW, OC/4, 4] — vectorize over 4 OC.
            if (x->type == NNR_DATA_TYPE_FLOAT32 && !depthwise && group == 1
                && iC <= 4 && kH >= 3 && MM >= 4) {
                size_t psz = pack_weights_first_layer_neon_size(MM, iC, kH, kW);
                w_first_layer.resize(psz);
                pack_weights_first_layer_neon(w_first_layer.data(),
                    (const float*)w->data, MM, iC, kH, kW);
            } else {
                w_first_layer.clear();
            }

            // Last-layer direct conv: small-OC Conv (e.g., RGB output).
            // Vectorizes over output width (4 pixels per NEON q-reg), skipping im2col.
            // IC tiling in kernel keeps pad_buf working set in L2.
            // OC ≤ 4: GEMM 8×8 micro-kernel wastes ≥50% for M≤4; for M≥8 GEMM is efficient.
            if (x->type == NNR_DATA_TYPE_FLOAT32 && !depthwise && group == 1
                && w_first_layer.empty()
                && MM <= 4 && kH >= 3
                && strides.size() >= 2 && strides[0] == 1 && strides[1] == 1
                && dilations[0] == 1 && dilations[1] == 1) {
                use_last_layer = true;
                const int nk2 = (int)kernels.size();
                ws_last_layer = conv_last_layer_workspace(
                    iC, x->dims[2], x->dims[3],
                    cpads[0], cpads[nk2], cpads[1], cpads[nk2 + 1]);
            } else {
                use_last_layer = false;
                ws_last_layer = 0;
            }
#endif
        }
        // FP16/BF16: pre-convert weights and bias to float32 (one-time cost in reshape)
        nnr_aligned_free(w_f32); w_f32 = nullptr;
        nnr_aligned_free(bias_f32); bias_f32 = nullptr;
        if (x->type == NNR_DATA_TYPE_FLOAT16) {
            size_t wn = w->ndata;
            w_f32 = (float*)nnr_aligned_alloc(wn * sizeof(float), 64);
            convert_f16_to_f32(w_f32, (const float16_t*)w->data, wn);
            if (inputs.size() > 2 && inputs[2]->data) {
                size_t bn = inputs[2]->ndata;
                bias_f32 = (float*)nnr_aligned_alloc(bn * sizeof(float), 64);
                convert_f16_to_f32(bias_f32, (const float16_t*)inputs[2]->data, bn);
            }
            // FP16 native NCHW direct conv pack — ARM64 only, eligible shapes:
            //   4D conv, group=1, dilation=1, not depthwise.
            // The pack is small relative to the FP32 convert above and leaves
            // the FP32 fallback intact for shapes the kernel can't handle.
#ifdef NNR_ARCH_ARM64
            w_fp16_direct.clear();
            w_fp16_nhwc_direct.clear();
            w_fp16_dw_nhwc.clear();
            if (has_neon_fp16() && x->ndim == 4
                && dilations.size() >= 2) {
                const int M_oc = w->dims[0];
                const int kC_w = w->dims[1];
                const int kH_w = w->dims[2], kW_w = w->dims[3];
                const int iC_x = x->dims[1];
                const bool depthwise_fp16 = (group == iC_x && kC_w == 1);
                if (depthwise_fp16 && M_oc == iC_x) {
                    // Depthwise FP16 NHWC pack: [C, 1, kH, kW] → [kH*kW, C] FP16.
                    size_t psz_dw = nnr::fp16::neon::repack_weights_depthwise_fp16_nhwc_size(
                        iC_x, kH_w, kW_w);
                    if (psz_dw > 0) {
                        w_fp16_dw_nhwc.resize(psz_dw / sizeof(uint16_t));
                        nnr::fp16::neon::repack_weights_depthwise_fp16_nhwc(
                            w_fp16_dw_nhwc.data(),
                            (const uint16_t*)w->data,
                            iC_x, kH_w, kW_w);
                    }
                }
                if (!depthwise_fp16 && group == 1
                    && dilations[0] == 1 && dilations[1] == 1) {
                    size_t psz = nnr::fp16::neon::pack_weights_fp16_direct_neon_size(
                        M_oc, kC_w, kH_w, kW_w);
                    if (psz > 0) {
                        w_fp16_direct.resize(psz / sizeof(uint16_t));
                        nnr::fp16::neon::pack_weights_fp16_direct_neon(
                            w_fp16_direct.data(),
                            (const uint16_t*)w->data,
                            M_oc, kC_w, kH_w, kW_w);
                    }
                    size_t psz_nhwc = nnr::fp16::neon::pack_weights_fp16_direct_nhwc_neon_size(
                        M_oc, kC_w, kH_w, kW_w);
                    if (psz_nhwc > 0) {
                        w_fp16_nhwc_direct.resize(psz_nhwc / sizeof(uint16_t));
                        nnr::fp16::neon::pack_weights_fp16_direct_nhwc_neon(
                            w_fp16_nhwc_direct.data(),
                            (const uint16_t*)w->data,
                            M_oc, kC_w, kH_w, kW_w);
                    }
                }
            }
#endif
        } else if (x->type == NNR_DATA_TYPE_BFLOAT16) {
            size_t wn = w->ndata;
            w_f32 = (float*)nnr_aligned_alloc(wn * sizeof(float), 64);
            convert_bf16_to_f32(w_f32, (const bfloat16_t*)w->data, wn);
            if (inputs.size() > 2 && inputs[2]->data) {
                size_t bn = inputs[2]->ndata;
                bias_f32 = (float*)nnr_aligned_alloc(bn * sizeof(float), 64);
                convert_bf16_to_f32(bias_f32, (const bfloat16_t*)inputs[2]->data, bn);
            }
        }

        return y->reshape(dims, x->type);
    }
