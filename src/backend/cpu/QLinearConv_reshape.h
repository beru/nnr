// -*- C++ -*-
// QLinearConv_operator::reshape() — extracted from QLinearConv.cpp.
//
// This file is NOT a standalone TU. It is #included inside the
// QLinearConv_operator class body in QLinearConv.cpp. It references
// class members (auto_pad, group, kernels, dilations, pads, strides,
//  cpads, per_channel_zp_, layout_mask, w_vnni_buf, w_row_sums_buf,
//  w_packed_f32, w_direct_buf, w_direct_sums, w_packed_nr48,
//  w_packed_nr48_col_sums, w_gather_packed, w_gather_col_sums,
//  w_dw_repacked, dw_ind_buf, dw_zero_buf, x_pad_buf, x_pad_nhwc_buf,
//  x_pad_zp, y_i32_buf, y_nhwc_buf, k_off_base, k_off_oh_all,
//  rq_output_scales, rq_combined_scales, rq_bias_f, rq_cached,
//  inputs, outputs) and the anonymous-namespace enum auto_pad_t
//  directly — they must all be in scope at the include site.
//
// Cold path: runs once per tensor-shape change, never per inference.

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        const tensor_t* w = inputs[3];
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
        // NHWC layout support: gather-GEMM (M=spatial, N=OC) writes contiguous NHWC.
        // Enable when: AVX-512 VNNI, stride=1, group=1, uint8, no per-channel ZP.
        // Also enable for depthwise (group==OC, C_per_group==1) — any stride.
#ifdef NNR_ARCH_X64
        if (x->ndim == 4 && has_avx512() && cpu_features().avx512vnni
            && x->type == NNR_DATA_TYPE_UINT8 && !per_channel_zp_) {
            bool is_depthwise = (group == w->dims[0] && w->dims[1] == 1);
            if (is_depthwise || group == 1)
                layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
        }
#endif

        // Pre-pack weights for the fast path.
        w_vnni_buf.clear();
        w_row_sums_buf.clear();
        w_packed_f32.clear();
        w_direct_buf.clear();
        w_direct_sums.clear();
        w_first_layer_int8.clear();
        w_first_layer_int8_sums.clear();
        w_packed_nr48.clear();
        w_packed_nr48_col_sums.clear();
        x_pad_nhwc_buf.clear();
        k_off_base.clear();
        k_off_oh_all.clear();
        x_pad_buf.clear();
        y_i32_buf.clear();
        w_dw_repacked.clear();
        dw_ind_buf.clear();
        dw_zero_buf.clear();
        if (x->ndim == 4 && (x->type == NNR_DATA_TYPE_UINT8 || x->type == NNR_DATA_TYPE_INT8)) {
            const tensor_t* wt = inputs[3];
            const tensor_t* w_zp_t = inputs[5];
            int M_total = wt->dims[0];
            int CHW_val = wt->dims[1] * wt->dims[2] * wt->dims[3];
            int MM_val = M_total / group;

            // Convert W to float32 (subtract zero-point) and pre-pack with pack_a
            float* w_f32_tmp = (float*)_aligned_malloc(wt->ndata * sizeof(float), 64);
            for (int oc = 0; oc < M_total; oc++) {
                int32_t wzp = 0;
                if (!per_channel_zp_ && w_zp_t->ndata > 0) {
                    if (wt->type == NNR_DATA_TYPE_INT8) wzp = *(int8_t*)w_zp_t->data;
                    else wzp = *(uint8_t*)w_zp_t->data;
                } else if (per_channel_zp_) {
                    if (wt->type == NNR_DATA_TYPE_INT8) wzp = ((int8_t*)w_zp_t->data)[oc];
                    else wzp = ((uint8_t*)w_zp_t->data)[oc];
                }
                const int8_t* wrow = (const int8_t*)wt->data + (size_t)oc * CHW_val;
                float* wf = w_f32_tmp + (size_t)oc * CHW_val;
                for (int i = 0; i < CHW_val; i++)
                    wf[i] = (float)((int32_t)wrow[i] - wzp);
            }

            // Pre-pack per group using pack_a (same format as FP32 Conv)
            size_t per_group = pack_a_size(MM_val, CHW_val);
            w_packed_f32.resize(per_group * group);
            for (int g_idx = 0; g_idx < group; g_idx++)
                pack_a(w_packed_f32.data() + g_idx * per_group,
                    w_f32_tmp + (size_t)g_idx * MM_val * CHW_val, MM_val, CHW_val);
            _aligned_free(w_f32_tmp);
        }
#ifdef NNR_ARCH_X64
        // Also pre-shift W for VNNI path (if available)
        if (has_avx512() && cpu_features().avx512vnni && x->ndim == 4
            && x->type == NNR_DATA_TYPE_UINT8 && !per_channel_zp_) {
            const tensor_t* wt = inputs[3];
            int CHW_val = wt->dims[1] * wt->dims[2] * wt->dims[3];

            w_vnni_buf.resize(wt->ndata);
            const int8_t* wdata = (const int8_t*)wt->data;
            for (size_t i = 0; i < wt->ndata; i++)
                w_vnni_buf[i] = (uint8_t)((int)wdata[i] + 128);

            int M_total = wt->dims[0];
            w_row_sums_buf.resize(M_total);
            for (int m = 0; m < M_total; m++) {
                int32_t sum = 0;
                for (int k = 0; k < CHW_val; k++)
                    sum += (int32_t)w_vnni_buf[(size_t)m * CHW_val + k];
                w_row_sums_buf[m] = sum;
            }
        }

        // Pre-compute fused im2col k_off base table (oh-independent offsets).
        // Avoids per-inference division/modulo in conv_int8_fused_gemm.
        if (!w_vnni_buf.empty() && strides[0] == 1 && strides[1] == 1) {
            const tensor_t* wt = inputs[3];
            int C_val = wt->dims[1], kH_val = wt->dims[2], kW_val = wt->dims[3];
            int CHW_val = C_val * kH_val * kW_val;
            int K4 = (CHW_val + 3) & ~3;
            int iH_val = x->dims[2], iW_val = x->dims[3];
            int pH_val = cpads[0], pW_val = cpads[1];
            int padded_H = iH_val + pH_val + cpads[2];
            int padded_W = iW_val + pW_val + cpads[3];
            size_t plane = (size_t)padded_H * padded_W;

            k_off_base.resize(K4);
            for (int k = 0; k < CHW_val; k++) {
                int c = k / (kH_val * kW_val);
                int rem = k % (kH_val * kW_val);
                int kh = rem / kW_val;
                int kw = rem % kW_val;
                k_off_base[k] = (size_t)c * plane
                    + (size_t)(kh * dilations[0]) * padded_W
                    + (size_t)(kw * dilations[1]);
            }
            for (int k = CHW_val; k < K4; k++)
                k_off_base[k] = 0;

            // Also build oh-expanded table: k_off_oh_all[oh * K4 + k] = k_off_base[k] + oh * padW
            int oH_val = dims[2];
            int oW_val = dims[3];
            int MM_val = wt->dims[0] / group;
            k_off_oh_all.resize((size_t)oH_val * K4);
            for (int oh = 0; oh < oH_val; oh++) {
                size_t oh_off = (size_t)oh * padded_W;
                size_t* dst = k_off_oh_all.data() + (size_t)oh * K4;
                for (int k = 0; k < K4; k++)
                    dst[k] = k_off_base[k] + oh_off;
            }

            // Pre-allocate work buffers (aligned for AVX-512)
            int iC_val = x->dims[1];
            x_pad_buf.resize((size_t)iC_val * plane + 64);
            // Pre-fill with x_zp so exec only needs to copy actual pixels (no memset)
            x_pad_zp = (inputs[2]->ndata > 0) ? ((uint8_t*)inputs[2]->data)[0] : 0;
            memset(x_pad_buf.data(), x_pad_zp, x_pad_buf.size());
            constexpr size_t Y_TILE_BYTES = 2 * 1024 * 1024;
            int y_tile_h = std::max(1, (int)(Y_TILE_BYTES / ((size_t)MM_val * oW_val * sizeof(int32_t))));
            y_tile_h = std::min(y_tile_h, oH_val);
            y_i32_buf.resize((size_t)MM_val * y_tile_h * oW_val);

            // Gather-GEMM: pre-pack weights for group=1, stride=1 convs.
            if (group == 1) {
                const int8_t* wdata = (const int8_t*)wt->data;
                w_gather_packed.resize(int8::pack_weights_gather_nr48_size(MM_val, CHW_val));
                w_gather_col_sums.resize(MM_val);
                int8::pack_weights_gather_nr48(
                    w_gather_packed.data(), w_gather_col_sums.data(),
                    wdata, MM_val, CHW_val);
            }
        }

        // Packed NR=48: pre-pack weights for group=1 convs (any stride).
        // Sub-stride format with (kh,kw,ic) K-order for NHWC pack_a.
        if (!w_vnni_buf.empty() && group == 1) {
            const tensor_t* wt = inputs[3];
            int C_val = wt->dims[1], kH_val = wt->dims[2], kW_val = wt->dims[3];
            int CHW_val = C_val * kH_val * kW_val;
            int MM_val = wt->dims[0] / group;

            w_packed_nr48.resize(int8::pack_weights_nr48_substride_size(MM_val, CHW_val));
            w_packed_nr48_col_sums.resize(MM_val);
            int8::pack_weights_nr48_nhwc(
                w_packed_nr48.data(), w_packed_nr48_col_sums.data(),
                (const int8_t*)wt->data, MM_val, C_val, kH_val, kW_val);

            // Pre-allocate NCHW→NHWC transpose buffer (only used when input is NCHW).
            x_pad_nhwc_buf.resize((size_t)x->dims[2] * x->dims[3] * C_val + 64);

            // Pre-allocate NHWC output temp buffer (only used when output is NCHW).
            // The packed NR=48 GEMM writes [spatial × OC] NHWC; we transpose to NCHW after.
            // Pad OC to NR=48 to avoid JIT requantize writes overflowing row boundaries.
            // Use local dims (not outputs[0]->dims) since y hasn't been reshaped yet.
            int oH_val = dims[2], oW_val = dims[3];
            int oc_padded = (MM_val + 47) & ~47;  // round up to NR=48
            size_t buf_sz = (size_t)oH_val * oW_val * oc_padded;
            y_nhwc_buf.resize(buf_sz);

            // NR=16 NHWC-direct (memcpy-free) path: kernel reads raw input
            // via per-pixel indirection and substitutes x_zp for OOB
            // kernel pixels — no pre-pad buffer needed. Dispatch predicate
            // is checked at call time in QLinearConv.cpp.
        }

        // Direct int8 conv: pack weights for VNNI direct path (no im2col).
        // Eligible when: group==1, dilation==1, non-depthwise.
        if (has_avx512() && cpu_features().avx512vnni && x->ndim == 4
            && x->type == NNR_DATA_TYPE_UINT8 && !per_channel_zp_
            && group == 1
            && dilations[0] == 1 && dilations[1] == 1) {
            const tensor_t* wt = inputs[3];
            int OC_val = wt->dims[0];
            int IC_val = wt->dims[1];
            int kH_val = wt->dims[2];
            int kW_val = wt->dims[3];

            w_direct_buf.resize(pack_weights_int8_direct_size(OC_val, IC_val, kH_val, kW_val));
            pack_weights_int8_direct(w_direct_buf.data(),
                (const int8_t*)wt->data, OC_val, IC_val, kH_val, kW_val);

            w_direct_sums.resize(OC_val);
            compute_weight_sums_int8_direct(w_direct_sums.data(),
                (const int8_t*)wt->data, OC_val, IC_val, kH_val, kW_val);

            // First-layer int8 direct path: eligible for small IC (RGB-style stem)
            // when output is NCHW, weights are symmetric, and kernel is big enough
            // to amortize prepad. Skips im2col + packed GEMM for a 9x speedup on
            // the ResNet stem (IC=3, 7x7, stride=2).
            const tensor_t* w_zp_t2 = inputs[5];
            bool w_zp_symmetric = true;
            if (w_zp_t2->ndata > 0) {
                // Must be 0 for VPDPBUSD path (kernel assumes zp=0).
                int32_t z0 = 0;
                if (wt->type == NNR_DATA_TYPE_INT8) z0 = ((int8_t*)w_zp_t2->data)[0];
                else z0 = ((uint8_t*)w_zp_t2->data)[0];
                if (z0 != 0) w_zp_symmetric = false;
            }
            if (IC_val <= 4 && kH_val >= 3 && OC_val >= 16
                && wt->type == NNR_DATA_TYPE_INT8 && w_zp_symmetric) {
                w_first_layer_int8.resize(
                    pack_weights_first_layer_int8_size(OC_val, IC_val, kH_val, kW_val));
                pack_weights_first_layer_int8(
                    w_first_layer_int8.data(),
                    (const int8_t*)wt->data, OC_val, IC_val, kH_val, kW_val);
                w_first_layer_int8_sums.resize(OC_val);
                compute_weight_sums_first_layer_int8(
                    w_first_layer_int8_sums.data(),
                    (const int8_t*)wt->data, OC_val, IC_val, kH_val, kW_val);
            }
        }

        // Depthwise int8: repack weights and pre-allocate indirection buffer.
        // Eligible: group==OC, C_per_group==1, uint8, VNNI.
        if (has_avx512() && cpu_features().avx512vnni && x->ndim == 4
            && x->type == NNR_DATA_TYPE_UINT8 && !per_channel_zp_) {
            const tensor_t* wt = inputs[3];
            int OC_val = wt->dims[0];
            int kH_val = wt->dims[2];
            int kW_val = wt->dims[3];
            if (group == OC_val && wt->dims[1] == 1) {
                int kHW = kH_val * kW_val;
                int oH_val = dims[2];
                int oW_val = dims[3];
                int C_pad = (OC_val + 15) & ~15;

                w_dw_repacked.resize(int8::repack_depthwise_weights_size(OC_val, kH_val, kW_val));
                int8::repack_depthwise_weights(
                    w_dw_repacked.data(), (const int8_t*)wt->data,
                    OC_val, kH_val, kW_val);

                // Pre-allocate indirection buffer (rebuilt per inference with actual x pointer)
                dw_ind_buf.resize((size_t)oH_val * oW_val * kHW);

                // Zero buffer for padding pixels (filled with x_zp at exec time)
                dw_zero_buf.resize(C_pad);
            }
        }
#endif

#ifdef NNR_ARCH_ARM64
        // Depthwise int8 on ARM: repack weights into [kH*kW, C_pad] layout (C_pad = round_up(OC, 4))
        // so the NEON depthwise_int8_nhwc kernel can load 8 channels per iter.
        // Eligible: group == OC, C_per_group == 1, uint8 input, symmetric per-tensor w_zp, dotprod CPU.
        if (has_neon_dotprod() && x->ndim == 4
            && x->type == NNR_DATA_TYPE_UINT8 && !per_channel_zp_) {
            const tensor_t* wt = inputs[3];
            int OC_val = wt->dims[0];
            int kH_val = wt->dims[2];
            int kW_val = wt->dims[3];
            if (group == OC_val && wt->dims[1] == 1) {
                int kHW = kH_val * kW_val;
                int oH_val = dims[2];
                int oW_val = dims[3];
                int C_pad = (OC_val + 3) & ~3;  // ARM NEON aligns to 4 (vs 16 on AVX-512)

                w_dw_repacked.resize(int8::neon::repack_depthwise_weights_neon_size(OC_val, kH_val, kW_val));
                int8::neon::repack_depthwise_weights_neon(
                    w_dw_repacked.data(), (const int8_t*)wt->data,
                    OC_val, kH_val, kW_val);

                dw_ind_buf.resize((size_t)oH_val * oW_val * kHW);
                dw_zero_buf.resize(C_pad);
            }
        }

        // Direct int8 conv on ARM: pack weights into SDOT (OC4×IC4) tile layout.
        // Eligible: group==1, dilation==1, uint8 input, symmetric per-tensor ZP, dotprod CPU.
        if (has_neon_dotprod() && x->ndim == 4
            && x->type == NNR_DATA_TYPE_UINT8 && !per_channel_zp_
            && group == 1
            && dilations[0] == 1 && dilations[1] == 1) {
            const tensor_t* wt = inputs[3];
            int OC_val = wt->dims[0];
            int IC_val = wt->dims[1];
            int kH_val = wt->dims[2];
            int kW_val = wt->dims[3];

            w_direct_buf.resize(pack_weights_int8_direct_neon_size(OC_val, IC_val, kH_val, kW_val));
            pack_weights_int8_direct_neon(w_direct_buf.data(),
                (const int8_t*)wt->data, OC_val, IC_val, kH_val, kW_val);

            w_direct_sums.resize(OC_val);
            compute_weight_sums_int8_direct_neon(w_direct_sums.data(),
                (const int8_t*)wt->data, OC_val, IC_val, kH_val, kW_val);

  #if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(_MSC_VER) && defined(_M_ARM64))
            // When the CPU has i8mm, also build the SMMLA pack (2-OC × 8-IC tile).
            // The same pack feeds both the NHWC exec path (direct) and the NCHW exec
            // path (which pre-transposes activations to NHWC in a scratch buffer and
            // runs the SMMLA body with NCHW-layout stores). SDOT pack above remains
            // as fallback for non-i8mm chips and for the NHWC path's initial build.
            if (has_neon_i8mm()) {
                w_direct_smmla_buf.resize(
                    pack_weights_int8_direct_nhwc_smmla_size(OC_val, IC_val, kH_val, kW_val));
                pack_weights_int8_direct_nhwc_smmla(w_direct_smmla_buf.data(),
                    (const int8_t*)wt->data, OC_val, IC_val, kH_val, kW_val);
            } else {
                w_direct_smmla_buf.clear();
            }
  #endif

            // The same SDOT pack feeds exec_direct_int8_nhwc_neon — advertise
            // NHWC so the graph optimizer can avoid NCHW↔NHWC transposes at op
            // boundaries (mirrors the x64 VNNI gate above).
            layout_mask = LAYOUT_NCHW | LAYOUT_NHWC;
        }
#endif

        // Pre-compute requantize scale/bias vectors (avoid per-inference heap alloc)
        if (x->ndim == 4 && (x->type == NNR_DATA_TYPE_UINT8 || x->type == NNR_DATA_TYPE_INT8)) {
            const tensor_t* wt = inputs[3];
            int M_total = wt->dims[0];
            float x_sc = *(float*)inputs[1]->data;
            float y_sc = *(float*)inputs[6]->data;
            float inv_y = 1.0f / y_sc;
            const float* w_sc = (const float*)inputs[4]->data;
            bool per_ch = (inputs[4]->ndata > 1);
            const int32_t* pb = (inputs.size() > 8 && inputs[8] && inputs[8]->ndata > 0)
                ? (const int32_t*)inputs[8]->data : nullptr;

            int32_t y_zp_val = 0;
            if (inputs[7] && inputs[7]->ndata > 0) {
                if (x->type == NNR_DATA_TYPE_UINT8) y_zp_val = *(uint8_t*)inputs[7]->data;
                else y_zp_val = *(int8_t*)inputs[7]->data;
            }
            int clamp_lo = (x->type == NNR_DATA_TYPE_UINT8) ? 0 : -128;
            int clamp_hi = (x->type == NNR_DATA_TYPE_UINT8) ? 255 : 127;

            rq_output_scales.resize(M_total);
            rq_combined_scales.resize(M_total);
            rq_bias_f.clear();
            for (int oc = 0; oc < M_total; oc++) {
                float ws = per_ch ? w_sc[oc] : w_sc[0];
                rq_combined_scales[oc] = x_sc * ws;
                rq_output_scales[oc] = rq_combined_scales[oc] * inv_y;
            }
            if (pb) {
                rq_bias_f.resize(M_total);
                for (int oc = 0; oc < M_total; oc++)
                    rq_bias_f[oc] = (float)pb[oc] * rq_combined_scales[oc];
            }

#ifdef NNR_ARCH_X64
            // Pre-fill constant rq_params fields (x64 int8 JIT kernels only)
            memset(&rq_cached, 0, sizeof(rq_cached));
            rq_cached.output_scales   = rq_output_scales.data();
            rq_cached.bias_int32      = pb;
            rq_cached.rq_qmin         = (float)clamp_lo - (float)y_zp_val;
            rq_cached.rq_qmax         = (float)clamp_hi - (float)y_zp_val;
            rq_cached.y_zp_int        = y_zp_val;
            rq_cached.combined_scales  = rq_combined_scales.data();
            rq_cached.bias_vals        = pb ? rq_bias_f.data() : nullptr;
            rq_cached.inv_y_scale      = inv_y;
            rq_cached.y_zp             = (float)y_zp_val;
            rq_cached.qmin             = (float)clamp_lo;
            rq_cached.qmax             = (float)clamp_hi;
            rq_cached.y_out_stride     = M_total / group;  // NHWC: stride = OC per group
#endif
        }

        return y->reshape(dims, x->type);
    }
