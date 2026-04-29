#pragma once
// NCHWc (BLOCKED_16) convolution execution, included from Conv.cpp.
// Requires: Conv_operator members (w_nchwc, bias_nchwc, strides, cpads, etc.)
// Requires: backend/x64/ headers included from Conv.cpp.

bool exec_conv_nchwc(float* bias) {
    tensor_t* y = outputs[0];
    const tensor_t* x = inputs[0];
    const tensor_t* w = inputs[1];
    const int M = w->dims[0], kC = w->dims[1];
    const int kH = w->dims[2], kW = w->dims[3];
    const int iC = x->dims[1], iH = x->dims[2], iW = x->dims[3];
    const int oN = y->dims[0], oH = y->dims[2], oW = y->dims[3];

    constexpr int block = 16;
    const bool is1x1_exec = (kH == 1 && kW == 1
        && strides.size() >= 2 && strides[0] == 1 && strides[1] == 1
        && cpads[0] == 0 && cpads[1] == 0);

    // Lazy-pack NCHWc weights on first entry
    if (w_nchwc.empty()) {
        const int OC = w->dims[0];
        const int OCb = (OC + block - 1) / block;
        if (is1x1_exec) {
            w_nchwc.resize((size_t)OCb * iC * block);
            pack_weight_nchwc_1x1(w_nchwc.data(), (const float*)w->data, OC, iC, block);
        } else {
            const int ICb = (iC + block - 1) / block;
            w_nchwc.resize((size_t)OCb * ICb * kH * kW * block * block);
            pack_weight_nchwc(w_nchwc.data(), (const float*)w->data, OC, iC, kH, kW, block);
        }
        bias_nchwc.resize(nchwc_padded_channels(OC, block));
        const float* bias_src = (inputs.size() > 2) ? (const float*)inputs[2]->data : nullptr;
        pack_bias_nchwc(bias_nchwc.data(), bias_src, OC, block);
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
        if (x->format == memory_layout_t::BLOCKED_16) {
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
        if (x->format == memory_layout_t::BLOCKED_16) {
            in_nchwc = (float*)x->data;
        } else {
            in_nchwc = (float*)ctx->workspace;
            nchw_to_nchwc(in_nchwc, (const float*)x->data, oN, iC, iH, iW, block);
        }
    }

    float* out_nchwc = (float*)y->data;

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
        const int OCb = (M + block - 1) / block;
        const int total = oN * OCb * oH * oW * block;
        nnr::apply_post_fn_parallel(post_fn, out_nchwc, total, fused_op);
    }

    y->format = memory_layout_t::BLOCKED_16;
    return true;
}
