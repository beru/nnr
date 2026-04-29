#include "nnr.h"
#include "tensor_view.h"
#include "aligned_alloc.h"
#include "util.h"
#include "profiler.h"
#include "conv_shape.h"
#include "kernel/conv.h"
#include "kernel/conv_direct.h"
#include "kernel/winograd.h"
#include "kernel/layout.h"
#include "kernel/conv_nchwc.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/depthwise_avx512.h"
#include "backend/x64/depthwise_avx2.h"
#include "backend/x64/conv_direct_avx512.h"
#include "backend/x64/conv_first_layer_avx512.h"
#include "backend/x64/conv_last_layer_avx512.h"
#include "backend/x64/depthwise_nhwc_x64.h"
#endif
#ifdef NNR_ARCH_ARM64
#include "backend/arm/depthwise_neon.h"
#include "backend/arm/conv_first_layer_neon.h"
#include "backend/arm/conv_last_layer_neon.h"
#include "backend/arm/conv_direct_neon_inline.h"
#include "backend/arm/depthwise_nhwc_neon.h"
#include "backend/arm/depthwise_nchwc_neon.h"
#include "backend/arm/conv_fp16_direct_neon.h"
#include "backend/arm/conv_fp16_nhwc_direct_neon.h"
#include "backend/arm/depthwise_fp16_neon.h"
#ifdef NNR_USE_XBYAK_AARCH64
#include "backend/arm/jit_gemm_driver.h"
#endif
#endif
#include "cpu_features.h"
#include "layout_cost.h"
#include "kernel/f16_convert.h"

namespace nnr {

namespace {

// im2col tiling threshold: tile spatially when the im2col buffer exceeds
// the per-core L2 cache so the im2col data stays cache-resident for GEMM.
static inline size_t im2col_tile_bytes() {
    return (size_t)cpu_features().l2_kb * 1024;
}

// Minimum Winograd F(4x4,3x3) tiles for the transform overhead to amortize.
// Each tile covers 4x4 output pixels. At 7x7 spatial, tiles = 2x2 = 4 (too few).
// Threshold 16 = 4x4 grid → oH,oW >= 13 for Winograd eligibility.
static constexpr int WINOGRAD_MIN_TILES = 16;

// NOTE: ctx.set_force_nchwc(true) used to bypass NCHWc cost penalties
// (1x1 and Winograd) in the cost model. After the ORT-style eligibility
// refactor in assign_blocked_layouts.cpp there is no NCHWc cost model
// left, and the flag is currently a no-op. Kept in the public API for
// a future version that restores its intent by relaxing rules 4/5 in
// the eligibility gate.

enum auto_pad_t {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
};

// @nnr-meta-op mt=cost layout=[NCHW,NHWC,BLOCKED_16] workspace=yes prepack=yes scroll=yes fusion=[post_op,binary,bn]
struct Conv_operator : public operator_t {
    auto_pad_t auto_pad = NOTSET;
    int group = 0;
    small_vector<int> kernels;
    small_vector<int> dilations;
    small_vector<int, MAX_NDIM * 2> pads;
    small_vector<int> strides;
    int cpads[32] = {0};         // computed paddings: [begin_pads..., end_pads...] for each spatial dim
    // Cached workspace dimensions for workspace_size()
    int ws_CHW = 0;              // kC * kH * kW — im2col column height
    int ws_spatial = 0;          // oH * oW — output spatial size
    int ws_kHW = 0;              // kH * kW — kernel spatial size
    int ws_oW = 0;               // oW — output width (for tiled im2col workspace)
    // NHWC 1×1 Conv: transposed weight [C × M] and workspace for input reorder
    std::vector<float> w_nhwc;
    // NHWC depthwise Conv: repacked weights [kH, kW, C]
    std::vector<float> w_dw_nhwc;
    // NHWC general Conv: repacked weights [(kH*kW*kC) × MM] per group
    std::vector<float> w_gemm_nhwc;
    // Pre-packed GEMM B-panels for NHWC (eliminates per-tile B-copy)
    std::vector<float> w_packed;
#ifdef NNR_USE_XBYAK_AARCH64
    // JIT 6×16 packed weights: [M/16, K, 16] panels for dgemm_jit
    std::vector<float> w_packed_jit;
#endif
    // Pre-packed GEMM A-panels for NCHW (eliminates per-tile A-copy)
    std::vector<float> w_packed_nchw;
    // Winograd F(4×4, 3×3): pre-transformed filters [36][M][kC]
    std::vector<float> w_winograd;
    // Winograd: pre-packed A-panels for 36 position-wise GEMMs (NCHW)
    std::vector<float> w_winograd_packed;
    // Winograd NHWC: transposed weights [36][kC][M] and pre-packed B-panels
    std::vector<float> w_winograd_nhwc;
    std::vector<float> w_winograd_packed_nhwc;
    int ws_winograd = 0;         // workspace size for Winograd path (in floats)
    int wino_group = WINO_GROUP_DEFAULT; // per-layer tile group size
    // First-layer direct conv: repacked weights [IC, KH, KW, OC/16, 16]
    std::vector<float> w_first_layer;
    // Last-layer direct conv: flag + workspace size for small-OC Conv
    bool use_last_layer = false;
    size_t ws_last_layer = 0;
    size_t ws_nhwc_reorder = 0;  // workspace needed for NCHW→NHWC input reorder at boundaries
    // NCHWc (blocked layout) packed weights and bias
    std::vector<float> w_nchwc;       // [OCb, IC, block] for 1×1, or [OCb, IC, kH, kW, block] for general
    std::vector<float> bias_nchwc;    // [OC_padded] zero-padded to block boundary
    std::vector<float> w_dw_nchwc;    // DW Conv: [Cb, kH, kW, block] packed depthwise weights
    std::vector<float> bias_dw_nchwc; // DW Conv: [C_padded] zero-padded bias
    // Winograd F(4,3) NCHWc: pre-transformed filters [36][OCb][iC][block]
    // Lazy-packed in exec() only when a chain actually dispatches the Wino
    // NCHWc kernel (matches w_nchwc lazy-pack pattern).
    std::vector<float> w_wino_nchwc;
    size_t ws_nchwc_reorder = 0;      // workspace for NCHW→NCHWc input reorder at boundaries
    // FP16 I/O: pre-converted float32 weights and bias (populated in reshape)
    float* w_f32 = nullptr;           // [M, kC, kH, kW] as float32
    float* bias_f32 = nullptr;        // [M] as float32
#ifdef NNR_ARCH_ARM64
    // FP16 native NCHW direct conv (has_neon_fp16): pre-packed FP16 weights
    // in the `conv_fp16_direct_neon.h` layout.  When populated, the FP16
    // trampoline picks the native path in preference to convert-to-FP32.
    std::vector<uint16_t> w_fp16_direct;
    // FP16 native NHWC direct conv: pre-packed FP16 weights in the
    // `conv_fp16_nhwc_direct_neon.h` layout. Selected at exec time when
    // x->format is NHWC; otherwise w_fp16_direct (NCHW) wins.
    std::vector<uint16_t> w_fp16_nhwc_direct;
    // FP16 native NHWC depthwise conv: repacked FP16 depthwise weights in
    // [kH*kW, C] layout for depthwise_fp16_nhwc_neon.
    std::vector<uint16_t> w_fp16_dw_nhwc;
#endif

    ~Conv_operator() {
        nnr_aligned_free(w_f32);
        nnr_aligned_free(bias_f32);
    }

    bool init() override {
        if (!(inputs.size() >= 2 && outputs.size() == 1))
            return false;
        int64_t* ints = nullptr;
        int i, l;
        auto_pad = string2enum(attribute(attr_key_t::auto_pad, "NOTSET"), NOTSET);
        group = attribute(attr_key_t::group, 1);
        int nk = attribute(attr_key_t::kernel_shape, ints);
        if (nk > 0) {
            kernels.resize(nk);
            for (i = 0; i < nk; ++i) kernels[i] = ints[i];
            dilations.resize(nk);
            l = attribute(attr_key_t::dilations, ints);
            for (i = 0; i < l; ++i) dilations[i] = ints[i];
            for (; i < nk; ++i) dilations[i] = 1;
            pads.resize(nk * 2);
            l = attribute(attr_key_t::pads, ints);
            for (i = 0; i < l; ++i) pads[i] = ints[i];
            for (; i < nk * 2; ++i) pads[i] = 0;
            strides.resize(nk);
            l = attribute(attr_key_t::strides, ints);
            for (i = 0; i < l; ++i) strides[i] = ints[i];
            for (; i < nk; ++i) strides[i] = 1;
        }
        return true;
    }

#include "Conv_reshape.h"

    // Compute tile_h for im2col spatial tiling.
    // Returns oH (no tiling) or a smaller value that caps the buffer to
    // the L2-derived im2col threshold.
    int im2col_tile_h() const {
        if (ws_CHW <= 0 || ws_oW <= 0) return 0;
        int oH = ws_spatial / ws_oW;
        size_t full_bytes = (size_t)ws_CHW * ws_spatial * sizeof(float);
        const size_t cap = im2col_tile_bytes();
        if (full_bytes <= cap) return oH;
        return std::max(1, (int)(cap / ((size_t)ws_CHW * ws_oW * sizeof(float))));
    }

    size_t workspace_size() const override {
        // FP16/BF16: need float32 scratch for X, Y, and im2col buffer
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT16
            || inputs[0]->type == NNR_DATA_TYPE_BFLOAT16) {
            size_t ws = inputs[0]->ndata * sizeof(float);   // X_f32
            ws += outputs[0]->ndata * sizeof(float);        // Y_f32
            if (ws_CHW > 0) {
                int tile_h = im2col_tile_h();
                int tile_spatial = tile_h * ws_oW;
                ws += (size_t)ws_CHW * tile_spatial * sizeof(float);  // im2col
            }
            return ws;
        }

        size_t ws = 0;
        if (ws_CHW > 0) {
            size_t elem = data_type_sizeof(inputs[0]->type);
            int tile_h = im2col_tile_h();
            int tile_spatial = tile_h * ws_oW;
            ws = (size_t)ws_CHW * tile_spatial * elem;
            // NCHW tiled path uses an MM × tile_spatial scatter buffer.
            // The non-strip exec only needs it when tile_h < oH (multiple
            // tiles), but the strip exec (Conv_exec_entry.h) always uses
            // it because per-channel strip pitch differs from tile_spatial.
            // Reserve unconditionally so a scrolling conv never overruns
            // workspace into adjacent allocations during prune_segments'
            // trial run (which corrupted adjacent weights and gave flaky
            // path_tests/scroll_chain_*conv outputs).
            int MM = inputs[1]->dims[0] / group;
            ws += (size_t)MM * tile_spatial * elem;
            // NHWC general conv with groups > 1 needs extra scatter buffer
            if (!w_gemm_nhwc.empty() && group > 1) {
                int MM = inputs[1]->dims[0] / group;
                ws += (size_t)tile_spatial * MM * sizeof(float);
            }
            // General NHWC conv at NCHW boundary: add per-batch reorder buffer
            if (ws_nhwc_reorder > 0) {
                int N = std::max(1, (int)inputs[0]->dims[0]);
                ws += ws_nhwc_reorder / N;
            }
        } else {
            ws = ws_nhwc_reorder;
        }
        if (ws_winograd > 0)
            ws = std::max(ws, (size_t)ws_winograd * sizeof(float));
        // NCHWc reorder workspace: needed at NCHW→NCHWc chain boundary
        if (ws_nchwc_reorder > 0)
            ws = std::max(ws, ws_nchwc_reorder);
        // Last-layer direct conv: pre-padded input buffer
        if (ws_last_layer > 0)
            ws = std::max(ws, ws_last_layer);
        return ws;
    }

    int64_t num_ops() const override {
        const tensor_t* w = inputs[1];
        const tensor_t* y = outputs[0];
        if (y->ndim != 4) return 0;
        // 2 * N * M * oH * oW * (C/group) * kH * kW
        return (int64_t)2 * y->dims[0] * w->dims[0] * y->dims[2] * y->dims[3]
            * w->dims[1] * w->dims[2] * w->dims[3];
    }

    small_vector<op_cost_t, 8> estimate_costs(bool input_nhwc) const override {
        small_vector<op_cost_t, 8> out;
        if (inputs.size() < 2 || !inputs[1]) return out;
        auto* x = inputs[0];
        auto* w = inputs[1];
        auto* y = outputs.empty() ? nullptr : outputs[0];
        if (!x || !y || x->ndim != 4 || w->ndim != 4 || y->ndim != 4) return out;

        int C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int M = w->dims[0], kC = w->dims[1], kH = w->dims[2], kW = w->dims[3];
        int oH = y->dims[2], oW = y->dims[3];
        int groups = (kC > 0) ? C / kC : 1;
        int K = kH * kW * kC;
        int spatial = oH * oW;

        float input_bytes  = (float)C * H * W * 4;
        float weight_bytes = (float)M * K * 4;
        float output_bytes = (float)M * oH * oW * 4;
        float flops = 2.0f * M * K * spatial;

        // Depthwise
        if (groups == C) {
            float dw_weight = (float)C * kH * kW * 4;
            float dw_flops = 2.0f * C * oH * oW * kH * kW;
            // NCHW depthwise
            op_cost_t nchw{};
            nchw.layout = memory_layout_t::NCHW;
            nchw.read_bytes = input_bytes + dw_weight;
            nchw.write_bytes = output_bytes;
            nchw.read_sequential = 0.9f;
            nchw.compute_ops = dw_flops;
            nchw.working_set_bytes = (float)kH * W * 4;
            nchw.max_threads = C;
            nchw.scrollable = true;
            nchw.fusable_post_op = true;
            out.push_back(nchw);
            // NHWC depthwise
            op_cost_t nhwc{};
            nhwc.layout = memory_layout_t::NHWC;
            nhwc.read_bytes = input_bytes + dw_weight;
            nhwc.write_bytes = output_bytes;
            nhwc.read_sequential = nhwc_patch_util(C);
            nhwc.compute_ops = dw_flops;
            nhwc.working_set_bytes = (float)kH * C * 4;
            nhwc.max_threads = oH * oW;
            nhwc.fusable_post_op = true;
            out.push_back(nhwc);
            // NCHWc depthwise: C%16==0, no dilation
            bool dw_d1 = dilations.empty() || (dilations.size() >= 2 && dilations[0] == 1 && dilations[1] == 1);
            if (C % 16 == 0 && C >= 16 && dw_d1) {
                op_cost_t blk{};
                blk.layout = memory_layout_t::BLOCKED_16;
                blk.read_bytes = input_bytes + dw_weight;
                blk.write_bytes = output_bytes;
                blk.read_sequential = 0.95f;
                blk.compute_ops = dw_flops;
                blk.working_set_bytes = (float)kH * 16 * W * 4;  // one 16-channel block
                blk.max_threads = C / 16 * oH;
                blk.scrollable = true;
                blk.fusable_post_op = true;
                out.push_back(blk);
            }
            return out;
        }

        bool s1 = strides.empty() || (strides.size() >= 2 && strides[0] == 1 && strides[1] == 1);
        bool d1 = dilations.empty() || (dilations.size() >= 2 && dilations[0] == 1 && dilations[1] == 1);

        // Winograd eligibility
        bool winograd = false;
        int wino_tiles = 0;
        if (kH == 3 && kW == 3 && groups == 1 && s1 && d1) {
            wino_tiles = ((oH + 3) / 4) * ((oW + 3) / 4);
            winograd = wino_tiles >= WINOGRAD_MIN_TILES;
        }

        // --- NCHW candidates ---

        if (kH == 1 && kW == 1) {
            // 1x1: direct GEMM, no im2col
            op_cost_t c{};
            c.layout = memory_layout_t::NCHW;
            c.read_bytes = input_bytes + weight_bytes;
            c.write_bytes = output_bytes;
            c.read_sequential = 1.0f;
            c.compute_ops = flops;
            c.working_set_bytes = weight_bytes;  // B-panel
            c.max_threads = std::max(1, M / 8);  // MR=8 row groups
            c.scrollable = true;
            c.fusable_post_op = true;
            out.push_back(c);
        } else {
            if (winograd) {
                // NCHW Winograd: 36 batched GEMMs
                op_cost_t c{};
                c.layout = memory_layout_t::NCHW;
                c.read_bytes = input_bytes + weight_bytes;
                c.write_bytes = output_bytes;
                c.read_sequential = 0.9f;
                c.compute_ops = 2.0f * 36 * M * kC * wino_tiles;
                c.working_set_bytes = (float)wino_tiles * kC * 4;  // transform tile
                c.max_threads = wino_tiles;
                c.fusable_post_op = true;
                // Winograd is not scrollable
                out.push_back(c);
            }
            // NCHW im2col + GEMM (always available as fallback)
            {
                float im2col_buf = (float)K * spatial * 4 * groups;
                float input_seq = nchw_stride_util(H, W);
                float total_read = input_bytes + im2col_buf + weight_bytes;
                // effective_read = input/seq + im2col + weight (sequential parts at face value)
                float eff_read = input_bytes / input_seq + im2col_buf + weight_bytes;
                op_cost_t c{};
                c.layout = memory_layout_t::NCHW;
                c.read_bytes = total_read;
                c.write_bytes = im2col_buf + output_bytes;
                c.read_sequential = total_read / eff_read;
                c.compute_ops = flops;
                c.working_set_bytes = std::min(im2col_buf, (float)im2col_tile_bytes()) + weight_bytes;
                c.max_threads = std::max(1, M / 8);
                c.scrollable = (kH <= 3);  // exec_strip supports kH <= 3
                c.fusable_post_op = true;
                out.push_back(c);
            }
        }

        // --- NHWC candidates ---

        if (kH == 1 && kW == 1) {
            op_cost_t c{};
            c.layout = memory_layout_t::NHWC;
            c.read_bytes = input_bytes + weight_bytes;
            c.write_bytes = output_bytes;
            c.read_sequential = 1.0f;
            c.compute_ops = flops;
            c.working_set_bytes = weight_bytes;
            c.max_threads = std::max(1, M / 8);
            c.fusable_post_op = true;
            out.push_back(c);
        } else {
            if (winograd) {
                // NHWC Winograd
                op_cost_t c{};
                c.layout = memory_layout_t::NHWC;
                c.read_bytes = input_bytes + weight_bytes;
                c.write_bytes = output_bytes;
                c.read_sequential = 0.9f;
                c.compute_ops = 2.0f * 36 * M * kC * wino_tiles;
                c.working_set_bytes = (float)wino_tiles * kC * 4;
                c.max_threads = wino_tiles;
                c.fusable_post_op = true;
                out.push_back(c);
            }
            // NHWC im2col + GEMM
            {
                float im2col_buf = (float)K * spatial * 4 * groups;
                float input_seq = input_nhwc ? nhwc_patch_util(C) : nchw_stride_util(H, W);
                float total_read = input_bytes + im2col_buf + weight_bytes;
                float eff_read = input_bytes / input_seq + im2col_buf + weight_bytes;
                // Apply GEMM throughput penalty for NHWC
                float gemm_penalty_bytes = (im2col_buf + weight_bytes + output_bytes) * (GEMM_NHWC_PENALTY - 1.0f);
                op_cost_t c{};
                c.layout = memory_layout_t::NHWC;
                c.read_bytes = total_read;
                c.write_bytes = im2col_buf + output_bytes + gemm_penalty_bytes;
                c.read_sequential = total_read / eff_read;
                c.compute_ops = flops;
                c.working_set_bytes = std::min(im2col_buf, (float)im2col_tile_bytes()) + weight_bytes;
                c.max_threads = std::max(1, M / 8);
                c.fusable_post_op = true;
                out.push_back(c);
            }
        }

        // --- NCHWc (BLOCKED_16) candidate ---
        if (groups == 1 && C % 16 == 0 && M % 16 == 0) {
            op_cost_t c{};
            c.layout = memory_layout_t::BLOCKED_16;
            c.read_bytes = input_bytes + weight_bytes;
            c.write_bytes = output_bytes;
            c.read_sequential = 0.95f;
            c.compute_ops = flops;
            c.working_set_bytes = (float)(C / 16) * kH * kW * 16 * 4;  // IC blocks
            c.max_threads = oH;
            c.scrollable = true;
            c.fusable_post_op = true;
            out.push_back(c);
        }

        return out;
    }

    // NHWC channel-axis Concat alias: this Conv can write into a parent
    // buffer with a wider LDC than its local channel count. M2 covered NHWC
    // 1×1; M3 widens to general 2-D Conv (im2col + GEMM via dgemm_generic_ldc).
    // group>1 still excluded — the scatter path (line ~744 in Conv_exec.h)
    // doesn't yet honor LDC.
    bool supports_strided_output(memory_layout_t format) const override {
        if (format != memory_layout_t::NHWC) return false;
        if (outputs.empty() || !outputs[0]) return false;
        if (outputs[0]->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (group != 1) return false;
        if (kernels.size() != 2) return false;
        return true;
    }

    // Scalar cost formula for NCHW / NHWC / BLOCKED layout comparison.
    // BLOCKED uses the same structural model as NCHW (channel-major outer
    // layout, spatial inner) but replaces NCHW's stride-W cache-line waste
    // with c-block packing — every channel-block load is one SIMD register.
    // Eligible chains have C aligned to NATIVE_BLOCK so block_simd_util=1.0.
    float layout_cost(memory_layout_t layout, bool input_nhwc) const override {
        if (inputs.size() < 2 || !inputs[1]) return 0;
        auto* x = inputs[0];
        auto* w = inputs[1];
        auto* y = outputs.empty() ? nullptr : outputs[0];
        if (!x || !y || x->ndim != 4 || w->ndim != 4 || y->ndim != 4) return 0;

        int C = x->dims[1], H = x->dims[2], W = x->dims[3];
        int M = w->dims[0], kC = w->dims[1], kH = w->dims[2], kW = w->dims[3];
        int oH = y->dims[2], oW = y->dims[3];
        int groups = (kC > 0) ? C / kC : 1;
        int K = kH * kW * kC;
        int spatial = oH * oW;
        bool nhwc = (layout == memory_layout_t::NHWC);
        bool blocked = (layout == memory_layout_t::BLOCKED_16 ||
                        layout == memory_layout_t::BLOCKED_8);

        float input_bytes  = (float)C * H * W * 4;
        float weight_bytes = (float)M * K * 4;
        float output_bytes = (float)M * oH * oW * 4;

        if (blocked) {
            int block = (layout == memory_layout_t::BLOCKED_16) ? 16 : 8;
            float bu = block_simd_util(C, block);
            // DW NCHWc has the same per-op cost as DW NCHW
            // (see assign_blocked_layouts.cpp:253). bu doesn't apply: DW
            // kernels are per-channel and don't pack channels into SIMD lanes.
            if (groups == C) {
                float dw_bytes = input_bytes + output_bytes + (float)C * kH * kW * 4;
                return dw_bytes * 1.1f;
            }
            // 1×1: c-block-packed GEMM (channels are GEMM K).
            if (kH == 1 && kW == 1) {
                return weight_bytes + input_bytes / bu + output_bytes;
            }
            // 3×3 stride-1 dilation-1: NCHWc Winograd path (no im2col).
            bool wino = false;
            if (kH == 3 && kW == 3 && groups == 1) {
                bool s1 = strides.empty() || (strides.size() >= 2 && strides[0] == 1 && strides[1] == 1);
                bool d1 = dilations.empty() || (dilations.size() >= 2 && dilations[0] == 1 && dilations[1] == 1);
                int num_tiles = ((oH + 3) / 4) * ((oW + 3) / 4);
                wino = s1 && d1 && num_tiles >= WINOGRAD_MIN_TILES;
            }
            if (wino) return input_bytes / bu + output_bytes + weight_bytes;
            // General: c-block im2col + packed GEMM. Same im2col bandwidth
            // as NCHW but the spatial reads avoid the stride-W cache-line waste.
            float im2col_bytes = (float)K * spatial * 4 * groups;
            float im2col_eff = input_bytes / bu + im2col_bytes;
            float gemm_eff = weight_bytes + im2col_bytes + output_bytes;
            return im2col_eff + gemm_eff;
        }

        if (groups == C) {
            float dw_bytes = input_bytes + output_bytes + (float)C * kH * kW * 4;
            if (nhwc) return dw_bytes * 1.1f / nhwc_patch_util(C);
            return dw_bytes * 1.1f;
        }
        if (kH == 1 && kW == 1) {
            float gemm_bytes = weight_bytes + input_bytes + output_bytes;
            return nhwc ? gemm_bytes * GEMM_NHWC_1x1_PENALTY : gemm_bytes;
        }
        bool winograd = false;
        if (kH == 3 && kW == 3 && groups == 1) {
            bool s1 = strides.empty() || (strides.size() >= 2 && strides[0] == 1 && strides[1] == 1);
            bool d1 = dilations.empty() || (dilations.size() >= 2 && dilations[0] == 1 && dilations[1] == 1);
            int num_tiles = ((oH + 3) / 4) * ((oW + 3) / 4);
            winograd = s1 && d1 && num_tiles >= WINOGRAD_MIN_TILES;
        }
        if (winograd && !nhwc) return input_bytes + output_bytes + weight_bytes;
        if (winograd && nhwc)  return (input_bytes + output_bytes + weight_bytes) * WINO_NHWC_PENALTY;

        float im2col_bytes = (float)K * spatial * 4 * groups;
        float im2col_eff;
        if (nhwc) {
            float read_util = input_nhwc ? nhwc_patch_util(C) : nchw_stride_util(H, W);
            im2col_eff = input_bytes / read_util + im2col_bytes;
        } else {
            im2col_eff = input_bytes / nchw_stride_util(H, W) + im2col_bytes;
        }
        float gemm_eff = weight_bytes + im2col_bytes + output_bytes;
        if (nhwc) gemm_eff *= GEMM_NHWC_PENALTY;
        return im2col_eff + gemm_eff;
    }

    // Depthwise convolution (included from separate header for readability)
#include "Conv_depthwise.h"

#include "Conv_exec.h"



    scroll_info_t scroll_info() const override {
        if (kernels.size() != 2) return {};
        if (!inputs[0] || !inputs[1]) return {};
        if (inputs[0]->ndim != 4) return {};
        if (inputs[0]->type != NNR_DATA_TYPE_FLOAT32) return {};
#ifdef NNR_EXPLICIT_REORDERS
        // T3 follow-up A: exec_strip handles three layout combos:
        //   - NCHW/NCHW (whole-tensor uniform)
        //   - BLOCKED/BLOCKED (DW/1×1/Wino/general NCHWc strip kernels)
        //   - NCHW/BLOCKED (general K×K boundary Conv with on-entry strip
        //     reorder of input — covers residual-path 1×1 s=2 downsample
        //     whose input tensor is shared with a NCHW-uniform main-path
        //     Conv and never gets an upstream Reorder)
        // BLOCKED/NCHW (terminal exit) and grouped non-DW + dilated Convs
        // opt out — kernels are whole-tensor only.
        {
            const auto in_lo  = inputs[0]->declared_layout;
            const auto out_lo = outputs[0] ? outputs[0]->declared_layout : in_lo;
            const bool is_depthwise = (group == (int)inputs[0]->dims[1]
                                       && inputs[1]->dims[1] == 1);
            const bool nchw_uniform    = (in_lo == memory_layout_t::NCHW
                                          && out_lo == memory_layout_t::NCHW);
            const bool blocked_uniform = (in_lo == NATIVE_BLOCKED_FMT
                                          && out_lo == NATIVE_BLOCKED_FMT
                                          && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW);
            // NCHW→BLOCKED: only the general K×K NCHWc strip path covers it
            // (1×1 s=1 fast path requires BLOCKED-in; DW requires BLOCKED-in).
            // Restrict to the general path's shape envelope so exec_strip
            // never bails mid-strip on this combo.
            const bool mixed_to_blocked = (in_lo == memory_layout_t::NCHW
                                           && out_lo == NATIVE_BLOCKED_FMT
                                           && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW
                                           && group == 1
                                           && (int)inputs[0]->dims[1] % NATIVE_BLOCK == 0
                                           && (int)inputs[1]->dims[0] % NATIVE_BLOCK == 0);
            if (!nchw_uniform && !blocked_uniform && !mixed_to_blocked) return {};
            if (blocked_uniform || mixed_to_blocked) {
                if (!is_depthwise && group != 1) return {};
                if (!dilations.empty()
                    && (dilations[0] != 1 || dilations[1] != 1))
                    return {};
            }
        }
#endif
        // Skip grouped Conv (except depthwise)
        int iC = inputs[0]->dims[1];
        int kC = inputs[1]->dims[1];
        bool is_depthwise = (group == iC && kC == 1);
        if (group > 1 && !is_depthwise) return {};
        // Skip NHWC Conv (exec_strip only supports NCHW im2col path)
        if (!is_depthwise && outputs[0] && outputs[0]->format == memory_layout_t::NHWC) return {};
        // Skip large-kernel Conv (halo overhead dominates, workspace may be undersized)
        if (!is_depthwise && kernels[0] > 3) return {};
        int kH = kernels[0];
        int dH = dilations[0];
        int sH = strides[0];
        int pad_top = cpads[0];
        int eff_kH = (kH - 1) * dH + 1;
        return {
            .scrollable = true,
            .halo_top = pad_top,
            .halo_bottom = eff_kH - 1 - pad_top,
            .stride_h = sH,
        };
    }

    void depthwise_strip_channel(
        const float* xc, const float* wc, float* yc, float bv,
        int kH, int kW, int iH, int iW, int oW,
        int sH, int sW, int dH, int dW, int pH, int pW,
        int out_row_start, int out_end, int spatial_strip,
        int tensor_offset = 0)
    {
#ifdef NNR_ARCH_X64
        if (has_avx512()
            && depthwise_strip_channel_avx512(xc, wc, yc, bv,
                kH, kW, iH, iW, oW, sH, sW, dH, dW, pH, pW,
                out_row_start, out_end, spatial_strip,
                post_fn, fused_op, tensor_offset))
            return;
#elifdef NNR_ARCH_ARM64
        if (dH == 1 && dW == 1) {
            nnr::conv_direct_strip_neon(yc, xc, wc, bv,
                kH, kW, iH, iW, oW, sH, sW, pH, pW, dH, dW,
                out_row_start, out_end);
            if (post_fn) post_fn(yc, 1, spatial_strip, spatial_strip, fused_op, nullptr, tensor_offset);
            return;
        }
#endif
        // Scalar fallback (generic stride/dilation)
        for (int oh = out_row_start; oh < out_end; ++oh) {
            int ih0 = oh * sH - pH;
            for (int ow = 0; ow < oW; ++ow) {
                int iw0 = ow * sW - pW;
                float sum = bv;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = ih0 + kh * dH;
                    if (ih < 0 || ih >= iH) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = iw0 + kw * dW;
                        if (iw < 0 || iw >= iW) continue;
                        sum += xc[ih * iW + iw] * wc[kh * kW + kw];
                    }
                }
                yc[(oh - out_row_start) * oW + ow] = sum;
            }
        }
        if (post_fn) post_fn(yc, 1, spatial_strip, spatial_strip, fused_op, nullptr, tensor_offset);
    }
#include "Conv_exec_entry.h"
};

} // namespace

operator_t* resolver_default_op_Conv(int opset, pool_t& pool) { return pool_new<Conv_operator>(pool); }

} // namespace nnr
