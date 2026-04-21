// WebGPU Einsum. Covers the common case: 2 inputs, no ellipsis, no
// repeated indices on either operand, 1D-4D output, up to 8 unique
// labels total.
//
// Anything outside that scope returns false from init/reshape and the
// registry falls back to CPU. For realistic transformer Einsums ("ij,jk->ik",
// "bij,bjk->bik", "bid,bjd->bij", "bijh,bikh->bjkh", etc.) this covers the
// cases real ONNX exports produce.
//
// Algorithm at reshape time:
//   1. Parse the equation into input_labels[2][ndim] and output_labels[].
//   2. Compute label_sizes[L] by matching each input dim to a label.
//   3. Classify each label as output (appears in output) or contraction
//      (doesn't). Output shape = label_sizes for output labels in order.
//   4. Build per-input stride tables: stride[label] = input's element
//      stride along the dim that uses that label, else 0 (i.e., label
//      not present in this input → no contribution, input broadcast
//      across that label).
//   5. Runtime-compile a WGSL shader that:
//        - Unflattens the output index into label coords.
//        - Loops over contraction-label products.
//        - Computes each input's flat offset via dot(coord, stride).
//        - Multiply-accumulates and writes.
//
// One thread per output slot; inner loop iterates the Cartesian product
// of contraction labels. O(N_output × N_contract) ops — same complexity
// class as a naive matmul, competitive for small/medium shapes.

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

constexpr int MAX_LABELS = 8;
constexpr uint32_t WG = 64;

struct Einsum_op_webgpu : public operator_t {
    // Parsed at init.
    int n_inputs_parsed = 0;
    int num_labels      = 0;
    int output_ndim     = 0;

    // For each of up to 2 inputs, the label index at each of its dims.
    int  input_labels[2][MAX_LABELS] = {};
    int  input_ndim[2]               = {};
    int  output_labels[MAX_LABELS]   = {};

    // Populated at reshape time.
    uint32_t label_sizes[MAX_LABELS] = {};
    uint32_t output_label_idx[MAX_LABELS] = {};    // label index for each output dim
    uint32_t contract_label_idx[MAX_LABELS] = {};  // label index for each contraction axis
    int n_contract = 0;
    uint32_t contract_prod = 0;
    uint32_t output_total  = 0;
    // Per-label element stride in each input. 0 if the input doesn't use
    // that label, else the product of strides for dims after this label's
    // dim in that input.
    uint32_t a_stride_for_label[MAX_LABELS] = {};
    uint32_t b_stride_for_label[MAX_LABELS] = {};

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;
    bool built = false;

    // Cached BindGroup. Tensor-backed slots: [A, B, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[3] = {};
    // Remember the contraction+output label layout used when we compiled
    // the pipeline so that reshape can detect a label-layout change and
    // rebuild. Most models don't change their einsum equation across
    // runs, so this only fires on the very first call.

    // Parse the equation string into label arrays. Returns false on
    // unsupported patterns (ellipsis, 3+ inputs, repeated labels on a
    // single operand).
    bool parse_equation(std::string_view eq) {
        char eq_buf[128] = {};
        int eq_len = 0;
        for (char c : eq) {
            if (c == ' ') continue;
            if (eq_len >= 127) return false;
            eq_buf[eq_len++] = c;
        }
        if (strstr(eq_buf, "...")) return false;   // ellipsis → CPU

        const char* arrow = strstr(eq_buf, "->");
        const char* inputs_end = arrow ? arrow : eq_buf + eq_len;

        // Split on ','.
        const char* p = eq_buf;
        n_inputs_parsed = 0;
        while (p < inputs_end) {
            if (n_inputs_parsed >= 2) return false;
            const char* comma = p;
            while (comma < inputs_end && *comma != ',') ++comma;
            int len = (int)(comma - p);
            if (len > MAX_LABELS) return false;
            input_ndim[n_inputs_parsed] = len;
            for (int i = 0; i < len; ++i) input_labels[n_inputs_parsed][i] = (int)(unsigned char)p[i];
            ++n_inputs_parsed;
            p = (comma < inputs_end) ? comma + 1 : inputs_end;
        }
        if (n_inputs_parsed != 2) return false;     // scope: 2 inputs only

        // No repeated labels within one operand.
        for (int i = 0; i < 2; ++i) {
            for (int a = 0; a < input_ndim[i]; ++a)
                for (int b = a + 1; b < input_ndim[i]; ++b)
                    if (input_labels[i][a] == input_labels[i][b]) return false;
        }

        // Output labels: either explicit after "->" or implicit (all
        // labels appearing in exactly one input, in sorted order).
        if (arrow) {
            p = arrow + 2;
            int len = (int)(eq_buf + eq_len - p);
            if (len > MAX_LABELS) return false;
            output_ndim = len;
            for (int i = 0; i < len; ++i) output_labels[i] = (int)(unsigned char)p[i];
        } else {
            // Implicit: labels in only one operand, alphabetical.
            int counts[128] = {};
            for (int i = 0; i < 2; ++i)
                for (int a = 0; a < input_ndim[i]; ++a)
                    counts[input_labels[i][a]]++;
            output_ndim = 0;
            for (int c = 0; c < 128; ++c) {
                if (counts[c] == 1) {
                    if (output_ndim >= MAX_LABELS) return false;
                    output_labels[output_ndim++] = c;
                }
            }
        }

        // Re-assign labels to dense 0..num_labels-1 indices (original
        // storage was ASCII chars). Collect all unique labels.
        int char_to_idx[128];
        for (int i = 0; i < 128; ++i) char_to_idx[i] = -1;
        num_labels = 0;
        auto remap = [&](int c) -> int {
            if (char_to_idx[c] == -1) {
                if (num_labels >= MAX_LABELS) return -1;
                char_to_idx[c] = num_labels++;
            }
            return char_to_idx[c];
        };
        for (int i = 0; i < 2; ++i)
            for (int a = 0; a < input_ndim[i]; ++a) {
                int r = remap(input_labels[i][a]);
                if (r < 0) return false;
                input_labels[i][a] = r;
            }
        for (int a = 0; a < output_ndim; ++a) {
            int r = remap(output_labels[a]);
            if (r < 0) return false;
            output_labels[a] = r;
        }
        // Every output label must appear in at least one input.
        for (int a = 0; a < output_ndim; ++a) {
            int lbl = output_labels[a];
            bool found = false;
            for (int i = 0; i < 2 && !found; ++i)
                for (int k = 0; k < input_ndim[i] && !found; ++k)
                    if (input_labels[i][k] == lbl) found = true;
            if (!found) return false;
        }
        return true;
    }

    bool init() override {
        if (!webgpu::device_ready()) return false;
        if (outputs.size() != 1) return false;
        if (inputs.size() != 2) return false;   // scope: 2 inputs

        std::string_view eq = attribute(attr_key_t::equation, "");
        if (eq.empty()) return false;
        if (!parse_equation(eq)) return false;
        if (input_ndim[0] != (int)inputs[0]->ndim) return false;
        if (input_ndim[1] != (int)inputs[1]->ndim) return false;

        auto& dev = webgpu::get_device();
        wgpu::BindGroupLayoutEntry e[4] = {};
        e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
        e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
        e[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::Storage;
        e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
        e[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 4; bgld.entries = e;
        bgl = dev.device.CreateBindGroupLayout(&bgld);

        wgpu::BufferDescriptor md = {};
        md.size  = 256;   // header + per-label arrays (fits MAX_LABELS=8)
        md.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        meta_buf = dev.device.CreateBuffer(&md);
        return true;
    }

    bool build_pipeline() {
        // Generate a shader hardcoded for (output_ndim, n_contract). The
        // unflatten-output and contraction loops have compile-time bounds,
        // which lets Tint unroll and avoids runtime loop overhead.
        auto& dev = webgpu::get_device();
        std::string src;
        src =
            "struct Meta {\n"
            "  total            : u32,\n"
            "  _pad             : u32,\n"
            "  _pad2            : u32,\n"
            "  _pad3            : u32,\n"
            "  out_dims_lo      : vec4<u32>,\n"
            "  out_dims_hi      : vec4<u32>,\n"
            "  contract_dims_lo : vec4<u32>,\n"
            "  contract_dims_hi : vec4<u32>,\n"
            "  a_stride_lo      : vec4<u32>,\n"
            "  a_stride_hi      : vec4<u32>,\n"
            "  b_stride_lo      : vec4<u32>,\n"
            "  b_stride_hi      : vec4<u32>,\n"
            "  out_lbl_lo       : vec4<u32>,\n"
            "  out_lbl_hi       : vec4<u32>,\n"
            "  ctr_lbl_lo       : vec4<u32>,\n"
            "  ctr_lbl_hi       : vec4<u32>,\n"
            "};\n"
            "@group(0) @binding(0) var<storage, read>       A  : array<f32>;\n"
            "@group(0) @binding(1) var<storage, read>       B  : array<f32>;\n"
            "@group(0) @binding(2) var<storage, read_write> Y  : array<f32>;\n"
            "@group(0) @binding(3) var<storage, read>       md : Meta;\n"
            "fn get_out_dim(i : u32) -> u32 { if (i < 4u) { return md.out_dims_lo[i]; } return md.out_dims_hi[i - 4u]; }\n"
            "fn get_ctr_dim(i : u32) -> u32 { if (i < 4u) { return md.contract_dims_lo[i]; } return md.contract_dims_hi[i - 4u]; }\n"
            "fn get_a_stride(i : u32) -> u32 { if (i < 4u) { return md.a_stride_lo[i]; } return md.a_stride_hi[i - 4u]; }\n"
            "fn get_b_stride(i : u32) -> u32 { if (i < 4u) { return md.b_stride_lo[i]; } return md.b_stride_hi[i - 4u]; }\n"
            "fn get_out_lbl(i : u32) -> u32 { if (i < 4u) { return md.out_lbl_lo[i]; } return md.out_lbl_hi[i - 4u]; }\n"
            "fn get_ctr_lbl(i : u32) -> u32 { if (i < 4u) { return md.ctr_lbl_lo[i]; } return md.ctr_lbl_hi[i - 4u]; }\n"
            "@compute @workgroup_size(64)\n"
            "fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
            "  let o = gid.x;\n"
            "  if (o >= md.total) { return; }\n"
            // Unflatten o into per-label coords using output_ndim dims.
            "  var coord : array<u32, 8>;\n"
            "  for (var i : u32 = 0u; i < 8u; i = i + 1u) { coord[i] = 0u; }\n"
            "  var rem : u32 = o;\n";
        src += "  for (var d : i32 = i32(";
        src += std::to_string(output_ndim);
        src += ") - 1; d >= 0; d = d - 1) {\n";
        src += "    let dim = get_out_dim(u32(d));\n"
               "    let lbl = get_out_lbl(u32(d));\n"
               "    coord[lbl] = rem % dim;\n"
               "    rem = rem / dim;\n"
               "  }\n";
        // Base offset contribution from output labels.
        src += "  var a_base : u32 = 0u;\n"
               "  var b_base : u32 = 0u;\n"
               "  for (var d : u32 = 0u; d < ";
        src += std::to_string(output_ndim);
        src += "u; d = d + 1u) {\n";
        src += "    let lbl = get_out_lbl(d);\n"
               "    let c = coord[lbl];\n"
               "    a_base = a_base + c * get_a_stride(lbl);\n"
               "    b_base = b_base + c * get_b_stride(lbl);\n"
               "  }\n";
        // Contraction loop: nested over all contraction labels.
        src += "  var acc : f32 = 0.0;\n"
               "  var ctr_coord : array<u32, 8>;\n"
               "  for (var i : u32 = 0u; i < 8u; i = i + 1u) { ctr_coord[i] = 0u; }\n";
        // Linear counter trick: flatten the contraction space and
        // unflatten per iter — keeps the shader small and WGSL-friendly
        // for any n_contract.
        src += "  let total_ctr = ";
        if (n_contract == 0) {
            src += "1u;\n";
        } else {
            for (int i = 0; i < n_contract; ++i) {
                if (i > 0) src += " * ";
                src += "get_ctr_dim(" + std::to_string(i) + "u)";
            }
            src += ";\n";
        }
        src += "  for (var t : u32 = 0u; t < total_ctr; t = t + 1u) {\n"
               "    var rem2 : u32 = t;\n";
        src += "    for (var i : i32 = i32(";
        src += std::to_string(n_contract);
        src += ") - 1; i >= 0; i = i - 1) {\n";
        src += "      let dim = get_ctr_dim(u32(i));\n"
               "      let lbl = get_ctr_lbl(u32(i));\n"
               "      ctr_coord[lbl] = rem2 % dim;\n"
               "      rem2 = rem2 / dim;\n"
               "    }\n";
        // Add contraction contribution to A / B offsets.
        src += "    var a_off : u32 = a_base;\n"
               "    var b_off : u32 = b_base;\n"
               "    for (var i : u32 = 0u; i < ";
        src += std::to_string(n_contract);
        src += "u; i = i + 1u) {\n";
        src += "      let lbl = get_ctr_lbl(i);\n"
               "      let c = ctr_coord[lbl];\n"
               "      a_off = a_off + c * get_a_stride(lbl);\n"
               "      b_off = b_off + c * get_b_stride(lbl);\n"
               "    }\n"
               "    acc = acc + A[a_off] * B[b_off];\n"
               "  }\n"
               "  Y[o] = acc;\n"
               "}\n";

        wgpu::ShaderSourceWGSL w = {};
        w.code = src.c_str();
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &w;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline = dev.device.CreateComputePipeline(&cpd);
        built = true;
        return true;
    }

    bool reshape() override {
        const tensor_t* A = inputs[0];
        const tensor_t* B = inputs[1];
        tensor_t*       Y = outputs[0];
        if (A->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (B->type != NNR_DATA_TYPE_FLOAT32) return false;

        // Map each label to its size. Check that labels appearing in
        // multiple inputs agree on the size.
        for (int i = 0; i < num_labels; ++i) label_sizes[i] = 0;
        auto register_label = [&](int lbl, int size) -> bool {
            if (label_sizes[lbl] == 0) label_sizes[lbl] = (uint32_t)size;
            else if (label_sizes[lbl] != (uint32_t)size) return false;
            return true;
        };
        const tensor_t* ins[2] = { A, B };
        for (int i = 0; i < 2; ++i)
            for (int d = 0; d < input_ndim[i]; ++d)
                if (!register_label(input_labels[i][d], ins[i]->dims[d])) return false;
        // Every label has a size now (at least one input had it).
        for (int l = 0; l < num_labels; ++l)
            if (label_sizes[l] == 0) return false;

        // Classify labels: output or contraction.
        bool is_output_label[MAX_LABELS] = {};
        for (int a = 0; a < output_ndim; ++a) is_output_label[output_labels[a]] = true;
        n_contract = 0;
        for (int l = 0; l < num_labels; ++l)
            if (!is_output_label[l]) contract_label_idx[n_contract++] = (uint32_t)l;

        // Output shape.
        int out_dims[MAX_LABELS] = {};
        for (int d = 0; d < output_ndim; ++d) {
            out_dims[d] = (int)label_sizes[output_labels[d]];
            output_label_idx[d] = (uint32_t)output_labels[d];
        }
        if (!Y->reshape(std::span<const int>(out_dims, output_ndim), NNR_DATA_TYPE_FLOAT32)) return false;

        // Per-input stride per label.
        auto fill_strides = [&](int idx, uint32_t dest[MAX_LABELS]) {
            for (int l = 0; l < MAX_LABELS; ++l) dest[l] = 0;
            uint32_t run = 1;
            for (int d = input_ndim[idx] - 1; d >= 0; --d) {
                dest[input_labels[idx][d]] = run;
                run *= (uint32_t)ins[idx]->dims[d];
            }
        };
        fill_strides(0, a_stride_for_label);
        fill_strides(1, b_stride_for_label);

        output_total = 1;
        for (int d = 0; d < output_ndim; ++d) output_total *= (uint32_t)out_dims[d];
        contract_prod = 1;
        for (int c = 0; c < n_contract; ++c) contract_prod *= label_sizes[contract_label_idx[c]];

        if (!built && !build_pipeline()) return false;

        webgpu::ensure_buffer(A, A->ndata * sizeof(float));
        webgpu::ensure_buffer(B, B->ndata * sizeof(float));
        webgpu::ensure_buffer(Y, Y->ndata * sizeof(float));

        // Meta layout:
        //   [0]  total (output element count)
        //   [16..48)    out_dims[8]         — size for each output dim
        //   [48..80)    contract_dims[8]    — size for each contract label in ctr_lbl order
        //   [80..112)   a_stride[8]         — per-label stride in A
        //   [112..144)  b_stride[8]         — per-label stride in B
        //   [144..176)  out_lbl[8]          — label index for each output dim
        //   [176..208)  ctr_lbl[8]          — label index for each contraction axis
        uint8_t buf[256] = {};
        auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(buf + off, &v, 4); };
        put_u32(0, output_total);
        for (int d = 0; d < output_ndim; ++d) put_u32(16 + d * 4, label_sizes[output_labels[d]]);
        for (int c = 0; c < n_contract; ++c) put_u32(48 + c * 4, label_sizes[contract_label_idx[c]]);
        for (int l = 0; l < num_labels; ++l) {
            put_u32(80  + l * 4, a_stride_for_label[l]);
            put_u32(112 + l * 4, b_stride_for_label[l]);
        }
        for (int d = 0; d < output_ndim; ++d) put_u32(144 + d * 4, (uint32_t)output_labels[d]);
        for (int c = 0; c < n_contract; ++c)  put_u32(176 + c * 4, contract_label_idx[c]);
        webgpu::get_device().queue.WriteBuffer(meta_buf, 0, buf, sizeof(buf));
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();
        webgpu::upload_if_needed(inputs[0]);
        webgpu::upload_if_needed(inputs[1]);

        auto* ra = webgpu::find(inputs[0]);
        auto* rb = webgpu::find(inputs[1]);
        auto* ry = webgpu::find(outputs[0]);

        uint32_t gen_a = webgpu::generation_of(inputs[0]);
        uint32_t gen_b = webgpu::generation_of(inputs[1]);
        uint32_t gen_y = webgpu::generation_of(outputs[0]);
        if (!cached_bg || gen_a != cached_gen[0] || gen_b != cached_gen[1] || gen_y != cached_gen[2]) {
            wgpu::BindGroupEntry be[4] = {};
            be[0].binding = 0; be[0].buffer = ra->buf;   be[0].offset = 0; be[0].size = ra->size;
            be[1].binding = 1; be[1].buffer = rb->buf;   be[1].offset = 0; be[1].size = rb->size;
            be[2].binding = 2; be[2].buffer = ry->buf;   be[2].offset = 0; be[2].size = ry->size;
            be[3].binding = 3; be[3].buffer = meta_buf;  be[3].offset = 0; be[3].size = 256;
            wgpu::BindGroupDescriptor bgd = {};
            bgd.layout = bgl; bgd.entryCount = 4; bgd.entries = be;
            cached_bg = dev.device.CreateBindGroup(&bgd);
            cached_gen[0] = gen_a;
            cached_gen[1] = gen_b;
            cached_gen[2] = gen_y;
        }

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, cached_bg);
        uint32_t groups = (output_total + WG - 1) / WG;
        pass.DispatchWorkgroups(groups, 1, 1);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_default_op_Einsum_webgpu(int, pool_t& pool) {
    return pool_new<Einsum_op_webgpu>(pool);
}

} // namespace nnr
