#include "nnr.h"
#include "aligned_alloc.h"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace nnr {

static bool is_inplace_safe(std::string_view op_type) {
    static constexpr std::string_view safe[] = {
        "Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh",
        "Exp", "Log", "Sqrt", "Abs", "Neg", "Ceil", "Floor",
        "Round", "Sign", "Erf", "Not", "Reciprocal",
        "BitwiseNot",
        "Selu", "Elu", "Celu", "HardSigmoid", "HardSwish",
        "Softplus", "Softsign", "ThresholdedRelu", "Gelu", "Mish",
        "Identity", "Dropout",
        "Clip",
        "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max",
        "And", "Or", "Xor", "BitShift",
        "BatchNormalization",
        // View ops: only change shape metadata, never data
        "Reshape", "Squeeze", "Unsqueeze", "Flatten",
    };
    for (auto s : safe)
        if (s == op_type) return true;
    return false;
}

void delete_data(void* data, data_type_t type);

void memory_planner_t::analyze(context_t* ctx)
{
    ctx_ = ctx;

    // Use the excluded set populated by the format loader
    const auto& excluded = ctx->memory_plan_excluded;

    // Exclude outputs of ops that call reinit() during exec()
    std::unordered_set<std::string_view> local_excluded(excluded.begin(), excluded.end());
    // Collect view ops: output shares input's memory (Slice with contiguous region, etc.)
    // view_source[output_tensor] = input_tensor for zero-copy view ops
    std::unordered_map<tensor_t*, tensor_t*> view_source;
    for (auto* node : ctx->graph->nodes) {
        const std::string_view op(node->op_type);
        if (op == "Constant" || op == "ConstantOfShape" ||
            op == "NonMaxSuppression" || op == "Resize")
            for (auto* out : node->outputs)
                if (out) local_excluded.insert(out->name);
        // Skip nodes: at runtime, outputs[0]->data is aliased to inputs[0]->data
        // (nnr.cpp::node_action_t::SKIP handler). Mirror that aliasing here so
        // downstream consumers (Concat alias detection, in-place inference)
        // can look through the skip chain to find the actual lifetime entry.
        if (node->skip && !node->inputs.empty() && !node->outputs.empty()
            && node->inputs[0] && node->outputs[0]) {
            auto* src = node->inputs[0];
            while (view_source.count(src)) src = view_source[src];
            view_source[node->outputs[0]] = src;
            local_excluded.insert(node->outputs[0]->name);
        }
        // View ops: output borrows input's memory — no pool allocation needed
        // Handles both single-output (Reshape, Unsqueeze, Slice) and
        // multi-output (Split) view ops.
        int vi = node->view_input_index();
        if (vi >= 0 && vi < (int)node->inputs.size() && !node->outputs.empty()) {
            auto* raw_src = node->inputs[vi];
            if (raw_src) {
                // Trace through chains of views to find the ultimate source
                auto* src = raw_src;
                while (view_source.count(src))
                    src = view_source[src];
                for (auto* dst : node->outputs) {
                    if (dst) {
                        view_source[dst] = src;
                        local_excluded.insert(dst->name);
                    }
                }
            }
        }
    }

    std::unordered_map<tensor_t*, int> lt_index;
    auto& nodes = ctx->graph->nodes;

    for (int i = 0; i < (int)nodes.size(); ++i) {
        auto* node = nodes[i];
        if (node->skip || node->folded) continue;  // skip fused/folded nodes
        for (auto* out : node->outputs) {
            if (!out || local_excluded.count(out->name)) continue;
            if (out->ndata == 0) continue;
            if (out->type == NNR_DATA_TYPE_STRING ||
                out->type == NNR_DATA_TYPE_SEQUENCE) continue;
            if (out->ndata <= 16) continue;  // skip tiny tensors — pooling overhead not worth it

            int idx = (int)lifetimes_.size();
            lt_index[out] = idx;
            lifetimes_.push_back({
                out, i, i,
                out->ndata * data_type_sizeof(out->type)
            });
        }
        for (auto* in : node->inputs) {
            // If this input is a view, extend the ultimate source's lifetime
            tensor_t* source = in;
            auto vit = view_source.find(in);
            if (vit != view_source.end())
                source = vit->second;
            if (source && lt_index.count(source))
                lifetimes_[lt_index[source]].last_consumer = i;
        }
    }

    num_intermediates = (int)lifetimes_.size();
    total_unpooled_bytes = 0;
    for (auto& lt : lifetimes_)
        total_unpooled_bytes += lt.size_bytes;
}

void memory_planner_t::plan()
{
    auto& nodes = ctx_->graph->nodes;

    std::unordered_map<tensor_t*, int> lt_index;
    for (int i = 0; i < (int)lifetimes_.size(); ++i)
        lt_index[lifetimes_[i].tensor] = i;
    // Skip-chain map: a node marked `skip` aliases its output[0] to its
    // input[0] at runtime (nnr.cpp::SKIP handler). Concat inputs that ride
    // through Conv→Relu(skip) reach Concat as Relu's output tensor — but
    // lt_index has only the producer Conv's output. Build a redirect so
    // alias eligibility looks up the underlying lifetime entry.
    std::unordered_map<tensor_t*, tensor_t*> skip_alias;
    for (auto* node : nodes) {
        if (!node->skip || node->inputs.empty() || node->outputs.empty()) continue;
        if (!node->inputs[0] || !node->outputs[0]) continue;
        auto* src = node->inputs[0];
        auto sit = skip_alias.find(src);
        while (sit != skip_alias.end()) { src = sit->second; sit = skip_alias.find(src); }
        skip_alias[node->outputs[0]] = src;
    }
    auto resolve_skip = [&](tensor_t* t) -> tensor_t* {
        while (t) {
            if (lt_index.count(t)) return t;
            auto sit = skip_alias.find(t);
            if (sit == skip_alias.end()) return t;
            t = sit->second;
        }
        return t;
    };

    // Phase 1: In-place detection
    // If an op is safe for in-place execution (e.g., Relu, Add) and its input
    // has the same size and no other consumers after this op, the output can
    // reuse the input's buffer.
    for (int i = 0; i < (int)lifetimes_.size(); ++i) {
        auto& lt = lifetimes_[i];
        auto* node = nodes[lt.producer];
        if (!is_inplace_safe(node->op_type)) continue;

        for (auto* in : node->inputs) {
            if (!in || !lt_index.count(in)) continue;
            int parent_idx = lt_index[in];
            auto& in_lt = lifetimes_[parent_idx];

            // Can reuse if: parent's last consumer is exactly this op (no later readers)
            // and buffer sizes match exactly (no partial overlap).
            if (in_lt.last_consumer == lt.producer &&
                in_lt.size_bytes == lt.size_bytes) {
                lt.inplace = true;
                lt.inplace_parent = parent_idx;
                // Propagate lifetime extension up the in-place alias chain:
                // since all aliases share the same physical buffer, the buffer
                // must live until the last consumer of ANY alias in the chain.
                int idx = parent_idx;
                while (idx >= 0) {
                    if (lt.last_consumer > lifetimes_[idx].last_consumer)
                        lifetimes_[idx].last_consumer = lt.last_consumer;
                    idx = lifetimes_[idx].inplace_parent;
                }
                break;
            }
        }
    }

    // Phase 1b: Concat alias detection
    // For Concat on an axis where each input owns a contiguous byte range in
    // the output, inputs can alias into the output buffer. Producers then
    // write directly into the output — Concat becomes a no-op.
    //
    // The "contiguous run" property requires the product of *physical* dims
    // *before* the concat-axis's *physical* position to be 1. Tensor `dims[]`
    // are in logical (NCHW) order regardless of `format`; physical order is:
    //   NCHW / BLOCKED_*  : [N, C, ...]    (logical order, axis index direct)
    //   NHWC              : [N, H, W, C]   (C moved to inner position 3)
    //
    //   - NCHW axis=0 always; axis=1 N=1; axis=2 N=C=1 (rare).
    //   - NHWC axis=0 always (batch slabs contiguous).
    //   - NHWC axis=2 N=1 (H concat — H-slabs of W*C contiguous).
    //   - NHWC axis=3 N=H=1 (W concat — W*C contiguous within an H-row).
    //   - NHWC axis=1 only N=H=W=1 (channel concat in degenerate shape).
    //   - BLOCKED axis=0 always; axis=1 N=1 + C%block==0.
    //
    // BLOCKED axis ≥ 2 splits a c-block across the spatial dim. NHWC axis=1
    // in the common (H>1 or W>1) case requires producer kernels to write
    // with the parent's wider C stride — separate architectural extension
    // tracked in kb/nhwc_concat_strided.md.
    // Per-format×axis tally for Phase 1a contig alias firings, so we can see
    // which paths nullify Concat in real models. Phase 1b-strided handles only
    // NHWC axis=1 with H>1 or W>1; everything else lands here.
    int p1a_total = 0;
    int p1a_fmt_axis[4][4] = {0};   // [layout: nchw/nhwc/blk8/blk16][axis 0-3]
    int p1a_aliased = 0;
    int p1a_rej_format = 0, p1a_rej_blockaxis = 0, p1a_rej_outer = 0,
        p1a_rej_input = 0, p1a_rej_inplace = 0, p1a_rej_alreadyparent = 0,
        p1a_rej_consumer = 0, p1a_rej_layoutmismatch = 0,
        p1a_rej_blockalign = 0, p1a_rej_dimsmatch = 0, p1a_rej_sizesum = 0;
    for (int i = 0; i < (int)lifetimes_.size(); ++i) {
        auto& lt = lifetimes_[i];
        auto* node = nodes[lt.producer];
        if (node->op_type != "Concat") continue;
        if (node->inputs.empty() || node->outputs.empty()) continue;
        auto* y = node->outputs[0];
        if (!y || y->ndim < 2 || y->type == NNR_DATA_TYPE_STRING) continue;
        ++p1a_total;

        int block = 0;
        bool is_blocked = false;
        int fmt_idx = -1;
        switch (y->format) {
        case memory_layout_t::NCHW:       fmt_idx = 0; break;
        case memory_layout_t::NHWC:       fmt_idx = 1; break;
        case memory_layout_t::BLOCKED_8:  fmt_idx = 2; block = 8;  is_blocked = true; break;
        case memory_layout_t::BLOCKED_16: fmt_idx = 3; block = 16; is_blocked = true; break;
        default:                          ++p1a_rej_format; continue;
        }

        int axis = node->attribute("axis", (int32_t)0);
        if (axis < 0) axis += y->ndim;

        // BLOCKED layouts: spatial concat (axis>=2) splits inside c-blocks.
        // Only axis=0 (batch) and axis=1 (channel) are tractable.
        if (is_blocked && axis != 0 && axis != 1) { ++p1a_rej_blockaxis; continue; }

        // Each input must occupy a single contiguous byte range in the output
        // buffer. That requires the product of *physical* dims before the
        // concat-axis's *physical* position to be 1. Tensor `dims[]` are in
        // logical (NCHW) order regardless of `format`; the physical mapping
        // depends on layout:
        //   NCHW / BLOCKED_*  : physical = [N, C, ...]    — logical order
        //   NHWC (4D only)    : physical = [N, H, W, C]   — C moved to inner
        // The is_blocked check above already restricted blocked to axis ∈ {0,1},
        // where the dims-before-axis product happens to match physically.
        size_t physical_outer = 1;
        if (y->format == memory_layout_t::NHWC) {
            if (y->ndim != 4) { ++p1a_rej_format; continue; }
            switch (axis) {
            case 0: physical_outer = 1; break;
            case 1: physical_outer =
                        (size_t)y->dims[0] * y->dims[2] * y->dims[3]; break;
            case 2: physical_outer = (size_t)y->dims[0]; break;
            case 3: physical_outer = (size_t)y->dims[0] * y->dims[2]; break;
            default: ++p1a_rej_format; continue;
            }
        } else {
            for (int d = 0; d < axis; d++) physical_outer *= y->dims[d];
        }
        if (physical_outer != 1) { ++p1a_rej_outer; continue; }

        // Check all inputs: must be in lifetime table, must have matching spatial dims
        bool eligible = true;
        int reject_reason = 0;  // 0=ok, 1=input, 2=inplace, 3=already_parent,
                                // 4=consumer, 5=layout_mismatch, 6=block_align, 7=dims_match
        size_t elem_sz = data_type_sizeof(y->type);
        size_t offset = 0;
        for (auto* inp_raw : node->inputs) {
            tensor_t* inp = resolve_skip(inp_raw);
            if (!inp || !lt_index.count(inp)) { eligible = false; reject_reason = 1; break; }
            auto& in_lt = lifetimes_[lt_index[inp]];
            // Skip if input is already inplace-aliased to something else
            if (in_lt.inplace) { eligible = false; reject_reason = 2; break; }
            // Skip if input is a graph output or has special handling
            if (in_lt.concat_parent >= 0) { eligible = false; reject_reason = 3; break; }
            // Input's last consumer must be this Concat — otherwise aliasing
            // breaks in-place chains for ops that consume the input later.
            if (in_lt.last_consumer != lt.producer) { eligible = false; reject_reason = 4; break; }
            // Storage must match the output's layout exactly (alias is a
            // direct byte view; layout mismatches would scramble the output).
            if (inp->format != y->format) {
                static int s_logged = 0;
                if (s_logged < 12) {
                    auto fname = [](memory_layout_t f) {
                        switch (f) {
                        case memory_layout_t::NCHW: return "NCHW";
                        case memory_layout_t::NHWC: return "NHWC";
                        case memory_layout_t::BLOCKED_8: return "BLK8";
                        case memory_layout_t::BLOCKED_16: return "BLK16";
                        }
                        return "?";
                    };
                    fprintf(stderr,
                        "[concat-alias] layout_mismatch concat='%.*s' axis=%d "
                        "y.fmt=%s inp.fmt=%s inp_name='%.*s'\n",
                        (int)y->name.size(), y->name.data(), axis,
                        fname(y->format), fname(inp->format),
                        (int)inp->name.size(), inp->name.data());
                    ++s_logged;
                }
                eligible = false; reject_reason = 5; break;
            }
            // BLOCKED axis=1: channel slabs must align on whole c-blocks.
            if (is_blocked && axis == 1 && (inp->dims[axis] % block) != 0) {
                eligible = false; reject_reason = 6; break;
            }
            // Check spatial dims match (all dims except axis must match)
            for (int d = 0; d < y->ndim; d++) {
                if (d == axis) continue;
                if (inp->dims[d] != y->dims[d]) { eligible = false; reject_reason = 7; break; }
            }
            if (!eligible) break;
            offset += in_lt.size_bytes;
        }
        if (!eligible) {
            switch (reject_reason) {
            case 1: ++p1a_rej_input; break;
            case 2: ++p1a_rej_inplace; break;
            case 3: ++p1a_rej_alreadyparent; break;
            case 4: ++p1a_rej_consumer; break;
            case 5: ++p1a_rej_layoutmismatch; break;
            case 6: ++p1a_rej_blockalign; break;
            case 7: ++p1a_rej_dimsmatch; break;
            }
            continue;
        }
        if (offset != lt.size_bytes) { ++p1a_rej_sizesum; continue; }

        // All checks passed — set up concat aliases. Resolve through skip
        // chain so we wire concat_parent on the actual lifetime entry.
        offset = 0;
        for (auto* inp_raw : node->inputs) {
            tensor_t* inp = resolve_skip(inp_raw);
            int inp_idx = lt_index[inp];
            auto& in_lt = lifetimes_[inp_idx];
            in_lt.concat_parent = i;
            in_lt.concat_offset = offset;
            offset += in_lt.size_bytes;
            // Extend Concat output lifetime to cover the input's producer
            // (input is allocated from the output's slot, so output must exist)
            if (in_lt.producer < lt.producer)
                lt.producer = in_lt.producer;
            // Also extend to cover the input's last consumer
            if (in_lt.last_consumer > lt.last_consumer)
                lt.last_consumer = in_lt.last_consumer;
        }
        ++p1a_aliased;
        if (fmt_idx >= 0 && axis >= 0 && axis < 4) ++p1a_fmt_axis[fmt_idx][axis];
    }
    if (p1a_total > 0) {
        const char* fnames[4] = {"NCHW", "NHWC", "BLK8", "BLK16"};
        fprintf(stderr, "[concat-alias] phase1a contig: total=%d aliased=%d",
                p1a_total, p1a_aliased);
        for (int f = 0; f < 4; ++f)
            for (int a = 0; a < 4; ++a)
                if (p1a_fmt_axis[f][a])
                    fprintf(stderr, " %s/ax%d=%d", fnames[f], a, p1a_fmt_axis[f][a]);
        fprintf(stderr, "\n");
        fprintf(stderr, "[concat-alias] phase1a rejected: format=%d blkaxis=%d outer=%d "
                "input=%d inplace=%d already_parent=%d consumer=%d layout_mismatch=%d "
                "block_align=%d dims_match=%d size_sum=%d\n",
                p1a_rej_format, p1a_rej_blockaxis, p1a_rej_outer,
                p1a_rej_input, p1a_rej_inplace, p1a_rej_alreadyparent,
                p1a_rej_consumer, p1a_rej_layoutmismatch,
                p1a_rej_blockalign, p1a_rej_dimsmatch, p1a_rej_sizesum);
    }

    // Phase 1b-strided: NHWC channel-axis Concat alias with strided producers.
    // Covers the common case (caxis=1, H>1 or W>1) where the contiguous-block
    // alias above rejected because physical_outer != 1. Each input occupies a
    // sub-channel stripe at every (n,h,w) position; the producer kernel must
    // write with the parent's wider C stride (LDC = parent->dims[1]) instead
    // of its local channel count. Eligibility checked via
    // operator_t::supports_strided_output(); for M2 this is Conv NHWC 1×1 fp32.
    int considered = 0, rejected_format = 0, rejected_axis = 0,
        rejected_outer = 0, rejected_eligible = 0, accepted = 0;
    for (int i = 0; i < (int)lifetimes_.size(); ++i) {
        auto& lt = lifetimes_[i];
        if (lt.concat_parent >= 0) continue;  // already aliased by contig branch
        auto* node = nodes[lt.producer];
        if (node->op_type != "Concat") continue;
        if (node->inputs.empty() || node->outputs.empty()) continue;
        ++considered;
        auto* y = node->outputs[0];
        if (!y || y->ndim != 4) { ++rejected_format; continue; }
        if (y->format != memory_layout_t::NHWC) { ++rejected_format; continue; }
        // Type gating moved to producer's supports_strided_output override —
        // each op decides whether its NHWC kernel honors strides_set for its
        // dtype (M2-M5: fp32 Conv/Pool/AvgPool/BN/Add; M6+: int8 Conv).

        int axis = node->attribute("axis", (int32_t)0);
        if (axis < 0) axis += y->ndim;
        if (axis != 1) { ++rejected_axis; continue; }  // M2: channel axis only

        // We only enter this branch when physical_outer > 1 (else the contig
        // branch already aliased this Concat).
        size_t physical_outer = (size_t)y->dims[0] * y->dims[2] * y->dims[3];
        if (physical_outer == 1) { ++rejected_outer; continue; }

        // Eligibility: every input must be NHWC-fp32, last consumer is this
        // Concat, dims match (except the concat axis), producer honors
        // strides_set on output, and no prior alias claim.
        bool eligible = true;
        size_t elem_sz = data_type_sizeof(y->type);
        for (auto* inp : node->inputs) {
            if (!inp || !lt_index.count(inp)) { eligible = false; break; }
            auto& in_lt = lifetimes_[lt_index[inp]];
            if (in_lt.inplace) { eligible = false; break; }
            if (in_lt.concat_parent >= 0) { eligible = false; break; }
            if (in_lt.last_consumer != lt.producer) { eligible = false; break; }
            if (inp->format != memory_layout_t::NHWC) { eligible = false; break; }
            if (inp->type != y->type) { eligible = false; break; }
            if (in_lt.producer < 0) { eligible = false; break; }
            auto* prod = nodes[in_lt.producer];
            if (!prod || !prod->supports_strided_output(memory_layout_t::NHWC)) {
                eligible = false; break;
            }
            // Binary fusion (e.g., Add residual) reads a paired tensor at a
            // linear offset that assumes contiguous dst. Strided dst would
            // misalign that lookup, so skip alias when binary fusion is set.
            if (prod->fused_tensor != nullptr) { eligible = false; break; }
            for (int d = 0; d < y->ndim; ++d) {
                if (d == axis) continue;
                if (inp->dims[d] != y->dims[d]) { eligible = false; break; }
            }
            if (!eligible) break;
        }
        if (!eligible) { ++rejected_eligible; continue; }
        ++accepted;

        // Wire alias: parent C in elements, per-input start C in elements.
        const int parent_C = y->dims[1];
        size_t start_C_elem = 0;
        for (auto* inp : node->inputs) {
            int inp_idx = lt_index[inp];
            auto& in_lt = lifetimes_[inp_idx];
            in_lt.concat_parent = i;
            in_lt.concat_offset = start_C_elem * elem_sz;
            // Publish parent's inner-channel stride on the input. Producers
            // read this via tensor_stride(t, 1) when strides_set is true.
            in_lt.tensor->strides[1] = parent_C;
            in_lt.tensor->strides_set = true;
            start_C_elem += (size_t)inp->dims[1];
            if (in_lt.producer < lt.producer)
                lt.producer = in_lt.producer;
            if (in_lt.last_consumer > lt.last_consumer)
                lt.last_consumer = in_lt.last_consumer;
        }

        // Cache-line occupancy report. Strided alias is a clean win only when
        // each branch boundary lands on a cache line; otherwise neighboring
        // producers share a line and pay an RMW + false-sharing tax. Print
        // one line per activated Concat so users can see which aliases are
        // clean vs sub-line-bound. Fixed line=64 B (x64); CUDA backend should
        // reuse this report with line=128 once it adopts strided alias.
        constexpr int LINE = 64;
        const size_t parent_bytes = (size_t)parent_C * elem_sz;
        const int total_lines = (int)((parent_bytes + LINE - 1) / LINE);
        // A line is "shared" if it's straddled by 2+ producers. Equivalently:
        // a producer is line-clean iff both its start and end byte boundaries
        // are line-aligned. shared_lines = count of boundaries (start_C*es)
        // that are NOT line-aligned (excluding the trivial start_C=0).
        int shared_lines = 0;
        size_t cursor = 0;
        for (auto* inp : node->inputs) {
            cursor += (size_t)inp->dims[1] * elem_sz;
            // Last boundary lands at parent_bytes; aligned iff parent itself is.
            // Interior boundaries cause a shared line if not line-aligned.
            const bool is_last = (cursor == parent_bytes);
            if (!is_last && (cursor % LINE) != 0) ++shared_lines;
        }
        const float occupancy = (float)parent_bytes / (float)((size_t)total_lines * LINE);
        fprintf(stderr,
            "[concat-alias] parent='%.*s' branches=%d parent_C*es=%zuB lines=%d shared=%d occ=%.0f%% %s\n",
            (int)y->name.size(), y->name.data(),
            (int)node->inputs.size(), parent_bytes, total_lines, shared_lines,
            occupancy * 100.0f,
            (shared_lines == 0 ? "[clean]" : "[dirty]"));
    }
    if (considered > 0) {
        fprintf(stderr,
            "[concat-alias] phase1b-strided summary: considered=%d accepted=%d "
            "rejected: format=%d axis=%d outer=%d eligible=%d\n",
            considered, accepted, rejected_format, rejected_axis,
            rejected_outer, rejected_eligible);
    }

    // Phase 2: Sort non-inplace, non-concat-aliased tensors by size descending
    std::vector<int> order;
    for (int i = 0; i < (int)lifetimes_.size(); ++i)
        if (!lifetimes_[i].inplace && lifetimes_[i].concat_parent < 0)
            order.push_back(i);

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return lifetimes_[a].size_bytes > lifetimes_[b].size_bytes;
    });

    // Phase 3: Assign buffer slots (best-fit decreasing)
    // Sorted by size descending so large tensors get allocated first (better packing).
    // For each tensor, find an existing slot that: (a) is free (not in use), and
    // (b) minimizes wasted space (cost = how many extra bytes the slot must grow).
    // A new slot is created if no good fit exists (cost > tensor size = too wasteful).
    slots_.clear();

    for (int idx : order) {
        auto& lt = lifetimes_[idx];

        int best = -1;
        size_t best_cost = SIZE_MAX;

        for (int s = 0; s < (int)slots_.size(); ++s) {
            if (slots_[s].free_after >= lt.producer) continue;  // slot still in use

            // cost = bytes the slot must grow (0 if slot is already big enough)
            size_t cost = (lt.size_bytes > slots_[s].size)
                        ? (lt.size_bytes - slots_[s].size) : 0;
            if (cost < best_cost) {
                best = s;
                best_cost = cost;
            }
        }

        if (best >= 0 && best_cost <= lt.size_bytes) {
            lt.slot_id = best;
            if (lt.size_bytes > slots_[best].size)
                slots_[best].size = lt.size_bytes;
            slots_[best].free_after = lt.last_consumer;
        } else {
            lt.slot_id = (int)slots_.size();
            slots_.push_back({lt.size_bytes, lt.last_consumer});
        }
    }

    // Phase 4: Propagate in-place slot assignments from parent
    for (auto& lt : lifetimes_) {
        if (!lt.inplace) continue;
        if (lt.inplace_parent >= 0 && lifetimes_[lt.inplace_parent].slot_id >= 0) {
            lt.slot_id = lifetimes_[lt.inplace_parent].slot_id;
        } else {
            lt.inplace = false;
            lt.slot_id = (int)slots_.size();
            slots_.push_back({lt.size_bytes, lt.last_consumer});
        }
    }

    num_slots = (int)slots_.size();
    num_inplace = 0;
    total_pool_bytes = 0;
    for (auto& lt : lifetimes_)
        if (lt.inplace) num_inplace++;
    for (auto& s : slots_)
        total_pool_bytes += s.size;
}

void memory_planner_t::apply()
{
    pool_.resize(slots_.size());
    for (int i = 0; i < (int)slots_.size(); ++i)
        pool_[i] = nnr_aligned_alloc(slots_[i].size, 64);
    // Fresh buffers are uninitialized; the next zero_pool() call must wipe
    // them before the first inference. After that, ops overwrite their own
    // outputs and per-run zeroing becomes redundant.
    pool_zeroed_ = false;

    for (auto& lt : lifetimes_) {
        if (lt.slot_id < 0 || !pool_[lt.slot_id]) continue;
        tensor_t* t = lt.tensor;

        if (t->owns_data && t->data && t->ndata > 0)
            delete_data(t->data, t->type);

        t->data = pool_[lt.slot_id];
        t->owns_data = false;
    }

    // Assign concat-aliased tensors: point into parent's pool slot at offset.
    // Also publish the alias to the tensor itself so GPU backends (gpu_cache)
    // can mirror the aliasing on the device side — producer kernels then
    // write directly into the Concat output's device buffer.
    for (auto& lt : lifetimes_) {
        if (lt.concat_parent < 0) continue;
        auto& parent_lt = lifetimes_[lt.concat_parent];
        if (parent_lt.slot_id < 0 || !pool_[parent_lt.slot_id]) continue;
        tensor_t* t = lt.tensor;
        if (t->owns_data && t->data && t->ndata > 0)
            delete_data(t->data, t->type);
        t->data = (char*)pool_[parent_lt.slot_id] + lt.concat_offset;
        t->owns_data = false;
        t->concat_parent = parent_lt.tensor;
        t->concat_offset = lt.concat_offset;
    }

    // Activate zero-copy views for view ops whose outputs were excluded from
    // the pool. The source tensor's lifetime has been extended to cover all
    // consumers, so it's safe to alias. Free the view output's first-run
    // allocation since it won't be used anymore.
    // Handles both single-output and multi-output view ops.
    // Skip folded nodes: their outputs contain folded constants that must persist.
    for (auto* node : ctx_->graph->nodes) {
        int vi = node->view_input_index();
        if (vi < 0 || node->folded) continue;
        for (auto* y : node->outputs) {
            if (y && y->owns_data && y->data && y->ndata > 0) {
                delete_data(y->data, y->type);
                y->data = nullptr;
                y->owns_data = false;
            }
        }
    }
}

void memory_planner_t::zero_pool()
{
    for (int i = 0; i < (int)slots_.size(); ++i)
        if (pool_[i]) memset(pool_[i], 0, slots_[i].size);
    pool_zeroed_ = true;
}

void memory_planner_t::release()
{
    for (auto& lt : lifetimes_) {
        tensor_t* t = lt.tensor;
        if (!t->owns_data) {
            t->data = nullptr;
            t->ndata = 0;
            t->owns_data = true;
        }
        t->concat_parent = nullptr;
        t->concat_offset = 0;
        t->strides_set = false;
    }
    // Skip-aliased tensors (e.g. BN/Relu outputs fused into Conv) are not in
    // lifetimes_ but still hold non-owning pointers into pool memory — set up
    // by the SKIP handler in run_graph_impl, or by view ops at exec time.
    // Without resetting them here, the subsequent pool free leaves their
    // `data` dangling. fold_run's second-run pass then walks those tensors
    // (e.g. MaxPool reading the fused Conv's relu output) and crashes.
    // Guard on pool_ — when release() is called a second time (~planner via
    // ~context_t after context's explicit pre-clean), the map's tensor_t
    // objects have already been deleted; the pool is also empty so this work
    // would be a no-op anyway.
    if (ctx_ && !pool_.empty()) {
        for (auto& [k, t] : ctx_->map) {
            if (t && !t->owns_data) {
                t->data = nullptr;
                t->ndata = 0;
                t->owns_data = true;
                t->concat_parent = nullptr;
                t->concat_offset = 0;
                t->strides_set = false;
            }
        }
    }

    for (auto* p : pool_)
        nnr_aligned_free(p);
    pool_.clear();
    slots_.clear();
    lifetimes_.clear();
    pool_zeroed_ = false;

    num_intermediates = 0;
    num_slots = 0;
    num_inplace = 0;
    total_pool_bytes = 0;
    total_unpooled_bytes = 0;
}

memory_planner_t::~memory_planner_t()
{
    release();
}

} // namespace nnr
