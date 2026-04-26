#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// T3 M1 step 4 — explicit Reorder ops at layout boundaries.
//
// When NNR_EXPLICIT_REORDERS is OFF (M1 default), this is a no-op.
//
// When ON, walks the graph after assign_*_layouts have committed
// declared_layout on every chain tensor. For each layout-aware op (Conv,
// BatchNormalization, MaxPool/AveragePool/GAP, Concat, binary Add/Mul/Sub/Div,
// QLinearAdd/QLinearMul), checks each "data input" against the op's own
// declared output layout. If they differ AND a single-step reorder is
// supported, splices a Reorder node onto that input edge.
//
// This subsumes both Conv input-edge handling (the M1 step 3d scaffold) and
// Conv output-edge handling (T3 task 4) — every op-input edge is the same
// edge as the upstream op's output. Plumbing it on the consumer side handles
// every layout transition in one place. cancel_reorders.cpp then collapses
// adjacent round-trips and merges duplicate sibling reorders that fan out
// to multiple consumers.
//
// Unsupported pairs (e.g. NHWC↔BLOCKED) need a two-step chain via NCHW and
// are deferred — `reorder_supported` returns false and the edge is left
// alone. Layout-transparent ops (Reshape, Cast, view ops, pointwise unary)
// aren't in the layout-aware set; their `outputs[0]->declared_layout` is
// either left at default NCHW (no boundary detected) or kept in sync with
// upstream by assign_layouts (chain-uniform).

namespace {

// Returns true when (from, to) is a supported single-step reorder.
bool reorder_supported(memory_layout_t from, memory_layout_t to)
{
    if (from == to) return false;
    using L = memory_layout_t;
    auto is_blocked = [](L l) { return l == L::BLOCKED_16 || l == L::BLOCKED_8; };
    if (from == L::NCHW && to == L::NHWC)            return true;
    if (from == L::NHWC && to == L::NCHW)            return true;
    if (from == L::NCHW && is_blocked(to))           return true;
    if (is_blocked(from) && to == L::NCHW)           return true;
    return false;  // NHWC↔BLOCKED needs a two-step chain (deferred)
}

// layout_mask capability bit for `l`. Used to mirror nnr.cpp:1118's
// `consumes_blocked` exemption: if the consumer's kernel advertises the
// source layout, no Reorder is needed — the kernel handles the conversion
// internally (e.g., Conv's NCHWc path runs nchw_to_nchwc on entry, or
// nchwc_to_nchw on terminal-Conv exit).
uint8_t layout_capability_bit(memory_layout_t l)
{
    using L = memory_layout_t;
    switch (l) {
    case L::NCHW:       return LAYOUT_NCHW;
    case L::NHWC:       return LAYOUT_NHWC;
    case L::BLOCKED_8:  return LAYOUT_BLOCKED_8;
    case L::BLOCKED_16: return LAYOUT_BLOCKED_16;
    }
    return 0;
}

// Indices of data inputs whose declared_layout must equal the op's own
// output declared_layout. Returns empty for layout-transparent / non-4D ops.
// Constants and scalar attribute inputs (weights, scales, zero-points,
// running mean/var) are excluded — they don't carry a chain layout.
//
// Capacity 64 covers DenseNet-201's deepest dense blocks (~32 layers feeding
// a single Concat) with margin. The previous capacity-8 small_vector
// overflowed silently on densenetblur121d.
small_vector<int, 64> layout_aware_data_inputs(operator_t* n)
{
    small_vector<int, 64> result;
    if (!n) return result;
    std::string_view t = n->op_type;
    auto in_count = (int)n->inputs.size();
    auto push_if = [&](int i) {
        if (i >= 0 && i < in_count && n->inputs[i]) result.push_back(i);
    };
    if (t == "Conv" || t == "QLinearConv"
            || t == "BatchNormalization"
            || t == "MaxPool" || t == "AveragePool" || t == "GlobalAveragePool") {
        push_if(0);
    } else if (t == "Concat") {
        for (int i = 0; i < in_count; ++i) push_if(i);
    } else if (t == "Add" || t == "Mul" || t == "Sub" || t == "Div") {
        push_if(0);
        if (in_count >= 2) push_if(1);
    } else if (t == "QLinearAdd" || t == "QLinearMul") {
        // Layout: A=0, A_scale=1, A_zp=2, B=3, B_scale=4, B_zp=5, Y_scale=6, Y_zp=7
        push_if(0);
        if (in_count >= 4) push_if(3);
    }
    return result;
}

} // namespace

void insert_reorders(context_t* ctx)
{
    if (!ctx || !ctx->graph) return;
#ifndef NNR_EXPLICIT_REORDERS
    (void)ctx;
    return;
#else
    auto& nodes = ctx->graph->nodes;
    if (nodes.empty()) return;

    // Default opset for newly-synthesized ops (matches decompose.cpp).
    int default_opset = 13;
    for (auto& [domain, version] : ctx->meta_opsets) {
        if (domain == "ai.onnx" || domain.empty()) {
            default_opset = (int)version; break;
        }
    }
    const auto backend = static_cast<backend_t>(ctx->preferred_backend);

    using attr_pair_t = std::pair<attr_key_t, attr_t>;
    int reorder_counter = 0;

    std::vector<operator_t*> new_nodes;
    new_nodes.reserve(nodes.size() + 16);

    auto synth_reorder = [&](tensor_t* x, memory_layout_t from,
                             memory_layout_t to) -> tensor_t* {
        // --- Synthesize intermediate tensor ---
        std::string nm = std::string(x->name) + "_reorder"
                       + std::to_string(reorder_counter++);
        size_t len = nm.size();
        char* name_buf = (char*)ctx->attr_pool.alloc(len + 1, 1);
        memcpy(name_buf, nm.data(), len + 1);
        std::string_view intermed_name{name_buf, len};

        tensor_t* intermed = new (std::nothrow) tensor_t(intermed_name, x->type, {});
        if (!intermed) return nullptr;
        intermed->declared_layout = to;
        intermed->format = to;
        ctx->map.emplace_back(intermed_name, intermed);

        // --- Synthesize Reorder op ---
        operator_t* rop = solve_operator("Reorder", default_opset, ctx->attr_pool, backend);
        if (!rop) return nullptr;
        rop->ctx     = ctx;
        rop->opset   = default_opset;
        rop->op_type = "Reorder";
        rop->domain  = "";

        tensor_t** rin  = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        tensor_t** rout = ctx->attr_pool.alloc_arr<tensor_t*>(1);
        rin[0]  = x;
        rout[0] = intermed;
        rop->inputs  = {rin, 1};
        rop->outputs = {rout, 1};

        attr_pair_t* attr_arr = ctx->attr_pool.alloc_arr<attr_pair_t>(2);
        attr_arr[0].first = attr_key_t::from_layout;
        attr_arr[0].second.kind = attr_t::kind_t::INT;
        attr_arr[0].second.i    = (int64_t)from;
        attr_arr[1].first = attr_key_t::to_layout;
        attr_arr[1].second.kind = attr_t::kind_t::INT;
        attr_arr[1].second.i    = (int64_t)to;
        rop->attrs = {attr_arr, 2};

        rop->init();
        // insert_reorders runs AFTER first_run; new ops bypass the normal
        // reshape sweep. Reshape now so the intermediate is sized/allocated
        // before any exec / pre-pass.
        rop->reshape();

        new_nodes.push_back(rop);
        return intermed;
    };

    for (auto* n : nodes) {
        if (!n || n->skip || n->folded
                || n->inputs.empty() || n->outputs.empty() || !n->outputs[0]) {
            new_nodes.push_back(n);
            continue;
        }

        auto idxs = layout_aware_data_inputs(n);
        if (idxs.empty()) {
            new_nodes.push_back(n);
            continue;
        }

        memory_layout_t to = n->outputs[0]->declared_layout;

        // Lazily clone the input array on first rewrite; mutating the
        // pool-backed span in place is unsafe because adjacent ops may share
        // input arrays through how onnx_loader allocates them.
        tensor_t** mut_in = nullptr;
        const size_t nin = n->inputs.size();
        bool any_rewired = false;

        for (int idx : idxs) {
            tensor_t* x = n->inputs[idx];
            if (!x) continue;
            memory_layout_t from = x->declared_layout;
            // Mirror nnr.cpp:1118 (`consumes_blocked`): if the consumer's
            // kernel advertises support for `from`, no Reorder is needed —
            // its exec converts internally (Conv NCHWc path entry/terminal).
            // Without this, mobile chains paid spurious BLOCKED↔NCHW reorders
            // around terminal Convs that the implicit path naturally avoided,
            // causing flag-ON regressions of +50–80% on mobilenet/efficientnet.
            if (n->layout_mask & layout_capability_bit(from)) continue;
            if (!reorder_supported(from, to)) continue;

            tensor_t* intermed = synth_reorder(x, from, to);
            if (!intermed) continue;

            if (!mut_in) {
                mut_in = ctx->attr_pool.alloc_arr<tensor_t*>(nin);
                for (size_t i = 0; i < nin; ++i) mut_in[i] = n->inputs[i];
            }
            mut_in[idx] = intermed;
            any_rewired = true;
        }

        if (any_rewired) {
            n->inputs = {mut_in, nin};
            // CRITICAL: Conv pre-packs weights at reshape() time based on
            // input format. Other ops' reshape is a near-no-op. Re-running
            // it after rewiring covers both — picks the kernel/pre-pack
            // that matches the new input layout.
            n->reshape();
        }

        new_nodes.push_back(n);
    }

    if (new_nodes.size() != nodes.size())
        nodes = std::move(new_nodes);
    (void)reorder_counter;
#endif
}

} // namespace nnr
