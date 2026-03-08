#include "nnr.h"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstring>
#include <malloc.h>

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
    // For Concat on axis where each input is a contiguous block in the output
    // (axis=0, or axis=1 with N=1), inputs can alias into the output buffer.
    // Producers then write directly into the output — Concat becomes a no-op.
    for (int i = 0; i < (int)lifetimes_.size(); ++i) {
        auto& lt = lifetimes_[i];
        auto* node = nodes[lt.producer];
        if (node->op_type != "Concat") continue;
        if (node->inputs.empty() || node->outputs.empty()) continue;
        auto* y = node->outputs[0];
        if (!y || y->ndim < 2 || y->type == NNR_DATA_TYPE_STRING) continue;
        if (y->format != memory_layout_t::NCHW) continue;

        int axis = node->attribute("axis", (int32_t)0);
        if (axis < 0) axis += y->ndim;

        // Check: all dims before concat axis must be 1 (ensures contiguous blocks)
        int outer = 1;
        for (int d = 0; d < axis; d++) outer *= y->dims[d];
        if (outer != 1) continue;

        // Check all inputs: must be in lifetime table, must have matching spatial dims
        bool eligible = true;
        size_t elem_sz = data_type_sizeof(y->type);
        size_t offset = 0;
        for (auto* inp : node->inputs) {
            if (!inp || !lt_index.count(inp)) { eligible = false; break; }
            auto& in_lt = lifetimes_[lt_index[inp]];
            // Skip if input is already inplace-aliased to something else
            if (in_lt.inplace) { eligible = false; break; }
            // Skip if input is a graph output or has special handling
            if (in_lt.concat_parent >= 0) { eligible = false; break; }
            // Input's last consumer must be this Concat — otherwise aliasing
            // breaks in-place chains for ops that consume the input later.
            if (in_lt.last_consumer != lt.producer) { eligible = false; break; }
            // Check spatial dims match (all dims except axis must match)
            for (int d = 0; d < y->ndim; d++) {
                if (d == axis) continue;
                if (inp->dims[d] != y->dims[d]) { eligible = false; break; }
            }
            if (!eligible) break;
            offset += in_lt.size_bytes;
        }
        if (!eligible) continue;
        if (offset != lt.size_bytes) continue; // sizes don't sum up correctly

        // All checks passed — set up concat aliases
        offset = 0;
        for (auto* inp : node->inputs) {
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
        pool_[i] = _aligned_malloc(slots_[i].size, 64);

    for (auto& lt : lifetimes_) {
        if (lt.slot_id < 0 || !pool_[lt.slot_id]) continue;
        tensor_t* t = lt.tensor;

        if (t->owns_data && t->data && t->ndata > 0)
            delete_data(t->data, t->type);

        t->data = pool_[lt.slot_id];
        t->owns_data = false;
    }

    // Assign concat-aliased tensors: point into parent's pool slot at offset
    for (auto& lt : lifetimes_) {
        if (lt.concat_parent < 0) continue;
        auto& parent_lt = lifetimes_[lt.concat_parent];
        if (parent_lt.slot_id < 0 || !pool_[parent_lt.slot_id]) continue;
        tensor_t* t = lt.tensor;
        if (t->owns_data && t->data && t->ndata > 0)
            delete_data(t->data, t->type);
        t->data = (char*)pool_[parent_lt.slot_id] + lt.concat_offset;
        t->owns_data = false;
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
            if (y && y->owns_data && y->data) {
                if (y->type == NNR_DATA_TYPE_STRING)
                    delete[] (std::string*)y->data;
                else
                    delete[] (char*)y->data;
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
    }

    for (auto* p : pool_)
        _aligned_free(p);
    pool_.clear();
    slots_.clear();
    lifetimes_.clear();

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
