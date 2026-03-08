#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------
// After the first inference, all tensor data is populated. Nodes whose
// inputs are all constants (initializers or outputs of other constant nodes)
// produce constant outputs. Mark them as folded so exec() is skipped on
// subsequent runs — the output data from the first run is retained.

void fold_constants(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;

    // Seed: all initializer tensors are constant
    std::unordered_set<const tensor_t*> constants;
    for (auto& [name, tensor] : ctx->map) {
        if (ctx->initializer_names.count(name))
            constants.insert(tensor);
    }

    // Ops not safe to fold (non-deterministic / side effects)
    auto is_unsafe = [](std::string_view op) -> bool {
        return op == "RandomNormal" || op == "RandomNormalLike" ||
               op == "RandomUniform" || op == "RandomUniformLike" ||
               op == "Multinomial" || op == "Bernoulli";
    };

    // Propagate: if all inputs are constant and op is safe, outputs are constant
    for (auto* n : nodes) {
        if (n->skip) continue;
        if (is_unsafe(n->op_type)) continue;

        bool all_const = true;
        for (auto* t : n->inputs) {
            if (t && !constants.count(t)) {
                all_const = false;
                break;
            }
        }

        if (all_const) {
            n->folded = true;
            for (auto* t : n->outputs) {
                if (t) {
                    constants.insert(t);
                    // Exclude from memory planning so the buffer isn't reused
                    ctx->memory_plan_excluded.insert(t->name);
                }
            }
        }
    }
}


} // namespace nnr
