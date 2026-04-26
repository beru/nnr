#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// ---------------------------------------------------------------------------
// Layout assignment — cost model is now in operator_t::layout_cost() overrides.
// Reorder cost utilities are in layout_cost.h.
// ---------------------------------------------------------------------------

// Runs after fusion. For each Conv that supports NHWC (layout_mask & NHWC),
// propagates NHWC format forward through layout-agnostic ops (layout_mask == ALL).
// Stops at NCHW-only ops, multi-consumer tensors, multi-input ops with mixed
// formats, or graph outputs. The run loop handles reorder at boundaries.
// Uses cache-oblivious cost model to decide whether NHWC is profitable.

void assign_layouts(context_t* ctx)
{
    // CPU-only pass. NHWC-vs-NCHW chain decisions use CPU layout_cost
    // formulas (cache-oblivious bandwidth model). Other backends own
    // their own layout decision pipelines (CUDA: ad-hoc NHWC int8 pass
    // in the CUDA backend). Single-backend-per-run is enforced via
    // ctx->preferred_backend at op-resolution time.
    if (static_cast<backend_t>(ctx->preferred_backend) != backend_t::CPU) return;

    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());
    if (n == 0) return;
    const bool debug_layout = ctx->optimizer->debug_layout;

    std::unordered_map<tensor_t*, int> consumer_count;
    for (int i = 0; i < n; i++) {
        auto* op = nodes[i];
        if (op->folded) continue;
        for (auto* t : op->inputs)
            if (t) consumer_count[t]++;
    }

    std::unordered_set<tensor_t*> graph_outputs;
    for (auto& name : ctx->graph_outputs) {
        tensor_t* t = ctx->search_tensor(name);
        if (t) graph_outputs.insert(t);
    }

    std::unordered_map<tensor_t*, int> tensor_producer;
    for (int i = 0; i < n; i++) {
        auto* op = nodes[i];
        for (auto* t : op->outputs)
            if (t) tensor_producer[t] = i;
    }

    // Follow skip-chain: resolve a tensor through skipped ops, optionally
    // calling fn(op) on each skipped op encountered along the way.
    auto follow_skip_chain = [&](tensor_t* t, auto&& fn) -> tensor_t* {
        for (int iter = 0; iter < n; iter++) {
            bool found = [&]() {
                for (int i = 0; i < n; i++) {
                    auto* op = nodes[i];
                    if (!op->skip) continue;
                    for (auto* inp : op->inputs) {
                        if (inp == t && !op->outputs.empty() && op->outputs[0]) {
                            fn(op);
                            t = op->outputs[0];
                            return true;
                        }
                    }
                }
                return false;
            }();
            if (!found) break;
        }
        return t;
    };

    auto effective_tensor = [&](tensor_t* t) -> tensor_t* {
        return follow_skip_chain(t, [](auto*) {});
    };

    auto find_consumer = [&](tensor_t* t) -> int {
        tensor_t* eff = effective_tensor(t);
        for (int j = 0; j < n; j++) {
            auto* cand = nodes[j];
            if (cand->skip || cand->folded) continue;
            for (auto* inp : cand->inputs)
                if (inp == eff) return j;
        }
        return -1;
    };

    std::vector<bool> visited(n, false);

    // Aggregate telemetry (emitted at end if debug_layout).
    int stat_chains_seen = 0, stat_chains_full = 0, stat_chains_prefix = 0;
    int stat_chains_rejected = 0, stat_ops_committed = 0;

    for (int start = 0; start < n; start++) {
        auto* op = nodes[start];
        if (op->skip || op->folded) continue;
        if (op->op_type != "Conv" && op->op_type != "QLinearConv") continue;
        if (!(op->layout_mask & LAYOUT_NHWC)) continue;
        if (visited[start]) continue;
        // Skip Convs already committed to NCHWc (assigned by assign_blocked_layouts)
        if (!op->outputs.empty() && op->outputs[0]
            && op->outputs[0]->format == NATIVE_BLOCKED_FMT
            && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW) continue;

        struct chain_entry { int node_idx; };
        std::vector<chain_entry> chain;
        int nhwc_conv_count = 0;
        int cur = start;

        while (cur >= 0 && cur < n) {
            auto* nd = nodes[cur];
            if (nd->skip || nd->folded) { cur++; continue; }
            if (!(nd->layout_mask & LAYOUT_NHWC)) break;
            // Stop at ops whose output is already native-blocked (NCHWc chain)
            if (!nd->outputs.empty() && nd->outputs[0]
                && nd->outputs[0]->format == NATIVE_BLOCKED_FMT
                && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW) break;

            // Multi-input ops (Add, Sum, Concat, etc.): all 4D activation inputs
            // must come from within the chain or already be assigned NHWC from
            // an earlier chain.  Otherwise the elementwise operation mixes NHWC
            // and NCHW data.  Single-input ops (Conv, Pool) handle the boundary
            // case themselves (workspace reorder).
            {
                int n4d = 0, chain_4d = 0;
                for (auto* t : nd->inputs) {
                    if (!t || t->ndim != 4) continue;
                    // Skip initializers (weights, constants) — they're not activation
                    // tensors and don't have a spatial layout to conflict with.
                    if (ctx->initializer_names.count(t->name)) continue;
                    n4d++;
                    bool ok = false;
                    auto it = tensor_producer.find(t);
                    if (it != tensor_producer.end()) {
                        int prod = it->second;
                        for (auto& e : chain)
                            if (e.node_idx == prod) { ok = true; break; }
                    }
                    // Accept tensors already assigned NHWC from earlier chains
                    if (!ok && t->format == memory_layout_t::NHWC) ok = true;
                    if (ok) chain_4d++;
                }
                if (n4d > 1 && chain_4d < n4d) break;
            }

            // Also reject ops with fused post-ops that have external inputs
            // (fused Add reads from the external tensor at the same flat offset).
            if (nd->post_fn && nd->fused_op) {
                bool has_external = false;
                for (auto* t : nd->fused_op->inputs) {
                    if (!t) continue;
                    bool is_internal = false;
                    for (auto* co : nd->outputs)
                        if (co == t) { is_internal = true; break; }
                    if (!is_internal) { has_external = true; break; }
                }
                if (has_external) break;
            }

            chain.push_back({ cur });
            if ((nd->op_type == "Conv" || nd->op_type == "QLinearConv")
                && (nd->layout_mask & LAYOUT_NHWC))
                nhwc_conv_count++;
            visited[cur] = true;

            if (nd->outputs.empty()) break;
            tensor_t* out = nd->outputs[0];
            tensor_t* eff = effective_tensor(out);
            if (graph_outputs.count(eff)) break;
            if (consumer_count[eff] > 1) break;

            int next = find_consumer(out);
            if (next < 0) break;
            if (visited[next]) break;
            cur = next;
        }

        if (nhwc_conv_count < 1) continue;

        // Detect special chain types that bypass the cost model:
        // - Depthwise: NHWC compute savings (13×) dwarf bandwidth costs
        // - All-int8 (QLinearConv only): packed NR=48 NHWC GEMM is strictly
        //   faster than NCHW fused im2col for all shapes and strides
        bool has_depthwise = false;
        bool all_int8_conv = true;
        for (auto& entry : chain) {
            auto* nd = nodes[entry.node_idx];
            // Float Conv: weight is inputs[1], shape [M, C/group, kH, kW].
            // QLinearConv: weight is inputs[3] (after x_scale, x_zp, w).
            const tensor_t* w_t = nullptr;
            if (nd->op_type == "Conv" && nd->inputs.size() >= 2)
                w_t = nd->inputs[1];
            else if (nd->op_type == "QLinearConv" && nd->inputs.size() >= 4)
                w_t = nd->inputs[3];
            if (w_t && w_t->ndim == 4 && w_t->dims[1] == 1) {
                int grp = const_cast<operator_t*>(nd)->attribute(attr_key_t::group, (int32_t)1);
                if (grp == w_t->dims[0])
                    has_depthwise = true;
            }
            if (nd->op_type == "Conv" || nd->op_type == "QLinearConv") {
                if (nd->op_type != "QLinearConv")
                    all_int8_conv = false;
            }
        }

        // -------------------------------------------------------------------
        // Decide how much of this chain should run in NHWC.
        // -------------------------------------------------------------------
        //
        // Background:
        //   Tensors can live in memory as NCHW (channels-major) or NHWC
        //   (channels-last). Most Conv kernels run faster in one layout than
        //   the other, but switching layouts mid-graph costs a "reorder" —
        //   essentially a memcpy that shuffles data. So the question isn't
        //   "is NHWC faster for this one op?" — it's "is NHWC faster for
        //   *this run of ops* after paying the reorder at entry and exit?"
        //
        //   The chain we built above is a linear sequence of ops that *could*
        //   all run in NHWC. But that doesn't mean they *should* — the tail
        //   of the chain might include an op (e.g. GlobalAvgPool, Reshape)
        //   that doesn't benefit from NHWC and drags the total cost up.
        //
        // Strategy — "best prefix":
        //   Score every possible prefix of the chain (the first 1 op,
        //   the first 2 ops, ..., up to all K ops), pick the prefix with
        //   the best predicted speedup, and commit only those ops to NHWC.
        //   The rest stay NCHW. If no prefix is a win, reject the whole
        //   chain and the existing NCHW execution stands.
        //
        //   Example: chain = [Conv, Conv, Conv, GlobalAvgPool].
        //     prefix 1  (Conv)                       → speedup 0.95, reject
        //     prefix 2  (Conv+Conv)                  → speedup 0.85, accept
        //     prefix 3  (Conv+Conv+Conv)             → speedup 0.80, accept
        //     prefix 4  (full chain, incl. Pool)     → speedup 0.93, reject
        //   Pick prefix 3 — the first three Convs become NHWC, the Pool
        //   stays NCHW, and we pay one exit reorder between them.
        //
        // How we avoid doing O(K²) work:
        //   We need "sum of per-op costs over the first k ops" for every k.
        //   Naïve: for each k, re-sum the first k ops — O(K²) total.
        //   Better: compute cumulative arrays pfx_nhwc[k] / pfx_nchw[k]
        //   in one walk, then each prefix cost is a single array lookup —
        //   O(K) total. Same trick as "running sum" in any array problem.
        //
        //   pfx_nhwc[k] = sum of NHWC per-op costs for ops [0 .. k-1]
        //   pfx_nchw[k] = same for NCHW
        //   pfx_convs[k] = how many NHWC-capable Convs fall in [0 .. k-1]
        //                  (we require at least 1 to bother committing a prefix)
        // -------------------------------------------------------------------
        {
            const int K = (int)chain.size();
            // Size K+1 so that index k stores the total for the first k ops,
            // with pfx_*[0] = 0 (the empty prefix).
            std::vector<float> pfx_nhwc(K + 1, 0.0f);
            std::vector<float> pfx_nchw(K + 1, 0.0f);
            std::vector<int>   pfx_convs(K + 1, 0);
            // is_first_conv tracks whether we've walked past the first Conv
            // yet. The very first Conv is special: its NHWC cost estimate
            // must not assume the input is already NHWC (it isn't — the
            // input arrives as NCHW and we pay an entry reorder). Subsequent
            // Convs get to assume their input is NHWC because the previous
            // op in the chain produced it in NHWC.
            bool is_first_conv = true;
            int dbg_conv_count = 0;
            stat_chains_seen++;
            if (debug_layout)
                fprintf(stderr, "  [layout] chain: %d ops, %d convs\n",
                    K, nhwc_conv_count);

            // Walk every op in the chain exactly once, accumulating the
            // cumulative sums. After this loop, pfx_nhwc[k] / pfx_nchw[k]
            // hold the total per-op cost for the first k ops in each layout.
            for (int k = 0; k < K; k++) {
                auto* nd = nodes[chain[k].node_idx];
                // Ask the operator how much work it would do in each layout.
                // Returns 0 for layout-neutral ops (Relu, Add, ...) whose cost
                // doesn't depend on NHWC vs NCHW — those contribute nothing
                // to the comparison and we skip adding them below.
                float ne = nd->layout_cost(memory_layout_t::NHWC, !is_first_conv);
                float ce = nd->layout_cost(memory_layout_t::NCHW, false);
                // "Cumulative sum" pattern: start from the previous total,
                // add this op's contribution (if any), store the running total.
                pfx_nhwc[k + 1] = pfx_nhwc[k];
                pfx_nchw[k + 1] = pfx_nchw[k];
                if (ne > 0 || ce > 0) {
                    pfx_nhwc[k + 1] += ne;
                    pfx_nchw[k + 1] += ce;
                }
                // Count NHWC-capable Convs in the prefix — a prefix containing
                // zero Convs has no real compute to speed up, so we don't
                // bother committing it even if the numbers look favorable.
                pfx_convs[k + 1] = pfx_convs[k];
                if ((nd->op_type == "Conv" || nd->op_type == "QLinearConv")
                    && (nd->layout_mask & LAYOUT_NHWC))
                    pfx_convs[k + 1]++;

                if (debug_layout && nd->op_type == "Conv" && nd->inputs.size() >= 2 && nd->inputs[1]) {
                    auto* w = nd->inputs[1];
                    int dbg_M = w->dims[0], dbg_kC = w->dims[1];
                    int dbg_kH = w->dims[2], dbg_kW = w->dims[3];
                    auto* dbg_y = nd->outputs[0];
                    int dbg_oH = dbg_y->dims[2], dbg_oW = dbg_y->dims[3];
                    int dbg_sp = dbg_oH * dbg_oW;
                    int dbg_groups = (dbg_kC > 0) ? (int)nd->inputs[0]->dims[1] / dbg_kC : 1;
                    bool dbg_wino = (dbg_kH == 3 && dbg_kW == 3 && dbg_groups == 1);
                    if (dbg_wino) {
                        int64_t* ints = nullptr;
                        int sl = const_cast<operator_t*>(nd)->attribute(attr_key_t::strides, ints);
                        bool s1 = (sl == 0) || (sl >= 2 && ints[0] == 1 && ints[1] == 1);
                        int dl = const_cast<operator_t*>(nd)->attribute(attr_key_t::dilations, ints);
                        bool d1 = (dl == 0) || (dl >= 2 && ints[0] == 1 && ints[1] == 1);
                        int nt = ((dbg_oH + 3) / 4) * ((dbg_oW + 3) / 4);
                        dbg_wino = s1 && d1 && nt >= 16;
                    }
                    bool dbg_dw = (dbg_groups == (int)nd->inputs[0]->dims[1]);
                    bool dbg_1x1 = (dbg_kH == 1 && dbg_kW == 1);
                    fprintf(stderr, "    conv[%d]: M=%d kC=%d k=%dx%d sp=%d "
                        "nhwc=%.0f nchw=%.0f ratio=%.3f%s%s%s\n",
                        dbg_conv_count++, dbg_M, dbg_kC, dbg_kH, dbg_kW,
                        dbg_sp, ne, ce, ce > 0 ? ne / ce : 0.0f,
                        dbg_wino ? " [WINOGRAD]" : "",
                        dbg_dw ? " [DW]" : "",
                        dbg_1x1 ? " [1x1]" : "");
                }
                if (nd->op_type == "Conv" || nd->op_type == "QLinearConv") is_first_conv = false;
            }

            // Entry reorder: cost of converting the chain's input tensor from
            // NCHW to NHWC at the chain boundary. This is paid once regardless
            // of which prefix we choose, so we compute it here and add it to
            // every candidate below.
            auto* first_op = nodes[chain.front().node_idx];
            float entry_reorder = 0.0f;
            if (!first_op->inputs.empty() && first_op->inputs[0])
                entry_reorder = reorder_cost(first_op->inputs[0]);

            // Two cases where the cost model is overridden and we always
            // accept the full chain (no prefix search):
            //   - has_depthwise: depthwise Convs get huge compute wins in NHWC
            //     that the bandwidth-focused cost model systematically
            //     underestimates.
            //   - all_int8_conv: all-int8 QLinearConv chains have NHWC-only
            //     fast paths that beat NCHW for all shapes.
            const bool force_accept = has_depthwise || all_int8_conv;

            // Compute the full-chain totals once. Used for debug output, and
            // as the values we commit with when force_accept skips the search.
            float full_nhwc = 0, full_nchw = 0, full_ratio = 0;
            {
                auto* tail = nodes[chain.back().node_idx];
                float er = 0; // exit reorder for the full chain
                if (!tail->outputs.empty() && tail->outputs[0])
                    er = reorder_cost(tail->outputs[0]);
                full_nhwc = pfx_nhwc[K] + entry_reorder + er;
                full_nchw = pfx_nchw[K];
                full_ratio = full_nchw > 0 ? full_nhwc / full_nchw : 0.0f;
            }

            // -----------------------------------------------------------
            // Prefix search: which prefix length k maximizes the speedup?
            // -----------------------------------------------------------
            //
            // For each candidate prefix length k from 1 to K:
            //   1. Look up the per-op NHWC and NCHW totals from the
            //      cumulative arrays — O(1), no re-summation.
            //   2. Add the entry reorder (fixed) and the *exit* reorder
            //      for this specific prefix (depends on where the prefix
            //      ends — the tensor produced by chain[k-1]).
            //   3. Compute the ratio nhwc_total / nchw_total. Below 1.0
            //      means NHWC is cheaper. We require the ratio to drop
            //      below 0.90 before accepting — a 10% margin of safety
            //      against cost-model error (the model can be off by a
            //      few percent in either direction).
            //   4. Remember the k that gives the lowest ratio.
            //
            // Tie-break rule: if two prefixes produce the same ratio (within
            // floating-point noise), prefer the *longer* one — committing
            // more ops is usually right when the cost is the same.
            // -----------------------------------------------------------
            int best_k = 0;                  // 0 = "no prefix is worth committing"
            float best_ratio = 0.90f;        // acceptance threshold; candidates must beat this
            float best_nhwc = 0, best_nchw = 0;
            for (int k = 1; k <= K; k++) {
                if (pfx_convs[k] < 1) continue; // need at least one NHWC Conv

                // Exit reorder for this prefix: the output tensor of the
                // last op in the prefix must be converted back to NCHW so
                // downstream consumers (who are still NCHW) see the layout
                // they expect.
                auto* tail = nodes[chain[k - 1].node_idx];
                float exit_reorder = 0.0f;
                if (!tail->outputs.empty() && tail->outputs[0])
                    exit_reorder = reorder_cost(tail->outputs[0]);

                float nhwc_total = pfx_nhwc[k] + entry_reorder + exit_reorder;
                float nchw_total = pfx_nchw[k];
                if (nchw_total <= 0) continue; // guard against all-neutral prefixes
                float ratio = nhwc_total / nchw_total;
                if (ratio < best_ratio ||
                    (ratio <= best_ratio + 1e-6f && k > best_k)) {
                    best_k = k;
                    best_ratio = ratio;
                    best_nhwc = nhwc_total;
                    best_nchw = nchw_total;
                }
            }

            // Override: depthwise / int8 chains always take the full chain,
            // regardless of what the cost model said.
            if (force_accept) {
                best_k = K;
                best_nhwc = full_nhwc;
                best_nchw = full_nchw;
                best_ratio = full_ratio;
            }

            if (debug_layout) {
                if (best_k == 0) {
                    fprintf(stderr, "    total: no winning prefix → REJECT\n");
                } else if (best_k == K) {
                    fprintf(stderr, "    total: nhwc=%.0f nchw=%.0f ratio=%.3f → ACCEPT full%s\n",
                        best_nhwc, best_nchw, best_nchw > 0 ? best_nhwc / best_nchw : 0.0f,
                        force_accept ? (all_int8_conv ? " (int8)" : " (dw)") : "");
                } else {
                    fprintf(stderr, "    total: nhwc=%.0f nchw=%.0f ratio=%.3f → ACCEPT prefix %d/%d\n",
                        best_nhwc, best_nchw, best_ratio, best_k, K);
                }
            }

            // No prefix was worth committing → leave the whole chain as NCHW.
            if (best_k == 0) { stat_chains_rejected++; continue; }

            // Drop the losing tail of the chain (if any) before the commit
            // loop below runs. Two things happen when we truncate:
            //   1. `chain` shrinks to just the winning prefix, so the commit
            //      loop naturally only touches ops we want to mark NHWC.
            //   2. The ops we dropped had their `visited[]` flag set when we
            //      originally walked the chain. We clear it so those ops can
            //      be re-examined as the start of a *later* chain — otherwise
            //      a Conv we skipped here would be permanently invisible.
            if (best_k < K) {
                for (int k = best_k; k < K; k++)
                    visited[chain[k].node_idx] = false;
                chain.resize(best_k);
                stat_chains_prefix++;
            } else {
                stat_chains_full++;
            }
            stat_ops_committed += best_k;
        }

        for (auto& entry : chain) {
            auto* nd = nodes[entry.node_idx];
            for (auto* t : nd->outputs) {
                if (t && t->ndim == 4 && (t->type == NNR_DATA_TYPE_FLOAT32
                    || t->type == NNR_DATA_TYPE_FLOAT16
                    || t->type == NNR_DATA_TYPE_UINT8 || t->type == NNR_DATA_TYPE_INT8)) {
                    t->format = memory_layout_t::NHWC;
                    t->declared_layout = memory_layout_t::NHWC;  // T3 M1 step 3b
                    ctx->optimizer->nhwc_tensors.push_back(t);
                }
            }
            if (!nd->outputs.empty()) {
                follow_skip_chain(nd->outputs[0], [&](auto* op) {
                    op->outputs[0]->format = memory_layout_t::NHWC;
                    op->outputs[0]->declared_layout = memory_layout_t::NHWC;  // T3 M1 step 3b
                    ctx->optimizer->nhwc_tensors.push_back(op->outputs[0]);
                });
            }
        }

    }

    // Propagate NHWC through inter-chain nodes (DQ/Q/Add/Concat/BN between
    // QLinearConv chains). Without this, reset_formats() only sets chain
    // tensors to NHWC, leaving inter-chain uint8/float tensors NCHW. This
    // forces QLinearConv at chain boundaries to do expensive NCHW→NHWC
    // transpose per inference.
    //
    // Safety rules: only promote a node's output to NHWC if
    //   (a) the node advertises LAYOUT_NHWC (LAYOUT_ALL or explicit),
    //   (b) every 4D activation input is already NHWC (for multi-input ops
    //       like Concat/QLinearAdd this prevents mixed-layout execution), and
    //   (c) every consumer of the output is either already NHWC, LAYOUT_ALL,
    //       or also advertises LAYOUT_NHWC.
    {
        std::unordered_set<tensor_t*> nhwc_set(
            ctx->optimizer->nhwc_tensors.begin(),
            ctx->optimizer->nhwc_tensors.end());

        // Check if all active consumers of a tensor are safe for NHWC
        auto all_consumers_nhwc_safe = [&](tensor_t* tensor) -> bool {
            for (int j = 0; j < n; j++) {
                auto* cnd = nodes[j];
                if (cnd->skip || cnd->folded) continue;
                for (auto* inp : cnd->inputs) {
                    if (inp == tensor) {
                        // Consumer is NHWC chain node, LAYOUT_ALL, or explicitly
                        // supports NHWC → safe. (The NHWC-capable consumer will
                        // read x->format at exec time to pick its NHWC branch.)
                        bool out_nhwc = !cnd->outputs.empty() && cnd->outputs[0]
                            && nhwc_set.count(cnd->outputs[0]);
                        bool supports_nhwc = (cnd->layout_mask == LAYOUT_ALL)
                            || (cnd->layout_mask & LAYOUT_NHWC);
                        if (!out_nhwc && !supports_nhwc)
                            return false;
                        break;
                    }
                }
            }
            return true;
        };

        // Check that every non-initializer 4D activation input of `nd` is
        // already in nhwc_set. Required before we can promote nd's output.
        auto all_inputs_nhwc = [&](operator_t* nd) -> bool {
            for (auto* t : nd->inputs) {
                if (!t || t->ndim != 4) continue;
                if (ctx->initializer_names.count(t->name)) continue;
                if (!nhwc_set.count(t)) return false;
            }
            return true;
        };

        bool changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < n; i++) {
                auto* nd = nodes[i];
                if (nd->skip || nd->folded) continue;
                bool is_layout_all = (nd->layout_mask == LAYOUT_ALL);
                bool supports_nhwc = is_layout_all || (nd->layout_mask & LAYOUT_NHWC);
                if (!supports_nhwc) continue;
                bool has_nhwc_in = false;
                for (auto* t : nd->inputs)
                    if (t && nhwc_set.count(t)) { has_nhwc_in = true; break; }
                if (!has_nhwc_in) continue;
                // For non-LAYOUT_ALL ops, all 4D activation inputs must already
                // be NHWC — mixed-layout inputs would corrupt output.
                if (!is_layout_all && !all_inputs_nhwc(nd)) continue;
                for (auto* t : nd->outputs) {
                    if (!t || t->ndim != 4 || nhwc_set.count(t)) continue;
                    if (!all_consumers_nhwc_safe(t)) continue;
                    t->format = memory_layout_t::NHWC;
                    t->declared_layout = memory_layout_t::NHWC;  // T3 M1 step 3b
                    ctx->optimizer->nhwc_tensors.push_back(t);
                    nhwc_set.insert(t);
                    changed = true;
                }
            }
        }
    }

    size_t max_reorder = 0;
    for (int i = 0; i < n; i++) {
        auto* op = nodes[i];
        if (op->skip || op->folded) continue;
        for (auto* t : op->inputs) {
            if (t && t->format == memory_layout_t::NHWC
                && !(op->layout_mask & LAYOUT_NHWC)) {
                size_t sz = t->ndata * sizeof(float);
                if (sz > max_reorder) max_reorder = sz;
            }
        }
    }
    for (auto* t : graph_outputs) {
        if (t && t->format == memory_layout_t::NHWC) {
            size_t sz = t->ndata * sizeof(float);
            if (sz > max_reorder) max_reorder = sz;
        }
    }
    ctx->optimizer->layout_reorder_ws = max_reorder;

    if (debug_layout) {
        fprintf(stderr, "[layout] summary: %d chains seen, %d full-accept, %d prefix-accept, "
            "%d rejected, %d ops committed, %zu nhwc tensors\n",
            stat_chains_seen, stat_chains_full, stat_chains_prefix,
            stat_chains_rejected, stat_ops_committed,
            ctx->optimizer->nhwc_tensors.size());
    }
}

} // namespace nnr
