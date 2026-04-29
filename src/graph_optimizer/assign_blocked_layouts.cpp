// ORT-style eligibility-based NCHWc layout assignment.
//
// Earlier revisions ran a per-chain cost model that summed kernel and
// boundary-reorder costs for every NCHWc candidate and accepted only if
// the ratio beat a hard threshold (0.90, or 1.15 for chains containing a
// non-1x1 DW). That model repeatedly misfired: the 1.15× NCHWc 1x1
// pessimism got summed twice per ResNet bottleneck (2× 1x1 per 3x3) and
// dragged chain ratios from ~0.97 to ~1.07, rejecting entire models
// (fcn-resnet50-11 ran 2.45× slower than ORT purely because of this).
//
// ORT solves the same problem structurally (see
// onnxruntime/core/optimizer/nchwc_transformer.cc::TransformConv): if the
// shape passes the structural gate, rewrite unconditionally, and rely on
// boundary reorders to collapse at runtime when adjacent ops are also
// blocked. We now do the same thing. Correctness of the decision is
// pushed entirely into the eligibility gate — every rule there reflects
// a real kernel capability, not a cost estimate. The chain walker is
// retained only so debug output still groups convs by connectivity.
//
// Block width: NATIVE_BLOCK (nnr.h) — 16 on x64 (AVX-512 ZMM width),
// 8 on ARM64 (2 NEON qregs). The pass is parameterized over block size
// so the same eligibility rules apply on both architectures; only the
// alignment constant differs. NATIVE_BLOCKED_FMT is the matching enum
// value (BLOCKED_16 on x64, BLOCKED_8 on ARM64).
//
// Eligibility rules (each maps to a kernel or runtime capability — never
// to a cost estimate):
//
//   1. op_type == "Conv", fp32, 4D.
//   2. group == 1, OR depthwise (group == iC && kC == 1 && oC == iC).
//   3. iC % NATIVE_BLOCK == 0 (input channel block alignment).
//      oC % NATIVE_BLOCK == 0 is NOT required — Convs with OC tails
//      become "terminal blocked consumers" that accept BLOCKED_16 input
//      but produce NCHW output (see Conv_exec_nchwc.h terminal path).
//      Their output tensors are NOT registered as blocked.
//   4. No dilation — NCHWc Conv kernel has no dilated path.
//
// (Rule 5 retired: NCHWc Winograd carve-out at MT removed — scrollable
// Winograd in scroll chains eliminates the workspace-driven MT regression,
// and the three scroll+fusion crash bugs are fixed.)
//
// Anything passing all four rules gets NCHWc unconditionally. Boundary
// reorder cost is not modeled; adjacent blocked ops collapse their
// reorders naturally at runtime (see nnr.cpp wants_blocked handling).
#include "graph_optimizer/graph_optimizer_internal.h"
#include "graph_optimizer/qdq_helpers.h"
#include "thread_pool.h"
#include "cpu_features.h"

namespace nnr {

// Matches WINOGRAD_MIN_TILES in src/backend/cpu/Conv.cpp.
// Used only for the MT Winograd carve-out below.
static constexpr int NCHWC_WINOGRAD_MIN_TILES = 16;

void assign_blocked_layouts(context_t* ctx)
{
    // CPU-only pass. NCHWc/BLOCKED is an x64 (AVX-512) and ARM64 (NEON)
    // layout — no GPU equivalent. CUDA backend has its own layout
    // pipeline (NHWC int8 for WMMA). Single-backend-per-run gate.
    if (static_cast<backend_t>(ctx->preferred_backend) != backend_t::CPU) return;

    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());
    if (n == 0) return;

    // Rule 5 (retired): NCHWc Winograd dispatch was carved out at MT because
    // the NCHWc kernel regressed ssd-12 by ~130 ms without scroll chains.
    // Three crash bugs blocked scroll activation (all fixed):
    //   1. num_physical() integer division truncation → max(1,...) fix
    //   2. exec_scroll_segment mixed BLOCKED_16/NCHW segments → bail-out guard
    //   3. ring buffer saved_fmt used planned format instead of actual → t->format
    // With scrollable Winograd in scroll chains, the L2-resident strip execution
    // replaces the full-workspace materialization that caused the MT regression.
    const bool carve_out_winograd_nchwc = false;

    // No native blocked layout for this build target — nothing to do.
    // (x86 without AVX-512, or non-x64/non-ARM64 builds.)
    if (NATIVE_BLOCKED_FMT == memory_layout_t::NCHW) return;

#ifdef NNR_ARCH_X64
    // BLOCKED_16 NCHWc kernels (Conv, Winograd, depthwise) are AVX-512 only.
    // On AVX-2-only x64 hardware skip blocked layout entirely — the NCHW path
    // dispatches AVX-2 GEMM/depthwise instead.
    if (!has_avx512()) return;
#endif

    const bool debug_layout = ctx->optimizer->debug_layout;
    const bool force_nchwc = ctx->optimizer->force_nchwc;
    constexpr int block = NATIVE_BLOCK;

    // Phase 1: structural eligibility gate. See file header comment.
    std::vector<bool> eligible(n, false);
    for (int i = 0; i < n; ++i) {
        auto* nd = nodes[i];
        if (nd->skip || nd->folded) continue;
        if (nd->op_type != "Conv") continue;
        if (nd->inputs.size() < 2) continue;
        auto* x = nd->inputs[0];
        auto* w = nd->inputs[1];
        auto* y = nd->outputs.empty() ? nullptr : nd->outputs[0];
        if (!x || !w || !y || x->type != NNR_DATA_TYPE_FLOAT32) continue;
        if (x->ndim != 4 || w->ndim != 4 || y->ndim != 4) continue;

        // Capability check: the Conv must actually advertise a native-blocked
        // kernel in its reshape()-time layout_mask. Without this, shape-based
        // eligibility alone would pull DW/general convs into chains on arches
        // where only a subset of kernels exist (e.g. ARM M1 ships a 1×1-only
        // NCHWc kernel — a shape-eligible DW conv would be grouped into the
        // chain, force boundary reorders, and regress the DW-heavy models).
        if (!(nd->layout_mask & LAYOUT_NATIVE_BLOCKED)) continue;

        int iC = x->dims[1];
        int oC = w->dims[0];
        int kH = w->dims[2], kW = w->dims[3];
        int oH = y->dims[2], oW = y->dims[3];

        // Rule 2: group == 1 or pure depthwise.
        int group = nd->attribute("group", (int32_t)1);
        bool is_dw = (group == iC && w->dims[1] == 1 && oC == iC);
        if (group != 1 && !is_dw) continue;

        // Rule 3: IC block alignment. OC-tail Convs are allowed as terminal
        // consumers — the kernel handles zero-padded last-OCb automatically.
        // IC-tail (iC % block != 0): also OK at chain entry — nchw_to_nchwc
        // zero-fills the partial last block, pack_weight_nchwc_blocked
        // matches with zero-padded weights, and the FMA over zero lanes
        // contributes 0. Rejection of iC < block keeps first-layer (iC=3
        // RGB) Convs out — those have a dedicated conv_first_layer path.
        if (iC < block || oC < 1) continue;

        // Rule 4: no dilation — NCHWc kernel has no dilated path.
        int64_t* dilations = nullptr;
        int ndil = nd->attribute("dilations", dilations);
        bool has_dilation = false;
        for (int di = 0; di < ndil; ++di) if (dilations[di] != 1) has_dilation = true;
        if (has_dilation) continue;

        eligible[i] = true;
    }

    // Phase 2: chain building (debug-only grouping).
    // BFS through LAYOUT_NATIVE_BLOCKED-compatible transparent ops to group
    // connected eligible Convs. The groups are not used for accept/reject
    // decisions — they exist so --debug-layout can print chain summaries.
    std::unordered_map<tensor_t*, std::vector<int>> tensor_consumers;
    for (int i = 0; i < n; ++i) {
        for (auto* t : nodes[i]->inputs) {
            if (t) tensor_consumers[t].push_back(i);
        }
    }
    auto tensor_producer = qdq_helpers::build_producer_map(
        nodes, /*include_folded=*/true);

    auto find_downstream_convs = [&](int start_idx) {
        std::vector<int> queue;
        std::vector<bool> visited(n, false);
        visited[start_idx] = true;

        for (auto* t : nodes[start_idx]->outputs) {
            if (!t) continue;
            auto it = tensor_consumers.find(t);
            if (it == tensor_consumers.end()) continue;
            for (int c : it->second) {
                if (!visited[c]) {
                    visited[c] = true;
                    queue.push_back(c);
                }
            }
        }

        std::vector<int> found_convs;
        for (int qi = 0; qi < (int)queue.size(); ++qi) {
            int idx = queue[qi];
            auto* nd = nodes[idx];

            if (nd->skip || nd->folded) {
                for (auto* t : nd->outputs) {
                    if (!t) continue;
                    auto it = tensor_consumers.find(t);
                    if (it == tensor_consumers.end()) continue;
                    for (int c : it->second) {
                        if (!visited[c]) {
                            visited[c] = true;
                            queue.push_back(c);
                        }
                    }
                }
                continue;
            }

            if (eligible[idx]) {
                found_convs.push_back(idx);
                continue;
            }

            if (!(nd->layout_mask & LAYOUT_NATIVE_BLOCKED)) continue;

            bool has_broadcast = false;
            for (auto* t : nd->inputs) {
                if (t && t->ndim > 0 && t->ndim < 4 && t->ndata > 1) {
                    has_broadcast = true;
                    break;
                }
            }
            if (has_broadcast) continue;

            for (auto* t : nd->outputs) {
                if (!t) continue;
                auto it = tensor_consumers.find(t);
                if (it == tensor_consumers.end()) continue;
                for (int c : it->second) {
                    if (!visited[c]) {
                        visited[c] = true;
                        queue.push_back(c);
                    }
                }
            }
        }
        return found_convs;
    };

    std::vector<int> parent(n, -1);
    for (int i = 0; i < n; ++i)
        if (eligible[i]) parent[i] = i;

    auto uf_find = [&](int x) -> int {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto uf_unite = [&](int a, int b) {
        a = uf_find(a); b = uf_find(b);
        if (a != b) parent[a] = b;
    };

    for (int i = 0; i < n; ++i) {
        if (!eligible[i]) continue;
        auto downstream = find_downstream_convs(i);
        for (int d : downstream)
            uf_unite(i, d);
    }

    std::unordered_map<int, std::vector<int>> chain_groups;
    for (int i = 0; i < n; ++i) {
        if (!eligible[i] || parent[i] < 0) continue;
        chain_groups[uf_find(i)].push_back(i);
    }

    // Phase 3: per-chain structural gate. Greedy-accept every chain,
    // EXCEPT chains whose contents offer no NCHWc compute win:
    //
    //   (a) all-1×1: 1×1 convs are bandwidth-bound — NCHWc and NCHW both
    //       reduce them to the same GEMM. In branching architectures
    //       (squeezenet fire modules, inception blocks) the 1×1 chain
    //       almost always terminates at a Concat whose other branch is
    //       NCHW (its 3×3 was Winograd-excluded), and the resulting
    //       NCHWc→NCHW exit reorder wipes out any marginal benefit.
    //
    //   (b) all-DW: the DW NCHWc kernel has the same per-op cost as the
    //       DW NCHW kernel — see Conv::layout_cost's early-return in the
    //       `groups == C` branch. An isolated DW chain gains no compute
    //       but still pays BLOCKED_16↔NCHW reorders at entry and exit.
    //       DW convs inside a larger mixed chain are still accepted
    //       because they avoid intermediate reorders with the surrounding
    //       pointwise convs.
    //
    // Both are kernel-capability rules, not cost estimates: they reflect
    // what our kernels actually do, not what a bandwidth model predicts.
    auto is_dw_node = [&](int ci) -> bool {
        auto* nd = nodes[ci];
        auto* x = nd->inputs[0];
        auto* w = nd->inputs[1];
        if (!x || !w) return false;
        int iC = x->dims[1];
        int group = nd->attribute("group", (int32_t)1);
        return (group == iC && w->dims[1] == 1);
    };
    // Direct per-chain NHWC vs BLOCKED comparison using each op's
    // layout_cost(BLOCKED_*) override (T1: see Conv.cpp / Pool / QLinearConv).
    // Boundary reorder costs are equal under reorder_cost() (NCHW↔NHWC and
    // NCHW↔BLOCKED both ~ 2.5× bytes), so they cancel out when comparing
    // the two within a chain — no boundary term needed here.
    //
    // Replaces the historical NCHW-vs-NHWC proxy with magic 0.60 threshold;
    // see git history for the per-arch tuning issues that motivated T1.
    auto chain_nhwc_better_than_blocked = [&](const std::vector<int>& convs) -> bool {
        float nhwc_cost = 0, blocked_cost = 0;
        bool first = true;
        for (int ci : convs) {
            auto* nd = nodes[ci];
            nhwc_cost    += nd->layout_cost(memory_layout_t::NHWC, !first);
            blocked_cost += nd->layout_cost(NATIVE_BLOCKED_FMT,    false);
            first = false;
        }
        return blocked_cost > 0 && nhwc_cost < blocked_cost;
    };

    std::vector<bool> accepted(n, false);
    std::unordered_map<int, const char*> reject_reason;
    std::unordered_map<int, std::vector<int>> all_1x1_chains;  // root → conv idxs
    for (auto& [root, convs] : chain_groups) {
        bool all_1x1 = true;
        bool all_dw = true;
        for (int ci : convs) {
            auto* w = nodes[ci]->inputs[1];
            if (!w) continue;
            bool k1 = (w->dims[2] == 1 && w->dims[3] == 1);
            if (!k1) all_1x1 = false;
            if (!is_dw_node(ci)) all_dw = false;
            if (!all_1x1 && !all_dw) break;
        }
        // `force_nchwc` is a dev flag: bypass the all-1x1 / all-DW
        // rejection so individual kernels can be validated in isolation
        // (e.g., the ARM M1 plan lands a 1x1 NCHWc kernel before any
        // mixed-chain candidate exists). Production leaves this off.
        if (all_1x1 && !force_nchwc) {
            reject_reason[root] = "all-1x1";
            all_1x1_chains[root] = convs;
            continue;
        }
        if (all_dw  && !force_nchwc) { reject_reason[root] = "all-dw";  continue; }
        if (!force_nchwc && chain_nhwc_better_than_blocked(convs)) {
            reject_reason[root] = "nhwc-better-than-blocked";
            continue;
        }
        for (int ci : convs) accepted[ci] = true;
    }

    // Phase 3.5 (forward-look): promote rejected all-1×1 chains when their
    // output feeds a Phase-4.5-eligible Concat whose OTHER inputs are already
    // (or will also be) BLOCKED. Without this promotion, inception / squeezenet
    // / densenet Concats see mixed-format inputs and Phase 4.5 falls through,
    // forcing the planner to materialize per-input memcpys instead of aliasing
    // each producer directly into the Concat output buffer.
    //
    // Each promoted chain pays NCHW↔BLOCKED entry/exit reorders, so we only
    // promote when the chain itself can register a BLOCKED output (every
    // conv has OC % block == 0 — otherwise Phase 4 skips its output as a
    // "terminal blocked consumer" producing NCHW, defeating the upgrade).
    auto chain_terminal_outputs = [&](const std::vector<int>& convs) {
        std::vector<tensor_t*> ys;
        std::unordered_set<int> in_chain(convs.begin(), convs.end());
        for (int ci : convs) {
            for (auto* y : nodes[ci]->outputs) {
                if (!y) continue;
                auto cit = tensor_consumers.find(y);
                if (cit == tensor_consumers.end()) { ys.push_back(y); continue; }
                bool external = false;
                for (int c_idx : cit->second) if (!in_chain.count(c_idx)) { external = true; break; }
                if (external) ys.push_back(y);
            }
        }
        return ys;
    };

    auto chain_oc_aligned = [&](const std::vector<int>& convs) {
        for (int ci : convs) {
            auto* w = nodes[ci]->inputs[1];
            if (!w) return false;
            if (w->dims[0] % block != 0) return false;
        }
        return true;
    };

    auto concat_phase45_eligible = [&](operator_t* nd) -> bool {
        if (nd->op_type != "Concat") return false;
        if (nd->skip || nd->folded) return false;
        if (nd->inputs.empty() || nd->outputs.empty()) return false;
        auto* y = nd->outputs[0];
        if (!y || y->ndim != 4 || y->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (y->dims[1] % block != 0) return false;
        int axis = nd->attribute("axis", (int32_t)0);
        if (axis < 0) axis += y->ndim;
        if (axis != 0 && axis != 1) return false;
        if (axis == 1 && y->dims[0] != 1) return false;
        if (axis == 1) {
            for (auto* in : nd->inputs)
                if (!in || in->dims[1] % block != 0) return false;
        }
        return true;
    };

    if (!force_nchwc && !all_1x1_chains.empty()) {
        // Currently-blocked outputs after Phase 3 (will-also-be-promoted set
        // grows during the forward-look loop below).
        std::unordered_set<tensor_t*> will_be_blocked;
        for (int i = 0; i < n; ++i) {
            if (!accepted[i]) continue;
            for (auto* y : nodes[i]->outputs)
                if (y && y->ndim == 4 && y->type == NNR_DATA_TYPE_FLOAT32
                    && y->dims[1] % block == 0)
                    will_be_blocked.insert(y);
        }
        // Map each candidate all-1×1 chain output → root, so the
        // "all OTHER inputs of the Concat will be blocked" check can
        // see promotable chains as also-blocked.
        std::unordered_map<tensor_t*, int> candidate_output_to_root;
        for (auto& [root, convs] : all_1x1_chains) {
            if (!chain_oc_aligned(convs)) continue;
            for (auto* y : chain_terminal_outputs(convs))
                candidate_output_to_root[y] = root;
        }
        // Walk forward from `y` through skip/folded transparent ops
        // (e.g., Conv→Relu where Relu is post_fn-fused → marked folded but
        // still in graph). Returns the first non-skip consumer-tensor pair
        // (consumer_node_idx, tensor reaching it). Stops at first eligible
        // Concat consumer; otherwise returns -1.
        auto find_concat_through_folded = [&](tensor_t* y_in) -> std::pair<int, tensor_t*> {
            tensor_t* y = y_in;
            for (int hop = 0; hop < 8; ++hop) {
                auto cit = tensor_consumers.find(y);
                if (cit == tensor_consumers.end()) return {-1, nullptr};
                if (cit->second.size() != 1) {
                    // Multiple consumers — only proceed if one of them is
                    // an eligible Concat directly.
                    for (int c_idx : cit->second) {
                        auto* c_nd = nodes[c_idx];
                        if (concat_phase45_eligible(c_nd)) return {c_idx, y};
                    }
                    return {-1, nullptr};
                }
                int c_idx = cit->second[0];
                auto* c_nd = nodes[c_idx];
                if (concat_phase45_eligible(c_nd)) return {c_idx, y};
                if (!(c_nd->skip || c_nd->folded)) return {-1, nullptr};
                if (c_nd->outputs.empty() || !c_nd->outputs[0]) return {-1, nullptr};
                y = c_nd->outputs[0];
            }
            return {-1, nullptr};
        };

        // Per-chain rejection log for the forward-look loop (debug_layout only).
        // Surfaces why all-1×1 chains aren't getting promoted to BLOCKED on
        // inception/squeezenet — which silently kills Concat aliasing wins.
        struct fwd_diag_t {
            const char* reason = nullptr;
            tensor_t* blocking_in = nullptr;
            int concat_idx = -1;
        };
        std::unordered_map<int, fwd_diag_t> last_diag;

        bool fwd_changed = true;
        while (fwd_changed) {
            fwd_changed = false;
            for (auto& [root, convs] : all_1x1_chains) {
                if (accepted[convs.front()]) continue;
                if (!chain_oc_aligned(convs)) {
                    if (debug_layout) last_diag[root] = {"oc-not-aligned", nullptr, -1};
                    continue;
                }
                auto outs = chain_terminal_outputs(convs);
                bool unlock = false;
                fwd_diag_t worst{"no-eligible-concat", nullptr, -1};
                for (auto* y : outs) {
                    auto [c_idx, y_at_concat] = find_concat_through_folded(y);
                    if (c_idx < 0) continue;
                    worst = {"some-input-not-blocked", nullptr, c_idx};
                    auto* c_nd = nodes[c_idx];
                    bool all_ok = true;
                    for (auto* in : c_nd->inputs) {
                        if (!in) { all_ok = false; worst.blocking_in = nullptr; break; }
                        if (in == y_at_concat) continue;
                        if (will_be_blocked.count(in)) continue;
                        if (candidate_output_to_root.count(in)) continue;
                        // Walk back through folded transparent ops to find the
                        // ultimate producer's Conv output (mirrors the forward
                        // walk above).
                        tensor_t* src = in;
                        bool found = false;
                        for (int hop = 0; hop < 8; ++hop) {
                            auto pit = tensor_producer.find(src);
                            if (pit == tensor_producer.end()) break;
                            auto* p_nd = nodes[pit->second];
                            if (!(p_nd->skip || p_nd->folded)) break;
                            if (p_nd->inputs.empty() || !p_nd->inputs[0]) break;
                            src = p_nd->inputs[0];
                            if (will_be_blocked.count(src)) { found = true; break; }
                            if (candidate_output_to_root.count(src)) { found = true; break; }
                        }
                        if (!found) { all_ok = false; worst.blocking_in = in; break; }
                    }
                    if (all_ok) { unlock = true; worst.reason = nullptr; break; }
                }
                if (unlock) {
                    for (int ci : convs) accepted[ci] = true;
                    for (auto* y : outs) will_be_blocked.insert(y);
                    reject_reason.erase(root);
                    fwd_changed = true;
                    last_diag.erase(root);
                } else if (debug_layout) {
                    last_diag[root] = worst;
                }
            }
        }
        if (debug_layout) {
            for (auto& [root, d] : last_diag) {
                if (!d.reason) continue;
                const char* concat_name = "?";
                size_t concat_name_len = 1;
                if (d.concat_idx >= 0 && !nodes[d.concat_idx]->outputs.empty()
                    && nodes[d.concat_idx]->outputs[0]) {
                    auto& nm = nodes[d.concat_idx]->outputs[0]->name;
                    concat_name = nm.data();
                    concat_name_len = nm.size();
                }
                const char* blk_name = "?";
                size_t blk_name_len = 1;
                if (d.blocking_in) {
                    blk_name = d.blocking_in->name.data();
                    blk_name_len = d.blocking_in->name.size();
                }
                // For "some-input-not-blocked", trace back through skip chain
                // to find the producer Conv and report whether it was
                // accepted, its OC, and OC%block (the OC-tail case).
                char tail_info[128] = "";
                if (d.blocking_in && d.reason
                    && std::string_view(d.reason) == "some-input-not-blocked") {
                    tensor_t* src = d.blocking_in;
                    int prod_idx = -1;
                    for (int hop = 0; hop < 8; ++hop) {
                        auto pit = tensor_producer.find(src);
                        if (pit == tensor_producer.end()) break;
                        prod_idx = pit->second;
                        auto* p = nodes[prod_idx];
                        if (!(p->skip || p->folded)) break;
                        if (p->inputs.empty() || !p->inputs[0]) break;
                        src = p->inputs[0];
                    }
                    if (prod_idx >= 0) {
                        auto* p = nodes[prod_idx];
                        int oc = -1, ic = -1, kH = -1, kW = -1;
                        const char* prod_name = "?";
                        size_t prod_name_len = 1;
                        if (!p->outputs.empty() && p->outputs[0]
                            && p->outputs[0]->ndim >= 2) {
                            oc = p->outputs[0]->dims[1];
                            prod_name = p->outputs[0]->name.data();
                            prod_name_len = p->outputs[0]->name.size();
                        }
                        if (p->inputs.size() >= 2 && p->inputs[0] && p->inputs[1]
                            && p->inputs[0]->ndim >= 2 && p->inputs[1]->ndim == 4) {
                            ic = p->inputs[0]->dims[1];
                            kH = p->inputs[1]->dims[2];
                            kW = p->inputs[1]->dims[3];
                        }
                        snprintf(tail_info, sizeof(tail_info),
                                 " prod='%.*s' op='%.*s' accepted=%d "
                                 "iC=%d (iC%%blk=%d) oC=%d (oC%%blk=%d) k=%dx%d",
                                 (int)prod_name_len, prod_name,
                                 (int)p->op_type.size(), p->op_type.data(),
                                 (int)accepted[prod_idx],
                                 ic, ic >= 0 ? ic % block : -1,
                                 oc, oc >= 0 ? oc % block : -1,
                                 kH, kW);
                    }
                }
                fprintf(stderr,
                    "[nchwc] fwdlook: chain root=%d not promoted (%s, "
                    "concat='%.*s', blocking_input='%.*s'%s)\n",
                    root, d.reason,
                    (int)concat_name_len, concat_name,
                    (int)blk_name_len, blk_name,
                    tail_info);
            }
        }
    }

    if (debug_layout) {
        for (auto& [root, convs] : chain_groups) {
            bool chain_accepted = !convs.empty() && accepted[convs.front()];
            const char* reason = chain_accepted ? "" : reject_reason[root];
            fprintf(stderr, "[nchwc] chain (id=%d, %d convs): %s%s%s\n",
                root, (int)convs.size(),
                chain_accepted ? "ACCEPT" : "REJECT (",
                chain_accepted ? "" : (reason ? reason : "?"),
                chain_accepted ? "" : ")");
            for (int ci : convs) {
                auto* nd = nodes[ci];
                auto* x = nd->inputs[0];
                auto* w = nd->inputs[1];
                auto* y = nd->outputs.empty() ? nullptr : nd->outputs[0];
                int kH = w ? w->dims[2] : 0, kW = w ? w->dims[3] : 0;
                int iC = x ? x->dims[1] : 0, oC = w ? w->dims[0] : 0;
                int oH = y ? y->dims[2] : 0, oW = y ? y->dims[3] : 0;
                int64_t* strides = nullptr;
                int nstr = nd->attribute("strides", strides);
                int sH = nstr >= 1 ? (int)strides[0] : 1;
                int sW = nstr >= 2 ? (int)strides[1] : 1;
                fprintf(stderr, "[nchwc]   node %3d k%dx%d s%dx%d %d->%d %dx%d\n",
                    ci, kH, kW, sH, sW, iC, oC, oH, oW);
            }
        }
    }

    // Phase 4: register native-blocked tensors in accepted chain nodes.
    // Don't set format yet — the data is still NCHW from the first run.
    // reset_formats() will set the format when the plan is built and a
    // proper run produces blocked data.
    //
    // T3 M1 step 3c — also write declared_layout = NATIVE_BLOCKED_FMT so
    // insert_reorders.cpp can query intended layouts BEFORE first run /
    // reset_formats. This is type-checked path — declared_layout commits
    // synchronously here even though `format` propagates lazily.
    auto& blocked_tensors = ctx->optimizer->blocked_tensors;
    std::unordered_set<tensor_t*> seen;
    for (int i = 0; i < n; ++i) {
        if (!accepted[i]) continue;
        for (auto* t : nodes[i]->outputs) {
            if (t && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32
                && t->dims[1] % block == 0 && !seen.count(t)) {
                blocked_tensors.push_back(t);
                t->declared_layout = NATIVE_BLOCKED_FMT;  // T3 M1 step 3c
                seen.insert(t);
            }
        }
    }

    // Phase 4.5: extend BLOCKED through eligible Concat ops.
    // When every Concat input is already a BLOCKED tensor and the byte layout
    // matches under c-block storage (axis=0, or axis=1 with N=1 + C divisible
    // by block), the Concat output keeps the BLOCKED format. memory_planner
    // can then nullify the Concat by aliasing inputs into the output buffer.
    // The Concat fallback exec_impl is just memcpy, which produces correct
    // BLOCKED output bytes for these cases, so a non-aliased fallback is safe.
    // Iterate to a fixed point so Concat-of-Concat chains propagate.
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < n; ++i) {
            auto* nd = nodes[i];
            if (nd->op_type != "Concat") continue;
            if (nd->skip || nd->folded) continue;
            if (nd->inputs.empty() || nd->outputs.empty()) continue;
            auto* y = nd->outputs[0];
            if (!y || y->ndim != 4 || y->type != NNR_DATA_TYPE_FLOAT32) continue;
            if (seen.count(y)) continue;
            if (y->dims[1] % block != 0) continue;
            int axis = nd->attribute("axis", (int32_t)0);
            if (axis < 0) axis += y->ndim;
            if (axis != 0 && axis != 1) continue;
            if (axis == 1 && y->dims[0] != 1) continue;
            bool all_in = true;
            for (auto* in : nd->inputs) {
                if (!in) { all_in = false; break; }
                // Walk back through skip/folded transparent ops (e.g.,
                // Conv→Relu(skip)→Concat). Phase 4 only registered raw Conv
                // outputs in `seen`; without this walk, fused-Relu Concats
                // never get extended and downstream alias rejects on
                // layout_mismatch.
                tensor_t* src = in;
                for (int hop = 0; hop < 8; ++hop) {
                    if (seen.count(src)) break;
                    auto pit = tensor_producer.find(src);
                    if (pit == tensor_producer.end()) break;
                    auto* p_nd = nodes[pit->second];
                    if (!(p_nd->skip || p_nd->folded)) break;
                    if (p_nd->inputs.empty() || !p_nd->inputs[0]) break;
                    src = p_nd->inputs[0];
                }
                if (!seen.count(src)) { all_in = false; break; }
                if (axis == 1 && src->dims[1] % block != 0) { all_in = false; break; }
            }
            if (!all_in) continue;
            blocked_tensors.push_back(y);
            y->declared_layout = NATIVE_BLOCKED_FMT;  // T3 M1 step 3c (Phase 4.5)
            seen.insert(y);
            changed = true;
        }
    }

    // Phase 4.55: propagate BLOCKED through skip/folded post-op-fused ops
    // (Conv-fused Relu/Clip/Sigmoid/Tanh/LeakyRelu/Elu/BN). Their output
    // tensor is a SEPARATE tensor_t from input even though their data
    // pointers alias at runtime — so the input's declared_layout = BLOCKED
    // doesn't automatically reach the output. Without this, downstream
    // Conv::scroll_info sees the consumer's input declared NCHW and
    // rejects the chain. Whitelisted by op_type to avoid touching ops
    // that change shape/dtype (Reshape/Cast/Squeeze/...).
    auto is_post_op_fused = [](std::string_view t) {
        return t == "Relu" || t == "Clip" || t == "Sigmoid"
            || t == "Tanh" || t == "LeakyRelu" || t == "Elu"
            || t == "BatchNormalization";
    };
    // Phase 4.55 + 4.6 share an outer fixed-point. Without it, an active
    // Conv→Add→Relu chain stalls because Phase 4.55 visits Relu before
    // Phase 4.6 marks Add's output BLOCKED, and Relu's output is left NCHW.
    // Looping both passes together lets Add→Relu fan-out propagate cleanly.
    {
        bool outer_changed = true;
        while (outer_changed) {
            outer_changed = false;

            // Phase 4.55: skip/folded + active post-op-fused unary ops
            // (element-wise on flat data, BLOCKED passes through unchanged).
            for (int i = 0; i < n; ++i) {
                auto* nd = nodes[i];
                if (!is_post_op_fused(nd->op_type)) continue;
                if (nd->inputs.empty() || nd->outputs.empty()) continue;
                auto* x = nd->inputs[0];
                auto* y = nd->outputs[0];
                if (!x || !y) continue;
                if (y->ndim != 4 || y->type != NNR_DATA_TYPE_FLOAT32) continue;
                if (seen.count(y)) continue;
                if (!seen.count(x)) continue;
                if (y->dims[1] % block != 0) continue;
                blocked_tensors.push_back(y);
                y->declared_layout = NATIVE_BLOCKED_FMT;
                seen.insert(y);
                outer_changed = true;
            }

            // Phase 4.6: active binary same-shape (Add/Mul/Sub/Div) + active BN.
            for (int i = 0; i < n; ++i) {
                auto* nd = nodes[i];
                if (nd->skip || nd->folded) continue;
                if (nd->inputs.empty() || nd->outputs.empty()) continue;
                std::string_view t = nd->op_type;
                const bool is_binary = (t == "Add" || t == "Mul"
                                     || t == "Sub" || t == "Div");
                const bool is_bn = (t == "BatchNormalization");
                if (!is_binary && !is_bn) continue;
                auto* y = nd->outputs[0];
                if (!y || y->ndim != 4 || y->type != NNR_DATA_TYPE_FLOAT32) continue;
                if (seen.count(y)) continue;
                if (y->dims[1] % block != 0) continue;
                bool all_in = true;
                const int n_check = is_binary ? 2 : 1;
                for (int k = 0; k < n_check && k < (int)nd->inputs.size(); ++k) {
                    auto* in = nd->inputs[k];
                    if (!in) { all_in = false; break; }
                    if (in->ndim != 4 || in->type != NNR_DATA_TYPE_FLOAT32) {
                        all_in = false; break;
                    }
                    if (!seen.count(in)) { all_in = false; break; }
                }
                if (!all_in) continue;
                blocked_tensors.push_back(y);
                y->declared_layout = NATIVE_BLOCKED_FMT;
                seen.insert(y);
                outer_changed = true;
            }
        }
    }

    size_t max_reorder = ctx->optimizer->layout_reorder_ws;
    for (auto* t : blocked_tensors) {
        int C = t->dims[1];
        int Cp = nchwc_padded_channels(C, block);
        size_t sz = (size_t)t->dims[0] * Cp * t->dims[2] * t->dims[3] * sizeof(float);
        if (sz > max_reorder) max_reorder = sz;
    }
    ctx->optimizer->layout_reorder_ws = max_reorder;
}

} // namespace nnr
