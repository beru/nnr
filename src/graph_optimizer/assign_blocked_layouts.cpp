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
#include "thread_pool.h"

namespace nnr {

// Matches WINOGRAD_MIN_TILES in src/backend/cpu/Conv.cpp.
// Used only for the MT Winograd carve-out below.
static constexpr int NCHWC_WINOGRAD_MIN_TILES = 16;

void assign_blocked_layouts(context_t* ctx)
{
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
        if (iC % block != 0 || iC < block || oC < 1) continue;

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
    std::unordered_map<tensor_t*, int> tensor_producer;
    for (int i = 0; i < n; ++i) {
        for (auto* t : nodes[i]->inputs) {
            if (t) tensor_consumers[t].push_back(i);
        }
        for (auto* t : nodes[i]->outputs) {
            if (t) tensor_producer[t] = i;
        }
    }

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
    std::vector<bool> accepted(n, false);
    std::unordered_map<int, const char*> reject_reason;
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
        if (all_1x1 && !force_nchwc) { reject_reason[root] = "all-1x1"; continue; }
        if (all_dw  && !force_nchwc) { reject_reason[root] = "all-dw";  continue; }
        for (int ci : convs) accepted[ci] = true;
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
    auto& blocked_tensors = ctx->optimizer->blocked_tensors;
    std::unordered_set<tensor_t*> seen;
    for (int i = 0; i < n; ++i) {
        if (!accepted[i]) continue;
        for (auto* t : nodes[i]->outputs) {
            if (t && t->ndim == 4 && t->type == NNR_DATA_TYPE_FLOAT32
                && t->dims[1] % block == 0 && !seen.count(t)) {
                blocked_tensors.push_back(t);
                seen.insert(t);
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
