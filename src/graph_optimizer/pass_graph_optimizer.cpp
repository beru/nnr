// Pass-framework graph_optimizer_t subclass.
//
// History:
//  - M1: scaffold, one pass registered, rest fall through to legacy free fns.
//  - M2: every pass from preprocess()/optimize() migrated to a pass_t subclass.
//        preprocess() / optimize() bodies collapsed to mgr_.run_level() calls
//        over PREPROCESS / FUSION / LAYOUT / SCHEDULING levels.
//  - M3: FUSION-level passes flipped to once()=false with fingerprint-based
//        change detection, enabling fixed-point iteration so cascading
//        fusions don't require hand-ordered sequencing.
//
// Remaining inline code:
//  - preprocess(): AveragePool re-init loop after fuse_pad (state reduction,
//    not a graph transform).
//  - optimize():   scroll_mode → scrolling_resolved switch (state-machine
//    finalization after SCHEDULING passes).

#include "graph_optimizer/graph_optimizer_internal.h"
#include "graph_optimizer/pass_framework.h"
#include "cpu_features.h"  // NNR_ARCH_X64 / NNR_ARCH_ARM64
#include "nnr.h"

#include <cstdint>
#include <cstring>

namespace nnr {

// ---------------------------------------------------------------------------
// Change-detection helper
// ---------------------------------------------------------------------------
//
// Legacy fusion free functions return void, so the pass wrappers can't
// directly tell pass_manager_t::run_level() whether anything was mutated.
// Instead, we snapshot a coarse fingerprint of the graph (node count + the
// skip/folded flag distribution) before and after each apply(), and signal
// "changed" when the fingerprint moves. This is sufficient for fixed-point
// iteration at FUSION level: every fusion/fold we currently run flips at
// least one skip or folded bit when it succeeds.
//
// The fingerprint intentionally ignores attribute-level edits (layout hints,
// etc.) — those are handled by once() passes that don't need iteration.
struct graph_fingerprint_t {
    uint64_t node_count = 0;
    uint64_t skip_count = 0;
    uint64_t folded_count = 0;

    bool operator==(const graph_fingerprint_t& o) const
    {
        return node_count == o.node_count
            && skip_count == o.skip_count
            && folded_count == o.folded_count;
    }
    bool operator!=(const graph_fingerprint_t& o) const { return !(*this == o); }
};

static graph_fingerprint_t fingerprint(context_t* ctx)
{
    graph_fingerprint_t fp;
    if (!ctx || !ctx->graph) return fp;
    fp.node_count = ctx->graph->nodes.size();
    for (auto* n : ctx->graph->nodes) {
        if (n->skip)   ++fp.skip_count;
        if (n->folded) ++fp.folded_count;
    }
    return fp;
}

// ---------------------------------------------------------------------------
// pass_manager_t implementation
// ---------------------------------------------------------------------------

void pass_manager_t::add(std::unique_ptr<pass_t> p)
{
    passes_.push_back(std::move(p));
}

pass_t* pass_manager_t::find(const char* name)
{
    for (auto& p : passes_) {
        if (std::strcmp(p->name(), name) == 0) return p.get();
    }
    return nullptr;
}

int pass_manager_t::run_level(pass_level_t lvl, graph_optimizer_t& opt,
                               context_t* ctx, int max_rounds)
{
    int round = 0;
    for (; round < max_rounds; ++round) {
        bool changed = false;
        for (auto& p : passes_) {
            if (p->level() != lvl) continue;
            if (round > 0 && p->once()) continue;
            if (p->apply(opt, ctx)) changed = true;
        }
        if (!changed) break;
    }
    return round + 1;
}

// ---------------------------------------------------------------------------
// Registered passes
// ---------------------------------------------------------------------------
//
// The vast majority of passes follow one of three apply() shapes:
//
//   SIMPLE        — no gate; just call the free function. (decompose_ops,
//                   the WebGPU passes — those gate themselves internally on
//                   preferred_backend.)
//   GATED         — return false unless opt.fusion_enabled. (Most fusion /
//                   layout helpers.)
//   FUSION_FP     — gated, FUSION level, fingerprint-cascade for fixed-point
//                   iteration.
//
// The handful of passes with compound gates (assign_*_layouts,
// optimize_transposes, detect_scroll_chains, prune_segments) stay
// hand-written below.

namespace {

#define DEFINE_SIMPLE_PASS(CLASS_NAME, FUNC, NAME_STR, LVL, ONCE)              \
    struct CLASS_NAME : pass_t {                                               \
        bool apply(graph_optimizer_t&, context_t* ctx) override                \
        {                                                                      \
            FUNC(ctx);                                                         \
            return false;                                                      \
        }                                                                      \
        const char*  name()  const override { return NAME_STR; }               \
        pass_level_t level() const override { return pass_level_t::LVL; }      \
        bool         once()  const override { return ONCE; }                   \
    }

#define DEFINE_GATED_PASS(CLASS_NAME, FUNC, NAME_STR, LVL, ONCE)               \
    struct CLASS_NAME : pass_t {                                               \
        bool apply(graph_optimizer_t& opt, context_t* ctx) override            \
        {                                                                      \
            if (!opt.fusion_enabled) return false;                             \
            FUNC(ctx);                                                         \
            return false;                                                      \
        }                                                                      \
        const char*  name()  const override { return NAME_STR; }               \
        pass_level_t level() const override { return pass_level_t::LVL; }      \
        bool         once()  const override { return ONCE; }                   \
    }

// FUSION-level passes use fingerprint-based change detection so run_level()
// can iterate to a fixed point. The free functions are idempotent on a
// stable graph, so calling them in a second round when nothing changed is a
// cheap no-op — they scan and return without touching anything.
#define DEFINE_FUSION_FP_PASS(CLASS_NAME, FUNC, NAME_STR)                      \
    struct CLASS_NAME : pass_t {                                               \
        bool apply(graph_optimizer_t& opt, context_t* ctx) override            \
        {                                                                      \
            if (!opt.fusion_enabled) return false;                             \
            auto before = fingerprint(ctx);                                    \
            FUNC(ctx);                                                         \
            return fingerprint(ctx) != before;                                 \
        }                                                                      \
        const char*  name()  const override { return NAME_STR; }               \
        pass_level_t level() const override                                    \
        {                                                                      \
            return pass_level_t::FUSION;                                       \
        }                                                                      \
        bool once() const override { return false; }                           \
    }

DEFINE_SIMPLE_PASS(decompose_ops_pass_t, decompose_ops, "decompose_ops",
                   PREPROCESS, true);

// fuse_conv_bn leaves BN as a SKIP-handled alias to its inputs, and the alias
// copy needs the input tensor to be backed by a real allocation — so it must
// run at FUSION level (post fold_run), not PREPROCESS. Otherwise the SKIP
// aliasing reads null data.
DEFINE_GATED_PASS(fuse_conv_bn_pass_t, fuse_conv_bn, "fuse_conv_bn",
                  FUSION, true);

DEFINE_GATED_PASS(fuse_pad_pass_t, fuse_pad, "fuse_pad", PREPROCESS, true);
DEFINE_GATED_PASS(fuse_layer_norm_pass_t, fuse_layer_norm, "fuse_layer_norm",
                  PREPROCESS, true);
DEFINE_GATED_PASS(fuse_gelu_pass_t, fuse_gelu, "fuse_gelu", PREPROCESS, true);
DEFINE_GATED_PASS(fuse_silu_pass_t, fuse_silu, "fuse_silu", PREPROCESS, true);

// FUSION (not PREPROCESS): needs shape inference to have run so
// q_tensor->dims[...] is populated for the shape-compatibility checks.
DEFINE_GATED_PASS(fuse_sdpa_pass_t, fuse_sdpa, "fuse_sdpa", FUSION, true);

// WebGPU-only: fold consecutive same-shape f32 unary elementwise ops into one
// dispatch. Gated internally on ctx->preferred_backend == WEBGPU, so it's a
// no-op for the CPU backend even when fusion is enabled. Kept independent of
// opt.fusion_enabled — that flag governs the CPU-side rewrite set and is
// often disabled for WebGPU paths (see webgpu_e2e_graph.cpp) while the
// WebGPU-native fusion below is always desirable.
//
// Registered at FUSION level, not PREPROCESS: PREPROCESS only runs from
// context_t::run()'s first_run path, not from prepare() — callers that call
// prepare() before run() (the e2e tests do) would otherwise silently skip
// fusion. FUSION runs via both optimize() paths and post fold_run, which
// has the added benefit of guaranteeing all input tensor shapes are
// propagated by the time we build the fused op.
DEFINE_SIMPLE_PASS(fuse_webgpu_elementwise_pass_t, fuse_webgpu_elementwise,
                   "fuse_webgpu_elementwise", FUSION, true);

// WebGPU-only: absorb a FusedElementwiseChain that directly follows a
// MatMul into the MatMul's output epilogue. Must run AFTER
// fuse_webgpu_elementwise has built the chain nodes (registration order
// controls FUSION-level ordering within a round). Gated internally on
// preferred_backend == WEBGPU.
DEFINE_SIMPLE_PASS(fuse_webgpu_matmul_chain_pass_t, fuse_webgpu_matmul_chain,
                   "fuse_webgpu_matmul_chain", FUSION, true);

DEFINE_GATED_PASS(fold_bn_qdq_pass_t, fold_bn_qdq, "fold_bn_qdq",
                  PREPROCESS, true);

DEFINE_FUSION_FP_PASS(fold_constants_pass_t, fold_constants, "fold_constants");

// Depends on fold_constants having folded weight-side DQ nodes. Registered
// after fold_constants_pass_t and runs in the same FUSION-level sweep.
DEFINE_FUSION_FP_PASS(fuse_qdq_compute_pass_t, fuse_qdq_compute,
                      "fuse_qdq_compute");

DEFINE_FUSION_FP_PASS(fuse_qdq_pass_t, fuse_qdq, "fuse_qdq");

// Must run after fold_constants (Clip min/max constants need folding first).
DEFINE_FUSION_FP_PASS(fuse_post_ops_pass_t, fuse_post_ops, "fuse_post_ops");

// Layout rewrites (NCHWc blocked, NHWC) are CPU-specific: the AVX/NEON
// kernels are packed-aware, but the WebGPU Conv/BN/Pool kernels assume
// plain NCHW. Skip these passes when the preferred backend is WebGPU,
// otherwise the graph gets rewritten into layouts the WebGPU ops can't
// consume and performance collapses (measured ~35% slowdown on cnn_bench
// before this gate landed).
static bool skip_for_webgpu(context_t* ctx) {
    return ctx->preferred_backend == static_cast<uint8_t>(backend_t::WEBGPU);
}

struct assign_blocked_layouts_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled || opt.no_blocked || skip_for_webgpu(ctx)) return false;
        assign_blocked_layouts(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "assign_blocked_layouts"; }
    pass_level_t level() const override { return pass_level_t::LAYOUT; }
    bool         once()  const override { return true; }
};

struct assign_layouts_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled || opt.no_nhwc || skip_for_webgpu(ctx)) return false;
        assign_layouts(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "assign_layouts"; }
    pass_level_t level() const override { return pass_level_t::LAYOUT; }
    bool         once()  const override { return true; }
};

// T3 M1 step 1: skeleton passes for explicit Reorder ops. Both are no-ops
// when NNR_EXPLICIT_REORDERS is OFF (the M1 default). Wired into LAYOUT
// level after assign_*_layouts and before optimize_transposes so the
// transpose-elimination cleanup still runs on the post-reorder graph.
DEFINE_GATED_PASS(insert_reorders_pass_t, insert_reorders, "insert_reorders",
                  LAYOUT, true);
DEFINE_GATED_PASS(cancel_reorders_pass_t, cancel_reorders, "cancel_reorders",
                  LAYOUT, true);

// SiLU post-op fusion. Runs at LAYOUT (not FUSION) so declared_layout is
// committed before we decide whether to bridge a Conv→Sigmoid SKIP-alias —
// see kb/conv_silu_fusion_blocker.md.
DEFINE_GATED_PASS(fuse_post_ops_silu_pass_t, fuse_post_ops_silu,
                  "fuse_post_ops_silu", LAYOUT, true);

// M7+ #2: eliminate redundant NCHW<->NHWC reorders left by assign_layouts.
// Currently a no-op scaffold — real logic lives in optimize_transposes.cpp.
struct optimize_transposes_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled || opt.no_nhwc || skip_for_webgpu(ctx)) return false;
        optimize_transposes(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "optimize_transposes"; }
    pass_level_t level() const override { return pass_level_t::LAYOUT; }
    bool         once()  const override { return true; }
};

struct detect_scroll_chains_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (opt.scroll_mode == scroll_mode_t::OFF) return false;
        if (opt.scroll_detection_done) return false;
        detect_scroll_chains(&opt, ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "detect_scroll_chains"; }
    pass_level_t level() const override { return pass_level_t::SCHEDULING; }
    bool         once()  const override { return true; }
};

// AUTO mode only: prune empty/expensive segments after detection.
struct prune_segments_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (opt.scroll_mode != scroll_mode_t::AUTO) return false;
        if (opt.scroll_segments.empty()) return false;
        // Skip on shape-change rerun: prune_segments executes ops to time
        // them, but the memory plan + activation buffers are mid-rebuild
        // and ops can read uninitialized pool slots → crash. The
        // first-run prune decision (which segments are profitable) holds
        // for shape-stable reruns; only strip_height needs refresh, and
        // that's done by detect_scroll_chains.
        if (opt.scheduling_in_rerun) return false;
        opt.prune_segments(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "prune_segments"; }
    pass_level_t level() const override { return pass_level_t::SCHEDULING; }
    bool         once()  const override { return true; }
};

#undef DEFINE_SIMPLE_PASS
#undef DEFINE_GATED_PASS
#undef DEFINE_FUSION_FP_PASS

// ---------------------------------------------------------------------------
// pass_graph_optimizer_t — arch-agnostic base
// ---------------------------------------------------------------------------
//
// As of M3, preprocess() and optimize() are driven entirely by run_level()
// calls; only the AveragePool re-init loop and the scroll state-machine
// reduction remain inline. FUSION passes iterate to a fixed point via
// fingerprint-based change detection.
//
// M2.5 split: the base class owns the pass_manager_t and registers passes
// whose behavior is arch-agnostic (decompose, fold, fusion math, transpose
// elimination). Arch-specific subclasses below register the passes that
// need a per-architecture cost model — today just assign_blocked_layouts
// and the scroll chain detection/pruning pair. The subclass constructor
// runs after this one, so its registrations append to mgr_ in order.
struct pass_graph_optimizer_t : graph_optimizer_t {
    pass_graph_optimizer_t()
    {
        // PREPROCESS level — graph rewrites independent of shape/data.
        mgr_.add(std::make_unique<decompose_ops_pass_t>());
        mgr_.add(std::make_unique<fuse_pad_pass_t>());
        mgr_.add(std::make_unique<fuse_layer_norm_pass_t>());
        mgr_.add(std::make_unique<fuse_gelu_pass_t>());
        mgr_.add(std::make_unique<fuse_silu_pass_t>());
        mgr_.add(std::make_unique<fold_bn_qdq_pass_t>());

        // FUSION level — shape-dependent rewrites, run post fold_run.
        mgr_.add(std::make_unique<fuse_conv_bn_pass_t>());
        mgr_.add(std::make_unique<fold_constants_pass_t>());
        mgr_.add(std::make_unique<fuse_qdq_compute_pass_t>());
        mgr_.add(std::make_unique<fuse_qdq_pass_t>());
        mgr_.add(std::make_unique<fuse_sdpa_pass_t>());
        mgr_.add(std::make_unique<fuse_post_ops_pass_t>());
        mgr_.add(std::make_unique<fuse_webgpu_elementwise_pass_t>());
        mgr_.add(std::make_unique<fuse_webgpu_matmul_chain_pass_t>());

        // LAYOUT and SCHEDULING are registered by the arch-specific subclass
        // because the blocked-layout cost model differs per arch, and order
        // within the level matters (blocked before non-blocked, non-blocked
        // before transpose elimination).
    }

    void preprocess(context_t* ctx) override
    {
        if (!ctx || !ctx->graph) return;
        if (preprocess_done) return;

        // Run registered PREPROCESS-level passes.
        mgr_.run_level(pass_level_t::PREPROCESS, *this, ctx);

        // Inline tail: AveragePool re-init after fuse_pad may have changed
        // count_include_pad. Not a pass (per-node reduction, not a graph
        // transform).
        if (fusion_enabled) {
            // Re-init AveragePool: Pad fusion may have changed count_include_pad.
            for (auto* n : ctx->graph->nodes) {
                if (!n->skip && !n->folded && n->op_type == "AveragePool")
                    n->init();
            }
        }
        preprocess_done = true;
    }

    void optimize(context_t* ctx) override
    {
        if (!ctx || !ctx->graph) return;
        if (!fusion_done) {
            // Run registered FUSION and LAYOUT passes.
            mgr_.run_level(pass_level_t::FUSION, *this, ctx);
            mgr_.run_level(pass_level_t::LAYOUT, *this, ctx);
            fusion_done = true;
        }
        // Run registered SCHEDULING passes (detect_scroll_chains, prune_segments).
        mgr_.run_level(pass_level_t::SCHEDULING, *this, ctx);

        resolve_scrolling();
    }

    void rerun_after_shape_change(context_t* ctx) override
    {
        if (!ctx || !ctx->graph) return;

        // Reset SCHEDULING-level state. detect_scroll_chains gates on
        // scroll_detection_done and clears scroll_segments on entry; clearing
        // both here forces a re-derivation against the new shapes. plan +
        // exec_steps must also be invalidated because they embed the
        // strip_height / segment boundaries.
        scroll_detection_done = false;
        scrolling_resolved    = false;
        scroll_segments.clear();
        plan_scroll_seg.clear();
        plan.clear();
        exec_steps.clear();
        plan_built            = false;

        // Re-run SCHEDULING passes only. PREPROCESS / FUSION / LAYOUT stay
        // committed (graph mutations: decompose, fuse_*, insert_reorders).
        // Set scheduling_in_rerun so prune_segments skips its trial-run
        // (buffers are mid-rebuild — see prune_segments_pass_t::apply).
        scheduling_in_rerun = true;
        mgr_.run_level(pass_level_t::SCHEDULING, *this, ctx);
        scheduling_in_rerun = false;
        resolve_scrolling();

        // Rebuild plan + exec_steps from the new scroll segments.
        build_plan(ctx);
    }

protected:
    pass_manager_t mgr_;

private:
    // Scroll state finalization — not a pass, just a state-machine reduction
    // over scroll_mode and the post-pass scroll_segments vector.
    void resolve_scrolling()
    {
        switch (scroll_mode) {
        case scroll_mode_t::AUTO: scrolling_resolved = !scroll_segments.empty(); break;
        case scroll_mode_t::ON:   scrolling_resolved = true;  break;
        case scroll_mode_t::OFF:  scrolling_resolved = false; break;
        }
    }
};

// ---------------------------------------------------------------------------
// Arch-specific subclasses
// ---------------------------------------------------------------------------
// Each arch registers its own blocked-layout and scroll passes. Today both
// subclasses register the same pass types; the split exists so a future
// M2.6+ session can swap in e.g. an arm_assign_blocked_layouts_pass_t with
// an ARM-aware chain-boundary cost model without touching x64.
//
// Only one subclass is compiled per arch (compile-time dispatch via
// NNR_ARCH_*), matching the kernel layer in src/backend/{x64,arm}/.

#ifdef NNR_ARCH_X64
struct x64_pass_graph_optimizer_t final : pass_graph_optimizer_t {
    x64_pass_graph_optimizer_t()
    {
        // LAYOUT — order matters: blocked assignment must precede the
        // non-blocked layout pass because the latter reads BLOCKED bits
        // set by the former. optimize_transposes cleans up NCHW↔NHWC
        // reorder pairs left by assign_layouts.
        mgr_.add(std::make_unique<assign_blocked_layouts_pass_t>());
        mgr_.add(std::make_unique<assign_layouts_pass_t>());
        // T3 M1: explicit Reorder ops (skeleton; no-op when flag OFF).
        mgr_.add(std::make_unique<insert_reorders_pass_t>());
        mgr_.add(std::make_unique<cancel_reorders_pass_t>());
        mgr_.add(std::make_unique<optimize_transposes_pass_t>());
        // SiLU fusion runs after layout commit so we can gate on layout
        // agreement. Must precede detect_scroll_chains so the scroll pass
        // sees the post-fuse skip flags.
        mgr_.add(std::make_unique<fuse_post_ops_silu_pass_t>());

        // SCHEDULING — scroll detection + segment pruning.
        mgr_.add(std::make_unique<detect_scroll_chains_pass_t>());
        mgr_.add(std::make_unique<prune_segments_pass_t>());
    }
};
#elifdef NNR_ARCH_ARM64
struct arm_pass_graph_optimizer_t final : pass_graph_optimizer_t {
    arm_pass_graph_optimizer_t()
    {
        // LAYOUT — same pass types as x64 today. The slot for an
        // arm_assign_blocked_layouts_pass_t with a NEON-aware chain-boundary
        // cost model is here when M2.6 lands.
        mgr_.add(std::make_unique<assign_blocked_layouts_pass_t>());
        mgr_.add(std::make_unique<assign_layouts_pass_t>());
        // T3 M1: explicit Reorder ops (skeleton; no-op when flag OFF).
        mgr_.add(std::make_unique<insert_reorders_pass_t>());
        mgr_.add(std::make_unique<cancel_reorders_pass_t>());
        mgr_.add(std::make_unique<optimize_transposes_pass_t>());
        mgr_.add(std::make_unique<fuse_post_ops_silu_pass_t>());

        // SCHEDULING — scroll detection + segment pruning.
        mgr_.add(std::make_unique<detect_scroll_chains_pass_t>());
        mgr_.add(std::make_unique<prune_segments_pass_t>());
    }
};
#endif

} // namespace

std::unique_ptr<graph_optimizer_t> make_pass_graph_optimizer()
{
#ifdef NNR_ARCH_X64
    return std::make_unique<x64_pass_graph_optimizer_t>();
#elifdef NNR_ARCH_ARM64
    return std::make_unique<arm_pass_graph_optimizer_t>();
#else
#error "NNR_ARCH_X64 or NNR_ARCH_ARM64 must be defined"
#endif
}

} // namespace nnr
