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
// Registered passes (M1: just decompose_ops)
// ---------------------------------------------------------------------------

namespace {

struct decompose_ops_pass_t : pass_t {
    bool apply(graph_optimizer_t& /*opt*/, context_t* ctx) override
    {
        decompose_ops(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "decompose_ops"; }
    pass_level_t level() const override { return pass_level_t::PREPROCESS; }
    bool         once()  const override { return true; }
};

// fuse_conv_bn leaves BN as a SKIP-handled alias to its inputs, and the alias
// copy needs the input tensor to be backed by a real allocation — so it must
// run at FUSION level (post fold_run), not PREPROCESS. Otherwise the SKIP
// aliasing reads null data.
struct fuse_conv_bn_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        fuse_conv_bn(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "fuse_conv_bn"; }
    pass_level_t level() const override { return pass_level_t::FUSION; }
    bool         once()  const override { return true; }
};

struct fuse_pad_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        fuse_pad(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "fuse_pad"; }
    pass_level_t level() const override { return pass_level_t::PREPROCESS; }
    bool         once()  const override { return true; }
};

struct fuse_layer_norm_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        fuse_layer_norm(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "fuse_layer_norm"; }
    pass_level_t level() const override { return pass_level_t::PREPROCESS; }
    bool         once()  const override { return true; }
};

struct fuse_gelu_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        fuse_gelu(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "fuse_gelu"; }
    pass_level_t level() const override { return pass_level_t::PREPROCESS; }
    bool         once()  const override { return true; }
};

struct fuse_silu_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        fuse_silu(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "fuse_silu"; }
    pass_level_t level() const override { return pass_level_t::PREPROCESS; }
    bool         once()  const override { return true; }
};

struct fold_bn_qdq_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        fold_bn_qdq(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "fold_bn_qdq"; }
    pass_level_t level() const override { return pass_level_t::PREPROCESS; }
    bool         once()  const override { return true; }
};

// FUSION-level passes use fingerprint-based change detection so run_level()
// can iterate to a fixed point. The free functions are idempotent on a
// stable graph, so calling them in a second round when nothing changed is a
// cheap no-op — they scan and return without touching anything.
struct fold_constants_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        auto before = fingerprint(ctx);
        fold_constants(ctx);
        return fingerprint(ctx) != before;
    }
    const char*  name()  const override { return "fold_constants"; }
    pass_level_t level() const override { return pass_level_t::FUSION; }
    bool         once()  const override { return false; }
};

// Depends on fold_constants having folded weight-side DQ nodes. Registered
// after fold_constants_pass_t and runs in the same FUSION-level sweep.
struct fuse_qdq_compute_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        auto before = fingerprint(ctx);
        fuse_qdq_compute(ctx);
        return fingerprint(ctx) != before;
    }
    const char*  name()  const override { return "fuse_qdq_compute"; }
    pass_level_t level() const override { return pass_level_t::FUSION; }
    bool         once()  const override { return false; }
};

struct fuse_qdq_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        auto before = fingerprint(ctx);
        fuse_qdq(ctx);
        return fingerprint(ctx) != before;
    }
    const char*  name()  const override { return "fuse_qdq"; }
    pass_level_t level() const override { return pass_level_t::FUSION; }
    bool         once()  const override { return false; }
};

// Must run after fold_constants (Clip min/max constants need folding first).
struct fuse_post_ops_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled) return false;
        auto before = fingerprint(ctx);
        fuse_post_ops(ctx);
        return fingerprint(ctx) != before;
    }
    const char*  name()  const override { return "fuse_post_ops"; }
    pass_level_t level() const override { return pass_level_t::FUSION; }
    bool         once()  const override { return false; }
};

struct assign_blocked_layouts_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled || opt.no_blocked) return false;
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
        if (!opt.fusion_enabled || opt.no_nhwc) return false;
        assign_layouts(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "assign_layouts"; }
    pass_level_t level() const override { return pass_level_t::LAYOUT; }
    bool         once()  const override { return true; }
};

// M7+ #2: eliminate redundant NCHW<->NHWC reorders left by assign_layouts.
// Currently a no-op scaffold — real logic lives in optimize_transposes.cpp.
struct optimize_transposes_pass_t : pass_t {
    bool apply(graph_optimizer_t& opt, context_t* ctx) override
    {
        if (!opt.fusion_enabled || opt.no_nhwc) return false;
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
        opt.prune_segments(ctx);
        return false;  // idempotent; no cascade
    }
    const char*  name()  const override { return "prune_segments"; }
    pass_level_t level() const override { return pass_level_t::SCHEDULING; }
    bool         once()  const override { return true; }
};

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
        mgr_.add(std::make_unique<fuse_post_ops_pass_t>());

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
            // fuse_sdpa(ctx);  // TODO: needs non-threaded GEMM to avoid deadlock
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

        // Scroll state finalization — not a pass, just a state-machine reduction
        // over scroll_mode and the post-pass scroll_segments vector.
        switch (scroll_mode) {
        case scroll_mode_t::AUTO: scrolling_resolved = !scroll_segments.empty(); break;
        case scroll_mode_t::ON:   scrolling_resolved = true;  break;
        case scroll_mode_t::OFF:  scrolling_resolved = false; break;
        }
    }

protected:
    pass_manager_t mgr_;
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
        mgr_.add(std::make_unique<optimize_transposes_pass_t>());

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
        mgr_.add(std::make_unique<optimize_transposes_pass_t>());

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
