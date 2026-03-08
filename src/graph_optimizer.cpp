// Top-level graph_optimizer translation unit — class methods only.
// Free pass functions live under src/graph_optimizer/. See
// graph_optimizer_internal.h for the forward declarations.
#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// preprocess() and optimize() are pure virtual on graph_optimizer_t; the sole
// implementation lives in src/graph_optimizer/pass_graph_optimizer.cpp
// (pass_graph_optimizer_t). This file retains the shared services
// (build_plan, build_exec_steps, save, load, reset_formats).

void graph_optimizer_t::build_plan(context_t* ctx)
{
    if (!ctx || !ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());

    plan.resize(n);
    plan_scroll_seg.resize(n);

    // Default: everything is EXEC
    for (int i = 0; i < n; ++i) {
        plan[i] = node_action_t::EXEC;
        plan_scroll_seg[i] = -1;
    }

    // Mark skip and folded nodes
    for (int i = 0; i < n; ++i) {
        if (nodes[i]->skip)
            plan[i] = node_action_t::SKIP;
        else if (nodes[i]->folded)
            plan[i] = node_action_t::FOLDED;
    }

    // Mark scroll segments
    if (scrolling_resolved && scroll_detection_done) {
        for (int s = 0; s < (int)scroll_segments.size(); ++s) {
            auto& seg = scroll_segments[s];
            for (int i = seg.start; i < seg.end; ++i) {
                if (plan[i] == node_action_t::SKIP || plan[i] == node_action_t::FOLDED)
                    continue;  // keep skip/folded as-is
                plan[i] = (i == seg.start) ? node_action_t::SCROLL_START : node_action_t::SCROLL_INSIDE;
                if (i == seg.start)
                    plan_scroll_seg[i] = (int16_t)s;
            }
        }
    }

    plan_built = true;
    // Formats must be set before build_exec_steps reads them to decide
    // FLAG_WANTS_BLOCKED / FLAG_WANTS_NHWC. Without this, tensors are still
    // NCHW from fold_run and every NCHWc Conv triggers a spurious B16→NCHW
    // boundary reorder on its inputs.
    reset_formats();
    build_exec_steps(ctx);
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
// Binary format:
//   [0..7]   magic   "NNR_OPT\0"
//   [8..11]  version (uint32_t, currently 1)
//   [12..15] num_segments (uint32_t)
//   For each segment:
//     [+0..+3]  start       (int32_t)
//     [+4..+7]  end         (int32_t)
//     [+8..+11] strip_height(int32_t)
//
// Fusion state is NOT serialized — fusion modifies weights in-place and
// must re-run each time the model is loaded.  Only scroll segments (the
// expensive O(n²) detection) are cached.

static constexpr char     OPTCACHE_MAGIC[8] = {'N','N','R','_','O','P','T','\0'};
static constexpr uint32_t OPTCACHE_VERSION  = 1;

bool graph_optimizer_t::save(const char* path) const
{
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    fwrite(OPTCACHE_MAGIC, 1, 8, f);
    uint32_t ver = OPTCACHE_VERSION;
    fwrite(&ver, 4, 1, f);

    uint32_t nseg = (uint32_t)scroll_segments.size();
    fwrite(&nseg, 4, 1, f);
    for (auto& seg : scroll_segments) {
        int32_t vals[3] = { seg.start, seg.end, seg.strip_height };
        fwrite(vals, 4, 3, f);
    }

    fclose(f);
    return true;
}

bool graph_optimizer_t::load(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, OPTCACHE_MAGIC, 8) != 0) {
        fclose(f);
        return false;
    }

    uint32_t ver;
    if (fread(&ver, 4, 1, f) != 1 || ver != OPTCACHE_VERSION) {
        fclose(f);
        return false;
    }

    uint32_t nseg;
    if (fread(&nseg, 4, 1, f) != 1 || nseg > 10000) {
        fclose(f);
        return false;
    }

    scroll_segments.resize(nseg);
    for (uint32_t i = 0; i < nseg; ++i) {
        int32_t vals[3];
        if (fread(vals, 4, 3, f) != 3) {
            fclose(f);
            scroll_segments.clear();
            return false;
        }
        scroll_segments[i] = { vals[0], vals[1], vals[2] };
    }

    fclose(f);
    scroll_detection_done = true;  // skip detect_scroll_chains()
    return true;
}

void graph_optimizer_t::reset_formats()
{
    for (auto* t : nhwc_tensors)
        t->format = memory_layout_t::NHWC;
    for (auto* t : blocked_tensors)
        t->format = NATIVE_BLOCKED_FMT;
}

void graph_optimizer_t::build_exec_steps(context_t* ctx)
{
    if (!ctx || !ctx->graph || !plan_built) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());

    exec_steps.clear();
    exec_steps.reserve(n / 2);  // typically ~50% are EXEC

    for (int i = 0; i < n; ++i) {
        auto action = plan[i];

        // SKIP: forward data pointer (must stay in the loop)
        if (action == node_action_t::SKIP) {
            auto* nd = nodes[i];
            if (!nd->inputs.empty() && !nd->outputs.empty()
                && nd->inputs[0] && nd->outputs[0]) {
                exec_step_t step;
                step.node_idx = i;
                step.op = nd;
                step.flags = FLAG_NONE;
                step.scroll_seg = -2;  // sentinel: SKIP action
                exec_steps.push_back(step);
            }
            continue;
        }

        if (action == node_action_t::FOLDED || action == node_action_t::SCROLL_INSIDE)
            continue;

        auto* nd = nodes[i];
        uint8_t flags = FLAG_NONE;

        if (!nd->outputs.empty() && nd->outputs[0]) {
            if (nd->outputs[0]->format == memory_layout_t::NHWC)
                flags |= FLAG_WANTS_NHWC;
            if (nd->outputs[0]->format == NATIVE_BLOCKED_FMT
                && NATIVE_BLOCKED_FMT != memory_layout_t::NCHW)
                flags |= FLAG_WANTS_BLOCKED;
        }

        if (nd->layout_mask == LAYOUT_ALL)
            flags |= FLAG_LAYOUT_ALL;

        // Pre-check broadcast inputs (for BLOCKED_16 propagation safety)
        for (auto* t : nd->inputs) {
            if (t && t->ndim > 0 && t->ndim < 4 && t->ndata > 1) {
                flags |= FLAG_HAS_BROADCAST;
                break;
            }
        }

        exec_step_t step;
        step.node_idx = i;
        step.op = nd;
        step.flags = flags;
        step.scroll_seg = (action == node_action_t::SCROLL_START)
            ? plan_scroll_seg[i] : -1;
        exec_steps.push_back(step);
    }
}


} // namespace nnr
