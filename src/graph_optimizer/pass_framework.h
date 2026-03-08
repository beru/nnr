#pragma once
// Pass framework for graph_optimizer_t.
//
// Models ORT's GraphTransformer/GraphTransformerManager, simplified for NNR.
// A pass_t is a self-contained graph transformation; pass_manager_t owns a
// set of passes and runs them level-by-level with fixed-point iteration.
//
// In M1 only decompose_ops is registered as a pass_t; everything else still
// runs via direct free-function calls in pass_graph_optimizer_t. M2–M4
// migrate passes one at a time.

#include <cstdint>
#include <memory>
#include <vector>

namespace nnr {

struct context_t;
struct graph_optimizer_t;

// Conceptual grouping — mirrors ORT TransformerLevel:
//   PREPROCESS — shape-independent rewrites (runs before fold_run)
//   FUSION     — shape-dependent fusion (runs after fold_run)
//   LAYOUT     — format assignment (NCHWc / NHWC)
//   SCHEDULING — scroll chain detection, segment pruning
enum class pass_level_t : uint8_t {
    PREPROCESS = 0,
    FUSION     = 1,
    LAYOUT     = 2,
    SCHEDULING = 3,
};

struct pass_t {
    virtual ~pass_t() = default;

    // Apply the pass. Return true if the graph was modified in a way that
    // might enable further opportunities (triggers another fixed-point round).
    // Return false for no-op / idempotent-after-first-run passes.
    virtual bool apply(graph_optimizer_t& opt, context_t* ctx) = 0;

    virtual const char*  name()  const = 0;
    virtual pass_level_t level() const = 0;

    // Passes that are idempotent after round 0 set this true so the
    // fixed-point loop skips them on subsequent rounds.
    virtual bool once() const { return false; }
};

struct pass_manager_t {
    void    add(std::unique_ptr<pass_t> p);
    pass_t* find(const char* name);

    // Runs all passes at `lvl`, iterating until no pass reports a change
    // or `max_rounds` is hit. Returns the number of rounds executed.
    int run_level(pass_level_t lvl, graph_optimizer_t& opt, context_t* ctx,
                  int max_rounds = 8);

private:
    std::vector<std::unique_ptr<pass_t>> passes_;
};

} // namespace nnr
