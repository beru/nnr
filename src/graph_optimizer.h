#pragma once
// Graph optimizer — fusion passes, scroll chain detection, exec plan.
//
// Owns all optimization state that was previously scattered across context_t.
// Usage:
//   ctx.optimizer->enable_fusion();       // enable (default)
//   ctx.optimizer->enable_scrolling();    // opt-in
//   // After first run:
//   ctx.optimizer->optimize(&ctx);        // runs fusion + scroll detection
//   ctx.optimizer->build_plan(&ctx);      // builds per-node exec plan

#include "nnrconf.h"  // small_vector
#include <memory>
#include <vector>

namespace nnr {

struct context_t;
struct operator_t;

// Per-node action in the execution plan.
enum class node_action_t : uint8_t {
    EXEC,            // normal execution
    SKIP,            // skip (fused into predecessor), forward data pointer
    FOLDED,          // constant-folded, output already computed
    SCROLL_START,    // first node of a scroll segment — triggers strip exec
    SCROLL_INSIDE,   // inside a scroll segment — already executed by SCROLL_START
};

// Scroll segment descriptor.
struct scroll_segment_t {
    int start = 0;           // first node index (inclusive)
    int end = 0;             // last node index (exclusive)
    int strip_height = 0;    // chosen strip height (output rows per iteration)
};

// Scrolling mode: AUTO detects and enables only when beneficial.
enum class scroll_mode_t : uint8_t { AUTO, ON, OFF };

struct graph_optimizer_t {
    virtual ~graph_optimizer_t() = default;

    // --- Configuration (set before first run) ---
    bool fusion_enabled = true;
    scroll_mode_t scroll_mode = scroll_mode_t::AUTO;
    bool debug_layout = false;    // print NHWC/NCHWc chain decisions to stderr
    bool no_blocked = false;      // disable NCHWc blocked layout
    bool no_nhwc = false;         // disable NHWC layout
    bool force_nchwc = false;     // force NCHWc layout (ignore cost model penalties)

    // --- Enable helpers ---
    void enable_fusion(bool enable = true) { fusion_enabled = enable; }
    // Force scrolling on/off. Default AUTO enables only when beneficial segments are found.
    void enable_scrolling(bool enable = true) { scroll_mode = enable ? scroll_mode_t::ON : scroll_mode_t::OFF; }
    void set_scroll_mode(scroll_mode_t mode) { scroll_mode = mode; }

    // After optimize(), true if scrolling will actually be used (resolved from AUTO).
    bool scrolling_active() const { return scrolling_resolved; }

    // Trial-run each segment both ways, discard those that don't help.
    void prune_segments(context_t* ctx);

    // --- Pre-execution graph rewriting (decompose unsupported ops) ---
    // Must be called before first inference.
    // Virtual: implemented by pass_graph_optimizer_t (pass framework).
    virtual void preprocess(context_t* ctx) = 0;

    // --- Run all optimization passes (call after first inference) ---
    // Virtual: same rationale as preprocess().
    virtual void optimize(context_t* ctx) = 0;

    // --- Build execution plan from current graph state ---
    // Must be called after optimize() and whenever skip/folded flags change.
    void build_plan(context_t* ctx);

    // --- Execute a scroll segment strip-by-strip ---
    static bool exec_scroll_segment(context_t* ctx, int seg_start, int seg_end, int strip_height);

    // --- Execution plan (indexed by node index) ---
    std::vector<node_action_t> plan;

    // --- Scroll segments ---
    std::vector<scroll_segment_t> scroll_segments;

    // Node-to-segment mapping: plan_scroll_seg[i] = segment index for SCROLL_START nodes, -1 otherwise
    std::vector<int16_t> plan_scroll_seg;

    // --- Serialization ---
    // Save/load optimizer state (scroll segments) to skip expensive
    // detection on subsequent runs of the same model.
    // Fusion still re-runs on load (it modifies weights in-place).
    bool save(const char* path) const;
    bool load(const char* path);

    // --- Internal state ---
    bool preprocess_done = false;
    bool fusion_done = false;
    bool scroll_detection_done = false;
    bool plan_built = false;
    bool scrolling_resolved = false;  // final decision after AUTO resolution

    // Layout optimization: workspace needed for NHWC↔NCHW reorder at boundaries
    size_t layout_reorder_ws = 0;

    // Tensors assigned NHWC format by assign_layouts().
    // Reset at the start of each run (boundary reorder modifies format to NCHW).
    std::vector<tensor_t*> nhwc_tensors;

    // Tensors assigned BLOCKED_16 format by assign_blocked_layouts().
    std::vector<tensor_t*> blocked_tensors;

    void reset_formats();

    // --- Pre-compiled execution steps (flat, no SKIP/FOLDED/SCROLL_INSIDE) ---
    // Built by build_exec_steps() after build_plan(). Replaces per-node
    // dispatch loop with a compact array of only actionable steps.
    enum exec_flag : uint8_t {
        FLAG_NONE           = 0,
        FLAG_WANTS_NHWC     = 1,  // output assigned NHWC format
        FLAG_WANTS_BLOCKED  = 2,  // output assigned BLOCKED_16 format
        FLAG_LAYOUT_ALL     = 4,  // op is layout-agnostic (element-wise)
        FLAG_HAS_BROADCAST  = 8,  // has non-scalar <4D inputs (blocks BLOCKED_16 propagation)
    };
    struct exec_step_t {
        int node_idx;             // index into nodes[] (for profiler)
        operator_t* op;           // direct pointer — no indirection through nodes[]
        uint8_t flags;            // pre-computed exec_flags
        int8_t scroll_seg;        // scroll segment index (-1 if not SCROLL_START)
    };
    std::vector<exec_step_t> exec_steps;
    void build_exec_steps(context_t* ctx);
};

// Factory: the pass-framework optimizer is the only implementation (M6+).
// All preprocess/optimize passes are registered with pass_manager_t and
// driven by run_level() sweeps, with fixed-point iteration at FUSION level.
std::unique_ptr<graph_optimizer_t> make_pass_graph_optimizer();

} // namespace nnr
