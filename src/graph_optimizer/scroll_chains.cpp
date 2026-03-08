// this file's wrapper. Content below was moved verbatim from the monolithic
// graph_optimizer.cpp (2026-04-10 refactor).
#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// ---------------------------------------------------------------------------
// Scrollable chain detection
// ---------------------------------------------------------------------------
// After fusion and constant folding, identify maximal contiguous segments of
// operators that support strip-based (scrolling) execution. Each segment is
// a linear chain where:
//   1. Every op declares scroll_info().scrollable == true
//   2. Each op's output feeds exactly one consumer (the next op)
//   3. All ops work on 4D NCHW tensors (spatial decomposable)
//   4. The chain has at least 2 ops (single op has no benefit)
//
// Chain detection is conservative: any op that breaks these rules becomes
// a segment boundary.

void detect_scroll_chains(graph_optimizer_t* opt, context_t* ctx)
{
    if (!ctx->graph) return;
    if (opt->scroll_detection_done) return;
    opt->scroll_segments.clear();

    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());
    if (n < 2) { opt->scroll_detection_done = true; return; }

    // Count consumers of each tensor. Skip nodes with empty inputs
    // (e.g., bypassed Pads whose inputs were cleared) are excluded since
    // they no longer represent a real data dependency.
    auto count_users = [&](tensor_t* tensor, int producer_idx) -> int {
        int users = 0;
        for (int j = 0; j < n; ++j) {
            if (j == producer_idx) continue;
            if (nodes[j]->skip && nodes[j]->inputs.empty()) continue;
            for (auto* t : nodes[j]->inputs)
                if (t == tensor) users++;
        }
        // Also check if it's a graph output (must be fully materialized)
        for (auto& name : ctx->graph_outputs)
            if (tensor->name == name) users += 100;
        return users;
    };

    int chain_start = -1;

    auto try_emit_chain = [&](int chain_end) {
        // chain_end is the last node in the chain (inclusive)
        if (chain_start >= 0 && chain_end >= chain_start) {
            // Count active (non-skip/folded) ops — need at least 2
            int active_count = 0;
            for (int j = chain_start; j <= chain_end; ++j)
                if (!nodes[j]->skip && !nodes[j]->folded) active_count++;
            if (active_count < 2) { chain_start = -1; return; }

            int seg_start = chain_start;
            int seg_end = chain_end + 1;  // exclusive

            // Auto strip_height: pick the largest strip that keeps the widest
            // intermediate tensor's strip in L2.  Target: one strip of the
            // largest intermediate ≤ 256 KB (half a typical per-core L2).
            int max_CW = 0;
            for (int i = seg_start; i < seg_end; ++i) {
                if (nodes[i]->skip || nodes[i]->folded) continue;
                auto* t = nodes[i]->outputs[0];
                if (t && t->ndim == 4) {
                    int CW = t->dims[1] * t->dims[3]; // C * W
                    if (CW > max_CW) max_CW = CW;
                }
            }
            // Check if the chain contains a Conv (Winograd needs larger strips
            // for efficient 36-GEMM batches — playground sweep shows strip_h≥20
            // is needed to break even, with 20-24 being the sweet spot).
            bool has_conv = false;
            for (int i = seg_start; i < seg_end; ++i) {
                if (nodes[i]->skip || nodes[i]->folded) continue;
                if (nodes[i]->op_type == "Conv") { has_conv = true; break; }
            }

            int strip_h;
            if (max_CW > 0) {
                // strip_h * C * W * 4 bytes ≤ 256 KB
                strip_h = (256 * 1024) / (max_CW * (int)sizeof(float));
                strip_h = std::max(strip_h, 2);  // minimum 2 rows
                strip_h = std::min(strip_h, 32); // cap at 32
            } else {
                strip_h = 8;
            }
            // Winograd Conv: raise floor to 20, align to tile height (4 rows)
            // to eliminate tile-row overlap and keep GEMMs efficient.
            if (has_conv) {
                strip_h = std::max(strip_h, 20);
                strip_h = (strip_h + 3) & ~3;  // round up to multiple of 4
            }

            // Verify all active ops in the segment have 4D float32 tensors
            bool valid = true;
            for (int i = seg_start; i < seg_end; ++i) {
                auto* op = nodes[i];
                if (op->skip || op->folded) continue;
                if (op->outputs.empty() || !op->outputs[0]) { valid = false; break; }
                if (op->outputs[0]->ndim != 4) { valid = false; break; }
                if (op->inputs.empty() || !op->inputs[0]) { valid = false; break; }
                if (op->inputs[0]->ndim != 4) { valid = false; break; }
                // Only float32 for now
                if (op->outputs[0]->type != NNR_DATA_TYPE_FLOAT32) { valid = false; break; }
            }
            if (valid) {
                opt->scroll_segments.push_back({seg_start, seg_end, strip_h});
            }
        }
        chain_start = -1;
    };

    for (int i = 0; i < n; ++i) {
        auto* op = nodes[i];
        if (op->skip || op->folded) {
            // Folded/skipped nodes don't participate in chains but don't
            // break them — the executor skips them, and the prev_idx scan
            // (line 122) already walks backwards over them.
            continue;
        }

        auto info = op->scroll_info();
        if (!info.scrollable) {
            try_emit_chain(i - 1);
            continue;
        }

        // Limit: at most one Conv per chain.  Strip execution overhead
        // (36 Winograd GEMMs, im2col workspace) compounds with multiple
        // Convs; chains are designed for one heavy Conv + light post-ops.
        if (chain_start >= 0 && op->op_type == "Conv") {
            bool has_conv = false;
            for (int j = chain_start; j < i; ++j)
                if (!nodes[j]->skip && !nodes[j]->folded
                    && nodes[j]->op_type == "Conv")
                    { has_conv = true; break; }
            if (has_conv) {
                try_emit_chain(i - 1);
                chain_start = i;
                continue;
            }
        }

        // Check this op can chain with the previous one
        if (chain_start >= 0) {
            // Verify the previous op's output feeds only into this op
            auto* prev = nodes[i - 1];
            // Handle skipped nodes: find the actual previous non-skipped node
            int prev_idx = i - 1;
            while (prev_idx >= chain_start && (nodes[prev_idx]->skip || nodes[prev_idx]->folded))
                prev_idx--;
            if (prev_idx < chain_start) {
                try_emit_chain(prev_idx);
                chain_start = i;
                continue;
            }
            prev = nodes[prev_idx];

            if (prev->outputs.empty() || !prev->outputs[0]) {
                try_emit_chain(prev_idx - 1);
                chain_start = i;
                continue;
            }

            // Check single-consumer constraint
            if (count_users(prev->outputs[0], prev_idx) != 1) {
                // prev's output has multiple consumers — still include prev as
                // chain end since the last op's output is not ring-buffered.
                try_emit_chain(prev_idx);
                chain_start = i;
                continue;
            }

            // Check data flows from prev to this op, tracing through
            // intermediate skip/folded nodes whose tensor objects differ
            // but whose data pointers are aliased at execution time.
            tensor_t* chain_tensor = prev->outputs[0];
            for (int j = prev_idx + 1; j < i; ++j) {
                auto* mid = nodes[j];
                if (!mid->skip && !mid->folded) break;
                for (size_t k = 0; k < mid->inputs.size(); ++k) {
                    if (mid->inputs[k] == chain_tensor) {
                        chain_tensor = mid->outputs[0];
                        break;
                    }
                }
            }
            int chain_in = -1;
            if (!op->inputs.empty() && op->inputs[0] == chain_tensor)
                chain_in = 0;
            else if (op->inputs.size() == 2 && op->inputs[1] == chain_tensor)
                chain_in = 1;

            if (chain_in < 0) {
                try_emit_chain(prev_idx);
                chain_start = i;
                continue;
            }

            // For 2-input ops (e.g. residual Add): the non-chain input (skip)
            // must be produced before the chain started so it's a full tensor.
            if (op->inputs.size() == 2) {
                tensor_t* skip = op->inputs[1 - chain_in];
                int skip_producer = -1;
                for (int j = 0; j < n; ++j) {
                    for (auto* t : nodes[j]->outputs)
                        if (t == skip) { skip_producer = j; break; }
                    if (skip_producer >= 0) break;
                }
                // skip_producer < 0: graph input/initializer (always safe)
                // skip_producer < chain_start: produced before chain (safe)
                if (skip_producer >= chain_start) {
                    try_emit_chain(prev_idx);
                    chain_start = i;
                    continue;
                }
                // Ensure chain input is inputs[0] for executor compatibility
                if (chain_in == 1)
                    std::swap(op->inputs[0], op->inputs[1]);
            }
        } else {
            chain_start = i;
        }
    }
    try_emit_chain(n - 1);

    opt->scroll_detection_done = true;
}

// ---------------------------------------------------------------------------
// Scroll executor
// ---------------------------------------------------------------------------
// For a detected scrollable segment, execute all ops strip-by-strip instead
// of layer-by-layer. Each strip of output rows is produced by running the
// entire chain on the corresponding input rows (with halos).
//
// Ring buffer optimization: intermediate tensors between ops in the chain are
// replaced with small ring buffers sized to (strip_height + halos) rows.
// This reduces memory from O(H) to O(strip_height) per intermediate, improves
// TLB locality, and avoids L2 writeback pollution from dead strip data.
// The approach uses a "virtual data pointer": tensor->data is set to
// ring_buf - base_row * W so that normal absolute-row addressing in exec_strip
// transparently hits the correct ring buffer position.

bool graph_optimizer_t::exec_scroll_segment(context_t* ctx, int seg_start, int seg_end, int strip_height)
{
    auto& nodes = ctx->graph->nodes;

    // Collect non-skipped ops in the segment
    small_vector<operator_t*, 32> ops;
    for (int i = seg_start; i < seg_end; ++i) {
        if (nodes[i]->skip || nodes[i]->folded) continue;
        ops.push_back(nodes[i]);
    }
    if (ops.size() < 2) return false;  // not worth scrolling

    // Bail out if the segment mixes BLOCKED_16 and non-BLOCKED_16 formats.
    // The ring buffer virtual pointer and exec_strip addressing differ between
    // NCHW and BLOCKED_16 layouts. When assign_blocked_layouts marks a Conv for
    // BLOCKED_16 but the Conv's exec_strip can't produce blocked data (e.g.,
    // NCHWc weights not yet lazy-packed), the NCHW fallback writes wrong-layout
    // data into the ring buffer, corrupting downstream ops.
    if (NATIVE_BLOCKED_FMT != memory_layout_t::NCHW) {
        bool has_blocked = false, has_nchw = false;
        for (auto* op : ops) {
            auto* t = op->outputs[0];
            if (t && t->format == NATIVE_BLOCKED_FMT) has_blocked = true;
            else has_nchw = true;
        }
        if (has_blocked && has_nchw) {
            return false;
        }
    }

    // Get the output height from the last op's output tensor
    auto* last_op = ops[ops.size() - 1];
    int output_H = last_op->outputs[0]->dims[2];

    // Compute compound halos: for each op, how many extra output rows
    // does it need to produce so that all downstream ops have enough input?
    // Work backwards through the chain.
    int num_ops = (int)ops.size();
    small_vector<int, 32> compound_halo_top(num_ops);
    small_vector<int, 32> compound_halo_bot(num_ops);
    memset(compound_halo_top.data(), 0, num_ops * sizeof(int));
    memset(compound_halo_bot.data(), 0, num_ops * sizeof(int));

    // Start from the last op: it needs exactly strip_height output rows.
    // Each op before it needs to produce enough rows to cover the downstream
    // op's input requirement = output_rows * stride + halo.
    for (int k = num_ops - 1; k >= 1; --k) {
        auto info = ops[k]->scroll_info();
        int extra_top = (compound_halo_top[k] + info.halo_top);
        int extra_bot = (compound_halo_bot[k] + info.halo_bottom);
        compound_halo_top[k - 1] = extra_top * info.stride_h;
        compound_halo_bot[k - 1] = extra_bot * info.stride_h;
    }

    // --- Ring buffer setup ---
    // Compute ring_H for each op's output: max rows per strip (via backward
    // expansion from the last op's strip_height through strides and halos).
    small_vector<int, 32> ring_rows(num_ops);
    ring_rows[num_ops - 1] = strip_height;
    for (int k = num_ops - 1; k >= 1; --k) {
        auto info = ops[k]->scroll_info();
        ring_rows[k - 1] = (ring_rows[k] - 1) * info.stride_h
            + info.halo_top + info.halo_bottom + 1;
    }

    // Pre-pass: for ops that need global statistics (e.g., InstanceNorm),
    // run preceding chain ops in full to produce their inputs, then compute stats.
    // Track how far we've fully executed so we can skip ring buffers and strip
    // re-execution for those ops.
    int pre_pass_done_to = -1;
    {
        int done_to = -1;
        for (int k = 0; k < num_ops; ++k) {
            auto info = ops[k]->scroll_info();
            if (!info.needs_pre_pass) continue;
            for (int j = done_to + 1; j < k; ++j)
                ops[j]->exec();
            ops[k]->scroll_pre_exec();
            ops[k]->exec();
            done_to = k;
        }
        pre_pass_done_to = done_to;
    }

    // Allocate ring buffers for intermediate tensors (ops 0..num_ops-2).
    // Skip if: binary fusion present, ring wouldn't save memory, or
    // the op was fully executed in the pre-pass.
    //
    // All ring buffers are allocated as a single arena block to avoid the
    // arena reallocating its backing buffer between individual alloc() calls,
    // which would invalidate previously returned pointers.
    struct ring_alloc_t {
        void* buf = nullptr;
        int ring_H = 0;
        int base_row = 0;       // previous strip's base_row (for halo copy offset)
        void* saved_data = nullptr;
        int saved_dims2 = 0;
        size_t saved_ndata = 0;
        memory_layout_t saved_fmt = memory_layout_t::NCHW;
    };
    small_vector<ring_alloc_t, 32> rings(num_ops);
    arena_scope_t ring_arena(ctx->arena);

    // First pass: compute per-ring sizes and total allocation.
    constexpr size_t RING_ALIGN = 64;
    small_vector<size_t, 32> ring_sizes(num_ops);
    memset(ring_sizes.data(), 0, num_ops * sizeof(size_t));
    size_t total_ring_bytes = 0;
    for (int k = 0; k < num_ops - 1; ++k) {
        if (k <= pre_pass_done_to) continue;
        auto* t = ops[k]->outputs[0];
        if (!t || t->ndim != 4) continue;
        if (ops[k]->fused_tensor) continue;
        if (ring_rows[k] >= t->dims[2]) continue;

        size_t sz = (size_t)t->dims[0] * t->dims[1] * ring_rows[k] * t->dims[3]
            * data_type_sizeof(t->type);
        ring_sizes[k] = sz;
        total_ring_bytes += (sz + RING_ALIGN - 1) & ~(RING_ALIGN - 1);
    }

    // Single allocation for all ring buffers.
    char* ring_block = nullptr;
    if (total_ring_bytes > 0) {
        ring_block = (char*)ctx->arena.alloc(total_ring_bytes, RING_ALIGN);
    }

    // Second pass: subdivide the block and fill ring_alloc_t entries.
    if (ring_block) {
        char* cursor = ring_block;
        for (int k = 0; k < num_ops - 1; ++k) {
            if (ring_sizes[k] == 0) continue;
            auto* t = ops[k]->outputs[0];
            rings[k].buf = cursor;
            rings[k].ring_H = ring_rows[k];
            rings[k].saved_data = t->data;
            rings[k].saved_dims2 = t->dims[2];
            rings[k].saved_ndata = t->ndata;
            rings[k].saved_fmt = t->format;
            cursor += (ring_sizes[k] + RING_ALIGN - 1) & ~(RING_ALIGN - 1);
        }
    }

    bool ok = true;

    // Track which rows each op has already computed (persists across strips).
    // valid_end[k] = first row NOT yet computed by op k. Rows [0..valid_end) are valid.
    // Pre-pass ops are marked fully computed so the strip loop skips them.
    small_vector<int, 32> valid_end(num_ops);
    memset(valid_end.data(), 0, num_ops * sizeof(int));
    for (int k = 0; k <= pre_pass_done_to && k < num_ops; ++k) {
        auto* t = ops[k]->outputs[0];
        if (t && t->ndim >= 3)
            valid_end[k] = t->dims[t->ndim - 2];  // full output H
    }

    // Execute strip by strip
    for (int strip_start = 0; strip_start < output_H; strip_start += strip_height) {
        int strip_rows = std::min(strip_height, output_H - strip_start);

        small_vector<int, 32> out_start(num_ops);
        small_vector<int, 32> out_rows(num_ops);

        out_start[num_ops - 1] = strip_start;
        out_rows[num_ops - 1] = strip_rows;

        // Work backwards: compute each op's needed output row range
        for (int k = num_ops - 1; k >= 1; --k) {
            auto info = ops[k]->scroll_info();
            int in_start = out_start[k] * info.stride_h - info.halo_top;
            int in_end = (out_start[k] + out_rows[k] - 1) * info.stride_h + info.halo_bottom + 1;

            // Use original H for boundary clamping (not ring_H)
            int prev_oH = rings[k - 1].buf
                ? rings[k - 1].saved_dims2
                : ops[k - 1]->outputs[0]->dims[2];
            in_start = std::max(0, in_start);
            in_end = std::min(in_end, prev_oH);

            out_start[k - 1] = in_start;
            out_rows[k - 1] = in_end - in_start;
        }

        // Halo reuse: for ring-buffered intermediates, copy overlapping rows
        // from the previous strip's position to the new strip's position.
        // This avoids re-executing ops on rows that are already computed.
        for (int k = 0; k < num_ops - 1; ++k) {
            if (!rings[k].buf) continue;
            int halo_rows = std::max(0, valid_end[k] - out_start[k]);
            if (halo_rows <= 0) continue;

            auto* t = ops[k]->outputs[0];
            int W = t->dims[3];
            size_t elem_sz = data_type_sizeof(t->type);
            // BLOCKED_16/8: row width is W*block per channel block, with
            // N*Cb planes.  NCHW: row width is W per channel, N*C planes.
            // Total bytes are identical: N*C*H*W*elem_sz.
            bool blocked = (rings[k].saved_fmt == NATIVE_BLOCKED_FMT);
            int block = blocked ? NATIVE_BLOCK : 1;
            int NC = blocked
                ? t->dims[0] * (t->dims[1] / block)
                : t->dims[0] * t->dims[1];
            size_t row_bytes = (size_t)W * block * elem_sz;
            int src_off = out_start[k] - rings[k].base_row;  // physical offset of halo start

            for (int nc = 0; nc < NC; ++nc) {
                char* ch_base = (char*)rings[k].buf + (size_t)nc * rings[k].ring_H * row_bytes;
                memmove(ch_base, ch_base + (size_t)src_off * row_bytes,
                    (size_t)halo_rows * row_bytes);
            }
        }

        // Set up ring buffer virtual data pointers for this strip.
        // Virtual pointer trick: set tensor->data = ring_buf - base_row * row_bytes
        // so that indexing at the logical row hits the right ring position.
        // For BLOCKED_16, the row stride per channel block is W*16*elem_sz.
        for (int k = 0; k < num_ops - 1; ++k) {
            if (!rings[k].buf) continue;
            auto* t = ops[k]->outputs[0];
            int W = t->dims[3];
            int base_row = out_start[k];
            size_t elem_sz = data_type_sizeof(t->type);
            bool blocked = (rings[k].saved_fmt == NATIVE_BLOCKED_FMT);
            int block = blocked ? NATIVE_BLOCK : 1;
            size_t row_bytes = (size_t)W * block * elem_sz;

            t->data = (char*)rings[k].buf - (size_t)base_row * row_bytes;
            t->dims[2] = rings[k].ring_H;
            t->ndata = (size_t)t->dims[0] * t->dims[1] * rings[k].ring_H * W;
            rings[k].base_row = base_row;
        }

        // Set ring_in/ring_out on operators for boundary/padding checks
        for (int k = 0; k < num_ops; ++k) {
            ops[k]->ring_in = {};
            ops[k]->ring_out = {};

            if (k > 0 && rings[k - 1].buf) {
                ops[k]->ring_in.ring_H = rings[k - 1].ring_H;
                ops[k]->ring_in.base_row = out_start[k - 1];
                ops[k]->ring_in.orig_H = rings[k - 1].saved_dims2;
            }
            if (k < num_ops - 1 && rings[k].buf) {
                ops[k]->ring_out.ring_H = rings[k].ring_H;
                ops[k]->ring_out.base_row = out_start[k];
                ops[k]->ring_out.orig_H = rings[k].saved_dims2;
            }
        }

        // Execute forward through the chain, skipping already-computed rows
        for (int k = 0; k < num_ops; ++k) {
            int needed_end = out_start[k] + out_rows[k];
            int exec_start = std::max(out_start[k], valid_end[k]);
            int exec_rows = needed_end - exec_start;
            if (exec_rows <= 0) {
                valid_end[k] = std::max(valid_end[k], needed_end);
                continue;
            }

            auto info = ops[k]->scroll_info();
            int in_start = exec_start * info.stride_h - info.halo_top;
            int in_end = (exec_start + exec_rows - 1) * info.stride_h + info.halo_bottom + 1;
            int iH = (k > 0 && rings[k - 1].buf)
                ? rings[k - 1].saved_dims2
                : ops[k]->inputs[0]->dims[2];
            in_start = std::max(0, in_start);
            in_end = std::min(in_end, iH);

            if (!ops[k]->exec_strip(exec_start, exec_rows, in_start, in_end - in_start)) {
                ok = false;
                break;
            }
            valid_end[k] = needed_end;
        }
        if (!ok) break;
    }

    // Restore tensor state (ring_arena scope frees all ring buffers on return)
    for (int k = 0; k < num_ops - 1; ++k) {
        if (!rings[k].buf) continue;
        auto* t = ops[k]->outputs[0];
        t->data = rings[k].saved_data;
        t->dims[2] = rings[k].saved_dims2;
        t->ndata = rings[k].saved_ndata;
    }

    // Clear ring state on operators
    for (int k = 0; k < num_ops; ++k) {
        ops[k]->ring_in = {};
        ops[k]->ring_out = {};
    }

    return ok;
}

// ---------------------------------------------------------------------------
// graph_optimizer_t methods
// ---------------------------------------------------------------------------

// Trial-run each scroll segment both ways (scrolled vs layer-by-layer).
// Segments that don't improve performance are removed.
// Called during AUTO mode in optimize().
void graph_optimizer_t::prune_segments(context_t* ctx)
{
    using clock = std::chrono::steady_clock;
    constexpr int TRIALS = 5;

    auto& nodes = ctx->graph->nodes;

    if (scroll_segments.empty()) return;

    ctx->run_for_warmup();

    std::vector<scroll_segment_t> kept;

    auto run_layer = [&](const scroll_segment_t& seg) {
        for (int i = seg.start; i < seg.end; ++i) {
            auto* n = nodes[i];
            if (n->skip || n->folded) continue;
            n->exec();
        }
    };

    for (auto& seg : scroll_segments) {
        // Warm up both paths
        run_layer(seg);
        { arena_scope_t scope(ctx->arena);
          exec_scroll_segment(ctx, seg.start, seg.end, seg.strip_height); }

        // Interleave measurements to neutralize thermal effects:
        // alternate layer/scroll so both see the same thermal state.
        uint64_t layer_times[TRIALS];
        uint64_t scroll_times[TRIALS];
        for (int t = 0; t < TRIALS; ++t) {
            {
                auto t0 = clock::now();
                run_layer(seg);
                auto t1 = clock::now();
                layer_times[t] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            }
            {
                arena_scope_t scope(ctx->arena);
                auto t0 = clock::now();
                exec_scroll_segment(ctx, seg.start, seg.end, seg.strip_height);
                auto t1 = clock::now();
                scroll_times[t] = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            }
        }

        // Use median for stable comparison
        std::sort(layer_times, layer_times + TRIALS);
        std::sort(scroll_times, scroll_times + TRIALS);
        uint64_t ns_layer  = layer_times[TRIALS / 2];
        uint64_t ns_scroll = scroll_times[TRIALS / 2];

        // Keep segment only if scrolling is at least 5% faster
        bool keep = (ns_scroll * 100 < ns_layer * 95);
        if (keep) {
            kept.push_back(seg);
        }
    }

    scroll_segments = std::move(kept);
}


} // namespace nnr
