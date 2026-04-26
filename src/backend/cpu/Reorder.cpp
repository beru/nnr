// Internal-only "Reorder" op: explicit layout-conversion node inserted by
// graph_optimizer/insert_reorders.cpp at producer→consumer edges where the
// declared layouts mismatch.
//
// Status (T3 M1, step 2b): exec(), exec_strip(), and scroll_info() implemented
// using strip-aware helpers from reorder_helpers.h. Reorder participates in
// scroll segments as a normal scrollable op with no halos and stride 1, so
// scroll_chains.cpp can drive it strip-by-strip without special casing.
//
// Attributes: from_layout, to_layout (memory_layout_t enum values stored as
// int32). Both are required.
//
// Aliasing (M1): full-alloc, no view-shaped reorders. M2 may teach the
// memory planner about eliminate-able views.

#include "nnr.h"
#include "attr_key.h"
#include "pool.h"
#include "kernel/layout.h"

#include <cstdint>

namespace nnr {

namespace reorder_detail {
template<int B> struct elem_for;
template<> struct elem_for<1> { using type = uint8_t;  };
template<> struct elem_for<2> { using type = uint16_t; };
template<> struct elem_for<4> { using type = uint32_t; };
template<> struct elem_for<8> { using type = uint64_t; };
template<int B> using elem_t = typename elem_for<B>::type;
} // namespace reorder_detail

namespace {

struct Reorder_operator : public operator_t {
    memory_layout_t from_layout = memory_layout_t::NCHW;
    memory_layout_t to_layout   = memory_layout_t::NCHW;

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        // Read attributes set by insert_reorders. Stored as int32.
        from_layout = static_cast<memory_layout_t>(
            attribute(attr_key_t::from_layout, (int32_t)memory_layout_t::NCHW));
        to_layout = static_cast<memory_layout_t>(
            attribute(attr_key_t::to_layout, (int32_t)memory_layout_t::NCHW));
        return true;
    }

    bool reshape() override {
        if (!inputs[0] || !outputs[0]) return false;
        auto* x = inputs[0];
        auto* y = outputs[0];
        small_vector<int> dims(x->ndim);
        for (int i = 0; i < x->ndim; ++i) dims[i] = x->dims[i];
        if (!y->reshape(dims, x->type)) return false;
        y->format = to_layout;
        y->declared_layout = to_layout;
        return true;
    }

    // View-eligibility: NCHW ↔ NHWC is bitwise-identical when the inner
    // permutation degenerates — i.e. C == 1 or H*W == 1. In those cases
    // the memory_planner can alias the output buffer to the input buffer
    // and the runtime exec path is a pointer-set, no data movement.
    int view_input_index() const override {
        if (inputs.empty() || outputs.empty() || !inputs[0]) return -1;
        auto* x = inputs[0];
        if (x->ndim != 4) return -1;
        using L = memory_layout_t;
        const bool nchw_nhwc =
            (from_layout == L::NCHW && to_layout == L::NHWC)
         || (from_layout == L::NHWC && to_layout == L::NCHW);
        if (!nchw_nhwc) return -1;
        const int C = x->dims[1];
        const int64_t HW = (int64_t)x->dims[2] * x->dims[3];
        if (C == 1 || HW == 1) return 0;
        return -1;
    }

    scroll_info_t scroll_info() const override {
        // Reorder is row-local: each output row depends only on the same
        // input row. No halos, no stride.
        //
        // BLOCKED layouts are excluded from scroll: the strip kernels
        // `nchw_to_nchwc` / `nchwc_to_nchw` use a per-channel-block stride
        // that doesn't compose with scroll_chains' ring-buffer virtual
        // pointer trick (which assumes a single contiguous H stride per
        // tensor). Driving them strip-by-strip in ring mode addresses
        // out-of-bounds. NCHW↔NHWC strips are safe — both are interleaved
        // along H * W with no cross-channel block striding.
        //
        // View-eligible cases also opt out — those are handled by the
        // pointer-aliasing exec() path, not strip-by-strip.
        using L = memory_layout_t;
        const bool blocked = (from_layout == L::BLOCKED_16 || from_layout == L::BLOCKED_8
                            || to_layout   == L::BLOCKED_16 || to_layout   == L::BLOCKED_8);
        if (blocked) return {};
        if (view_input_index() == 0) return {};
        return { .scrollable = true, .needs_pre_pass = false,
                 .halo_top = 0, .halo_bottom = 0, .stride_h = 1 };
    }

    template<int B>
    bool exec_strip_typed(int row_start, int row_end) {
        auto* x = inputs[0];
        auto* y = outputs[0];
        if (!x || !y || x->ndim != 4) return false;
        // NOTE: in ring-buffer mode (scroll_chains exec_scroll_segment),
        // x->dims[2] is replaced with ring_H (small, < orig H), and
        // x->data is virtual-pointer-offset by -base_row*row_bytes so that
        // addressing data + logical_row*row_bytes lands in the right ring
        // slot. The stride math in the strip helpers uses H_stride for
        // per-channel/per-batch offsets (which equal ring_H in ring mode —
        // correct, since each channel's ring slab is ring_H rows). Row
        // indices [row_start, row_end) come from the caller in *logical*
        // tensor space and must NOT be clamped against H_stride.
        int N = x->dims[0], C = x->dims[1], W = x->dims[3];
        int H_stride = x->dims[2];   // ring_H when ring-buffered, else orig H
        if (row_end <= row_start) return true;  // nothing to do

        using elem_t = reorder_detail::elem_t<B>;
        auto* dst = static_cast<elem_t*>(y->data);
        auto* src = static_cast<const elem_t*>(x->data);

        const auto F = from_layout;
        const auto T = to_layout;
        const auto NCHW = memory_layout_t::NCHW;
        const auto NHWC = memory_layout_t::NHWC;
        const auto BLK16 = memory_layout_t::BLOCKED_16;
        const auto BLK8  = memory_layout_t::BLOCKED_8;

        if (F == NCHW && T == NHWC) {
            nchw_to_nhwc(dst, src, N, C, H_stride, W, row_start, row_end);
        } else if (F == NHWC && T == NCHW) {
            nhwc_to_nchw(dst, src, N, C, H_stride, W, row_start, row_end);
        } else if (F == NCHW && (T == BLK16 || T == BLK8)) {
            int block = (T == BLK16) ? 16 : 8;
            nchw_to_nchwc(dst, src, N, C, H_stride, W, block, row_start, row_end);
        } else if ((F == BLK16 || F == BLK8) && T == NCHW) {
            int block = (F == BLK16) ? 16 : 8;
            nchwc_to_nchw(dst, src, N, C, H_stride, W, block, row_start, row_end);
        } else {
            // Unsupported pair (e.g. NHWC↔BLOCKED): caller must split into
            // a two-step chain via NCHW. insert_reorders enforces this by
            // never emitting such a Reorder directly.
            return false;
        }
        return true;
    }

    bool exec_strip(int out_row_start, int out_rows,
                    int /*in_row_start*/, int /*in_rows*/) override {
        // Reorder has stride_h=1 and no halos, so out and in rows coincide.
        // We use the out range as the row bound for both source and dest.
        if (!inputs[0]) return false;
        size_t bytes = data_type_sizeof(inputs[0]->type);
        const int row_end = out_row_start + out_rows;
        switch (bytes) {
        case 1: return exec_strip_typed<1>(out_row_start, row_end);
        case 2: return exec_strip_typed<2>(out_row_start, row_end);
        case 4: return exec_strip_typed<4>(out_row_start, row_end);
        case 8: return exec_strip_typed<8>(out_row_start, row_end);
        default: return false;
        }
    }

    bool exec() override {
        if (!inputs[0] || inputs[0]->ndim != 4) return false;
        // View case: memory_planner already excluded the output from the
        // pool. Set up the alias here so downstream consumers see the
        // input's bytes labeled as `to_layout`.
        if (view_input_index() == 0) {
            auto* x = inputs[0];
            auto* y = outputs[0];
            if (!x || !y) return false;
            y->data = x->data;
            y->owns_data = false;
            return true;
        }
        // Whole-tensor: full row range.
        const int H = inputs[0]->dims[2];
        return exec_strip(0, H, 0, H);
    }

    float layout_cost(memory_layout_t /*layout*/, bool /*input_nhwc*/) const override {
        // Reorder cost equals reorder_cost(input). T1's chain comparisons
        // call layout_cost on Conv/Pool/etc., not on Reorder, so this is
        // mostly informational until cancel_reorders queries it.
        if (!inputs[0]) return 0;
        auto* x = inputs[0];
        if (x->ndim != 4) return 0;
        float bytes = (float)x->dims[0] * x->dims[1] * x->dims[2] * x->dims[3]
                    * (float)data_type_sizeof(x->type);
        return 2.5f * bytes;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Reorder(int /*opset*/, pool_t& pool) {
    return pool_new<Reorder_operator>(pool);
}

} // namespace nnr
