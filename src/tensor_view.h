// -*- C++ -*-
// addr_view_t — unified scatter/gather addressing for NNR tensors.
//
// Captures (base ptr, per-dim physical byte strides, optional ring-mod) so
// producer/consumer kernels read addresses from the view instead of
// recomputing from `tensor_t::dims[]` + `tensor_t::format`. Subsumes alias
// (M1-M7), broadcast, slice, transpose, ring-buffer scroll as different
// `addr_view_t` configurations. Same machinery works for gather (input)
// and scatter (output).
//
// Logical-index convention: at(n, c, h, w) regardless of physical layout.
// The factory translates to the right physical byte stride per format.
//
// 4-D only for now. BLOCKED_8 / BLOCKED_16 keep their existing kernels.

#pragma once

#include "nnr.h"

namespace nnr {

struct addr_view_t {
    char*     base = nullptr;        // already shifted by alias / slice / scroll offsets
    ptrdiff_t s[MAX_NDIM] = {};      // physical byte strides per logical dim (NCHW order)
    int       ndim = 0;
    int       ring_mod = 0;          // 0 = no ring; else wrap dim `ring_axis` by this
    int       ring_axis = -1;        // logical dim the ring applies to (default H = 2)

    // Logical-dim address of (i0, i1, ..., i_ndim-1) → byte ptr.
    // Inline; the compiler hoists strides out of inner loops.
    template <class T>
    T* at(int i0, int i1 = 0, int i2 = 0, int i3 = 0) const {
        if (ring_mod && ring_axis == 2) i2 %= ring_mod;
        return (T*)(base + (ptrdiff_t)i0 * s[0] + (ptrdiff_t)i1 * s[1]
                          + (ptrdiff_t)i2 * s[2] + (ptrdiff_t)i3 * s[3]);
    }

    // Row pointer for 3-D views (dropping the last dim). Faster than at()
    // when the inner axis is a dense SIMD loop.
    template <class T>
    T* row(int i0, int i1, int i2) const {
        if (ring_mod && ring_axis == 2) i2 %= ring_mod;
        return (T*)(base + (ptrdiff_t)i0 * s[0] + (ptrdiff_t)i1 * s[1]
                          + (ptrdiff_t)i2 * s[2]);
    }

    // Element stride for logical dim `d` (in elements, not bytes). Useful as
    // an LDC parameter for SIMD GEMMs that take the C row stride directly.
    template <class T>
    int elem_stride(int d) const {
        return (int)(s[d] / (ptrdiff_t)sizeof(T));
    }
};

// Factory: derive view from a tensor_t. Logical index is (n, c, h, w);
// physical byte stride per dim is derived from t->format and dims[].
//
// Convention for `tensor_t::strides[]` when `strides_set` is true:
//   - NHWC alias (M1-M7 surface): strides[1] holds the parent's C count
//     (channel-axis row stride for the [spatial × C_total] matrix view).
//     Other entries in strides[] are not consulted.
//   - NCHW: strides[] is element-strides per logical dim (row-major).
//
// BLOCKED_8 / BLOCKED_16 fall through to default contiguous derivation
// (existing blocked kernels read those tensors directly).
inline addr_view_t make_addr(const tensor_t* t) {
    addr_view_t v;
    v.base = (char*)t->data;
    v.ndim = t->ndim;
    const ptrdiff_t es = (ptrdiff_t)data_type_sizeof(t);

    if (t->ndim == 4 && t->format == memory_layout_t::NHWC) {
        // Physical [N, H, W, C]. The "row stride" — per-spatial-position
        // C count — is parent_C when aliased, else dims[1].
        const ptrdiff_t C_row = t->strides_set ? (ptrdiff_t)t->strides[1] : (ptrdiff_t)t->dims[1];
        const ptrdiff_t W_str = C_row * es;                          // W step = C_row bytes
        const ptrdiff_t H_str = (ptrdiff_t)t->dims[3] * W_str;       // H step = W * C_row bytes
        const ptrdiff_t N_str = (ptrdiff_t)t->dims[2] * H_str;       // N step = H * W * C_row bytes
        v.s[0] = N_str;
        v.s[1] = es;       // C step = 1 element (innermost)
        v.s[2] = H_str;
        v.s[3] = W_str;
        return v;
    }

    // Default (NCHW or arbitrary rank): trust strides[] as element strides.
    // reinit() populates it row-major over the logical dim order.
    for (int d = 0; d < t->ndim; ++d) {
        v.s[d] = (ptrdiff_t)t->strides[d] * es;
    }
    return v;
}

} // namespace nnr
