#pragma once

#include "nnr.h"

#include <webgpu/webgpu_cpp.h>

namespace nnr::webgpu {

// Shared base for 2D NCHW pooling (MaxPool, AveragePool). One output element
// per thread; inner loops scan the kernel window, guarding each (ih, iw)
// against the input shape — pad regions contribute nothing and don't count
// toward the average when count_include_pad = 0.
//
// Subclasses supply three WGSL strings:
//   init_expr():     initial accumulator
//   combine_expr():  update using `acc`, `v`
//   finalize_expr(): final value using `acc`, `n_div` (either valid count
//                    or the full kernel size, depending on count_include_pad)
struct pool_elementwise_t : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;   // 16 u32 fields, stored in 128-byte buffer

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    uint32_t total_u = 0;
    // 16 matched fields + 1 trailing slot for grid_stride_x (2D dispatch split).
    uint32_t meta_vals[17] = {};
    // Dispatch grid cached at reshape() time.
    uint32_t dispatch_gx = 0;
    uint32_t dispatch_gy = 0;

    virtual const char* init_expr()     const = 0;
    virtual const char* combine_expr()  const = 0;
    virtual const char* finalize_expr() const = 0;

    bool init() override;
    bool reshape() override;
    bool exec() override;

    // Subclasses (e.g. GlobalAveragePool) that override reshape() to bake
    // their own meta_vals must call this at the end so the dispatch grid
    // is computed and meta_buf is written. Base reshape() calls this too.
    void finalize_meta_for_dispatch();
};

} // namespace nnr::webgpu
