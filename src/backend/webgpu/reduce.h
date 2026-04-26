#pragma once

#include "nnr.h"

#include <webgpu/webgpu_cpp.h>

namespace nnr::webgpu {

// Shared base for float32 reduction ops along one or more axes.
//
// The kernel does two reductions per output: (1) a strided per-thread
// accumulation over the reduce domain, then (2) a workgroup tree
// reduction over the 64 per-thread partial accumulators. Subclasses
// parameterize via four WGSL expressions:
//   - init_expr():      initial accumulator value (e.g. "0.0")
//   - transform_expr(): applied to each element `v : f32` before
//                       accumulation (e.g. "v", "v * v", "abs(v)");
//                       defaults to `v` if not overridden.
//   - merge_expr():     associative combiner of two partial accumulators
//                       using free variables `a : f32, b : f32`
//                       (e.g. "a + b", "max(a, b)", "a * b").
//   - finalize_expr():  post-processing using `acc : f32, n : u32`
//                       (reduce count) — default `acc`.
//
// The stream step in the shader is: `acc = merge(acc, transform(v))`.
// The tree step is: `acc = merge(a, b)` over shared-memory slots.
//
// The base runtime-compiles a generic multi-axis reducer. It supports
// `axes` from either an int32/int64 input (opset 13+ for ReduceSum, 18+
// for the others) or an attribute; empty axes means "reduce all". The
// keepdims attribute controls whether reduced axes are 1-sized in the
// output shape or removed entirely.
struct reduce_elementwise_t : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;   // 112B storage

    // Cached BindGroup. Tensor-backed slots: [X, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    uint32_t total      = 0;     // output element count
    uint32_t ndim       = 0;     // input rank
    uint32_t red_count  = 0;     // product of reduce-axis dims (≥ 1)
    uint32_t in_dims_u[8]    = {};
    uint32_t in_strides_u[8] = {};
    uint32_t is_reduce_u[8]  = {};

    virtual const char* init_expr()      const = 0;
    virtual const char* transform_expr() const { return "v"; }
    virtual const char* merge_expr()     const = 0;
    virtual const char* finalize_expr()  const { return "acc"; }

    bool init() override;
    bool reshape() override;
    bool exec() override;
};

} // namespace nnr::webgpu
