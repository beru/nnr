#pragma once

#include "nnr.h"

#include <string>
#include <string_view>
#include <vector>
#include <webgpu/webgpu_cpp.h>

namespace nnr::webgpu {

// Shared base for unary float32 elementwise ops (Relu, Sigmoid, Tanh, ...).
// Subclasses only have to return the per-element expression in WGSL where
// `v : <input_wgsl_ty>` is the input value. Output shape equals input shape.
//
// Dtype: defaults to f32 in/out. Override `input_wgsl_ty()` / `output_wgsl_ty()`
// / `input_dtype()` / `output_dtype()` to switch to u32 or i32 (used by Not
// for u32 mask inversion). The op_expr must evaluate to a value of the
// output type.
struct unary_elementwise_t : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;  // {n, 0, 0, 0}

    // Cached BindGroup. Tensor-backed slots: [X, Y]. Rebuilt when either
    // input or output reallocates its GPU buffer.
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[2] = {};

    // Dispatch grid + workgroup count cached at reshape() time so exec()
    // doesn't recompute or re-dispatch the uniform write.
    uint32_t        dispatch_gx = 0;
    uint32_t        dispatch_gy = 0;

    // WGSL expression producing the output element from `v`.
    // Example: "max(v, 0.0)" for Relu.
    virtual const char* op_expr() const = 0;
    virtual const char* input_wgsl_ty()  const { return "f32"; }
    virtual const char* output_wgsl_ty() const { return "f32"; }
    virtual data_type_t input_dtype()    const { return NNR_DATA_TYPE_FLOAT32; }
    virtual data_type_t output_dtype()   const { return NNR_DATA_TYPE_FLOAT32; }

    bool init() override;
    bool reshape() override;
    bool exec() override;
};

// Shared base for binary float32 elementwise ops (Add, Mul, ...) with full
// NumPy/ONNX broadcasting. Subclasses return the expression in terms of
// `a : f32, b : f32`. Supports up to rank 8.
//
// Broadcast rule: right-align input shapes; each pair of dims must be equal,
// or one must be 1. Missing leading axes are treated as 1. Output shape is
// the elementwise max of the aligned input shapes. An input's stride for a
// given output axis is 0 when that axis is size-1 in the input (or missing),
// otherwise it's the natural row-major stride of the input.
//
// Output dtype: defaults to f32. Subclasses that produce a different dtype
// (e.g., comparison ops returning u32 masks) override `output_wgsl_ty()` and
// `output_dtype()`. The op_expr then must evaluate to a value of that dtype
// — WGSL `bool` results need to be converted explicitly, e.g.
// `select(0u, 1u, a == b)`.
struct binary_elementwise_t : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;   // storage: {total, ndim, out_dims[8], a_strides[8], b_strides[8]}

    // Cached BindGroup. Tensor-backed slots: [A, B, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[3] = {};

    // Dispatch grid cached at reshape() time so exec() can dispatch
    // without recomputing it (and the matching grid_stride_x in the
    // meta buffer is written in reshape()).
    uint32_t        dispatch_gx = 0;
    uint32_t        dispatch_gy = 0;

    uint32_t total        = 0;
    uint32_t ndim         = 0;
    uint32_t out_dims_u[8]  = {};
    uint32_t a_strides_u[8] = {};
    uint32_t b_strides_u[8] = {};

    virtual const char* op_expr() const = 0;
    virtual const char* input_wgsl_ty()  const { return "f32"; }
    virtual const char* output_wgsl_ty() const { return "f32"; }
    virtual data_type_t input_dtype()    const { return NNR_DATA_TYPE_FLOAT32; }
    virtual data_type_t output_dtype()   const { return NNR_DATA_TYPE_FLOAT32; }

    bool init() override;
    bool reshape() override;
    bool exec() override;
};

// Fused chain of N ≥ 2 same-output-shape f32 elementwise ops. Composes the
// per-stage WGSL expressions into a single shader so all N stages run in
// one dispatch, eliminating the N-1 intermediate buffer round-trips.
//
// Chain links come in two flavors:
//   Unary:    v = <unary_expr(v)>                   // no side input
//   Binary*:  v = v <op> S<k>[idx]                  // one side input
// (*pipe_first: inputs[0] of the original op must be the previous stage's
//  output; inputs[1] becomes the side input. Reversed configurations are
//  rejected by the fusion pass and fall back to unfused execution.)
//
// Shader modes:
//   Path U (needs_meta=false): every side has the same shape as the pipe;
//     stage_wgsl references `S<k>[i]` directly, uniform holds just {n, 0, 0, 0}.
//   Path M (needs_meta=true): at least one side broadcasts INTO the pipe
//     (same output shape, but side has rank ≤ pipe and size-1 / missing axes).
//     Stage_wgsl references `S<k>[side_<k>_flat]`; a storage meta buffer
//     carries {total, ndim, out_dims, per-side stride tables}, and the shader
//     unflattens `i` per axis to compute each side_<k>_flat.
//
// Inputs layout: [pipe, side_0, side_1, ..., side_{n_sides-1}].
// Outputs: [y].
//
// Built by the graph optimizer (see graph_optimizer/fuse_webgpu_elementwise.cpp)
// — not registered with solve_operator, never created by the ONNX loader.
struct fused_elementwise_chain_t : public operator_t {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;  // Path U: {n, 0, 0, 0}
    wgpu::Buffer          meta_buf;  // Path M: {total, ndim, out_dims, side strides}

    // Per-stage WGSL RHS expressions, in execution order. The fusion pass
    // has already substituted `$s` per stage — with `S<k>[i]` (Path U) or
    // `S<k>[side_<k>_flat]` (Path M).
    std::vector<std::string> stage_wgsl;
    // Number of side input tensors (= count of binary stages).
    int n_sides = 0;
    // Set by the fusion pass when at least one side needs broadcast. Drives
    // shader generation (init) and bind group layout.
    bool needs_meta = false;

    // Cached BindGroup. Tensor-backed slots: [pipe, side_0..side_{n_sides-1}, Y]
    // = `1 + n_sides + 1` slots; upper-bounded by MAX_NDIM + 2 = 10, but
    // n_sides never exceeds the 8-way chain depth the fusion pass permits
    // in practice. A fixed 16 is plenty.
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[16] = {};

    bool init() override;
    bool reshape() override;
    bool exec() override;
};

// Returns the WGSL expression for an op_type that is a fusable same-shape
// f32 unary. Returns nullptr for any op that isn't in the fusable set
// (Clip / LeakyRelu / u32-typed ops / multi-input ops / etc.).
const char* fusable_unary_op_expr(std::string_view op_type);

// Returns the pipe-first WGSL pattern for a same-shape f32 binary op type,
// or nullptr if the op isn't in the fusable binary set. The pattern uses
// `v` for the pipe and `$s` as a placeholder for the side input access.
const char* fusable_binary_op_pattern(std::string_view op_type);

// MatMul (2D f32) with an elementwise epilogue fused into the output write.
// Produced by the graph optimizer when a MatMul is followed by a
// `fused_elementwise_chain_t` that consumes only the MatMul's output — the
// chain's `stage_wgsl` list is absorbed into the MatMul kernel's final
// epilogue, so the N-stage chain runs inside the same dispatch that computes
// the accumulator, eliminating one full-tensor read+write compared to the
// two-dispatch (MatMul → chain) sequence.
//
// Inputs layout: [A, B, side_0, ..., side_{n_sides-1}]
// Output:        [Y]  (shape = [M, N])
//
// Pipe of the absorbed chain is Y. Sides follow the same Path U / Path M
// convention as `fused_elementwise_chain_t`:
//   Path U (needs_meta=false): all sides have shape [M, N]; stage expressions
//     reference `S<k>[i]` where i = row*N + col.
//   Path M (needs_meta=true): at least one side broadcasts INTO [M, N]
//     (a 1D bias [N], a column [M,1], a scalar, ...). Stage expressions
//     reference `S<k>[side_<k>_flat]`; the shader computes each
//     side_<k>_flat from `row`/`col` using a per-side (row_stride, col_stride)
//     pair packed in a small storage meta buffer. Size-1 / missing axes map
//     to stride 0.
//
// Never registered with solve_operator — only the fusion pass constructs it.
struct matmul_fused_chain_t : public operator_t {
    int  m = 0, n = 0, k = 0;
    int  n_sides    = 0;
    bool needs_meta = false;
    std::vector<std::string> stage_wgsl;

    wgpu::ComputePipeline pipeline;        // 16x16 tiled matmul (general M)
    wgpu::ComputePipeline gemv_pipeline;   // split-K GEMV variant, M=1 only
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;  // {M, N, K, 0}
    wgpu::Buffer          meta_buf;  // Path M only: per-side (stride_row, stride_col) as vec4<u32>

    bool is_gemv = false;                  // set in reshape() once m is known

    // Cached BindGroup. Tensor slots: [A, B, side_0..side_{n_sides-1}, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[16] = {};

    bool init() override;
    bool reshape() override;
    bool exec() override;

    int64_t num_ops() const override { return (int64_t)2 * m * n * k; }
};

// Gemm (2D f32) with an elementwise epilogue fused into the output write.
// Analogue of `matmul_fused_chain_t` for ONNX Gemm — adds α, β, transA,
// transB, and an optional bias `C` (shapes {}, [N], [M,N] supported; other
// broadcasts cause the pre-init sanity check to refuse fusion so the
// Gemm+chain pair stays split).
//
// Inputs layout: [A, B] ++ (has_bias ? [C] : []) ++ [side_0, ..., side_{n_sides-1}]
// Output:        [Y]  (shape = [M, N])
//
// The chain's stages (already `$s`-substituted to `S<k>[i]` Path U or
// `S<k>[side_<k>_flat]` Path M by the elementwise fusion pass) run on `v`
// AFTER the Gemm produces `v = α·A·B + β·C` — matching the semantics of
// Gemm followed by the original chain.
struct gemm_fused_chain_t : public operator_t {
    int  m = 0, n = 0, k = 0;
    int  n_sides    = 0;
    bool needs_meta = false;

    // Gemm-specific config — set by init() from attrs / input shape.
    float alpha     = 1.0f;
    float beta      = 1.0f;
    int   transA    = 0;
    int   transB    = 0;
    bool  has_bias  = false;
    int   bias_kind = 0;  // 0=none, 1=per-col [N], 2=full [M,N]; set in reshape()

    std::vector<std::string> stage_wgsl;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          cfg_buf;    // 48-byte Cfg (matches base Gemm layout + reuses runtime bias branch)
    wgpu::Buffer          zero_bias;  // bound on the C slot when !has_bias
    wgpu::Buffer          meta_buf;   // Path M only: per-side (stride_row, stride_col)

    // Cached BindGroup. Tensor slots: [A, B, C|0, side_0..side_{n_sides-1}, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[16] = {};

    bool init() override;
    bool reshape() override;
    bool exec() override;

    int64_t num_ops() const override { return (int64_t)2 * m * n * k; }
};

// 2D NCHW Conv with an elementwise epilogue fused into the output write.
// Analogue of `matmul_fused_chain_t` / `gemm_fused_chain_t` for Conv —
// absorbs a FusedElementwiseChain that directly consumes the Conv's output
// into the Conv kernel itself. The pipe shape is 4D
// `[N, C_out, H_out, W_out]`; sides may be same-shape (Path U) or
// broadcast-into-pipe with per-axis size-1 / missing-leading axes
// (Path M). Typical payload: Conv → bias-add → ReLU collapses to one
// dispatch.
//
// Inputs layout: [X, W] ++ (has_bias ? [B] : []) ++ [side_0, ..., side_{n_sides-1}]
// Output:        [Y]  (shape = [N, C_out, H_out, W_out])
//
// Pipeline cache key: (mode, n_sides, stage list). Strides/pads/dilations/
// groups/auto_pad all flow through the runtime meta buffer (same layout as
// the base Conv op plus a per-side stride table for Path M), so they don't
// split the cache.
struct conv_fused_chain_t : public operator_t {
    // Output shape + dispatch total.
    int n_out = 0, c_out = 0, h_out = 0, w_out = 0;
    uint32_t total_u = 0;

    // Config baked at reshape() time.
    uint32_t meta_vals[18] = {};   // matches base Conv Meta layout
    bool  has_bias   = false;
    int   n_sides    = 0;
    bool  needs_meta = false;      // Path U if false

    std::vector<std::string> stage_wgsl;

    wgpu::ComputePipeline pipeline;               // plain per-output-thread kernel
    wgpu::ComputePipeline tiled_pipeline;         // 8x8 shared-weight cache variant
    wgpu::ComputePipeline regtile_pipeline;       // 8x8 WG, 4 output cols per thread
    wgpu::ComputePipeline regtile_wide_pipeline;  // 8x8 WG, 8 output cols per thread
    wgpu::ComputePipeline regtile_2d_pipeline;    // 8x8 WG, 2 rows × 4 cols per thread
    wgpu::ComputePipeline regtile_3x3_pipeline;   // specialized kH=kW=3, stride=dilation=1, pad=1
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          meta_buf;    // 128 B base Conv meta
    wgpu::Buffer          side_md;     // Path M only: per-side vec4<u32> strides
    wgpu::Buffer          dummy_bias;  // 16 B zero buffer when !has_bias

    bool use_tiled        = false;     // set in reshape() when the weight tile fits shared mem
    bool use_regtile      = false;     // set in reshape() when register tiling is a net win
    bool use_regtile_wide = false;     // set in reshape() when the wider 8-col tile fits cleanly
    bool use_regtile_2d   = false;     // set in reshape() for tall-or-wide outputs with iC small
    bool has_3x3_s1       = false;     // set in init() if attrs match kH=kW=3, stride=dilation=1, pad=1
    bool use_regtile_3x3  = false;     // set in reshape() when has_3x3_s1 is true and regtile is on

    // Cached BindGroup. Tensor slots: [X, W, bias|0, side_0..side_{n_sides-1}, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[16] = {};

    bool init() override;
    bool reshape() override;
    bool exec() override;

    int64_t num_ops() const override { return 2 * (int64_t)total_u; }  // rough
};

// LayerNormalization (last-axis) with an elementwise epilogue fused into
// the final normalized-output write. Same shape / stage-substitution
// conventions as the matmul / gemm / conv variants, but the producer is a
// reducing op (2-pass mean/variance + elementwise apply). The chain stages
// run on each output element after the scale/bias apply, inside the same
// workgroup-level pass that writes Y.
//
// Inputs layout: [X, scale] ++ (has_bias ? [bias] : []) ++ [side_0, ..., side_{n_sides-1}]
// Output:        [Y]  (same shape as X)
//
// Sides follow the same Path U / Path M convention as
// `fused_elementwise_chain_t`:
//   Path U (needs_meta=false): every side has the same shape as X; stage
//     expressions reference `S<k>[i]`.
//   Path M (needs_meta=true): at least one side broadcasts INTO X (same
//     output shape, but side has rank ≤ X and size-1 / missing axes). A
//     separate storage meta buffer carries the output shape + per-side
//     stride tables; the shader unflattens the linear output index in the
//     final pass to compute each `side_<k>_flat`. Supports rank ≤ 8.
struct layer_norm_fused_chain_t : public operator_t {
    int64_t axis_attr = -1;
    float   epsilon   = 1e-5f;
    int     outer     = 0;
    int     N         = 0;

    bool  has_bias   = false;
    int   n_sides    = 0;
    bool  needs_meta = false;

    std::vector<std::string> stage_wgsl;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms_dims;   // {outer, N, 0, 0}
    wgpu::Buffer          uniforms_cfg;    // {eps, 0, 0, 0}
    wgpu::Buffer          zero_bias;       // bound when !has_bias
    wgpu::Buffer          side_md;         // Path M only: ndim + out_dims + per-side strides

    // Cached BindGroup. Tensor slots: [X, scale, bias|0, side_0..side_{n_sides-1}, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[16] = {};

    bool init() override;
    bool reshape() override;
    bool exec() override;
};

// Softmax (any axis) with an elementwise epilogue fused into the final
// normalized-output write. Three-pass row reduction (max, exp-sum,
// divide-and-write) dispatched one workgroup per (outer, inner) pair;
// the chain stages run on each output element after the divide, inside
// the same workgroup-level pass. Last-axis case (inner==1) folds the
// stride multiply — same dispatch as the non-fused Softmax.
//
// Inputs layout: [X] ++ [side_0, ..., side_{n_sides-1}]
// Output:        [Y]  (same shape as X)
//
// Sides follow the same Path U / Path M convention as
// `layer_norm_fused_chain_t`.
struct softmax_fused_chain_t : public operator_t {
    int axis_attr  = -1;
    int outer      = 0;
    int N          = 0;
    int inner      = 0;
    int n_sides    = 0;
    bool needs_meta = false;

    std::vector<std::string> stage_wgsl;

    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;       // {outer, N, inner, 0}
    wgpu::Buffer          side_md;        // Path M only: ndim + out_dims + per-side strides

    // Cached BindGroup. Tensor slots: [X, side_0..side_{n_sides-1}, Y].
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[16] = {};

    bool init() override;
    bool reshape() override;
    bool exec() override;
};

} // namespace nnr::webgpu
