#pragma once

#include "nnr.h"

#include <webgpu/webgpu_cpp.h>

#include <unordered_map>

namespace nnr::webgpu {

// Per-tensor GPU residency record. Tracks where the current valid copy of
// the tensor's data lives. Backend ops never force a download between GPU
// kernels — downloads happen only at backend boundaries (CPU op consuming a
// GPU-resident tensor, or user code reading a final output).
//
// Invariant: at least one of gpu_valid / cpu_valid is true whenever the
// tensor has meaningful data. Both true means coherent.
//
// `generation` is a monotonic ID bumped every time `ensure_buffer` allocates
// a fresh underlying `buf`. Ops cache their BindGroup and compare the
// generation of each bound tensor across execs; if all match, the cached
// BindGroup is reused and no per-exec `CreateBindGroup` happens.
//
// `staging` is a persistent MapRead|CopyDst buffer used by
// `download_if_needed` to read back GPU data into CPU memory. Grow-only:
// resized only when a larger readback is needed. Avoids creating a fresh
// staging buffer on every cross-backend boundary.
struct tensor_gpu_t {
    wgpu::Buffer buf;
    size_t       capacity = 0;  // bytes allocated on GPU
    size_t       size     = 0;  // bytes logically in use
    uint32_t     generation = 0;  // bumped on each (re)allocation of `buf`
    wgpu::Buffer staging;
    size_t       staging_capacity = 0;
    bool         gpu_valid = false;
    bool         cpu_valid = true;
    bool         is_weight = false;  // initializer: upload once, never again
};

// Returns the current generation of `t`'s GPU buffer, or 0 if no record
// exists. Callers compare this against a cached snapshot to detect when
// their BindGroup must be rebuilt (generation mismatch ⇒ underlying
// `buf` changed ⇒ BindGroup is stale).
uint32_t generation_of(const tensor_t* t);

// Returns the active per-run command encoder, allocating a fresh one on
// first use after each `flush_encoder()`. Ops encode their compute pass
// (and any CopyBufferToBuffer they need) into this single shared encoder
// instead of allocating one per `exec()`. The runtime calls
// `flush_encoder()` once at the end of `context_t::run()` and at any
// cross-backend boundary (via `download_if_needed`) so buffered work
// actually executes when needed.
//
// Lifetime caveat: the encoder owns refs to whatever BindGroups the ops
// recorded. Each op's cached BindGroup also retains its referenced
// buffers, so buffer reallocation between dispatches is safe — the old
// buffer outlives the encoder via the BindGroup's refcount.
wgpu::CommandEncoder& shared_encoder();

// Finishes + submits the active shared encoder if one exists, then
// clears the active flag so the next `shared_encoder()` allocates fresh.
// No-op when no encoder is active. Called by:
//   1. `download_if_needed` (must flush so the readback sees pending
//      kernel writes)
//   2. `context_t::run()` epilogue (so user code that reads outputs
//      after run() returns sees the work, even without an explicit
//      readback)
void flush_encoder();

// Allocates (or re-uses if large enough) a storage buffer for `t` sized
// `bytes` with STORAGE | COPY_SRC | COPY_DST usage. Does not touch contents.
tensor_gpu_t& ensure_buffer(const tensor_t* t, size_t bytes);

// Returns the record for `t` if one exists, else nullptr.
tensor_gpu_t* find(const tensor_t* t);

// Upload tensor->data to the GPU buffer if gpu is stale. No-op if gpu_valid.
void upload_if_needed(const tensor_t* t);

// Download GPU buffer into tensor->data if cpu is stale. Blocks on the
// current queue until the readback completes. No-op if cpu_valid.
void download_if_needed(tensor_t* t);

// After a compute pass writes to `t`'s buffer.
void mark_gpu_written(const tensor_t* t);

// After CPU code writes directly into tensor->data (e.g., non-WEBGPU op).
void mark_cpu_written(const tensor_t* t);

// Evict any GPU resources associated with `t`. Call when the tensor is
// about to be destroyed or resized.
void forget(const tensor_t* t);

// Make `dst` share `src`'s GPU buffer. Used by view ops (Reshape, Squeeze,
// Unsqueeze, Flatten, Identity) so the output sees the same VRAM as the
// input with zero copy. Residency flags are copied too — if src is gpu_valid
// then dst becomes gpu_valid.
void alias(const tensor_t* dst, const tensor_t* src);

// If `op` is resolved on CPU, download any GPU-resident inputs it will read.
// No-op when `op->resolved_backend == WEBGPU` (WebGPU ops read directly from
// the resident GPU buffer) or when no input has a GPU record. Called from
// context_t::run() at each exec site to preserve cross-backend correctness
// without per-op transfer thrashing.
void sync_inputs_if_cpu_op(operator_t* op);

// After a CPU op executes, mark any of its outputs that already have a GPU
// residency record as cpu-written — the CPU exec just overwrote tensor->data
// so the GPU buffer is now stale. Without this, a subsequent WebGPU op that
// consumes the same tensor would see gpu_valid=true (carried over from an
// earlier run where a WebGPU op wrote the same tensor) and skip upload_if_
// needed, reading stale GPU data. No-op for WEBGPU ops (mark_gpu_written
// handles their case) and for CPU-only tensors with no GPU record yet.
void sync_outputs_if_cpu_op(operator_t* op);

} // namespace nnr::webgpu
