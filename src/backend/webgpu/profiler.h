#pragma once

#include <vector>

namespace nnr::webgpu {

// Per-op GPU timestamp profiling. Zero-cost when disabled.
//
// Enabled by NNR_WEBGPU_OP_TIMINGS=1 at process start, AND by the adapter
// exposing the TimestampQuery feature (see device.cpp). If either gate
// fails, op_profiling_enabled() returns false and all hook functions are
// cheap no-ops.
//
// Design (the "boundary-pair" trick): rather than wrap every op's
// BeginComputePass call with timestampWrites (would touch 40+ files), we
// emit one *empty* compute pass just before and just after each op's
// exec(). Each empty pass takes a beginningOfPassWriteIndex, capturing a
// single GPU timestamp. The delta between surrounding empty passes gives
// the op's GPU execution time plus a small pipeline-barrier overhead
// shared across all ops (fixed bias, cancels for comparisons).
//
// Lifecycle per context_t::run():
//   1. op_profiler_begin_run(n_ops)            — resizes QuerySet if needed
//   2. For each live op i:
//        op_profiler_op_begin(i); n->exec(); op_profiler_op_end(i);
//   3. op_profiler_pre_flush()                 — adds ResolveQuerySet +
//      CopyBufferToBuffer to the shared encoder
//   4. flush_encoder()                         — finishes + submits
//   5. op_profiler_post_flush()                — MapAsync + wait + read;
//      populates last_op_times_us()
//
// Timestamps are reported in nanoseconds by WebGPU; we divide by 1000 to
// present μs in last_op_times_us().
bool        op_profiling_enabled();

void        op_profiler_begin_run(int n_ops);
void        op_profiler_op_begin(int op_idx);
void        op_profiler_op_end(int op_idx);
void        op_profiler_pre_flush();
void        op_profiler_post_flush();

const std::vector<double>& op_profiler_last_times_us();

} // namespace nnr::webgpu
