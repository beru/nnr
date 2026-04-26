#pragma once
// CUPTI Activity API profiler for per-kernel GPU timing.
//
// During CUDA Graph replay, host-side timers around operator_t::exec() report
// ~0 ms because the captured graph runs as one cudaGraphLaunch — individual
// exec() calls are no-ops. CUPTI records each kernel's actual GPU start/end
// timestamp on the device, regardless of whether it ran via direct launch or
// graph replay.
//
// Usage:
//   cupti_profiler_t prof;
//   prof.init();
//   prof.push_op(op_idx);
//     <run op or replay graph segment>
//   prof.pop_op();
//   ...
//   prof.flush();                  // sync + drain CUPTI buffers
//   for (auto& r : prof.records()) // r.op_idx, r.duration_ns
//
// All correlation IDs map back to the `op_idx` pushed via push_op().
// Activity API needs no special permissions for KERNEL records.
//
// No-op when NNR_USE_CUPTI is undefined.

#if defined(NNR_USE_CUDA)

#include <cstdint>
#include <vector>

namespace nnr::gpu {

struct cupti_record_t {
    uint32_t op_idx;        // matches the op_idx push_op'd around the launch
    uint64_t start_ns;      // device timestamp (CUPTI clock)
    uint64_t end_ns;
    const char* kernel;     // demangled kernel name (CUPTI-owned, lifetime = profiler)
};

struct cupti_profiler_t {
    cupti_profiler_t();
    ~cupti_profiler_t();

    // Returns true if CUPTI was successfully initialized (NNR_USE_CUPTI=ON
    // at build time AND runtime subscribe succeeded). When false, all other
    // methods are cheap no-ops.
    bool init();

    // Tag every kernel launched between push_op and pop_op with op_idx.
    // Safe to call when init() failed — becomes a no-op.
    void push_op(uint32_t op_idx);
    void pop_op();

    // Force CUPTI to drain any pending activity buffers. Call after
    // cudaStreamSynchronize / cudaGraphLaunch + sync.
    void flush();

    // Drain accumulated records (clears internal storage).
    std::vector<cupti_record_t> drain();

    // Disable activity recording (init re-enables).
    void shutdown();

private:
    bool initialized_ = false;
};

// Process-wide singleton. The CUPTI subscriber is a global resource — only one
// can be active at a time per process, so this is a singleton by necessity.
cupti_profiler_t& cupti_profiler();

} // namespace nnr::gpu

#endif // NNR_USE_CUDA
