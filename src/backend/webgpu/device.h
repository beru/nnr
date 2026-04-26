#pragma once

#include <webgpu/webgpu_cpp.h>

namespace nnr::webgpu {

// Lazily-initialized process-wide WebGPU device. First call boots up Dawn,
// picks a non-CPU adapter, and creates a device + queue. Subsequent calls
// return the same handles. Returns false if initialization fails.
struct device_t {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
};

device_t& get_device();
bool      device_ready();

// True iff the adapter picked in init_once() exposed TimestampQuery and the
// device was created with it. Callers should gate op-profiler begin/end
// emission on this, otherwise BeginComputePass with timestampWrites fails
// validation.
bool      has_timestamp_query();

// 1D-dispatch splitter. Given a total workgroup count `wg_count` (already
// divided by workgroup_size), returns (gx, gy) with gx, gy ≤ 65535 and
// gx*gy ≥ wg_count. Callers pass `gx * workgroup_size` as a stride uniform
// so the shader can reconstruct the flat id from (gid.x, gid.y):
//   let i = gid.y * stride + gid.x;
// WebGPU's per-dim workgroup cap is 65535; exceeding it silently drops the
// whole dispatch on Dawn. This helper keeps 1D ops working for
// 65535-workgroup-plus tensors (shufflenet channel shuffle, resnet with
// large batches, etc).
inline void dispatch_1d_grid(uint32_t wg_count, uint32_t& gx, uint32_t& gy) {
    constexpr uint32_t MAX_DIM = 65535u;
    if (wg_count <= MAX_DIM) { gx = wg_count; gy = 1; return; }
    gy = (wg_count + MAX_DIM - 1) / MAX_DIM;
    gx = (wg_count + gy - 1) / gy;
}

} // namespace nnr::webgpu
