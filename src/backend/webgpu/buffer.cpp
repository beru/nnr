#include "buffer.h"

#include "device.h"
#include "registry.h"  // backend_t

#include <atomic>
#include <cstring>
#include <mutex>

namespace nnr::webgpu {

namespace {

std::mutex                                               g_mu;
std::unordered_map<const tensor_t*, tensor_gpu_t>        g_map;

// Monotonic counter handed out every time a fresh GPU buffer is allocated
// inside `ensure_buffer`. Starts at 1 so 0 remains a reliable "never
// allocated" sentinel for stale BindGroup-cache snapshots.
std::atomic<uint32_t> g_next_generation{1};

// Shared per-run encoder. Lazily created by `shared_encoder()` and
// finished + submitted by `flush_encoder()`. This is process-global —
// nnr's runtime is single-threaded for inference (the worker pool is
// for intra-op parallelism), so no lock is needed.
wgpu::CommandEncoder g_encoder;
bool                 g_encoder_active = false;

size_t round_up_4(size_t n) { return (n + 3u) & ~size_t{3}; }

void submit_sync() {
    auto& d = get_device();
    // Poll until the queue is idle. WorkDone future gives us a precise
    // "everything submitted so far has completed" signal.
    wgpu::Future f = d.queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
    d.instance.WaitAny(f, UINT64_MAX);
}

} // namespace

tensor_gpu_t& ensure_buffer(const tensor_t* t, size_t bytes) {
    // BOOL tensors are stored as u8 on CPU (1 byte per element) but WGSL
    // can only address them as `array<u32>` — storage buffers have no
    // u8 access. We widen to u32 on upload, so the GPU buffer needs
    // `ndata * 4` bytes regardless of what the caller passed.
    if (t && t->type == NNR_DATA_TYPE_BOOL) {
        bytes = (size_t)t->ndata * 4;
    }
    // INT64 tensors are 8 bytes per element on CPU but WGSL has no i64.
    // Narrow to i32 on upload — safe for the common use cases (Gather /
    // ArgMax indices, model axes) where values fit comfortably in i32.
    // Callers that pass `t->ndata * 8` by habit get silently corrected.
    if (t && t->type == NNR_DATA_TYPE_INT64) {
        bytes = (size_t)t->ndata * 4;
    }

    std::lock_guard<std::mutex> lk(g_mu);
    auto& r = g_map[t];
    r.size = bytes;
    size_t needed = round_up_4(bytes);
    if (r.capacity >= needed && r.buf) return r;

    wgpu::BufferDescriptor d = {};
    d.size = needed;
    d.usage = wgpu::BufferUsage::Storage
            | wgpu::BufferUsage::CopySrc
            | wgpu::BufferUsage::CopyDst;
    r.buf = get_device().device.CreateBuffer(&d);
    r.capacity = needed;
    r.gpu_valid = false;
    // Bump generation so every op that cached a BindGroup referencing the
    // previous `buf` detects the mismatch on its next exec and rebuilds.
    r.generation = g_next_generation.fetch_add(1, std::memory_order_relaxed);
    return r;
}

uint32_t generation_of(const tensor_t* t) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_map.find(t);
    return it == g_map.end() ? 0u : it->second.generation;
}

wgpu::CommandEncoder& shared_encoder() {
    if (!g_encoder_active) {
        g_encoder = get_device().device.CreateCommandEncoder();
        g_encoder_active = true;
    }
    return g_encoder;
}

void flush_encoder() {
    if (!g_encoder_active) return;
    wgpu::CommandBuffer cb = g_encoder.Finish();
    get_device().queue.Submit(1, &cb);
    g_encoder = nullptr;
    g_encoder_active = false;
}

tensor_gpu_t* find(const tensor_t* t) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_map.find(t);
    return it == g_map.end() ? nullptr : &it->second;
}

void upload_if_needed(const tensor_t* t) {
    tensor_gpu_t* r;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_map.find(t);
        if (it == g_map.end()) return;
        r = &it->second;
        if (r->gpu_valid) return;
    }
    // BOOL tensors are u8 on CPU but u32 on the GPU (WGSL has no u8 storage).
    // Widen each byte to a u32 in a scratch buffer and upload that.
    if (t->type == NNR_DATA_TYPE_BOOL) {
        const uint8_t* src = (const uint8_t*)t->data;
        const size_t n = (size_t)t->ndata;
        std::vector<uint32_t> widened(n);
        for (size_t i = 0; i < n; ++i) widened[i] = src[i] ? 1u : 0u;
        get_device().queue.WriteBuffer(r->buf, 0, widened.data(), n * sizeof(uint32_t));
    } else if (t->type == NNR_DATA_TYPE_INT64) {
        // INT64 → i32 truncation. Values outside i32 range lose their high
        // bits silently — that's the documented trade-off; callers with
        // real int64 magnitudes must stay on CPU.
        const int64_t* src = (const int64_t*)t->data;
        const size_t n = (size_t)t->ndata;
        std::vector<int32_t> narrowed(n);
        for (size_t i = 0; i < n; ++i) narrowed[i] = (int32_t)src[i];
        get_device().queue.WriteBuffer(r->buf, 0, narrowed.data(), n * sizeof(int32_t));
    } else {
        // WriteBuffer requires a 4-byte-multiple size. Pad up using rounded capacity.
        size_t n = round_up_4(r->size);
        get_device().queue.WriteBuffer(r->buf, 0, t->data, n);
    }
    std::lock_guard<std::mutex> lk(g_mu);
    r->gpu_valid = true;
    r->cpu_valid = true; // WriteBuffer is a copy; both sides coherent.
}

void download_if_needed(tensor_t* t) {
    tensor_gpu_t* r;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_map.find(t);
        if (it == g_map.end()) return;
        r = &it->second;
        if (r->cpu_valid) return;
    }

    // Any pending kernel writes are still sitting in the shared encoder.
    // Flush before the readback or the staging copy will see stale data.
    flush_encoder();

    auto& d = get_device();
    size_t n = round_up_4(r->size);

    // Reuse the tensor's persistent staging buffer when it's big enough,
    // else grow it in place. This is the hot path for any CPU-consumer
    // boundary (e.g. mixed-backend ops, graph outputs) — allocating a
    // fresh staging buffer per call dominated the boundary cost before.
    if (r->staging_capacity < n) {
        wgpu::BufferDescriptor sd = {};
        sd.size  = n;
        sd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        r->staging          = d.device.CreateBuffer(&sd);
        r->staging_capacity = n;
    }
    wgpu::Buffer& staging = r->staging;

    wgpu::CommandEncoder enc = d.device.CreateCommandEncoder();
    enc.CopyBufferToBuffer(r->buf, 0, staging, 0, n);
    wgpu::CommandBuffer cb = enc.Finish();
    d.queue.Submit(1, &cb);

    wgpu::Future f = staging.MapAsync(
        wgpu::MapMode::Read, 0, n,
        wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus, wgpu::StringView) {});
    d.instance.WaitAny(f, UINT64_MAX);

    const void* p = staging.GetConstMappedRange(0, n);
    if (t->type == NNR_DATA_TYPE_BOOL) {
        // Narrow u32 → u8: any non-zero u32 becomes 1, zero stays 0.
        const uint32_t* src = (const uint32_t*)p;
        uint8_t* dst = (uint8_t*)t->data;
        const size_t ne = (size_t)t->ndata;
        for (size_t i = 0; i < ne; ++i) dst[i] = src[i] ? 1u : 0u;
    } else if (t->type == NNR_DATA_TYPE_INT64) {
        // Widen i32 → int64 with sign extension.
        const int32_t* src = (const int32_t*)p;
        int64_t* dst = (int64_t*)t->data;
        const size_t ne = (size_t)t->ndata;
        for (size_t i = 0; i < ne; ++i) dst[i] = (int64_t)src[i];
    } else {
        std::memcpy(t->data, p, r->size);
    }
    // Unmap so the staging buffer is ready for the next MapAsync call.
    staging.Unmap();

    std::lock_guard<std::mutex> lk(g_mu);
    r->cpu_valid = true;
}

void mark_gpu_written(const tensor_t* t) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_map.find(t);
    if (it == g_map.end()) return;
    it->second.gpu_valid = true;
    it->second.cpu_valid = false;
}

void mark_cpu_written(const tensor_t* t) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_map.find(t);
    if (it == g_map.end()) return;
    it->second.gpu_valid = false;
    it->second.cpu_valid = true;
}

void forget(const tensor_t* t) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_map.erase(t);
}

void alias(const tensor_t* dst, const tensor_t* src) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_map.find(src);
    if (it == g_map.end()) return;
    g_map[dst] = it->second;  // wgpu::Buffer holds a shared ref count
}

void sync_inputs_if_cpu_op(operator_t* op) {
    if (!op) return;
    if (op->resolved_backend == static_cast<uint8_t>(backend_t::WEBGPU)) return;
    for (auto* t : op->inputs) {
        if (t) download_if_needed(t);
    }
}

void sync_outputs_if_cpu_op(operator_t* op) {
    if (!op) return;
    if (op->resolved_backend == static_cast<uint8_t>(backend_t::WEBGPU)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto* t : op->outputs) {
        if (!t) continue;
        auto it = g_map.find(t);
        if (it == g_map.end()) continue;
        it->second.gpu_valid = false;
        it->second.cpu_valid = true;
    }
}

} // namespace nnr::webgpu
