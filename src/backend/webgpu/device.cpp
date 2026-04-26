#include "device.h"

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <webgpu/webgpu.h>

#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

namespace nnr::webgpu {

namespace {

std::once_flag                         g_once;
std::unique_ptr<dawn::native::Instance> g_native_instance;
device_t                               g_dev;
bool                                   g_ok = false;
bool                                   g_has_timestamp_query = false;

void init_once() {
    dawnProcSetProcs(&dawn::native::GetProcs());

    // Enable TimedWaitAny so WaitAny(future, UINT64_MAX) works — we use it
    // to block on GPU work from synchronous call sites (smoke tests, and
    // the end-of-run download step).
    static constexpr wgpu::InstanceFeatureName kTimedWaitAny =
        wgpu::InstanceFeatureName::TimedWaitAny;
    wgpu::InstanceDescriptor id = {};
    id.requiredFeatureCount = 1;
    id.requiredFeatures = &kTimedWaitAny;

    g_native_instance = std::make_unique<dawn::native::Instance>(&id);
    g_dev.instance = wgpu::Instance(g_native_instance->Get());

    std::vector<dawn::native::Adapter> adapters = g_native_instance->EnumerateAdapters();
    if (adapters.empty()) {
        fprintf(stderr, "[nnr::webgpu] no adapters found\n");
        return;
    }

    dawn::native::Adapter picked;
    for (auto& a : adapters) {
        WGPUAdapterInfo info = {};
        wgpuAdapterGetInfo(a.Get(), &info);
        bool is_cpu = info.adapterType == WGPUAdapterType_CPU;
        wgpuAdapterInfoFreeMembers(info);
        if (!is_cpu) { picked = a; break; }
    }
    if (!picked) picked = adapters.front();

    g_dev.adapter = wgpu::Adapter(picked.Get());

    // Request TimestampQuery if the adapter supports it — enables per-op
    // GPU profiling via op_profiler. Optional: device creation falls back
    // to a profiler-less device if the feature is missing on this adapter
    // (some CPU/SwiftShader adapters don't expose it).
    std::vector<wgpu::FeatureName> requested_features;
    if (g_dev.adapter.HasFeature(wgpu::FeatureName::TimestampQuery)) {
        requested_features.push_back(wgpu::FeatureName::TimestampQuery);
        g_has_timestamp_query = true;
    }

    // Request the adapter's full advertised limits. Defaults are conservative
    // (256 MB max buffer on most GPUs); without this, ops like vgg16-12-qdq's
    // FC layer hit Dawn's CreateBuffer error path on multi-hundred-MB
    // tensors. The adapter typically supports much more (~2 GB on dGPUs).
    wgpu::Limits adapterLimits = {};
    g_dev.adapter.GetLimits(&adapterLimits);

    wgpu::DeviceDescriptor desc = {};
    desc.requiredLimits = &adapterLimits;
    if (!requested_features.empty()) {
        desc.requiredFeatureCount = requested_features.size();
        desc.requiredFeatures     = requested_features.data();
    }
    desc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView msg) {
            fprintf(stderr, "[nnr::webgpu] device error (%d): %.*s\n",
                    (int)type, (int)msg.length, msg.data);
        });
    desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView msg) {
            fprintf(stderr, "[nnr::webgpu] device LOST (reason=%d): %.*s\n",
                    (int)reason, (int)msg.length, msg.data);
        });

    WGPUDevice raw = picked.CreateDevice(&desc);
    if (!raw) {
        fprintf(stderr, "[nnr::webgpu] CreateDevice failed\n");
        return;
    }
    g_dev.device = wgpu::Device::Acquire(raw);
    g_dev.queue = g_dev.device.GetQueue();
    g_ok = true;
}

} // namespace

bool has_timestamp_query() {
    std::call_once(g_once, init_once);
    return g_has_timestamp_query;
}

device_t& get_device() {
    std::call_once(g_once, init_once);
    return g_dev;
}

bool device_ready() {
    std::call_once(g_once, init_once);
    return g_ok;
}

} // namespace nnr::webgpu
