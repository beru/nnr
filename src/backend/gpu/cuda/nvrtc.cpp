#if defined(NNR_USE_CUDA)

#include "nvrtc.h"
#include "cuda_device.h"

#include <nvrtc.h>
#include <cstdio>
#include <cstring>
#include <vector>

namespace nnr::gpu {

#define NVRTC_CHECK(call) do {                                                 \
    nvrtcResult _r = (call);                                                   \
    if (_r != NVRTC_SUCCESS) {                                                 \
        fprintf(stderr, "NVRTC error %s:%d: %s\n",                             \
                __FILE__, __LINE__, nvrtcGetErrorString(_r));                  \
        return nullptr;                                                        \
    }                                                                          \
} while (0)

#define CU_CHECK(call) do {                                                    \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
        const char* _s = nullptr;                                              \
        cuGetErrorString(_r, &_s);                                             \
        fprintf(stderr, "CUDA driver error %s:%d: %s\n",                       \
                __FILE__, __LINE__, _s ? _s : "?");                            \
        return nullptr;                                                        \
    }                                                                          \
} while (0)

CUfunction nvrtc_cache_t::get(const std::string& key,
                              const char* source,
                              const char* name,
                              const char* options)
{
    // Cache is keyed per (source-key, kernel-name): many kernels share one
    // compiled source (e.g. the elementwise module has 30+ functions), and
    // each needs its own CUfunction handle.
    std::string lookup_key = key;
    lookup_key += '/';
    lookup_key += name;

    auto it = cache_.find(lookup_key);
    if (it != cache_.end()) return it->second.func;

    // If we've already compiled this source (under its own key with any kernel
    // name appended), look up the other functions in the same module without
    // recompiling.
    for (auto& [k, v] : cache_) {
        if (k.size() > key.size() && k.compare(0, key.size(), key) == 0 && k[key.size()] == '/') {
            // Same source, different kernel. Reuse the module.
            CUfunction func = nullptr;
            if (cuModuleGetFunction(&func, v.module, name) == CUDA_SUCCESS) {
                cache_[lookup_key] = { v.module, func };  // NOTE: module shared
                return func;
            }
            break;
        }
    }

    nvrtcProgram prog = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, source, "nnr_nvrtc.cu", 0, nullptr, nullptr));

    std::vector<const char*> opts;
    opts.push_back("-default-device");
    opts.push_back("--std=c++17");
    if (options && *options) opts.push_back(options);
    // Include CUDA's bundled headers (mma.h, cuda_fp16.h, etc.). Path embedded
    // at configure time (src/CMakeLists.txt → NNR_CUDA_INCLUDE_DIR).
#ifdef NNR_CUDA_INCLUDE_DIR
    static const std::string include_opt = std::string("-I") + NNR_CUDA_INCLUDE_DIR;
    opts.push_back(include_opt.c_str());
#endif

    nvrtcResult compile_r = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (compile_r != NVRTC_SUCCESS) {
        size_t log_sz = 0;
        nvrtcGetProgramLogSize(prog, &log_sz);
        std::string log(log_sz, '\0');
        nvrtcGetProgramLog(prog, log.data());
        fprintf(stderr, "NVRTC compile failed for '%s':\n%s\n", name, log.c_str());
        nvrtcDestroyProgram(&prog);
        return nullptr;
    }

    size_t ptx_sz = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_sz));
    std::string ptx(ptx_sz, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    nvrtcDestroyProgram(&prog);

    CUmodule mod = nullptr;
    CU_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));

    CUfunction func = nullptr;
    CUresult fr = cuModuleGetFunction(&func, mod, name);
    if (fr != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(fr, &s);
        fprintf(stderr, "cuModuleGetFunction('%s') failed: %s\n", name, s ? s : "?");
        cuModuleUnload(mod);
        return nullptr;
    }

    cache_[lookup_key] = { mod, func };
    return func;
}

nvrtc_cache_t::~nvrtc_cache_t() {
    // Multiple entries may share the same module (one per kernel name); unload
    // each module exactly once.
    std::unordered_map<CUmodule, bool> seen;
    for (auto& [k, v] : cache_) {
        if (v.module && !seen[v.module]) {
            cuModuleUnload(v.module);
            seen[v.module] = true;
        }
    }
    cache_.clear();
}

bool nvrtc_launch(cuda_device_t* dev,
                  CUfunction func,
                  unsigned grid_x, unsigned grid_y, unsigned grid_z,
                  unsigned block_x, unsigned block_y, unsigned block_z,
                  void** args,
                  unsigned shared_bytes)
{
    if (!dev || !func) return false;
    CUresult r = cuLaunchKernel(func,
                                grid_x, grid_y, grid_z,
                                block_x, block_y, block_z,
                                shared_bytes,
                                (CUstream)dev->compute_stream(),
                                args, nullptr);
    if (r != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(r, &s);
        fprintf(stderr, "cuLaunchKernel failed: %s\n", s ? s : "?");
        return false;
    }
    return true;
}

const char* nvrtc_arch_option(cuda_device_t* dev) {
    // Cache per-device by id. Returns e.g. "-arch=compute_86".
    static std::unordered_map<int, std::string> cache;
    int id = dev ? dev->device_id() : 0;
    auto it = cache.find(id);
    if (it != cache.end()) return it->second.c_str();

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, id) != cudaSuccess)
        return "-arch=compute_75";  // safe default

    char buf[64];
    std::snprintf(buf, sizeof(buf), "-arch=compute_%d%d", prop.major, prop.minor);
    cache[id] = buf;
    return cache[id].c_str();
}

} // namespace nnr::gpu

#endif // NNR_USE_CUDA
