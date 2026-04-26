#if defined(NNR_USE_CUDA)

#include "cuda_backend.h"
#include "gemm_wmma.h"
#include "nnr.h"
#include "registry.h"
#include "graph_optimizer.h"

#include <unordered_map>

namespace nnr::gpu {

cuda_backend_t* get_or_create_cuda_backend(context_t* ctx) {
    auto& slot = ctx->backends[static_cast<uint8_t>(backend_t::CUDA)];
    if (slot.data) return static_cast<cuda_backend_t*>(slot.data);

    // Refuse to create the backend during constant folding / graph prep:
    // fold_run() calls op->exec() directly and expects output->data (host) to
    // have the result. Running on GPU would leave data on device. Our CUDA ops
    // then fall back to CPU.
    if (!ctx->optimizer || !ctx->optimizer->plan_built) return nullptr;

    auto* dev = static_cast<cuda_device_t*>(create_device(gpu_backend_t::CUDA, 0));
    if (!dev) return nullptr;

    auto* b = new cuda_backend_t;
    b->device = dev;
    b->cache  = new gpu_cache_t(dev);

    slot.data = b;
    slot.free_fn = [](void* p) { delete static_cast<cuda_backend_t*>(p); };
    slot.writeback_fn = [](void* p, tensor_t* t) {
        static_cast<cuda_backend_t*>(p)->cache->writeback(t);
    };
    return b;
}

// --- CUDA Graph capture / replay lifecycle ---

void cuda_backend_t::invalidate_graph() {
    if (graph_exec && device) device->destroy_graph(graph_exec);
    graph_exec = nullptr;
    mode = exec_mode::normal;
}

cuda_backend_t::exec_mode cuda_backend_t::plan_run(bool all_ops_gpu, bool shapes_dirty) {
    if (shapes_dirty) {
        invalidate_graph();
        mode = exec_mode::normal;
        return mode;
    }
    if (!all_ops_gpu) {
        mode = exec_mode::normal;
        return mode;
    }
    if (graph_exec) {
        mode = exec_mode::replaying;
        return mode;
    }
    if (run_count < 1) {
        mode = exec_mode::normal;
        return mode;
    }
    if (device && device->begin_capture()) {
        mode = exec_mode::capturing;
    } else {
        mode = exec_mode::normal;
    }
    return mode;
}

void cuda_backend_t::finalize_capture() {
    if (mode != exec_mode::capturing || !device) return;
    graph_exec = device->end_capture();
    mode = exec_mode::normal;
}

bool cuda_backend_t::replay() {
    if (!graph_exec || !device) return false;
    return device->launch_graph(graph_exec);
}

// ---------------------------------------------------------------------------
// GEMM dispatch (WMMA TF32 fast path + scalar fallback). No cuBLAS.
// ---------------------------------------------------------------------------

bool gemm_device_f32(cuda_backend_t* be,
                     const float* A, const float* B, float* C,
                     int M, int N, int K,
                     int transA, int transB)
{
    if (!be || M <= 0 || N <= 0 || K <= 0) return false;

    // Two-tier dispatch:
    //   - gemm_tc_tf32_blocked: A*B (no transpose). 4 warps, 64×64 tile,
    //     shared-memory staging with zero-pad + bounds-checked store —
    //     handles any shape on the TensorCores.
    //   - gemm_scalar_f32: any transpose flags. Shared-memory tiled CUDA
    //     core kernel.
    if (transA == 0 && transB == 0) {
        // Big-matrix fast path: 128×128 WMMA (cp.async, double-buffered,
        // dynamic shared memory). Halves HBM bandwidth per FLOP vs the 64×64
        // kernel. The tile wastes warps when M or N are small, and low
        // occupancy hurts when the problem doesn't have many blocks. Only
        // dispatch for shapes where the 128×128 tile is fully populated and
        // block count is high enough for good SM utilization.
        if (M >= 256 && N >= 256 && (unsigned long long)M * N >= 128 * 1024) {
            CUfunction f = be->nvrtc.get("nnr_gemm_wmma", gemm_source(),
                                         "gemm_tc_tf32_128",
                                         nvrtc_arch_option(be->device));
            if (f) {
                static std::unordered_map<CUfunction, bool> s_opted_in;
                if (!s_opted_in[f]) {
                    cuFuncSetAttribute(f,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        98304);
                    s_opted_in[f] = true;
                }
                unsigned grid_x = (unsigned)((N + 127) / 128);
                unsigned grid_y = (unsigned)((M + 127) / 128);
                void* args[] = { (void*)&A, (void*)&B, (void*)&C, &M, &N, &K };
                return nvrtc_launch(be->device, f, grid_x, grid_y, 1,
                                    256, 1, 1, args, 98304);
            }
        }
        CUfunction f = be->nvrtc.get("nnr_gemm_wmma", gemm_source(),
                                     "gemm_tc_tf32_blocked_async",
                                     nvrtc_arch_option(be->device));
        if (!f) return false;
        unsigned grid_x = (unsigned)((N + 63) / 64);
        unsigned grid_y = (unsigned)((M + 63) / 64);
        void* args[] = { (void*)&A, (void*)&B, (void*)&C, &M, &N, &K };
        return nvrtc_launch(be->device, f, grid_x, grid_y, 1, 128, 1, 1, args);
    }
    CUfunction f = be->nvrtc.get("nnr_gemm_wmma", gemm_source(),
                                 "gemm_scalar_f32",
                                 nvrtc_arch_option(be->device));
    if (!f) return false;
    unsigned grid_x = (unsigned)((N + 15) / 16);
    unsigned grid_y = (unsigned)((M + 15) / 16);
    int ta = transA, tb = transB;
    void* args[] = { (void*)&A, (void*)&B, (void*)&C, &M, &N, &K, &ta, &tb };
    return nvrtc_launch(be->device, f, grid_x, grid_y, 1, 16, 16, 1, args);
}

bool gemm_device_s8s32(cuda_backend_t* be,
                       const void* A, const void* B, int* C,
                       int M, int N, int K, bool is_signed)
{
    if (!be || M <= 0 || N <= 0 || K <= 0) return false;
    const char* name = is_signed ? "gemm_tc_s8s32_blocked"
                                 : "gemm_tc_u8s32_blocked";
    CUfunction f = be->nvrtc.get("nnr_gemm_wmma", gemm_source(), name,
                                 nvrtc_arch_option(be->device));
    if (!f) return false;
    unsigned grid_x = (unsigned)((N + 63) / 64);
    unsigned grid_y = (unsigned)((M + 63) / 64);
    void* args[] = { (void*)&A, (void*)&B, (void*)&C, &M, &N, &K };
    return nvrtc_launch(be->device, f, grid_x, grid_y, 1, 128, 1, 1, args);
}

bool gemm_epilogue_f32(cuda_backend_t* be,
                       float* Y, const float* bias,
                       int M, int N, float alpha, float beta, int bias_kind)
{
    if (!be || M <= 0 || N <= 0) return false;
    CUfunction f = be->nvrtc.get("nnr_gemm_wmma", gemm_source(),
                                 "gemm_epilogue_f32", nvrtc_arch_option(be->device));
    if (!f) return false;
    int total = M * N;
    unsigned grid = (unsigned)((total + 255) / 256);
    float _a = alpha, _b = beta;
    int _M = M, _N = N, _bk = bias_kind;
    void* args[] = { (void*)&Y, (void*)&bias, &_M, &_N, &_a, &_b, &_bk };
    return nvrtc_launch(be->device, f, grid, 1, 1, 256, 1, 1, args);
}

} // namespace nnr::gpu

// --- Free-function hooks used by run_graph_impl (nnr.cpp). No CUDA headers
//     leak into nnr.cpp; mode is just an int.
namespace nnr {

// mode values: 0=normal, 1=replaying, 2=capturing
int cuda_begin_run(context_t* ctx, bool all_ops_gpu, bool shapes_dirty) {
    auto& slot = ctx->backends[static_cast<uint8_t>(backend_t::CUDA)];
    if (!slot.data) return 0;
    auto* be = static_cast<gpu::cuda_backend_t*>(slot.data);
    auto m = be->plan_run(all_ops_gpu, shapes_dirty);
    switch (m) {
        case gpu::cuda_backend_t::exec_mode::replaying: return 1;
        case gpu::cuda_backend_t::exec_mode::capturing: return 2;
        default:                                        return 0;
    }
}

void cuda_end_run(context_t* ctx, int mode) {
    auto& slot = ctx->backends[static_cast<uint8_t>(backend_t::CUDA)];
    if (!slot.data) return;
    auto* be = static_cast<gpu::cuda_backend_t*>(slot.data);
    if (mode == 2) be->finalize_capture();
    ++be->run_count;
}

bool cuda_replay_and_sync(context_t* ctx) {
    auto& slot = ctx->backends[static_cast<uint8_t>(backend_t::CUDA)];
    if (!slot.data) return false;
    auto* be = static_cast<gpu::cuda_backend_t*>(slot.data);
    if (!be->replay()) return false;
    be->device->sync();
    return true;
}

} // namespace nnr

#endif // NNR_USE_CUDA
