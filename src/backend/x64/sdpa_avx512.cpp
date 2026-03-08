#include "cpu_features.h"
#ifdef NNR_ARCH_X64
#include "sdpa_avx512.h"
#include "thread_pool.h"

namespace nnr {

void sdpa_multihead_avx512(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int seq_len, int head_dim)
{
    // Scratch per head: K^T [head_dim × seq_len] + score tile [BR × seq_len]
    // BR=64, seq_len=1500, head_dim=64 → (64×1500 + 64×1500)×4 = 768KB
    size_t scratch_per_head = ((size_t)head_dim * seq_len + 64 * seq_len) * sizeof(float);
    thread_pool_t::get().ensure_scratch(scratch_per_head);

    size_t head_stride = (size_t)seq_len * head_dim;

    // Process heads sequentially — each head's GEMM calls use the thread pool
    // internally. Nested for_static would deadlock.
    // Scratch is used only by the main thread (GEMM does its own threading).
    float* scratch = (float*)thread_pool_t::get().scratch(0);
    for (int h = 0; h < batch; h++) {
        sdpa_head_avx512(
            Q + h * head_stride,
            K + h * head_stride,
            V + h * head_stride,
            O + h * head_stride,
            scratch, seq_len, head_dim);
    }
}

} // namespace nnr
#endif // NNR_ARCH_X64
