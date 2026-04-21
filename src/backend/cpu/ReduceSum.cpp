#include "reduce_base.h"
#include "thread_pool.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/ops_x64.h"
#elifdef NNR_ARCH_ARM64
#include "backend/arm/conv_neon.h"
#endif

namespace nnr {

namespace {

struct ReduceSum_operator : reduce_base_t {
    bool init() override {
        axes_since_opset = 13;
        return reduce_base_t::init();
    }

    template <typename T>
    bool exec() {
#ifdef NNR_ARCH_X64
        // AVX-512 + threaded fast path for contiguous float ReduceSum
        if constexpr (std::is_same_v<T, float>) {
            const tensor_t* x = inputs[0];
            const float* px = (const float*)x->data;
            float* py = (float*)outputs[0]->data;
            if (x->ndata == 0) {
                for (size_t i = 0; i < outputs[0]->ndata; ++i) py[i] = 0.f;
                return true;
            }
            if (caxes.empty()) { memcpy(py, px, x->ndata * sizeof(float)); return true; }
            auto plan = plan_reduce(x->dims, x->ndim, caxes.data(), (int)caxes.size());
            if (plan.contiguous) {
                int batch = plan.batch_size, red = plan.reduce_size, tail = plan.tail_size;
                if (tail == 1) {
                    // Reduce contiguous elements: vectorized horizontal sum
                    nnr::for_static(0, batch, batch > 4, [&](int b) {
                        py[b] = reduce_sum_avx512(px + (size_t)b * red, red);
                    });
                } else if (tail >= 16) {
                    // Vectorize over tail dimension, reduce sequentially
                    nnr::for_static(0, batch, batch > 4, [&](int b) {
                        reduce_sum_tail_avx512(
                            py + (size_t)b * tail,
                            px + (size_t)b * red * tail,
                            red, tail);
                    });
                } else {
                    // Small tail: scalar with threading over batch*tail.
                    // for_static's iteration bound is int; if batch*tail overflows int
                    // the threaded path wraps and produces garbage — fall through to
                    // the generic reduce path below which walks the tensor with size_t.
                    int64_t total64 = (int64_t)batch * tail;
                    if (total64 > INT_MAX) goto scalar_fallback_x64;
                    int total = (int)total64;
                    nnr::for_static(0, total, total > 16, [&](int bt) {
                        int b = bt / tail, t = bt % tail;
                        float sum = 0;
                        for (int r = 0; r < red; r++)
                            sum += px[((size_t)b * red + r) * tail + t];
                        py[bt] = sum;
                    });
                }
                return true;
            }
        }
scalar_fallback_x64:;
#elifdef NNR_ARCH_ARM64
        // NEON + threaded fast path for contiguous float ReduceSum
        if constexpr (std::is_same_v<T, float>) {
            const tensor_t* x = inputs[0];
            const float* px = (const float*)x->data;
            float* py = (float*)outputs[0]->data;
            if (x->ndata == 0) {
                for (size_t i = 0; i < outputs[0]->ndata; ++i) py[i] = 0.f;
                return true;
            }
            if (caxes.empty()) { memcpy(py, px, x->ndata * sizeof(float)); return true; }
            auto plan = plan_reduce(x->dims, x->ndim, caxes.data(), (int)caxes.size());
            if (plan.contiguous) {
                int batch = plan.batch_size, red = plan.reduce_size, tail = plan.tail_size;
                if (tail == 1) {
                    nnr::for_static(0, batch, batch > 4, [&](int b) {
                        py[b] = reduce_sum_neon(px + (size_t)b * red, red);
                    });
                } else if (tail >= 4) {
                    nnr::for_static(0, batch, batch > 4, [&](int b) {
                        reduce_sum_tail_neon(
                            py + (size_t)b * tail,
                            px + (size_t)b * red * tail,
                            red, tail);
                    });
                } else {
                    int64_t total64 = (int64_t)batch * tail;
                    if (total64 > INT_MAX) goto scalar_fallback_arm64;
                    int total = (int)total64;
                    nnr::for_static(0, total, total > 16, [&](int bt) {
                        int b = bt / tail, t = bt % tail;
                        float sum = 0;
                        for (int r = 0; r < red; r++)
                            sum += px[((size_t)b * red + r) * tail + t];
                        py[bt] = sum;
                    });
                }
                return true;
            }
        }
scalar_fallback_arm64:;
#endif
        using AccT = typename acc_widen<T>::type;
        return reduce_exec_accum<T, acc_widen>(this, T(0), AccT(0),
            [](AccT acc, T v) { return acc + (AccT)v; },
            [](AccT acc) { return (T)acc; });
    }

    bool exec() override {
        if (inputs[0]->is_quantized())
            return exec_quantized_via_float(this, [this]() { return exec<float>(); });
        return typed_exec<ReduceSum_operator,
            opset_t<13, int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t, bfloat16_t, float16_t, float, double>,
            opset_t<1, int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=static
operator_t* resolver_default_op_ReduceSum(int opset, pool_t& pool) { return pool_new<ReduceSum_operator>(pool); }

} // namespace nnr
