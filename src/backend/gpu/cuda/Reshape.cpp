#if defined(NNR_USE_CUDA)

// Zero-copy reshape family: Reshape, Flatten, Squeeze, Unsqueeze.
// These ops only rename dimensions — no data movement.
//
// Strategy:
//   1) Run the CPU fallback, which performs the host-side pointer alias
//      (output->data = input->data, output->owns_data = false) and sets
//      up output metadata correctly.
//   2) If the input has a device buffer in the GPU cache, create an alias
//      entry for the output. A subsequent CUDA op reading `output` will
//      see the device buffer immediately (no re-upload).
//
// device_tag is set to CUDA in reshape() so sync_inputs_to_host doesn't
// download the input during this op (the fallback is pure pointer-fiddling).

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"

namespace nnr {

operator_t* resolver_default_op_Reshape  (int opset, pool_t& pool);
operator_t* resolver_default_op_Flatten  (int opset, pool_t& pool);
operator_t* resolver_default_op_Squeeze  (int opset, pool_t& pool);
operator_t* resolver_default_op_Unsqueeze(int opset, pool_t& pool);
operator_t* resolver_default_op_Identity (int opset, pool_t& pool);
operator_t* resolver_default_op_Dropout  (int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

struct Alias_cuda : public operator_t {
    operator_t* fallback = nullptr;

    virtual operator_t* make_fallback() = 0;

    bool init() override {
        fallback = make_fallback();
        fallback->ctx = ctx;
        fallback->opset = opset;
        fallback->op_type = op_type;
        fallback->inputs = inputs;
        fallback->outputs = outputs;
        fallback->attrs = attrs;
        fallback->init();
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        // CPU fallback does the host-side pointer alias (output = input).
        if (!fallback->exec()) return false;

        // Mirror on device: if input has a device buffer, alias it for output.
        // Not an error if the input isn't on device — output just doesn't have
        // a device buffer either, which is correct.
        if (inputs.empty() || outputs.empty() || !inputs[0] || !outputs[0])
            return true;

        auto& slot = ctx->backends[static_cast<uint8_t>(backend_t::CUDA)];
        if (!slot.data) return true;
        auto* be = static_cast<gpu::cuda_backend_t*>(slot.data);
        be->cache->alias(inputs[0], outputs[0]);
        return true;
    }
};

struct Reshape_cuda : Alias_cuda {
    operator_t* make_fallback() override { return resolver_default_op_Reshape(opset, ctx->attr_pool); }
};

struct Flatten_cuda : Alias_cuda {
    operator_t* make_fallback() override { return resolver_default_op_Flatten(opset, ctx->attr_pool); }
};

struct Squeeze_cuda : Alias_cuda {
    operator_t* make_fallback() override { return resolver_default_op_Squeeze(opset, ctx->attr_pool); }
};

struct Unsqueeze_cuda : Alias_cuda {
    operator_t* make_fallback() override { return resolver_default_op_Unsqueeze(opset, ctx->attr_pool); }
};

struct Identity_cuda : Alias_cuda {
    operator_t* make_fallback() override { return resolver_default_op_Identity(opset, ctx->attr_pool); }
};

// Dropout (inference): identity-alias when training_mode is 0 (the default).
// NNR runs inference only, so this always aliases.
struct Dropout_cuda : Alias_cuda {
    operator_t* make_fallback() override { return resolver_default_op_Dropout(opset, ctx->attr_pool); }
};

} // namespace

operator_t* resolver_cuda_op_Reshape  (int opset, pool_t& pool) { return pool_new<Reshape_cuda>  (pool); }
operator_t* resolver_cuda_op_Flatten  (int opset, pool_t& pool) { return pool_new<Flatten_cuda>  (pool); }
operator_t* resolver_cuda_op_Squeeze  (int opset, pool_t& pool) { return pool_new<Squeeze_cuda>  (pool); }
operator_t* resolver_cuda_op_Unsqueeze(int opset, pool_t& pool) { return pool_new<Unsqueeze_cuda>(pool); }
operator_t* resolver_cuda_op_Identity (int opset, pool_t& pool) { return pool_new<Identity_cuda> (pool); }
operator_t* resolver_cuda_op_Dropout  (int opset, pool_t& pool) { return pool_new<Dropout_cuda> (pool); }

} // namespace nnr

#endif // NNR_USE_CUDA
