#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// ---------------------------------------------------------------------------
// Conv + BatchNormalization fusion
// ---------------------------------------------------------------------------
// Folds BN scale/bias/mean/var into Conv weights and bias so that the BN
// node can be skipped entirely.  Handles Conv nodes with or without an
// existing bias input.
//
// Pattern:  Conv -> BatchNormalization   (adjacent, single consumer)
// Requires: BN parameters are all graph initializers (constant weights).

void fuse_conv_bn(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());

    for (int i = 0; i + 1 < n; ++i) {
        operator_t* conv = nodes[i];
        operator_t* bn = nodes[i + 1];
        if (conv->skip || bn->skip) continue;
        if (conv->op_type != "Conv" || bn->op_type != "BatchNormalization") continue;
        if (conv->outputs.empty() || bn->inputs.empty()) continue;
        if (conv->outputs[0] != bn->inputs[0]) continue;
        // BN needs 5 inputs: x, scale, bias, mean, var
        if (bn->inputs.size() < 5) continue;
        // Conv output must feed only into BN
        int users = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            for (auto* t : nodes[j]->inputs)
                if (t == conv->outputs[0]) users++;
        }
        if (users != 1) continue;
        // All BN parameters must be initializers (constant weights)
        bool all_init = true;
        for (int k = 1; k <= 4; ++k) {
            if (!bn->inputs[k] || bn->inputs[k]->ndata == 0 ||
                ctx->initializer_names.find(bn->inputs[k]->name) == ctx->initializer_names.end()) {
                all_init = false;
                break;
            }
        }
        if (!all_init) continue;
        // Only handle float32 for now
        tensor_t* w = conv->inputs[1]; // Conv weight [M, C/g, kH, kW]
        if (!w || w->type != NNR_DATA_TYPE_FLOAT32) continue;

        const tensor_t* bn_scale = bn->inputs[1];
        const tensor_t* bn_bias  = bn->inputs[2];
        const tensor_t* bn_mean  = bn->inputs[3];
        const tensor_t* bn_var   = bn->inputs[4];
        float epsilon = 1e-5f;  // ONNX BatchNormalization default epsilon
        auto* eps_attr = bn->find_attr(attr_key_t::epsilon);
        if (eps_attr && eps_attr->kind == attr_t::kind_t::FLOAT) epsilon = eps_attr->f;

        int M = w->dims[0]; // number of output channels
        int wpc = (int)(w->ndata / M); // weights per output channel
        float* pw = (float*)w->data;
        const float* sc = (const float*)bn_scale->data;
        const float* bi = (const float*)bn_bias->data;
        const float* me = (const float*)bn_mean->data;
        const float* va = (const float*)bn_var->data;

        // Fuse into Conv weights: W_new[m] = W_old[m] * factor[m]
        // where factor[m] = scale[m] / sqrt(var[m] + eps)
        for (int m = 0; m < M; ++m) {
            float factor = sc[m] / sqrtf(va[m] + epsilon);
            float* wm = pw + (size_t)m * wpc;
            for (int j = 0; j < wpc; ++j)
                wm[j] *= factor;
        }

        // Fuse into Conv bias (create if doesn't exist)
        tensor_t* conv_bias = (conv->inputs.size() > 2) ? conv->inputs[2] : nullptr;
        if (!conv_bias || conv_bias->ndata == 0) {
            // Conv has no bias — create a new bias tensor
            small_vector<int> bdims(1);
            bdims[0] = M;
            auto* new_bias = new tensor_t("__fused_bn_bias__", NNR_DATA_TYPE_FLOAT32, bdims);
            ctx->map.emplace_back(new_bias->name, new_bias);
            for (int m = 0; m < M; ++m) {
                float factor = sc[m] / sqrtf(va[m] + epsilon);
                ((float*)new_bias->data)[m] = factor * (-me[m]) + bi[m];
            }
            if (conv->inputs.size() >= 3) {
                // Slot exists but is null — just assign
                conv->inputs[2] = new_bias;
            } else {
                // Need to extend inputs span: allocate new 3-element buffer from attr_pool
                size_t new_count = 3;
                tensor_t** buf = ctx->attr_pool.alloc_arr<tensor_t*>(new_count);
                for (size_t k = 0; k < conv->inputs.size(); ++k)
                    buf[k] = conv->inputs[k];
                buf[2] = new_bias;
                conv->inputs = std::span<tensor_t*>(buf, new_count);
            }
        } else {
            // Conv already has bias — fuse BN into it
            float* bp = (float*)conv_bias->data;
            for (int m = 0; m < M; ++m) {
                float factor = sc[m] / sqrtf(va[m] + epsilon);
                bp[m] = factor * (bp[m] - me[m]) + bi[m];
            }
        }

        // Skip BN — wire its output to Conv's output
        bn->skip = true;
    }
}

} // namespace nnr
