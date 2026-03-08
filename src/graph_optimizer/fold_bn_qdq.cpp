#include "graph_optimizer/graph_optimizer_internal.h"

namespace nnr {

// ---------------------------------------------------------------------------
// QDQ fusion: fold DequantizeLinear/QuantizeLinear into tensor metadata.
// ---------------------------------------------------------------------------
// Detects patterns where DequantizeLinear feeds an operator whose output
// goes to QuantizeLinear. Propagates scale/zero_point onto tensor_t's
// quant_* fields and marks the DQ/Q nodes as skip. The target operator
// ---------------------------------------------------------------------------
// Fold DQ → BatchNormalization → Q into a per-channel uint8→uint8 requantize.
// Eliminates float round-trip: DQ(uint8→float) + BN(float→float) + Q(float→uint8)
// becomes BN(uint8→uint8) with precomputed combined scale/offset.
//
// Combined transform per channel c:
//   A[c] = dq_scale * gamma[c] / q_scale
//   B[c] = (beta[c] - dq_zp * dq_scale * gamma[c]) / q_scale + q_zp
//   y = clamp(round(x * A[c] + B[c]), 0, 255)
// where gamma[c] = bn_scale[c] / sqrt(bn_var[c] + eps)
//       beta[c]  = bn_bias[c] - gamma[c] * bn_mean[c]
// ---------------------------------------------------------------------------
void fold_bn_qdq(context_t* ctx)
{
    if (!ctx->graph) return;
    auto& nodes = ctx->graph->nodes;
    const int n = static_cast<int>(nodes.size());
    int folded_count = 0;

    // Build producer map
    std::unordered_map<tensor_t*, int> producer;
    for (int i = 0; i < n; i++) {
        if (nodes[i]->skip || nodes[i]->folded) continue;
        for (auto* t : nodes[i]->outputs)
            producer[t] = i;
    }

    // Count consumers
    auto count_consumers = [&](tensor_t* tensor, int skip_idx) -> int {
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (j == skip_idx || nodes[j]->skip || nodes[j]->folded) continue;
            for (auto* t : nodes[j]->inputs)
                if (t == tensor) count++;
        }
        return count;
    };

    for (int i = 0; i < n; i++) {
        operator_t* dq = nodes[i];
        if (dq->skip || dq->folded) continue;
        if (dq->op_type != "DequantizeLinear") continue;
        if (dq->outputs.empty() || dq->inputs.size() < 2) continue;

        tensor_t* dq_out = dq->outputs[0];
        if (count_consumers(dq_out, i) != 1) continue;

        // Find BN consumer
        int bn_idx = -1;
        for (int j = 0; j < n; j++) {
            if (j == i || nodes[j]->skip || nodes[j]->folded) continue;
            if (nodes[j]->op_type != "BatchNormalization") continue;
            if (!nodes[j]->inputs.empty() && nodes[j]->inputs[0] == dq_out) {
                bn_idx = j; break;
            }
        }
        if (bn_idx < 0) continue;

        operator_t* bn = nodes[bn_idx];
        if (bn->inputs.size() < 5 || bn->outputs.empty()) continue;
        tensor_t* bn_out = bn->outputs[0];
        if (count_consumers(bn_out, bn_idx) != 1) continue;

        // Find Q consumer of BN
        int q_idx = -1;
        for (int j = 0; j < n; j++) {
            if (j == bn_idx || nodes[j]->skip || nodes[j]->folded) continue;
            if (nodes[j]->op_type != "QuantizeLinear") continue;
            if (!nodes[j]->inputs.empty() && nodes[j]->inputs[0] == bn_out) {
                q_idx = j; break;
            }
        }
        if (q_idx < 0) continue;

        operator_t* q = nodes[q_idx];
        if (q->inputs.size() < 2 || q->outputs.empty()) continue;

        // Extract DQ scale (per-tensor only)
        tensor_t* dq_scale_t = dq->inputs[1];
        if (!dq_scale_t || dq_scale_t->ndata != 1 || dq_scale_t->type != NNR_DATA_TYPE_FLOAT32)
            continue;
        float dq_scale = *(float*)dq_scale_t->data;

        int32_t dq_zp = 0;
        if (dq->inputs.size() >= 3 && dq->inputs[2] && dq->inputs[2]->ndata > 0) {
            tensor_t* zpt = dq->inputs[2];
            if (zpt->type == NNR_DATA_TYPE_UINT8) dq_zp = *(uint8_t*)zpt->data;
            else if (zpt->type == NNR_DATA_TYPE_INT8) dq_zp = *(int8_t*)zpt->data;
        }

        // Extract Q scale (per-tensor only)
        tensor_t* q_scale_t = q->inputs[1];
        if (!q_scale_t || q_scale_t->ndata != 1 || q_scale_t->type != NNR_DATA_TYPE_FLOAT32)
            continue;
        float q_scale = *(float*)q_scale_t->data;

        int32_t q_zp = 0;
        if (q->inputs.size() >= 3 && q->inputs[2] && q->inputs[2]->ndata > 0) {
            tensor_t* zpt = q->inputs[2];
            if (zpt->type == NNR_DATA_TYPE_UINT8) q_zp = *(uint8_t*)zpt->data;
            else if (zpt->type == NNR_DATA_TYPE_INT8) q_zp = *(int8_t*)zpt->data;
        }

        // All BN parameters must be float32 initializers
        bool all_init = true;
        for (int k = 1; k <= 4; ++k) {
            tensor_t* t = bn->inputs[k];
            if (!t || t->ndata == 0 || t->type != NNR_DATA_TYPE_FLOAT32 ||
                ctx->initializer_names.find(t->name) == ctx->initializer_names.end()) {
                all_init = false; break;
            }
        }
        if (!all_init) continue;

        // Compute combined per-channel A[c] and B[c]
        tensor_t* bn_scale_t = bn->inputs[1];
        tensor_t* bn_bias_t  = bn->inputs[2];
        tensor_t* bn_mean_t  = bn->inputs[3];
        tensor_t* bn_var_t   = bn->inputs[4];
        float epsilon = 1e-5f;
        auto* eps_attr = bn->find_attr(attr_key_t::epsilon);
        if (eps_attr && eps_attr->kind == attr_t::kind_t::FLOAT) epsilon = eps_attr->f;

        int C = (int)bn_scale_t->ndata;
        float* sc = (float*)bn_scale_t->data;
        float* bi = (float*)bn_bias_t->data;
        const float* me = (const float*)bn_mean_t->data;
        const float* va = (const float*)bn_var_t->data;

        // Overwrite BN scale/bias with combined A[c]/B[c]
        for (int c = 0; c < C; ++c) {
            float gamma = sc[c] / sqrtf(va[c] + epsilon);
            float beta  = bi[c] - gamma * me[c];
            float A = dq_scale * gamma / q_scale;
            float B = (beta - dq_zp * dq_scale * gamma) / q_scale + (float)q_zp;
            sc[c] = A;
            bi[c] = B;
        }

        // Rewire: BN reads DQ's uint8 input, writes Q's uint8 output
        tensor_t* uint8_in  = dq->inputs[0];
        tensor_t* uint8_out = q->outputs[0];

        bn->inputs[0] = uint8_in;
        bn->outputs[0] = uint8_out;
        uint8_out->reinit(uint8_in->type, bn_out->dim_span());

        // Mark DQ and Q as skip
        dq->skip = true;
        q->skip = true;

        bn->reshape();
        folded_count++;
    }

    // Second pass: fold BN→Q where BN input is fp32 (no DQ predecessor).
    // Combined: q = round(A[c] * x + B[c]) where
    //   A[c] = gamma[c] / (sqrt(var+eps) * q_scale)
    //   B[c] = (beta[c] - gamma[c]*mean[c]/sqrt(var+eps)) / q_scale + q_zp
    int folded_bnq = 0;
    for (int i = 0; i < n; i++) {
        operator_t* bn = nodes[i];
        if (bn->skip || bn->folded) continue;
        if (bn->op_type != "BatchNormalization") continue;
        if (bn->inputs.size() < 5 || bn->outputs.empty()) continue;

        tensor_t* bn_in = bn->inputs[0];
        if (!bn_in || bn_in->type != NNR_DATA_TYPE_FLOAT32) continue;

        tensor_t* bn_out = bn->outputs[0];
        if (count_consumers(bn_out, i) != 1) continue;

        // Find Q consumer of BN
        int q_idx = -1;
        for (int j = 0; j < n; j++) {
            if (j == i || nodes[j]->skip || nodes[j]->folded) continue;
            if (nodes[j]->op_type != "QuantizeLinear") continue;
            if (!nodes[j]->inputs.empty() && nodes[j]->inputs[0] == bn_out) {
                q_idx = j; break;
            }
        }
        if (q_idx < 0) continue;

        operator_t* q = nodes[q_idx];
        if (q->inputs.size() < 2 || q->outputs.empty()) continue;

        // Extract Q scale (per-tensor only)
        tensor_t* q_scale_t = q->inputs[1];
        if (!q_scale_t || q_scale_t->ndata != 1 || q_scale_t->type != NNR_DATA_TYPE_FLOAT32)
            continue;
        float q_scale = *(float*)q_scale_t->data;

        int32_t q_zp = 0;
        if (q->inputs.size() >= 3 && q->inputs[2] && q->inputs[2]->ndata > 0) {
            tensor_t* zpt = q->inputs[2];
            if (zpt->type == NNR_DATA_TYPE_UINT8) q_zp = *(uint8_t*)zpt->data;
            else if (zpt->type == NNR_DATA_TYPE_INT8) q_zp = *(int8_t*)zpt->data;
        }

        // All BN parameters must be float32 initializers
        bool all_init = true;
        for (int k = 1; k <= 4; ++k) {
            tensor_t* t = bn->inputs[k];
            if (!t || t->ndata == 0 || t->type != NNR_DATA_TYPE_FLOAT32 ||
                ctx->initializer_names.find(t->name) == ctx->initializer_names.end()) {
                all_init = false; break;
            }
        }
        if (!all_init) continue;

        // Compute combined per-channel A[c] and B[c]
        tensor_t* bn_scale_t = bn->inputs[1];
        tensor_t* bn_bias_t  = bn->inputs[2];
        tensor_t* bn_mean_t  = bn->inputs[3];
        tensor_t* bn_var_t   = bn->inputs[4];
        float epsilon = 1e-5f;
        auto* eps_attr = bn->find_attr(attr_key_t::epsilon);
        if (eps_attr && eps_attr->kind == attr_t::kind_t::FLOAT) epsilon = eps_attr->f;

        int C = (int)bn_scale_t->ndata;
        float* sc = (float*)bn_scale_t->data;
        float* bi = (float*)bn_bias_t->data;
        const float* me = (const float*)bn_mean_t->data;
        const float* va = (const float*)bn_var_t->data;

        // Overwrite BN scale/bias with combined A[c]/B[c]
        // A[c] = gamma / (sqrt(var+eps) * q_scale)
        // B[c] = (beta - gamma*mean/sqrt(var+eps)) / q_scale + q_zp
        for (int c = 0; c < C; ++c) {
            float gamma = sc[c] / sqrtf(va[c] + epsilon);
            float beta  = bi[c] - gamma * me[c];
            sc[c] = gamma / q_scale;
            bi[c] = beta / q_scale + (float)q_zp;
        }

        // Rewire: BN output becomes Q's uint8 output
        // Don't call bn->reshape() — it would overwrite uint8 type with fp32
        tensor_t* uint8_out = q->outputs[0];
        bn->outputs[0] = uint8_out;
        uint8_out->reinit(NNR_DATA_TYPE_UINT8, bn_out->dim_span());

        q->skip = true;
        folded_bnq++;
    }

    if (folded_count > 0 || folded_bnq > 0)
        fprintf(stderr, "[NNR] Folded %d DQ->BN->Q + %d BN->Q patterns\n",
                folded_count, folded_bnq);
}

} // namespace nnr
