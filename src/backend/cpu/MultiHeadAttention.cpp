// com.microsoft MultiHeadAttention contrib op.
// Inputs: Q[B,S,H], K[B,S,H], V[B,S,H], bias(unused), key_padding_mask(unused),
//         attention_mask[B,1,S,S], past_key(unused), past_value(unused)
// Output: output[B,S,H]
// Attributes: num_heads (int), scale (float)
//
// Q, K, V arrive pre-projected. The op does multi-head scaled dot-product attention:
//   reshape to [B, num_heads, S, head_dim], compute Q@K^T * scale + mask, softmax, @V.

#include "nnr.h"
#include "arena.h"

namespace nnr {

namespace {

struct MultiHeadAttention_operator : public operator_t {
    int num_heads = 0;
    float scale_val = 0.0f;

    bool init() override {
        if (inputs.size() < 3 || outputs.empty()) return false;
        num_heads = attribute(attr_key_t::num_heads, (int32_t)0);
        scale_val = attribute(attr_key_t::scale, 0.0f);
        if (num_heads <= 0) return false;
        return true;
    }

    bool reshape() override {
        // Output has same shape as Q
        return outputs[0]->reshape_identity(inputs[0]);
    }

    size_t workspace_size() const override {
        // Pre-size arena for scores buffer: seq_len floats per query row.
        const tensor_t* Q = inputs[0];
        if (Q->ndim < 2) return 0;
        int seq_len = Q->dims[Q->ndim - 2];
        return (size_t)seq_len * sizeof(double);
    }

    bool exec() override {
        arena_scope_t scope(ctx->arena);
        const tensor_t* Q_in = inputs[0]; // [B, S, H]
        const tensor_t* K_in = inputs[1];
        const tensor_t* V_in = inputs[2];
        // Input 3 = bias (unused), 4 = key_padding_mask (unused), 5 = attention_mask
        const tensor_t* mask_in = (inputs.size() > 5 && inputs[5] && inputs[5]->ndata > 0)
                                  ? inputs[5] : nullptr;

        tensor_t* Y = outputs[0];

        if (Q_in->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (Q_in->ndim < 2) return false;

        const int nh = num_heads;
        const int H = Q_in->dims[Q_in->ndim - 1];
        const int head_dim = H / nh;
        const int S = Q_in->dims[Q_in->ndim - 2];
        int batch = 1;
        for (int i = 0; i < Q_in->ndim - 2; i++) batch *= Q_in->dims[i];

        const float sc = (scale_val != 0.0f) ? scale_val : 1.0f / std::sqrt((float)head_dim);

        const float* pQ = (const float*)Q_in->data;
        const float* pK = (const float*)K_in->data;
        const float* pV = (const float*)V_in->data;
        float* pY = (float*)Y->data;

        // Mask shape: [B, 1, S, S] (additive) — head dim is broadcast
        const float* pM = mask_in ? (const float*)mask_in->data : nullptr;
        // mask_stride_batch = 1 * S * S (the "1" is the broadcast head dim)
        const int mask_stride_batch = S * S;

        double* scores = scope.alloc_arr<double>(S);

        // Q/K/V layout: [B, S, num_heads * head_dim]
        // We treat heads as interleaved in the last dimension:
        //   Q[b, s, h * head_dim + d] = Q for batch b, seq s, head h, dim d
        const size_t seq_stride = (size_t)H; // stride between consecutive seq positions
        const size_t batch_stride = (size_t)S * H;

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < nh; h++) {
                for (int sq = 0; sq < S; sq++) {
                    // Q row: Q[b, sq, h*head_dim .. (h+1)*head_dim]
                    const float* q_row = pQ + b * batch_stride + sq * seq_stride + h * head_dim;

                    // Compute scores[sk] = dot(Q[sq], K[sk]) * scale
                    for (int sk = 0; sk < S; sk++) {
                        const float* k_row = pK + b * batch_stride + sk * seq_stride + h * head_dim;
                        double dot = 0;
                        for (int d = 0; d < head_dim; d++)
                            dot += (double)q_row[d] * (double)k_row[d];
                        scores[sk] = dot * sc;
                    }

                    // Add attention mask (additive, broadcast over heads)
                    if (pM) {
                        int mb = (mask_in->dims[0] == 1) ? 0 : b;
                        const float* mask_row = pM + mb * mask_stride_batch + sq * S;
                        for (int sk = 0; sk < S; sk++)
                            scores[sk] += (double)mask_row[sk];
                    }

                    // Softmax
                    double max_score = scores[0];
                    for (int sk = 1; sk < S; sk++)
                        if (scores[sk] > max_score) max_score = scores[sk];
                    double sum_exp = 0;
                    for (int sk = 0; sk < S; sk++) {
                        scores[sk] = std::exp(scores[sk] - max_score);
                        sum_exp += scores[sk];
                    }
                    double inv = 1.0 / sum_exp;
                    for (int sk = 0; sk < S; sk++)
                        scores[sk] *= inv;

                    // Weighted sum of V
                    float* y_row = pY + b * batch_stride + sq * seq_stride + h * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        double sum = 0;
                        for (int sk = 0; sk < S; sk++) {
                            const float* v_row = pV + b * batch_stride + sk * seq_stride + h * head_dim;
                            sum += scores[sk] * (double)v_row[d];
                        }
                        y_row[d] = (float)sum;
                    }
                }
            }
        }
        return true;
    }

    int64_t num_ops() const override {
        const tensor_t* Q = inputs[0];
        if (Q->ndim < 2) return 0;
        int S = Q->dims[Q->ndim - 2];
        int H = Q->dims[Q->ndim - 1];
        int batch = 1;
        for (int i = 0; i < Q->ndim - 2; i++) batch *= Q->dims[i];
        // Per head: S * S * head_dim (QK^T) + S * S * head_dim (attn@V) = 2*S*S*head_dim
        // Total: batch * num_heads * 2 * S * S * head_dim = batch * 2 * S * S * H
        return (int64_t)batch * 2 * S * S * H;
    }
};

} // namespace

// @nnr-meta-op mt=no workspace=yes
operator_t* resolver_default_op_MultiHeadAttention(int opset, pool_t& pool) {
    return pool_new<MultiHeadAttention_operator>(pool);
}

} // namespace nnr
