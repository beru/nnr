// Fused Scaled Dot-Product Attention operator.
// Created by graph optimizer when it detects: MatMul(Q, K^T) → Softmax → MatMul(attn, V)
// Inputs: Q[B,S,D], K[B,S,D] (not transposed), V[B,S,D]
// Output: O[B,S,D]

#include "nnr.h"
#include "util.h"
#include "cpu_features.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/sdpa_avx512.h"
#elif defined(NNR_ARCH_ARM64)
#include "backend/arm/sdpa_neon.h"
#endif

namespace nnr {

namespace {

struct SDPA_operator : public operator_t {
    bool init() override {
        if (inputs.size() != 3 || outputs.empty()) return false;
        return true;
    }

    bool reshape() override {
        // Output has same shape as Q
        return outputs[0]->reshape_identity(inputs[0]);
    }

    bool exec() override {
        const tensor_t* Q = inputs[0]; // [B, S, D]
        const tensor_t* K = inputs[1]; // [B, S, D]
        const tensor_t* V = inputs[2]; // [B, S, D]
        tensor_t* O = outputs[0];

        if (Q->ndim < 2 || K->ndim < 2 || V->ndim < 2) return false;

        int head_dim = Q->dims[Q->ndim - 1];
        int seq_len  = Q->dims[Q->ndim - 2];
        int batch = 1;
        for (int i = 0; i < Q->ndim - 2; i++) batch *= Q->dims[i];

#ifdef NNR_ARCH_X64
        if (Q->type == NNR_DATA_TYPE_FLOAT32) {
            sdpa_multihead_avx512(
                (const float*)Q->data, (const float*)K->data,
                (const float*)V->data, (float*)O->data,
                batch, seq_len, head_dim);
            return true;
        }
#elif defined(NNR_ARCH_ARM64)
        if (Q->type == NNR_DATA_TYPE_FLOAT32) {
            sdpa_multihead_neon(
                (const float*)Q->data, (const float*)K->data,
                (const float*)V->data, (float*)O->data,
                batch, seq_len, head_dim);
            return true;
        }
#endif
        // Scalar fallback
        size_t head_stride = (size_t)seq_len * head_dim;
        std::vector<float> scores(seq_len);

        for (int b = 0; b < batch; b++) {
            const float* pq = (const float*)Q->data + b * head_stride;
            const float* pk = (const float*)K->data + b * head_stride;
            const float* pv = (const float*)V->data + b * head_stride;
            float* po = (float*)O->data + b * head_stride;

            for (int i = 0; i < seq_len; i++) {
                // scores[j] = dot(Q[i,:], K[j,:])
                for (int j = 0; j < seq_len; j++) {
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++)
                        dot += pq[i * head_dim + d] * pk[j * head_dim + d];
                    scores[j] = dot;
                }
                // softmax
                float mx = *std::max_element(scores.begin(), scores.begin() + seq_len);
                float sum = 0;
                for (int j = 0; j < seq_len; j++) {
                    scores[j] = expf(scores[j] - mx);
                    sum += scores[j];
                }
                float inv = 1.0f / sum;
                for (int j = 0; j < seq_len; j++) scores[j] *= inv;
                // output[i,:] = scores × V
                for (int d = 0; d < head_dim; d++) {
                    float v = 0;
                    for (int j = 0; j < seq_len; j++)
                        v += scores[j] * pv[j * head_dim + d];
                    po[i * head_dim + d] = v;
                }
            }
        }
        return true;
    }
};

} // namespace

operator_t* create_sdpa_operator(pool_t& pool) {
    return pool_new<SDPA_operator>(pool);
}

} // namespace nnr
