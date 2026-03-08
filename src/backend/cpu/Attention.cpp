#include <cmath>
#include <limits>
#include "nnr.h"
#include "util.h"
#include "arena.h"

namespace nnr {

namespace {

struct Attention_operator : public operator_t {
    int is_causal;
    int64_t q_num_heads;
    int64_t kv_num_heads;
    float scale;
    float softcap;
    int qk_matmul_output_mode;

    bool init() override {
        if (inputs.size() < 3 || outputs.empty()) return false;
        is_causal = (int)attribute(attr_key_t::is_causal, (int64_t)0);
        q_num_heads = attribute(attr_key_t::q_num_heads, (int64_t)0);
        kv_num_heads = attribute(attr_key_t::kv_num_heads, (int64_t)0);
        scale = attribute(attr_key_t::scale, 0.0f);
        softcap = attribute(attr_key_t::softcap, 0.0f);
        qk_matmul_output_mode = (int)attribute(attr_key_t::qk_matmul_output_mode, (int64_t)0);
        return true;
    }

    bool reshape() override {
        const tensor_t* Q = inputs[0];
        const tensor_t* K = inputs[1];
        const tensor_t* V = inputs[2];
        tensor_t* Y = outputs[0];

        int qh, kvh, vd;
        if (Q->ndim == 3) {
            qh = (int)q_num_heads; if (qh == 0) qh = 1;
            kvh = (int)kv_num_heads; if (kvh == 0) kvh = qh;
            vd = V->dims[2] / kvh;
            small_vector<int> dims(3);
            dims[0] = Q->dims[0]; dims[1] = Q->dims[1]; dims[2] = qh * vd;
            Y->reshape(dims, Q->type);
        } else if (Q->ndim == 4) {
            qh = Q->dims[1]; kvh = K->dims[1]; vd = V->dims[3];
            small_vector<int> dims(4);
            dims[0] = Q->dims[0]; dims[1] = qh; dims[2] = Q->dims[2]; dims[3] = vd;
            Y->reshape(dims, Q->type);
        } else {
            return false;
        }

        // Determine total KV sequence length for present outputs
        const tensor_t* past_key_t = (inputs.size() > 4 && inputs[4] && inputs[4]->ndata > 0) ? inputs[4] : nullptr;
        if (past_key_t) {
            int past_seq = past_key_t->dims[2];
            int kd_r, vd_r, new_seq;
            if (Q->ndim == 4) {
                kd_r = K->dims[3]; vd_r = V->dims[3];
                new_seq = past_seq + K->dims[2];
            } else {
                kd_r = K->dims[2] / kvh; vd_r = V->dims[2] / kvh;
                new_seq = past_seq + K->dims[1];
            }
            if (outputs.size() > 1 && outputs[1]) {
                small_vector<int> dims(4);
                dims[0] = Q->dims[0]; dims[1] = kvh; dims[2] = new_seq; dims[3] = kd_r;
                outputs[1]->reshape(dims, K->type);
            }
            if (outputs.size() > 2 && outputs[2] && inputs.size() >= 6 && inputs[5]) {
                small_vector<int> dims(4);
                dims[0] = Q->dims[0]; dims[1] = kvh; dims[2] = new_seq; dims[3] = vd_r;
                outputs[2]->reshape(dims, V->type);
            }
        }

        // qk_matmul_output (output[3])
        if (outputs.size() > 3 && outputs[3]) {
            int seq_q_v = (Q->ndim == 4) ? Q->dims[2] : Q->dims[1];
            int seq_kv_v = (Q->ndim == 4) ? K->dims[2] : K->dims[1];
            if (past_key_t) seq_kv_v += past_key_t->dims[2];
            small_vector<int> dims(4);
            dims[0] = Q->dims[0]; dims[1] = qh; dims[2] = seq_q_v; dims[3] = seq_kv_v;
            outputs[3]->reshape(dims, Q->type);
        }
        return true;
    }

    template <typename T>
    bool exec() {
        arena_scope_t scope(ctx->arena);
        const tensor_t* Q_in = inputs[0];
        const tensor_t* K_in = inputs[1];
        const tensor_t* V_in = inputs[2];
        const tensor_t* mask_in = (inputs.size() > 3 && inputs[3] && inputs[3]->ndata > 0) ? inputs[3] : nullptr;
        const tensor_t* past_key = (inputs.size() > 4 && inputs[4] && inputs[4]->ndata > 0) ? inputs[4] : nullptr;
        const tensor_t* past_value = (inputs.size() > 5 && inputs[5] && inputs[5]->ndata > 0) ? inputs[5] : nullptr;
        tensor_t* Y = outputs[0];
        tensor_t* qk_out = (outputs.size() > 3 && outputs[3] && outputs[3]->ndata > 0) ? outputs[3] : nullptr;
        int qk_mode = qk_matmul_output_mode;

        const T* pQ = (const T*)Q_in->data;
        const T* pK = (const T*)K_in->data;
        const T* pV = (const T*)V_in->data;
        T* pY = (T*)Y->data;

        int batch, qh, kvh, seq_q, seq_kv, qd, kd, vd;

        if (Q_in->ndim == 4) {
            batch = Q_in->dims[0]; qh = Q_in->dims[1]; kvh = K_in->dims[1];
            seq_q = Q_in->dims[2]; seq_kv = K_in->dims[2];
            qd = Q_in->dims[3]; kd = K_in->dims[3]; vd = V_in->dims[3];
        } else {
            batch = Q_in->dims[0];
            qh = (int)q_num_heads; kvh = (int)kv_num_heads;
            if (qh == 0) qh = 1; if (kvh == 0) kvh = qh;
            seq_q = Q_in->dims[1]; seq_kv = K_in->dims[1];
            qd = Q_in->dims[2] / qh; kd = K_in->dims[2] / kvh; vd = V_in->dims[2] / kvh;
        }

        // Build full key/value with past concatenation
        int total_kv_seq = seq_kv;
        int past_seq = 0;
        T* full_key = nullptr;
        T* full_value = nullptr;
        const T* key_ptr = pK;
        const T* val_ptr = pV;
        int key_stride_batch, key_stride_head, key_stride_seq;
        int val_stride_batch, val_stride_head, val_stride_seq;

        if (past_key) {
            past_seq = past_key->dims[2];
            total_kv_seq = past_seq + seq_kv;
            full_key = scope.alloc_arr<T>(batch * kvh * total_kv_seq * kd);
            full_value = scope.alloc_arr<T>(batch * kvh * total_kv_seq * vd);
            const T* ppk = (const T*)past_key->data;
            const T* ppv = past_value ? (const T*)past_value->data : nullptr;

            if (Q_in->ndim == 4) {
                for (int b = 0; b < batch; ++b) {
                    for (int h = 0; h < kvh; ++h) {
                        T* dk = full_key + (b * kvh + h) * total_kv_seq * kd;
                        memcpy(dk, ppk + (b * kvh + h) * past_seq * kd, past_seq * kd * sizeof(T));
                        memcpy(dk + past_seq * kd, pK + (b * kvh + h) * seq_kv * kd, seq_kv * kd * sizeof(T));
                        if (ppv) {
                            T* dv = full_value + (b * kvh + h) * total_kv_seq * vd;
                            memcpy(dv, ppv + (b * kvh + h) * past_seq * vd, past_seq * vd * sizeof(T));
                            memcpy(dv + past_seq * vd, pV + (b * kvh + h) * seq_kv * vd, seq_kv * vd * sizeof(T));
                        }
                    }
                }
            } else {
                for (int b = 0; b < batch; ++b) {
                    for (int h = 0; h < kvh; ++h) {
                        T* dk = full_key + (b * kvh + h) * total_kv_seq * kd;
                        memcpy(dk, ppk + (b * kvh + h) * past_seq * kd, past_seq * kd * sizeof(T));
                        for (int s = 0; s < seq_kv; ++s) {
                            for (int d = 0; d < kd; ++d) {
                                dk[(past_seq + s) * kd + d] = pK[b * seq_kv * kvh * kd + s * kvh * kd + h * kd + d];
                            }
                        }
                        if (ppv) {
                            T* dv = full_value + (b * kvh + h) * total_kv_seq * vd;
                            memcpy(dv, ppv + (b * kvh + h) * past_seq * vd, past_seq * vd * sizeof(T));
                            for (int s = 0; s < seq_kv; ++s) {
                                for (int d = 0; d < vd; ++d) {
                                    dv[(past_seq + s) * vd + d] = pV[b * seq_kv * kvh * vd + s * kvh * vd + h * vd + d];
                                }
                            }
                        }
                    }
                }
            }
            key_ptr = full_key;
            val_ptr = full_value;
            key_stride_batch = kvh * total_kv_seq * kd;
            key_stride_head = total_kv_seq * kd;
            key_stride_seq = kd;
            val_stride_batch = kvh * total_kv_seq * vd;
            val_stride_head = total_kv_seq * vd;
            val_stride_seq = vd;

            // Write present outputs
            if (outputs.size() > 1 && outputs[1] && outputs[1]->ndata > 0)
                memcpy(outputs[1]->data, full_key, (size_t)(batch * kvh * total_kv_seq * kd) * sizeof(T));
            if (outputs.size() > 2 && outputs[2] && outputs[2]->ndata > 0)
                memcpy(outputs[2]->data, full_value, (size_t)(batch * kvh * total_kv_seq * vd) * sizeof(T));
        } else if (Q_in->ndim == 4) {
            key_stride_batch = kvh * seq_kv * kd;
            key_stride_head = seq_kv * kd;
            key_stride_seq = kd;
            val_stride_batch = kvh * seq_kv * vd;
            val_stride_head = seq_kv * vd;
            val_stride_seq = vd;
        } else {
            key_stride_batch = seq_kv * kvh * kd;
            key_stride_head = kd;
            key_stride_seq = kvh * kd;
            val_stride_batch = seq_kv * kvh * vd;
            val_stride_head = vd;
            val_stride_seq = kvh * vd;
        }

        int kv_seq = total_kv_seq;
        float sc = scale;
        if (sc == 0.0f) sc = 1.0f / std::sqrt((float)qd);
        int gqa_ratio = qh / kvh;

        bool mask_is_bool = mask_in && mask_in->type == NNR_DATA_TYPE_BOOL;

        double* scores = scope.alloc_arr<double>(kv_seq);
        T* pQK = qk_out ? (T*)qk_out->data : nullptr;

        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < qh; ++h) {
                int kv_h = h / gqa_ratio;
                for (int sq = 0; sq < seq_q; ++sq) {
                    const T* q_row;
                    if (Q_in->ndim == 4) {
                        q_row = pQ + b * qh * seq_q * qd + h * seq_q * qd + sq * qd;
                    } else {
                        q_row = pQ + b * seq_q * qh * qd + sq * qh * qd + h * qd;
                    }

                    // Q * K^T * scale
                    for (int sk = 0; sk < kv_seq; ++sk) {
                        const T* k_row = key_ptr + b * key_stride_batch + kv_h * key_stride_head + sk * key_stride_seq;
                        double dot = 0;
                        for (int d = 0; d < qd; ++d)
                            dot += (double)q_row[d] * (double)k_row[d];
                        scores[sk] = dot * sc;
                    }


                    // Write qk_matmul_output mode 0: Q*K^T * scale (before mask)
                    if (pQK && qk_mode == 0) {
                        int qk_idx = ((b * qh + h) * seq_q + sq) * kv_seq;
                        for (int sk = 0; sk < kv_seq; ++sk)
                            pQK[qk_idx + sk] = (T)scores[sk];
                    }
                    // Apply attention mask (additive bias)
                    if (mask_in) {
                        if (mask_is_bool) {
                            const uint8_t* pm_bool = (const uint8_t*)mask_in->data;
                            for (int sk = 0; sk < kv_seq; ++sk) {
                                int mi;
                                if (mask_in->ndim == 2) {
                                    mi = sq * mask_in->dims[1] + sk;
                                } else if (mask_in->ndim == 3) {
                                    int mb = (mask_in->dims[0] == 1) ? 0 : b;
                                    mi = (mb * mask_in->dims[1] + sq) * mask_in->dims[2] + sk;
                                } else { // 4D
                                    int mb = (mask_in->dims[0] == 1) ? 0 : b;
                                    int mh = (mask_in->dims[1] == 1) ? 0 : h;
                                    mi = ((mb * mask_in->dims[1] + mh) * mask_in->dims[2] + sq) * mask_in->dims[3] + sk;
                                }
                                if (!pm_bool[mi])
                                    scores[sk] = -std::numeric_limits<double>::infinity();
                            }
                        } else {
                            const T* pM = (const T*)mask_in->data;
                            for (int sk = 0; sk < kv_seq; ++sk) {
                                int mi;
                                if (mask_in->ndim == 2) {
                                    mi = sq * mask_in->dims[1] + sk;
                                } else if (mask_in->ndim == 3) {
                                    int mb = (mask_in->dims[0] == 1) ? 0 : b;
                                    mi = (mb * mask_in->dims[1] + sq) * mask_in->dims[2] + sk;
                                } else { // 4D
                                    int mb = (mask_in->dims[0] == 1) ? 0 : b;
                                    int mh = (mask_in->dims[1] == 1) ? 0 : h;
                                    mi = ((mb * mask_in->dims[1] + mh) * mask_in->dims[2] + sq) * mask_in->dims[3] + sk;
                                }
                                scores[sk] += (double)pM[mi];
                            }
                        }
                    }


                    // Apply softcap
                    if (softcap > 0) {
                        for (int sk = 0; sk < kv_seq; ++sk)
                            scores[sk] = softcap * std::tanh(scores[sk] / softcap);
                    }

                    // Write qk_matmul_output mode 2: after softcap
                    if (pQK && qk_mode == 2) {
                        int qk_idx = ((b * qh + h) * seq_q + sq) * kv_seq;
                        for (int sk = 0; sk < kv_seq; ++sk)
                            pQK[qk_idx + sk] = (T)scores[sk];
                    }

                    // Apply causal mask (account for past sequence offset)
                    if (is_causal) {
                        int offset = past_seq; // query position sq corresponds to absolute position offset + sq
                        for (int sk = 0; sk < kv_seq; ++sk) {
                            if (sk > offset + sq)
                                scores[sk] = -std::numeric_limits<double>::infinity();
                        }
                    }

                    // Write qk_matmul_output mode 1: after scale + mask + causal
                    if (pQK && qk_mode == 1) {
                        int qk_idx = ((b * qh + h) * seq_q + sq) * kv_seq;
                        for (int sk = 0; sk < kv_seq; ++sk)
                            pQK[qk_idx + sk] = (T)scores[sk];
                    }

                    // Softmax
                    double max_score = scores[0];
                    for (int sk = 1; sk < kv_seq; ++sk)
                        if (scores[sk] > max_score) max_score = scores[sk];
                    double sum_exp = 0;
                    for (int sk = 0; sk < kv_seq; ++sk) {
                        scores[sk] = std::exp(scores[sk] - max_score);
                        sum_exp += scores[sk];
                    }
                    for (int sk = 0; sk < kv_seq; ++sk)
                        scores[sk] /= sum_exp;

                    // Write qk_matmul_output mode 3: after softmax
                    if (pQK && qk_mode == 3) {
                        int qk_idx = ((b * qh + h) * seq_q + sq) * kv_seq;
                        for (int sk = 0; sk < kv_seq; ++sk)
                            pQK[qk_idx + sk] = (T)scores[sk];
                    }

                    // Weighted sum of V
                    T* y_row;
                    if (Q_in->ndim == 4) {
                        y_row = pY + b * qh * seq_q * vd + h * seq_q * vd + sq * vd;
                    } else {
                        y_row = pY + b * seq_q * qh * vd + sq * qh * vd + h * vd;
                    }
                    for (int d = 0; d < vd; ++d) {
                        double sum = 0;
                        for (int sk = 0; sk < kv_seq; ++sk) {
                            const T* v_row = val_ptr + b * val_stride_batch + kv_h * val_stride_head + sk * val_stride_seq;
                            sum += scores[sk] * (double)v_row[d];
                        }
                        y_row[d] = (T)sum;
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        data_type_t type = inputs[0]->type;
        return typed_exec<Attention_operator,
            float16_t, float, double, bfloat16_t
        >(this, type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Attention(int opset, pool_t& pool) { return pool_new<Attention_operator>(pool); }

} // namespace nnr
