#include <cmath>
#include <cstring>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct RotaryEmbedding_operator : public operator_t {
    int interleaved;
    int num_heads;
    int rotary_dim_attr;

    bool init() override {
        if (inputs.size() < 3 || outputs.empty()) return false;
        interleaved = attribute(attr_key_t::interleaved, (int32_t)0);
        num_heads = attribute(attr_key_t::num_heads, (int32_t)0);
        rotary_dim_attr = attribute(attr_key_t::rotary_embedding_dim, (int32_t)0);
        return true;
    }

    bool reshape() override {
        return outputs[0]->reshape_identity(inputs[0]);
    }

    template <typename T>
    bool exec() {
        const tensor_t* x = inputs[0];
        const tensor_t* cos_cache = inputs[1];
        const tensor_t* sin_cache = inputs[2];
        const tensor_t* position_ids = (inputs.size() > 3 && inputs[3] && inputs[3]->ndata > 0) ? inputs[3] : nullptr;
        tensor_t* y = outputs[0];

        const T* px = (const T*)x->data;
        const T* pcos = (const T*)cos_cache->data;
        const T* psin = (const T*)sin_cache->data;
        T* py = (T*)y->data;

        // Copy input to output first
        memcpy(py, px, x->ndata * sizeof(T));

        bool is_3d = (x->ndim == 3);
        int batch, n_heads, seq_len, head_dim;

        if (is_3d) {
            // [batch, seq_len, hidden_dim]
            batch = x->dims[0];
            seq_len = x->dims[1];
            int hidden_dim = x->dims[2];
            n_heads = num_heads > 0 ? num_heads : 1;
            head_dim = hidden_dim / n_heads;
        } else {
            // [batch, num_heads, seq_len, head_dim]
            batch = x->dims[0];
            n_heads = x->dims[1];
            seq_len = x->dims[2];
            head_dim = x->dims[3];
        }

        int rotary_dim = rotary_dim_attr > 0 ? rotary_dim_attr : head_dim;
        int half_rotary = rotary_dim / 2;

        // cos_cache/sin_cache shape:
        // If 2D: [max_seq_len, half_rotary]
        // If 3D: [batch, seq_len, half_rotary]
        bool cache_3d = (cos_cache->ndim == 3);
        int cache_dim = cos_cache->dims[cos_cache->ndim - 1]; // half_rotary

        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                // Get position
                int pos;
                if (position_ids) {
                    const int64_t* pids = (const int64_t*)position_ids->data;
                    pos = (int)pids[b * seq_len + s];
                } else {
                    pos = s; // default sequential
                }

                // Get cos/sin for this position
                const T* cos_ptr;
                const T* sin_ptr;
                if (cache_3d) {
                    cos_ptr = pcos + (b * seq_len + s) * cache_dim;
                    sin_ptr = psin + (b * seq_len + s) * cache_dim;
                } else {
                    cos_ptr = pcos + pos * cache_dim;
                    sin_ptr = psin + pos * cache_dim;
                }

                for (int h = 0; h < n_heads; ++h) {
                    T* out;
                    if (is_3d) {
                        out = py + (b * seq_len + s) * (n_heads * head_dim) + h * head_dim;
                    } else {
                        out = py + ((b * n_heads + h) * seq_len + s) * head_dim;
                    }

                    if (interleaved) {
                        // Interleaved: pairs (x0,x1), (x2,x3), ...
                        for (int i = 0; i < half_rotary; ++i) {
                            double x0 = (double)out[2*i];
                            double x1 = (double)out[2*i+1];
                            double c = (double)cos_ptr[i];
                            double sn = (double)sin_ptr[i];
                            out[2*i]   = (T)(x0 * c - x1 * sn);
                            out[2*i+1] = (T)(x0 * sn + x1 * c);
                        }
                    } else {
                        // Non-interleaved: first half and second half
                        for (int i = 0; i < half_rotary; ++i) {
                            double x0 = (double)out[i];
                            double x1 = (double)out[i + half_rotary];
                            double c = (double)cos_ptr[i];
                            double sn = (double)sin_ptr[i];
                            out[i]              = (T)(x0 * c - x1 * sn);
                            out[i + half_rotary] = (T)(x0 * sn + x1 * c);
                        }
                    }
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<RotaryEmbedding_operator,
            float16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_RotaryEmbedding(int opset, pool_t& pool) { return pool_new<RotaryEmbedding_operator>(pool); }

} // namespace nnr
