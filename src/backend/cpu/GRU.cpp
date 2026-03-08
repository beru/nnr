#include "nnr.h"
#include "util.h"
#include "arena.h"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace nnr {

namespace {

struct GRU_operator : public operator_t {
    int hidden_size_ = 0;
    int linear_before_reset_ = 0;
    int layout_ = 0;
    int num_directions_ = 1;
    std::string_view direction_;

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    bool init() override {
        // Inputs: X, W, R, [B], [sequence_lens], [initial_h]
        // Outputs: Y, Y_h (both optional but at least one)
        if (inputs.size() < 3 || outputs.size() < 1) {
            return false;
        }
        hidden_size_ = static_cast<int>(attribute(attr_key_t::hidden_size, (int64_t)0));
        linear_before_reset_ = static_cast<int>(attribute(attr_key_t::linear_before_reset, (int64_t)0));
        layout_ = static_cast<int>(attribute(attr_key_t::layout, (int64_t)0));
        direction_ = attribute(attr_key_t::direction, "forward");
        if (direction_ == "bidirectional") {
            num_directions_ = 2;
        } else {
            num_directions_ = 1;
        }
        return true;
    }

    bool reshape() override {
        const tensor_t* X = inputs[0];
        const tensor_t* W = inputs[1];

        // Infer hidden_size from W if not set via attribute
        // W shape: [num_directions, 3*hidden_size, input_size]
        if (hidden_size_ == 0) {
            hidden_size_ = W->dims[1] / 3;
        }

        int seq_length, batch_size;
        if (layout_ == 0) {
            seq_length = X->dims[0];
            batch_size = X->dims[1];
        } else {
            batch_size = X->dims[0];
            seq_length = X->dims[1];
        }

        // Output Y: [seq_length, num_directions, batch_size, hidden_size] (layout=0)
        //        or [batch_size, seq_length, num_directions, hidden_size] (layout=1)
        if (outputs.size() > 0 && outputs[0]) {
            if (layout_ == 0) {
                int dims[] = { seq_length, num_directions_, batch_size, hidden_size_ };
                if (!outputs[0]->reshape(dims, X->type)) return false;
            } else {
                int dims[] = { batch_size, seq_length, num_directions_, hidden_size_ };
                if (!outputs[0]->reshape(dims, X->type)) return false;
            }
        }
        // Output Y_h: [num_directions, batch_size, hidden_size] (layout=0)
        //          or [batch_size, num_directions, hidden_size] (layout=1)
        if (outputs.size() > 1 && outputs[1]) {
            if (layout_ == 0) {
                int dims[] = { num_directions_, batch_size, hidden_size_ };
                if (!outputs[1]->reshape(dims, X->type)) return false;
            } else {
                int dims[] = { batch_size, num_directions_, hidden_size_ };
                if (!outputs[1]->reshape(dims, X->type)) return false;
            }
        }
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* X = inputs[0];
        const tensor_t* W = inputs[1];
        const tensor_t* R = inputs[2];
        const tensor_t* B = (inputs.size() > 3 && inputs[3]) ? inputs[3] : nullptr;
        const tensor_t* seq_lens_t = (inputs.size() > 4 && inputs[4]) ? inputs[4] : nullptr;
        const tensor_t* init_h_t = (inputs.size() > 5 && inputs[5]) ? inputs[5] : nullptr;

        int seq_length, batch_size, input_size;
        if (layout_ == 0) {
            seq_length = X->dims[0];
            batch_size = X->dims[1];
            input_size = X->dims[2];
        } else {
            batch_size = X->dims[0];
            seq_length = X->dims[1];
            input_size = X->dims[2];
        }

        const int HS = hidden_size_;

        // Pointers
        const T* px = (const T*)X->data;
        const T* pw = (const T*)W->data;
        const T* pr = (const T*)R->data;
        const T* pb = B ? (const T*)B->data : nullptr;
        const int* p_seq_lens = seq_lens_t ? (const int*)seq_lens_t->data : nullptr;

        tensor_t* Y_out = (outputs.size() > 0) ? outputs[0] : nullptr;
        tensor_t* Yh_out = (outputs.size() > 1) ? outputs[1] : nullptr;

        T* py = Y_out ? (T*)Y_out->data : nullptr;
        T* pyh = Yh_out ? (T*)Yh_out->data : nullptr;

        // Temp buffers for one direction
        arena_scope_t scope(ctx->arena);
        T* H    = scope.alloc_arr<T>(batch_size * HS);
        T* Hnew = scope.alloc_arr<T>(batch_size * HS);
        T* zt_vec = scope.alloc_arr<T>(HS);
        T* rt_vec = scope.alloc_arr<T>(HS);
        std::fill(H, H + batch_size * HS, T(0));

        for (int dir = 0; dir < num_directions_; ++dir) {
            bool is_reverse = (direction_ == "reverse") || (direction_ == "bidirectional" && dir == 1);

            // W[dir]: shape [3*HS, input_size] — gates in ZRH order
            const T* Wz = pw + dir * 3 * HS * input_size;
            const T* Wr = Wz + HS * input_size;
            const T* Wh = Wr + HS * input_size;

            // R[dir]: shape [3*HS, HS]
            const T* Rz = pr + dir * 3 * HS * HS;
            const T* Rr = Rz + HS * HS;
            const T* Rh = Rr + HS * HS;

            // B[dir]: shape [6*HS] — Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h
            const T* Wbz = nullptr, *Wbr = nullptr, *Wbh = nullptr;
            const T* Rbz = nullptr, *Rbr = nullptr, *Rbh = nullptr;
            if (pb) {
                const T* bias = pb + dir * 6 * HS;
                Wbz = bias;
                Wbr = bias + HS;
                Wbh = bias + 2 * HS;
                Rbz = bias + 3 * HS;
                Rbr = bias + 4 * HS;
                Rbh = bias + 5 * HS;
            }

            // Initialize hidden state
            std::fill(H, H + batch_size * HS, T(0));
            if (init_h_t) {
                const T* p_init = (const T*)init_h_t->data + dir * batch_size * HS;
                std::memcpy(H, p_init, batch_size * HS * sizeof(T));
            }

            for (int ti = 0; ti < seq_length; ++ti) {
                int t = is_reverse ? (seq_length - 1 - ti) : ti;

                for (int b = 0; b < batch_size; ++b) {
                    // Check sequence length
                    if (p_seq_lens && t >= p_seq_lens[b]) {
                        // Past the end of this sequence — keep H unchanged, write to output
                        if (py) {
                            if (layout_ == 0) {
                                T* dst = py + ((size_t)t * num_directions_ * batch_size + dir * batch_size + b) * HS;
                                std::memset(dst, 0, HS * sizeof(T));
                            } else {
                                T* dst = py + ((size_t)b * seq_length * num_directions_ + t * num_directions_ + dir) * HS;
                                std::memset(dst, 0, HS * sizeof(T));
                            }
                        }
                        continue;
                    }

                    // Get input for this timestep and batch
                    const T* xt;
                    if (layout_ == 0) {
                        xt = px + ((size_t)t * batch_size + b) * input_size;
                    } else {
                        xt = px + ((size_t)b * seq_length + t) * input_size;
                    }

                    const T* Hb = H + b * HS;
                    T* Hb_new = Hnew + b * HS;

                    // First pass: compute z and r gates for all units
                    for (int h = 0; h < HS; ++h) {
                        T val_z = T(0);
                        for (int k = 0; k < input_size; ++k)
                            val_z += xt[k] * Wz[h * input_size + k];
                        for (int k = 0; k < HS; ++k)
                            val_z += Hb[k] * Rz[h * HS + k];
                        if (Wbz) val_z += Wbz[h];
                        if (Rbz) val_z += Rbz[h];
                        zt_vec[h] = (T)sigmoid((float)val_z);

                        T val_r = T(0);
                        for (int k = 0; k < input_size; ++k)
                            val_r += xt[k] * Wr[h * input_size + k];
                        for (int k = 0; k < HS; ++k)
                            val_r += Hb[k] * Rr[h * HS + k];
                        if (Wbr) val_r += Wbr[h];
                        if (Rbr) val_r += Rbr[h];
                        rt_vec[h] = (T)sigmoid((float)val_r);
                    }

                    // Second pass: compute h gate and update
                    for (int h = 0; h < HS; ++h) {
                        T val_h = T(0);
                        for (int k = 0; k < input_size; ++k)
                            val_h += xt[k] * Wh[h * input_size + k];
                        if (Wbh) val_h += Wbh[h];

                        if (linear_before_reset_) {
                            T rh_dot = T(0);
                            for (int k = 0; k < HS; ++k)
                                rh_dot += Hb[k] * Rh[h * HS + k];
                            if (Rbh) rh_dot += Rbh[h];
                            val_h += rt_vec[h] * rh_dot;
                        } else {
                            // (rt o H) * Rh^T: use rt_vec[k] for each k
                            T rh_dot = T(0);
                            for (int k = 0; k < HS; ++k)
                                rh_dot += (rt_vec[k] * Hb[k]) * Rh[h * HS + k];
                            if (Rbh) rh_dot += Rbh[h];
                            val_h += rh_dot;
                        }
                        T ht = (T)std::tanh((float)val_h);
                        Hb_new[h] = (T(1) - zt_vec[h]) * ht + zt_vec[h] * Hb[h];
                    }

                    std::memcpy(const_cast<T*>(Hb), Hb_new, HS * sizeof(T));

                    // Write to Y output
                    if (py) {
                        T* dst;
                        if (layout_ == 0) {
                            dst = py + ((size_t)t * num_directions_ * batch_size + dir * batch_size + b) * HS;
                        } else {
                            dst = py + ((size_t)b * seq_length * num_directions_ + t * num_directions_ + dir) * HS;
                        }
                        std::memcpy(dst, Hb, HS * sizeof(T));
                    }
                }
            }

            // Write Y_h (final hidden state)
            if (pyh) {
                if (layout_ == 0) {
                    // Y_h: [num_directions, batch, HS]
                    T* dst = pyh + dir * batch_size * HS;
                    std::memcpy(dst, H, batch_size * HS * sizeof(T));
                } else {
                    // Y_h: [batch, num_directions, HS]
                    for (int b = 0; b < batch_size; ++b) {
                        T* dst = pyh + (b * num_directions_ + dir) * HS;
                        std::memcpy(dst, H + b * HS, HS * sizeof(T));
                    }
                }
            }
        }

        return true;
    }

    bool exec() override {
        return typed_exec<GRU_operator, float, double>(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_GRU(int opset, pool_t& pool)
{
    return pool_new<GRU_operator>(pool);
}

} // namespace nnr
