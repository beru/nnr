#include "nnr.h"
#include "util.h"
#include "arena.h"

#include <cmath>
#include <cstring>

namespace nnr {

namespace {

struct LSTM_operator : public operator_t {
    int hidden_size_ = 0;
    int input_forget_ = 0;
    int layout_ = 0;
    float clip_ = 0.0f;
    int num_directions_ = 1;
    bool is_reverse_ = false;
    bool is_bidirectional_ = false;

    bool init() override {
        if (inputs.size() < 3 || outputs.size() < 1) {
            return false;
        }
        hidden_size_ = (int)attribute(attr_key_t::hidden_size, (int64_t)0);
        input_forget_ = (int)attribute(attr_key_t::input_forget, (int64_t)0);
        layout_ = (int)attribute(attr_key_t::layout, (int64_t)0);
        clip_ = attribute(attr_key_t::clip, 0.0f);

        auto dir = attribute(attr_key_t::direction, "forward");
        is_reverse_ = (dir == "reverse");
        is_bidirectional_ = (dir == "bidirectional");
        num_directions_ = is_bidirectional_ ? 2 : 1;

        return true;
    }

    bool reshape() override {
        const tensor_t* X = inputs[0];
        tensor_t* Y = outputs[0];

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

        if (Y) {
            if (layout_ == 0) {
                int dims[] = { seq_length, num_directions_, batch_size, hidden_size_ };
                if (!Y->reshape(dims, X->type)) return false;
            } else {
                int dims[] = { batch_size, seq_length, num_directions_, hidden_size_ };
                if (!Y->reshape(dims, X->type)) return false;
            }
        }

        if (outputs.size() > 1 && outputs[1]) {
            if (layout_ == 0) {
                int dims[] = { num_directions_, batch_size, hidden_size_ };
                if (!outputs[1]->reshape(dims, X->type)) return false;
            } else {
                int dims[] = { batch_size, num_directions_, hidden_size_ };
                if (!outputs[1]->reshape(dims, X->type)) return false;
            }
        }

        if (outputs.size() > 2 && outputs[2]) {
            if (layout_ == 0) {
                int dims[] = { num_directions_, batch_size, hidden_size_ };
                if (!outputs[2]->reshape(dims, X->type)) return false;
            } else {
                int dims[] = { batch_size, num_directions_, hidden_size_ };
                if (!outputs[2]->reshape(dims, X->type)) return false;
            }
        }

        return true;
    }

    template <typename T>
    static T sigmoid(T x) {
        return (T)1 / ((T)1 + std::exp(-x));
    }

    template <typename T>
    static T clip_val(T x, float clip) {
        if (clip > 0.0f) {
            if (x > (T)clip) return (T)clip;
            if (x < (T)(-clip)) return (T)(-clip);
        }
        return x;
    }

    template <typename T>
    void lstm_one_direction(
        const T* X_data, int seq_length, int batch_size, int input_size,
        const T* W, const T* R, const T* B, const T* P,
        const T* init_h, const T* init_c,
        const int* seq_lens,
        T* Y_out, T* Y_h_out, T* Y_c_out,
        bool reverse, int dir_idx)
    {
        const int HS = hidden_size_;

        // Allocate working buffers for H and C: [batch, hidden]
        arena_scope_t scope(ctx->arena);
        T* H     = scope.alloc_arr<T>(batch_size * HS);
        T* C     = scope.alloc_arr<T>(batch_size * HS);
        T* gates = scope.alloc_arr<T>(batch_size * 4 * HS);
        memset(H, 0, batch_size * HS * sizeof(T));
        memset(C, 0, batch_size * HS * sizeof(T));

        // Initialize from initial_h, initial_c if provided
        if (init_h) {
            // init_h shape: [num_directions, batch, hidden] (layout=0) or [batch, num_directions, hidden] (layout=1)
            for (int b = 0; b < batch_size; ++b) {
                const T* src;
                if (layout_ == 0) {
                    src = init_h + dir_idx * batch_size * HS + b * HS;
                } else {
                    src = init_h + b * num_directions_ * HS + dir_idx * HS;
                }
                std::memcpy(H + b * HS, src, HS * sizeof(T));
            }
        }
        if (init_c) {
            for (int b = 0; b < batch_size; ++b) {
                const T* src;
                if (layout_ == 0) {
                    src = init_c + dir_idx * batch_size * HS + b * HS;
                } else {
                    src = init_c + b * num_directions_ * HS + dir_idx * HS;
                }
                std::memcpy(C + b * HS, src, HS * sizeof(T));
            }
        }

        // W: [4*HS, input_size] — gate order: IOFC
        // R: [4*HS, HS]
        // B: [8*HS] — Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c
        // P: [3*HS] — Pi, Po, Pf

        // Bias pointers
        const T* Wb_i = B ? B : nullptr;
        const T* Wb_o = B ? B + HS : nullptr;
        const T* Wb_f = B ? B + 2 * HS : nullptr;
        const T* Wb_c = B ? B + 3 * HS : nullptr;
        const T* Rb_i = B ? B + 4 * HS : nullptr;
        const T* Rb_o = B ? B + 5 * HS : nullptr;
        const T* Rb_f = B ? B + 6 * HS : nullptr;
        const T* Rb_c = B ? B + 7 * HS : nullptr;

        // Peephole pointers
        const T* Pi = P ? P : nullptr;
        const T* Po = P ? P + HS : nullptr;
        const T* Pf = P ? P + 2 * HS : nullptr;

        // W row pointers for each gate
        const T* Wi = W;               // [HS, input_size]
        const T* Wo = W + HS * input_size;
        const T* Wf = W + 2 * HS * input_size;
        const T* Wc = W + 3 * HS * input_size;

        const T* Ri = R;               // [HS, HS]
        const T* Ro = R + HS * HS;
        const T* Rf = R + 2 * HS * HS;
        const T* Rc = R + 3 * HS * HS;

        for (int t_step = 0; t_step < seq_length; ++t_step) {
            int t = reverse ? (seq_length - 1 - t_step) : t_step;

            for (int b = 0; b < batch_size; ++b) {
                // Check sequence length
                if (seq_lens && t >= seq_lens[b]) {
                    // Past end of sequence; output zeros, keep state unchanged
                    if (Y_out) {
                        T* y_ptr;
                        if (layout_ == 0) {
                            y_ptr = Y_out + t * num_directions_ * batch_size * HS + dir_idx * batch_size * HS + b * HS;
                        } else {
                            y_ptr = Y_out + b * seq_length * num_directions_ * HS + t * num_directions_ * HS + dir_idx * HS;
                        }
                        std::memset(y_ptr, 0, HS * sizeof(T));
                    }
                    continue;
                }

                // Get input pointer for this timestep and batch
                const T* x_t;
                if (layout_ == 0) {
                    x_t = X_data + t * batch_size * input_size + b * input_size;
                } else {
                    x_t = X_data + b * seq_length * input_size + t * input_size;
                }

                T* h_b = H + b * HS;
                T* c_b = C + b * HS;

                for (int h = 0; h < HS; ++h) {
                    // Xi = x_t * Wi^T, etc.
                    T xi = (T)0, xo = (T)0, xf = (T)0, xc = (T)0;
                    for (int k = 0; k < input_size; ++k) {
                        xi += x_t[k] * Wi[h * input_size + k];
                        xo += x_t[k] * Wo[h * input_size + k];
                        xf += x_t[k] * Wf[h * input_size + k];
                        xc += x_t[k] * Wc[h * input_size + k];
                    }

                    // Ri = h_prev * Ri^T
                    T ri = (T)0, ro = (T)0, rf = (T)0, rc = (T)0;
                    for (int k = 0; k < HS; ++k) {
                        ri += h_b[k] * Ri[h * HS + k];
                        ro += h_b[k] * Ro[h * HS + k];
                        rf += h_b[k] * Rf[h * HS + k];
                        rc += h_b[k] * Rc[h * HS + k];
                    }

                    // Accumulate bias
                    T bi = (T)0, bo = (T)0, bf = (T)0, bc = (T)0;
                    if (B) {
                        bi = Wb_i[h] + Rb_i[h];
                        bo = Wb_o[h] + Rb_o[h];
                        bf = Wb_f[h] + Rb_f[h];
                        bc = Wb_c[h] + Rb_c[h];
                    }

                    // it = sigmoid(Xi + Ri + Pi*Ct-1 + bi)
                    T it_val = xi + ri + bi;
                    if (P) it_val += Pi[h] * c_b[h];
                    it_val = sigmoid(clip_val(it_val, clip_));

                    // ft = sigmoid(Xf + Rf + Pf*Ct-1 + bf)
                    T ft_val = xf + rf + bf;
                    if (P) ft_val += Pf[h] * c_b[h];
                    ft_val = sigmoid(clip_val(ft_val, clip_));

                    // input_forget: couple input and forget gates
                    if (input_forget_) {
                        ft_val = (T)1 - it_val;
                    }

                    // ct = tanh(Xc + Rc + bc)
                    T ct_val = std::tanh(clip_val(xc + rc + bc, clip_));

                    // Ct = ft * Ct-1 + it * ct
                    T Ct_new = ft_val * c_b[h] + it_val * ct_val;

                    // ot = sigmoid(Xo + Ro + Po*Ct + bo)
                    T ot_val = xo + ro + bo;
                    if (P) ot_val += Po[h] * Ct_new;
                    ot_val = sigmoid(clip_val(ot_val, clip_));

                    // Ht = ot * tanh(Ct)
                    T Ht_new = ot_val * std::tanh(Ct_new);

                    // Store to temp gates buffer (we need to write all at once since h_b is read in inner loop)
                    gates[b * 4 * HS + h] = Ht_new;           // new H
                    gates[b * 4 * HS + HS + h] = Ct_new;       // new C
                }
            }

            // Update H and C from gates buffer, write to Y
            for (int b = 0; b < batch_size; ++b) {
                if (seq_lens && t >= seq_lens[b]) continue;

                T* h_b = H + b * HS;
                T* c_b = C + b * HS;

                for (int h = 0; h < HS; ++h) {
                    h_b[h] = gates[b * 4 * HS + h];
                    c_b[h] = gates[b * 4 * HS + HS + h];
                }

                // Write to Y output
                if (Y_out) {
                    T* y_ptr;
                    if (layout_ == 0) {
                        y_ptr = Y_out + t * num_directions_ * batch_size * HS + dir_idx * batch_size * HS + b * HS;
                    } else {
                        y_ptr = Y_out + b * seq_length * num_directions_ * HS + t * num_directions_ * HS + dir_idx * HS;
                    }
                    std::memcpy(y_ptr, h_b, HS * sizeof(T));
                }
            }
        }

        // Write final H and C to Y_h and Y_c
        if (Y_h_out) {
            for (int b = 0; b < batch_size; ++b) {
                T* dst;
                if (layout_ == 0) {
                    dst = Y_h_out + dir_idx * batch_size * HS + b * HS;
                } else {
                    dst = Y_h_out + b * num_directions_ * HS + dir_idx * HS;
                }
                std::memcpy(dst, H + b * HS, HS * sizeof(T));
            }
        }
        if (Y_c_out) {
            for (int b = 0; b < batch_size; ++b) {
                T* dst;
                if (layout_ == 0) {
                    dst = Y_c_out + dir_idx * batch_size * HS + b * HS;
                } else {
                    dst = Y_c_out + b * num_directions_ * HS + dir_idx * HS;
                }
                std::memcpy(dst, C + b * HS, HS * sizeof(T));
            }
        }
    }

    template <typename T>
    bool exec() {
        const tensor_t* X = inputs[0];
        const tensor_t* W_t = inputs[1];
        const tensor_t* R_t = inputs[2];

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

        const T* X_data = (const T*)X->data;
        const T* W_data = (const T*)W_t->data;
        const T* R_data = (const T*)R_t->data;

        // Optional inputs
        const T* B_data = nullptr;
        if (inputs.size() > 3 && inputs[3] && inputs[3]->ndata > 0) {
            B_data = (const T*)inputs[3]->data;
        }

        const int* seq_lens = nullptr;
        if (inputs.size() > 4 && inputs[4] && inputs[4]->ndata > 0) {
            seq_lens = (const int*)inputs[4]->data;
        }

        const T* init_h = nullptr;
        if (inputs.size() > 5 && inputs[5] && inputs[5]->ndata > 0) {
            init_h = (const T*)inputs[5]->data;
        }

        const T* init_c = nullptr;
        if (inputs.size() > 6 && inputs[6] && inputs[6]->ndata > 0) {
            init_c = (const T*)inputs[6]->data;
        }

        const T* P_data = nullptr;
        if (inputs.size() > 7 && inputs[7] && inputs[7]->ndata > 0) {
            P_data = (const T*)inputs[7]->data;
        }

        tensor_t* Y = outputs[0];
        tensor_t* Y_h = (outputs.size() > 1) ? outputs[1] : nullptr;
        tensor_t* Y_c = (outputs.size() > 2) ? outputs[2] : nullptr;

        T* Y_data = Y ? (T*)Y->data : nullptr;
        T* Y_h_data = (Y_h && Y_h->ndata > 0) ? (T*)Y_h->data : nullptr;
        T* Y_c_data = (Y_c && Y_c->ndata > 0) ? (T*)Y_c->data : nullptr;

        const int HS = hidden_size_;

        // W shape: [num_directions, 4*HS, input_size]
        // R shape: [num_directions, 4*HS, HS]
        // B shape: [num_directions, 8*HS]
        // P shape: [num_directions, 3*HS]

        int W_dir_stride = 4 * HS * input_size;
        int R_dir_stride = 4 * HS * HS;
        int B_dir_stride = 8 * HS;
        int P_dir_stride = 3 * HS;

        // Forward direction (or single direction for "forward"/"reverse")
        {
            int dir_idx = 0;
            bool reverse = is_reverse_;
            const T* W_dir = W_data + dir_idx * W_dir_stride;
            const T* R_dir = R_data + dir_idx * R_dir_stride;
            const T* B_dir = B_data ? B_data + dir_idx * B_dir_stride : nullptr;
            const T* P_dir = P_data ? P_data + dir_idx * P_dir_stride : nullptr;

            lstm_one_direction<T>(
                X_data, seq_length, batch_size, input_size,
                W_dir, R_dir, B_dir, P_dir,
                init_h, init_c, seq_lens,
                Y_data, Y_h_data, Y_c_data,
                reverse, dir_idx);
        }

        // Backward direction (for bidirectional)
        if (is_bidirectional_) {
            int dir_idx = 1;
            const T* W_dir = W_data + dir_idx * W_dir_stride;
            const T* R_dir = R_data + dir_idx * R_dir_stride;
            const T* B_dir = B_data ? B_data + dir_idx * B_dir_stride : nullptr;
            const T* P_dir = P_data ? P_data + dir_idx * P_dir_stride : nullptr;

            lstm_one_direction<T>(
                X_data, seq_length, batch_size, input_size,
                W_dir, R_dir, B_dir, P_dir,
                init_h, init_c, seq_lens,
                Y_data, Y_h_data, Y_c_data,
                true, dir_idx);
        }

        return true;
    }

    bool exec() override {
        return typed_exec<LSTM_operator, float, double>(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no workspace=yes
operator_t* resolver_default_op_LSTM(int opset, pool_t& pool)
{
    return pool_new<LSTM_operator>(pool);
}

} // namespace nnr
