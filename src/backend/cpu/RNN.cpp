#include "nnr.h"
#include "util.h"
#include "arena.h"

#include <cmath>
#include <cstring>

namespace nnr {

namespace {

struct RNN_operator : public operator_t {
    int hidden_size_ = 0;
    int layout_ = 0;
    int num_directions_ = 1;
    std::string_view direction_;

    bool init() override {
        if (inputs.size() < 3 || outputs.size() < 1) return false;
        hidden_size_ = (int)attribute(attr_key_t::hidden_size, (int64_t)0);
        layout_ = (int)attribute(attr_key_t::layout, (int64_t)0);
        direction_ = attribute(attr_key_t::direction, "forward");
        num_directions_ = (direction_ == "bidirectional") ? 2 : 1;
        return true;
    }

    bool reshape() override {
        const tensor_t* X = inputs[0];
        const tensor_t* W = inputs[1];

        if (hidden_size_ == 0) hidden_size_ = W->dims[1];

        int seq_length, batch_size;
        if (layout_ == 0) {
            seq_length = X->dims[0]; batch_size = X->dims[1];
        } else {
            batch_size = X->dims[0]; seq_length = X->dims[1];
        }

        if (outputs.size() > 0 && outputs[0]) {
            if (layout_ == 0) {
                int dims[] = { seq_length, num_directions_, batch_size, hidden_size_ };
                if (!outputs[0]->reshape(dims, X->type)) return false;
            } else {
                int dims[] = { batch_size, seq_length, num_directions_, hidden_size_ };
                if (!outputs[0]->reshape(dims, X->type)) return false;
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
        return true;
    }

    template <typename T>
    bool exec() {
        const tensor_t* X = inputs[0];
        const tensor_t* W = inputs[1];
        const tensor_t* R = inputs[2];
        const tensor_t* B = (inputs.size() > 3 && inputs[3] && inputs[3]->ndata > 0) ? inputs[3] : nullptr;
        const tensor_t* seq_lens_t = (inputs.size() > 4 && inputs[4] && inputs[4]->ndata > 0) ? inputs[4] : nullptr;
        const tensor_t* init_h_t = (inputs.size() > 5 && inputs[5] && inputs[5]->ndata > 0) ? inputs[5] : nullptr;

        int seq_length, batch_size, input_size;
        if (layout_ == 0) {
            seq_length = X->dims[0]; batch_size = X->dims[1]; input_size = X->dims[2];
        } else {
            batch_size = X->dims[0]; seq_length = X->dims[1]; input_size = X->dims[2];
        }

        const int HS = hidden_size_;
        const T* px = (const T*)X->data;
        const T* pw = (const T*)W->data;
        const T* pr = (const T*)R->data;
        const T* pb = B ? (const T*)B->data : nullptr;

        tensor_t* Y_out = (outputs.size() > 0) ? outputs[0] : nullptr;
        tensor_t* Yh_out = (outputs.size() > 1) ? outputs[1] : nullptr;
        T* py = Y_out ? (T*)Y_out->data : nullptr;
        T* pyh = Yh_out ? (T*)Yh_out->data : nullptr;

        arena_scope_t scope(ctx->arena);
        T* H    = scope.alloc_arr<T>(batch_size * HS);
        T* Hnew = scope.alloc_arr<T>(HS);
        std::fill(H, H + batch_size * HS, T(0));

        for (int dir = 0; dir < num_directions_; ++dir) {
            bool is_reverse = (direction_ == "reverse") || (direction_ == "bidirectional" && dir == 1);

            const T* Wi = pw + dir * HS * input_size;
            const T* Ri = pr + dir * HS * HS;

            const T* Wb = nullptr, *Rb = nullptr;
            if (pb) {
                Wb = pb + dir * 2 * HS;
                Rb = Wb + HS;
            }

            std::fill(H, H + batch_size * HS, T(0));
            if (init_h_t) {
                const T* p_init = (const T*)init_h_t->data + dir * batch_size * HS;
                std::memcpy(H, p_init, batch_size * HS * sizeof(T));
            }

            for (int ti = 0; ti < seq_length; ++ti) {
                int t = is_reverse ? (seq_length - 1 - ti) : ti;

                for (int b = 0; b < batch_size; ++b) {
                    if (seq_lens_t && t >= ((const int*)seq_lens_t->data)[b]) {
                        if (py) {
                            T* dst;
                            if (layout_ == 0)
                                dst = py + ((size_t)t * num_directions_ * batch_size + dir * batch_size + b) * HS;
                            else
                                dst = py + ((size_t)b * seq_length * num_directions_ + t * num_directions_ + dir) * HS;
                            std::memset(dst, 0, HS * sizeof(T));
                        }
                        continue;
                    }

                    const T* xt;
                    if (layout_ == 0)
                        xt = px + ((size_t)t * batch_size + b) * input_size;
                    else
                        xt = px + ((size_t)b * seq_length + t) * input_size;

                    T* Hb = H + b * HS;

                    for (int h = 0; h < HS; ++h) {
                        T val = T(0);
                        for (int k = 0; k < input_size; ++k)
                            val += xt[k] * Wi[h * input_size + k];
                        for (int k = 0; k < HS; ++k)
                            val += Hb[k] * Ri[h * HS + k];
                        if (Wb) val += Wb[h];
                        if (Rb) val += Rb[h];
                        Hnew[h] = (T)std::tanh((float)val);
                    }
                    std::memcpy(Hb, Hnew, HS * sizeof(T));

                    if (py) {
                        T* dst;
                        if (layout_ == 0)
                            dst = py + ((size_t)t * num_directions_ * batch_size + dir * batch_size + b) * HS;
                        else
                            dst = py + ((size_t)b * seq_length * num_directions_ + t * num_directions_ + dir) * HS;
                        std::memcpy(dst, Hb, HS * sizeof(T));
                    }
                }
            }

            if (pyh) {
                if (layout_ == 0) {
                    std::memcpy(pyh + dir * batch_size * HS, H, batch_size * HS * sizeof(T));
                } else {
                    for (int b = 0; b < batch_size; ++b)
                        std::memcpy(pyh + (b * num_directions_ + dir) * HS, H + b * HS, HS * sizeof(T));
                }
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<RNN_operator, float, double>(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no workspace=yes
operator_t* resolver_default_op_RNN(int opset, pool_t& pool) { return pool_new<RNN_operator>(pool); }

} // namespace nnr
