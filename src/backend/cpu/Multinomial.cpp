#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Multinomial_operator : public operator_t {
    data_type_t dtype;
    int sample_size;
    float seed;

    bool init() override {
        if (!is_inout_size(1, 1)) {
            return false;
        }
        dtype = (data_type_t)attribute(attr_key_t::dtype, NNR_DATA_TYPE_INT32);
        sample_size = attribute(attr_key_t::sample_size, 1);
        seed = attribute(attr_key_t::seed, 0.0f);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        return y->reshape_identity(x, dtype);
    }

    template <typename XT, typename YT>
    bool exec() {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const int bsz = x->dims[0];
        const int csz = x->dims[1];
        const XT* px = (const XT*)x->data;
        arena_scope_t scope(ctx->arena);
        XT* cum = scope.alloc_arr<XT>(csz);

        if (seed != 0.0) {
            srand((unsigned int)seed);
        }

        YT* py = (YT*)y->data;
        for (int i = 0; i < bsz; ++i) {
            for (int j = 0; j < sample_size; ++j) {
                cum[0] = px[i * csz];
                for (int k = 1; k < csz; ++k) {
                    cum[k] = cum[k - 1] + px[i * csz + k];
                }
                int l = csz - 1;
                for (int k = 0; k < csz - 1; ++k) {
                    if ((XT)rand() / (XT)(RAND_MAX) < cum[k]) {
                        l = k;
                        break;
                    }
                }
                int o = i * csz + l;
                py[o]++;
            }
        }
        return true;
    }

    template <typename XT>
    bool exec() {
        tensor_t* y = outputs[0];
        switch (y->type) {
        case NNR_DATA_TYPE_INT32:
            return exec<XT, int32_t>();
        case NNR_DATA_TYPE_INT64:
            return exec<XT, int64_t>();
        default:
            return false;
        }
    }

    bool exec() override {
        return typed_exec<Multinomial_operator,
            opset_t<7, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }

};

} // namespace {

// @nnr-meta-op mt=no workspace=yes
operator_t* resolver_default_op_Multinomial(int opset, pool_t& pool) { return pool_new<Multinomial_operator>(pool); }

} // namespace nnr
