#include <cmath>
#include <cstdlib>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

template <typename T>
void fill_normal(T* py, size_t n, float mean, float scale) {
    using F = std::conditional_t<std::is_same_v<T, double>, double, float>;
    constexpr F one = 1;
    for (size_t i = 0; i < n; ++i) {
        F ty = (F)rand() / (RAND_MAX + one);
        F tx = (F)rand() / (RAND_MAX + one);
        py[i] = (T)(mean + scale * std::sqrt(F(-2) * std::log(tx)) * std::cos(F(2) * std::acos(-one) * ty));
    }
}

template <typename T>
void fill_uniform(T* py, size_t n, float high, float low) {
    for (size_t i = 0; i < n; ++i)
        py[i] = (T)(((float)rand() / (float)RAND_MAX) * (high - low) + low);
}

struct RandomNormal_op : operator_t {
    float mean, scale, seed;
    small_vector<int> shape;
    data_type_t dtype;

    bool init() override {
        if (outputs.size() != 1) return false;
        int64_t* ints;
        int nshape = attribute(attr_key_t::shape, ints);
        if (nshape <= 0) return false;
        dtype = (data_type_t)attribute(attr_key_t::dtype, NNR_DATA_TYPE_FLOAT32);
        mean = attribute(attr_key_t::mean, 0.0f);
        scale = attribute(attr_key_t::scale, 1.0f);
        seed = attribute(attr_key_t::seed, 0.0f);
        shape.resize(nshape);
        for (int i = 0; i < nshape; ++i) shape[i] = (int)ints[i];
        return true;
    }
    bool reshape() override { return outputs[0]->reshape(shape, dtype); }
    template <typename T> bool exec() { fill_normal((T*)outputs[0]->data, outputs[0]->ndata, mean, scale); return true; }
    bool exec() override {
        if (seed != 0.0f) srand((unsigned int)seed);
        return typed_exec<RandomNormal_op, float16_t, float, double>(this, outputs[0]->type);
    }
};

struct RandomNormalLike_op : operator_t {
    float mean, scale, seed;
    data_type_t dtype;

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        dtype = (data_type_t)attribute(attr_key_t::dtype, NNR_DATA_TYPE_UNDEFINED);
        mean = attribute(attr_key_t::mean, 0.0f);
        scale = attribute(attr_key_t::scale, 1.0f);
        seed = attribute(attr_key_t::seed, 0.0f);
        return true;
    }
    bool reshape() override {
        data_type_t type = (dtype != NNR_DATA_TYPE_UNDEFINED) ? dtype : inputs[0]->type;
        if (type != NNR_DATA_TYPE_FLOAT16 && type != NNR_DATA_TYPE_FLOAT32 && type != NNR_DATA_TYPE_FLOAT64) return false;
        return outputs[0]->reshape(inputs[0]->dims, type);
    }
    template <typename T> bool exec() { fill_normal((T*)outputs[0]->data, outputs[0]->ndata, mean, scale); return true; }
    bool exec() override {
        if (seed != 0.0f) srand((unsigned int)seed);
        return typed_exec<RandomNormalLike_op, float16_t, float, double>(this, outputs[0]->type);
    }
};

struct RandomUniform_op : operator_t {
    float high, low, seed;
    small_vector<int> shape;
    data_type_t dtype;

    bool init() override {
        if (outputs.size() != 1) return false;
        int64_t* ints;
        int nshape = attribute(attr_key_t::shape, ints);
        if (nshape <= 0) return false;
        dtype = (data_type_t)attribute(attr_key_t::dtype, NNR_DATA_TYPE_FLOAT32);
        high = attribute(attr_key_t::high, 1.0f);
        low = attribute(attr_key_t::low, 0.0f);
        seed = attribute(attr_key_t::seed, 0.0f);
        shape.resize(nshape);
        for (int i = 0; i < nshape; ++i) shape[i] = (int)ints[i];
        return true;
    }
    bool reshape() override { return outputs[0]->reshape(shape, dtype); }
    template <typename T> bool exec() { fill_uniform((T*)outputs[0]->data, outputs[0]->ndata, high, low); return true; }
    bool exec() override {
        if (seed != 0.0f) srand((unsigned int)seed);
        return typed_exec<RandomUniform_op, float16_t, float, double>(this, outputs[0]->type);
    }
};

struct RandomUniformLike_op : operator_t {
    float high, low, seed;
    data_type_t dtype;

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        dtype = (data_type_t)attribute(attr_key_t::dtype, NNR_DATA_TYPE_UNDEFINED);
        high = attribute(attr_key_t::high, 1.0f);
        low = attribute(attr_key_t::low, 0.0f);
        seed = attribute(attr_key_t::seed, 0.0f);
        return true;
    }
    bool reshape() override {
        data_type_t type = (dtype != NNR_DATA_TYPE_UNDEFINED) ? dtype : inputs[0]->type;
        if (type != NNR_DATA_TYPE_FLOAT16 && type != NNR_DATA_TYPE_FLOAT32 && type != NNR_DATA_TYPE_FLOAT64) return false;
        return outputs[0]->reshape(inputs[0]->dims, type);
    }
    template <typename T> bool exec() { fill_uniform((T*)outputs[0]->data, outputs[0]->ndata, high, low); return true; }
    bool exec() override {
        if (seed != 0.0f) srand((unsigned int)seed);
        return typed_exec<RandomUniformLike_op, float16_t, float, double>(this, outputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op op=RandomNormal mt=no
operator_t* resolver_default_op_RandomNormal(int opset, pool_t& pool) { return pool_new<RandomNormal_op>(pool); }
// @nnr-meta-op op=RandomNormalLike mt=no
operator_t* resolver_default_op_RandomNormalLike(int opset, pool_t& pool) { return pool_new<RandomNormalLike_op>(pool); }
// @nnr-meta-op op=RandomUniform mt=no
operator_t* resolver_default_op_RandomUniform(int opset, pool_t& pool) { return pool_new<RandomUniform_op>(pool); }
// @nnr-meta-op op=RandomUniformLike mt=no
operator_t* resolver_default_op_RandomUniformLike(int opset, pool_t& pool) { return pool_new<RandomUniformLike_op>(pool); }

} // namespace nnr
