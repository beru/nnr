#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct Constant_operator : public operator_t {
    bool init() override {
        if (outputs.size() != 1 || attrs.size() != 1)
            return false;
        const auto& [aname, a] = attrs[0];
        tensor_t* y = outputs[0];
        if (aname == attr_key_t::value) {
            return attribute(attr_key_t::value, y);
        }
        if (aname == attr_key_t::value_float) {
            if (!y->reinit(NNR_DATA_TYPE_FLOAT32, {})) return false;
            y->apply(&a.f, sizeof(float));
            return true;
        }
        if (aname == attr_key_t::value_int) {
            if (!y->reinit(NNR_DATA_TYPE_INT64, {})) return false;
            y->apply(&a.i, sizeof(int64_t));
            return true;
        }
        if (aname == attr_key_t::value_floats) {
            if (a.floats.empty()) return false;
            int dims[] = { (int)a.floats.size() };
            if (!y->reinit(NNR_DATA_TYPE_FLOAT32, dims)) return false;
            y->apply(a.floats.data(), a.floats.size() * sizeof(float));
            return true;
        }
        if (aname == attr_key_t::value_ints) {
            if (a.ints.empty()) return false;
            int dims[] = { (int)a.ints.size() };
            if (!y->reinit(NNR_DATA_TYPE_INT64, dims)) return false;
            y->apply(a.ints.data(), a.ints.size() * sizeof(int64_t));
            return true;
        }
        if (aname == attr_key_t::value_string) {
            int dims[] = { 1 };
            if (!y->reinit(NNR_DATA_TYPE_STRING, dims)) return false;
            if (y->data) ((std::string*)y->data)[0] = a.s;
            return true;
        }
        return false;
    }

    bool reshape() override { return true; }
    bool exec() override { return true; }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Constant(int opset, pool_t& pool) { return pool_new<Constant_operator>(pool); }

} // namespace nnr
