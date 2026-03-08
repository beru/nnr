#include <cmath>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

static constexpr double PI = 3.14159265358979323846;

struct window_base : operator_t {
    int periodic;
    data_type_t output_datatype;

    bool init() override {
        if (inputs.size() != 1 || outputs.empty()) return false;
        periodic = attribute(attr_key_t::periodic, (int32_t)1);
        output_datatype = (data_type_t)attribute(attr_key_t::output_datatype, (int32_t)NNR_DATA_TYPE_FLOAT32);
        return true;
    }

    bool reshape() override {
        const tensor_t* size_tensor = inputs[0];
        int64_t N = 0;
        if (size_tensor->type == NNR_DATA_TYPE_INT64)
            N = *(const int64_t*)size_tensor->data;
        else if (size_tensor->type == NNR_DATA_TYPE_INT32)
            N = *(const int32_t*)size_tensor->data;
        small_vector<int> dims(1);
        dims[0] = (int)N;
        return outputs[0]->reshape(dims, output_datatype);
    }
};

#define NNR_WINDOW_OP(Name, formula) \
struct Name##_op : window_base { \
    template <typename T> bool exec() { \
        T* py = (T*)outputs[0]->data; \
        int N = outputs[0]->dims[0]; \
        double denom = periodic ? (double)N : (double)(N - 1); \
        if (denom == 0) denom = 1; \
        for (int i = 0; i < N; ++i) py[i] = (T)(formula); \
        return true; \
    } \
    bool exec() override { return typed_exec<Name##_op, float16_t, float, double>(this, outputs[0]->type); } \
};

NNR_WINDOW_OP(HammingWindow,  25.0/46.0 - 21.0/46.0 * std::cos(2.0 * PI * i / denom))
NNR_WINDOW_OP(HannWindow,     0.5 - 0.5 * std::cos(2.0 * PI * i / denom))
NNR_WINDOW_OP(BlackmanWindow, 0.42 - 0.5 * std::cos(2.0 * PI * i / denom) + 0.08 * std::cos(4.0 * PI * i / denom))

#undef NNR_WINDOW_OP

} // namespace

// @nnr-meta-op op=HammingWindow mt=no
operator_t* resolver_default_op_HammingWindow(int opset, pool_t& pool) { return pool_new<HammingWindow_op>(pool); }
// @nnr-meta-op op=HannWindow mt=no
operator_t* resolver_default_op_HannWindow(int opset, pool_t& pool) { return pool_new<HannWindow_op>(pool); }
// @nnr-meta-op op=BlackmanWindow mt=no
operator_t* resolver_default_op_BlackmanWindow(int opset, pool_t& pool) { return pool_new<BlackmanWindow_op>(pool); }

} // namespace nnr
