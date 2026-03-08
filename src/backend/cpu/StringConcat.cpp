#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct StringConcat_operator : public operator_t {
    bool init() override {
        return is_inout_size(2, 1);
    }

    bool reshape() override {
        return outputs[0]->reshape_multi_broadcast(inputs[0], inputs[1], NNR_DATA_TYPE_STRING);
    }

    bool exec() override {
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        tensor_t* y = outputs[0];
        const std::string* pa = (const std::string*)a->data;
        const std::string* pb = (const std::string*)b->data;
        std::string* py = (std::string*)y->data;

        for (size_t i = 0; i < y->ndata; ++i) {
            const std::string* va = (const std::string*)a->broadcast_map_address(y, (int)i);
            const std::string* vb = (const std::string*)b->broadcast_map_address(y, (int)i);
            py[i] = *va + *vb;
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_StringConcat(int opset, pool_t& pool) { return pool_new<StringConcat_operator>(pool); }

} // namespace nnr
