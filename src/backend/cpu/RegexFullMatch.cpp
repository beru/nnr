#include <regex>
#include <string>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct RegexFullMatch_operator : public operator_t {
    std::string pattern_str;

    bool init() override {
        if (inputs.empty() || outputs.empty()) return false;
        pattern_str = std::string(attribute(attr_key_t::pattern, std::string_view("")));
        return true;
    }

    bool reshape() override {
        return outputs[0]->reshape_identity(inputs[0], NNR_DATA_TYPE_BOOL);
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const std::string* px = (const std::string*)x->data;
        bool* py = (bool*)y->data;

        std::regex re(pattern_str);

        for (size_t i = 0; i < x->ndata; ++i) {
            py[i] = std::regex_match(px[i], re);
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_RegexFullMatch(int opset, pool_t& pool) { return pool_new<RegexFullMatch_operator>(pool); }

} // namespace nnr
