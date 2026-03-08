#include <string>
#include <vector>
#include <cstring>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct StringSplit_operator : public operator_t {
    std::string delimiter;
    int maxsplit;

    bool init() override {
        if (inputs.empty() || outputs.empty()) return false;
        delimiter = std::string(attribute(attr_key_t::delimiter, std::string_view("")));
        maxsplit = attribute(attr_key_t::maxsplit, (int32_t)-1);
        return true;
    }

    bool reshape() override {
        // Cannot know output shape without executing
        return true;
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0]; // split strings
        tensor_t* num_splits = outputs.size() > 1 ? outputs[1] : nullptr; // number of splits per element

        const std::string* px = (const std::string*)x->data;
        size_t N = x->ndata;

        // First pass: split all strings to find max number of parts
        std::vector<std::vector<std::string>> all_parts(N);
        int max_parts = 0;

        for (size_t i = 0; i < N; ++i) {
            const std::string& s = px[i];
            std::vector<std::string>& parts = all_parts[i];

            if (delimiter.empty()) {
                // Split on whitespace
                size_t pos = 0;
                int splits = 0;
                while (pos < s.size()) {
                    // Skip whitespace
                    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
                        pos++;
                    if (pos >= s.size()) break;
                    if (maxsplit >= 0 && splits >= maxsplit) {
                        parts.push_back(s.substr(pos));
                        break;
                    }
                    size_t end = pos;
                    while (end < s.size() && s[end] != ' ' && s[end] != '\t' && s[end] != '\n' && s[end] != '\r')
                        end++;
                    parts.push_back(s.substr(pos, end - pos));
                    pos = end;
                    splits++;
                }
            } else {
                // Split on delimiter
                size_t pos = 0;
                int splits = 0;
                while (pos <= s.size()) {
                    if (maxsplit >= 0 && splits >= maxsplit) {
                        parts.push_back(s.substr(pos));
                        break;
                    }
                    size_t found = s.find(delimiter, pos);
                    if (found == std::string::npos) {
                        parts.push_back(s.substr(pos));
                        break;
                    }
                    parts.push_back(s.substr(pos, found - pos));
                    pos = found + delimiter.size();
                    splits++;
                }
            }
            if ((int)parts.size() > max_parts)
                max_parts = (int)parts.size();
        }

        // Output Y: input_shape + [max_parts]
        {
            small_vector<int> dims(x->ndim + 1);
            for (int d = 0; d < x->ndim; ++d)
                dims[d] = x->dims[d];
            dims[x->ndim] = max_parts;
            y->reshape(dims, NNR_DATA_TYPE_STRING);
            std::string* py = (std::string*)y->data;
            for (size_t i = 0; i < N; ++i) {
                for (int j = 0; j < max_parts; ++j) {
                    size_t idx = i * max_parts + j;
                    if (j < (int)all_parts[i].size())
                        py[idx] = all_parts[i][j];
                    else
                        py[idx] = "";
                }
            }
        }

        // Output num_splits
        if (num_splits) {
            small_vector<int> dims(x->ndim);
            for (int d = 0; d < x->ndim; ++d)
                dims[d] = x->dims[d];
            num_splits->reshape(dims, NNR_DATA_TYPE_INT64);
            int64_t* pn = (int64_t*)num_splits->data;
            for (size_t i = 0; i < N; ++i)
                pn[i] = (int64_t)all_parts[i].size();
        }

        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_StringSplit(int opset, pool_t& pool) { return pool_new<StringSplit_operator>(pool); }

} // namespace nnr
