#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include "nnr.h"
#include "util.h"

namespace nnr {

namespace {

struct StringNormalizer_operator : public operator_t {
    std::string case_change; // "NONE", "LOWER", "UPPER"
    int is_case_sensitive;
    std::span<const std::string_view> stopwords;

    bool init() override {
        if (inputs.empty() || outputs.empty()) return false;
        case_change = std::string(attribute(attr_key_t::case_change_action, std::string_view("NONE")));
        is_case_sensitive = attribute(attr_key_t::is_case_sensitive, (int32_t)0);

        // Read stopwords from attribute
        if (attr_t* a = find_attr("stopwords"))
            stopwords = a->strings;
        return true;
    }

    bool reshape() override {
        // Output shape may differ if stopwords remove elements
        return true;
    }

    bool is_stopword(std::string_view s) {
        if (is_case_sensitive) {
            return std::find(stopwords.begin(), stopwords.end(), s) != stopwords.end();
        } else {
            std::string lower(s);
            for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
            for (auto& sw : stopwords) {
                std::string sw_lower(sw);
                for (auto& c : sw_lower) c = (char)std::tolower((unsigned char)c);
                if (lower == sw_lower) return true;
            }
            return false;
        }
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        const std::string* px = (const std::string*)x->data;

        if (x->ndim == 1) {
            // 1D: filter stopwords, then apply case change
            std::vector<std::string> result;
            for (size_t i = 0; i < x->ndata; ++i) {
                if (!stopwords.empty() && is_stopword(px[i])) continue;
                result.push_back(px[i]);
            }

            // Apply case change
            for (auto& s : result) {
                if (case_change == "LOWER") {
                    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
                } else if (case_change == "UPPER") {
                    for (auto& c : s) c = (char)std::toupper((unsigned char)c);
                }
            }

            if (result.empty()) {
                // Output empty string
                small_vector<int> dims(1);
                dims[0] = 1;
                y->reshape(dims, NNR_DATA_TYPE_STRING);
                std::string* py = (std::string*)y->data;
                py[0] = "";
            } else {
                small_vector<int> dims(1);
                dims[0] = (int)result.size();
                y->reshape(dims, NNR_DATA_TYPE_STRING);
                std::string* py = (std::string*)y->data;
                for (size_t i = 0; i < result.size(); ++i)
                    py[i] = result[i];
            }
        } else if (x->ndim == 2) {
            // 2D: process each row independently
            int rows = x->dims[0], cols = x->dims[1];
            std::vector<std::vector<std::string>> all_results(rows);
            int max_cols = 0;

            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const std::string& s = px[r * cols + c];
                    if (!stopwords.empty() && is_stopword(s)) continue;
                    std::string out = s;
                    if (case_change == "LOWER") {
                        for (auto& ch : out) ch = (char)std::tolower((unsigned char)ch);
                    } else if (case_change == "UPPER") {
                        for (auto& ch : out) ch = (char)std::toupper((unsigned char)ch);
                    }
                    all_results[r].push_back(out);
                }
                if ((int)all_results[r].size() > max_cols)
                    max_cols = (int)all_results[r].size();
            }

            if (max_cols == 0) max_cols = 1;
            small_vector<int> dims(2);
            dims[0] = rows;
            dims[1] = max_cols;
            y->reshape(dims, NNR_DATA_TYPE_STRING);
            std::string* py = (std::string*)y->data;
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < max_cols; ++c) {
                    if (c < (int)all_results[r].size())
                        py[r * max_cols + c] = all_results[r][c];
                    else
                        py[r * max_cols + c] = "";
                }
            }
        } else {
            y->reshape_identity(x);
            std::string* py = (std::string*)y->data;
            for (size_t i = 0; i < x->ndata; ++i) {
                py[i] = px[i];
                if (case_change == "LOWER")
                    for (auto& c : py[i]) c = (char)std::tolower((unsigned char)c);
                else if (case_change == "UPPER")
                    for (auto& c : py[i]) c = (char)std::toupper((unsigned char)c);
            }
        }

        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_StringNormalizer(int opset, pool_t& pool) { return pool_new<StringNormalizer_operator>(pool); }

} // namespace nnr
