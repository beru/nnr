#include "nnr.h"
#include "util.h"
#include <vector>
#include <string>
#include <cstring>

namespace nnr {

namespace {

struct TfIdfVectorizer_operator : public operator_t {
    int max_gram_length;
    int max_skip_count;
    int min_gram_length;
    std::string mode_str;
    std::vector<int64_t> ngram_counts;
    std::vector<int64_t> ngram_indexes;
    std::vector<int64_t> pool_int64s;
    std::vector<float> weights;

    struct NgramEntry {
        std::vector<int64_t> values;
        int output_index;
    };
    std::vector<std::vector<NgramEntry>> ngrams_by_length;

    bool init() override {
        if (inputs.size() != 1 || outputs.size() != 1) return false;

        max_gram_length = attribute(attr_key_t::max_gram_length, 1);
        max_skip_count = attribute(attr_key_t::max_skip_count, 0);
        min_gram_length = attribute(attr_key_t::min_gram_length, 1);
        mode_str = std::string(attribute(attr_key_t::mode, "TF"));

        int64_t* counts_ptr;
        int n_counts = attribute(attr_key_t::ngram_counts, counts_ptr);
        ngram_counts.assign(counts_ptr, counts_ptr + n_counts);

        int64_t* indexes_ptr;
        int n_indexes = attribute(attr_key_t::ngram_indexes, indexes_ptr);
        ngram_indexes.assign(indexes_ptr, indexes_ptr + n_indexes);

        int64_t* pool_ptr;
        int n_pool = attribute(attr_key_t::pool_int64s, pool_ptr);
        if (n_pool > 0) {
            pool_int64s.assign(pool_ptr, pool_ptr + n_pool);
        }

        float* weights_ptr;
        int n_weights = attribute(attr_key_t::weights, weights_ptr);
        if (n_weights > 0) {
            weights.assign(weights_ptr, weights_ptr + n_weights);
        }

        ngrams_by_length.resize(max_gram_length);
        int ngram_idx = 0;

        for (int g = 1; g <= max_gram_length; ++g) {
            if (g - 1 >= (int)ngram_counts.size()) break;
            int pool_start = (int)ngram_counts[g - 1];
            int pool_end = (g < (int)ngram_counts.size()) ? (int)ngram_counts[g] : (int)pool_int64s.size();
            int num_ngrams = (pool_end - pool_start) / g;

            for (int n = 0; n < num_ngrams; ++n) {
                NgramEntry entry;
                entry.values.resize(g);
                for (int k = 0; k < g; ++k) {
                    entry.values[k] = pool_int64s[pool_start + n * g + k];
                }
                entry.output_index = (ngram_idx < (int)ngram_indexes.size()) ? (int)ngram_indexes[ngram_idx] : 0;
                ngrams_by_length[g - 1].push_back(entry);
                ngram_idx++;
            }
        }

        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];

        int out_size = 0;
        for (auto idx : ngram_indexes) {
            if ((int)idx + 1 > out_size) out_size = (int)idx + 1;
        }

        if (x->ndim == 1) {
            small_vector<int> dims = {out_size};
            return y->reshape(dims, NNR_DATA_TYPE_FLOAT32);
        } else if (x->ndim == 2) {
            small_vector<int> dims = {x->dims[0], out_size};
            return y->reshape(dims, NNR_DATA_TYPE_FLOAT32);
        }
        return false;
    }

    void collect_and_match(int seq_len, int pos, int remaining, int skip_used,
                           std::vector<int>& positions,
                           const int64_t* seq, const std::vector<NgramEntry>& entries,
                           float* out) {
        if (remaining == 0) {
            int g = (int)positions.size();
            for (const auto& entry : entries) {
                bool match = true;
                for (int k = 0; k < g; ++k) {
                    if (seq[positions[k]] != entry.values[k]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    int oi = entry.output_index;
                    if (mode_str == "IDF" && !weights.empty() && oi < (int)weights.size()) {
                        out[oi] += weights[oi];
                    } else {
                        out[oi] += 1.0f;
                    }
                }
            }
            return;
        }

        int max_next = pos + max_skip_count - skip_used + 1;
        if (max_next > seq_len - remaining + 1) max_next = seq_len - remaining + 1;

        for (int next = pos; next < max_next; ++next) {
            int skip = next - pos;
            positions.push_back(next);
            collect_and_match(seq_len, next + 1, remaining - 1, skip_used + skip,
                              positions, seq, entries, out);
            positions.pop_back();
        }
    }

    bool exec() override {
        const tensor_t* x = inputs[0];
        tensor_t* y = outputs[0];
        float* py = (float*)y->data;

        int batch_size = (x->ndim == 2) ? x->dims[0] : 1;
        int seq_len = (x->ndim == 2) ? x->dims[1] : x->dims[0];
        int out_size = (x->ndim == 2) ? y->dims[1] : y->dims[0];

        memset(py, 0, y->ndata * sizeof(float));

        // Convert input to int64 sequence
        std::vector<int64_t> seq(seq_len);

        for (int b = 0; b < batch_size; ++b) {
            float* out = py + b * out_size;

            // Extract sequence
            if (x->type == NNR_DATA_TYPE_INT32) {
                const int32_t* px = (const int32_t*)x->data + b * seq_len;
                for (int i = 0; i < seq_len; ++i) seq[i] = px[i];
            } else if (x->type == NNR_DATA_TYPE_INT64) {
                const int64_t* px = (const int64_t*)x->data + b * seq_len;
                for (int i = 0; i < seq_len; ++i) seq[i] = px[i];
            } else {
                return false;
            }

            for (int g = min_gram_length; g <= max_gram_length; ++g) {
                if (g - 1 >= (int)ngrams_by_length.size()) continue;
                const auto& entries = ngrams_by_length[g - 1];
                if (entries.empty()) continue;

                for (int start = 0; start <= seq_len - g; ++start) {
                    std::vector<int> positions;
                    positions.reserve(g);
                    positions.push_back(start);
                    collect_and_match(seq_len, start + 1, g - 1, 0,
                                      positions, seq.data(), entries, out);
                }
            }
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_TfIdfVectorizer(int opset, pool_t& pool) { return pool_new<TfIdfVectorizer_operator>(pool); }

} // namespace nnr
