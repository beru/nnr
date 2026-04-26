#include "nnr.h"
#include "util.h"
#include "allocator.h"
#include <cstring>

namespace nnr {

namespace {

struct Loop_operator : public operator_t {
    graph_t* body = nullptr;
    std::span<const std::string_view> body_input_names;
    std::span<const std::string_view> body_output_names;

    bool init() override {
        if (inputs.size() < 2 || outputs.empty())
            return false;
        body = attribute_subgraph("body");
        if (!body) return false;
        attr_t* body_attr = find_attr("body");
        if (!body_attr) return false;
        body_input_names  = body_attr->subgraph_inputs;
        body_output_names = body_attr->subgraph_outputs;
        return true;
    }

    bool reshape() override {
        if (!body) return false;
        int num_carried = (int)inputs.size() - 2;
        int num_scan = (int)body_output_names.size() - 1 - num_carried;
        if (num_scan < 0) num_scan = 0;

        // Seed body inputs (iter=0, cond=true, carried=initial state) so the
        // body can run a shape-propagation pass. Use real data, not just
        // shape, since cheap shape-driven ops inside the body (Shape, Gather,
        // Slice, Concat, Cast) feed sizes into Reshape/Resize and need values.
        small_vector<int> scalar_dims;
        if (body_input_names.size() > 0) {
            tensor_t* t = ctx->search_tensor(body_input_names[0]);
            if (t && t->reshape(scalar_dims, NNR_DATA_TYPE_INT64) && t->data)
                *(int64_t*)t->data = 0;
        }
        if (body_input_names.size() > 1) {
            tensor_t* t = ctx->search_tensor(body_input_names[1]);
            if (t && t->reshape(scalar_dims, NNR_DATA_TYPE_BOOL) && t->data)
                *(bool*)t->data = true;
        }
        for (int i = 0; i < num_carried; ++i) {
            if (body_input_names.size() <= (size_t)(i + 2)) break;
            const tensor_t* src = inputs[i + 2];
            if (!src) continue;
            tensor_t* t = ctx->search_tensor(body_input_names[i + 2]);
            if (!t) continue;
            if (src->data) t->apply(*src);
            else t->reshape_identity(src);
        }

        // Body shape propagation: mirrors fold_run for the subgraph. Reshape
        // every node and best-effort exec the cheap ones so shape-driven data
        // (Shape→Slice→Cast→Concat producing Reshape/Resize sizes) is valid
        // by the time downstream ops reshape. Expensive/control-flow/random
        // ops are skipped — their reshape alone is enough for shape inference.
        auto is_unsafe = [](std::string_view op) {
            return op == "RandomNormal"     || op == "RandomNormalLike" ||
                   op == "RandomUniform"    || op == "RandomUniformLike" ||
                   op == "Multinomial"      || op == "Bernoulli";
        };
        auto is_expensive = [](std::string_view op) {
            return op == "Conv"        || op == "ConvTranspose"  ||
                   op == "MatMul"      || op == "Gemm"           ||
                   op == "ConvInteger" || op == "MatMulInteger"  ||
                   op == "QLinearConv" || op == "QLinearMatMul"  ||
                   op == "LSTM"        || op == "GRU"            || op == "RNN" ||
                   op == "Loop"        || op == "If"             || op == "Scan" ||
                   op == "NonMaxSuppression" || op == "TopK"     || op == "RoiAlign";
        };
        for (auto* n : body->nodes) {
            if (!n->reshape()) continue;
            if (is_unsafe(n->op_type) || is_expensive(n->op_type)) continue;
            bool has_null = false;
            for (auto* t : n->inputs)
                if (t && (!t->data || t->type == NNR_DATA_TYPE_UNDEFINED)) { has_null = true; break; }
            for (auto* t : n->outputs)
                if (t && !t->data && t->type != NNR_DATA_TYPE_UNDEFINED) { has_null = true; break; }
            if (!has_null) n->exec();
        }

        // Carried final outputs inherit body output shapes.
        for (int i = 0; i < num_carried && i < (int)outputs.size(); ++i) {
            size_t out_idx = (size_t)(1 + i);
            if (body_output_names.size() <= out_idx) break;
            const tensor_t* src = ctx->search_tensor(body_output_names[out_idx]);
            if (src && outputs[i] && src->type != NNR_DATA_TYPE_UNDEFINED)
                outputs[i]->reshape_identity(src);
        }

        // Scan outputs prepend an iteration-count dim. If M (trip count) is a
        // static initializer we use it; otherwise default to 1 — the typical
        // batched case (e.g. ssd_mobilenet's cond-driven loop with batch=1).
        // Downstream Shape→Gather→Reshape chains read dim[0] through the
        // backbone, so 0 would zero-out feature maps and break Concat alignment.
        // exec() rewrites the real shape once iteration count is known.
        int placeholder_iters = 1;
        if (inputs[0] && inputs[0]->ndata > 0 && inputs[0]->data) {
            int64_t m = *(const int64_t*)inputs[0]->data;
            if (m > 0 && m < INT_MAX) placeholder_iters = (int)m;
        }
        for (int i = 0; i < num_scan && (num_carried + i) < (int)outputs.size(); ++i) {
            size_t out_idx = (size_t)(1 + num_carried + i);
            if (body_output_names.size() <= out_idx) break;
            const tensor_t* src = ctx->search_tensor(body_output_names[out_idx]);
            if (!src || src->type == NNR_DATA_TYPE_UNDEFINED) continue;
            tensor_t* dst = outputs[num_carried + i];
            if (!dst) continue;
            small_vector<int> dims(src->ndim + 1);
            dims[0] = placeholder_iters;
            for (int d = 0; d < src->ndim; ++d) dims[d + 1] = src->dims[d];
            dst->reshape(dims, src->type);
        }
        return true;
    }

    bool exec() override {
        int64_t max_trip = INT64_MAX;
        if (inputs[0] && inputs[0]->ndata > 0 && inputs[0]->data)
            max_trip = *(const int64_t*)inputs[0]->data;
        bool cond = true;
        if (inputs[1] && inputs[1]->ndata > 0 && inputs[1]->data)
            cond = *(const bool*)inputs[1]->data;

        int num_carried = (int)inputs.size() - 2;
        int num_scan = (int)body_output_names.size() - 1 - num_carried;
        if (num_scan < 0) num_scan = 0;

        // Temp storage for carried variable state (to avoid aliasing input/output tensors)
        // Arena-backed: fixed size, freed automatically when scope exits.
        arena_scope_t scope(ctx->arena);
        tensor_t** carried = scope.alloc_arr<tensor_t*>(num_carried);
        for (int i = 0; i < num_carried; ++i) carried[i] = nullptr;
        for (int i = 0; i < num_carried; ++i) {
            const tensor_t* src = inputs[i + 2];
            if (src && src->data && src->ndata > 0) {
                carried[i] = new (std::nothrow) tensor_t("", src->type, src->dim_span());
                if (carried[i]) copy_data(carried[i], src);
            }
        }
        struct carried_guard_t {
            tensor_t** v; int n;
            ~carried_guard_t() { for (int i = 0; i < n; ++i) delete v[i]; }
        } carried_guard{carried, num_carried};

        // Scan output accumulators: outer array fixed-size (num_scan), inner grows per iteration.
        // Both backed by arena — freed in bulk when scope exits.
        using avi = arena_vector<tensor_t*>;
        avi* scan_iters = scope.alloc_arr<avi>(num_scan);
        for (int i = 0; i < num_scan; ++i)
            new (&scan_iters[i]) avi(arena_allocator<tensor_t*>{scope.arena});

        for (int64_t iter = 0; iter < max_trip && cond; ++iter) {
            // Set iteration number
            if (body_input_names.size() > 0) {
                tensor_t* t = ctx->search_tensor(body_input_names[0]);
                if (t) {
                    small_vector<int> sd;
                    t->reshape(sd, NNR_DATA_TYPE_INT64);
                    *(int64_t*)t->data = iter;
                }
            }
            // Set condition
            if (body_input_names.size() > 1) {
                tensor_t* t = ctx->search_tensor(body_input_names[1]);
                if (t) {
                    small_vector<int> sd;
                    t->reshape(sd, NNR_DATA_TYPE_BOOL);
                    *(bool*)t->data = cond;
                }
            }
            // Set carried variables
            for (int i = 0; i < num_carried; ++i) {
                if (body_input_names.size() > (size_t)(i + 2) && carried[i]) {
                    tensor_t* t = ctx->search_tensor(body_input_names[i + 2]);
                    if (t && !t->apply(*carried[i])) return false;
                }
            }

            // Execute body. Skip ops whose reshape rejects the input or whose
            // inputs/outputs have null data — without this guard, a failed
            // reshape upstream cascades to downstream ops dereferencing nulls
            // (e.g. ssd_mobilenet's NMS body has 5800 nodes; one bad Gather
            // makes Squeeze.exec crash on a null pointer).
            for (auto* n : body->nodes) {
                if (!n->reshape()) continue;
                bool safe = true;
                for (auto* t : n->inputs)
                    if (t && (!t->data || t->type == NNR_DATA_TYPE_UNDEFINED)) { safe = false; break; }
                for (auto* t : n->outputs)
                    if (t && !t->data && t->type != NNR_DATA_TYPE_UNDEFINED) { safe = false; break; }
                if (safe) n->exec();
            }

            // Read condition output
            if (body_output_names.size() > 0) {
                tensor_t* t = ctx->search_tensor(body_output_names[0]);
                if (t && t->data && t->ndata > 0) cond = *(const bool*)t->data;
            }
            // Read carried variable outputs
            for (int i = 0; i < num_carried; ++i) {
                size_t out_idx = (size_t)(1 + i);
                if (body_output_names.size() > out_idx) {
                    tensor_t* t = ctx->search_tensor(body_output_names[out_idx]);
                    if (t && t->ndata > 0 && t->data) {
                        if (!carried[i])
                            carried[i] = new (std::nothrow) tensor_t("", t->type, t->dim_span());
                        if (carried[i] && !carried[i]->allocation_failed) {
                            if (!carried[i]->apply(*t)) return false;
                        }
                    }
                }
            }
            // Accumulate scan outputs
            for (int i = 0; i < num_scan; ++i) {
                size_t out_idx = (size_t)(1 + num_carried + i);
                if (body_output_names.size() > out_idx) {
                    tensor_t* t = ctx->search_tensor(body_output_names[out_idx]);
                    if (t && t->ndata > 0) {
                        tensor_t* snap = new (std::nothrow) tensor_t("", t->type, t->dim_span());
                        if (snap) { copy_data(snap, t); scan_iters[i].push_back(snap); }
                    }
                }
            }
        }

        // Write final carried variables to outputs
        for (int i = 0; i < num_carried && i < (int)outputs.size(); ++i) {
            if (outputs[i] && carried[i]) {
                if (!outputs[i]->apply(*carried[i])) return false;
            }
        }

        // Write scan outputs
        for (int i = 0; i < num_scan && (num_carried + i) < (int)outputs.size(); ++i) {
            tensor_t* out = outputs[num_carried + i];
            if (!out) continue;
            auto& iters = scan_iters[i];
            int n_iters = (int)iters.size();
            if (n_iters == 0) continue;
            const tensor_t* first = iters[0];
            if (first->type == NNR_DATA_TYPE_SEQUENCE) {
                out->reshape({}, NNR_DATA_TYPE_SEQUENCE);
                sequence_t* out_seq = tensor_get_sequence(out);
                if (out_seq) {
                    for (auto* t : out_seq->tensors) delete t;
                    out_seq->tensors.clear();
                    for (auto* t : iters) {
                        tensor_t* elem = new (std::nothrow) tensor_t("", t->type, t->dim_span());
                        if (elem) copy_data(elem, t);
                        out_seq->tensors.push_back(elem);
                    }
                }
            } else {
                small_vector<int> dims(first->ndim + 1);
                dims[0] = n_iters;
                for (int d = 0; d < first->ndim; ++d) dims[d + 1] = first->dims[d];
                out->reshape(dims, first->type);
                size_t elem_sz = data_type_sizeof(first->type) * first->ndata;
                for (int j = 0; j < n_iters; ++j)
                    memcpy((uint8_t*)out->data + j * elem_sz, iters[j]->data, elem_sz);
            }
            for (auto* t : iters) delete t;
            iters.clear();
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no workspace=yes
operator_t* resolver_default_op_Loop(int opset, pool_t& pool)
{
    return pool_new<Loop_operator>(pool);
}

} // namespace nnr
