#include "nnr.h"
#include "util.h"
#include "allocator.h"
#include <cstring>
#include <algorithm>

namespace nnr {

namespace {

using byte_vec = arena_vector<uint8_t>;

struct Scan_operator : public operator_t {
    graph_t* body = nullptr;
    int num_scan_inputs = 0;
    std::span<const std::string_view> body_input_names;
    std::span<const std::string_view> body_output_names;
    std::span<const int64_t> scan_input_axes;
    std::span<const int64_t> scan_input_directions;
    std::span<const int64_t> scan_output_axes;
    std::span<const int64_t> scan_output_directions;

    // Allocate a span of `count` int64 values filled with `val` from attr_pool.
    std::span<const int64_t> alloc_defaults(int count, int64_t val)
    {
        int64_t* p = ctx->attr_pool.alloc_arr<int64_t>(count);
        std::fill(p, p + count, val);
        return {p, (size_t)count};
    }

    bool init() override {
        if (outputs.empty()) return false;
        body = attribute_subgraph("body");
        if (!body) return false;
        attr_t* body_attr = find_attr("body");
        if (!body_attr) return false;
        body_input_names  = body_attr->subgraph_inputs;
        body_output_names = body_attr->subgraph_outputs;

        num_scan_inputs = attribute(attr_key_t::num_scan_inputs, 0);
        if (num_scan_inputs <= 0) return false;

        int64_t* ptr;
        int n;
        int default_axis = (opset < 9) ? 1 : 0;

        n = attribute(attr_key_t::scan_input_axes, ptr);
        scan_input_axes = (n > 0) ? std::span<const int64_t>(ptr, n)
                                   : alloc_defaults(num_scan_inputs, default_axis);

        n = attribute(attr_key_t::scan_input_directions, ptr);
        scan_input_directions = (n > 0) ? std::span<const int64_t>(ptr, n)
                                         : alloc_defaults(num_scan_inputs, 0);

        int input_offset = (opset < 9) ? 1 : 0;
        int num_state = (int)inputs.size() - input_offset - num_scan_inputs;
        int num_scan_outputs = (int)body_output_names.size() - num_state;
        if (num_scan_outputs < 0) num_scan_outputs = 0;

        n = attribute(attr_key_t::scan_output_axes, ptr);
        scan_output_axes = (n > 0) ? std::span<const int64_t>(ptr, n)
                                    : alloc_defaults(num_scan_outputs, default_axis);

        n = attribute(attr_key_t::scan_output_directions, ptr);
        scan_output_directions = (n > 0) ? std::span<const int64_t>(ptr, n)
                                          : alloc_defaults(num_scan_outputs, 0);

        return true;
    }

    bool reshape() override { return true; }

    // Extract a slice along `axis` at index `idx` from `src`, writing bytes to `out_data`.
    void extract_slice(const tensor_t* src, int axis, int idx,
                       byte_vec& out_data,
                       small_vector<int>& out_dims,
                       data_type_t& out_type) {
        out_type = src->type;
        out_dims.clear();
        for (int d = 0; d < src->ndim; ++d)
            if (d != axis) out_dims.push_back(src->dims[d]);

        size_t elem_sz = data_type_sizeof(src->type);
        size_t outer = 1, inner = 1;
        for (int d = 0; d < axis; ++d) outer *= src->dims[d];
        for (int d = axis + 1; d < src->ndim; ++d) inner *= src->dims[d];

        out_data.resize(outer * inner * elem_sz);
        const uint8_t* sp = (const uint8_t*)src->data;
        uint8_t* dp = out_data.data();
        for (size_t o = 0; o < outer; ++o)
            memcpy(dp + o * inner * elem_sz,
                   sp + (o * src->dims[axis] + idx) * inner * elem_sz,
                   inner * elem_sz);
    }

    bool exec() override {
        int input_offset = (opset < 9) ? 1 : 0;
        int num_state = (int)inputs.size() - input_offset - num_scan_inputs;
        int num_scan_outputs = (int)outputs.size() - num_state;
        if (num_scan_outputs < 0) num_scan_outputs = 0;

        if (opset < 9) return exec_v8(num_state, num_scan_outputs);
        return exec_v9(num_state, num_scan_outputs);
    }

    bool exec_v9(int num_state, int num_scan_outputs) {
        int n_iters = 0;
        for (int i = 0; i < num_scan_inputs; ++i) {
            const tensor_t* si = inputs[num_state + i];
            if (!si) continue;
            int axis = (i < (int)scan_input_axes.size()) ? (int)scan_input_axes[i] : 0;
            if (axis < 0) axis += si->ndim;
            n_iters = si->dims[axis]; break;
        }
        if (n_iters <= 0) return false;

        arena_scope_t scope(ctx->arena);
        arena_allocator<uint8_t> byte_alloc{scope.arena};

        // Current state storage (overwritten each iteration)
        byte_vec*                    state_data  = scope.alloc_arr<byte_vec>(num_state);
        data_type_t*               state_types = scope.alloc_arr<data_type_t>(num_state);
        small_vector<int>* state_dims  = scope.alloc_arr<small_vector<int>>(num_state);
        for (int i = 0; i < num_state; ++i) {
            new (&state_data[i]) byte_vec(byte_alloc);
            new (&state_dims[i]) small_vector<int>();
            state_types[i] = NNR_DATA_TYPE_UNDEFINED;
        }
        for (int i = 0; i < num_state; ++i) {
            const tensor_t* src = inputs[i];
            if (src && src->data && src->ndata > 0) {
                size_t sz = data_type_sizeof(src->type) * src->ndata;
                state_data[i].resize(sz);
                memcpy(state_data[i].data(), src->data, sz);
                state_types[i] = src->type;
                state_dims[i].assign(src->dims, src->dims + src->ndim);
            }
        }

        // Scan output accumulation: flat byte buffer per output, concatenated across iters
        byte_vec*                    scan_out_bufs     = scope.alloc_arr<byte_vec>(num_scan_outputs);
        size_t*                      scan_out_iter_sz  = scope.alloc_arr<size_t>(num_scan_outputs);
        data_type_t*               scan_out_types    = scope.alloc_arr<data_type_t>(num_scan_outputs);
        small_vector<int>* scan_out_elem_dims = scope.alloc_arr<small_vector<int>>(num_scan_outputs);
        for (int i = 0; i < num_scan_outputs; ++i) {
            new (&scan_out_bufs[i]) byte_vec(byte_alloc);
            new (&scan_out_elem_dims[i]) small_vector<int>();
            scan_out_iter_sz[i] = 0;
            scan_out_types[i] = NNR_DATA_TYPE_UNDEFINED;
        }

        // Per-iteration scan input slices (reused across iterations)
        byte_vec*                    scan_slices = scope.alloc_arr<byte_vec>(num_scan_inputs);
        data_type_t*               scan_types  = scope.alloc_arr<data_type_t>(num_scan_inputs);
        small_vector<int>* scan_dims   = scope.alloc_arr<small_vector<int>>(num_scan_inputs);
        for (int i = 0; i < num_scan_inputs; ++i) {
            new (&scan_slices[i]) byte_vec(byte_alloc);
            new (&scan_dims[i]) small_vector<int>();
            scan_types[i] = NNR_DATA_TYPE_UNDEFINED;
        }

        for (int iter = 0; iter < n_iters; ++iter) {
            // Extract scan input slices
            for (int i = 0; i < num_scan_inputs; ++i) {
                const tensor_t* src = inputs[num_state + i];
                if (!src || !src->data) continue;
                int axis = (i < (int)scan_input_axes.size()) ? (int)scan_input_axes[i] : 0;
                if (axis < 0) axis += src->ndim;
                int dir = (i < (int)scan_input_directions.size()) ? (int)scan_input_directions[i] : 0;
                extract_slice(src, axis, dir ? (n_iters - 1 - iter) : iter,
                              scan_slices[i], scan_dims[i], scan_types[i]);
            }

            // Set state inputs
            for (int i = 0; i < num_state; ++i) {
                if ((int)body_input_names.size() <= i || state_data[i].empty()) continue;
                tensor_t* t = ctx->search_tensor(body_input_names[i]);
                if (t) {
                    t->reshape(state_dims[i], state_types[i]);
                    memcpy(t->data, state_data[i].data(), state_data[i].size());
                }
            }
            // Set scan input slices
            for (int i = 0; i < num_scan_inputs; ++i) {
                int body_idx = num_state + i;
                if ((int)body_input_names.size() <= body_idx) continue;
                tensor_t* t = ctx->search_tensor(body_input_names[body_idx]);
                if (t && !scan_slices[i].empty()) {
                    t->reshape(scan_dims[i], scan_types[i]);
                    memcpy(t->data, scan_slices[i].data(), scan_slices[i].size());
                }
            }

            // Execute body
            for (auto* n : body->nodes) { n->reshape(); n->exec(); }

            // Read state outputs
            for (int i = 0; i < num_state; ++i) {
                if ((int)body_output_names.size() <= i) continue;
                tensor_t* t = ctx->search_tensor(body_output_names[i]);
                if (t && t->data && t->ndata > 0) {
                    size_t sz = data_type_sizeof(t->type) * t->ndata;
                    state_data[i].resize(sz);
                    memcpy(state_data[i].data(), t->data, sz);
                    state_types[i] = t->type;
                    state_dims[i].assign(t->dims, t->dims + t->ndim);
                }
            }
            // Append scan outputs to flat buffers
            for (int i = 0; i < num_scan_outputs; ++i) {
                int body_idx = num_state + i;
                if ((int)body_output_names.size() <= body_idx) continue;
                tensor_t* t = ctx->search_tensor(body_output_names[body_idx]);
                if (t && t->data && t->ndata > 0) {
                    size_t sz = data_type_sizeof(t->type) * t->ndata;
                    scan_out_iter_sz[i] = sz;
                    scan_out_types[i] = t->type;
                    scan_out_elem_dims[i].assign(t->dims, t->dims + t->ndim);
                    size_t old_sz = scan_out_bufs[i].size();
                    scan_out_bufs[i].resize(old_sz + sz);
                    memcpy(scan_out_bufs[i].data() + old_sz, t->data, sz);
                }
            }
        }

        // Write final state to outputs
        for (int i = 0; i < num_state && i < (int)outputs.size(); ++i)
            if (outputs[i] && !state_data[i].empty()) {
                outputs[i]->reshape(state_dims[i], state_types[i]);
                memcpy(outputs[i]->data, state_data[i].data(), state_data[i].size());
            }

        // Write scan outputs
        for (int i = 0; i < num_scan_outputs && (num_state + i) < (int)outputs.size(); ++i) {
            tensor_t* out = outputs[num_state + i];
            if (!out || scan_out_bufs[i].empty()) continue;
            size_t iter_sz = scan_out_iter_sz[i];
            int ni = (iter_sz > 0) ? (int)(scan_out_bufs[i].size() / iter_sz) : 0;
            if (ni == 0) continue;

            int out_axis = (i < (int)scan_output_axes.size()) ? (int)scan_output_axes[i] : 0;
            int out_dir  = (i < (int)scan_output_directions.size()) ? (int)scan_output_directions[i] : 0;
            int elem_ndim = (int)scan_out_elem_dims[i].size();

            small_vector<int> dims;
            for (int d = 0; d < elem_ndim; ++d) {
                if (d == out_axis) dims.push_back(ni);
                dims.push_back(scan_out_elem_dims[i][d]);
            }
            if (out_axis >= elem_ndim) dims.push_back(ni);

            out->reshape(dims, scan_out_types[i]);
            size_t type_sz = data_type_sizeof(scan_out_types[i]);

            if (out_axis == 0) {
                for (int j = 0; j < ni; ++j) {
                    int src_j = out_dir ? (ni - 1 - j) : j;
                    memcpy((uint8_t*)out->data + j * iter_sz,
                           scan_out_bufs[i].data() + src_j * iter_sz, iter_sz);
                }
            } else {
                size_t outer = 1, inner = 1;
                for (int d = 0; d < out_axis; ++d) outer *= scan_out_elem_dims[i][d];
                for (int d = out_axis; d < elem_ndim; ++d) inner *= scan_out_elem_dims[i][d];
                for (int j = 0; j < ni; ++j) {
                    int src_j = out_dir ? (ni - 1 - j) : j;
                    const uint8_t* src_buf = scan_out_bufs[i].data() + src_j * iter_sz;
                    for (size_t o = 0; o < outer; ++o)
                        memcpy((uint8_t*)out->data + (o * ni + j) * inner * type_sz,
                               src_buf + o * inner * type_sz, inner * type_sz);
                }
            }
        }

        return true;
    }

    bool exec_v8(int num_state, int num_scan_outputs) {
        int batch_size = 1, n_iters = 0;
        for (int i = 0; i < num_scan_inputs; ++i) {
            const tensor_t* si = inputs[1 + num_state + i];
            if (!si || si->ndim < 2) continue;
            batch_size = si->dims[0];
            int axis = (i < (int)scan_input_axes.size()) ? (int)scan_input_axes[i] : 1;
            if (axis < 0) axis += si->ndim;
            n_iters = si->dims[axis]; break;
        }
        if (n_iters <= 0) return false;

        arena_scope_t scope(ctx->arena);
        arena_allocator<uint8_t> byte_alloc{scope.arena};
        int bs_ns = batch_size * num_state;
        int bs_nso = batch_size * num_scan_outputs;

        // Per-batch state: [batch * num_state]
        byte_vec*                    batch_state_data  = scope.alloc_arr<byte_vec>(bs_ns);
        data_type_t*               batch_state_types = scope.alloc_arr<data_type_t>(bs_ns);
        small_vector<int>* batch_state_dims  = scope.alloc_arr<small_vector<int>>(bs_ns);
        for (int k = 0; k < bs_ns; ++k) {
            new (&batch_state_data[k]) byte_vec(byte_alloc);
            new (&batch_state_dims[k]) small_vector<int>();
            batch_state_types[k] = NNR_DATA_TYPE_UNDEFINED;
        }

        // Per-batch scan outputs: [batch * num_scan_outputs]
        byte_vec*                    batch_scan_bufs     = scope.alloc_arr<byte_vec>(bs_nso);
        size_t*                      batch_scan_iter_sz  = scope.alloc_arr<size_t>(bs_nso);
        data_type_t*               batch_scan_types    = scope.alloc_arr<data_type_t>(bs_nso);
        small_vector<int>* batch_scan_elem_dims = scope.alloc_arr<small_vector<int>>(bs_nso);
        for (int k = 0; k < bs_nso; ++k) {
            new (&batch_scan_bufs[k]) byte_vec(byte_alloc);
            new (&batch_scan_elem_dims[k]) small_vector<int>();
            batch_scan_iter_sz[k] = 0;
            batch_scan_types[k] = NNR_DATA_TYPE_UNDEFINED;
        }

        // Reusable per-iteration temporaries
        byte_vec                    batch_slice(byte_alloc);
        small_vector<int> bslice_dims;
        byte_vec*                    scan_slices = scope.alloc_arr<byte_vec>(num_scan_inputs);
        data_type_t*               scan_types  = scope.alloc_arr<data_type_t>(num_scan_inputs);
        small_vector<int>* scan_dims   = scope.alloc_arr<small_vector<int>>(num_scan_inputs);
        for (int i = 0; i < num_scan_inputs; ++i) {
            new (&scan_slices[i]) byte_vec(byte_alloc);
            new (&scan_dims[i]) small_vector<int>();
            scan_types[i] = NNR_DATA_TYPE_UNDEFINED;
        }

        for (int b = 0; b < batch_size; ++b) {
            // Initialize state for this batch from inputs
            for (int i = 0; i < num_state; ++i) {
                const tensor_t* src = inputs[1 + i];
                if (!src || !src->data || src->ndim < 1) continue;
                data_type_t stype;
                extract_slice(src, 0, b,
                              batch_state_data[b * num_state + i],
                              batch_state_dims[b * num_state + i],
                              batch_state_types[b * num_state + i]);
            }

            for (int iter = 0; iter < n_iters; ++iter) {
                // Extract scan input slices for this batch/iter
                for (int i = 0; i < num_scan_inputs; ++i) {
                    const tensor_t* src = inputs[1 + num_state + i];
                    if (!src || !src->data) continue;
                    int axis = (i < (int)scan_input_axes.size()) ? (int)scan_input_axes[i] : 1;
                    if (axis < 0) axis += src->ndim;
                    int dir = (i < (int)scan_input_directions.size()) ? (int)scan_input_directions[i] : 0;

                    data_type_t stype;
                    extract_slice(src, 0, b, batch_slice, bslice_dims, stype);

                    int new_axis = axis - 1;
                    size_t elem_sz = data_type_sizeof(stype);
                    size_t outer = 1, inner = 1;
                    for (int d = 0; d < new_axis; ++d) outer *= bslice_dims[d];
                    for (int d = new_axis + 1; d < (int)bslice_dims.size(); ++d) inner *= bslice_dims[d];

                    scan_dims[i].clear();
                    for (int d = 0; d < (int)bslice_dims.size(); ++d)
                        if (d != new_axis) scan_dims[i].push_back(bslice_dims[d]);
                    scan_types[i] = stype;

                    int slice_idx = dir ? (n_iters - 1 - iter) : iter;
                    size_t slice_sz = 1;
                    for (auto d : scan_dims[i]) slice_sz *= d;
                    scan_slices[i].resize(slice_sz * elem_sz);
                    const uint8_t* sp = batch_slice.data();
                    uint8_t* dp = scan_slices[i].data();
                    for (size_t o = 0; o < outer; ++o)
                        memcpy(dp + o * inner * elem_sz,
                               sp + (o * bslice_dims[new_axis] + slice_idx) * inner * elem_sz,
                               inner * elem_sz);
                }

                // Set state inputs
                for (int i = 0; i < num_state; ++i) {
                    if ((int)body_input_names.size() <= i) continue;
                    byte_vec& sd = batch_state_data[b * num_state + i];
                    if (sd.empty()) continue;
                    tensor_t* t = ctx->search_tensor(body_input_names[i]);
                    if (t) {
                        t->reshape(batch_state_dims[b * num_state + i], batch_state_types[b * num_state + i]);
                        memcpy(t->data, sd.data(), sd.size());
                    }
                }
                // Set scan input slices
                for (int i = 0; i < num_scan_inputs; ++i) {
                    int body_idx = num_state + i;
                    if ((int)body_input_names.size() <= body_idx) continue;
                    tensor_t* t = ctx->search_tensor(body_input_names[body_idx]);
                    if (t && !scan_slices[i].empty()) {
                        t->reshape(scan_dims[i], scan_types[i]);
                        memcpy(t->data, scan_slices[i].data(), scan_slices[i].size());
                    }
                }

                // Execute body
                for (auto* n : body->nodes) { n->reshape(); n->exec(); }

                // Read state outputs
                for (int i = 0; i < num_state; ++i) {
                    if ((int)body_output_names.size() <= i) continue;
                    tensor_t* t = ctx->search_tensor(body_output_names[i]);
                    if (t && t->data && t->ndata > 0) {
                        size_t sz = data_type_sizeof(t->type) * t->ndata;
                        byte_vec& sd = batch_state_data[b * num_state + i];
                        sd.resize(sz);
                        memcpy(sd.data(), t->data, sz);
                        batch_state_types[b * num_state + i] = t->type;
                        batch_state_dims[b * num_state + i].assign(t->dims, t->dims + t->ndim);
                    }
                }
                // Append scan outputs
                for (int i = 0; i < num_scan_outputs; ++i) {
                    int body_idx = num_state + i;
                    if ((int)body_output_names.size() <= body_idx) continue;
                    tensor_t* t = ctx->search_tensor(body_output_names[body_idx]);
                    if (t && t->data && t->ndata > 0) {
                        size_t sz = data_type_sizeof(t->type) * t->ndata;
                        int k = b * num_scan_outputs + i;
                        batch_scan_iter_sz[k] = sz;
                        batch_scan_types[k] = t->type;
                        batch_scan_elem_dims[k].assign(t->dims, t->dims + t->ndim);
                        size_t old_sz = batch_scan_bufs[k].size();
                        batch_scan_bufs[k].resize(old_sz + sz);
                        memcpy(batch_scan_bufs[k].data() + old_sz, t->data, sz);
                    }
                }
            }
        }

        // Assemble state outputs with batch dimension
        for (int i = 0; i < num_state && i < (int)outputs.size(); ++i) {
            if (!outputs[i] || batch_state_data[i].empty()) continue;
            small_vector<int> dims;
            dims.push_back(batch_size);
            for (auto d : batch_state_dims[i]) dims.push_back(d);
            outputs[i]->reshape(dims, batch_state_types[i]);
            size_t elem_sz = batch_state_data[i].size();
            for (int b = 0; b < batch_size; ++b)
                memcpy((uint8_t*)outputs[i]->data + b * elem_sz,
                       batch_state_data[b * num_state + i].data(), elem_sz);
        }

        // Assemble scan outputs with batch dimension
        for (int i = 0; i < num_scan_outputs && (num_state + i) < (int)outputs.size(); ++i) {
            tensor_t* out = outputs[num_state + i];
            if (!out) continue;
            size_t iter_sz = batch_scan_iter_sz[i];  // use batch 0
            int ni = (iter_sz > 0) ? (int)(batch_scan_bufs[i].size() / iter_sz) : 0;
            if (ni == 0) continue;

            int out_axis = (i < (int)scan_output_axes.size()) ? (int)scan_output_axes[i] : 1;
            int out_dir  = (i < (int)scan_output_directions.size()) ? (int)scan_output_directions[i] : 0;
            auto& edims = batch_scan_elem_dims[i];  // from batch 0
            int elem_ndim = (int)edims.size();
            int body_out_axis = out_axis - 1;

            small_vector<int> dims;
            dims.push_back(batch_size);
            for (int d = 0; d < elem_ndim; ++d) {
                if (d == body_out_axis) dims.push_back(ni);
                dims.push_back(edims[d]);
            }
            if (body_out_axis >= elem_ndim) dims.push_back(ni);

            out->reshape(dims, batch_scan_types[i]);

            size_t type_sz = data_type_sizeof(batch_scan_types[i]);

            for (int b = 0; b < batch_size; ++b) {
                int k = b * num_scan_outputs + i;
                size_t b_iter_sz = batch_scan_iter_sz[k];
                int b_ni = (b_iter_sz > 0) ? (int)(batch_scan_bufs[k].size() / b_iter_sz) : 0;
                uint8_t* base = (uint8_t*)out->data + b * b_ni * b_iter_sz;
                if (body_out_axis <= 0) {
                    for (int j = 0; j < b_ni; ++j) {
                        int src_j = out_dir ? (b_ni - 1 - j) : j;
                        memcpy(base + j * b_iter_sz,
                               batch_scan_bufs[k].data() + src_j * b_iter_sz, b_iter_sz);
                    }
                } else {
                    size_t outer = 1, inner = 1;
                    for (int d = 0; d < body_out_axis; ++d) outer *= edims[d];
                    for (int d = body_out_axis; d < elem_ndim; ++d) inner *= edims[d];
                    for (int j = 0; j < b_ni; ++j) {
                        int src_j = out_dir ? (b_ni - 1 - j) : j;
                        const uint8_t* src_buf = batch_scan_bufs[k].data() + src_j * b_iter_sz;
                        for (size_t o = 0; o < outer; ++o)
                            memcpy(base + (o * b_ni + j) * inner * type_sz,
                                   src_buf + o * inner * type_sz, inner * type_sz);
                    }
                }
            }
        }

        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_Scan(int opset, pool_t& pool) { return pool_new<Scan_operator>(pool); }

} // namespace nnr
