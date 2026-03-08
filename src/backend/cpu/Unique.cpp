#include "nnr.h"
#include "util.h"
#include "allocator.h"
#include <algorithm>
#include <cstring>

namespace nnr {

namespace {

struct Unique_operator : public operator_t {
    int axis_attr;
    int sorted;
    bool has_axis;

    bool init() override {
        if (inputs.empty() || outputs.empty()) return false;
        sorted = attribute(attr_key_t::sorted, (int32_t)1);

        // Check if axis attribute exists
        has_axis = false;
        if (attr_t* a = find_attr("axis")) {
            has_axis = true;
            axis_attr = (int)a->i;
        }
        return true;
    }

    bool reshape() override {
        // Cannot determine output shape without running - defer to exec
        return true;
    }

    template <typename T>
    bool exec() {
        arena_scope_t scope(ctx->arena);
        const tensor_t* x = inputs[0];
        const T* px = (const T*)x->data;

        if (!has_axis) {
            // Flatten mode: treat input as 1D
            size_t N = x->ndata;
            // Collect (value, original_index) pairs
            struct Entry { T val; int idx; };
            Entry* entries = scope.alloc_arr<Entry>(N);
            for (size_t i = 0; i < N; ++i) {
                entries[i] = {px[i], (int)i};
            }

            // Find unique values
            arena_vector<int> first_indices(arena_allocator<int>{scope.arena});
            int* inverse = scope.alloc_arr<int>(N);
            arena_vector<int> counts_vec(arena_allocator<int>{scope.arena});

            if (sorted) {
                // Sort by value
                int* order = scope.alloc_arr<int>(N);
                for (size_t i = 0; i < N; ++i) order[i] = (int)i;
                std::sort(order, order + N, [&](int a, int b) {
                    return (double)px[a] < (double)px[b];
                });

                int unique_count = 0;
                for (size_t i = 0; i < N; ++i) {
                    int oi = order[i];
                    if (i == 0 || !((double)px[oi] == (double)px[order[i-1]])) {
                        first_indices.push_back(oi);
                        counts_vec.push_back(1);
                        unique_count++;
                    } else {
                        counts_vec.back()++;
                        // Update first_indices to be the minimum original index
                        if (oi < first_indices.back())
                            first_indices.back() = oi;
                    }
                }

                // Build unique values in sorted order
                T* unique_vals = scope.alloc_arr<T>(unique_count);
                for (int i = 0; i < unique_count; ++i)
                    unique_vals[i] = px[first_indices[i]];

                // Build inverse mapping
                for (size_t i = 0; i < N; ++i) {
                    T v = px[i];
                    // Binary search in unique_vals
                    for (int j = 0; j < unique_count; ++j) {
                        if ((double)unique_vals[j] == (double)v) {
                            inverse[i] = j;
                            break;
                        }
                    }
                }

                // Output 0: unique values
                {
                    small_vector<int> dims(1);
                    dims[0] = unique_count;
                    outputs[0]->reshape(dims, x->type);
                    T* py = (T*)outputs[0]->data;
                    for (int i = 0; i < unique_count; ++i)
                        py[i] = unique_vals[i];
                }
            } else {
                // Unsorted: preserve order of first occurrence
                arena_vector<T> unique_vals(arena_allocator<T>{scope.arena});
                for (size_t i = 0; i < N; ++i) {
                    bool found = false;
                    for (int j = 0; j < (int)unique_vals.size(); ++j) {
                        if ((double)unique_vals[j] == (double)px[i]) {
                            inverse[i] = j;
                            counts_vec[j]++;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        inverse[i] = (int)unique_vals.size();
                        unique_vals.push_back(px[i]);
                        first_indices.push_back((int)i);
                        counts_vec.push_back(1);
                    }
                }

                int unique_count = (int)unique_vals.size();
                small_vector<int> dims(1);
                dims[0] = unique_count;
                outputs[0]->reshape(dims, x->type);
                T* py = (T*)outputs[0]->data;
                for (int i = 0; i < unique_count; ++i)
                    py[i] = unique_vals[i];
            }

            // Output 1: indices (first occurrence positions)
            if (outputs.size() > 1) {
                int unique_count = (int)first_indices.size();
                small_vector<int> dims(1);
                dims[0] = unique_count;
                outputs[1]->reshape(dims, NNR_DATA_TYPE_INT64);
                int64_t* pi = (int64_t*)outputs[1]->data;
                for (int i = 0; i < unique_count; ++i)
                    pi[i] = first_indices[i];
            }

            // Output 2: inverse_indices
            if (outputs.size() > 2) {
                small_vector<int> dims(1);
                dims[0] = (int)N;
                outputs[2]->reshape(dims, NNR_DATA_TYPE_INT64);
                int64_t* pi = (int64_t*)outputs[2]->data;
                for (size_t i = 0; i < N; ++i)
                    pi[i] = inverse[i];
            }

            // Output 3: counts
            if (outputs.size() > 3) {
                int unique_count = (int)counts_vec.size();
                small_vector<int> dims(1);
                dims[0] = unique_count;
                outputs[3]->reshape(dims, NNR_DATA_TYPE_INT64);
                int64_t* pi = (int64_t*)outputs[3]->data;
                for (int i = 0; i < unique_count; ++i)
                    pi[i] = counts_vec[i];
            }
        } else {
            // Axis mode: find unique slices along the given axis
            int axis = axis_attr;
            if (axis < 0) axis += x->ndim;

            int axis_dim = x->dims[axis];
            // Compute slice size (elements per slice along axis)
            size_t slice_size = 1;
            for (int d = 0; d < x->ndim; ++d)
                if (d != axis) slice_size *= x->dims[d];

            // Extract slices as byte arrays for comparison
            size_t elem_size = x->ndata / (axis_dim > 0 ? axis_dim : 1);
            // Actually compute stride for axis
            size_t outer = 1, inner = 1;
            for (int d = 0; d < axis; ++d) outer *= x->dims[d];
            for (int d = axis + 1; d < x->ndim; ++d) inner *= x->dims[d];

            // Function to compare two slices along axis
            auto slices_equal = [&](int a, int b) -> bool {
                for (size_t o = 0; o < outer; ++o) {
                    for (size_t i = 0; i < inner; ++i) {
                        size_t idx_a = (o * axis_dim + a) * inner + i;
                        size_t idx_b = (o * axis_dim + b) * inner + i;
                        if (!((double)px[idx_a] == (double)px[idx_b]))
                            return false;
                    }
                }
                return true;
            };

            auto slice_less = [&](int a, int b) -> bool {
                for (size_t o = 0; o < outer; ++o) {
                    for (size_t i = 0; i < inner; ++i) {
                        size_t idx_a = (o * axis_dim + a) * inner + i;
                        size_t idx_b = (o * axis_dim + b) * inner + i;
                        double va = (double)px[idx_a], vb = (double)px[idx_b];
                        if (va < vb) return true;
                        if (va > vb) return false;
                    }
                }
                return false;
            };

            int* order = scope.alloc_arr<int>(axis_dim);
            for (int i = 0; i < axis_dim; ++i) order[i] = i;

            if (sorted) {
                std::sort(order, order + axis_dim, [&](int a, int b) {
                    return slice_less(a, b);
                });
            }

            // Find unique slices in order
            arena_vector<int> unique_indices(arena_allocator<int>{scope.arena});
            int* inverse = scope.alloc_arr<int>(axis_dim);
            arena_vector<int> counts_vec(arena_allocator<int>{scope.arena});

            for (int i = 0; i < axis_dim; ++i) {
                int oi = order[i];
                bool found = false;
                if (sorted && i > 0 && slices_equal(order[i-1], oi)) {
                    inverse[oi] = inverse[order[i-1]];
                    counts_vec.back()++;
                    found = true;
                }
                if (!found && !sorted) {
                    for (int j = 0; j < (int)unique_indices.size(); ++j) {
                        if (slices_equal(unique_indices[j], oi)) {
                            inverse[oi] = j;
                            counts_vec[j]++;
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    inverse[oi] = (int)unique_indices.size();
                    unique_indices.push_back(oi);
                    counts_vec.push_back(1);
                }
            }

            int unique_count = (int)unique_indices.size();

            // Output 0: unique tensor
            {
                small_vector<int> dims(x->ndim);
                for (int d = 0; d < x->ndim; ++d)
                    dims[d] = (d == axis) ? unique_count : x->dims[d];
                outputs[0]->reshape(dims, x->type);
                T* py = (T*)outputs[0]->data;

                for (int u = 0; u < unique_count; ++u) {
                    int src_slice = unique_indices[u];
                    for (size_t o = 0; o < outer; ++o) {
                        for (size_t i = 0; i < inner; ++i) {
                            size_t src_idx = (o * axis_dim + src_slice) * inner + i;
                            size_t dst_idx = (o * unique_count + u) * inner + i;
                            py[dst_idx] = px[src_idx];
                        }
                    }
                }
            }

            // Output 1: indices
            if (outputs.size() > 1) {
                small_vector<int> dims(1);
                dims[0] = unique_count;
                outputs[1]->reshape(dims, NNR_DATA_TYPE_INT64);
                int64_t* pi = (int64_t*)outputs[1]->data;
                for (int i = 0; i < unique_count; ++i)
                    pi[i] = unique_indices[i];
            }

            // Output 2: inverse_indices
            if (outputs.size() > 2) {
                small_vector<int> dims(1);
                dims[0] = axis_dim;
                outputs[2]->reshape(dims, NNR_DATA_TYPE_INT64);
                int64_t* pi = (int64_t*)outputs[2]->data;
                for (int i = 0; i < axis_dim; ++i)
                    pi[i] = inverse[i];
            }

            // Output 3: counts
            if (outputs.size() > 3) {
                small_vector<int> dims(1);
                dims[0] = unique_count;
                outputs[3]->reshape(dims, NNR_DATA_TYPE_INT64);
                int64_t* pi = (int64_t*)outputs[3]->data;
                for (int i = 0; i < unique_count; ++i)
                    pi[i] = counts_vec[i];
            }
        }
        return true;
    }

    bool exec() override {
        return typed_exec<Unique_operator,
            int8_t, int16_t, int32_t, int64_t,
            uint8_t, uint16_t, uint32_t, uint64_t,
            float16_t, bfloat16_t, float, double
        >(this, inputs[0]->type);
    }
};

} // namespace

// @nnr-meta-op mt=no workspace=yes
operator_t* resolver_default_op_Unique(int opset, pool_t& pool)
{
    return pool_new<Unique_operator>(pool);
}

} // namespace nnr
