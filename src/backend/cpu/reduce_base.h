#pragma once
#include "nnr.h"
#include "util.h"
#include "kernel/quant_exec.h"

namespace nnr {

// acc_widen: integers widen to 64-bit, half-floats widen to float.
// Used by ReduceSum, ReduceMean, ReduceL1, ReduceSumSquare.
template <typename T> struct acc_widen { using type = T; };
template <> struct acc_widen<int8_t>     { using type = int64_t; };
template <> struct acc_widen<int32_t>    { using type = int64_t; };
template <> struct acc_widen<int64_t>    { using type = int64_t; };
template <> struct acc_widen<uint8_t>    { using type = uint64_t; };
template <> struct acc_widen<uint32_t>   { using type = uint64_t; };
template <> struct acc_widen<uint64_t>   { using type = uint64_t; };
template <> struct acc_widen<bfloat16_t> { using type = float; };
template <> struct acc_widen<float16_t>  { using type = float; };

// acc_float: everything accumulates in float, except double.
// Used by ReduceL2, ReduceProd, ReduceLogSum, ReduceLogSumExp.
template <typename T> struct acc_float { using type = float; };
template <> struct acc_float<double>   { using type = double; };

// mean_div: reciprocal multiplication for floats, integer division for ints.
template <typename AccT>
struct mean_div {
    AccT val;
    mean_div(int count) {
        if constexpr (std::is_floating_point_v<AccT>) val = AccT(1.0) / count;
        else val = (AccT)count;
    }
    AccT operator()(AccT sum) const {
        if constexpr (std::is_floating_point_v<AccT>) return sum * val;
        else return sum / val;
    }
};

// Shared base for Reduce* operators: handles init() and reshape().
struct reduce_base_t : public operator_t {
    int keepdims;
    int noop_with_empty_axes;
    small_vector<int> caxes;
    int axes_since_opset = 18;  // ReduceSum overrides to 13

    bool init() override {
        if (outputs.size() != 1) return false;
        keepdims = attribute(attr_key_t::keepdims, 1);
        noop_with_empty_axes = attribute(attr_key_t::noop_with_empty_axes, 0);
        return true;
    }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* x = inputs[0];
        small_vector<int> dims;

        if (opset >= axes_since_opset && inputs.size() > 1 && inputs[1] && inputs[1]->ndata > 0) {
            const tensor_t* a = inputs[1];
            const int64_t* pa = (const int64_t*)a->data;
            caxes.clear();
            for (size_t i = 0; i < a->ndata; ++i) {
                int axis = (int)pa[i];
                if (axis < 0) axis += x->ndim;
                if (axis < 0 || axis >= x->ndim) return false;
                caxes.push_back(axis);
            }
        }else if (opset >= axes_since_opset) {
            if (noop_with_empty_axes) {
                caxes.clear();
            }else {
                caxes.resize(x->ndim);
                for (int i = 0; i < x->ndim; ++i) caxes[i] = i;
            }
        }else {
            int64_t* ints;
            int nint = attribute(attr_key_t::axes, ints);
            if (nint > 0) {
                caxes.resize(nint);
                for (int i = 0; i < nint; ++i) {
                    int axis = (int)ints[i];
                    if (axis < 0) axis += x->ndim;
                    if (axis < 0 || axis >= x->ndim) return false;
                    caxes[i] = axis;
                }
            }else {
                caxes.resize(x->ndim);
                for (int i = 0; i < x->ndim; ++i) caxes[i] = i;
            }
        }
        if (caxes.empty()) return y->reshape_identity(x);
        if (keepdims) {
            dims.assign(x->dims, x->dims + x->ndim);
            for (int i = 0; i < caxes.size(); ++i) dims[caxes[i]] = 1;
        }else {
            for (int i = 0; i < x->ndim; ++i) {
                bool found = false;
                for (int j = 0; j < caxes.size(); ++j) {
                    if (i == caxes[j]) { found = true; break; }
                }
                if (!found) dims.push_back(x->dims[i]);
            }
        }
        return y->reshape(dims, x->type);
    }
};

// Scatter reduce: iterates over output elements and calls fn(px, offset, iter, strides, maxes)
// for each. fn performs the inner reduction over the reduce-axes and returns one output value.
template <typename T, typename Fn>
void reduce_scatter_loop(reduce_base_t* self, const T* px, T* py, Fn fn) {
    const tensor_t* x = self->inputs[0];
    int nax = self->caxes.size();
    int ndim_out = x->ndim - nax;
    small_vector<int> out_max(ndim_out), out_iter(ndim_out), out_strides(ndim_out);
    small_vector<int> in_max(nax), in_iter(nax), in_strides(nax);
    uint32_t mask = 0;
    for (int i = 0; i < nax; ++i) mask |= (1 << self->caxes[i]);
    for (int i = 0, j = 0, k = 0; i < x->ndim; ++i) {
        if (mask & (1 << i)) {
            in_strides[j] = x->strides[i];
            in_max[j] = x->dims[i];
            ++j;
        }else {
            out_strides[k] = x->strides[i];
            out_max[k] = x->dims[i];
            ++k;
        }
    }
    int i = 0;
    do {
        std::fill(in_iter.begin(), in_iter.end(), 0);
        py[i++] = fn(px, stride_offset(out_iter, out_strides), in_iter, in_strides, in_max);
    } while (dim_next(out_iter, out_max));
}

// reduce_exec_accum: shared exec() body for accumulation-based reduce operators.
// AccTrait: acc_widen or acc_float — maps T to the accumulator type.
// empty_val: fill value when input is empty.
// init_val: accumulator initial value (e.g. 0 for sum, 1 for product).
// accum_fn(AccT acc, T val) -> AccT: accumulate one element.
// finalize_fn(AccT acc) -> T: final transform (identity, sqrt, log, etc).
template <typename T, template<typename> class AccTrait, typename AccumFn, typename FinalizeFn>
bool reduce_exec_accum(reduce_base_t* self, T empty_val, typename AccTrait<T>::type init_val,
    AccumFn accum_fn, FinalizeFn finalize_fn)
{
    using AccT = typename AccTrait<T>::type;
    const tensor_t* x = self->inputs[0];
    const T* px = (const T*)x->data;
    T* py = (T*)self->outputs[0]->data;
    if (x->ndata == 0) {
        for (size_t i = 0; i < self->outputs[0]->ndata; ++i) py[i] = empty_val;
        return true;
    }
    if (self->caxes.empty()) { memcpy(py, px, x->ndata * sizeof(T)); return true; }
    {
        auto plan = plan_reduce(x->dims, x->ndim, self->caxes.data(), (int)self->caxes.size());
        if (plan.contiguous) {
            int batch = plan.batch_size, red = plan.reduce_size, tail = plan.tail_size;
            for (int b = 0; b < batch; ++b)
                for (int t = 0; t < tail; ++t) {
                    AccT acc = init_val;
                    for (int r = 0; r < red; ++r)
                        acc = accum_fn(acc, px[(b * red + r) * tail + t]);
                    py[b * tail + t] = finalize_fn(acc);
                }
            return true;
        }
    }
    reduce_scatter_loop<T>(self, px, py, [&](const T* px, int o, auto& iter, auto& strides, auto& maxes) -> T {
        AccT acc = init_val;
        do { acc = accum_fn(acc, px[o + stride_offset(iter, strides)]); } while (dim_next(iter, maxes));
        return finalize_fn(acc);
    });
    return true;
}

} // namespace nnr
