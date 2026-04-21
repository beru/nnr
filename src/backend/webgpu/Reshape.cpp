#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/buffer.h"

#include <cstring>

// Reshape / Squeeze / Unsqueeze / Flatten / Identity — all view ops that
// share the input's storage. On WebGPU we just alias the buffer record:
// no kernel dispatch, no copy. Output metadata (dims) is computed by each
// op's reshape(); the data pointer on CPU is still the planner's allocation
// (memory planner skips alloc for view ops via view_input_index), and the
// GPU buffer is shared via webgpu::alias().

namespace nnr {

namespace {

// View-op exec: mirror CPU behavior on the tensor side and re-alias the
// GPU residency record on the GPU side. Both branches are needed:
//   (a) During fold_run(), a constant-folded view op sees a freshly-
//       allocated y buffer distinct from x. reshape()'s alias() was a
//       no-op because x had no GPU record yet (initializers never touch
//       the GPU). CPU exec_impl does `copy_data(y, x)` in this case, so
//       we do the same; the next WebGPU consumer's ensure_buffer +
//       upload_if_needed will then push y's CPU data.
//   (b) At run time, x has been overwritten by its upstream producer on
//       the GPU. y's GPU record (snapshot from reshape() time, possibly
//       empty) is stale. We re-alias y → x so y sees x's current record,
//       then point y->data at x->data to match CPU semantics.
// CPU Reshape does exactly the same thing in `exec_impl`.
static inline void view_exec_sync(tensor_t* y, const tensor_t* x) {
    if (!y->owns_data) {
        y->data = x->data;
    } else if (y->data != x->data) {
        copy_data(y, x);
    }
    // Re-alias the GPU residency record. Covers (b): x's record may not
    // have existed when reshape() aliased, so propagate the current one.
    webgpu::alias(y, x);
}

// --- Reshape (opset 5+: dims come from inputs[1]) ---------------------------
struct Reshape_op_webgpu : public operator_t {
    int view_input_index() const override { return 0; }

    bool init() override { return is_inout_size(2, 1); }

    bool reshape() override {
        const tensor_t* x    = inputs[0];
        const tensor_t* shape = inputs[1];
        tensor_t*       y    = outputs[0];
        if (shape->type != NNR_DATA_TYPE_INT64) return false;

        const int64_t* sp = (const int64_t*)shape->data;
        int n = (int)shape->ndata;
        small_vector<int> out(n);
        int64_t total = 1;
        int     unk   = -1;
        for (int i = 0; i < n; ++i) {
            if (sp[i] == -1)     { unk = i; out[i] = 1; }
            else if (sp[i] == 0) { out[i] = x->dims[i]; total *= out[i]; }
            else                 { out[i] = (int)sp[i]; total *= sp[i]; }
        }
        if (unk >= 0) {
            if (total == 0) return false;
            out[unk] = (int)(x->ndata / total);
        }
        if (!y->reshape(out, x->type)) return false;
        webgpu::alias(y, x);
        return true;
    }

    bool exec() override { view_exec_sync(outputs[0], inputs[0]); return true; }
};

// --- Identity ---------------------------------------------------------------
struct Identity_op_webgpu : public operator_t {
    int view_input_index() const override { return 0; }
    bool init() override    { return is_inout_size(1, 1); }
    bool reshape() override {
        if (!outputs[0]->reshape_identity(inputs[0])) return false;
        webgpu::alias(outputs[0], inputs[0]);
        return true;
    }
    bool exec() override    { return true; }
};

// --- Squeeze / Unsqueeze helpers -------------------------------------------
static int64_t read_axis_i64(const tensor_t* t, int i) {
    if (t->type == NNR_DATA_TYPE_INT32) return ((const int32_t*)t->data)[i];
    return ((const int64_t*)t->data)[i];
}

// --- Squeeze (drop size-1 axes). Opset 13+ axes via input; older via attr.
struct Squeeze_op_webgpu : public operator_t {
    int view_input_index() const override { return 0; }
    bool init() override    { return (inputs.size() == 1 || inputs.size() == 2) && outputs.size() == 1; }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        int in_ndim = x->ndim;
        if (in_ndim > 8) return false;

        bool drop[8] = {};
        int n_drops_requested = 0;
        bool explicit_axes = false;
        auto mark = [&](int a) {
            if (a < 0) a += in_ndim;
            if (a < 0 || a >= in_ndim) return false;
            if (x->dims[a] != 1) return false;
            if (!drop[a]) { drop[a] = true; n_drops_requested += 1; }
            return true;
        };
        if (inputs.size() == 2 && inputs[1] && inputs[1]->data && inputs[1]->ndata > 0) {
            explicit_axes = true;
            const tensor_t* t = inputs[1];
            for (int i = 0; i < (int)t->ndata; ++i)
                if (!mark((int)read_axis_i64(t, i))) return false;
        } else {
            int64_t* attr_axes = nullptr;
            int nax = attribute(attr_key_t::axes, attr_axes);
            if (nax > 0) {
                explicit_axes = true;
                for (int i = 0; i < nax; ++i)
                    if (!mark((int)attr_axes[i])) return false;
            }
        }
        if (!explicit_axes) {
            for (int a = 0; a < in_ndim; ++a)
                if (x->dims[a] == 1) { drop[a] = true; n_drops_requested += 1; }
        }

        int out[8];
        int out_n = 0;
        for (int a = 0; a < in_ndim; ++a)
            if (!drop[a]) out[out_n++] = x->dims[a];
        // Fully-squeezed scalar — keep rank ≥ 1 so downstream ops don't trip.
        if (out_n == 0) { out[0] = 1; out_n = 1; }
        if (!y->reshape(std::span<const int>(out, out_n), x->type)) return false;
        webgpu::alias(y, x);
        return true;
    }
    bool exec() override { view_exec_sync(outputs[0], inputs[0]); return true; }
};

// --- Unsqueeze (insert size-1 axes at the specified positions). -------------
struct Unsqueeze_op_webgpu : public operator_t {
    int view_input_index() const override { return 0; }
    bool init() override    { return (inputs.size() == 1 || inputs.size() == 2) && outputs.size() == 1; }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        int in_ndim = x->ndim;

        // Collect requested insertion axes into an int buffer.
        int req_axes[8];
        int n_req = 0;
        if (inputs.size() == 2 && inputs[1] && inputs[1]->data && inputs[1]->ndata > 0) {
            const tensor_t* t = inputs[1];
            n_req = (int)t->ndata;
            if (n_req > 8) return false;
            for (int i = 0; i < n_req; ++i) req_axes[i] = (int)read_axis_i64(t, i);
        } else {
            int64_t* attr_axes = nullptr;
            int nax = attribute(attr_key_t::axes, attr_axes);
            if (nax <= 0 || nax > 8) return false;
            n_req = nax;
            for (int i = 0; i < n_req; ++i) req_axes[i] = (int)attr_axes[i];
        }

        int out_ndim = in_ndim + n_req;
        if (out_ndim > 8) return false;

        // Normalize negative axes against the *output* rank (ONNX Unsqueeze
        // spec) and bucket them onto the output-axis slots.
        bool is_new[8] = {};
        for (int i = 0; i < n_req; ++i) {
            int a = req_axes[i];
            if (a < 0) a += out_ndim;
            if (a < 0 || a >= out_ndim) return false;
            if (is_new[a]) return false;   // duplicate axis
            is_new[a] = true;
        }

        int out[8];
        int ix = 0;
        for (int a = 0; a < out_ndim; ++a) {
            if (is_new[a]) out[a] = 1;
            else           out[a] = x->dims[ix++];
        }
        if (!y->reshape(std::span<const int>(out, out_ndim), x->type)) return false;
        webgpu::alias(y, x);
        return true;
    }
    bool exec() override { view_exec_sync(outputs[0], inputs[0]); return true; }
};

// --- Dropout at inference. ONNX Dropout has up to 3 inputs (data, ratio,
//     training_mode) and up to 2 outputs (data, mask). We only support the
//     inference path: single output, training_mode absent or 0. In that case
//     data passes through untouched — same as Identity — so we can alias.
struct Dropout_op_webgpu : public operator_t {
    int view_input_index() const override { return 0; }
    bool init() override {
        return (inputs.size() >= 1 && inputs.size() <= 3) && outputs.size() == 1;
    }
    bool reshape() override {
        // Reject training mode (mask output path not implemented). Ratio
        // input is irrelevant at inference so we don't even look at it.
        if (inputs.size() >= 3 && inputs[2] && inputs[2]->data) {
            const tensor_t* tm = inputs[2];
            if (tm->ndata >= 1) {
                bool training = false;
                if      (tm->type == NNR_DATA_TYPE_BOOL)  training = ((const uint8_t*)tm->data)[0] != 0;
                else if (tm->type == NNR_DATA_TYPE_INT32) training = ((const int32_t*)tm->data)[0] != 0;
                else if (tm->type == NNR_DATA_TYPE_INT64) training = ((const int64_t*)tm->data)[0] != 0;
                if (training) return false;
            }
        }
        if (!outputs[0]->reshape_identity(inputs[0])) return false;
        webgpu::alias(outputs[0], inputs[0]);
        return true;
    }
    bool exec() override { view_exec_sync(outputs[0], inputs[0]); return true; }
};

// --- Flatten (collapse to 2D around axis) -----------------------------------
struct Flatten_op_webgpu : public operator_t {
    int axis_attr = 1;
    int view_input_index() const override { return 0; }

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        axis_attr = attribute(attr_key_t::axis, 1);
        return true;
    }

    bool reshape() override {
        const tensor_t* x = inputs[0];
        tensor_t*       y = outputs[0];
        int a = axis_attr < 0 ? axis_attr + x->ndim : axis_attr;
        if (a < 0 || a > x->ndim) return false;
        int outer = 1, inner = 1;
        for (int i = 0; i < a;       ++i) outer *= x->dims[i];
        for (int i = a; i < x->ndim; ++i) inner *= x->dims[i];
        int dims[2] = { outer, inner };
        if (!y->reshape(std::span<const int>(dims, 2), x->type)) return false;
        webgpu::alias(y, x);
        return true;
    }

    bool exec() override { view_exec_sync(outputs[0], inputs[0]); return true; }
};

} // namespace

operator_t* resolver_default_op_Reshape_webgpu   (int, pool_t& pool) { return pool_new<Reshape_op_webgpu>   (pool); }
operator_t* resolver_default_op_Identity_webgpu  (int, pool_t& pool) { return pool_new<Identity_op_webgpu>  (pool); }
operator_t* resolver_default_op_Flatten_webgpu   (int, pool_t& pool) { return pool_new<Flatten_op_webgpu>   (pool); }
operator_t* resolver_default_op_Squeeze_webgpu   (int, pool_t& pool) { return pool_new<Squeeze_op_webgpu>   (pool); }
operator_t* resolver_default_op_Unsqueeze_webgpu (int, pool_t& pool) { return pool_new<Unsqueeze_op_webgpu> (pool); }
operator_t* resolver_default_op_Dropout_webgpu   (int, pool_t& pool) { return pool_new<Dropout_op_webgpu>   (pool); }

} // namespace nnr
