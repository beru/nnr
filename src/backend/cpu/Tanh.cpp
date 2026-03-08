#include "nnr.h"
#include "util.h"
#include "thread_pool.h"

namespace nnr {

namespace {

struct Tanh_operator : public operator_t {

    static void apply_inplace(float* data, int rows, int cols, int stride,
                              const operator_t* self, const float* bias, int offset) {
        for (int r = 0; r < rows; r++) {
            float* row = data + (size_t)r * stride;
            float bv = bias ? bias[r] : 0.0f;
            for (int i = 0; i < cols; ++i)
                row[i] = tanhf(row[i] + bv);
        }
        if (self->post_fn)
            self->post_fn(data, rows, cols, stride, self->fused_op, nullptr, offset);
    }

    bool init() override {
        if (!is_inout_size(1, 1)) return false;
        fusable_apply = &apply_inplace;
        layout_mask = LAYOUT_ALL;
        return true;
    }

    scroll_info_t scroll_info() const override {
        if (inputs[0]->ndim < 3) return {};
        return { .scrollable = true };
    }

    bool exec_strip(int out_row_start, int out_rows,
        int in_row_start, int in_rows) override
    {
        const tensor_t* x = inputs[0];
        const tensor_t* y = outputs[0];
        return exec_strip_elementwise((const float*)x->data, (float*)y->data,
            x->ndata, x->dims, x->ndim, out_row_start, out_rows,
            [](float v) { return tanhf(v); }, ring_out.orig_H,
            y->dims, y->ndata);
    }

    template <typename T>
    bool exec() {
        foreach_tensor<T>([](auto x){ return tanh(x); });
        return true;
    }

    bool exec() override {
        return typed_exec<Tanh_operator,
            opset_t<13, bfloat16_t, float16_t, float, double>,
            opset_t<1, float16_t, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=no layout=[NCHW,NHWC,BLOCKED_16,BLOCKED_8] scroll=yes fusion=post_op
operator_t* resolver_default_op_Tanh(int opset, pool_t& pool)
{
    return pool_new<Tanh_operator>(pool);
}

} // namespace nnr
