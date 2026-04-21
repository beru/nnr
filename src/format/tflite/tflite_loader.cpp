#include "tflite_loader.h"
#include "tflite_schema.h"
#include "tflite_ops.h"
#include "nnr.h"
#include "attr_key.h"
#include "backend/cpu/solve_operator.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace nnr {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Generate a tensor name from the TFLite tensor name or index.
static std::string_view intern_name(pool_t& pool, std::string_view name, int idx)
{
    if (!name.empty()) {
        size_t len = name.size();
        char* buf = (char*)pool.alloc(len + 1, 1);
        memcpy(buf, name.data(), len);
        buf[len] = '\0';
        return {buf, len};
    }
    // Fallback: "tfl_tensor_<idx>"
    char tmp[32];
    int n = snprintf(tmp, sizeof(tmp), "tfl_tensor_%d", idx);
    char* buf = (char*)pool.alloc(n + 1, 1);
    memcpy(buf, tmp, n + 1);
    return {buf, (size_t)n};
}

// Allocate an attr pair in the pool and populate it.
static std::pair<attr_key_t, attr_t>* alloc_attrs(pool_t& pool, size_t count)
{
    using pair_t = std::pair<attr_key_t, attr_t>;
    auto* buf = static_cast<pair_t*>(pool.alloc(count * sizeof(pair_t), alignof(pair_t)));
    for (size_t i = 0; i < count; ++i)
        new (&buf[i]) pair_t(attr_key_t::unknown, attr_t{});
    return buf;
}

static void set_attr_int(std::pair<attr_key_t, attr_t>& p, attr_key_t key, int64_t val)
{
    p.first = key;
    p.second.kind = attr_t::kind_t::INT;
    p.second.i = val;
}

static void set_attr_float(std::pair<attr_key_t, attr_t>& p, attr_key_t key, float val)
{
    p.first = key;
    p.second.kind = attr_t::kind_t::FLOAT;
    p.second.f = val;
}

static void set_attr_string(std::pair<attr_key_t, attr_t>& p, attr_key_t key,
                            const char* str, pool_t& pool)
{
    p.first = key;
    p.second.kind = attr_t::kind_t::STRING;
    size_t len = strlen(str);
    char* buf = (char*)pool.alloc(len + 1, 1);
    memcpy(buf, str, len + 1);
    p.second.s = {buf, len};
}

static void set_attr_ints(std::pair<attr_key_t, attr_t>& p, attr_key_t key,
                          const int64_t* vals, size_t count, pool_t& pool)
{
    p.first = key;
    p.second.kind = attr_t::kind_t::INTS;
    int64_t* buf = pool.alloc_arr<int64_t>(count);
    memcpy(buf, vals, count * sizeof(int64_t));
    p.second.ints = {buf, count};
}

// Insert a fused activation operator after op_idx in the graph.
// Creates a Relu, Clip, or Tanh node consuming the output of the previous op.
static void insert_fused_activation(context_t* ctx, graph_t* graph,
                                    int8_t activation, tensor_t* producer_output,
                                    int default_opset)
{
    const char* act_op = tflite::fused_activation_op(activation);
    if (!act_op) return;

    // Create an intermediate tensor — the producer's output becomes the activation input,
    // and we create a new tensor for the activation output with the original name.
    // Actually simpler: create a new intermediate tensor, rewire the producer to output to it,
    // and have the activation read from it and write to the original tensor.
    std::string tmp_name = std::string(producer_output->name) + "_pre_act";
    size_t len = tmp_name.size();
    char* name_buf = (char*)ctx->attr_pool.alloc(len + 1, 1);
    memcpy(name_buf, tmp_name.data(), len + 1);
    std::string_view intermed_name{name_buf, len};

    tensor_t* intermed = new (std::nothrow) tensor_t(intermed_name, producer_output->type, {});
    if (!intermed) return;
    ctx->map.emplace_back(intermed_name, intermed);

    // Create activation operator
    operator_t* act = solve_operator(act_op, default_opset, ctx->attr_pool,
                                     static_cast<backend_t>(ctx->preferred_backend));
    struct op_dummy : public operator_t { bool exec() override { return false; } };
    if (!act) act = pool_new<op_dummy>(ctx->attr_pool);

    act->ctx = ctx;
    act->opset = default_opset;
    act->op_type = act_op;
    act->domain = "ai.onnx";

    // Input: intermediate tensor
    tensor_t** in_buf = ctx->attr_pool.alloc_arr<tensor_t*>(1);
    in_buf[0] = intermed;
    act->inputs = {in_buf, 1};

    // Output: original producer output tensor
    tensor_t** out_buf = ctx->attr_pool.alloc_arr<tensor_t*>(1);
    out_buf[0] = producer_output;
    act->outputs = {out_buf, 1};

    // For Clip (RELU6 or RELU_N1_TO_1), set min/max attributes
    if (strcmp(act_op, "Clip") == 0) {
        // Clip needs min/max as input tensors (opset 11+) or attributes (opset 6)
        // For simplicity, create constant tensors for min and max
        float min_val = (activation == tflite::ActivationFunctionType_RELU6) ? 0.0f : -1.0f;
        float max_val = (activation == tflite::ActivationFunctionType_RELU6) ? 6.0f : 1.0f;

        // Reallocate inputs to hold [input, min, max]
        tensor_t** clip_in = ctx->attr_pool.alloc_arr<tensor_t*>(3);
        clip_in[0] = intermed;

        // Create min tensor
        tensor_t* t_min = new (std::nothrow) tensor_t("", NNR_DATA_TYPE_FLOAT32, {});
        if (!t_min || t_min->allocation_failed) { delete t_min; return; }
        *(float*)t_min->data = min_val;
        std::string min_name_s = std::string(producer_output->name) + "_clip_min";
        char* mn = (char*)ctx->attr_pool.alloc(min_name_s.size() + 1, 1);
        memcpy(mn, min_name_s.data(), min_name_s.size() + 1);
        t_min->name = {mn, min_name_s.size()};
        ctx->map.emplace_back(t_min->name, t_min);
        ctx->initializer_names.insert(t_min->name);
        clip_in[1] = t_min;

        // Create max tensor
        tensor_t* t_max = new (std::nothrow) tensor_t("", NNR_DATA_TYPE_FLOAT32, {});
        if (!t_max || t_max->allocation_failed) { delete t_max; return; }
        *(float*)t_max->data = max_val;
        std::string max_name_s = std::string(producer_output->name) + "_clip_max";
        char* xn = (char*)ctx->attr_pool.alloc(max_name_s.size() + 1, 1);
        memcpy(xn, max_name_s.data(), max_name_s.size() + 1);
        t_max->name = {xn, max_name_s.size()};
        ctx->map.emplace_back(t_max->name, t_max);
        ctx->initializer_names.insert(t_max->name);
        clip_in[2] = t_max;

        act->inputs = {clip_in, 3};
    }

    if (!act->init())
        fprintf(stderr, "nnr: tflite fused activation init failed: %s\n", act_op);

    // Rewire: the preceding operator now outputs to 'intermed'
    // Find the producer node (last node in graph) and patch its output
    if (!graph->nodes.empty()) {
        operator_t* prev = graph->nodes.back();
        for (size_t i = 0; i < prev->outputs.size(); ++i) {
            if (prev->outputs[i] == producer_output) {
                prev->outputs[i] = intermed;
                break;
            }
        }
    }

    graph->nodes.push_back(act);
}

// ---------------------------------------------------------------------------
// NHWC → NCHW axis conversion for 4D tensors
// ---------------------------------------------------------------------------

// Convert a TFLite NHWC axis to NCHW axis for 4D tensors.
// NHWC [N,H,W,C] → NCHW [N,C,H,W]: 0→0, 1→2, 2→3, 3→1
// Handles negative axes too.
static int32_t nhwc_to_nchw_axis(int32_t axis, int ndim)
{
    if (ndim != 4) return axis;
    if (axis < 0) axis += ndim;
    static const int map[4] = {0, 2, 3, 1};
    if (axis >= 0 && axis < 4) return map[axis];
    return axis;
}

// ---------------------------------------------------------------------------
// Transpose helpers for NHWC ↔ NCHW weight conversion
// ---------------------------------------------------------------------------

// TFLite Conv2D weights: [OC, KH, KW, IC] (OHWI) → ONNX: [OC, IC, KH, KW] (OIHW)
static bool safe_mul(size_t a, size_t b, size_t& out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    out = a * b;
    return true;
}

static bool safe_total_4d(int d0, int d1, int d2, int d3, size_t elem_sz, size_t& out) {
    if (d0 <= 0 || d1 <= 0 || d2 <= 0 || d3 <= 0 || elem_sz == 0) { out = 0; return false; }
    size_t a, b, c;
    if (!safe_mul((size_t)d0, (size_t)d1, a)) return false;
    if (!safe_mul(a, (size_t)d2, b)) return false;
    if (!safe_mul(b, (size_t)d3, c)) return false;
    if (!safe_mul(c, elem_sz, out)) return false;
    return true;
}

static void transpose_conv_weights_ohwi_to_oihw(tensor_t* t)
{
    if (!t || t->ndim != 4 || !t->data) return;
    int OC = t->dims[0], KH = t->dims[1], KW = t->dims[2], IC = t->dims[3];
    size_t elem_sz = data_type_sizeof(t->type);
    if (elem_sz == 0) return;
    size_t total;
    if (!safe_total_4d(OC, KH, KW, IC, elem_sz, total)) return;
    auto tmp = std::make_unique<uint8_t[]>(total);
    memcpy(tmp.get(), t->data, total);

    // Reindex: dst[oc][ic][kh][kw] = src[oc][kh][kw][ic]
    uint8_t* src = tmp.get();
    uint8_t* dst = (uint8_t*)t->data;
    for (int oc = 0; oc < OC; ++oc)
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw)
                for (int ic = 0; ic < IC; ++ic) {
                    size_t si = ((size_t)oc * KH * KW * IC + kh * KW * IC + kw * IC + ic) * elem_sz;
                    size_t di = ((size_t)oc * IC * KH * KW + ic * KH * KW + kh * KW + kw) * elem_sz;
                    memcpy(dst + di, src + si, elem_sz);
                }

    // Update dims and strides
    t->dims[0] = OC; t->dims[1] = IC; t->dims[2] = KH; t->dims[3] = KW;
    t->strides[3] = 1;
    t->strides[2] = KW;
    t->strides[1] = KH * KW;
    t->strides[0] = IC * KH * KW;
}

// TFLite DepthwiseConv2D weights: [1, KH, KW, OC] → ONNX: [OC, 1, KH, KW]
// (depth_multiplier is folded: OC = IC * depth_multiplier, group = IC)
static void transpose_dw_weights_to_oihw(tensor_t* t)
{
    if (!t || t->ndim != 4 || !t->data) return;
    int D0 = t->dims[0], KH = t->dims[1], KW = t->dims[2], OC = t->dims[3];
    // D0 should be 1 for standard depthwise
    size_t elem_sz = data_type_sizeof(t->type);
    if (elem_sz == 0) return;
    size_t total;
    if (!safe_total_4d(D0, KH, KW, OC, elem_sz, total)) return;
    auto tmp = std::make_unique<uint8_t[]>(total);
    memcpy(tmp.get(), t->data, total);

    // Reindex: dst[oc][0][kh][kw] = src[0][kh][kw][oc]
    uint8_t* src = tmp.get();
    uint8_t* dst = (uint8_t*)t->data;
    for (int oc = 0; oc < OC; ++oc)
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw) {
                size_t si = ((size_t)kh * KW * OC + kw * OC + oc) * elem_sz;
                size_t di = ((size_t)oc * 1 * KH * KW + kh * KW + kw) * elem_sz;
                memcpy(dst + di, src + si, elem_sz);
            }

    t->dims[0] = OC; t->dims[1] = 1; t->dims[2] = KH; t->dims[3] = KW;
    t->strides[3] = 1;
    t->strides[2] = KW;
    t->strides[1] = KH * KW;
    t->strides[0] = 1 * KH * KW;
}

// Transpose NHWC activation tensor to NCHW in-place.
static void transpose_nhwc_to_nchw(tensor_t* t)
{
    if (!t || t->ndim != 4 || !t->data) return;
    int N = t->dims[0], H = t->dims[1], W = t->dims[2], C = t->dims[3];
    size_t elem_sz = data_type_sizeof(t->type);
    if (elem_sz == 0) return;
    size_t total;
    if (!safe_total_4d(N, H, W, C, elem_sz, total)) return;
    auto tmp = std::make_unique<uint8_t[]>(total);
    memcpy(tmp.get(), t->data, total);

    uint8_t* src = tmp.get();
    uint8_t* dst = (uint8_t*)t->data;
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c) {
                    size_t si = ((size_t)n * H * W * C + h * W * C + w * C + c) * elem_sz;
                    size_t di = ((size_t)n * C * H * W + c * H * W + h * W + w) * elem_sz;
                    memcpy(dst + di, src + si, elem_sz);
                }

    t->dims[0] = N; t->dims[1] = C; t->dims[2] = H; t->dims[3] = W;
    t->strides[3] = 1;
    t->strides[2] = W;
    t->strides[1] = H * W;
    t->strides[0] = C * H * W;
}

// ---------------------------------------------------------------------------
// Main loader
// ---------------------------------------------------------------------------

// Per-op attribute extraction and weight transposition, extracted from
// load_tflite() for readability. Each case handles a TFLite builtin
// operator: converts its options struct to ONNX-style attributes,
// transposes weights where needed, and returns the fused activation
// (0 = none) for load_tflite() to append as a post-op node.
static int8_t build_tflite_op_attrs(
    context_t* ctx,
    operator_t* n,
    const tflite::Operator& fop,
    int32_t builtin_code,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t oi,
    const tflite::fb_vec_i32& f_inputs,
    uint32_t n_tensors,
    tensor_t** tensor_map,
    bool* already_transposed)
{
        auto opts = fop.builtin_options();
        int8_t fused_act = 0;

        switch (builtin_code) {
        case tflite::BuiltinOperator_CONV_2D: {
            tflite::Conv2DOptions co(opts);
            fused_act = co.fused_activation();

            // Transpose weights: OHWI → OIHW
            if (n_in >= 2 && n->inputs[1]) {
                transpose_conv_weights_ohwi_to_oihw(n->inputs[1]);
                int32_t widx = f_inputs[1];
                if (widx >= 0 && widx < (int32_t)n_tensors) already_transposed[widx] = true;
            }

            // Build attrs: auto_pad, strides, dilations, group, kernel_shape
            auto* attrs = alloc_attrs(ctx->attr_pool, 5);
            set_attr_string(attrs[0], attr_key_t::auto_pad,
                co.padding() == tflite::Padding_SAME ? "SAME_UPPER" : "VALID", ctx->attr_pool);
            int64_t strides[2] = {co.stride_h(), co.stride_w()};
            set_attr_ints(attrs[1], attr_key_t::strides, strides, 2, ctx->attr_pool);
            int64_t dilations[2] = {co.dilation_h(), co.dilation_w()};
            set_attr_ints(attrs[2], attr_key_t::dilations, dilations, 2, ctx->attr_pool);
            set_attr_int(attrs[3], attr_key_t::group, 1);
            // After OHWI→OIHW transpose, weight is [OC,IC,KH,KW]
            if (n_in >= 2 && n->inputs[1] && n->inputs[1]->ndim == 4) {
                int64_t ks[2] = {n->inputs[1]->dims[2], n->inputs[1]->dims[3]};
                set_attr_ints(attrs[4], attr_key_t::kernel_shape, ks, 2, ctx->attr_pool);
            }
            n->attrs = {attrs, 5};

            // TFLite Conv2D has [input, weight, bias] — same as ONNX Conv
            break;
        }
        case tflite::BuiltinOperator_DEPTHWISE_CONV_2D: {
            tflite::DepthwiseConv2DOptions dco(opts);
            fused_act = dco.fused_activation();

            // Transpose weights: [1, KH, KW, OC] → [OC, 1, KH, KW]
            if (n_in >= 2 && n->inputs[1]) {
                transpose_dw_weights_to_oihw(n->inputs[1]);
                int32_t widx = f_inputs[1];
                if (widx >= 0 && widx < (int32_t)n_tensors) already_transposed[widx] = true;
            }

            // Determine group count: OC / depth_multiplier = IC = group
            int group = 1;
            if (n_in >= 2 && n->inputs[1]) {
                // After transpose, weights are [OC, 1, KH, KW]
                int OC = n->inputs[1]->dims[0];
                int dm = dco.depth_multiplier();
                if (dm > 0) group = OC / dm;
                if (group <= 0) group = OC; // fallback: pure depthwise
            }

            auto* attrs = alloc_attrs(ctx->attr_pool, 5);
            set_attr_string(attrs[0], attr_key_t::auto_pad,
                dco.padding() == tflite::Padding_SAME ? "SAME_UPPER" : "VALID", ctx->attr_pool);
            int64_t strides[2] = {dco.stride_h(), dco.stride_w()};
            set_attr_ints(attrs[1], attr_key_t::strides, strides, 2, ctx->attr_pool);
            int64_t dilations[2] = {dco.dilation_h(), dco.dilation_w()};
            set_attr_ints(attrs[2], attr_key_t::dilations, dilations, 2, ctx->attr_pool);
            set_attr_int(attrs[3], attr_key_t::group, (int64_t)group);
            // After transpose, weight is [OC,1,KH,KW]
            if (n_in >= 2 && n->inputs[1] && n->inputs[1]->ndim == 4) {
                int64_t ks[2] = {n->inputs[1]->dims[2], n->inputs[1]->dims[3]};
                set_attr_ints(attrs[4], attr_key_t::kernel_shape, ks, 2, ctx->attr_pool);
            }
            n->attrs = {attrs, 5};
            break;
        }
        case tflite::BuiltinOperator_AVERAGE_POOL_2D:
        case tflite::BuiltinOperator_MAX_POOL_2D: {
            tflite::Pool2DOptions po(opts);
            fused_act = po.fused_activation();

            auto* attrs = alloc_attrs(ctx->attr_pool, 4);
            set_attr_string(attrs[0], attr_key_t::auto_pad,
                po.padding() == tflite::Padding_SAME ? "SAME_UPPER" : "VALID", ctx->attr_pool);
            int64_t strides[2] = {po.stride_h(), po.stride_w()};
            set_attr_ints(attrs[1], attr_key_t::strides, strides, 2, ctx->attr_pool);
            int64_t kernel[2] = {po.filter_height(), po.filter_width()};
            set_attr_ints(attrs[2], attr_key_t::kernel_shape, kernel, 2, ctx->attr_pool);
            if (builtin_code == tflite::BuiltinOperator_AVERAGE_POOL_2D)
                set_attr_int(attrs[3], attr_key_t::count_include_pad, 0);
            else
                set_attr_int(attrs[3], attr_key_t::ceil_mode, 0);
            n->attrs = {attrs, 4};
            break;
        }
        case tflite::BuiltinOperator_FULLY_CONNECTED: {
            tflite::FullyConnectedOptions fco(opts);
            fused_act = fco.fused_activation();

            // TFLite FC: output = input × weight^T + bias
            // ONNX Gemm: Y = alpha*A*B + beta*C, with transB=1
            // TFLite weight shape: [OC, IC] — already matches ONNX Gemm with transB=1
            auto* attrs = alloc_attrs(ctx->attr_pool, 2);
            set_attr_int(attrs[0], attr_key_t::transB, 1);
            set_attr_float(attrs[1], attr_key_t::alpha, 1.0f);
            n->attrs = {attrs, 2};

            // FC in TFLite: inputs = [input(2D or ND), weights, bias]
            // For ND input, we may need to flatten. The Gemm op handles 2D.
            // If input is > 2D, reshape is needed — for now trust the operator.
            break;
        }
        case tflite::BuiltinOperator_CONCATENATION: {
            tflite::ConcatenationOptions co(opts);
            fused_act = co.fused_activation();
            int out_ndim = (n_out > 0 && n->outputs[0]) ? n->outputs[0]->ndim : 0;
            auto* attrs = alloc_attrs(ctx->attr_pool, 1);
            set_attr_int(attrs[0], attr_key_t::axis, nhwc_to_nchw_axis(co.axis(), out_ndim));
            n->attrs = {attrs, 1};
            break;
        }
        case tflite::BuiltinOperator_SOFTMAX: {
            tflite::SoftmaxOptions so(opts);
            (void)so; // beta is typically 1.0, ONNX Softmax doesn't have beta
            auto* attrs = alloc_attrs(ctx->attr_pool, 1);
            set_attr_int(attrs[0], attr_key_t::axis, -1);
            n->attrs = {attrs, 1};
            break;
        }
        case tflite::BuiltinOperator_RESHAPE: {
            // Reshape: ONNX takes shape as second input tensor (INT64).
            // TFLite may store target shape in ReshapeOptions or as second input (INT32).
            // Always create an INT64 shape tensor for ONNX compatibility.
            int ndims = 0;
            small_vector<int64_t> shape_vals;

            // Try second input first
            if (n_in >= 2 && n->inputs[1] && n->inputs[1]->ndata > 0) {
                tensor_t* st = n->inputs[1];
                size_t st_n = st->ndata;
                // Reject rather than silently truncate: a shape vector longer
                // than MAX_NDIM cannot be represented by any downstream tensor
                // and was the vector for F-ADV-002 (OOB write into the
                // small_vector's int8_t-sized inline storage).
                if (st_n > (size_t)MAX_NDIM) return false;
                ndims = (int)st_n;
                shape_vals = small_vector<int64_t>(ndims);
                if (st->type == NNR_DATA_TYPE_INT32) {
                    int32_t* sd = (int32_t*)st->data;
                    for (int d = 0; d < ndims; ++d) shape_vals[d] = sd[d];
                } else if (st->type == NNR_DATA_TYPE_INT64) {
                    int64_t* sd = (int64_t*)st->data;
                    for (int d = 0; d < ndims; ++d) shape_vals[d] = sd[d];
                }
            }
            // Fallback to ReshapeOptions
            if (ndims == 0 && opts) {
                tflite::ReshapeOptions ro(opts);
                auto ns = ro.new_shape();
                if (ns.size() > (size_t)MAX_NDIM) return false;
                ndims = (int)ns.size();
                shape_vals = small_vector<int64_t>(ndims);
                for (int d = 0; d < ndims; ++d) shape_vals[d] = ns[d];
            }
            if (ndims > 0) {
                int shape_dims[1] = {ndims};
                std::string sname = std::string(n->outputs[0]->name) + "_shape";
                size_t slen = sname.size();
                char* sbuf = (char*)ctx->attr_pool.alloc(slen + 1, 1);
                memcpy(sbuf, sname.data(), slen + 1);
                std::string_view sv{sbuf, slen};

                tensor_t* shape_t = new (std::nothrow) tensor_t(sv, NNR_DATA_TYPE_INT64,
                    std::span<const int>(shape_dims, 1));
                int64_t* sdata = (int64_t*)shape_t->data;
                for (int d = 0; d < ndims; ++d)
                    sdata[d] = shape_vals[d];

                ctx->map.emplace_back(sv, shape_t);
                ctx->initializer_names.insert(sv);

                tensor_t** ibuf2 = ctx->attr_pool.alloc_arr<tensor_t*>(2);
                ibuf2[0] = n->inputs.size() > 0 ? n->inputs[0] : nullptr;
                ibuf2[1] = shape_t;
                n->inputs = {ibuf2, 2};
            }
            break;
        }
        case tflite::BuiltinOperator_ADD:
        case tflite::BuiltinOperator_SUB:
        case tflite::BuiltinOperator_MUL:
        case tflite::BuiltinOperator_DIV: {
            if (opts) {
                tflite::ArithOptions ao(opts);
                fused_act = ao.fused_activation();
            }
            break;
        }
        case tflite::BuiltinOperator_MEAN: {
            // ReduceMean: opset 13 reads axes from attribute, not input[1].
            // Extract axes from TFLite input[1] constant tensor, convert NHWC→NCHW.
            int keepdims = 0;
            if (opts) {
                tflite::ReducerOptions ro(opts);
                keepdims = ro.keep_dims() ? 1 : 0;
            }
            int n_axes = 0;
            int64_t axes_nchw[MAX_NDIM] = {};
            if (n_in >= 2 && n->inputs[1] && n->inputs[1]->data) {
                const tensor_t* axes_t = n->inputs[1];
                // Cap axes count against the fixed-size stack buffer. Any
                // legitimate MEAN over >MAX_NDIM axes cannot map onto a
                // tensor_t in the first place, so reject hard.
                if (axes_t->ndata > (size_t)MAX_NDIM) return false;
                n_axes = (int)axes_t->ndata;
                const int32_t* axes_data = (const int32_t*)axes_t->data;
                int in_ndim = (n_in >= 1 && n->inputs[0]) ? n->inputs[0]->ndim : 4;
                for (int i = 0; i < n_axes; ++i)
                    axes_nchw[i] = nhwc_to_nchw_axis(axes_data[i], in_ndim);
            }
            auto* attrs = alloc_attrs(ctx->attr_pool, 2);
            set_attr_int(attrs[0], attr_key_t::keepdims, keepdims);
            if (n_axes > 0)
                set_attr_ints(attrs[1], attr_key_t::axes, axes_nchw, n_axes, ctx->attr_pool);
            n->attrs = {attrs, (size_t)(n_axes > 0 ? 2 : 1)};
            // Null out axes input — opset 13 reads from attribute
            if (n_in >= 2) n->inputs[1] = nullptr;
            break;
        }
        case tflite::BuiltinOperator_RESIZE_BILINEAR: {
            if (opts) {
                tflite::ResizeBilinearOptions ro(opts);
                auto* attrs = alloc_attrs(ctx->attr_pool, 3);
                set_attr_string(attrs[0], attr_key_t::mode, "linear", ctx->attr_pool);
                if (ro.align_corners())
                    set_attr_string(attrs[1], attr_key_t::coordinate_transformation_mode,
                        "align_corners", ctx->attr_pool);
                else if (ro.half_pixel_centers())
                    set_attr_string(attrs[1], attr_key_t::coordinate_transformation_mode,
                        "half_pixel", ctx->attr_pool);
                else
                    set_attr_string(attrs[1], attr_key_t::coordinate_transformation_mode,
                        "asymmetric", ctx->attr_pool);
                set_attr_string(attrs[2], attr_key_t::nearest_mode, "floor", ctx->attr_pool);
                n->attrs = {attrs, 3};
            }
            // TFLite Resize input[1] is [new_height, new_width] as int32 tensor
            // ONNX Resize opset 11+ takes: X, roi, scales, sizes
            // We need to rearrange inputs: [X, sizes] → [X, empty_roi, empty_scales, sizes]
            if (n_in >= 2 && n->inputs[1]) {
                tensor_t** ibuf3 = ctx->attr_pool.alloc_arr<tensor_t*>(4);
                ibuf3[0] = n->inputs[0]; // X
                ibuf3[1] = nullptr;       // roi (empty)
                ibuf3[2] = nullptr;       // scales (empty)
                ibuf3[3] = n->inputs[1]; // sizes
                n->inputs = {ibuf3, 4};
            }
            break;
        }
        case tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: {
            if (opts) {
                tflite::ResizeNNOptions ro(opts);
                auto* attrs = alloc_attrs(ctx->attr_pool, 3);
                set_attr_string(attrs[0], attr_key_t::mode, "nearest", ctx->attr_pool);
                if (ro.align_corners())
                    set_attr_string(attrs[1], attr_key_t::coordinate_transformation_mode,
                        "align_corners", ctx->attr_pool);
                else if (ro.half_pixel_centers())
                    set_attr_string(attrs[1], attr_key_t::coordinate_transformation_mode,
                        "half_pixel", ctx->attr_pool);
                else
                    set_attr_string(attrs[1], attr_key_t::coordinate_transformation_mode,
                        "asymmetric", ctx->attr_pool);
                set_attr_string(attrs[2], attr_key_t::nearest_mode, "floor", ctx->attr_pool);
                n->attrs = {attrs, 3};
            }
            if (n_in >= 2 && n->inputs[1]) {
                tensor_t** ibuf3 = ctx->attr_pool.alloc_arr<tensor_t*>(4);
                ibuf3[0] = n->inputs[0];
                ibuf3[1] = nullptr;
                ibuf3[2] = nullptr;
                ibuf3[3] = n->inputs[1];
                n->inputs = {ibuf3, 4};
            }
            break;
        }
        case tflite::BuiltinOperator_GATHER: {
            if (opts) {
                tflite::GatherOptions go(opts);
                auto* attrs = alloc_attrs(ctx->attr_pool, 1);
                set_attr_int(attrs[0], attr_key_t::axis, go.axis());
                n->attrs = {attrs, 1};
            }
            break;
        }
        case tflite::BuiltinOperator_LEAKY_RELU: {
            if (opts) {
                tflite::LeakyReluOptions lr(opts);
                auto* attrs = alloc_attrs(ctx->attr_pool, 1);
                set_attr_float(attrs[0], attr_key_t::alpha, lr.alpha());
                n->attrs = {attrs, 1};
            }
            break;
        }
        case tflite::BuiltinOperator_BATCH_MATMUL: {
            if (opts) {
                tflite::BatchMatMulOptions bm(opts);
                auto* attrs = alloc_attrs(ctx->attr_pool, 2);
                set_attr_int(attrs[0], attr_key_t::transA, bm.adj_x() ? 1 : 0);
                set_attr_int(attrs[1], attr_key_t::transB, bm.adj_y() ? 1 : 0);
                n->attrs = {attrs, 2};
            }
            break;
        }
        case tflite::BuiltinOperator_PACK: {
            // TFLite Pack = Concat along new axis (stack)
            // This needs Unsqueeze on each input then Concat, but we approximate as Concat
            if (opts) {
                tflite::PackOptions po(opts);
                auto* attrs = alloc_attrs(ctx->attr_pool, 2);
                set_attr_int(attrs[0], attr_key_t::axis, po.axis());
                set_attr_int(attrs[1], attr_key_t::new_axis, 1);
                n->attrs = {attrs, 2};
            }
            break;
        }
        case tflite::BuiltinOperator_SPLIT: {
            // TFLite Split: inputs = [axis_tensor, input], outputs = N outputs
            // ONNX Split: input = [input, split_sizes?], attr axis
            // Rearrange inputs: swap input[0] and input[1]
            if (n_in >= 2) {
                // TFLite: [axis, data] → ONNX: [data]
                // Read axis from tensor data
                int64_t axis_val = 0;
                if (n->inputs[0] && n->inputs[0]->data) {
                    if (n->inputs[0]->type == NNR_DATA_TYPE_INT32)
                        axis_val = *(int32_t*)n->inputs[0]->data;
                    else if (n->inputs[0]->type == NNR_DATA_TYPE_INT64)
                        axis_val = *(int64_t*)n->inputs[0]->data;
                }
                auto* attrs = alloc_attrs(ctx->attr_pool, 1);
                set_attr_int(attrs[0], attr_key_t::axis, axis_val);
                n->attrs = {attrs, 1};

                // Remap inputs: just the data tensor
                tensor_t** ibuf1 = ctx->attr_pool.alloc_arr<tensor_t*>(1);
                ibuf1[0] = n->inputs[1];
                n->inputs = {ibuf1, 1};
            }
            break;
        }
        case tflite::BuiltinOperator_EXPAND_DIMS: {
            // Unsqueeze: ONNX opset 13 takes axes as input[1]
            // TFLite ExpandDims: input[0]=data, input[1]=axis (scalar tensor)
            // This maps directly if we name the op correctly (already mapped to Unsqueeze)
            // ONNX Unsqueeze opset 13: inputs = [data, axes_tensor]
            break;
        }
        case tflite::BuiltinOperator_SQUEEZE: {
            // TFLite Squeeze options may have squeeze_dims
            // For now, rely on ONNX Squeeze defaults (all size-1 dims)
            break;
        }
        case tflite::BuiltinOperator_RELU6: {
            // Already mapped to "Clip" — need min=0, max=6 as inputs
            // Create constant tensors
            tensor_t* t_min = new (std::nothrow) tensor_t("", NNR_DATA_TYPE_FLOAT32, {});
            if (!t_min || t_min->allocation_failed) { delete t_min; break; }
            *(float*)t_min->data = 0.0f;
            std::string mn_s = "relu6_min_" + std::to_string(oi);
            char* mn = (char*)ctx->attr_pool.alloc(mn_s.size() + 1, 1);
            memcpy(mn, mn_s.data(), mn_s.size() + 1);
            t_min->name = {mn, mn_s.size()};
            ctx->map.emplace_back(t_min->name, t_min);
            ctx->initializer_names.insert(t_min->name);

            tensor_t* t_max = new (std::nothrow) tensor_t("", NNR_DATA_TYPE_FLOAT32, {});
            if (!t_max || t_max->allocation_failed) { delete t_max; break; }
            *(float*)t_max->data = 6.0f;
            std::string xn_s = "relu6_max_" + std::to_string(oi);
            char* xn = (char*)ctx->attr_pool.alloc(xn_s.size() + 1, 1);
            memcpy(xn, xn_s.data(), xn_s.size() + 1);
            t_max->name = {xn, xn_s.size()};
            ctx->map.emplace_back(t_max->name, t_max);
            ctx->initializer_names.insert(t_max->name);

            tensor_t** ibuf3 = ctx->attr_pool.alloc_arr<tensor_t*>(3);
            ibuf3[0] = n->inputs.size() > 0 ? n->inputs[0] : nullptr;
            ibuf3[1] = t_min;
            ibuf3[2] = t_max;
            n->inputs = {ibuf3, 3};
            break;
        }
        default:
            // No special attribute handling needed
            break;
        }

    return fused_act;
}

bool load_tflite(context_t* ctx, const void* data, size_t size)
{
    if (!ctx || !data || size < 8) return false;

    const uint8_t* buf = (const uint8_t*)data;

    // Verify FlatBuffer identifier
    if (memcmp(buf + 4, "TFL3", 4) != 0) return false;

    if (ctx->mmap_data_) {
        // Data is mmap'd — alias directly, no copy needed.
        // mmap lifetime is managed by context_t.
    } else {
        // Data from user buffer — must copy for ownership.
        uint8_t* owned = new (std::nothrow) uint8_t[size];
        if (!owned) return false;
        memcpy(owned, buf, size);
        ctx->set_model_handle(owned, [](void* p) { delete[] (uint8_t*)p; });
        buf = owned;
    }

    // Create bounds context for all FlatBuffer accessors (stack-local, reentrant)
    tflite::fb_ctx fbc(buf, size);

    tflite::Model model(buf, fbc);
    if (model.version() < 3) {
        fprintf(stderr, "nnr: unsupported TFLite schema version %d\n", model.version());
        return false;
    }

    ctx->meta_producer_name = "tflite";
    if (auto desc = model.description(); !desc.empty())
        ctx->meta_domain = desc;

    // Use opset 13 as default — covers most ONNX operators we map to
    const int default_opset = 13;
    ctx->meta_opsets.emplace_back("ai.onnx", (int64_t)default_opset);

    // Load operator codes
    auto opcodes = model.operator_codes();
    uint32_t n_opcodes = opcodes.size();

    // Load buffers
    auto buffers = model.buffers();

    // Process first subgraph only (TFLite models typically have one)
    auto subgraphs = model.subgraphs();
    if (subgraphs.size() == 0) return false;

    tflite::SubGraph sg(subgraphs[0]);
    auto fl_tensors = sg.tensors();
    auto fl_inputs  = sg.inputs();
    auto fl_outputs = sg.outputs();
    auto fl_ops     = sg.operators();

    uint32_t n_tensors = fl_tensors.size();
    uint32_t n_ops     = fl_ops.size();
    // Reject unreasonable counts that could cause DoS via huge allocations
    if (n_tensors > 10'000'000 || n_ops > 10'000'000) return false;
    // Phase 1: Create all tensors and copy weight data
    // Build a mapping from TFLite tensor index → tensor_t*
    auto tensor_map = std::make_unique<tensor_t*[]>(n_tensors);
    // Track tensors whose layout was already transposed in Phase 2 (conv weights)
    auto already_transposed = std::make_unique<bool[]>(n_tensors);

    for (uint32_t i = 0; i < n_tensors; ++i) {
        tflite::Tensor ft(fl_tensors[i]);

        // Name
        std::string_view name = intern_name(ctx->attr_pool, ft.name(), (int)i);

        // Type
        data_type_t dtype = tflite::tflite_type_to_nnr(ft.type());

        // Shape (TFLite stores NHWC; we'll convert to NCHW for 4D tensors later)
        auto fshape = ft.shape();
        uint32_t ndim = fshape.size();
        // Reject shapes that would overflow the fixed-size dims[MAX_NDIM] /
        // strides[MAX_NDIM] members of tensor_t. Without this guard, an
        // attacker-crafted TFLite file can drive an OOB write into
        // small_vector's inline storage and the tensor_t stack object.
        if (ndim > (uint32_t)MAX_NDIM) return false;
        small_vector<int> dims((int)ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            int dim_val = fshape[d];
            dims[d] = (dim_val < 0) ? 0 : dim_val; // -1 → 0 (dynamic)
        }

        tensor_t* t = new (std::nothrow) tensor_t(name, dtype, dims);
        if (!t) return false;

        // Copy buffer data (weights/constants)
        uint32_t buf_idx = ft.buffer();
        if (buf_idx > 0 && buf_idx < buffers.size()) {
            tflite::Buffer fb(buffers[buf_idx]);
            auto bd = fb.data();
            if (bd.size() > 0) {
                size_t elem_sz = data_type_sizeof(dtype);
                if (elem_sz > 0 && t->data && t->ndata > 0) {
                    // Overflow-safe: check before multiplying
                    size_t expected = (t->ndata <= SIZE_MAX / elem_sz) ? t->ndata * elem_sz : SIZE_MAX;
                    size_t avail = bd.size();
                    memcpy(t->data, bd.data(), std::min(expected, avail));
                }
                ctx->initializer_names.insert(name);
            }
        }

        tensor_map[i] = t;
        ctx->map.emplace_back(name, t);
    }

    // Phase 1b: Determine which tensors need NHWC→NCHW transposition.
    // We need to know which 4D tensors are used as spatial activations (not weights).
    // Strategy: weights are initializers. 4D activation tensors in NHWC need transposing.
    // Weight tensors for Conv/DepthwiseConv have their own transpose (OHWI→OIHW).
    // We'll mark which weight tensors need special treatment per-operator in phase 2.

    // Track which tensors are Conv/DW weight tensors (need special transpose)
    // We'll handle them during operator building.

    // Graph I/O
    for (uint32_t i = 0; i < fl_inputs.size(); ++i) {
        int32_t idx = fl_inputs[i];
        if (idx >= 0 && idx < (int32_t)n_tensors && tensor_map[idx]) {
            ctx->graph_inputs.push_back(tensor_map[idx]->name);
            // Don't exclude graph inputs from memory planning unless they're initializers
            if (ctx->initializer_names.count(tensor_map[idx]->name))
                ctx->memory_plan_excluded.insert(tensor_map[idx]->name);
        }
    }
    for (uint32_t i = 0; i < fl_outputs.size(); ++i) {
        int32_t idx = fl_outputs[i];
        if (idx >= 0 && idx < (int32_t)n_tensors && tensor_map[idx]) {
            ctx->graph_outputs.push_back(tensor_map[idx]->name);
            ctx->memory_plan_excluded.insert(tensor_map[idx]->name);
        }
    }
    // Exclude initializers from memory planning
    for (auto& name : ctx->initializer_names)
        ctx->memory_plan_excluded.insert(name);

    // Phase 2: Build operators
    ctx->graph = std::make_unique<graph_t>();
    auto& nodes = ctx->graph->nodes;
    nodes.reserve(n_ops * 2); // extra room for fused activations

    for (uint32_t oi = 0; oi < n_ops; ++oi) {
        tflite::Operator fop(fl_ops[oi]);

        // Resolve opcode
        uint32_t oc_idx = fop.opcode_index();
        if (oc_idx >= n_opcodes) {
            fprintf(stderr, "nnr: tflite op %u has invalid opcode index %u\n", oi, oc_idx);
            continue;
        }
        tflite::OperatorCode oc(opcodes[oc_idx]);
        int32_t builtin_code = oc.builtin_code();
        const char* onnx_op = tflite::builtin_to_onnx(builtin_code);
        if (!onnx_op) {
            fprintf(stderr, "nnr: unsupported TFLite op: builtin_code=%d", builtin_code);
            if (auto cc = oc.custom_code(); !cc.empty())
                fprintf(stderr, " custom=%.*s", (int)cc.size(), cc.data());
            fprintf(stderr, "\n");
            continue;
        }

        // Resolve operator
        operator_t* n = solve_operator(onnx_op, default_opset, ctx->attr_pool,
                                       static_cast<backend_t>(ctx->preferred_backend));
        struct op_dummy : public operator_t { bool exec() override { return false; } };
        if (!n) n = pool_new<op_dummy>(ctx->attr_pool);

        n->ctx      = ctx;
        n->opset    = default_opset;
        n->op_type  = onnx_op;
        n->domain   = "ai.onnx";

        // Inputs
        auto f_inputs = fop.inputs();
        uint32_t n_in = f_inputs.size();
        if (n_in > 0) {
            tensor_t** ibuf = ctx->attr_pool.alloc_arr<tensor_t*>(n_in);
            for (uint32_t j = 0; j < n_in; ++j) {
                int32_t tidx = f_inputs[j];
                ibuf[j] = (tidx >= 0 && tidx < (int32_t)n_tensors) ? tensor_map[tidx] : nullptr;
            }
            n->inputs = {ibuf, n_in};
        }

        // Outputs
        auto f_outputs = fop.outputs();
        uint32_t n_out = f_outputs.size();
        if (n_out > 0) {
            tensor_t** obuf = ctx->attr_pool.alloc_arr<tensor_t*>(n_out);
            for (uint32_t j = 0; j < n_out; ++j) {
                int32_t tidx = f_outputs[j];
                obuf[j] = (tidx >= 0 && tidx < (int32_t)n_tensors) ? tensor_map[tidx] : nullptr;
            }
            n->outputs = {obuf, n_out};
        }

        // --- Per-op attribute extraction and weight transposition ---
        int8_t fused_act = build_tflite_op_attrs(
            ctx, n, fop, builtin_code, n_in, n_out, oi,
            f_inputs, n_tensors, tensor_map.get(), already_transposed.get());


        if (!n->init())
            fprintf(stderr, "nnr: tflite op init failed: %s (builtin_code=%d)\n",
                    onnx_op, builtin_code);

        nodes.push_back(n);

        // Insert fused activation if present
        if (fused_act != tflite::ActivationFunctionType_NONE && n_out > 0) {
            int32_t out_idx = f_outputs[0];
            if (out_idx >= 0 && out_idx < (int32_t)n_tensors)
                insert_fused_activation(ctx, ctx->graph.get(), fused_act,
                                        tensor_map[out_idx], default_opset);
        }
    }

    // Phase 3: Transpose 4D activation tensors from NHWC to NCHW.
    // Only transpose tensors that have buffer data (constants) and are 4D.
    // Non-constant 4D tensors: reshape dims from [N,H,W,C] to [N,C,H,W]
    // so the NCHW operators see the right shape. Actual data transposition
    // happens at runtime via the NHWC→NCHW reorder in context_t::run().

    for (uint32_t i = 0; i < n_tensors; ++i) {
        tensor_t* t = tensor_map[i];
        if (!t || t->ndim != 4) continue;
        if (already_transposed[i]) continue; // Conv/DW weights already in OIHW

        bool is_init = ctx->initializer_names.count(t->name) > 0;

        if (is_init) {
            transpose_nhwc_to_nchw(t);
        } else {
            int N = t->dims[0], H = t->dims[1], W = t->dims[2], C = t->dims[3];
            t->dims[0] = N; t->dims[1] = C; t->dims[2] = H; t->dims[3] = W;
            t->strides[3] = 1;
            t->strides[2] = W;
            t->strides[1] = H * W;
            t->strides[0] = C * H * W;
            t->ndata = (size_t)N * C * H * W;
        }
    }

    return true;
}

} // namespace nnr
