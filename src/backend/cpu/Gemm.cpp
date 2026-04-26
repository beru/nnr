#include "nnr.h"
#include "aligned_alloc.h"
#include "util.h"
#include "allocator.h"
#include "kernel/gemm.h"
#include "kernel/f16_convert.h"
#ifdef NNR_ARCH_X64
#include "backend/x64/vec_ops_avx512.h"
#include "backend/x64/vec_ops_avx2.h"
#include "backend/x64/gemm_bf16_avx512.h"
#endif

namespace nnr {

namespace {

struct Gemm_operator : public operator_t {
    float alpha;
    float beta;
    int transA;
    int transB;

    int m = 0;
    int n = 0;
    int k = 0;

    std::vector<float> b_packed;  // pre-packed B weights (float32)
    std::vector<uint16_t> b_packed_bf16;  // VNNI-packed BF16 weights for VDPBF16PS
    std::vector<uint16_t> b_packed_fp16;  // K-major FP16-packed weights for gemm_fp16_neon

    bool init() override {
        if (!(inputs.size() >= 2 && outputs.size() == 1)) {
            return false;
        }
        alpha = attribute(attr_key_t::alpha, 1.0f);
        beta = attribute(attr_key_t::beta, 1.0f);
        transA = attribute(attr_key_t::transA, 0);
        transB = attribute(attr_key_t::transB, 0);
        return true;
    }

    bool reshape() override {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        int lk;

        if (transA) {
            m = a->dims[1];
            k = a->dims[0];
        }else {
            m = a->dims[0];
            k = a->dims[1];
        }
        if (transB) {
            n = b->dims[0];
            lk = 1;
        }else {
            n = b->dims[1];
            lk = 0;
        }
        if (b->dims[lk] != k) {
            return false;
        }
        if (m <= 0 || n <= 0 || k <= 0) {
            return false;
        }
        int tmp[2] = { m, n };
        if ((inputs.size() > 2) && !inputs[2]->broadcast_is_valid(tmp)) {
            return false;
        }

        // Pre-pack B weights for constant tensors (float32 or float16)
        b_packed.clear();
        if ((b->type == NNR_DATA_TYPE_FLOAT32 || b->type == NNR_DATA_TYPE_FLOAT16)
            && ctx && ctx->initializer_names.count(b->name)) {
            size_t psz = pack_b_size(k, n);
            if (psz > 0) {
                // Get float32 view of B (convert FP16 → F32 if needed)
                size_t bn = (size_t)k * n;
                float* tmp = nullptr;
                const float* pb;
                if (b->type == NNR_DATA_TYPE_FLOAT16) {
                    tmp = (float*)nnr_aligned_alloc(bn * sizeof(float), 64);
                    convert_f16_to_f32(tmp, (const float16_t*)b->data, bn);
                    pb = tmp;
                } else {
                    pb = (const float*)b->data;
                }
                b_packed.resize(psz);
                if (transB) {
                    // B is [n × k], transpose to [k × n] then pack
                    float* bt = (float*)nnr_aligned_alloc(bn * sizeof(float), 64);
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < k; j++)
                            bt[(size_t)j * n + i] = pb[(size_t)i * k + j];
                    pack_b(b_packed.data(), bt, k, n);
                    nnr_aligned_free(bt);
                } else {
                    // B is [k × n], pack directly
                    pack_b(b_packed.data(), pb, k, n);
                }
                nnr_aligned_free(tmp);
            }
        }

        // Pre-pack FP16 weights in the gemm_fp16_neon K-major layout.
        // Allows the FP16 fast path in exec_f16_as_f32() to skip both the
        // B FP16→FP32 conversion and the FP32 pack on every call.
        b_packed_fp16.clear();
        if (b->type == NNR_DATA_TYPE_FLOAT16 && !transA
            && ctx && ctx->initializer_names.count(b->name)) {
            size_t psz = pack_b_fp16_size(k, n);
            if (psz > 0) {
                const uint16_t* pb = (const uint16_t*)b->data;
                b_packed_fp16.resize(psz / sizeof(uint16_t));
                if (transB) {
                    size_t bn = (size_t)k * n;
                    uint16_t* bt = (uint16_t*)_aligned_malloc(bn * sizeof(uint16_t), 64);
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < k; j++)
                            bt[(size_t)j * n + i] = pb[(size_t)i * k + j];
                    pack_b_fp16(b_packed_fp16.data(), bt, k, n);
                    _aligned_free(bt);
                } else {
                    pack_b_fp16(b_packed_fp16.data(), pb, k, n);
                }
            }
        }

        // Pre-pack BF16 weights into VNNI format for VDPBF16PS
        b_packed_bf16.clear();
#ifdef NNR_ARCH_X64
        if (b->type == NNR_DATA_TYPE_BFLOAT16 && has_avx512_bf16()
            && ctx && ctx->initializer_names.count(b->name)) {
            const uint16_t* pb = (const uint16_t*)b->data;
            if (transB) {
                // B is [n × k], transpose to [k × n] then pack
                uint16_t* bt = (uint16_t*)nnr_aligned_alloc((size_t)k * n * sizeof(uint16_t), 64);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < k; j++)
                        bt[(size_t)j * n + i] = pb[(size_t)i * k + j];
                size_t psz = bf16::pack_b_bf16_size(k, n);
                b_packed_bf16.resize(psz);
                bf16::pack_b_bf16(b_packed_bf16.data(), bt, k, n);
                nnr_aligned_free(bt);
            } else {
                size_t psz = bf16::pack_b_bf16_size(k, n);
                b_packed_bf16.resize(psz);
                bf16::pack_b_bf16(b_packed_bf16.data(), pb, k, n);
            }
        }
#endif

        return y->reshape(tmp, a->type);
    }

    int64_t num_ops() const override {
        return (int64_t)2 * m * n * k;
    }

    template <typename T>
    bool exec() {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        const tensor_t* c = (inputs.size() > 2) ? inputs[2] : nullptr;
        T* py = (T*)y->data;
        const T* pa = (const T*)a->data;
        const T* pb = (const T*)b->data;

        // Apply fused post-op to output (after alpha*A*B + beta*C)
        auto apply_post_op = [&]() {
            if constexpr (std::is_same_v<T, float>) {
                if (post_fn)
                    post_fn(py, 1, m * n, m * n, fused_op, nullptr, 0);
            }
        };

        // Pre-packed B path: handles !transA with either transB setting
        // (B was transposed and packed in reshape if transB was set)
        if constexpr (std::is_same_v<T, float>) {
            if (!b_packed.empty() && !transA) {
                // Fast path: fuse column-wise bias + post-op into GEMM tile loop
                bool fused = false;
                if (alpha == 1.0f) {
                    const float* col_bias = nullptr;
                    bool can_fuse = true;
                    if (c) {
                        if (beta == 1.0f && c->ndim == 1 && c->dims[0] == n)
                            col_bias = (const float*)c->data;
                        else
                            can_fuse = false;
                    }
                    if (can_fuse) {
                        gemm_post_nhwc_t gp;
                        gp.bias = col_bias;
                        gp.c_base = py;
                        gp.post_fn = post_fn;
                        gp.fused_op = fused_op;
                        gp.classify();
                        dgemm_packed_b(m, n, k, pa, b_packed.data(), py, gp);
                        fused = true;
                    }
                }
                if (!fused) {
                    // General case: packed GEMM + separate alpha/beta/post
                    dgemm_packed_b(m, n, k, pa, b_packed.data(), py);
                    if (alpha != 1.0f) {
                        float av = alpha;
                        for (int i = 0; i < m * n; ++i)
                            py[i] *= av;
                    }
                    if (c) {
                        for (int i = 0; i < m; ++i)
                            for (int j = 0; j < n; ++j) {
                                int oy = i * n + j;
                                py[oy] += (float)(beta * *(const float*)c->broadcast_map_address(y, oy));
                            }
                    }
                    apply_post_op();
                }
                return true;
            }
        }

        // Accumulate alpha * A * B using i-k-j order where beneficial
        if (!transA && !transB) {
            // Use shared GEMM kernel: C = A × B
            dgemm_generic(m, n, k, pa, pb, py);

            // Apply alpha scaling (skip if alpha == 1.0)
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                if (alpha != 1.0f) {
                    T av = (T)alpha;
                    for (int i = 0; i < m * n; ++i)
                        py[i] *= av;
                }
            }

            // Add beta * C (broadcast)
            if (c) {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        int oy = i * n + j;
                        py[oy] += (T)(beta * *(const T*)c->broadcast_map_address(y, oy));
                    }
            }

            apply_post_op();
            return true;
        }

        // Transpose cases: initialize output with beta * C or zero, then accumulate
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                int oy = i * n + j;
                py[oy] = c ? (T)(beta * *(const T*)c->broadcast_map_address(y, oy)) : T(0);
            }

        if (transA && !transB) {
            // A is k*m, A[w][i] = pa[w*m+i], B[w][j] = pb[w*n+j]
            for (int i = 0; i < m; ++i)
                for (int w = 0; w < k; ++w) {
                    T a_iw = (T)(alpha * pa[w * m + i]);
                    for (int j = 0; j < n; ++j)
                        py[i * n + j] += a_iw * pb[w * n + j];
                }
        } else if (!transA && transB) {
            // A[i][w] = pa[i*k+w], B is n*k, B[j][w] = pb[j*k+w]: w-loop sequential
            {
                bool done = false;
#ifdef NNR_ARCH_X64
                if constexpr (std::is_same_v<T, float>) {
                    if (has_avx512()) {
                        // Both A[i,:] and B[j,:] are contiguous length-k vectors â dot product
                        nnr::for_static(0, m, (int64_t)m * n > 64, [&](int i) {
                            const float* pa_row = pa + (size_t)i * k;
                            float* py_row = py + (size_t)i * n;
                            for (int j = 0; j < n; ++j) {
                                const float* pb_row = pb + (size_t)j * k;
                                py_row[j] += alpha * avx512::dot_product(pa_row, pb_row, k);
                            }
                        });
                        done = true;
                    } else if (detect_isa() == isa_t::avx2) {
                        nnr::for_static(0, m, (int64_t)m * n > 64, [&](int i) {
                            const float* pa_row = pa + (size_t)i * k;
                            float* py_row = py + (size_t)i * n;
                            for (int j = 0; j < n; ++j) {
                                const float* pb_row = pb + (size_t)j * k;
                                py_row[j] += alpha * avx2::dot_product(pa_row, pb_row, k);
                            }
                        });
                        done = true;
                    }
                }
#endif
                if (!done)
                    for (int i = 0; i < m; ++i)
                        for (int j = 0; j < n; ++j) {
                            T sum = T(0);
                            for (int w = 0; w < k; ++w)
                                sum += pa[i * k + w] * pb[j * k + w];
                            py[i * n + j] += (T)(alpha * sum);
                        }
            }
        } else { // transA && transB
            // A is k*m, B is n*k; A[w][i] = pa[w*m+i], B[j][w] = pb[j*k+w]
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    T sum = T(0);
                    for (int w = 0; w < k; ++w)
                        sum += pa[w * m + i] * pb[j * k + w];
                    py[i * n + j] += (T)(alpha * sum);
                }
        }
        apply_post_op();
        return true;
    }

    // FP16 I/O with FP32 compute: convert inputs, run float GEMM, convert output back.
    bool exec_f16_as_f32() {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        const tensor_t* c = (inputs.size() > 2) ? inputs[2] : nullptr;

        // Native FP16 GEMM path: FP16 A × packed FP16 B → FP32 Y, then alpha/
        // beta/post in FP32 and FP32→FP16 on output. Gated on having the FP16
        // pack (built in reshape when ARM64 + has_neon_fp16() + initializer B
        // + !transA).
        if (!b_packed_fp16.empty() && !transA) {
            float* y_f32_native = (float*)ctx->workspace;
            const uint16_t* pa_u16 = (const uint16_t*)a->data;
            if (dgemm_fp16(m, n, k, pa_u16, b_packed_fp16.data(), y_f32_native)) {
                if (alpha != 1.0f)
                    for (int i = 0; i < m * n; ++i)
                        y_f32_native[i] *= alpha;
                if (c) {
                    for (int i = 0; i < m; ++i)
                        for (int j = 0; j < n; ++j) {
                            int oy = i * n + j;
                            y_f32_native[oy] += beta * (float)*(const float16_t*)c->broadcast_map_address(y, oy);
                        }
                }
                if (post_fn)
                    post_fn(y_f32_native, 1, m * n, m * n, fused_op, nullptr, 0);
                convert_f32_to_f16((float16_t*)y->data, y_f32_native, (size_t)m * n);
                return true;
            }
        }

        // Workspace layout: [A_f32 | B_f32 (if not packed) | Y_f32]
        float* ws = (float*)ctx->workspace;
        float* a_f32 = ws;
        float* y_f32;
        float* b_f32;

        convert_f16_to_f32(a_f32, (const float16_t*)a->data, (size_t)m * k);

        if (!b_packed.empty()) {
            // B already packed as float32 in reshape()
            b_f32 = nullptr;
            y_f32 = a_f32 + (size_t)m * k;
        } else {
            b_f32 = a_f32 + (size_t)m * k;
            convert_f16_to_f32(b_f32, (const float16_t*)b->data, (size_t)k * n);
            y_f32 = b_f32 + (size_t)k * n;
        }

        // Run float32 GEMM
        auto apply_post_op = [&]() {
            if (post_fn)
                post_fn(y_f32, 1, m * n, m * n, fused_op, nullptr, 0);
        };

        if (!b_packed.empty() && !transA) {
            bool fused = false;
            if (alpha == 1.0f) {
                const float* col_bias = nullptr;
                bool can_fuse = true;
                if (c) {
                    if (beta == 1.0f && c->ndim == 1 && c->dims[0] == n) {
                        // Convert FP16 bias to float32
                        if (c->type == NNR_DATA_TYPE_FLOAT16) {
                            float* c_f32 = y_f32 + (size_t)m * n;
                            convert_f16_to_f32(c_f32, (const float16_t*)c->data, n);
                            col_bias = c_f32;
                        } else {
                            col_bias = (const float*)c->data;
                        }
                    } else {
                        can_fuse = false;
                    }
                }
                if (can_fuse) {
                    gemm_post_nhwc_t gp;
                    gp.bias = col_bias;
                    gp.c_base = y_f32;
                    gp.post_fn = post_fn;
                    gp.fused_op = fused_op;
                    gp.classify();
                    dgemm_packed_b(m, n, k, a_f32, b_packed.data(), y_f32, gp);
                    fused = true;
                }
            }
            if (!fused) {
                dgemm_packed_b(m, n, k, a_f32, b_packed.data(), y_f32);
                if (alpha != 1.0f)
                    for (int i = 0; i < m * n; ++i)
                        y_f32[i] *= alpha;
                if (c) {
                    for (int i = 0; i < m; ++i)
                        for (int j = 0; j < n; ++j) {
                            int oy = i * n + j;
                            y_f32[oy] += beta * (float)*(const float16_t*)c->broadcast_map_address(y, oy);
                        }
                }
                apply_post_op();
            }
        } else if (!transA && !transB) {
            dgemm_generic(m, n, k, a_f32, b_f32, y_f32);
            if (alpha != 1.0f)
                for (int i = 0; i < m * n; ++i)
                    y_f32[i] *= alpha;
            if (c) {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        int oy = i * n + j;
                        y_f32[oy] += beta * (float)*(const float16_t*)c->broadcast_map_address(y, oy);
                    }
            }
            apply_post_op();
        } else {
            // Transpose cases: initialize with beta*C or zero
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    int oy = i * n + j;
                    y_f32[oy] = c ? beta * (float)*(const float16_t*)c->broadcast_map_address(y, oy) : 0.0f;
                }
            if (transA && !transB) {
                for (int i = 0; i < m; ++i)
                    for (int w = 0; w < k; ++w) {
                        float a_iw = alpha * a_f32[w * m + i];
                        for (int j = 0; j < n; ++j)
                            y_f32[i * n + j] += a_iw * b_f32[w * n + j];
                    }
            } else if (!transA && transB) {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (int w = 0; w < k; ++w)
                            sum += a_f32[i * k + w] * b_f32[j * k + w];
                        y_f32[i * n + j] += alpha * sum;
                    }
            } else { // transA && transB
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (int w = 0; w < k; ++w)
                            sum += a_f32[w * m + i] * b_f32[j * k + w];
                        y_f32[i * n + j] += alpha * sum;
                    }
            }
            apply_post_op();
        }

        // Convert output back to FP16
        convert_f32_to_f16((float16_t*)y->data, y_f32, (size_t)m * n);
        return true;
    }

    // BF16 GEMM: native VDPBF16PS when available, else convert-to-FP32 fallback.
    bool exec_bf16() {
        tensor_t* y = outputs[0];
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        const tensor_t* c = (inputs.size() > 2) ? inputs[2] : nullptr;
        const uint16_t* pa = (const uint16_t*)a->data;

        // Workspace: Y_f32 output buffer (+ B_packed_bf16 if not pre-packed)
        float* y_f32 = (float*)ctx->workspace;

#ifdef NNR_ARCH_X64
        if (has_avx512_bf16() && !transA) {
            // Native VDPBF16PS path
            const uint16_t* pb_packed;
            uint16_t* tmp_pack = nullptr;
            if (!b_packed_bf16.empty()) {
                pb_packed = b_packed_bf16.data();
            } else {
                // Pack B on the fly
                const uint16_t* pb = (const uint16_t*)b->data;
                size_t psz = bf16::pack_b_bf16_size(k, n);
                tmp_pack = (uint16_t*)(y_f32 + (size_t)m * n);
                if (transB) {
                    uint16_t* bt = tmp_pack + psz;
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < k; j++)
                            bt[(size_t)j * n + i] = pb[(size_t)i * k + j];
                    bf16::pack_b_bf16(tmp_pack, bt, k, n);
                } else {
                    bf16::pack_b_bf16(tmp_pack, pb, k, n);
                }
                pb_packed = tmp_pack;
            }

            gemm_post_t post;
            bf16::dgemm_bf16(m, n, k, pa, pb_packed, y_f32, post);

            // Apply alpha, beta*C, post-op in FP32
            if (alpha != 1.0f)
                for (int i = 0; i < m * n; ++i)
                    y_f32[i] *= alpha;
            if (c) {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        int oy = i * n + j;
                        y_f32[oy] += beta * (float)*(const bfloat16_t*)c->broadcast_map_address(y, oy);
                    }
            }
            if (post_fn)
                post_fn(y_f32, 1, m * n, m * n, fused_op, nullptr, 0);

            convert_f32_to_bf16((bfloat16_t*)y->data, y_f32, (size_t)m * n);
            return true;
        }
#endif
        // Fallback: convert BF16 → FP32 and use FP32 GEMM
        float* a_f32 = y_f32 + (size_t)m * n;
        convert_bf16_to_f32(a_f32, (const bfloat16_t*)a->data, (size_t)m * k);
        float* b_f32 = a_f32 + (size_t)m * k;
        convert_bf16_to_f32(b_f32, (const bfloat16_t*)b->data, (size_t)k * n);

        if (!transA && !transB) {
            dgemm_generic(m, n, k, a_f32, b_f32, y_f32);
        } else {
            // Scalar fallback for transpose cases
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int w = 0; w < k; ++w) {
                        float av = transA ? a_f32[w * m + i] : a_f32[i * k + w];
                        float bv = transB ? b_f32[j * k + w] : b_f32[w * n + j];
                        sum += av * bv;
                    }
                    y_f32[i * n + j] = alpha * sum;
                }
            if (alpha != 1.0f && !transA && !transB)
                for (int i = 0; i < m * n; ++i) y_f32[i] *= alpha;
        }
        if (c) {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    int oy = i * n + j;
                    y_f32[oy] += beta * (float)*(const bfloat16_t*)c->broadcast_map_address(y, oy);
                }
        }
        if (post_fn)
            post_fn(y_f32, 1, m * n, m * n, fused_op, nullptr, 0);
        convert_f32_to_bf16((bfloat16_t*)y->data, y_f32, (size_t)m * n);
        return true;
    }

    size_t workspace_size() const override {
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT16) {
            size_t ws = (size_t)m * k + (size_t)m * n;  // A_f32 + Y_f32
            if (b_packed.empty())
                ws += (size_t)k * n;  // B_f32
            if (inputs.size() > 2 && inputs[2]->type == NNR_DATA_TYPE_FLOAT16)
                ws += n;
            return ws * sizeof(float);
        }
        if (inputs[0]->type == NNR_DATA_TYPE_BFLOAT16) {
            size_t ws = (size_t)m * n;  // Y_f32
#ifdef NNR_ARCH_X64
            if (has_avx512_bf16() && b_packed_bf16.empty()) {
                // Space for on-the-fly B packing + possible transpose
                ws += bf16::pack_b_bf16_size(k, n) / 2;  // pack buffer (in floats equiv)
                if (transB) ws += ((size_t)k * n + 1) / 2;  // transpose buffer
            }
            if (!has_avx512_bf16())
#endif
            {
                // Fallback: A_f32 + B_f32
                ws += (size_t)m * k + (size_t)k * n;
            }
            return ws * sizeof(float);
        }
        return 0;
    }

    bool exec() override {
        if (inputs[0]->type == NNR_DATA_TYPE_FLOAT16)
            return exec_f16_as_f32();
        if (inputs[0]->type == NNR_DATA_TYPE_BFLOAT16)
            return exec_bf16();
        return typed_exec<Gemm_operator,
            opset_t<13, int32_t, int64_t,
                uint32_t, uint64_t,
                float, double>,
            opset_t<9, int32_t, int64_t,
                uint32_t, uint64_t,
                float, double>,
            opset_t<7, float, double>
        >(this, opset, inputs[0]->type);
    }
};

} // namespace {

// @nnr-meta-op mt=static workspace=yes prepack=yes fusion=post_op
operator_t* resolver_default_op_Gemm(int opset, pool_t& pool) { return pool_new<Gemm_operator>(pool); }

} // namespace nnr
