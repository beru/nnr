#include "nnr.h"
#include "pool.h"

#include "backend/webgpu/device.h"
#include "backend/webgpu/buffer.h"

#include "shaders.h"

#include <cstring>
#include <webgpu/webgpu_cpp.h>

namespace nnr {

namespace {

// float32 matmul. Four shapes accepted, served by two tiled shaders:
//   (a) 2D:                 A[M, K]       @ B[K, N]       -> C[M, N]
//   (b) left-batched 2D:    A[..., M, K]  @ B[K, N]       -> C[..., M, N]
//       The batch dims are collapsed into M since B is shared; no Z
//       dispatch.
//   (c) same-rank batched:  A[b..., M, K] @ B[b..., K, N] -> C[b..., M, N]
//       Batch dims match exactly. Z dispatch with per-batch offset =
//       batch_idx * M * K (A) / batch_idx * K * N (B). Simple shader.
//   (d) broadcast batched:  A[a_b..., M, K] @ B[b_b..., K, N]
//       -> C[out_b..., M, N] where out_b = NumPy-broadcast of a_b and b_b.
//       Ranks may differ; dims must be equal or one is 1 after
//       left-padding. Z dispatch with per-side stride tables in a
//       dedicated shader (matmul_bcast.wgsl).
// Anything else (1D, non-float32, rank > 8) returns false and falls back
// to CPU via the registry.
struct MatMul_operator_webgpu : public operator_t {
    int m = 0, n = 0, k = 0;
    int batch = 1;                // 1 for (a)/(b); prod(batch dims) for (c)/(d)
    int mode  = 0;                // 0 = plain (shader `matmul`); 1 = broadcast (`matmul_bcast`)

    // Mode-1 (broadcast) batch tables, element strides in A / B per out-axis.
    int  nbatch = 0;
    uint32_t out_dims[8]  = {};
    uint32_t a_strides[8] = {};
    uint32_t b_strides[8] = {};

    // Plain pipeline (small shapes, 16×16 tile, 1 output per thread).
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Buffer          uniforms;       // 16 bytes: {M, N, K, batch}

    // Cached BindGroup for the plain (mode-0) kernel. Invalidated when any
    // of the three bound tensors (A, B, Y) reallocates its GPU buffer.
    wgpu::BindGroup cached_bg;
    uint32_t        cached_gen[3] = {};  // [A, B, Y]

    // Same idea for the broadcast (mode-1) kernel — distinct BGL, distinct
    // cache slot.
    wgpu::BindGroup cached_bg_bcast;
    uint32_t        cached_gen_bcast[3] = {};

    // Register-tiled pipeline (32×32 tile, 2×2 per thread; built lazily
    // on first use since small matmuls never touch it).
    wgpu::ComputePipeline pipeline_tiled;
    bool                  tiled_built = false;

    // Large-tile pipeline (64×64 output tile, 4×4 per thread; built
    // lazily). Highest-tier kernel — used when both M ≥ 64 and N ≥ 64.
    wgpu::ComputePipeline pipeline_big;
    bool                  big_built = false;

    // Broadcast pipeline (built lazily on first use)
    wgpu::ComputePipeline pipeline_bcast;
    wgpu::BindGroupLayout bgl_bcast;
    wgpu::Buffer          uniforms_bcast; // 112 bytes: {M,N,K,nbatch, out_dims[8], a_strides[8], b_strides[8]}
    bool                  bcast_built = false;

    // Which mode-0 kernel reshape picked: 0 = plain, 1 = 32×32 tiled,
    // 2 = 64×64 big.
    int tile_tier = 0;

    bool init() override {
        if (!is_inout_size(2, 1)) return false;
        // Early-reject non-f32 / 1D / rank > 8 so the loader falls back to
        // CPU. reshape()'s returns-false path has no backend-fallback.
        if (inputs[0] && inputs[0]->type != NNR_DATA_TYPE_UNDEFINED
            && inputs[0]->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (inputs[1] && inputs[1]->type != NNR_DATA_TYPE_UNDEFINED
            && inputs[1]->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (inputs[0] && inputs[0]->ndim > 0 && inputs[0]->ndim < 2) return false;
        if (inputs[1] && inputs[1]->ndim > 0 && inputs[1]->ndim < 1) return false;
        if (inputs[0] && inputs[0]->ndim > 8) return false;
        if (inputs[1] && inputs[1]->ndim > 8) return false;
        if (!webgpu::device_ready())  return false;

        auto& dev = webgpu::get_device();

        // --- Plain 16x16 tile pipeline (modes a/b/c) ---
        {
            wgpu::ShaderSourceWGSL wgslSrc = {};
            wgslSrc.code = webgpu::wgsl::matmul;
            wgpu::ShaderModuleDescriptor smd = {};
            smd.nextInChain = &wgslSrc;
            wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

            wgpu::BindGroupLayoutEntry e[4] = {};
            e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
            e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
            e[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
            e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
            e[2].buffer.type = wgpu::BufferBindingType::Storage;
            e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
            e[3].buffer.type = wgpu::BufferBindingType::Uniform;

            wgpu::BindGroupLayoutDescriptor bgld = {};
            bgld.entryCount = 4;
            bgld.entries = e;
            bgl = dev.device.CreateBindGroupLayout(&bgld);

            wgpu::PipelineLayoutDescriptor pld = {};
            pld.bindGroupLayoutCount = 1;
            pld.bindGroupLayouts = &bgl;
            wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

            wgpu::ComputePipelineDescriptor cpd = {};
            cpd.layout = pl;
            cpd.compute.module = sm;
            cpd.compute.entryPoint = "main";
            pipeline = dev.device.CreateComputePipeline(&cpd);

            wgpu::BufferDescriptor ud = {};
            ud.size = 16;
            ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            uniforms = dev.device.CreateBuffer(&ud);
        }

        return true;
    }

    // Build the register-tiled pipeline on first use. Shares the plain
    // bind-group layout (same bindings: A, B, C, Dims uniform).
    bool ensure_tiled_pipeline() {
        if (tiled_built) return true;
        auto& dev = webgpu::get_device();

        wgpu::ShaderSourceWGSL wgslSrc = {};
        wgslSrc.code = webgpu::wgsl::matmul_tiled;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &wgslSrc;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline_tiled = dev.device.CreateComputePipeline(&cpd);
        tiled_built = true;
        return true;
    }

    // Build the large-tile pipeline on first use. Same bind-group layout
    // as the plain/tiled kernels — only shader differs.
    bool ensure_big_pipeline() {
        if (big_built) return true;
        auto& dev = webgpu::get_device();

        wgpu::ShaderSourceWGSL wgslSrc = {};
        wgslSrc.code = webgpu::wgsl::matmul_big;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &wgslSrc;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline_big = dev.device.CreateComputePipeline(&cpd);
        big_built = true;
        return true;
    }

    // Build the broadcast pipeline on first use.
    bool ensure_bcast_pipeline() {
        if (bcast_built) return true;
        auto& dev = webgpu::get_device();

        wgpu::ShaderSourceWGSL wgslSrc = {};
        wgslSrc.code = webgpu::wgsl::matmul_bcast;
        wgpu::ShaderModuleDescriptor smd = {};
        smd.nextInChain = &wgslSrc;
        wgpu::ShaderModule sm = dev.device.CreateShaderModule(&smd);

        wgpu::BindGroupLayoutEntry e[4] = {};
        e[0].binding = 0; e[0].visibility = wgpu::ShaderStage::Compute;
        e[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[1].binding = 1; e[1].visibility = wgpu::ShaderStage::Compute;
        e[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        e[2].binding = 2; e[2].visibility = wgpu::ShaderStage::Compute;
        e[2].buffer.type = wgpu::BufferBindingType::Storage;
        e[3].binding = 3; e[3].visibility = wgpu::ShaderStage::Compute;
        e[3].buffer.type = wgpu::BufferBindingType::Uniform;

        wgpu::BindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = 4;
        bgld.entries = e;
        bgl_bcast = dev.device.CreateBindGroupLayout(&bgld);

        wgpu::PipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl_bcast;
        wgpu::PipelineLayout pl = dev.device.CreatePipelineLayout(&pld);

        wgpu::ComputePipelineDescriptor cpd = {};
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = "main";
        pipeline_bcast = dev.device.CreateComputePipeline(&cpd);

        wgpu::BufferDescriptor ud = {};
        // Layout: 16B header (M,N,K,nbatch) + 8 u32 out_dims (2×vec4)
        //       + 8 u32 a_strides (2×vec4) + 8 u32 b_strides (2×vec4)
        // Total = 16 + 32 + 32 + 32 = 112 bytes.
        ud.size = 112;
        ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniforms_bcast = dev.device.CreateBuffer(&ud);

        bcast_built = true;
        return true;
    }

    bool reshape() override {
        const tensor_t* a = inputs[0];
        const tensor_t* b = inputs[1];
        tensor_t*       y = outputs[0];

        if (a->type != NNR_DATA_TYPE_FLOAT32 || b->type != NNR_DATA_TYPE_FLOAT32) return false;
        if (a->ndim < 2 || b->ndim < 1) return false;
        if (a->ndim > 8 || b->ndim > 8) return false;

        batch = 1;
        mode  = 0;

        if (a->ndim == 2 && b->ndim == 2) {
            // Mode (a): straight 2D
            if (a->dims[1] != b->dims[0]) return false;
            m = a->dims[0];
            k = a->dims[1];
            n = b->dims[1];
            int out[2] = { m, n };
            if (!y->reshape(std::span<const int>(out, 2), NNR_DATA_TYPE_FLOAT32)) return false;
        }
        else if (a->ndim > 2 && b->ndim == 2) {
            // Mode (b): left-batched, right-2D. Collapse A's batch dims into M.
            const int arank = a->ndim;
            const int base_m = a->dims[arank - 2];
            k = a->dims[arank - 1];
            if (k != b->dims[0]) return false;
            n = b->dims[1];

            int batch_prod = 1;
            int out_dims_[MAX_NDIM];
            for (int i = 0; i < arank - 2; ++i) {
                out_dims_[i] = a->dims[i];
                batch_prod *= a->dims[i];
            }
            out_dims_[arank - 2] = base_m;
            out_dims_[arank - 1] = n;
            if (!y->reshape(std::span<const int>(out_dims_, arank), NNR_DATA_TYPE_FLOAT32)) return false;

            m = base_m * batch_prod;  // shader sees a flat [m, k] x [k, n]
        }
        else if (b->ndim >= 3 || (a->ndim >= 3 && b->ndim >= 2)) {
            // Mode (c) or (d): batched. The 2D × batched-B case treats A
            // as having zero batch dims (fully broadcast across the output
            // batch), which the mode-(d) stride-table path handles natively.
            const int arank = a->ndim;
            const int brank = b->ndim;

            m = a->dims[arank - 2];
            k = a->dims[arank - 1];
            if (k != b->dims[brank - 2]) return false;
            n = b->dims[brank - 1];

            const int a_nb = arank - 2;
            const int b_nb = brank - 2;
            const int out_nb = (a_nb > b_nb) ? a_nb : b_nb;
            if (out_nb > 8) return false;

            // Left-pad shorter batch to match rank; compute the NumPy-
            // broadcast output batch shape. Record stride-0 on broadcast
            // axes so the shader sees "same slice" across that out axis.
            int out_batch[8] = {};
            int a_batch_raw[8] = {};
            int b_batch_raw[8] = {};
            bool needs_broadcast = false;
            for (int i = 0; i < out_nb; ++i) {
                int ai = i - (out_nb - a_nb);
                int bi = i - (out_nb - b_nb);
                int av = (ai >= 0) ? a->dims[ai] : 1;
                int bv = (bi >= 0) ? b->dims[bi] : 1;
                if (av != bv && av != 1 && bv != 1) return false;  // invalid broadcast
                out_batch[i] = (av > bv) ? av : bv;
                a_batch_raw[i] = av;
                b_batch_raw[i] = bv;
                if (av != out_batch[i] || bv != out_batch[i]) needs_broadcast = true;
                if (a_nb != b_nb) needs_broadcast = true;
            }

            int batch_prod = 1;
            int out_shape[MAX_NDIM];
            for (int i = 0; i < out_nb; ++i) {
                out_shape[i] = out_batch[i];
                batch_prod  *= out_batch[i];
            }
            out_shape[out_nb]     = m;
            out_shape[out_nb + 1] = n;
            if (!y->reshape(std::span<const int>(out_shape, out_nb + 2), NNR_DATA_TYPE_FLOAT32)) return false;
            batch = batch_prod;

            if (!needs_broadcast) {
                // Mode (c): batch shapes match exactly, use the plain shader.
                mode = 0;
            } else {
                // Mode (d): broadcast batched, use matmul_bcast.
                if (!ensure_bcast_pipeline()) return false;
                mode = 1;
                nbatch = out_nb;
                // Compute element strides for each out-axis in A / B.
                // Axes present in A start at (out_nb - a_nb) and have the
                // usual stride = product of trailing A dims. Broadcast
                // axes (A dim == 1 while out dim > 1, or axis missing
                // entirely in A) get stride 0.
                for (int i = 0; i < 8; ++i) {
                    out_dims[i]  = 0;
                    a_strides[i] = 0;
                    b_strides[i] = 0;
                }
                // A batch axes: walk from last to first; effective stride is
                // M*K scaled by product of later A batch dims; 0 if broadcast.
                {
                    uint32_t run = (uint32_t)m * (uint32_t)k;
                    for (int i = a_nb - 1; i >= 0; --i) {
                        int out_i = i + (out_nb - a_nb);
                        a_strides[out_i] = (a_batch_raw[out_i] == 1) ? 0u : run;
                        run *= (uint32_t)a->dims[i];
                    }
                }
                {
                    uint32_t run = (uint32_t)k * (uint32_t)n;
                    for (int i = b_nb - 1; i >= 0; --i) {
                        int out_i = i + (out_nb - b_nb);
                        b_strides[out_i] = (b_batch_raw[out_i] == 1) ? 0u : run;
                        run *= (uint32_t)b->dims[i];
                    }
                }
                for (int i = 0; i < out_nb; ++i)
                    out_dims[i] = (uint32_t)out_batch[i];
            }
        }
        else {
            return false;
        }

        webgpu::ensure_buffer(a, a->ndata * sizeof(float));
        auto& br = webgpu::ensure_buffer(b, b->ndata * sizeof(float));
        if (ctx && ctx->initializer_names.count(b->name)) {
            br.is_weight = true;
        }
        webgpu::ensure_buffer(y, y->ndata * sizeof(float));

        // Pick a tile tier based on shape:
        //   tier 0 = plain kernel (small, anything goes)
        //   tier 1 = 32×32 tiled (M ≥ 32 && N ≥ 32 && K ≥ 16)
        //   tier 2 = 64×64 big   (M ≥ 64 && N ≥ 64 && K ≥ 16 &&
        //                         M*N*batch ≥ 65536)
        // The M*N*batch gate on tier-2 keeps "single 64×64 tile" dispatches
        // on the 32×32 kernel: at M=N=64 tier-2 fires only one workgroup per
        // batch, whereas tier-1 fires 2×2=4 per batch and fills more CUs.
        // 65536 ≈ 16 tier-2 workgroups' worth of output so a modern GPU
        // stays occupied. Broadcast path always uses its own simple shader.
        tile_tier = 0;
        if (mode == 0 && k >= 16) {
            const int64_t work = (int64_t)m * (int64_t)n * (int64_t)batch;
            if (m >= 64 && n >= 64 && work >= 65536) {
                if (ensure_big_pipeline()) tile_tier = 2;
                else if (ensure_tiled_pipeline()) tile_tier = 1;
            } else if (m >= 32 && n >= 32) {
                if (ensure_tiled_pipeline()) tile_tier = 1;
            }
        }

        // Per-call uniform/meta payload depends only on reshape-time data
        // (m, n, k, batch and the broadcast stride tables). Write it here
        // so exec() doesn't queue a WriteBuffer every dispatch.
        auto& dev = webgpu::get_device();
        if (mode == 0) {
            uint32_t dims[4] = { (uint32_t)m, (uint32_t)n, (uint32_t)k, (uint32_t)batch };
            dev.queue.WriteBuffer(uniforms, 0, dims, sizeof(dims));
        } else {
            uint8_t bbuf[112] = {};
            auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(bbuf + off, &v, 4); };
            put_u32(0,  (uint32_t)m);
            put_u32(4,  (uint32_t)n);
            put_u32(8,  (uint32_t)k);
            put_u32(12, (uint32_t)nbatch);
            for (int i = 0; i < 8; ++i) put_u32(16 + i * 4, out_dims[i]);
            for (int i = 0; i < 8; ++i) put_u32(48 + i * 4, a_strides[i]);
            for (int i = 0; i < 8; ++i) put_u32(80 + i * 4, b_strides[i]);
            dev.queue.WriteBuffer(uniforms_bcast, 0, bbuf, sizeof(bbuf));
        }
        return true;
    }

    bool exec() override {
        auto& dev = webgpu::get_device();

        webgpu::upload_if_needed(inputs[0]);
        webgpu::upload_if_needed(inputs[1]);

        auto* ra = webgpu::find(inputs[0]);
        auto* rb = webgpu::find(inputs[1]);
        auto* ry = webgpu::find(outputs[0]);

        wgpu::ComputePassEncoder pass = webgpu::shared_encoder().BeginComputePass();

        if (mode == 0) {
            // Uniform was written in reshape() — its contents depend only
            // on m/n/k/batch which are reshape-time constants. BindGroup
            // cache: rebuild only when any bound tensor's buffer was
            // (re)allocated since the last exec.
            uint32_t gen_a = webgpu::generation_of(inputs[0]);
            uint32_t gen_b = webgpu::generation_of(inputs[1]);
            uint32_t gen_y = webgpu::generation_of(outputs[0]);
            if (!cached_bg || gen_a != cached_gen[0]
                           || gen_b != cached_gen[1]
                           || gen_y != cached_gen[2]) {
                wgpu::BindGroupEntry be[4] = {};
                be[0].binding = 0; be[0].buffer = ra->buf;   be[0].offset = 0; be[0].size = ra->size;
                be[1].binding = 1; be[1].buffer = rb->buf;   be[1].offset = 0; be[1].size = rb->size;
                be[2].binding = 2; be[2].buffer = ry->buf;   be[2].offset = 0; be[2].size = ry->size;
                be[3].binding = 3; be[3].buffer = uniforms;  be[3].offset = 0; be[3].size = 16;
                wgpu::BindGroupDescriptor bgd = {};
                bgd.layout = bgl; bgd.entryCount = 4; bgd.entries = be;
                cached_bg = dev.device.CreateBindGroup(&bgd);
                cached_gen[0] = gen_a;
                cached_gen[1] = gen_b;
                cached_gen[2] = gen_y;
            }

            switch (tile_tier) {
                case 2:  pass.SetPipeline(pipeline_big);   break;
                case 1:  pass.SetPipeline(pipeline_tiled); break;
                default: pass.SetPipeline(pipeline);       break;
            }
            pass.SetBindGroup(0, cached_bg);
        } else {
            // Broadcast meta written in reshape().
            uint32_t gen_a = webgpu::generation_of(inputs[0]);
            uint32_t gen_b = webgpu::generation_of(inputs[1]);
            uint32_t gen_y = webgpu::generation_of(outputs[0]);
            if (!cached_bg_bcast || gen_a != cached_gen_bcast[0]
                                 || gen_b != cached_gen_bcast[1]
                                 || gen_y != cached_gen_bcast[2]) {
                wgpu::BindGroupEntry be[4] = {};
                be[0].binding = 0; be[0].buffer = ra->buf;        be[0].offset = 0; be[0].size = ra->size;
                be[1].binding = 1; be[1].buffer = rb->buf;        be[1].offset = 0; be[1].size = rb->size;
                be[2].binding = 2; be[2].buffer = ry->buf;        be[2].offset = 0; be[2].size = ry->size;
                be[3].binding = 3; be[3].buffer = uniforms_bcast; be[3].offset = 0; be[3].size = 112;
                wgpu::BindGroupDescriptor bgd = {};
                bgd.layout = bgl_bcast; bgd.entryCount = 4; bgd.entries = be;
                cached_bg_bcast = dev.device.CreateBindGroup(&bgd);
                cached_gen_bcast[0] = gen_a;
                cached_gen_bcast[1] = gen_b;
                cached_gen_bcast[2] = gen_y;
            }

            pass.SetPipeline(pipeline_bcast);
            pass.SetBindGroup(0, cached_bg_bcast);
        }

        // Tile size depends on which kernel we're dispatching:
        //   plain / broadcast → 16×16 output per workgroup
        //   32×32 tiled       → 32×32 output per workgroup
        //   64×64 big         → 64×64 output per workgroup
        uint32_t tile = 16u;
        if (mode == 0) {
            if      (tile_tier == 2) tile = 64u;
            else if (tile_tier == 1) tile = 32u;
        }
        const uint32_t gx = ((uint32_t)n + tile - 1) / tile;
        const uint32_t gy = ((uint32_t)m + tile - 1) / tile;
        const uint32_t gz = (uint32_t)batch;
        pass.DispatchWorkgroups(gx, gy, gz);
        pass.End();

        webgpu::mark_gpu_written(outputs[0]);
        return true;
    }

    int64_t num_ops() const override { return (int64_t)2 * batch * m * n * k; }
};

} // namespace

operator_t* resolver_default_op_MatMul_webgpu(int opset, pool_t& pool) {
    return pool_new<MatMul_operator_webgpu>(pool);
}

} // namespace nnr
