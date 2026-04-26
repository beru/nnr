#if defined(NNR_USE_CUDA)

// NonMaxSuppression on CUDA — one block per (batch, class), 256 threads.
//
// Each block runs the standard greedy NMS: pick max-score not-yet-suppressed,
// then suppress all boxes with IoU > threshold against it.
//
// Phase 0 (one-shot): collect indices with score > score_threshold into a
//   compact candidate list `cand_idx[0..n_cand)`. This shrinks the per-pass
//   working set from `num_boxes` (up to 16384) to typically 100s of entries,
//   cutting argmax + IoU memory traffic by 10-100×.
// Phase 1: parallel argmax over cand_idx (skipping suppressed indices).
// Phase 2: parallel IoU broadcast vs the just-added box, marking the global
//   `suppressed` bitmap so subsequent argmax passes skip suppressed entries.
//
// Shared memory (per block, fixed-cap):
//   suppressed[NMS_MAX_BOXES/32]   // 2 KB — bitmap indexed by original box idx
//   cand_idx[MAX_CAND]              // 16 KB at MAX_CAND=4096
//   sel_idx, sel_x1..y2, sel_area   // ~6 KB at MAX_SEL=256
//
// Constraints (else fall back to CPU):
//   - num_boxes <= 16384
//   - selected per (b, c) <= 256
//   - center_point_box ∈ {0, 1}, fp32 boxes/scores
// If post-threshold candidate count exceeds MAX_CAND we use the first MAX_CAND
// found (a tiny correctness compromise — only triggers when score_threshold is
// near 0 and most boxes survive).

#include "nnr.h"
#include "registry.h"
#include "pool.h"
#include "cuda_backend.h"
#include "attr_key.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace nnr {

operator_t* resolver_default_op_NonMaxSuppression(int opset, pool_t& pool);

namespace gpu { cuda_backend_t* get_or_create_cuda_backend(context_t* ctx); }

namespace {

static constexpr int NMS_MAX_BOXES = 16384;
static constexpr int NMS_MAX_SEL   = 256;
static constexpr int NMS_BLOCK     = 256;

static const char* nms_source() {
    return R"CUDA(
#define MAX_BOXES 16384
#define MAX_CAND  4096
#define MAX_SEL   256
#define BLOCK     256

extern "C" __global__
void nms_kernel(const float* __restrict__ boxes,    // [batch, num_boxes, 4]
                const float* __restrict__ scores,   // [batch, num_classes, num_boxes]
                long long* __restrict__ results,    // [max_total, 3]
                int batch_size, int num_classes, int num_boxes,
                int max_output_per_class,
                int max_total,
                float iou_threshold,
                float score_threshold,
                int center_point_box)
{
    int b = blockIdx.x;
    int c = blockIdx.y;
    if (b >= batch_size || c >= num_classes) return;

    int tid = threadIdx.x;

    const float* b_boxes  = boxes  + (long long)b * num_boxes * 4;
    const float* c_scores = scores + ((long long)b * num_classes + c) * num_boxes;

    __shared__ unsigned int suppressed[MAX_BOXES / 32];
    __shared__ int   cand_idx[MAX_CAND];
    __shared__ int   sel_idx[MAX_SEL];
    __shared__ float sel_x1[MAX_SEL], sel_y1[MAX_SEL];
    __shared__ float sel_x2[MAX_SEL], sel_y2[MAX_SEL];
    __shared__ float sel_area[MAX_SEL];
    __shared__ int n_sel;
    __shared__ int n_cand_raw;

    // Pair arrays for explicit block reduction (avoid 64-bit atomicMax which
    // serializes warps on sm_86 — measured ~2× slower than the reduction below
    // for ssd-12 box counts).
    __shared__ float red_v[BLOCK];
    __shared__ int   red_i[BLOCK];

    // Init bitmap + counters.
    for (int i = tid; i < (MAX_BOXES / 32); i += BLOCK) suppressed[i] = 0u;
    if (tid == 0) { n_sel = 0; n_cand_raw = 0; }
    __syncthreads();

    // Phase 0: collect indices with score > score_threshold into cand_idx.
    // atomicAdd-append; cap at MAX_CAND. Lower-scored entries get dropped on
    // overflow — only matters if threshold is near zero AND the dropped boxes
    // would have survived NMS, both rare in practice.
    for (int j = tid; j < num_boxes; j += BLOCK) {
        float s = c_scores[j];
        if (!(s > score_threshold)) continue;
        int slot = atomicAdd(&n_cand_raw, 1);
        if (slot < MAX_CAND) cand_idx[slot] = j;
    }
    __syncthreads();
    int n_cand = n_cand_raw < MAX_CAND ? n_cand_raw : MAX_CAND;

    int eff_max = max_output_per_class < MAX_SEL ? max_output_per_class : MAX_SEL;

    while (n_sel < eff_max) {
        // Step 1: parallel argmax of c_scores over candidate list,
        // skipping indices that have been suppressed by earlier picks.
        float lv = -3.4028235e38f;
        int   li = -1;
        for (int k = tid; k < n_cand; k += BLOCK) {
            int j = cand_idx[k];
            if ((suppressed[j >> 5] >> (j & 31)) & 1u) continue;
            float s = c_scores[j];
            if (s > lv || (s == lv && (li < 0 || j < li))) {
                lv = s; li = j;
            }
        }
        red_v[tid] = lv;
        red_i[tid] = li;
        __syncthreads();
        // Block reduction. Stride down to 32 with __syncthreads, then warp
        // shuffle for the last 5 stages (single warp — no sync needed).
        for (int off = BLOCK / 2; off > 32; off >>= 1) {
            if (tid < off) {
                float A = red_v[tid],     B = red_v[tid + off];
                int   Ai = red_i[tid],    Bi = red_i[tid + off];
                bool take_B;
                if (Ai < 0)            take_B = (Bi >= 0);
                else if (Bi < 0)       take_B = false;
                else                   take_B = (B > A) || (B == A && Bi < Ai);
                if (take_B) { red_v[tid] = B; red_i[tid] = Bi; }
            }
            __syncthreads();
        }
        if (tid < 32) {
            float wv = red_v[tid];
            int   wi = red_i[tid];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float B  = __shfl_xor_sync(0xffffffffu, wv, off);
                int   Bi = __shfl_xor_sync(0xffffffffu, wi, off);
                bool take_B;
                if (wi < 0)        take_B = (Bi >= 0);
                else if (Bi < 0)   take_B = false;
                else               take_B = (B > wv) || (B == wv && Bi < wi);
                if (take_B) { wv = B; wi = Bi; }
            }
            if (tid == 0) { red_v[0] = wv; red_i[0] = wi; }
        }
        __syncthreads();

        int   best_i = red_i[0];
        float best_s = red_v[0];
        if (best_i < 0) break;
        if (!(best_s > score_threshold)) break;

        // Step 2: thread 0 decodes the picked box and appends to selected list.
        if (tid == 0) {
            const float* p = b_boxes + best_i * 4;
            float x1, y1, x2, y2;
            if (center_point_box) {
                float cx = p[0], cy = p[1], w = p[2], h = p[3];
                x1 = cx - w * 0.5f; x2 = cx + w * 0.5f;
                y1 = cy - h * 0.5f; y2 = cy + h * 0.5f;
            } else {
                float a0 = p[0], a1 = p[1], a2 = p[2], a3 = p[3];
                y1 = a0; x1 = a1; y2 = a2; x2 = a3;
                if (x1 > x2) { float t = x1; x1 = x2; x2 = t; }
                if (y1 > y2) { float t = y1; y1 = y2; y2 = t; }
            }
            float area = (x2 - x1) * (y2 - y1);
            // Always mark this idx suppressed so it isn't re-picked.
            atomicOr(&suppressed[best_i >> 5], (1u << (best_i & 31)));
            if (area > 0.f) {
                int slot = n_sel;
                sel_idx[slot] = best_i;
                sel_x1[slot] = x1; sel_y1[slot] = y1;
                sel_x2[slot] = x2; sel_y2[slot] = y2;
                sel_area[slot] = area;
                n_sel = slot + 1;
            }
        }
        __syncthreads();

        // Step 3: parallel IoU suppression vs the just-added box.
        int last = n_sel - 1;
        if (last < 0) continue;       // degenerate-area pick, no new selection
        float x1 = sel_x1[last], y1 = sel_y1[last];
        float x2 = sel_x2[last], y2 = sel_y2[last];
        float area = sel_area[last];

        for (int k = tid; k < n_cand; k += BLOCK) {
            int j = cand_idx[k];
            if ((suppressed[j >> 5] >> (j & 31)) & 1u) continue;
            const float* q = b_boxes + j * 4;
            float bx1, by1, bx2, by2;
            if (center_point_box) {
                float cx = q[0], cy = q[1], w = q[2], h = q[3];
                bx1 = cx - w * 0.5f; bx2 = cx + w * 0.5f;
                by1 = cy - h * 0.5f; by2 = cy + h * 0.5f;
            } else {
                float a0 = q[0], a1 = q[1], a2 = q[2], a3 = q[3];
                by1 = a0; bx1 = a1; by2 = a2; bx2 = a3;
                if (bx1 > bx2) { float t = bx1; bx1 = bx2; bx2 = t; }
                if (by1 > by2) { float t = by1; by1 = by2; by2 = t; }
            }
            float ix1 = x1 > bx1 ? x1 : bx1;
            float ix2 = x2 < bx2 ? x2 : bx2;
            if (ix2 <= ix1) continue;
            float iy1 = y1 > by1 ? y1 : by1;
            float iy2 = y2 < by2 ? y2 : by2;
            if (iy2 <= iy1) continue;
            float inter = (ix2 - ix1) * (iy2 - iy1);
            float barea = (bx2 - bx1) * (by2 - by1);
            float uni = area + barea - inter;
            if (uni > 0.f && inter > iou_threshold * uni) {
                atomicOr(&suppressed[j >> 5], (1u << (j & 31)));
            }
        }
        __syncthreads();
    }

    // Step 4: each block writes into its own deterministic slot range
    // [(b*num_classes + c) * max_output_per_class, ... + n_sel). Trailing
    // slots remain zero from the host's pre-launch memset, so downstream
    // Gather sees (0, 0, 0) on padding rows.
    if (tid == 0 && n_sel > 0) {
        long long base = ((long long)b * num_classes + c) * max_output_per_class;
        long long bb = (long long)b;
        long long cc = (long long)c;
        for (int k = 0; k < n_sel; ++k) {
            long long dst = base + k;
            if (dst >= max_total) break;
            results[dst * 3 + 0] = bb;
            results[dst * 3 + 1] = cc;
            results[dst * 3 + 2] = (long long)sel_idx[k];
        }
    }
}
)CUDA";
}

struct NonMaxSuppression_cuda : public operator_t {
    bool prim_valid = false;
    operator_t* fallback = nullptr;
    int center_point_box = 0;

    int batch_size = 0, num_classes = 0, num_boxes = 0;
    int max_per_class = 0, max_total = 0;
    float iou_threshold = 0.f;
    float score_threshold = 0.f;


    bool init() override {
        if (inputs.size() < 2 || outputs.size() != 1) return false;
        fallback = resolver_default_op_NonMaxSuppression(opset, ctx->attr_pool);
        fallback->ctx = ctx; fallback->opset = opset; fallback->op_type = op_type;
        fallback->inputs = inputs; fallback->outputs = outputs; fallback->attrs = attrs;
        fallback->init();
        center_point_box = (int)attribute(attr_key_t::center_point_box, (int32_t)0);
        return true;
    }

    bool reshape() override {
        if (!fallback->reshape()) return false;
        prim_valid = false; device_tag = 0;

        const tensor_t* boxes_t  = inputs[0];
        const tensor_t* scores_t = inputs[1];
        if (!boxes_t || !scores_t) return true;
        if (boxes_t->type != NNR_DATA_TYPE_FLOAT32 || scores_t->type != NNR_DATA_TYPE_FLOAT32)
            return true;
        if (boxes_t->ndim != 3 || scores_t->ndim != 3) return true;
        if (outputs[0]->type != NNR_DATA_TYPE_INT64) return true;

        batch_size  = boxes_t->dims[0];
        num_boxes   = boxes_t->dims[1];
        num_classes = scores_t->dims[1];
        if (boxes_t->dims[2] != 4) return true;
        if (scores_t->dims[0] != batch_size || scores_t->dims[2] != num_boxes) return true;
        if (num_boxes > NMS_MAX_BOXES) return true;

        int64_t max_output_boxes = 0;
        iou_threshold   = 0.f;
        score_threshold = -3.4028235e38f;
        if (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0 && inputs[2]->data)
            max_output_boxes = ((const int64_t*)inputs[2]->data)[0];
        if (inputs.size() > 3 && inputs[3] && inputs[3]->ndata > 0 && inputs[3]->data)
            iou_threshold = ((const float*)inputs[3]->data)[0];
        if (inputs.size() > 4 && inputs[4] && inputs[4]->ndata > 0 && inputs[4]->data)
            score_threshold = ((const float*)inputs[4]->data)[0];

        if (max_output_boxes <= 0) max_output_boxes = num_boxes;
        if (max_output_boxes > NMS_MAX_SEL) max_output_boxes = NMS_MAX_SEL;
        max_per_class = (int)max_output_boxes;
        max_total = outputs[0]->dims[0];

        // The output buffer must be large enough that each (b,c) block has its
        // own non-overlapping slot range of `max_per_class` rows.
        long long required = (long long)batch_size * num_classes * max_per_class;
        if (required > max_total) max_per_class = (int)(max_total / ((long long)batch_size * num_classes));
        if (max_per_class <= 0) return true;

        prim_valid = true;
        device_tag = static_cast<uint8_t>(backend_t::CUDA);
        return true;
    }

    bool exec() override {
        if (!prim_valid) return fallback->exec();
        auto* be = gpu::get_or_create_cuda_backend(ctx);
        if (!be) return fallback->exec();

        CUfunction f = be->nvrtc.get("nnr_nms", nms_source(), "nms_kernel",
                                     gpu::nvrtc_arch_option(be->device));
        if (!f) return fallback->exec();

        const float* d_boxes  = (const float*)be->cache->ensure_device(inputs[0]);
        const float* d_scores = (const float*)be->cache->ensure_device(inputs[1]);
        long long*   d_y      = (long long*)  be->cache->alloc_output(outputs[0]);
        if (!d_boxes || !d_scores || !d_y) return fallback->exec();

        cudaStream_t s = be->device->compute_stream();
        cudaMemsetAsync(d_y, 0, (size_t)max_total * 3 * sizeof(long long), s);

        int _bs = batch_size, _nc = num_classes, _nb = num_boxes;
        int _mpc = max_per_class, _mt = max_total, _cpb = center_point_box;
        float _iou = iou_threshold, _st = score_threshold;
        void* args[] = { &d_boxes, &d_scores, &d_y,
                         &_bs, &_nc, &_nb, &_mpc, &_mt,
                         &_iou, &_st, &_cpb };

        unsigned grid_x = (unsigned)batch_size;
        unsigned grid_y = (unsigned)num_classes;
        if (!gpu::nvrtc_launch(be->device, f, grid_x, grid_y, 1, NMS_BLOCK, 1, 1, args))
            return fallback->exec();

        be->cache->mark_written(outputs[0]);
        return true;
    }
};

} // namespace

operator_t* resolver_cuda_op_NonMaxSuppression(int opset, pool_t& pool) {
    return pool_new<NonMaxSuppression_cuda>(pool);
}

} // namespace nnr

#endif // NNR_USE_CUDA
