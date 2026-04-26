#include "nnr.h"
#include "util.h"
#include "allocator.h"
#include <algorithm>
#include <vector>

namespace nnr {

namespace {

struct NonMaxSuppression_operator : public operator_t {
    int center_point_box = 0;

    bool init() override {
        if (inputs.size() < 2 || outputs.size() != 1)
            return false;
        center_point_box = attribute(attr_key_t::center_point_box, (int32_t)0);
        return true;
    }

    bool reshape() override {
        // Output is [num_selected, 3] with int64
        // We don't know num_selected yet, defer to exec
        // For now, allocate max possible
        const tensor_t* boxes = inputs[0];
        int N = boxes->dims[0];
        int num_boxes = boxes->dims[1];
        int max_output = num_boxes * N;
        if (inputs.size() > 2 && inputs[2]->ndata > 0) {
            int64_t mo = ((const int64_t*)inputs[2]->data)[0];
            if (mo > 0 && mo < max_output)
                max_output = (int)mo;
        }
        // Count classes
        int num_classes = 1;
        if (inputs.size() > 1) {
            num_classes = inputs[1]->dims[1];
        }
        max_output *= num_classes;
        small_vector<int> dims(2);
        dims[0] = max_output;
        dims[1] = 3;
        return outputs[0]->reshape(dims, NNR_DATA_TYPE_INT64);
    }

    bool exec() override {
        const tensor_t* boxes_t = inputs[0];
        const tensor_t* scores_t = inputs[1];
        if (!boxes_t->data || !scores_t->data || boxes_t->ndim < 2 || scores_t->ndim < 2)
            return false;

        int64_t max_output_boxes = 0;
        float iou_threshold = 0.0f;
        float score_threshold = -std::numeric_limits<float>::infinity();

        if (inputs.size() > 2 && inputs[2] && inputs[2]->ndata > 0 && inputs[2]->data)
            max_output_boxes = ((const int64_t*)inputs[2]->data)[0];
        if (inputs.size() > 3 && inputs[3] && inputs[3]->ndata > 0 && inputs[3]->data)
            iou_threshold = ((const float*)inputs[3]->data)[0];
        if (inputs.size() > 4 && inputs[4] && inputs[4]->ndata > 0 && inputs[4]->data)
            score_threshold = ((const float*)inputs[4]->data)[0];

        const float* boxes = (const float*)boxes_t->data;
        const float* scores = (const float*)scores_t->data;

        int batch_size = boxes_t->dims[0];
        int num_boxes = boxes_t->dims[1];
        int num_classes = scores_t->dims[1];
        if (max_output_boxes <= 0) max_output_boxes = num_boxes;

        struct Result { int64_t batch, cls, idx; };
        std::vector<Result> results;

        // Decode one box into normalized corners (xmin <= xmax, ymin <= ymax) and area.
        // center_point_box: 0 = [y1,x1,y2,x2], 1 = [cx,cy,w,h].
        auto decode_box = [&](const float* p, float& xmin, float& ymin, float& xmax, float& ymax, float& area) {
            float a0 = p[0], a1 = p[1], a2 = p[2], a3 = p[3];
            if (center_point_box) {
                float hw = a2 * 0.5f, hh = a3 * 0.5f;
                xmin = a0 - hw; xmax = a0 + hw;
                ymin = a1 - hh; ymax = a1 + hh;
            } else {
                ymin = a0; xmin = a1; ymax = a2; xmax = a3;
                if (xmin > xmax) std::swap(xmin, xmax);
                if (ymin > ymax) std::swap(ymin, ymax);
            }
            area = (xmax - xmin) * (ymax - ymin);
        };

        // Reusable per-class buffers (cleared/resized in-place each class).
        struct Scored { float score; int idx; };
        std::vector<Scored> cand;
        cand.reserve(num_boxes);
        std::vector<float> sel_xmin, sel_ymin, sel_xmax, sel_ymax, sel_area;
        sel_xmin.reserve((size_t)max_output_boxes);
        sel_ymin.reserve((size_t)max_output_boxes);
        sel_xmax.reserve((size_t)max_output_boxes);
        sel_ymax.reserve((size_t)max_output_boxes);
        sel_area.reserve((size_t)max_output_boxes);

        for (int b = 0; b < batch_size; ++b) {
            const float* b_boxes = boxes + b * num_boxes * 4;
            for (int c = 0; c < num_classes; ++c) {
                const float* c_scores = scores + (b * num_classes + c) * num_boxes;

                // Build (score, idx) candidates above threshold, then heapify in O(N).
                cand.clear();
                for (int i = 0; i < num_boxes; ++i) {
                    float s = c_scores[i];
                    if (s > score_threshold)
                        cand.push_back({s, i});
                }
                if (cand.empty()) continue;
                auto cmp = [](const Scored& a, const Scored& b) {
                    return a.score < b.score || (a.score == b.score && a.idx > b.idx);
                };
                std::make_heap(cand.begin(), cand.end(), cmp);

                sel_xmin.clear(); sel_ymin.clear();
                sel_xmax.clear(); sel_ymax.clear(); sel_area.clear();

                while (!cand.empty() && (int64_t)sel_xmin.size() < max_output_boxes) {
                    int i = cand.front().idx;
                    std::pop_heap(cand.begin(), cand.end(), cmp);
                    cand.pop_back();

                    float x1_i, y1_i, x2_i, y2_i, area_i;
                    decode_box(b_boxes + i * 4, x1_i, y1_i, x2_i, y2_i, area_i);
                    if (area_i <= 0.f) continue;

                    bool keep = true;
                    const int n_sel = (int)sel_xmin.size();
                    const float* sx1 = sel_xmin.data();
                    const float* sy1 = sel_ymin.data();
                    const float* sx2 = sel_xmax.data();
                    const float* sy2 = sel_ymax.data();
                    const float* sA  = sel_area.data();
                    for (int k = 0; k < n_sel; ++k) {
                        // Early-exit on no x-overlap (ORT-style): skips the rest before
                        // touching y coords / area math.
                        float ix1 = x1_i > sx1[k] ? x1_i : sx1[k];
                        float ix2 = x2_i < sx2[k] ? x2_i : sx2[k];
                        if (ix2 <= ix1) continue;
                        float iy1 = y1_i > sy1[k] ? y1_i : sy1[k];
                        float iy2 = y2_i < sy2[k] ? y2_i : sy2[k];
                        if (iy2 <= iy1) continue;
                        float inter = (ix2 - ix1) * (iy2 - iy1);
                        float uni = area_i + sA[k] - inter;
                        if (uni <= 0.f) continue;
                        if (inter > iou_threshold * uni) { keep = false; break; }
                    }

                    if (keep) {
                        results.push_back({b, c, i});
                        sel_xmin.push_back(x1_i);
                        sel_ymin.push_back(y1_i);
                        sel_xmax.push_back(x2_i);
                        sel_ymax.push_back(y2_i);
                        sel_area.push_back(area_i);
                    }
                }
            }
        }

        // Shape the output to the actual detection count. ONNX spec: output is
        // dynamic shape (num_selected, 3); downstream ops re-run reshape() each
        // graph iteration so they see the current shape.
        int nresults = (int)results.size();
        int dims[2] = {nresults > 0 ? nresults : 0, 3};
        if (!outputs[0]->reshape(std::span<const int>(dims, 2), NNR_DATA_TYPE_INT64))
            return false;
        int64_t* py = (int64_t*)outputs[0]->data;
        for (int i = 0; i < nresults; ++i) {
            py[i * 3 + 0] = results[i].batch;
            py[i * 3 + 1] = results[i].cls;
            py[i * 3 + 2] = results[i].idx;
        }
        return true;
    }
};

} // namespace

// @nnr-meta-op mt=no
operator_t* resolver_default_op_NonMaxSuppression(int opset, pool_t& pool)
{
    return pool_new<NonMaxSuppression_operator>(pool);
}

} // namespace nnr
