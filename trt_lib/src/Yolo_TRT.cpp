//
// Created by zxj on 20-6-30.
//

#include "DetectBox.h"
#include "Yolo_TRT.h"
#include <unistd.h>
using namespace std;

namespace tk { namespace dnn {


    Yolo_TRT::Yolo_TRT(const string& weight_dir_path, const string& cfg_path, float prob_thresh, bool use_fp16, uint n_batch) {
        this->Yolo.reset(new DetectionNN(weight_dir_path, cfg_path, prob_thresh, use_fp16, n_batch));
    }

    vector<vector<DetectBox>> Yolo_TRT::detect(const std::vector<cv::cuda::GpuMat>& imgs) {
        vector<vector<DetectBox>> results;


        vector<vector<tk::dnn::box>> batchDetected = Yolo->detect(imgs);
        for(uint i = 0; i < batchDetected.size(); ++i){
            vector<tk::dnn::box> per_img_boxes = batchDetected[i];
            vector<DetectBox> per_img_detect_boxes;
            for(const tk::dnn::box& box : per_img_boxes) {
                cv::Rect rect(box.x, box.y, box.w, box.h);
                per_img_detect_boxes.emplace_back(
                        DetectBox(rect & cv::Rect(0, 0, imgs[i].cols, imgs[i].rows), box.cl, box.prob));
            }
            results.emplace_back(per_img_detect_boxes);
        }
        return results;
    }

}}