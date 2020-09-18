//
// Created by zxj on 20-6-30.
//

#ifndef TKDNN_YOLO_TRT_H
#define TKDNN_YOLO_TRT_H

#include "DetectionNN.h"
#include <opencv2/dnn.hpp>

struct DetectBox;


namespace tk { namespace dnn {


using std::vector;

class Yolo_TRT {
public:
    Yolo_TRT(const string& weight_dir_path, const string& cfg_path, float prob_thresh, bool use_fp16, uint n_batch = 1);

    vector<vector<DetectBox>> detect(const vector<cv::cuda::GpuMat>& imgs);


private:
    std::unique_ptr<DetectionNN> Yolo;
};

}}
#endif //TKDNN_YOLO_TRT_H
