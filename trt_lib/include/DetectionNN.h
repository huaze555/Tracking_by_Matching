#ifndef DETECTIONNN_H
#define DETECTIONNN_H

#include <iostream>
#include <stdlib.h>    
#include <unistd.h>
#include "utils.h"

#include <opencv2/opencv.hpp>

#include "tkdnn.h"



namespace tk {  namespace dnn {
using std::string;

class DetectionNN {
public:

    DetectionNN(const string &weight_dir_path, const string &cfg_path, float conf_thresh, bool use_fp16,
                int n_batches = 1);

    ~DetectionNN();

    std::vector<std::vector<tk::dnn::box>> detect(const std::vector<cv::cuda::GpuMat> &frames);


private:


    void gen_engine_file(const string &weight_dir_path, const string &cfg_path, bool use_fp16, uint n_batch,
                         const string &output_engine_path);
    std::vector<std::vector<tk::dnn::box>> postprocess(int batchsize, std::vector<cv::Size> originalSize);

    void preprocess(const std::vector<cv::cuda::GpuMat>& frames);






    tk::dnn::NetworkRT *netRT = nullptr;
    tk::dnn::Yolo::detection *dets = nullptr;
    tk::dnn::Yolo *yolo[3];
    int classes;
    float confThreshold; /*threshold on the confidence of the boxes*/
    int nBatches;


    float* input_data_d;   //如果使用cuda进行预处理，此为gpu数据指针，长度为 batch * 3 * H * W * sizeof(float)
};


}
}

#endif /* DETECTIONNN_H*/
