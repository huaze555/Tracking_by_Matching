//
// Created by zxj on 20-7-1.
//
#include <opencv2/cudawarping.hpp>
#include "DetectionNN.h"
#include "DarknetParser.h"
namespace tk {   namespace dnn {

using namespace std;
using namespace cv;


DetectionNN::DetectionNN(const string& weight_dir_path, const string& cfg_path, float conf_thresh, bool use_fp16, int n_batches)
        : confThreshold(conf_thresh),
          nBatches(n_batches) {

    string output_engine_path = cv::format("%s_fp%d_batch%d.engine",
            cfg_path.substr(0, cfg_path.find_last_of(".")).c_str(),
            use_fp16 ? 16 : 32,
            n_batches);

    if (!fileExist(output_engine_path.c_str())) {
        std::cout << output_engine_path << " isn't exists, generate new one\n";
        gen_engine_file(weight_dir_path, cfg_path, use_fp16, n_batches, output_engine_path);
    } else
        std::cout << "Loading exists " << output_engine_path << endl;

    //convert network to tensorRT
    netRT = new tk::dnn::NetworkRT(nullptr, output_engine_path.c_str());

    //tk::dnn::dataDim_t idim = netRT->input_dim;

    if (netRT->pluginFactory->n_yolos < 2) {
        FatalError("this is not yolo3")
    }

    for (int i = 0; i < netRT->pluginFactory->n_yolos; ++i) {
        YoloRT *yRT = netRT->pluginFactory->yolos[i];
        classes = yRT->classes;
        int num = yRT->num;
        int nMasks = yRT->n_masks;

        // make a yolo layer to interpret predictions
        yolo[i] = new tk::dnn::Yolo(nullptr, classes, nMasks, ""); // yolo without input and bias
        yolo[i]->mask_h = new dnnType[nMasks];
        yolo[i]->bias_h = new dnnType[num * nMasks * 2];
        memcpy(yolo[i]->mask_h, yRT->mask, sizeof(dnnType) * nMasks);
        memcpy(yolo[i]->bias_h, yRT->bias, sizeof(dnnType) * num * nMasks * 2);
        yolo[i]->input_dim = yolo[i]->output_dim = tk::dnn::dataDim_t(1, yRT->c, yRT->h, yRT->w);
    }

    dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, classes);


    cudaMalloc((void **) &input_data_d, nBatches * 3 * netRT->input_dim.w * netRT->input_dim.h * sizeof(float));

}


void DetectionNN::gen_engine_file(const string& weight_dir_path, const string& cfg_path, bool use_fp16, uint n_batch,
        const string& output_engine_path) {

    // parse darknet network
    tk::dnn::Network *net = tk::dnn::darknetParser(cfg_path, weight_dir_path, use_fp16);

    net->maxBatchSize = n_batch;

    //convert network to tensorRT
    tk::dnn::NetworkRT *netRT_ = new tk::dnn::NetworkRT(net, output_engine_path.c_str());
    net->releaseLayers();
    delete net;
    delete netRT_;
}

DetectionNN::~DetectionNN() {
    if (netRT) {
        delete netRT;
        netRT = nullptr;
    }

    cudaFree(input_data_d);

}


std::vector<std::vector<tk::dnn::box>> DetectionNN::detect(const std::vector<cv::cuda::GpuMat> &frames) {
    const int cur_batches = frames.size();
    if (cur_batches > nBatches) FatalError("A batch size greater than nBatches cannot be used")

    std::vector<cv::Size> originalSize;

    // preprocess
    preprocess(frames);

    for (int bi = 0; bi < cur_batches; ++bi) {
        originalSize.emplace_back(frames[bi].size());
    }

    // inference
    tk::dnn::dataDim_t dim = netRT->input_dim;
    dim.n = cur_batches;
    netRT->infer(dim, input_data_d);

    // postprocess
    std::vector<std::vector<tk::dnn::box>> batchDetected = postprocess(cur_batches, originalSize);

    return batchDetected;
}


void yolo_cuda_preprocess(const std::vector<cv::cuda::GpuMat>& imgs, float* data_d);

void DetectionNN::preprocess(const std::vector<cv::cuda::GpuMat>& frames) {
    vector<cv::cuda::GpuMat> input_frames(frames.size());

    for(int i = 0; i < frames.size(); ++i) {
        cv::cuda::resize(frames[i], input_frames[i], Size(netRT->input_dim.w, netRT->input_dim.h));
    }
    yolo_cuda_preprocess(input_frames, input_data_d);
}

vector<vector<tk::dnn::box>>
DetectionNN::postprocess(int batchsize, std::vector<cv::Size> originalSize) {
    vector<std::vector<tk::dnn::box>> batchDetected;

    //get yolo outputs
    dnnType *rt_out[netRT->pluginFactory->n_yolos];

    for (int bi = 0; bi < batchsize; ++bi) {

        for (int i = 0; i < netRT->pluginFactory->n_yolos; i++)
            rt_out[i] = (dnnType *) netRT->buffersRT[i + 1] + netRT->buffersDIM[i + 1].tot() * bi;

        float x_ratio = float(originalSize[bi].width) / netRT->input_dim.w;
        float y_ratio = float(originalSize[bi].height) / netRT->input_dim.h;

        // compute dets
        int nDets = 0;
        for (int i = 0; i < netRT->pluginFactory->n_yolos; i++) {
            yolo[i]->dstData = rt_out[i];
            yolo[i]->computeDetections(dets, nDets, netRT->input_dim.w, netRT->input_dim.h, confThreshold);
        }
        tk::dnn::Yolo::mergeDetections(dets, nDets, classes);

        // fill detected
        std::vector<tk::dnn::box> cur_detected;
        for (int j = 0; j < nDets; ++j) {
            tk::dnn::Yolo::box b = dets[j].bbox;

            int obj_class = -1;
            float prob = 0;

            for (int c = 0; c < classes; c++) {
                float tmp_score = dets[j].prob[c];
                if (tmp_score >= confThreshold && tmp_score > prob) {
                    obj_class = c;
                    prob = tmp_score;
                }
            }
            if(obj_class != -1) {
                // convert to image coords
                float x0 = (b.x - b.w / 2.) * x_ratio;
                float x1 = (b.x + b.w / 2.) * x_ratio;
                float y0 = (b.y - b.h / 2.) * y_ratio;
                float y1 = (b.y + b.h / 2.) * y_ratio;

                tk::dnn::box res;
                res.cl = obj_class;
                res.prob = prob;
                res.x = x0;
                res.y = y0;
                res.w = x1 - x0;
                res.h = y1 - y0;
                cur_detected.emplace_back(res);
            }
        }
        batchDetected.push_back(cur_detected);
    }

    return batchDetected;
}

}}

