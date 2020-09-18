//
// Created by zxj on 20-9-18.
//

#ifndef TRACKING_BY_MATCHING_DETECTBOX_H
#define TRACKING_BY_MATCHING_DETECTBOX_H

#include <opencv2/core.hpp>
// ObjectDetect BBox  , Yolo, Faster R-CNN, etc
struct DetectBox {
    DetectBox(cv::Rect r, uint id, float conf): box(r), class_id(id), confidence(conf){}
    cv::Rect box;
    uint class_id;
    float confidence;
};



#endif //TRACKING_BY_MATCHING_DETECTBOX_H
