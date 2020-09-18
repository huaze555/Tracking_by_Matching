#include <iostream>
#include "TrackerByMatcher.hpp"
#include "Yolo_TRT.h"
#include <opencv2/opencv.hpp>
#include <map>
#include "DetectBox.h"
using namespace std;
using namespace cv;

std::map<uint, Scalar> COLORES;

RNG rng(time(0));

int main() {
    tk::dnn::Yolo_TRT detector("../models/export_weights", "../models/pcsm-yolov4.cfg", 0.6, false);


    VideoCapture capture("../../PCSM/videos/10.163.12.42.mp4");
    double fps = capture.get(CAP_PROP_FPS);

    tbm::TrackerByMatcher tracker(fps, {0});

    Mat frame;

    int frame_counter = 0;

    while(capture.read(frame)){
        ++frame_counter;

        cv::cuda::GpuMat gpu_frame(frame);

        vector<DetectBox> bbox = detector.detect({gpu_frame})[0];

        tbm::TrackedObjects a = tracker.track(frame, bbox, frame_counter);

        std::unordered_map<size_t, std::vector<cv::Point>> results = tracker.getTrackingPoints();
        for(const auto& A : results){
            uint tracking_id = A.first;
            vector<Point> points = A.second;

            if(COLORES.find(tracking_id) == COLORES.end()){
                Scalar new_color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                COLORES[tracking_id] = new_color;
            }

            for(const Point& p : points)
                circle(frame, p, 3, COLORES[tracking_id], -1);
        }

        cv::resize(frame, frame, Size(), 0.7, 0.7);
        imshow("", frame);
        waitKey(1);
    }

    return 0;
}