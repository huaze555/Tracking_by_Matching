#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <opencv2/cudawarping.hpp>

using namespace cv;

namespace tk {   namespace dnn {


__global__ void yolo_cuda_preprocess_core(cv::cuda::PtrStepSz<uchar3> *imgs, int batchsize, float *data_d) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < imgs[0].cols && y < imgs[0].rows) {
        int INPUT_H = imgs[0].rows;
        int INPUT_W = imgs[0].cols;

        int LEN = INPUT_H * INPUT_W;
        for (int k = 0; k < batchsize; ++k) {
            uint prefix_pos = k * 3 * LEN;
            data_d[prefix_pos + 0 * LEN + y * INPUT_W + x] = imgs[k](y, x).z / 255.0;
            data_d[prefix_pos + 1 * LEN + y * INPUT_W + x] = imgs[k](y, x).y / 255.0;
            data_d[prefix_pos + 2 * LEN + y * INPUT_W + x] = imgs[k](y, x).x / 255.0;
//            data_d[prefix_pos + 0 * INPUT_H * INPUT_W + y * INPUT_W + x] = imgs[k](y, x).z / 255.0;
//            data_d[prefix_pos + 1 * INPUT_H * INPUT_W + y * INPUT_W + x] = imgs[k](y, x).y / 255.0;
//            data_d[prefix_pos + 2 * INPUT_H * INPUT_W + y * INPUT_W + x] = imgs[k](y, x).x / 255.0;
        }
    }
}

void yolo_cuda_preprocess(const std::vector<cv::cuda::GpuMat> &imgs, float *data_d) {
    int batchsize = imgs.size();
    CV_Assert(batchsize > 0);
    CV_Assert(imgs[0].type() == CV_8UC3);

    dim3 block(32, 8);
    dim3 grid((imgs[0].cols + block.x - 1) / block.x, (imgs[0].rows + block.y - 1) / block.y);


    cv::cuda::PtrStepSz<uchar3> *h_ptrs = new cv::cuda::PtrStepSz<uchar3>[batchsize];
    for (int i = 0; i < batchsize; ++i) {
        h_ptrs[i] = imgs[i];
    }

    cv::cuda::PtrStepSz<uchar3> *d_ptrs = NULL;
    cudaMalloc(&d_ptrs, batchsize * sizeof(cv::cuda::PtrStepSz<uchar3>));
    cudaMemcpy(d_ptrs, h_ptrs, batchsize * sizeof(cv::cuda::PtrStepSz<uchar3>), cudaMemcpyHostToDevice);

    yolo_cuda_preprocess_core << < grid, block >> > (d_ptrs, batchsize, data_d);
    cudaDeviceSynchronize();


    delete[] h_ptrs;
    cudaFree(d_ptrs);
}

}}
