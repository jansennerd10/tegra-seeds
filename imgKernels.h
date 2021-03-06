#ifndef IMG_KERNELS_H
#define IMG_KERNELS_H

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>

namespace bcvgpu
{
    void cvtGreyscale(cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst);
    double calcThreshold(cv::gpu::GpuMat& hist);
    int countSeeds(cv::gpu::GpuMat& binImg);
}

#endif