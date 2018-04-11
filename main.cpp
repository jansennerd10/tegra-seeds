#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include "imgKernels.h"

int main(int argc, char * argv[])
{
    cv::Mat image, hist;
    image = cv::imread("sample.jpg", CV_LOAD_IMAGE_COLOR);
    
    if(!image.data)
    {
        std::cout << "Could not open or find the image." << std::endl;
        return -1;
    }
    
    std::cout << "Read the image. Matrix size is (" << image.size().width << ", " << image.size().height << ") with " << image.channels() << " channels." << std::endl;
    
    cv::gpu::GpuMat gpuImg, gpuGrey, gpuHist;
    gpuImg.upload(image);
    
    std::cout << "Copied image to GPGPU memory." << std::endl;
    
    bcvgpu::cvtGreyscale(gpuImg, gpuGrey);
    cv::gpu::calcHist(gpuGrey, gpuHist);
}