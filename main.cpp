#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include "imgKernels.h"
#include <sys/time.h>

#define usec(tv) (tv.tv_sec * 1000000 + tv.tv_usec)

int main(int argc, char * argv[])
{
    struct timeval begin, end, diff;
    cv::Mat image, hist;
    
    if(argc != 2)
    {
        std::cout << "Usage: main <filename>" << std::endl;
        return -1;
    }
    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    
    if(!image.data)
    {
        std::cout << "Could not open or find the image '" << argv[1] << "'." << std::endl;
        return -1;
    }
    
    std::cout << "Read the image. Matrix size is (" << image.size().width << ", " << image.size().height << ") with " << image.channels() << " channels." << std::endl;
    
    cv::gpu::GpuMat gpuImg, gpuGrey, gpuHist, gpuBinary, gpuDilated;
    gpuImg.upload(image);
    
    std::cout << "Copied image to GPGPU memory." << std::endl;
    
    bcvgpu::cvtGreyscale(gpuImg, gpuGrey);
    
    std::cout << "Converted to grayscale." << std::endl;
    
    cv::gpu::calcHist(gpuGrey, gpuHist);
    
    std::cout << "Calculated histogram." << std::endl;
    
    double threshold = bcvgpu::calcThreshold(gpuHist);
    std::cout << "Threshold is " << threshold << std::endl;
    
    cv::gpu::threshold(gpuGrey, gpuBinary, threshold, 255, cv::THRESH_BINARY);
    gpuBinary.download(image);
    cv::imwrite("sample_binary.png", image);
    
    cv::gpu::dilate(gpuBinary, gpuDilated, cv::Mat::ones(4, 4, CV_8UC1));
    gpuDilated.download(image);
    cv::imwrite("sample_dilated.png", image);
    
    std::cout << "Seeds: " << bcvgpu::countSeeds(gpuDilated) << std::endl;
}