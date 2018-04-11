#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>

namespace bcvgpu
{
    void cvtGreyscale(cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst)
    {
        if(src.type() != CV_8UC3)
        {
            std::cout << "Expected src to be of type CV_8UC3." << std::endl;
        }
        
        //If dst isn't already of the correc type, have it allocate space for the height and width of the source image but with 1 channel.
        dst.create(src.size().height, src.size().width, CV_8UC1);
    }
}