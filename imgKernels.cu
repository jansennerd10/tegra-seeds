#include "imgKernels.h"

#define THREADS_PER_BLOCK 256 //has to be a power of 2

namespace bcvgpu
{
    __global__ void cvtGreyscaleKernel(uchar * srcData, uchar * dstData, int width, int height, int srcStep, int dstStep)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if(x >= width) return;
        
        uchar * pixel = srcData + blockIdx.y * srcStep + x * 3;
        dstData[blockIdx.y * dstStep + x] = ((short)pixel[0] + pixel[1] + pixel[2]) / 3;
    }
    
    void cvtGreyscale(cv::gpu::GpuMat& src, cv::gpu::GpuMat& dst)
    {
        if(src.type() != CV_8UC3)
        {
            std::cout << "Expected src to be of type CV_8UC3." << std::endl;
        }
        
        //If dst isn't already of the correct type, have it allocate space for the height and width of the source image but with 1 channel.
        dst.create(src.size(), CV_8UC1);
        
        dim3 numBlocks((src.size().width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, src.size().height);
        std::cout << "Thread blocks: (" << numBlocks.x << ", " << numBlocks.y << ")" << std::endl;
        cvtGreyscaleKernel<<<numBlocks, THREADS_PER_BLOCK>>>(src.data, dst.data, src.size().width, src.size().height, src.step, dst.step);
    }
    
    __global__ void calcSumKernel(int * values, int n, long * out)
    {
        __shared__ long workingArray[THREADS_PER_BLOCK];
        int split = THREADS_PER_BLOCK / 2; //THREADS_PER_BLOCK has to be a power of 2, so this is OK
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n) return;
        
        workingArray[threadIdx.x] = values[i];
        __syncthreads();
        
        while(split > 0)
        {
            if(threadIdx.x < split || i < n - THREADS_PER_BLOCK + split) return;
            workingArray[threadIdx.x] += workingArray[threadIdx.x + split];
            split >>= 1;
            _syncthreads();
        }
        
        atomicAdd(out, workingArray[0]);
    }
    
    __global__ void calcWeightedVectorKernel(int * inValues, int * outValues, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n)
        {
            outValues[i] = i * inValues;
        }
    }
    
    uchar calcThreshold(cv::gpu::GpuMat& hist)
    {
        cv::gpu::GpuMat weightedHist;
        weightedHist.create(hist.size(), hist.type());
        
        
}