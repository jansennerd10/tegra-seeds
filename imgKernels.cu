#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#endif

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
    
    /**
     * Only works in a single-block configuration!!!
     * This is OK because we would only want to use this kernel on an array of size 256.
     */
    __global__ void calcThresholdKernel(int * histData, int * weighted, int length, int * threshold)
    {
        __shared__ int minDiff;
        int thresh = blockIdx.x * blockDim.x + threadIdx.x;
        if(thresh >= length) return;
        
        weighted[thresh] = histData[thresh] * thresh;
        if(thresh == 0)
        {
            minDiff = 256;
        }
        __syncthreads();
        
        int sum = 0;
        int pCount = 0;
        int ave;
        int i;
        
        //FUTURE OPTIMIZATION: Compute low and high averages at the same time?
        for(i = 0; i < thresh + 1; i++)
        {
            sum += weighted[i];
            pCount += histData[i];
        }
        ave = sum / pCount;
        
        sum = 0;
        pCount = 0;
        for(; i < length; i++)
        {
            sum += weighted[i];
            pCount += histData[i];
        }
        ave = (ave + sum / pCount) / 2 + 1;
        
        ave = __sad(ave, thresh, 0); // __sad(x, y, z) computes |x-y|+z; there is no abs() function in CUDA
        
        if(ave < minDiff)
        {
            int old = atomicMin(&minDiff, ave);
            if(old < ave) return;
        }
        else
        {
            return;
        }
        __syncthreads();
        
        //This last step is non-deterministic if we had more than one threshold candidate with the same diff score.
        //However, in this case all candidates are considered equally good, so we don't care which one eventually gets used.
        if(ave == minDiff) *threshold = thresh;
    }
    
    
    double calcThreshold(cv::gpu::GpuMat& hist)
    {
        int * weighted;
        cudaMalloc((void **) &weighted, sizeof(int) * 256);
        int * threshold;
        cudaMalloc((void **) &threshold, sizeof(int));
        int host_threshold;
        calcThresholdKernel<<<1, 256>>>((int *)hist.data, weighted, 256, threshold);
        
        cudaMemcpy(&host_threshold, threshold, sizeof(int), cudaMemcpyDeviceToHost);
        
        #ifdef DEBUG
        int * localWeighted = new int[256];
        cudaMemcpy(localWeighted, weighted, sizeof(int) * 256, cudaMemcpyDeviceToHost);
        for(int i = 0; i < 256; i++)
        {
            printf("%d, ", localWeighted[i]);
        }
        printf("\n");
        printf("%s\n", cudaGetErrorName(cudaGetLastError()));
        #endif
        
        return (double)host_threshold;
    }
}