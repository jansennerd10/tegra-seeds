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
        //NTSC luma coefficients
        dstData[blockIdx.y * dstStep + x] = (0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]);
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
        
        long sum = 0;
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
    
    __global__ void countContiguousKernel(uchar * img, int * totalCount, int width, int height, int step, int chunkWidth, int chunkHeight)
    {
        int hOffset = chunkWidth * threadIdx.x;
        int vOffset = chunkHeight * blockIdx.x;
        
        int row, col;
        int seedCount = 0;
        uchar pixel, lastPixel, shoulderPixel, lastShoulder;
        #ifdef DEBUG
        bool didAdd = false;
        #endif
        for(row = vOffset; row < height && row < vOffset + chunkHeight; row++)
        {
            //Get the pixel and shoulder immediately to the left of our chunk, if applicable.
            //This will prevent us from double counting seeds that cross chunk borders.
            pixel = hOffset == 0 ? 255 : img[row * step + hOffset - 1];
            shoulderPixel = (row == 0 || hOffset == 0) ? 255 : img[(row - 1) * step + hOffset - 1];
            for(col = hOffset; col < width && col < hOffset + chunkWidth; col++)
            {
                lastPixel = pixel;
                lastShoulder = shoulderPixel;
                pixel = img[row * step + col];
                shoulderPixel = row == 0 ? 255 : img[(row - 1) * step + col];
                
                //FOR EACH PIXEL:
                //If it's white, do nothing.
                if(pixel == 255) continue;
                
                //Otherwise, if it's the first black pixel in a segment of black pixels,
                //greedily assume the segment is part of a new seed.
                if(lastPixel == 255)
                {
                    #ifdef DEBUG
                    didAdd = true;
                    #endif
                    seedCount++;
                }

                //If we encounter a black pixel over the shoulder, then our previous greedy assumption was wrong.
                //We decrement for every new segment of black pixels over the shoulder, to account for the horseshoe case
                //(and the positive-slope left boundary case).
                //The algorithm will be incorrect for the donut case, but we donut care about that because - worst case -
                //we will count the seed twice, and 99.9% of seeds do not look like donuts.
                if(shoulderPixel == 0 && (lastPixel == 255 || shoulderPixel != lastShoulder))
                {
                    seedCount--;
                    #ifdef DEBUG
                    if(!didAdd)
                    {
                        printf("Thread %d in block %d at img[%d][%d] subtracted under suspicious circumstances.\n", threadIdx.x, blockIdx.x, row, col);
                    }
                    didAdd = false;
                    #endif
                }
            }
            #ifdef DEBUG
            didAdd = false;
            #endif
        }
        
        atomicAdd(totalCount, seedCount);
    }
    
    int countSeeds(cv::gpu::GpuMat& binImg)
    {
        int * seeds;
        int host_seeds;
        cudaMalloc(&seeds, sizeof(int));
        cudaMemset(seeds, 0, sizeof(int));
        dim3 threadsPerBlock((binImg.size().width + 255) / 256);
        dim3 numBlocks((binImg.size().height + 255) / 256);
        countContiguousKernel<<<numBlocks, threadsPerBlock>>>(binImg.data, seeds, binImg.size().width, binImg.size().height, binImg.step, 256, 256);
        cudaMemcpy(&host_seeds, seeds, sizeof(int), cudaMemcpyDeviceToHost);
        return host_seeds;
    }
}