#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
static const int wholeArraySize = 100000000;
static const int blockSize = 1024;
static const int gridSize = 24; // this number is hardware-dependent; usually #SM*2 is a good number.

__device__ bool lastBlock(int *counter)
{
    __threadfence(); // ensure that partial result is visible by all blocks
    int last = 0;
    if (threadIdx.x == 0)
        last = atomicAdd(counter, 1);
    return __syncthreads_or(last == gridDim.x - 1);
}

__device__ void sumNoncommSingleBlock(const int *gArr, int arraySize, int *out)
{
    int thIdx = threadIdx.x;
    __shared__ int shArr[blockSize * 2];
    __shared__ int offset;
    shArr[thIdx] = thIdx < arraySize ? gArr[thIdx] : 0;
    if (thIdx == 0)
        offset = blockSize;
    __syncthreads();
    while (offset < arraySize)
    { // uniform
        shArr[thIdx + blockSize] = thIdx + offset < arraySize ? gArr[thIdx + offset] : 0;
        __syncthreads();
        if (thIdx == 0)
            offset += blockSize;
        int sum = shArr[2 * thIdx] + shArr[2 * thIdx + 1];
        __syncthreads();
        shArr[thIdx] = sum;
    }
    __syncthreads();
    for (int stride = 1; stride < blockSize; stride *= 2)
    { // uniform
        int arrIdx = thIdx * stride * 2;
        if (arrIdx + stride < blockSize)
            shArr[arrIdx] += shArr[arrIdx + stride];
        __syncthreads();
    }
    if (thIdx == 0)
        *out = shArr[0];
}

__global__ void sumNoncommMultiBlock(const int *gArr, int *out, int *lastBlockCounter)
{
    int arraySizePerBlock = wholeArraySize / gridSize;
    const int *gArrForBlock = gArr + blockIdx.x * arraySizePerBlock;
    int arraySize = arraySizePerBlock;
    if (blockIdx.x == gridSize - 1)
        arraySize = wholeArraySize - blockIdx.x * arraySizePerBlock;
    sumNoncommSingleBlock(gArrForBlock, arraySize, &out[blockIdx.x]);
    if (lastBlock(lastBlockCounter))
        sumNoncommSingleBlock(out, gridSize, out);
}

int main()
{
    int *A = new int[100000000]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int *Ad;
    int sum = 0;
    int* sumd;
    int *lastBlock;

    cudaMalloc((void**)&Ad, wholeArraySize*sizeof(int));
    cudaMalloc((void**)&sumd, sizeof(int));
    cudaMalloc((void**)&lastBlock, sizeof(int));
    cudaMemcpy(Ad,A, sizeof(int)*wholeArraySize, cudaMemcpyHostToDevice);

    sumNoncommMultiBlock<<<gridSize, blockSize>>>(Ad, sumd, lastBlock);

    cudaMemcpy(&sum, sumd, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "The sum is " << sum;
    cudaFree(Ad);
    cudaFree(sumd);
    cudaFree(lastBlock);

    delete[] A;
}