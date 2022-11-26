#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
// Source: https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-
static const int N = 8192;
static const int blockSize = 1024;
static const int gridSize = 8; // this number is hardware-dependent; usually #SM*2 is a good number.

__global__ void sumCommMultiBlock(
    const double *gArrX, const double *gArrY,
    const double *gArrXY, const double *gArrXsq,
    const double *gArrYsq, double arraySize,
    double *gOutX, double *gOutY, double *gOutXY,
    double *gOutXsq, double *gOutYsq)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x * blockSize;
    const int gridSize = blockSize * gridDim.x;
    double sumX = 0;
    double sumY = 0;
    double sumXY = 0;
    double sumXsq = 0;
    double sumYsq = 0;

    for (int i = gthIdx; i < arraySize; i += gridSize)
    {
        sumX += gArrX[i];
        sumY += gArrY[i];
        sumXY += gArrXY[i];
        sumXsq += gArrXsq[i];
        sumYsq += gArrYsq[i];
    }

    __shared__ double shArrX[blockSize];
    __shared__ double shArrY[blockSize];
    __shared__ double shArrXY[blockSize];
    __shared__ double shArrXsq[blockSize];
    __shared__ double shArrYsq[blockSize];

    shArrX[thIdx] = sumX;
    shArrY[thIdx] = sumY;
    shArrXY[thIdx] = sumXY;
    shArrXsq[thIdx] = sumXsq;
    shArrYsq[thIdx] = sumYsq;

    __syncthreads();

    for (int size = blockSize / 2; size > 0; size /= 2)
    { // uniform
        if (thIdx < size)
        {
            shArrX[thIdx] += shArrX[thIdx + size];
            shArrY[thIdx] += shArrY[thIdx + size];
            shArrXY[thIdx] += shArrXY[thIdx + size];
            shArrXsq[thIdx] += shArrXsq[thIdx + size];
            shArrYsq[thIdx] += shArrYsq[thIdx + size];
        }
        __syncthreads();
    }

    if (thIdx == 0)
    {
        gOutX[blockIdx.x] = shArrX[0];
        gOutY[blockIdx.x] = shArrY[0];
        gOutXY[blockIdx.x] = shArrXY[0];
        gOutXsq[blockIdx.x] = shArrXsq[0];
        gOutYsq[blockIdx.x] = shArrYsq[0];
    }
}

__host__ double calcCorrCoefficient(double *X, double *Y, double *XY, double *Xsq, double *Ysq)
{
    double *dev_arrX;
    double *dev_arrY;
    double *dev_arrXY;
    double *dev_arrXsq;
    double *dev_arrYsq;

    cudaMalloc((void **)&dev_arrX, N * sizeof(double));
    cudaMemcpy(dev_arrX, X, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_arrY, N * sizeof(double));
    cudaMemcpy(dev_arrY, Y, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_arrXY, N * sizeof(double));
    cudaMemcpy(dev_arrXY, XY, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_arrXsq, N * sizeof(double));
    cudaMemcpy(dev_arrXsq, Xsq, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_arrYsq, N * sizeof(double));
    cudaMemcpy(dev_arrYsq, Ysq, N * sizeof(double), cudaMemcpyHostToDevice);

    double sumX;
    double sumY;
    double sumXY;
    double sumXsq;
    double sumYsq;

    double *dev_sumX;
    double *dev_sumY;
    double *dev_sumXY;
    double *dev_sumXsq;
    double *dev_sumYsq;

    cudaMalloc((void **)&dev_sumX, sizeof(double) * gridSize);
    cudaMalloc((void **)&dev_sumY, sizeof(double) * gridSize);
    cudaMalloc((void **)&dev_sumXY, sizeof(double) * gridSize);
    cudaMalloc((void **)&dev_sumXsq, sizeof(double) * gridSize);
    cudaMalloc((void **)&dev_sumYsq, sizeof(double) * gridSize);

    sumCommMultiBlock<<<gridSize, blockSize>>>(dev_arrX, dev_arrY, dev_arrXY, dev_arrXsq, dev_arrYsq, N, dev_sumX, dev_sumY, dev_sumXY, dev_sumXsq, dev_sumYsq);
    // dev_out now holds the partial result
    sumCommMultiBlock<<<1, blockSize>>>(dev_sumX, dev_sumY, dev_sumXY, dev_sumXsq, dev_sumYsq, gridSize, dev_sumX, dev_sumY, dev_sumXY, dev_sumXsq, dev_sumYsq);
    // dev_out[0] now holds the final result
    cudaDeviceSynchronize();

    cudaMemcpy(&sumX, dev_sumX, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumY, dev_sumY, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumXY, dev_sumXY, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumXsq, dev_sumXsq, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumYsq, dev_sumYsq, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_arrX);
    cudaFree(dev_arrY);
    cudaFree(dev_arrXY);
    cudaFree(dev_arrXsq);
    cudaFree(dev_arrYsq);

    cudaFree(dev_sumX);
    cudaFree(dev_sumY);
    cudaFree(dev_sumXY);
    cudaFree(dev_sumXsq);
    cudaFree(dev_sumYsq);

    double corr = (double)(N * sumXY - sumX * sumY) / sqrt((N * sumXsq - sumX * sumX) * (N * sumYsq - sumY * sumY));
    return corr;
}

int main()
{
    double *F = new double[N]{0};
    double* G = new double[N]{0};
    double* FG = new double[N]{0};
    double* FF = new double[N]{0};
    double* GG = new double[N]{0};

    F[0] = N;

    // Start sequence of G with 1
    G[0] = 1;
    
    FG[0] = F[0] * G[0];

    FF[0] = F[0] * F[0];

    GG[0] = G[0] * G[0];

    // Initialize F with values starting from 8192 down to 1
    // Initialize G with values starting from 1 up to 8192
    for (int n = 1; n < N; ++n)
    {
        // f(n) = f(n-1) - 1
        F[n] = F[n-1] - 1;
        // g(n) = g(n-1) + 1
        G[n] = G[n-1] + 1;

        FG[n] = F[n] * G[n]; 

        FF[n] = F[n] * F[n];

        GG[n] = G[n] * G[n];
    }

    double corr = calcCorrCoefficient(F, G, FG, FF, GG);

    std::cout << "The corr is " << corr;
}