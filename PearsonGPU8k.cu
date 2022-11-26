//***************************************************************************
// Pearson Product Moment Correlation Coefficient GPU (8k)
// Name: Angel Badillo, and James Nealley
// GPU Programming Date: 11/28/22
//***************************************************************************
// How to run:
// This program is to be run on the TACC cluster using the SBATCH
// shell script named "PearsonGPU16kScript".
// The command to be run in the bash terminal is:
// sbatch PearsonGPU16kScript
//
// Description:
// This program calculates the Pearson Product Moment Correlation Coefficient 
// (PPMCC) between 2 arrays, X and Y and prints out the result. The arrays X 
// and Y contain 8192 elements each, the arrays do not contain extreme 
// outliers and they also satisfy the assumptions for the PPMCC.
// First, the arrays for all values of XY, X^2, and Y^2 must be computed prior
// to performing the parallel reduction. Then, an initial call to the kernel
// will produce a partial result of the reduction, but a second call
// will to the kernel will add all the parts into one singular element.
// Once all the sums have been calculated, the PPMCC can finally be calculated.
//
// Source: 
// https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-
//
// We used the code form the section titled "Multi-block parallel reduction 
// for commutative operator". Minor modifications were performed to the code 
// from this source for the purposes of using it in our program. We altered 
// the code to accept values of type double rather than of type integer, 
// since the data can be composed of rational numbers. Furthermore,
// We wanted to perform multiple summations at once, thus we altered
// the code to perform parallel reduction on multiple arrays at once.
// Another minor adjustment was changing the grid size and size of the
// arrays.
//***************************************************************************
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timer.h"

using namespace std;

// Size of arrays
static const int N = 8192;

// Number of threads per block
static const int blockSize = 1024;

// Number of blocks
static const int gridSize = 8;

/**
 * @brief Kernel that performs parallel reduction on 5 arrays to calculate the sums of
 * all their elements. The first call reduces the arrays to an array
 * the size of the total number of blocks used, which is a partial sum. The second call 
 * reduces the partials sums to total sums.
 * 
 * @param Xd Array of doubles to be reduced.
 * @param Yd Array of doubles to be reduced.
 * @param XYd Array of doubles to be reduced.
 * @param Xsqd Array of doubles to be reduced.
 * @param Ysqd Array of doubles to be reduced.
 * @param arraySize Size of all the arrays.
 * @param sumXd Reduction array Xd (sum).
 * @param sumYd Reduction array Yd (sum).
 * @param sumXYd Reduction array XYd (sum).
 * @param sumXsqd Reduction array Xsqd (sum).
 * @param sumYsqd Reduction array Ysqd (sum).
 * @return void
 */
__global__ void sumCommMultiBlock(
    const double *Xd, const double *Yd,
    const double *XYd, const double *Xsqd,
    const double *Ysqd, int arraySize,
    double *sumXd, double *sumYd, double *sumXYd,
    double *sumXsqd, double *sumYsqd)
{
    // Local thread index
    int thIdx = threadIdx.x;

    // Global thread index
    int gthIdx = thIdx + blockIdx.x * blockSize;

    // Total Number of threads in the grid
    const int totalThreads = blockSize * gridDim.x;
    
    // Holds partial sums
    double sumX = 0;
    double sumY = 0;
    double sumXY = 0;
    double sumXsq = 0;
    double sumYsq = 0;

    // Non-cyclic data distribution in the event that
    // array size is greater than total number of threads
    for (int i = gthIdx; i < arraySize; i += totalThreads)
    {
        sumX += Xd[i];
        sumY += Yd[i];
        sumXY += XYd[i];
        sumXsq += Xsqd[i];
        sumYsq += Ysqd[i];
    }

    // Use of shared memory to provide significant speedup
    __shared__ double shArrX[blockSize];
    __shared__ double shArrY[blockSize];
    __shared__ double shArrXY[blockSize];
    __shared__ double shArrXsq[blockSize];
    __shared__ double shArrYsq[blockSize];

    // Load partial sums to respective shared memory arrays
    shArrX[thIdx] = sumX;
    shArrY[thIdx] = sumY;
    shArrXY[thIdx] = sumXY;
    shArrXsq[thIdx] = sumXsq;
    shArrYsq[thIdx] = sumYsq;

    // Ensure all threads are synced at this point
    __syncthreads();

    // Begin performing summation via parallel reduction
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
        // Ensure threads continue being synced
        __syncthreads();
    }

    // Put partial sums into an arrays of size equal to number of blocks
    // Second kernel call will result in complete sum
    if (thIdx == 0)
    {
        sumXd[blockIdx.x] = shArrX[0];
        sumYd[blockIdx.x] = shArrY[0];
        sumXYd[blockIdx.x] = shArrXY[0];
        sumXsqd[blockIdx.x] = shArrXsq[0];
        sumYsqd[blockIdx.x] = shArrYsq[0];
    }
}

int main()
{
    // Used for timing execution
    double startT, finishT, elapsedT;

    // Arrays used for calculation of PPMCC
    double X[N];
    double Y[N];
    double XY[N];
    double Xsq[N];
    double Ysq[N];

    // Device arrays
    double *Xd;
    double *Yd;
    double *XYd;
    double *Xsqd;
    double *Ysqd;

    // Sums of arrays passed in
    double sumX;
    double sumY;
    double sumXY;
    double sumXsq;
    double sumYsq;

    // Device arrays of sums
    double *sumXd;
    double *sumYd;
    double *sumXYd;
    double *sumXsqd;
    double *sumYsqd;

    // Start sequence of X with N
    X[0] = N;

    // Start sequence of Y with 1
    Y[0] = 1;
    
    // Intialize first element
    XY[0] = X[0] * Y[0];
    Xsq[0] = X[0] * X[0];
    Ysq[0] = Y[0] * Y[0];

    // Initialize X with values starting from 8192 down to 1
    // Initialize Y with values starting from 1 up to 8192
    // Calculate element-wise multiplication of X and Y
    // Calculate squares of elements in X and Y
    for (int n = 1; n < N; ++n)
    {
        // x(n) = x(n-1) - 1
        X[n] = X[n-1] - 1;

        // y(n) = y(n-1) + 1
        Y[n] = Y[n-1] + 1;

        XY[n] = X[n] * Y[n]; 

        Xsq[n] = X[n] * X[n];

        Ysq[n] = Y[n] * Y[n];
    }

    // Copy X to Xd
    cudaMalloc((void **)&Xd, N * sizeof(double));
    cudaMemcpy(Xd, X, N * sizeof(double), cudaMemcpyHostToDevice);

    // Copy Y to Yd
    cudaMalloc((void **)&Yd, N * sizeof(double));
    cudaMemcpy(Yd, Y, N * sizeof(double), cudaMemcpyHostToDevice);

    // Copy XY to XYd
    cudaMalloc((void **)&XYd, N * sizeof(double));
    cudaMemcpy(XYd, XY, N * sizeof(double), cudaMemcpyHostToDevice);

    // Copy Xsq to Xsqd
    cudaMalloc((void **)&Xsqd, N * sizeof(double));
    cudaMemcpy(Xsqd, Xsq, N * sizeof(double), cudaMemcpyHostToDevice);

    // Copy Ysq to Ysqd
    cudaMalloc((void **)&Ysqd, N * sizeof(double));
    cudaMemcpy(Ysqd, Ysq, N * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate global memory for device arrays of sums
    cudaMalloc((void **)&sumXd, sizeof(double) * gridSize);
    cudaMalloc((void **)&sumYd, sizeof(double) * gridSize);
    cudaMalloc((void **)&sumXYd, sizeof(double) * gridSize);
    cudaMalloc((void **)&sumXsqd, sizeof(double) * gridSize);
    cudaMalloc((void **)&sumYsqd, sizeof(double) * gridSize);

    // Start timing
    GET_TIME(startT);

    // Calculate partial sums (array of size of no. of blocks)
    sumCommMultiBlock<<<gridSize, blockSize>>>(Xd, Yd, XYd, Xsqd, Ysqd, N, sumXd, sumYd, sumXYd, sumXsqd, sumYsqd);

    // Calculate total sums (array of 1 element)
    sumCommMultiBlock<<<1, blockSize>>>(sumXd, sumYd, sumXYd, sumXsqd, sumYsqd, gridSize, sumXd, sumYd, sumXYd, sumXsqd, sumYsqd);
    
    // Ensure completion of GPU activity
    cudaDeviceSynchronize();

    // End timing
    GET_TIME(finishT);

    // Calculate execution time
    elapsedT = finishT - startT;

    // Copy total sums from device to host
    cudaMemcpy(&sumX, sumXd, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumY, sumYd, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumXY, sumXYd, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumXsq, sumXsqd, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumYsq, sumYsqd, sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate PPMCC using the sums
    double corr = (double)(N * sumXY - sumX * sumY) 
    / sqrt((N * sumXsq - sumX * sumX) * (N * sumYsq - sumY * sumY));

    // Print execution time and calculations
    cout << fixed;
    cout << "Pearson Product Moment Correlation Coefficient GPU 8k" << '\n';
    cout << "######################################################\n";
    cout << "Execution time of kernel: " << elapsedT << '\n';
    cout << "Sum of X:                 " << sumX << '\n';
    cout << "Sum of Y:                 " << sumY << '\n';
    cout << "Sum of XY:                " << sumXY << '\n';
    cout << "Sum of X^2:               " << sumXsq << '\n';
    cout << "Sum of Y^2:               " << sumYsq << '\n';
    cout << "PPMCC:                    " << corr << '\n';
    cout << "######################################################\n";

    // Free device arrays
    cudaFree(Xd);
    cudaFree(Yd);
    cudaFree(XYd);
    cudaFree(Xsqd);
    cudaFree(Ysqd);

    // Free device sums
    cudaFree(sumXd);
    cudaFree(sumYd);
    cudaFree(sumXYd);
    cudaFree(sumXsqd);
    cudaFree(sumYsqd);

    return EXIT_SUCCESS;
}