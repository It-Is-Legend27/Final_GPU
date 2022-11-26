//***************************************************************************
// Pearson Product Moment Correlation Coefficient GPU (16k)
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
// (PPMCC) between 2 arrays, F and G and prints out the result. The arrays F 
// and G contain 16384 elements each, the arrays do not contain extreme 
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

static const int N = 16384;
static const int blockSize = 1024;
static const int gridSize = 16;

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

/**
 * @brief Given an array X, Y, an array of the element-wise multiplication of X and Y,
 * an array of the squares of elements in X, and an array of the squares of elements in Y,
 * compute the Pearson Product Moment Correlation Coefficient (PPMCC).
 * 
 * @param X Array of doubles, size N.
 * @param Y Array of doubles, size N.
 * @param XY Element-wise multiplication of X and Y, size N.
 * @param Xsq Element-wise square of X, size N.
 * @param Ysq Element-wise square of Y, size N.
 * @return The PPMCC, a double 
 */
__host__ double calcCorrCoefficient(double *X, double *Y, double *XY, double *Xsq, double *Ysq)
{
    // Device copies of arrays passed in
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

    // Calculate partial sums (array of size of no. of blocks)
    sumCommMultiBlock<<<gridSize, blockSize>>>(Xd, Yd, XYd, Xsqd, Ysqd, N, sumXd, sumYd, sumXYd, sumXsqd, sumYsqd);

    // Calculate total sums (array of 1 element)
    sumCommMultiBlock<<<1, blockSize>>>(sumXd, sumYd, sumXYd, sumXsqd, sumYsqd, gridSize, sumXd, sumYd, sumXYd, sumXsqd, sumYsqd);
    
    // Ensure completion of GPU activity
    cudaDeviceSynchronize();

    // Copy total sums from device to host
    cudaMemcpy(&sumX, sumXd, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumY, sumYd, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumXY, sumXYd, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumXsq, sumXsqd, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sumYsq, sumYsqd, sizeof(double), cudaMemcpyDeviceToHost);

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

    // Calculate PPMCC using the sums
    double corr = (double)(N * sumXY - sumX * sumY) 
    / sqrt((N * sumXsq - sumX * sumX) * (N * sumYsq - sumY * sumY));

    // Return PPMCC
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

    // Initialize F with values starting from 16384 down to 1
    // Initialize G with values starting from 1 up to 16384
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