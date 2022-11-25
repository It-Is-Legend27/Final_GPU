//**************************************************************
// Assignment #4
// Name: Angel Badillo, and James Nealley
// GPU Programming Date: 11/03/22
//***************************************************************
// How to run:
// This program is to be run on the TACC cluster using the SBATCH
// shell script named "AngelBadilloA4Script".
// The command to be run in the bash terminal is:
// sbatch AngelBadilloA4Script
//
// Description:
// This program, written in CUDA C, aims to implement a Fast
// Fourier Transform (FTT) algorithm, specifically the
// Cooley-Turkey FFT algorithm (Radix-2). The program peforms the
// FFT for N Fourier coefficients (8192) from X(0) to X(N-1),
// given N samples (8192). The first 8 samples are hardcoded,
// to array R[N] and I[N], but the rest are initialized to 0.
// Once the full calculation for N Fourier coefficients are
// completed after invoking the kernel, the program will
// print the first 8 outputs for X(0) to X(7), as well as
// the "middle" 8 outputs for X(4096) to X(4103).
//*****************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// No. of samples and Fourier coefficients
const int N_SAMPLES = 8192;

// Constant for pi
const double PI = 3.1415926535897932;

// Byte size of array of doubles with 8192 elements;
const int NBYTE_SIZE = N_SAMPLES * sizeof(double);

/**
 * @brief Represents a Complex number.
 * Contains real part (double CmplxNum::a) and imaginary part
 * (double CmplxNum::bi).
 *
 */
struct CmplxNum
{
    double a;  // real part of complex number
    double bi; // imaginary part of complex number
};

//*******************************************************************
// CmplxAdd
// @param X The augend, a CmplxNum
// @param Y The addend, a CmplxNum
// @brief
// Calculates the addition of complex numbers (CmplxNum) X and Y, then 
// returns sum as CmplxNum.
// @return The sum as struct CmplxNum
//********************************************************************
__device__ struct CmplxNum CmplxAdd(struct CmplxNum X, struct CmplxNum Y)
{
    struct CmplxNum Z = {.a = X.a + Y.a, .bi = X.bi + Y.bi};
    return Z;
}


//*******************************************************************
// CmplxMult
// @param X The multiplicand, a CmplxNum
// @param Y The multiplier, a CmplxNum
// @brief
// Calculates the multiplication of complex numbers (CmplxNum) X and 
// Y, then returns product as CmplxNum.
// @return The product as struct CmplxNum
//********************************************************************
__device__ struct CmplxNum CmplxMult(struct CmplxNum X, struct CmplxNum Y)
{
    // X = a + bi
    // Y = c + di
    // XY = (a+bi)(c+di) = (ac−bd) + (ad+bc)i
    struct CmplxNum Z = {.a = (X.a * Y.a) - (X.bi * Y.bi), .bi = (X.a * Y.bi) + (X.bi * Y.a)};
    return Z;
}


//*******************************************************************
// computeFFT
// @param XR an array of doubles of size 8192 containing real parts 
// of output for N no. of FTT coefficients.
// @param XI an array of doubles of size 8192 containing imaginary 
// parts of output for N no. of FFT coefficients.
// @param R an array of doubles of size 8192 containing real part of
// samples for the function x(n).
// @param I an array of doubles of size 8192 containing imaginary 
// part of samples for the function x(n)
// @brief
// Implements the Cooley-Turkey FFT algorithm (AKA Radix-2).
// Computes the output for a total of N no. FFT coefficients with N no. of 
// samples."Returns" (or modifies via reference) array of doubles XR and XI to 
// have the output for N no. of FFT coefficients X(0) to X(N-1).
// @return void
//********************************************************************
__global__ void computeFFT(double *XR, double *XI, double *R, double *I)
{
    // Global index
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate twiddle factor
    struct CmplxNum tFactor = {.a = cos(2 * PI * k / N_SAMPLES), .bi = -sin(2 * PI * k / N_SAMPLES)};

    // Initialize variables for calculation of even and odd parts for calculation of X(k) and X(k + N/2)
    struct CmplxNum evenPart = {.a = 0, .bi = 0};
    struct CmplxNum oddPart = {.a = 0, .bi = 0};

    // Calculate the sums of the even and odd parts for calculation of X(k) and X(k + N/2)
    for (int m = 0; m <= N_SAMPLES / 2 - 1; m++)
    {
        // E(k)
        struct CmplxNum funcX2m = {.a = R[2 * m], .bi = I[2 * m]};
        struct CmplxNum eulerPart = {.a = cos(2 * PI * 2 * m * k / N_SAMPLES), .bi = -sin(2 * PI * 2 * m * k / N_SAMPLES)};
        struct CmplxNum resEven = CmplxMult(funcX2m, eulerPart);
        evenPart = CmplxAdd(evenPart, resEven);

        // O(k)
        struct CmplxNum funcX2mP1 = {.a = R[(2 * m) + 1], .bi = I[(2 * m) + 1]};
        struct CmplxNum resOdd = CmplxMult(funcX2mP1, eulerPart);
        oddPart = CmplxAdd(oddPart, resOdd);
    }

    // Adding even part for E(k)
    XR[k] = evenPart.a;
    XI[k] = evenPart.bi;

    // Adding even part for E(k + N/2)
    XR[k + N_SAMPLES / 2] = evenPart.a;
    XI[k + N_SAMPLES / 2] = evenPart.bi;

    // Adjusting odd parts by twiddle factor
    struct CmplxNum temp = CmplxMult(tFactor, oddPart);

    // Adding odd part for O(k)
    XR[k] += temp.a;
    XI[k] += temp.bi;

    // Subtracting odd part for O(k + N/2)
    XR[k + N_SAMPLES / 2] -= temp.a;
    XI[k + N_SAMPLES / 2] -= temp.bi;
}

int main()
{
    // Real and imaginary components of samples
    double R[N_SAMPLES] = {3.6, 2.9, 5.6, 4.8, 3.3, 5.9, 5.0, 4.3};
    double I[N_SAMPLES] = {2.6, 6.3, 4.0, 9.1, 0.4, 4.8, 2.6, 4.1};
    double *R_d;
    double *I_d;

    // Output for Fourier coefficients
    double XR[N_SAMPLES];
    double XI[N_SAMPLES];
    double *XR_d;
    double *XI_d;

    // Allocate memory for R_d, I_d, XR_d, and XI_d
    cudaMalloc((void **)&R_d, NBYTE_SIZE);
    cudaMalloc((void **)&I_d, NBYTE_SIZE);
    cudaMalloc((void **)&XR_d, NBYTE_SIZE);
    cudaMalloc((void **)&XI_d, NBYTE_SIZE);

    // Copy arrays R & I to R_d & I_d, respectively
    cudaMemcpy(R_d, R, NBYTE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(I_d, I, NBYTE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(XR_d, XR, NBYTE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(XI_d, XI, NBYTE_SIZE, cudaMemcpyHostToDevice);

    // Dimensions of grid & block
    dim3 dimGrid(4, 1);
    dim3 dimBlock(1024, 1);

    // Call kernel to compute FFT
    computeFFT<<<dimGrid, dimBlock>>>(XR_d, XI_d, R_d, I_d);

    // Copy XR_d & XI_d to XR & XI
    cudaMemcpy(XR, XR_d, NBYTE_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(XI, XI_d, NBYTE_SIZE, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(R_d);
    cudaFree(I_d);
    cudaFree(XR_d);
    cudaFree(XI_d);

    
    printf("==========================================================================\n");
    printf("TOTAL PROCESSED SAMPLES: %d\n", N_SAMPLES);
    printf("==========================================================================\n");

    // Print output for the first 8 Fourier Coefficients from X(0) to X(7)
    for (int i = 0; i < 8; ++i)
    {
        printf("XR[%d]: %.6f          XI[%d]: %.6fi\n", i, XR[i], i, XI[i]);
        printf("==========================================================================\n");
    }

    // Print the output for "middle" 8 coefficients from X(4096) to X(4103)
    for (int i = N_SAMPLES/2; i < N_SAMPLES/2 + 8; ++i)
    {
        printf("XR[%d]: %.6f          XI[%d]: %.6fi\n", i, XR[i], i, XI[i]);
        printf("==========================================================================\n");
    }

    return EXIT_SUCCESS;
}