//***************************************************************************
// Pearson Product Moment Correlation Coefficient Serial (16k)
// Name: Angel Badillo, and James Nealley
// GPU Programming Date: 11/28/22
//***************************************************************************
// How to run:
// This program is to be run on the TACC cluster using the SBATCH
// shell script named "PearsonSerial16kScript".
// The command to be run in the bash terminal is:
// sbatch PearsonSerial16kScript
//
// Description:
// This program calculates the Pearson Product Moment Correlation Coefficient 
// (PPMCC) between 2 arrays, F and G and prints out the result. The arrays F 
// and G contain 16384 elements each, the arrays do not contain extreme 
// outliers and they also satisfy the assumptions for the PPMCC.
//
// Source: https://www.geeksforgeeks.org/program-find-correlation-coefficient/
// Minor modifications were performed to the code from this source for the 
// purposes of using it in our program. We altered the code to accept values 
// of type double rather than of type integer, since the data can be
// composed of rational numbers.
//***************************************************************************
#include<iostream>
#include <cmath>
#include "timer.h"

using namespace std;

// Size of arrays F and G
const int N = 16384;

/**
 * @brief Calculates the Pearson Product Moment Correlation Coefficient
 * between two arrays of equal size.
 * 
 * @param X Array of type double, size of n.
 * @param Y Array of type double, size of n.
 * @param n Size of arrays X and Y.
 * @returns Pearson Product Moment Correlation Coefficient between arrays
 * X and Y. Value is of type double.
 */
double correlationCoefficient(double X[], double Y[], int n)
{
	// Sums required for the calculation of PPMCC
	double sum_X = 0, sum_Y = 0, sum_XY = 0;
	double squareSum_X = 0, squareSum_Y = 0;

	// Calculate all the sums
	for (int i = 0; i < n; i++)
	{
		// Sum of elements of array X
		sum_X = sum_X + X[i];

		// Sum of elements of array Y
		sum_Y = sum_Y + Y[i];

		// Sum of X[i] * Y[i].
		sum_XY = sum_XY + X[i] * Y[i];

		// Sum of square of array elements
		squareSum_X = squareSum_X + X[i] * X[i];
		squareSum_Y = squareSum_Y + Y[i] * Y[i];
	}

	// Calculate PPMCC using the formula for PPMCC
	double corr = (double)(n * sum_XY - sum_X * sum_Y)
				/ sqrt((n * squareSum_X - sum_X * sum_X)
					* (n * squareSum_Y - sum_Y * sum_Y));

	return corr;
}

int main()
{
	// Used for timing code
	double start, finish, elapsed;

	// Begin timing execution
	GET_TIME(start);

	// Arrays F and G for calculating correlation coefficient between them
    double F[N];
    double G[N];

	double corrCoefficient;

    // Start sequence of F with 16384
    F[0] = N;

    // Start sequence of G with 1
    G[0] = 1;
    
    // Initialize F with values starting from 16384 down to 1
    // Initialize G with values starting from 1 up to 16384
    for (int n = 1; n < N; ++n)
    {
        // f(n) = f(n-1) - 1
        F[n] = F[n-1] - 1;
        // g(n) = g(n-1) + 1
        G[n] = G[n-1] + 1;
    }

	// Calculate PPMCC
	corrCoefficient = correlationCoefficient(F, G, N);

	// Finish timing execution
	GET_TIME(finish);

	// Calculate elapsed time of execution
	elapsed = finish - start;

	// Print execution time and PPMCC
	cout << "Execution time: " << elapsed << '\n';
	cout << "Pearson Correlation Coefficient of F and G: " << corrCoefficient << '\n';
    
	return EXIT_SUCCESS;
}