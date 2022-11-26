//***************************************************************************
// Pearson Product Moment Correlation Coefficient Serial (8k)
// Name: Angel Badillo, and James Nealley
// GPU Programming Date: 11/28/22
//***************************************************************************
// How to run:
// This program is to be run on the TACC cluster using the SBATCH
// shell script named "PearsonSerial8kScript".
// The command to be run in the bash terminal is:
// sbatch PearsonSerial8kScript
//
// Description:
// This program calculates the Pearson Product Moment Correlation Coefficient 
// (PPMCC) between 2 arrays, X and Y and prints out the result. The arrays X 
// and Y contain 8192 elements each, the arrays do not contain extreme 
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

// Size of arrays X and Y
const int N = 8192;

int main()
{
	// Used for timing code
	double start, finish, elapsed;

	// Begin timing execution
	GET_TIME(start);

	// Arrays X and Y for calculating correlation coefficient between them
    double X[N];
    double Y[N];

	double corrCoefficient;

    // Start sequence of X with 16384
    X[0] = N;

    // Start sequence of Y with 1
    Y[0] = 1;
    
    // Initialize X with values starting from 8192 down to 1
    // Initialize Y with values starting from 1 up to 8192
    for (int n = 1; n < N; ++n)
    {
        // X(n) = X(n-1) - 1
        X[n] = X[n-1] - 1;

        // Y(n) = Y(n-1) + 1
        Y[n] = Y[n-1] + 1;
    }

	// Calculate PPMCC
	// Sums required for the calculation of PPMCC
	double sum_X = 0, sum_Y = 0, sum_XY = 0;
	double squareSum_X = 0, squareSum_Y = 0;

	// Calculate all the sums
	for (int i = 0; i < N; i++)
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
	double corr = (double)(N * sum_XY - sum_X * sum_Y)
				/ sqrt((N * squareSum_X - sum_X * sum_X)
					* (N * squareSum_Y - sum_Y * sum_Y));

	// Finish timing execution
	GET_TIME(finish);

	// Calculate elapsed time of execution
	elapsed = finish - start;

	// Print execution time and calculations
    cout << fixed;
    cout << "Pearson Product Moment Correlation Coefficient 8k" << '\n';
    cout << "###################################################\n";
    cout << "Execution time:           " << elapsed << '\n';
    cout << "Sum of X:                 " << sum_X << '\n';
    cout << "Sum of Y:                 " << sum_Y << '\n';
    cout << "Sum of XY:                " << sum_XY << '\n';
    cout << "Sum of X^2:               " << squareSum_X << '\n';
    cout << "Sum of Y^2:               " << squareSum_Y << '\n';
    cout << "PPMCC:                    " << corr << '\n';
    cout << "###################################################\n";
    
	return EXIT_SUCCESS;
}