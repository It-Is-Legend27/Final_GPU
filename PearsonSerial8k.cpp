//**************************************************************
// Pearson's Product Moment Correlation Coefficient Serial (8k)
// Name: Angel Badillo, and James Nealley
// GPU Programming Date: 11/28/22
//***************************************************************
// How to run:
// This program is to be run on the TACC cluster using the SBATCH
// shell script named "PearsonSerial8kScript".
// The command to be run in the bash terminal is:
// sbatch PearsonSerial8kScript
//
// Description:
// 
//
//
// Source: https://www.geeksforgeeks.org/program-find-correlation-coefficient/
// Minor modifications were performed to the code from this source for the purposes
// of using it in our program. We altered the code to accept values of type double
// rather than of type integer.
//*****************************************************************
#include<iostream>
#include <cmath>

using namespace std;

// Size of arrays F and G
const int N = 8192;

// function that returns correlation coefficient.
double correlationCoefficient(double X[], double Y[], int n)
{

	double sum_X = 0, sum_Y = 0, sum_XY = 0;
	double squareSum_X = 0, squareSum_Y = 0;

	for (int i = 0; i < n; i++)
	{
		// sum of elements of array X.
		sum_X = sum_X + X[i];

		// sum of elements of array Y.
		sum_Y = sum_Y + Y[i];

		// sum of X[i] * Y[i].
		sum_XY = sum_XY + X[i] * Y[i];

		// sum of square of array elements.
		squareSum_X = squareSum_X + X[i] * X[i];
		squareSum_Y = squareSum_Y + Y[i] * Y[i];
	}

	// use formula for calculating correlation coefficient.
	double corr = (double)(n * sum_XY - sum_X * sum_Y)
				/ sqrt((n * squareSum_X - sum_X * sum_X)
					* (n * squareSum_Y - sum_Y * sum_Y));

	return corr;
}

// Driver function
int main()
{
    double F[N];
    double G[N];

    // Start sequence of F with 8192
    F[0] = N;

    // Start sequence of G with 1
    G[0] = 1;
    
    // Initialize F with values starting from 8192 down to 1
    // Initialize G with values starting from 1 up to 8192
    for (int n = 1; n < N; ++n)
    {
        // f(n) = f(n-1) - 1
        F[n] = F[n-1] - 1;
        // g(n) = g(n-1) + 1
        G[n] = G[n-1] + 1;
    }

	//Function call to correlationCoefficient.
	cout << "Pearson Correlation Coefficient of F and G: " << correlationCoefficient(F, G, N) << '\n';
    
	return EXIT_SUCCESS;
}