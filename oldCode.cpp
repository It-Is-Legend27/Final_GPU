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
//*****************************************************************
#include <iostream>
#include <cmath>
using namespace std;

// Number of elements in F and G
const int N = 5;

int main()
{
    // Create arrays for sequences F and G
    double F[N];
    double G[N];

    // Accumulators for sums of all Xs, Ys, XYs, X^2s, and Y^2s
    double sumX = 0;
    double sumY = 0;
    double sumXY = 0;
    double sumXsq = 0;
    double sumYsq = 0;

    // Pearson Correlation Coefficient
    double R = 0;

    // Start sequence of F with 0.1
    F[0] = 0.1;

    // Start sequence of G with 1
    G[0] = 1;
    
    // Initialize F with values starting from
    // Initialize G with values starting from
    for (int n = 1; n < N; ++n)
    {
        // f(n) = f(n-1) + 0.5
        F[n] = F[n-1] + 0.1;
        // g(n) = g(n-1) + 1
        G[n] = G[n-1] + 1;
    }

    // Calculate XY, X^2, and Y^2 for each element of F and G
    // and add them to corresponding accumulators
    for (int n = 0;  n < N; ++n)
    {
        // Sum of Xs
        sumX += F[n];

        // Sum of Ys
        sumY += G[n];

        // Sum of XYs
        sumXY += F[n]*G[n];

        // Sum of X^2s
        sumXsq += F[n] * F[n];

        // Sum of Y^2s
        sumYsq += G[n] * G[n];
    }

    // Pearson Correlation Coefficient
    //R = (double)((N*sumXY)-(sumX*sumY)) / sqrt((N*sumXsq - (sumX*sumX)) * (N*sumYsq - (sumY*sumY)));

    cout << "Number of elements in F and G: " << N << "\n";
    cout << "Sum X = " << sumX << "\n";
    cout << "Sum Y = " << sumY << "\n";
    cout << "Sum XY = " << sumXY << "\n";
    cout << "Sum X^2 = " << sumXsq << "\n";
    cout << "Sum Y^2 = " << sumYsq << "\n"; 
}