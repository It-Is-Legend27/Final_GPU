//**************************************************************
// Assignment #5
// Name: Angel Badillo, and James Nealley
// GPU Programming Date: 11/07/22
//***************************************************************
// How to run:
// This program is to be run on the TACC cluster using the SBATCH
// shell script named "AngelBadilloA5Script".
// The command to be run in the bash terminal is:
// sbatch AngelBadilloA5Script
//
// Description:
// This program, written in C, creates 3 finite sequences (arrays),
// F, G, H, where F is 4096 elements, G is 1024 elements, and H
// is 5119 elements.
// Then, F will have a sequence of numbers from 1 up to 4096,
// and G will have a sequence of numbers from 1024 down to 1.
// H will be temporarily be intialized to all 0s.
// Next, the convolution of F and G, F * G, will be calculated and
// and stored in H. Finally, the contents of H will be printed
// in column fashion to AngelBadilloA5.csv.
//*****************************************************************
#include <stdio.h>
#include <stdlib.h>

// Number of elements in F
#define M 4096

// Number of elements in G
#define N 1024

// Number of elements in H
#define O N + M - 1

// Size of buffer to hold formatted string
#define STR_SZ 128

// Simple buffer to hold formatted string before printing
// Will be size 128, just to be safe and avoid accessing out
// of bounds.
typedef char string_buffer[STR_SZ];

int main()
{
    // For use to hold formatted string
    string_buffer output_string;

    // Create (if does not exist) and open file for write
    FILE *outputFilePtr = fopen("AngelBadilloA5.csv", "w");

    // Create arrays for sequences F, G, and resulting H
    long long F[M];
    long long G[N];
    long long H[O] = {0};

    // Checksum to be calculated once H is computed
    long long checkSum = 0;

    // Initialize F starting from 1 up to 4096
    for (int m = 0; m < M; m++)
    {
        F[m] = m + 1;
    }

    // Initialize G from 1024 down to 1
    for (int n = 0; n < N; n++)
    {
        G[n] = N - n;
    }

    // Computes all values of H(n) from 0 up to N+M-2
    for (int n = 0; n < O; n++)
    {
        // Compute each part the sum of F(m)*G(n-m), and add it
        // to the total sum in H(n)
        for (int m = 0; m <= n; m++)
        {
            // Only performed if indices are valid.
            // If m < 0 and m >= M and n < 0 and n <= N, this
            // instruction will be skipped as the value to be added to the sum
            // will be 0.
            if (m >= 0 && m < M && (n - m) >= 0 && (n - m) < N)
                H[n] += F[m] * G[n - m];
        }
        // Add to check sum
        checkSum += H[n];
    }

    // Printing number of elements in each array of values
    printf("Number of elements in f: %d\n", M);
    printf("Number of elements in g: %d\n", N);
    printf("Number of elements in h: %d\n", O);
    printf("Checksum results: %lld\n", checkSum);

    // Print labels for columns to .csv file
    fputs("n, H[n]\n", outputFilePtr);

    // Print every element of H to .csv file
    for (int i = 0; i < O; i++)
    {
        // Create formatted string, store in buffer, output_string
        sprintf(output_string, "%d, %lld\n", i, H[i]);

        // Print output_string to .csv file
        fputs(output_string, outputFilePtr);
    }

    // Close the file
    fclose(outputFilePtr);
}