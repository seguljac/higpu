
#include "common.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * Initialize a matrix with random elements.
 */
void randomInitArr(float *data, unsigned int size)
{
    unsigned int i;

    for (i = 0; i < size; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

/**
 * Output a matrix to the standard output.
 */
void printMatrix(float *mat, unsigned int h, unsigned int w)
{
    unsigned int i, j;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            printf("%.3f ", *(mat++));
        }
        printf("\n");
    }

    printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void computeGold(float* C, const float* A, const float* B,
        unsigned int hA, unsigned int wA, unsigned int wB)
{
    unsigned int i, j, k;

    for (i = 0; i < hA; ++i)
    {
        for (j = 0; j < wB; ++j)
        {
            double sum = 0;
            for (k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
    }
}

void compare_matrices(float *mat, float *ref, unsigned int nelems)
{
    int error = 0;

    unsigned int i;
    for (i = 0; i < nelems; ++i)
    {
        float diff = mat[i] - ref[i];
        if (diff < 0) diff = -diff;
        float avg = (mat[i] + ref[i]) / 2;
        if (diff / avg > 1e-5)
        {
            printf("Diff at index %u: %f, %f\n", i, mat[i], ref[i]);
            error = 1;
        }
    }
    
    if (!error) printf("PASSED!\n");
}
