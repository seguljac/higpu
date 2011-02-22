/**
 * This is the hiCUDA version of matrix multiplication, that divides the
 * computation into tiles of size TILE_SZ.
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include "common.h"

#define WA 1024     // width of matrix A
#define HA 1024     // height of matrix A
#define WB 1024     // width of matrix B

#define TILE_SZ 16

float A[HA][WA];
float B[WA][WB];
float C[HA][WB];
float reference[HA][WB];

int main(int argc, char **argv)
{
    int i, j, k, kk;

    // Randomly init A and B.
    srand(2008);
    randomInitArr((float*)A, HA*WA);
    randomInitArr((float*)B, WA*WB);

#pragma hicuda global alloc A[*][*] copyin
#pragma hicuda global alloc B[*][*] copyin
#pragma hicuda global alloc C[*][*]

    // Record the start time.
    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    // C = A * B
#pragma hicuda kernel matrixMul tblock(64,64) thread(16,16)

#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < HA; ++i)
    {
#pragma hicuda loop_partition over_tblock over_thread
        for (j = 0; j < WB; ++j)
        {
            float sum = 0;

            for (kk = 0; kk < WA; kk += TILE_SZ) {
#pragma hicuda shared alloc A[i][kk:kk+15] copyin
#pragma hicuda shared alloc B[kk:kk+15][j] copyin
#pragma hicuda barrier
                for (k = 0; k < TILE_SZ; ++k) {
                    sum += A[i][kk+k] * B[kk+k][j];
                }
#pragma hicuda barrier
#pragma hicuda shared remove A B
            }
            C[i][j] = sum;
        }
    }

#pragma hicuda kernel_end

#pragma hicuda global copyout C[*][*]

#pragma hicuda global free A B C

    // Record the end time.
    struct timeval end_time;
    gettimeofday(&end_time, NULL);

    printf("Time elapsed: %6f ms\n", get_time_diff(&start_time, &end_time));

    // Compute reference solution.
    computeGold((float*)reference, (float*)A, (float*)B, HA, WA, WB);

    // Check result.
    compare_matrices((float*)C, (float*)reference, HA*WB);

    // printMatrix((float*)C, HA, WB);

    return 0;
}

