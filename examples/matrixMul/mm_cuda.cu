/**
 * This is the CUDA version of matrix multiplication.
 */

// CUDA runtime header file
#include <cuda_runtime.h>

#include <stdio.h>

#include "common.h"

#define WA 1024     // width of matrix A
#define HA 1024     // height of matrix A
#define WB 1024     // width of matrix B

#define TILE_SZ 16

/** CODE ADDED/MODIFIED BEGIN **/

/**
 * matrixMul kernel
 * wA and wB are the width of matrix A and B respectively.
 */
__global__ void matrixMul(float *A, float *B, int wA, int wB, float *C)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first and last sub-matrix of A processed by the block
    int aBegin = wA * TILE_SZ * by;
    int aEnd = aBegin + wA - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep = TILE_SZ;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = TILE_SZ * bx;
    int bStep = TILE_SZ * wB;

    // The element of the block sub-matrix computed by the thread
    float Csub = 0;

    // Loop over all sub-matrices of A and B.
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Shared memory for the sub-matrix of A
        __shared__ float As[TILE_SZ][TILE_SZ];
        // Shared memory for the sub-matrix of B
        __shared__ float Bs[TILE_SZ][TILE_SZ];

        // Load sub-matrices from global memory to shared memory cooperatively.
        // NOTE: As[tx][ty] would cause bank conflicts in smem and would not
        // exploit coalescing in gmem.
        As[ty][tx] = A[a + wA*ty + tx];
        Bs[ty][tx] = B[b + wB*ty + tx];

        // Synch to make sure that the matrices are loaded.
        __syncthreads();

        // Multiply the sub-matrices together.
        // Each thread computes one element (ty,tx) of it.
        for (int k = 0; k < TILE_SZ; ++k) {
            // NOTE: As[tx][k] * Bs[k][ty] would cause bank conflicts in smem.
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synch again to make sure the computation is done before
        // data is loaded in the next iteration.
        __syncthreads();
    }

    // Write the block sub-matrix of C to global memory.
    int c = wB * TILE_SZ * by + TILE_SZ * bx;
    C[c + wB*ty + tx] = Csub;
}

/** CODE ADDED/MODIFIED END **/

int main(int argc, char **argv)
{
    int size;
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = (float*)malloc(HA*WA*sizeof(float));
    B = (float*)malloc(WA*WB*sizeof(float));
    C = (float*)malloc(HA*WB*sizeof(float));
    float *reference = (float*)malloc(HA*WB*sizeof(float));

    // Randomly init A and B.
    srand(2008);
    randomInitArr((float*)A, HA*WA);
    randomInitArr((float*)B, WA*WB);

    /** CODE ADDED BEGIN **/

    // Load A and B to the device.
    size = HA * WA * sizeof(float);
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    size = WA * WB * sizeof(float);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Allocate C on the device.
    size = HA * WB * sizeof(float);
    cudaMalloc((void**)&d_C, size);

    // Record the start time.
    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    // Compute the execution configuration: grid/block dimensions.
    dim3 dimBlock(TILE_SZ, TILE_SZ);
    dim3 dimGrid(WB/dimBlock.x, HA/dimBlock.y);

    // Launch the kernel.
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, WA, WB, d_C);
    // cudaThreadSynchronize();

    // Read C from the device.
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Record the end time.
    struct timeval end_time;
    gettimeofday(&end_time, NULL);

    printf("Time elapsed: %6f ms\n", get_time_diff(&start_time, &end_time));

    // Compute reference solution.
    computeGold((float*)reference, (float*)A, (float*)B, HA, WA, WB);

    // Check result.
    compare_matrices((float*)C, (float*)reference, HA*WB);

    // Free the device memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /** CODE ADDED END **/

    // printMatrix((float*)C, HA, WB);

    return 0;
}
