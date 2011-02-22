
/*****************************************************************************
 *
 * This program tests:
 * 1) kernel data access analysis, switch(incr[i]) in particular
 * 2) constant propagation on loop tripcount
 *
 ****************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#define WARP_SZ 32
#define N_WARPS_PER_TBLK 1
#define N_TBLKS 1

#define N_ITERS 100

int main(int argc, char **argv)
{
    float *arr, init_val;
    int *incr;
    int n_threads;
    int i, j;

    n_threads = N_TBLKS * N_WARPS_PER_TBLK;
    n_threads *= WARP_SZ;
    incr = (int*)malloc(n_threads * sizeof(int));
    arr = (float*)malloc(n_threads * sizeof(float));

#pragma hicuda shape incr[n_threads]
#pragma hicuda shape arr[n_threads]

    // Copy propagation should NOT happen.
    init_val = (float)argc + 1.0F;

    for (i = 0; i < n_threads; ++i)
    {
        incr[i] = (i % WARP_SZ);
        arr[i] = init_val;
    }

#pragma hicuda global alloc incr[*] copyin
#pragma hicuda global alloc arr[*] clear

#pragma hicuda kernel d_test tblock(N_TBLKS) thread(32)

#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < n_threads; ++i)
    {
        for (j = 0; j < N_ITERS; ++j)
        {
            switch (incr[i])
            {
                case 0: arr[i] += 0.0F; break;
                case 1: arr[i] += 1.0F; break;
                case 2: arr[i] += 2.0F; break;
                case 3: arr[i] += 3.0F; break;
                case 4: arr[i] += 4.0F; break;
                case 5: arr[i] += 5.0F; break;
                case 6: arr[i] += 6.0F; break;
                case 7: arr[i] += 7.0F; break;
                case 8: arr[i] += 8.0F; break;
                case 9: arr[i] += 9.0F; break;
                case 10: arr[i] += 10.0F; break;
                case 11: arr[i] += 11.0F; break;
                case 12: arr[i] += 12.0F; break;
                case 13: arr[i] += 13.0F; break;
                case 14: arr[i] += 14.0F; break;
                case 15: arr[i] += 15.0F; break;
                case 16: arr[i] += 16.0F; break;
                case 17: arr[i] += 17.0F; break;
                case 18: arr[i] += 18.0F; break;
                case 19: arr[i] += 19.0F; break;
                case 20: arr[i] += 20.0F; break;
                case 21: arr[i] += 21.0F; break;
                case 22: arr[i] += 22.0F; break;
                case 23: arr[i] += 23.0F; break;
                case 24: arr[i] += 24.0F; break;
                case 25: arr[i] += 25.0F; break;
                case 26: arr[i] += 26.0F; break;
                case 27: arr[i] += 27.0F; break;
                case 28: arr[i] += 28.0F; break;
                case 29: arr[i] += 29.0F; break;
                case 30: arr[i] += 30.0F; break;
                case 31: arr[i] += 31.0F; break;
                default: break;
            }
        }
    }

#pragma hicuda kernel_end

#pragma hicuda global copyout arr[*]

#pragma hicuda global free incr arr

    printf("\n");
    for (i = 0; i < WARP_SZ; ++i)
    {
        printf("%10.2f\n", arr[i]);
    }
    printf("\n");

    free(arr);

    return 0;
}

