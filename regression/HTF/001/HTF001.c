
#include <stdio.h>

#define A_SIZE 1024
#define N_BLOCKS 4
#define N_THREADS 256

int main()
{
    int i;
    float A[A_SIZE];

#pragma hicuda global alloc A[0:A_SIZE-1] copyin

#pragma hicuda kernel zero tblock(N_BLOCKS) thread(N_THREADS)

#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < A_SIZE; ++i)
    {
        A[i] = i;
    }

#pragma hicuda global copyout A[*]
#pragma hicuda global free A

    return 0;
}
