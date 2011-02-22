#include <stdlib.h>
#include <stdio.h>

#define ARR_LEN 1000

long compute(long x)
{
    return x * x - x;
}

int main(int argc, char **argv)
{
    long *arr;

    arr = (long*)malloc(ARR_LEN * sizeof(long));

#pragma hicuda shape arr[ARR_LEN]

#pragma hicuda global alloc arr[*]

#pragma hicuda kernel gpu_compute1 tblock(16) thread(256)
    int i;
#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < ARR_LEN; ++i)
    {
        arr[i] = compute(i);
    }
#pragma hicuda kernel_end

// #pragma hicuda kernel gpu_compute2 tblock(16) thread(256)
// #pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < ARR_LEN; ++i)
    {
        arr[i] = compute(arr[i]);
    }
// #pragma hicuda kernel_end

#pragma hicuda global copyout arr[*]
#pragma hicuda global free arr

    printf("arr[10] = %ld\n", arr[10]);

    return 0;
}
