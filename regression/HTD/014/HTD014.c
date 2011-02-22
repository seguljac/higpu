
/*****************************************************************************
 *
 * This program tests the construction of Def-Use chains for scalars across
 * kernel boundaries.
 *
 ****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

int main(int argc, char **argv)
{
    int i, n_threads;
    int x;
    int *arr;

    n_threads = atoi(argv[2]);
    arr = (int*)calloc(n_threads, sizeof(int));
    assert(arr != NULL);

    x = atoi(argv[1]);

#pragma hicuda shape arr[n_threads]
#pragma hicuda global alloc arr[*] copyin

#pragma hicuda kernel k_test tblock(1) thread(10)

#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < n_threads; ++i)
    {
        int y;

        if (x % 2) y = x;

        if (x % 2) arr[i] += y;
    }

#pragma hicuda kernel_end

#pragma hicuda global copyout arr[*]
#pragma hicuda global free arr

    printf("arr[0] = %d\n", arr[0]);

    free(arr);

    return 0;
}

