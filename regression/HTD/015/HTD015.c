
/*****************************************************************************
 *
 * This program tests kernel outlining when some variables accessed within are
 * static (either program-scoped or file-scoped).
 *
 ****************************************************************************/

#include <stdio.h>

static long n2;     // file-scoped
static long n3;

int main(int argc, char **argv)
{
    static long arr[10];
    static long n1;     // PU-scoped
    int i;

    // n2 = argc;
    n2 = n3;

#pragma hicuda global alloc arr[*] clear

#pragma hicuda kernel k_test tblock(1) thread(10)
#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < 10; ++i)
    {
        arr[i] += i + n1 + n2;
    }
#pragma hicuda kernel_end

#pragma hicuda global copyout arr[*]
#pragma hicuda global free arr

    printf("%d\n", arr[3]);

    return 0;
}

