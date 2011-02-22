#include <stdio.h>

int main1(int argc, char **argv)
{
    int A[8][7];
    int i, j;

// #pragma hicuda global alloc A[*][1:6] copyin A[0:3][1:3]
#pragma hicuda global alloc A[*][1:6] copyin

#pragma hicuda kernel k_test tblock(8) thread(7)

    A[0][2] = 0;

#pragma hicuda loop_partition over_tblock
    for (i = 0; i < 8; ++i)
    {
#pragma hicuda loop_partition over_thread
        for (j = 1; j < 7; ++j)
        {
            A[i][j]++;
        }
    }

#pragma hicuda kernel_end

#pragma hicuda global copyout A[4:8][4:6]
#pragma hicuda global free A

    printf("%d\n", A[0][0]);

    return 0;
}

int main(int argc, char **argv)
{
    return main1(argc, argv);
}
