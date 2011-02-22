
/*****************************************************************************
 *
 * This program tests reaching directive analysis.
 *
 * When <matrix> of <foo> is a static array, like a[6][5], there is no way for
 * the compiler to know that the shape of <matrix> is <height> x <width>,
 * unless it performs IP constant propagation.
 *
 * An alternative is to allow the user to override the shape of <a> to be
 * <height> x <width> outside or within <foo>.
 *
 ****************************************************************************/

#include <stdlib.h>

float a[6][5];

void foo(float *matrix, int height, int width)
{
    int i, j;

#pragma hicuda kernel k_rt012 tblock(4) thread(4)

#pragma hicuda loop_partition over_tblock
    for (i = 1; i < height-1; ++i)
#pragma hicuda loop_partition over_thread
        for (j = 1; j < width-1; ++j)
            *(matrix + i*width + j) = i + j;

#pragma hicuda kernel_end
}

int main(int argc, char **argv)
{
    int n = argc;
    int m = argc + 2;

    float *da = (float*)malloc(n * m * sizeof(float));

#pragma hicuda shape da[n][m]

#pragma hicuda global alloc da[1:n-1][1:m-1]
#pragma hicuda global alloc a[*][*]

    foo(da, n, m);
    foo((float*)a, 6, 5);

#pragma hicuda global copyout da[1:n-1][1:m-1]
#pragma hicuda global copyout a[1:4][1:3]
#pragma hicuda global free da a

    return 0;
}

