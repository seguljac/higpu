
/*****************************************************************************
 *
 * This program tests the compiler's ability to reconstruct multi-dimensional
 * array accesses.
 *
 ****************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#define HEIGHT 128
#define WIDTH 256

int main(int argc, char **argv)
{
    float *arr;
    int h = argc + HEIGHT, w = argc + WIDTH;    // prevent const propagation
    int i, j;

    arr = (float*)calloc(h * w, sizeof(float));

#pragma hicuda shape arr[h][w]

#pragma hicuda global alloc arr[*][*] clear

#pragma hicuda kernel k_test tblock(1,1) thread(1,1)

    for (i = 0; i < h; ++i)
    {
        int base_ofst = i * w;
        for (j = 0; j < w; ++j)
        {
            int ofst = base_ofst + j;
            arr[ofst] += 1;
        }
    }

#pragma hicuda kernel_end

#pragma hicuda global copyout arr[*][*]

#pragma hicuda global free arr

    free(arr);

    return 0;
}

