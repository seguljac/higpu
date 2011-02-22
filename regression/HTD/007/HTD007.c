
#include <stdlib.h>

float glob_arr[10];

void bar(float (*arr)[10])
{
    (*arr)[9] = 2 * glob_arr[9];
}

void foo(float (*arr)[10], int lbnd)
{
    int i;

    for (i = lbnd; i < 10; ++i)
        glob_arr[i] += (*arr)[i];

    bar(arr);
}

int main(int argc, char **argv)
{
    float local_arr[10];

    int n = min(argc, 9);

#pragma hicuda global alloc local_arr[n:9]
#pragma hicuda global alloc glob_arr[0:9]

    bar(&local_arr);

#pragma hicuda kernel k_test tblock(2) thread(4)

    foo(&local_arr, n);
    foo(&local_arr, n);

#pragma hicuda kernel_end

#pragma hicuda global copyout local_arr[n:9]
#pragma hicuda global free local_arr

    return 0;
}
