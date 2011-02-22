
#include <stdlib.h>

float glob_arr[10];

void bar(float (*arr)[10])
{
    (*arr)[9] = 0;
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
    float local_arr1[10];
    float local_arr2[10];

    int n = min(argc, 9);

#pragma hicuda global alloc local_arr1[n:9]
#pragma hicuda global alloc local_arr2[n:9]
#pragma hicuda global alloc glob_arr[0:9]

    bar(&local_arr1);

#pragma hicuda kernel k_test tblock(2) thread(4)

    foo(&local_arr1, n);
    foo(&local_arr2, n);

#pragma hicuda kernel_end

#pragma hicuda global copyout local_arr1[n:9]
#pragma hicuda global copyout local_arr2[n:9]
#pragma hicuda global free local_arr1 local_arr2

    return 0;
}
