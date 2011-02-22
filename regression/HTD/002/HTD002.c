#include <stdlib.h>

int arr[6];

int foo(int *f)
{
    int x = f[1];
    int j;

    f[0] = f[0] + 1;

#pragma hicuda loop_partition over_tblock
    for (j = 1; j < 5; ++j) {
        f[0] += f[j];
    }

    return x + f[0];
}

int sum1(int *b, int m)
{
    int result, i;

    b[m-1] = b[2] + 1;

#pragma hicuda global alloc b[*] copyin
#pragma hicuda global alloc result

#pragma hicuda kernel test_sum tblock(2) thread(2)

    result += foo(b);

#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < 4; ++i) {
        result += b[i];
    }

#pragma hicuda kernel_end

#pragma hicuda global free b result

    return result;
}

int foo1(int x)
{
    int j;

// #pragma hicuda kernel test_foo1 tblock(2) thread(3)
    x = x + 1;
// #pragma hicuda kernel_end

#pragma hicuda loop_partition over_tblock
    for (j = 1; j < 5; ++j) {
        arr[0] += arr[j];
    }

    return x + 3;
}

int main(int argc, char **argv)
{
    int x, i;
    // int b[5];

    int *a = arr + 1;

    int *c = (int*)malloc(6*sizeof(int));

#pragma hicuda shape c[6]

    int n = atoi(argv[0]);

#pragma hicuda shape a[n]

    x = sum1(a, n);
    x += sum1(&arr, 6);

    x += foo(c);

    x += foo1(3);
    x += foo1(4);

#pragma hicuda shape a[4]

    x += foo(a);

    // int (*b)[5] = 0;
    int (*b)[5];

    for (i = 0; i < 4; ++i) {
        x += (*b)[i];
    }

    arr[0] = 0;

    return x;
}

