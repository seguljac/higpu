#include <stdio.h>

void loop_test(float *a, float *b, float *c, float *d,
        int limit_b, int limit_c, int limit_d)
{
    int i, j;
    // Data depedences between the two loops will NOT be respected once turned
    // into CUDA code.
#pragma hicuda kernel MYKERNEL tblock(10) thread(16)

#pragma hicuda loop_partition over_tblock(CYCLIC) over_thread
    for (i = 0; i < limit_b; ++i)
    {
        b[i] += a[i] + i;
    }

#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < limit_c; ++i)
    {
        c[i] += a[i] + 1 + i;
    }

#pragma hicuda loop_partition over_tblock
    for (j = 0; j < limit_d; ++j)
    {
        d[j] += a[j+1];
    }
#pragma hicuda kernel_end
}

void loop_test1(float *a, float *b, float *c, float *d,
        int limit_b, int limit_c, int limit_d)
{
    int i, j;
    // Data depedences between the two loops will NOT be respected once turned
    // into CUDA code.
#pragma hicuda loop_partition over_tblock(CYCLIC) over_thread
    for (i = 0; i < limit_b; ++i)
    {
        b[i] += a[i] + i;
    }

#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < limit_c; ++i)
    {
        c[i] += a[i] + 1 + i;
    }

#pragma hicuda loop_partition over_tblock
    for (j = 0; j < limit_d; ++j)
    {
        d[j] += a[j+1];
    }
}

int main(int argc, char **argv)
{
    int i, j;

    float a[1000], b[1000], c[1000], d[1000];

    for (i = 0; i < 1000; ++i)
    {
        a[i] = 1;
        d[i] = 2;
    }

    int limit_b = 900, limit_c = 901, limit_d = 902;

#pragma hicuda constant copyin a[*]
#pragma hicuda global alloc b[*] clear
#pragma hicuda global alloc c[*] clear
#pragma hicuda global alloc d[*] copyin

    loop_test(a, b, c, d, limit_b, limit_c, limit_d);

#pragma hicuda global copyout b[*]
#pragma hicuda global copyout c[*]
#pragma hicuda global copyout d[*]

    for (i = 0; i < 1000; ++i)
    {
        float b_i = 0, c_i = 0, d_i = 2;
        if (i < limit_b) b_i = i + 1;
        if (i < limit_c) c_i = i + 2;
        if (i < limit_d) d_i = 3;

        if (b[i] != b_i) { printf("b[%d] is incorrect!\n", i); return 1; }
        if (c[i] != c_i) { printf("c[%d] is incorrect!\n", i); return 1; }
        if (d[i] != d_i) { printf("d[%d] is incorrect!\n", i); return 1; }
    }

    for (i = 0; i < 1000; ++i)
    {
        a[i] = 1;
        d[i] = 2;
    }

    limit_b = 902; limit_c = 901; limit_d = 900;

#pragma hicuda constant copyin a[*]
#pragma hicuda global alloc b[*] clear
#pragma hicuda global alloc c[*] clear
#pragma hicuda global alloc d[*] copyin

#pragma hicuda kernel MYKERNEL1 tblock(10) thread(16)
    loop_test1(a, b, c, d, limit_b, limit_c, limit_d);
#pragma hicuda kernel_end

#pragma hicuda global copyout b[*]
#pragma hicuda global copyout c[*]
#pragma hicuda global copyout d[*]

    for (i = 0; i < 1000; ++i)
    {
        float b_i = 0, c_i = 0, d_i = 2;
        if (i < limit_b) b_i = i + 1;
        if (i < limit_c) c_i = i + 2;
        if (i < limit_d) d_i = 3;

        if (b[i] != b_i) { printf("b[%d] is incorrect!\n", i); return 1; }
        if (c[i] != c_i) { printf("c[%d] is incorrect!\n", i); return 1; }
        if (d[i] != d_i) { printf("d[%d] is incorrect!\n", i); return 1; }
    }

    printf("PASSED\n");

    return 0;
}

