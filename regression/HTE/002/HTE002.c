
void bar(float *b, int end)
{
    int k;

#pragma hicuda loop_partition over_thread
    for (k = 0; k < end; ++k)
        b[k] -= b[end];
}

void foo(float *a, int start)
{
    int p;

#pragma hicuda loop_partition over_thread
    for (p = start; p < 1024; ++p)
        bar(a, p);
}

int main(int argc, char **argv)
{
    float arr[1024];
    int i, j;

#pragma hicuda global alloc arr[*] copyin

#pragma hicuda kernel k_example tblock(8) thread(2,3,4,8)

#pragma hicuda loop_partition over_tblock
    for (i = 0; i < 1024; ++i)
        foo(arr, i);

#pragma hicuda loop_partition over_tblock(CYCLIC) over_thread
    for (j = 1023; j >= 0; --j)
        bar(arr, j);

#pragma hicuda kernel_end
}
