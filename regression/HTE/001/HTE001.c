
void bar(float *b, int end)
{
    int i;

#pragma hicuda loop_partition over_thread
    for (i = 0; i < end; ++i)
        b[i] -= b[end];
}

void foo(float *a, int start)
{
    int i;

#pragma hicuda loop_partition over_thread
    for (i = start; i < 1024; ++i)
        a[i] = a[i-1] + a[start];

    bar(a, start);
}

int main(int argc, char **argv)
{
    float arr[1024];
    int i;

#pragma hicuda global alloc arr[*] copyin

#pragma hicuda kernel k_looppart_ex tblock(8) thread(2,3,4,8)

#pragma hicuda loop_partition over_tblock
    for (i = 0; i < 1024; ++i)
        foo(arr, i);

#pragma hicuda loop_partition over_tblock(CYCLIC) over_thread
    for (i = 1023; i >= 0; --i)
        bar(arr, i);

#pragma hicuda kernel_end

#pragma hicuda global free arr

    return 0;
}
