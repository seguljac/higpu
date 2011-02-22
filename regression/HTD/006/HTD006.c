
int arr[5];

int foo1(int x)
{
    return x + 1;
}

int foo(int (*b)[5], int (*at)[5])
{
    int x = (*at)[2];

    (*at)[0] = (*at)[0] + 1;

#pragma hicuda global alloc b[*] copyin

    x += foo1(x);

    sum(b);

#pragma hicuda global free b

    return x + (*at)[0];
}

int sum(int (*a)[5])
{
    int result, i;

#pragma hicuda global alloc result

    result = foo(a, a);

    result += foo(a, a);

#pragma hicuda kernel test_sum tblock(2) thread(2)

#pragma hicuda loop_partition over_tblock over_thread
    for (i = 0; i < 4; ++i) {
        result += (*a)[i];
    }

#pragma hicuda kernel_end

#pragma hicuda global free result

    return result;
}

int main(int argc, char **argv)
{
    int x, i;
    // int b[5];

    x = foo(&arr, &arr);
    // x += sum(&b);
    
    // int (*b)[5] = 0;
    int (*b)[5];

    for (i = 0; i < 4; ++i) {
        x += (*b)[i];
    }

    arr[0] = 0;

    return x;
}

