
int arr[6];

int foo(int *b, int *a)
{
    int x = a[1];

    a[0] = a[0] + 1;

#pragma hicuda global alloc b[*] copyin

    sum(b);

#pragma hicuda global free b

    return x + a[0];
}

int sum(int *a)
{
    int result, i;

    a[2] = a[2] + 1;

#pragma hicuda global alloc a[*] copyin
#pragma hicuda global alloc result

    result += foo(a, a);

#pragma hicuda kernel test_sum tblock(2) thread(2)

    for (i = 0; i < 4; ++i) {
        result += a[i];
    }

#pragma hicuda kernel_end

#pragma hicuda global free a result

    return result;
}

int foo1(int x)
{
// #pragma hicuda kernel test_foo1 tblock(2) thread(3)
    x = x + 1;
// #pragma hicuda kernel_end

    return x + 3;
}

int main(int argc, char **argv)
{
    int x, i;
    // int b[5];

    int *a = arr + 1;
#pragma hicuda shape a[4]

    x = sum(a);
    // x += sum(&b);
    
#pragma hicuda shape a[6]

    x += foo(a, a);

    x += foo1(3);
    x += foo1(4);

#pragma hicuda shape a[4]

    x += foo(a, a);

    // int (*b)[5] = 0;
    int (*b)[5];

    for (i = 0; i < 4; ++i) {
        x += (*b)[i];
    }

    arr[0] = 0;

    return x;
}

