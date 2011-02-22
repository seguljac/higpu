
int arr[5];

int size;

int foo(int (*b)[5], int (*at)[5])
{
    int x = (*at)[1];

    (*at)[0] = (*at)[0] + 1;

#pragma hicuda global alloc b[*] copyin

    sum(b);

#pragma hicuda global free b

    return x + (*at)[0];
}

int sum(int (*a)[5])
{
    int result, i;

    (*a)[2] = (*a)[2] + 1;

#pragma hicuda global alloc a[*] copyin
#pragma hicuda global alloc result

    result = foo(a, a);

#pragma hicuda kernel test_sum tblock(2) thread(2)

    for (i = 0; i < 4; ++i) {
        result += (*a)[i];
    }

#pragma hicuda kernel_end

#pragma hicuda global free a result

    return result;
}

int main(int argc, char **argv)
{
    int x, i;
    // int b[5];

    x = sum(&arr);
    // x += sum(&b);
    
    // int (*b)[5] = 0;
    int (*b)[5];

    for (i = 0; i < 4; ++i) {
        x += (*b)[i];
    }

    arr[0] = 0;

    return x;
}

