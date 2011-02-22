int arr[6];

int sum(int *a)
{
    int result, i;

    a[2] = a[2] + 1;

#pragma hicuda global alloc a[*] copyin
#pragma hicuda global alloc result

#pragma hicuda kernel test_sum tblock(2) thread(2)

    for (i = 0; i < 4; ++i) {
        result += a[i];
    }

#pragma hicuda kernel_end

#pragma hicuda global free a result

    return result;
}

int main(int argc, char **argv)
{
    int x, i;
    // int b[5];

    int *a = arr + 1;

#pragma hicuda shape a[x+4]

    x = sum(a);
    // x += sum(&b);
    
#pragma hicuda shape a[6]

#pragma hicuda shape a[4]

    // int (*b)[5] = 0;
    int (*b)[5];

    for (i = 0; i < 4; ++i) {
        x += (*b)[i];
    }

    arr[0] = 0;

    return x;
}

