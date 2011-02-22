int bar(int x)
{
    return x + 1;
}

void foo(int (*a)[10])
{
    (*a)[9] = bar((*a)[9]);
}

int main(int argc, char **argv)
{
    int arr[10];

    int lbnd = argc;
    if (lbnd < 0) lbnd = 0;
    if (lbnd > 9) lbnd = 9;

#pragma hicuda global alloc arr[lbnd:9]

    foo(arr);

#pragma hicuda kernel k_test tblock(2) thread(4)

    foo(arr);

#pragma hicuda kernel_end

#pragma hicuda global copyout arr[lbnd:9]
#pragma hicuda global free arr

    return 0;
}
