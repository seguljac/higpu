
int main(int argc, char **argv)
{
    int a[4][5];

#pragma hicuda global alloc a[*][*][*] copyin
// #pragma hicuda global alloc a[*][*] copyin

#pragma hicuda global copyout a[*]
#pragma hicuda global free a

    return 0;
}

