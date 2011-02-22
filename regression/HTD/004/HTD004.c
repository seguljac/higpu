
#include <stdlib.h>

void fooA(int*);
void fooB(int*);

void fooA(int *a)
{
    fooB(a);

#pragma hicuda global alloc a[*] copyin

    fooB(a);

#pragma hicuda kernel testA tblock(2) thread(3)

    fooB(a);

#pragma hicuda kernel_end

#pragma hicuda global free a
}

void fooB(int *b)
{
    fooA(b);
}

int main(int argv, char **argc)
{
    int *arr = (int*)malloc(5 * sizeof(int));

#pragma hicuda shape arr[5]

    fooA(arr);

    return 0;
}
