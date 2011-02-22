
/*****************************************************************************
 *
 * Test the code generation of array initializer.
 *
 ****************************************************************************/

#include <stdio.h>

int main(int argc, char **argv)
{
    float* arr[3] = {NULL, NULL, NULL};

    arr[2] = (float*)1;

    printf("%x, %x, %x\n", arr[0], arr[1], arr[2]);

    return 0;
}

