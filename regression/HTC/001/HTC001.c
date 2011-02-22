
#include "HTC001.h"

int glob_a[5];

void kernel_callee(float (*E)[8][8])
{
#pragma hicuda shared alloc E[*][*] copyin
    // (*E)[0][0] = 0;
#pragma hicuda shared remove E
}

int main(int argc, char **argv)
{
    float A[128], B[128], C[128], D[256], E[8][8];

#pragma hicuda global alloc E[*][*] copyin

#pragma hicuda constant copyin A[*]
#pragma hicuda constant copyin B[*]

#pragma hicuda kernel k_name tblock(2) thread(32)

// #pragma hicuda shared alloc C[*] copyin
    kernel_callee(&E);
// #pragma hicuda shared remove C

    E[0][0] = B[2] + A[1];

// #pragma hicuda shared alloc D[*] copyin
    kernel_callee(&E);
// #pragma hicuda shared remove D

#pragma hicuda kernel_end

#pragma hicuda constant remove A B

#pragma hicuda global free E

    return 0;
}
