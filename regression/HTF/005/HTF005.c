
/*****************************************************************************
 *
 * This program tests the generation of complex array accesses.
 *
 * Issues to be resolved:
 *
 * - elems[i].nodes[j] is not generated as an array access on a 64-bit
 *   machine. The reason is that Fold_Base exits early due to the mismatch of
 *   the array base type (U8) and the array index type (I4).
 *
 * - Somehow adding the line of elems[i].nodes[j] alters the generated code
 *   for the access a[i][j][k], to a[(int)i][j][k]. Although this does not
 *   affect correctness, it is probably worthwhile to know why.
 *
 ****************************************************************************/

struct elem_t
{
    int nodes[4];
};

int main(int argc, char **argv)
{
    int a[5][6][7];
    int t[5][6];
    struct elem_t *elems;

    int i,j,k;
    int x, b;

    int (*p)[6][7];

    // int c[i][6][k];

    x = elems[i].nodes[j];

    b = x + a[i][j][k];

    a[j][k+1][i] = 1 + b;

    b = t[j][k];

    p = &a[1];

    b += (*p)[j][k];
    b += p[i][j][k];

    return b;
}

