
/*****************************************************************************
 *
 * This program tests proper type cast for function arguments.
 *
 ****************************************************************************/

struct atom_t
{
    char type[1];
    float x, y;
};

int main (int argc, char * argv[])
{
    struct atom_t atom;
    char type[8];

    if (strcmp(atom.type, type) == 0) return 1;

    return 0;
}

