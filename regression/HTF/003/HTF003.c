
/*****************************************************************************
 *
 * Test the code generation of various standard C (or C99) functions.
 *
 ****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    int src, dest;

    sscanf(argv[0], "%d", &src);

    printf("Hello world!\n");
    memcpy(&dest, &src, 4);

    // fscanf
    FILE *fin = fopen("test.txt", "r");
    if (fin != NULL)
    {
        float x,y;

        if (fscanf(fin, "%f %f", &x, &y) == 2)
        {
            printf("%f %f", x, y);
        }

        fclose(fin); fin = NULL;
    }

    // getopt
    int c;
    while ((c = getopt(argc, argv, "a:b:c:d")) != EOF)
    {
        switch(c)
        {
            case 'a':
                printf("a"); break;
        }
    }

    return 0;
}

