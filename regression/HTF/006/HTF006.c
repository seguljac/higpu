
/*****************************************************************************
 *
 * Test the code generation for function declarations that are almost the same
 * as standard C functions.
 *
 ****************************************************************************/

#include <stdio.h>

extern int printf1 (__const char *__restrict __format, ...);

extern char func(__const char *__restrict p_c, const char c, ...)
{
    return p_c[0] + c + 1;
}

int main(int argc, char **argv)
{
    char c;

    c = '1';
    c = func(&c, c);

    printf1("%d\n", (int)c);
    printf("%d\n", (int)c);

    return (int)c;
}

