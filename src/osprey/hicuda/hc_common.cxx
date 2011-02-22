#include <stdarg.h>
#include <stdio.h>

#include "wn.h"

#include "hc_common.h"

#define HC_ERROR_CODE 1

void HC_dev_warn(const char *msg, ...)
{
    char *c = NULL;
    va_list arg_ptr;

    asprintf(&c, "\n!!! hiCUDA DevWarn: %s\n\n", msg);

    va_start(arg_ptr, msg);
    vfprintf(stderr, c, arg_ptr);
    va_end(arg_ptr);

    free(c); c = NULL;
}

void HC_warn(const char *msg, ...)
{
    char *c = NULL;
    va_list arg_ptr;

    // Could use <Sharps>.
    asprintf(&c, "\n!!! hiCUDA WARNING: %s\n\n", msg);

    va_start(arg_ptr, msg);
    vfprintf(stderr, c, arg_ptr);
    va_end(arg_ptr);

    free(c); c = NULL;
}

void HC_error(const char *msg, ...)
{
    char *c = NULL;
    va_list arg_ptr;

    asprintf(&c, "\n### hiCUDA ERROR: %s\n\n", msg);

    va_start(arg_ptr, msg);
    vfprintf(stderr, c, arg_ptr);
    va_end(arg_ptr);

    free(c); c = NULL;

    Signal_Cleanup(0);
    exit(HC_ERROR_CODE);
}

