#ifndef _HC_COMMON_H_
#define _HC_COMMON_H_

#include "defs.h"

/*****************************************************************************
 *
 * Common hiCUDA routines to be used by multiple modules
 * 
 * We should try to keep the file dependencies as simple as possible.
 *
 ****************************************************************************/

/**
 * Development warning within the hicuda compiler.
 */
extern void HC_dev_warn(const char *msg, ...);

/*****************************************************************************
 *
 * Output the given warning message to stderr.
 *
 ****************************************************************************/

extern void HC_warn(const char *msg, ...);

/*****************************************************************************
 *
 * hiCUDA-specific error reporting (like Is_True).
 *
 ****************************************************************************/

#define HC_assert(Cond, ParmList ) \
    ( Cond ? (void) 1  : ( HC_error ParmList ) )

extern void HC_error(const char *msg, ...);

#endif  // _HC_COMMON_H_
