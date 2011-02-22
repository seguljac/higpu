#ifndef _COMMON_H_
#define _COMMON_H_

#include <sys/time.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a matrix with random elements.
 */
void randomInitArr(float *data, unsigned int size);

/**
 * Output a matrix to the standard output.
 */
void printMatrix(float *mat, unsigned int h, unsigned int w);

void computeGold(float *C, const float *A, const float *B,
    unsigned int HA, unsigned int WA, unsigned int WB);

extern void compare_matrices(float *mat, float *ref, unsigned int nelems);

/**
 * Get the time difference in milliseconds between 'start' and 'end'.
 */
inline float
get_time_diff(const struct timeval *start, const struct timeval *end) {
    return  (float)(1000.0F * (end->tv_sec - start->tv_sec)
        + (0.001F * (end->tv_usec - start->tv_usec)));
}

#ifdef __cplusplus
}
#endif

#endif  /* _COMMON_H_ */
