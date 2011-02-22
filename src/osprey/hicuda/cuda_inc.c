/**
 * It appears to be very tricky to get the set of symbols in cuda_runtime.h 
 * that is the same as the set NVCC gets when including cuda_runtime.h. The
 * ultimate solution is to use NVCC to preprocess an empty .cu file and
 * convert it to a WHIRL file. However, using opencc to compile this .ii file
 * leads to syntax error.
 *
 * Compiling an empty C++ file with cuda_runtim.h included, using opencc,
 * does not work due to the same syntax error.
 *
 * For now, we only consider C headers that NVCC will see. They are obtained
 * by checking cuda_inc.cpp1.ii.
 */

// The following header files are re-included in the generated CUDA code.
// We must remove math functions because NVCC complains about the exception
// handling of these functions.
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

