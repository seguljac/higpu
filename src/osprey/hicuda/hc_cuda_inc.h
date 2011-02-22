/** DAVID CODE BEGIN **/

#ifndef _HICUDA_CUDA_INC_H_
#define _HICUDA_CUDA_INC_H_

#include "hc_symtab.h"

/*****************************************************************************
 *
 * Routines that mark global symbols and types declared in cuda_runtime.h
 * (including those declared in the C headers included in cuda_runtime.h) in
 * the current global symbol table.
 *
 * For symbols, only CLASS_VAR and CLASS_FUNC are considered. Each CUDA symbol
 * will have the attribute ST_ATTR_IS_CUDA_RUNTIME set.
 *
 * Each CUDA type will have the flag TY_IS_CUDA_RUNTIME set.
 *
 * The global symtab for cuda_runtime.h is obtained by loading the WHIRL file
 * 'cuda_inc.B'.
 *
 ****************************************************************************/

extern void HC_mark_cuda_runtime_symbols_and_types(hc_symtab *hcst);

#endif  // _HICUDA_CUDA_INC_H_

/*** DAVID CODE END ***/

