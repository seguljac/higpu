/** DAVID CODE BEGIN **/

#ifndef _HICUDA_CMP_ST_IDX_H_
#define _HICUDA_CMP_ST_IDX_H_

/**
 * Data structure for compact symbol index, given that no new symbols are
 * added when the compact indices are used.
 *
 * We can count the number of symbols in each scope and merge them into
 * a flat array. For example, GLOBAL_SYMTAB has N symbols and CURRENT_SYMTAB
 * has M symbols, the 1st symbol in CURRENT_SYMTAB would have compact index
 * N+1 (one-based). CMP_ST_IDX_ZERO is reserved for error.
 */

#include "symtab.h"
#include "bitvector.h"


typedef mUINT32 CMP_ST_IDX;

static const CMP_ST_IDX CMP_ST_IDX_ZERO = 0;


extern void init_cmp_st_idx();
extern void reset_cmp_st_idx();

extern int total_num_symbols();

extern CMP_ST_IDX cmp_st_idx(ST_IDX st_idx);

extern ST_IDX regular_st_idx(CMP_ST_IDX idx);

/**
 * Convert a bit-vector, whose bit-index is a CMP_ST_IDX relative to
 * CMP_ST_IDX_ZERO+1 (i.e. index N is CMP_ST_IDX_ZERO + N + 1), to an array
 * of ST_IDX whose CMP_ST_IDX is turned on in the bit-vector.
 *
 * Return the number of symbols filled in 'st_list'.
 * 
 * This is used to interpret DFA results.
 */
extern int bitvector_to_stlist(bit_vector *idx_bv, ST_IDX *st_list);


#endif  // _HICUDA_CMP_ST_IDX_H_

/*** DAVID CODE END ***/
