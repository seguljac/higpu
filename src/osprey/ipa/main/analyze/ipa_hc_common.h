/** DAVID CODE BEGIN **/

#ifndef _IPA_HICUDA_H_
#define _IPA_HICUDA_H_

#include <stdio.h>

#include "defs.h"
#include "cxx_template.h"
#include "cxx_hash.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef DYN_ARRAY<ST_IDX> HC_SYM_LIST;

typedef HASH_TABLE<ST_IDX,ST_IDX> HC_SYM_MAP;
typedef HASH_TABLE_ITER<ST_IDX,ST_IDX> HC_SYM_MAP_ITER;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

extern void IPA_Call_Graph_print(FILE *fp);

#endif  /* _IPA_HICUDA_H_ */

/*** DAVID CODE END ***/

