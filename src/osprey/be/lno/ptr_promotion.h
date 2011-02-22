/** DAVID CODE BEGIN **/

/*****************************************************************************
 *
 * Pre-optimizations needed by the hiCUDA compiler
 *
 ****************************************************************************/

#ifndef _LNO_PTR_PROMOTION_H_
#define _LNO_PTR_PROMOTION_H_

#include "wn.h"

#ifdef __cplusplus
extern "C" {
#endif

class DU_MANAGER;
class ALIAS_MANAGER;

extern WN* HC_extract_arr_base(WN *addr_wn, ST_IDX& arr_st_idx);

/*****************************************************************************
 *
 * Transform accesses like a[1] where <a> is of type int(*)[5], into ARRAYs.
 * This allows:
 * 1) array access redirection to GPU variables, and
 * 2) more accurate IP array section analysis later on.
 *
 * <func_wn> is a FUNC_ENTRY node.
 *
 ****************************************************************************/

extern void transform_ptr_access_to_array(WN *func_wn,
        DU_MANAGER *du_mgr, ALIAS_MANAGER *alias_mgr);

#ifdef __cplusplus
}
#endif

#endif  /* _LNO_PTR_PROMOTION_H_ */

/*** DAVID CODE END ***/
