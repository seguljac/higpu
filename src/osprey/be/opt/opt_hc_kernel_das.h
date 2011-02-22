/** DAVID CODE BEGIN **/
#ifdef HICUDA

#ifndef _OPT_HC_KERNEL_DAS_H_
#define _OPT_HC_KERNEL_DAS_H_

/*****************************************************************************
 *
 * Walk through the given procedure (FUNC_ENTRY), and do the following:
 * - Make sure that kernel regions are not nested.
 * - Make sure indirect loads/stores have ARRAY as the address.
 * - For each DEF/USE in the DU-chain, mark it inside/outside a kernel region
 *   (if inside, which kernel region).
 *
 * <parent_kernel_map> is a map from each WN node to the symbol of its parent
 * kernel region (or ST_IDX_ZERO). It should be empty before the initial call
 * and gets filled at the end.
 *
 ****************************************************************************/

extern void HC_kernel_preprocess(WN *wn, WN_MAP parent_kernel_map);

#endif  /* _OPT_HC_KERNEL_DAS_H_ */

#endif  // HICUDA
/*** DAVID CODE END ***/
