/** DAVID CODE BEGIN **/

#ifndef _HICUDA_LIVEVAR_H_
#define _HICUDA_LIVEVAR_H_

#include "hc_cfg.h"

/**
 * This function provides a plugin to the DFA framework so that it can do
 * Live Variable Analysis.
 *
 * The information the DFA solver (hc_dfa.h) needs is the GEN and KILL sets
 * of all BB's in 'hbl', which is computed here.
 *
 * This function must be called before the DFA solver.
 *
 * NOTE: the GEN and KILL set of entry and exit BBs are not determined
 * because they are not needed by DFA solver.
 */

extern void compute_lv_genkill_sets(hc_bblist *hbl, int n_syms);

#endif  // _HICUDA_LIVEVAR_H_

/*** DAVID CODE END ***/
