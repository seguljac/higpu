/** DAVID CODE BEGIN **/

#ifndef _HICUDA_CFG_H_
#define _HICUDA_CFG_H_

#include "wn.h"

#include "hc_bb.h"

/* Data structure for a control flow graph */

typedef struct hc_bblist_t hc_bblist;

struct hc_bblist_t {
    hc_bb *head;
    int nblocks;        // number of basic blocks

    hc_bb *entry;
    hc_bb *exit;        // only one exit

    // cached DFS traversals
    hc_bb **pre_dfs;
    hc_bb **post_dfs;
};

extern void free_hc_bblist(hc_bblist *hbl);

/* Build the control flow graph for a REGION node. */

/**
 * Return a list of basic blocks.
 */
extern hc_bblist* build_region_cfg(const WN *region);

#endif  // _HICUDA_CFG_H_

/*** DAVID CODE END ***/
