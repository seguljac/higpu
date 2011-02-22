/** DAVID CODE BEGIN **/

#ifndef _HICUDA_BB_H_
#define _HICUDA_BB_H_

#include "wn.h"

#include "hc_dfa.h"

/* Definition of a basic block at the VH WHIRL level. */

typedef struct hc_bb_t hc_bb;
typedef struct hc_bb_ptr_t hc_bb_ptr;

struct hc_bb_t {
    // let all nodes form a chain to facilitate deallocation
    hc_bb *prev;
    hc_bb *next;

    int id;

    // used in control flow graph (CFG)
    hc_bb_ptr *pred;
    hc_bb_ptr *succ;

    // info used in data flow analysis
    DFS_STATE dfs_state;
    dfa_info dfa[NUM_DFA];

    // a BLOCK node containing a list of WHIRL nodes in the BB
    WN *body_blk;
};

/* A linked list of basic block nodes */
struct hc_bb_ptr_t {
    hc_bb *node;
    hc_bb_ptr *next;
};

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/**
 * Manipulate the implicit BB list head.
 */
extern hc_bb* get_bblist_head(int *nbbs);
extern void reset_bblist();

/**
 * Add 'bb' to the end of 'hbp' if it does not exist.
 *
 * Return the new list head of 'hbp'.
 */
extern hc_bb_ptr* add_unique_hc_bb_ptr(hc_bb_ptr *hbp, hc_bb *bb);

/**
 * Create an empty basic block and link it in the implicit BB list.
 */
extern hc_bb* new_hc_bb();

/**
 * Add a predecessor/successor basic block.
 * Does uniqueness check.
 */
extern void add_pred(hc_bb *bb, hc_bb *pred_bb);
extern void add_succ(hc_bb *bb, hc_bb *succ_bb);

inline void
chain_bbs(hc_bb *bb1, hc_bb *bb2) {
    add_succ(bb1, bb2);
    add_pred(bb2, bb1);
}

/**
 * A copy of 'stmt' is made before being inserted.
 */
extern void append_stmt_to_bb(hc_bb *bb, WN *stmt);

/**
 * Deallocate the BB list.
 */
extern void cleanup_all_hc_bbs(hc_bb *bbs);

extern void print_all_hc_bbs(hc_bb *bbs);

#endif  // _HICUDA_BB_H_

/*** DAVID CODE END ***/
