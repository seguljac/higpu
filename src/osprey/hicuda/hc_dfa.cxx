/** DAVID CODE BEGIN **/

#include <assert.h>

#include "hc_bb.h"
#include "hc_cfg.h"
#include "hc_dfa.h"

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void
reset_dfa_info(dfa_info *di) {
    if (di->in != NULL) {
        free_bit_vector(di->in);
        di->in = NULL;
    }
    if (di->out != NULL) {
        free_bit_vector(di->out);
        di->out = NULL;
    }
    if (di->gen != NULL) {
        free_bit_vector(di->gen);
        di->gen = NULL;
    }
    if (di->kill != NULL) {
        free_bit_vector(di->kill);
        di->kill = NULL;
    }
}

void
get_analysis_property(DFA_TYPE type, bool *isAny, bool *direction) {
    switch (type) {
        case DFA_LIVE_VAR:
            *isAny = true; *direction = false;
            break;
        default:
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// This is to speed up the recursive call 'dfs_visit'.
static hc_bb **preorder = NULL;
static int prenext = 0;
static hc_bb **postorder = NULL;
static int postnext = 0;

static void
reset_dfs_visit_states(hc_bb **pre_order, hc_bb **post_order) {
    preorder = pre_order;
    prenext = 0;
    postorder = post_order;
    postnext = 0;
}

/**
 * Recursively traverse a BB list with entry 'bb' and store the pre-order
 * and post-order traversal in 'preorder' and 'postorder'.
 *
 * All BBs being traversed are reachable.
 */
static void
dfs_visit(hc_bb *bb) {
    assert(bb->dfs_state == DFS_NYS);

    preorder[prenext++] = bb;

    bb->dfs_state = DFS_WIP;

	// Go through its successors.
    hc_bb_ptr *curr = bb->succ;
	while (curr != NULL) {
        if (curr->node->dfs_state == DFS_NYS) dfs_visit(curr->node);
		curr = curr->next;
	}

	postorder[postnext++] = bb;

    bb->dfs_state = DFS_DONE;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/**
 * Performa MEET_OP on a reachable BB in a forward DFA.
 *
 * IN set = MEET_OP < OUT sets of all reachable predecessors >
 */
static void
forward_meetop(hc_bb *bb, DFA_TYPE type, bop_bits meetop) {
    dfa_info *di = &(bb->dfa[type]), *pred_di = NULL;
    hc_bb_ptr *hbp = bb->pred;

    /* We only perform MEET_OP with reachable predecessors. */

    bool first_reachable_pred = true;
    while (hbp != NULL) {
        pred_di = &(hbp->node->dfa[type]);

        if (pred_di->out != NULL) {
            if (first_reachable_pred) {
                copy_bits(di->in, pred_di->out);
                first_reachable_pred = false;
            } else {
                meetop(di->in, pred_di->out);
            }
        }

        hbp = hbp->next;
    }

    // We must have at least one reachable predecessor because this BB
    // is reachable as well.
    assert(!first_reachable_pred);
}

/**
 * Perform forward DFA (using the DFS preorder in 'list').
 *
 * We assume that the GEN and KILL set of each BB have been computed.
 */
static void
forward_dfa_solver(hc_bblist *list, DFA_TYPE type, bool isAny, int size) {
    hc_bb **dfsorder = list->pre_dfs;
    int i, nblocks = list->nblocks, n_dfs_blks;
    hc_bb *entry_bb = list->entry, *exit_bb = list->exit;

    hc_bb *bb;		    // temp var for a basic block
    bit_vector *bv;		// temp var for a bit-vector
    dfa_info *di;		// temp var for a DFA info struct

    // Prologue: init IN/OUT set of each reachable BB.
    assert(dfsorder[0] == entry_bb);

    // Init IN/OUT of the remaining BBs in the DFS.
    for (i = 0; i < nblocks && (bb = dfsorder[i]) != NULL; ++i) {
        di = &(bb->dfa[type]);

        // Allocate the IN and OUT set.
        assert(di->in == NULL);
        di->in = new_bit_vector(size);
        assert(di->out == NULL);
        di->out = new_bit_vector(size);

        // Initialization
        if (bb == entry_bb) {
            // Entry's IN and OUT are always empty.
            set_all_bits(di->in, false);
            set_all_bits(di->out, false);
        } else if (bb != exit_bb) {
            // We only need to init OUT sets for non-exit BBs.
            assert(di->gen != NULL && di->kill != NULL);
            if (isAny) {
                // OUT = GEN
                copy_bits(di->out, di->gen);
            } else {
                // OUT = GEN <UNION> (U - KILL)
                negate_bits(di->out, di->kill);
                or_bits(di->out, di->gen);
            }
        }
    }
    // Side effect: find the # of BB (i.e. reachable) in the DFS.
    n_dfs_blks = i;

    // Determine the meet operation.
    register bop_bits meetop = isAny ? or_bits : and_bits;

    // Main loop
    bool changed = true;
    bv = new_bit_vector(size);		// temp bit vector
    while (changed) {
        changed = false;

        // Skip the entry block.
        for (i = 1; i < n_dfs_blks; ++i) {
            bb = dfsorder[i];
            // Skip the exit block.
            if (bb == exit_bb) continue;

            // This BB must be reachable.
            // Meet operation.
            forward_meetop(bb, type, meetop);

            // Transfer operation (IN -> OUT).
            di = &(bb->dfa[type]);
            copy_bits(bv, di->in);
            subtract_bits(bv, di->kill);
            or_bits(bv, di->gen);

            // Check for changes in the OUT set.
            if (!bits_are_equal(bv, di->out)) {
                copy_bits(di->out, bv);
                changed = true;
            }
        }
    }
    // Clean up.
    free_bit_vector(bv); bv = NULL;

    // Epilogue: set exit's IN and OUT (if it is reachable).
    di = &(exit_bb->dfa[type]);
    if (di->in != NULL) {
        forward_meetop(exit_bb, type, meetop);
        // Set exit's OUT (same as IN).
        copy_bits(di->out, di->in);
    }
}

/**
 * Performa MEET_OP on a reachable BB in a backward DFA.
 *
 * OUT set = MEET_OP < IN sets of all reachable successors >
 */
static void
backward_meetop(hc_bb *bb, DFA_TYPE type, bop_bits meetop) {
    dfa_info *di = &(bb->dfa[type]), *succ_di = NULL;
    hc_bb_ptr *hbp = bb->succ;

    /* We only perform MEET_OP with reachable successors. */

    bool first_reachable_succ = true;
    while (hbp != NULL) {
        succ_di = &(hbp->node->dfa[type]);

        if (succ_di->in != NULL) {
            if (first_reachable_succ) {
                copy_bits(di->out, succ_di->in);
                first_reachable_succ = false;
            } else {
                meetop(di->out, succ_di->in);
            }
        }

        hbp = hbp->next;
    }

    // We must have at least one reachable successor because this BB
    // is reachable and has at least one successor.
    assert(!first_reachable_succ);
}

/**
 * Perform backward DFA (using the DFS postorder in 'list').
 *
 * We assume that the GEN and KILL set of each BB have been computed.
 */
static void
backward_dfa_solver(hc_bblist *list, DFA_TYPE type, bool isAny, int size) {
    hc_bb **dfsorder = list->post_dfs;
    int i, nblocks = list->nblocks, n_dfs_blks;
    hc_bb *entry_bb = list->entry, *exit_bb = list->exit;

    hc_bb *bb;		    // temp var for a basic block
    bit_vector *bv;		// temp var for a bit-vector
    dfa_info *di;		// temp var for a DFA info struct

    // Init IN/OUT set of each reachable in the DFS.
    for (i = 0; i < nblocks && (bb = dfsorder[i]) != NULL; ++i) {
        di = &(bb->dfa[type]);

        // Allocate the IN and OUT set.
        assert(di->in == NULL);
        di->in = new_bit_vector(size);
        assert(di->out == NULL);
        di->out = new_bit_vector(size);

        // Initialization
        if (bb == exit_bb) {
            // Exit's IN and OUT are always empty.
            set_all_bits(di->in, false);
            set_all_bits(di->out, false);
        } else if (bb != entry_bb) {
            // We only need to init IN sets for non-entry BBs.
            assert(di->gen != NULL && di->kill != NULL);
            if (isAny) {
                // IN = GEN
                copy_bits(di->in, di->gen);
            } else {
                // IN = GEN <UNION> (U - KILL)
                negate_bits(di->in, di->kill);
                or_bits(di->in, di->gen);
            }
        }
    }
    // Side effect: find the # of BB (i.e. reachable) in the DFS.
    n_dfs_blks = i;

    // Determine the meet operation.
    register bop_bits meetop = isAny ? or_bits : and_bits;

    // Main loop
    bool changed = true;
    bv = new_bit_vector(size);		// temp bit vector
    while (changed) {
        changed = false;

        // print_all_hc_bbs(list->head);

        for (i = 0; i < n_dfs_blks; ++i) {
            bb = dfsorder[i];

            // Skip the entry and exit BB.
            if (bb == entry_bb || bb == exit_bb) continue;

            // This BB must be reachable.
            // Meet operation
            backward_meetop(bb, type, meetop);

            // Transfer operation (OUT -> IN).
            di = &(bb->dfa[type]);
            copy_bits(bv, di->out);
            subtract_bits(bv, di->kill);
            or_bits(bv, di->gen);

            // Check for changes in the IN set.
            if (!bits_are_equal(bv, di->in)) {
                copy_bits(di->in, bv);
                changed = true;
            }
        }
    }
    // Clean up.
    free_bit_vector(bv); bv = NULL;

    // Epilogue: set entry's IN and OUT (entry is always reachable).
    di = &(entry_bb->dfa[type]);
    assert(di->out != NULL);
    backward_meetop(entry_bb, type, meetop);
    // Set entry's IN (same as OUT).
    copy_bits(di->in, di->out);
}

/**
 * A generic engine that can do ANY/ALL FOR/BACKWARD data flow analyses.
 */
void
dfa_solver(hc_bblist *list, DFA_TYPE type, int size) {
	// Determine the property of this analysis.
	bool isAny, isForward;
	get_analysis_property(type, &isAny, &isForward);

	// Generate the traversal order if necessary.
	if (list->pre_dfs == NULL) {
        assert(list->post_dfs == NULL);

		// Construct both pre/post-order DFS (init to 0).
        int nblocks = list->nblocks;
		list->pre_dfs = (hc_bb**)calloc(nblocks, sizeof(hc_bb*));
		list->post_dfs = (hc_bb**)calloc(nblocks, sizeof(hc_bb*));

		// Init each block's DFS state.
        hc_bb *bb = list->head;
        while (bb != NULL) {
            bb->dfs_state = DFS_NYS;
            bb = bb->next;
        }

		// Start DFS visit.
        reset_dfs_visit_states(list->pre_dfs, list->post_dfs);
		dfs_visit(list->entry);
	}

	// Do the actual analysis.
	if (isForward) {
		forward_dfa_solver(list, type, isAny, size);
	} else {
		backward_dfa_solver(list, type, isAny, size);
	}
}

/*** DAVID CODE END ***/
