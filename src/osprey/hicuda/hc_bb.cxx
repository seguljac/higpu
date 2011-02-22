/** DAVID CODE BEGIN **/

#include <stdio.h>
#include <assert.h>

#include "wn_util.h"

#include "hc_utils.h"
#include "hc_bb.h"

/* A list of BBs that are not freed */
static hc_bb *bb_list = NULL;
/* Gives an ID to each new BB, also act as a counter. */
static int bb_count = 0;

hc_bb*
get_bblist_head(int *nbbs) {
    if (nbbs != NULL) *nbbs = bb_count;
    return bb_list;
}

void
reset_bblist() {
    bb_list = NULL;
    bb_count = 0;
}

hc_bb*
new_hc_bb() {
    hc_bb *bb = (hc_bb*)malloc(sizeof(hc_bb));

    // Add it to the beginning of bb_list.
    bb->prev = NULL;
    bb->next = bb_list;
    if (bb_list != NULL) bb_list->prev = bb;
    bb_list = bb;

    bb->id = ++bb_count;
    bb->pred = bb->succ = NULL;

    for (unsigned i = 0; i < NUM_DFA; ++i) init_dfa_info(&(bb->dfa[i]));

    // Create an empty block.
    bb->body_blk = WN_CreateBlock();

    return bb;
}

/**
 * Add 'bb' to the end of 'hbp' if it does not exist.
 *
 * Return the new list head of 'hbp'.
 */
hc_bb_ptr*
add_unique_hc_bb_ptr(hc_bb_ptr *hbp, hc_bb *bb) {
    hc_bb_ptr *curr = hbp, *prev = NULL;

    // Check if it has already existed.
    while (curr != NULL) {
        if (curr->node == bb) return hbp;
        prev = curr;
        curr = curr->next;
    }

    curr = (hc_bb_ptr*)malloc(sizeof(hc_bb_ptr));
    curr->node = bb;
    curr->next = NULL;
    if (prev == NULL) {
        hbp = curr;
    } else {
        prev->next = curr;
    }

    return hbp;
}

void
add_pred(hc_bb *bb, hc_bb *pred_bb) {
    bb->pred = add_unique_hc_bb_ptr(bb->pred, pred_bb);
}

void
add_succ(hc_bb *bb, hc_bb *succ_bb) {
    bb->succ = add_unique_hc_bb_ptr(bb->succ, succ_bb);
}

void
append_stmt_to_bb(hc_bb *bb, WN *wn) {
    OPERATOR opr = WN_operator(wn);
    WN *stmt = NULL;

    if (OPERATOR_is_expression(opr)) {
        // Create an artificial EVAL statement.
        stmt = WN_CreateEval(WN_COPY_Tree(wn));
    } else {
        stmt = WN_COPY_Tree(wn);
    }

    // We only accept statements that are not structured-control-flow.
    opr = WN_operator(stmt);
    // NOTE: do not use OPERATOR_is_non_scf, because OPR_CALL, for instance,
    // does not have the non_scf flag set.
    assert(OPERATOR_is_stmt(opr) && !OPERATOR_is_scf(opr));

    WN_INSERT_BlockLast(bb->body_blk, stmt);
}

static void
free_hc_bb_ptr_list(hc_bb_ptr *hbp) {
    hc_bb_ptr *tmp = NULL;

    while (hbp != NULL) {
        tmp = hbp;
        hbp = hbp->next;
        // do not free the BB it points to
        free(tmp);
    }
}

static void
cleanup_hc_bb(hc_bb *bb) {
    // Free the chain of predecessors and successors.
    free_hc_bb_ptr_list(bb->pred);
    free_hc_bb_ptr_list(bb->succ);

    // Free DFA info
    for (unsigned i = 0; i < NUM_DFA; ++i) reset_dfa_info(&(bb->dfa[i]));

    // Free the body block (it is a copy!).
    WN_DELETE_Tree(bb->body_blk);
}

void
cleanup_all_hc_bbs(hc_bb *bbs) {
    hc_bb *tmp = NULL;

    while (bbs != NULL) {
        tmp = bbs;
        bbs = bbs->next;

        cleanup_hc_bb(tmp);
        free(tmp);
    }
}

void
print_all_hc_bbs(hc_bb *bbs) {
    hc_bb *bb = bbs;
    hc_bb_ptr *hbp = NULL;

    while (bb != NULL) {
        printf("BB %d: %u WHIRL nodes\n", bb->id, WN_kid_count(bb->body_blk));

        printf("  pred:");
        hbp = bb->pred;
        while (hbp != NULL) {
            printf(" %d", hbp->node->id);
            hbp = hbp->next;
        }
        printf("\n");

        printf("  succ:");
        hbp = bb->succ;
        while (hbp != NULL) {
            printf(" %d", hbp->node->id);
            hbp = hbp->next;
        }
        printf("\n");

        dfa_info *di = &(bb->dfa[DFA_LIVE_VAR]);
        if (di->in) {
            printf("IN:   ");
            fprint_bit_vector(stdout, di->in);
            printf("\n");
        }

        if (di->out) {
            printf("OUT:  ");
            fprint_bit_vector(stdout, di->out);
            printf("\n");
        }

        if (di->gen) {
            printf("GEN:  ");
            fprint_bit_vector(stdout, di->gen);
            printf("\n");
        }

        if (di->kill) {
            printf("KILL: ");
            fprint_bit_vector(stdout, di->kill);
            printf("\n");
        }

        bb = bb->next;
    }
}

/*** DAVID CODE END ***/
