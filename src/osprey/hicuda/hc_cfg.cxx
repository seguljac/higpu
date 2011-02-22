/** DAVID CODE BEGIN **/

#include <assert.h>

#include "hc_cfg.h"

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

static hc_bb *curr_bb = NULL;

inline void
add_wn_to_curr_bb(WN *wn) {
    append_stmt_to_bb(curr_bb, wn);
}

static void
push_new_bb() {
    hc_bb *new_bb = new_hc_bb();
    chain_bbs(curr_bb, new_bb);
    curr_bb = new_bb;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

typedef struct cfg_work_node_t cfg_work_node;

struct cfg_work_node_t {
    // They are all references.
    hc_bb *curr;
    hc_bb *exit;
    WN *blk;

    cfg_work_node *next;
};

static cfg_work_node *work_queue = NULL;

/**
 * Return the new exit, which should be the curr.
 */
static void
new_cfg_work(hc_bb *curr, hc_bb *exit, WN *blk) {
    cfg_work_node *node = (cfg_work_node*)malloc(sizeof(cfg_work_node));
    node->curr = curr;
    node->exit = exit;
    node->blk = blk;

    // Add it to the beginning of the work queue.
    node->next = work_queue;
    work_queue = node;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

typedef struct label_tab_t label_tab;

struct label_tab_t {
    LABEL_IDX label;
    hc_bb *label_bb;
    hc_bb_ptr *goto_bbs;

    label_tab *next;
};

static label_tab *ltab = NULL;

/* Create a new one if 'label' does not exist. */
static label_tab*
find_label(LABEL_IDX label) {
    label_tab *curr = ltab;

    while (curr != NULL) {
        if (curr->label == label) return curr;
        curr = curr->next;
    }

    curr = (label_tab*)malloc(sizeof(label_tab));
    curr->label = label;
    curr->label_bb = NULL;
    curr->goto_bbs = NULL;

    // Add it at the beginning.
    curr->next = ltab;
    ltab = curr;

    return curr;
}

void
add_goto_bb(LABEL_IDX label, hc_bb *goto_bb) {
    label_tab *lt = find_label(label);
    lt->goto_bbs = add_unique_hc_bb_ptr(lt->goto_bbs, goto_bb);
}

void
add_label_bb(LABEL_IDX label, hc_bb *label_bb) {
    label_tab *lt = find_label(label);
    assert(lt->label_bb == NULL);
    lt->label_bb = label_bb;
}

void
link_gotos_and_labels() {
    label_tab *tmp = NULL;
    hc_bb_ptr *hbp = NULL, *tmp_hbp = NULL;

    while (ltab != NULL) {
        assert(ltab->label_bb != NULL || ltab->goto_bbs == NULL);

        hbp = ltab->goto_bbs;
        while (hbp != NULL) {
            chain_bbs(hbp->node, ltab->label_bb);
            // Free the node.
            tmp_hbp = hbp;
            hbp = hbp->next;
            free(tmp_hbp);
        }

        // Free the node.
        tmp = ltab;
        ltab = ltab->next;
        free(tmp);
    }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

static hc_bb *region_exit = NULL;

static void
handle_blk_node(WN *node) {
    OPERATOR opr = WN_operator(node);

    // Dead code elimination
    if (curr_bb == NULL && opr != OPR_LABEL) return;

    switch (opr) {
        case OPR_FUNC_ENTRY:
        case OPR_BLOCK:
        case OPR_GOTO_OUTER_BLOCK:
            Fail_FmtAssertion("CFG: meet %s\n", OPERATOR_name(opr));

        // Are these two correct?
        case OPR_REGION_EXIT:
        case OPR_ALTENTRY:
        case OPR_SWITCH:
        case OPR_CASEGOTO:
        case OPR_COMPGOTO:
        case OPR_AGOTO:
            fprintf(stderr, "CFG: ignore %s nodes.\n", OPERATOR_name(opr));
            break;

        case OPR_REGION: {
            // A new piece of work on its body block
            hc_bb *new_exit = new_hc_bb();
            new_cfg_work(curr_bb, new_exit, WN_kid2(node));
            curr_bb = new_exit;

            break;
        }
        case OPR_DO_LOOP: {
            // Add the init statement to the current BB.
            add_wn_to_curr_bb(WN_kid1(node));
            // Creates a new BB for the check stmt.
            push_new_bb();
            // The new BB has only the comparison.
            add_wn_to_curr_bb(WN_kid2(node));
            // Create a new BB for the update stmt.
            hc_bb *update_bb = new_hc_bb();
            append_stmt_to_bb(update_bb, WN_kid(node,3));
            chain_bbs(update_bb, curr_bb);
            // Add work node for the loop body.
            hc_bb *body_curr_bb = new_hc_bb();
            chain_bbs(curr_bb, body_curr_bb);
            new_cfg_work(body_curr_bb, update_bb, WN_kid(node,4));
            // Create the loop exit.
            push_new_bb();

            break;
        }
        case OPR_WHILE_DO: {
            // Create a new BB for the check stmt.
            push_new_bb();
            add_wn_to_curr_bb(WN_kid0(node));
            // Create a new BB for the virtual loop end.
            hc_bb *loop_end_bb = new_hc_bb();
            chain_bbs(loop_end_bb, curr_bb);
            // Add work node for the loop body.
            hc_bb *body_curr_bb = new_hc_bb();
            chain_bbs(curr_bb, body_curr_bb);
            new_cfg_work(body_curr_bb, loop_end_bb, WN_kid1(node));
            // Create the loop exit.
            push_new_bb();

            break;
        }
        case OPR_DO_WHILE: {
            // Create a new BB for the loop beginning.
            push_new_bb();
            // Create a new BB for the check stmt at the loop end.
            hc_bb *loop_end_bb = new_hc_bb();
            append_stmt_to_bb(loop_end_bb, WN_kid0(node));
            chain_bbs(loop_end_bb, curr_bb);
            // Add work node for the loop body.
            new_cfg_work(curr_bb, loop_end_bb, WN_kid1(node));
            // Create the loop exit.
            curr_bb = new_hc_bb();
            chain_bbs(loop_end_bb, curr_bb);

            break;
        }
        case OPR_IF: {
            // Add the check stmt to the current BB.
            add_wn_to_curr_bb(WN_kid0(node));
            // Create an exit for the IF structure.
            hc_bb *exit_bb = new_hc_bb();
            // Add a work node for the if sub-block.
            hc_bb *if_bb = new_hc_bb();
            chain_bbs(curr_bb, if_bb);
            new_cfg_work(if_bb, exit_bb, WN_kid1(node));
            // Add a work node for the else sub-block.
            hc_bb *else_bb = new_hc_bb();
            chain_bbs(curr_bb, else_bb);
            new_cfg_work(else_bb, exit_bb, WN_kid2(node));
            // The exit node should be the current node.
            curr_bb = exit_bb;

            break;
        }
        case OPR_GOTO: {
            // Add this stmt to the current BB.
            // add_wn_to_curr_bb(node);
            // Register the current BB in the label table.
            add_goto_bb(WN_label_number(node), curr_bb);
            // Set curr_bb to be NULL to indicate that there is no link
            // between the next BB and the current BB.
            curr_bb = NULL;

            break;
        }
        case OPR_FALSEBR:
        case OPR_TRUEBR: {
            // Add the check stmt to the current BB.
            add_wn_to_curr_bb(WN_kid0(node));
            // Register the current BB in the label table.
            add_goto_bb(WN_label_number(node), curr_bb);
            // Create a new BB.
            push_new_bb();

            break;
        }
        case OPR_RETURN_VAL:
            // Add the return value expression to the current BB.
            add_wn_to_curr_bb(WN_kid0(node));
            // fall through
        case OPR_RETURN: {
            // Link the current BB with the region's exit BB.
            chain_bbs(curr_bb, region_exit);
            // End the current BB.
            curr_bb = NULL;

            break;
        }
        case OPR_LABEL: {
            if (curr_bb == NULL) {
                // Create a new BB (we cannot use push_new_bb!).
                curr_bb = new_hc_bb();
            } else {
                // Link with the previous BB.
                push_new_bb();
            }
            // Register it in the label table. It must be referenced
            // somewhere.
            add_label_bb(WN_label_number(node), curr_bb);

            break;
        }
        default: {
            if (!OPERATOR_is_not_executable(opr)) {
                add_wn_to_curr_bb(node);
            }
            break;
        }
    }
}

hc_bblist*
build_region_cfg(const WN *region) {
    assert(WN_operator(region) == OPR_REGION);

    // Create entry and exit BB.
    hc_bb *region_entry = new_hc_bb();
    assert(region_exit == NULL);
    region_exit = new_hc_bb();

    assert(curr_bb == NULL);
    curr_bb = new_hc_bb();
    chain_bbs(region_entry, curr_bb);

    assert(work_queue == NULL);
    new_cfg_work(curr_bb, region_exit, WN_kid2(region));

    while (work_queue != NULL) {
        // Get work from the front of the queue.
        cfg_work_node *work = work_queue;
        work_queue = work->next;
        work->next = NULL;

        curr_bb = work->curr;

        // It must be a BLOCK.
        WN *blk = work->blk;
        assert(WN_operator(blk) == OPR_BLOCK); 

        WN *node = WN_first(blk);
        while (node != NULL) {
            handle_blk_node(node);
            node = WN_next(node);
        }

        // Link the current BB with the exit BB.
        chain_bbs(curr_bb, work->exit);
        curr_bb = NULL;

        // Deallocate the work node.
        free(work);
    }

    // Link GOTO and LABEL basic blocks.
    link_gotos_and_labels();

    // Create an hc_bblist.
    hc_bblist *result = (hc_bblist*)malloc(sizeof(hc_bblist));

    result->head = get_bblist_head(&(result->nblocks));
    result->entry = region_entry;
    result->exit = region_exit;
    result->pre_dfs = result->post_dfs = NULL;

    reset_bblist();

    // IMPORTANT: reset state
    region_exit = NULL;

    return result;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void
free_hc_bblist(hc_bblist *hbl) {
    cleanup_all_hc_bbs(hbl->head);

    if (hbl->pre_dfs != NULL) free(hbl->pre_dfs);
    if (hbl->post_dfs != NULL) free(hbl->post_dfs);
}

/*** DAVID CODE END ***/
