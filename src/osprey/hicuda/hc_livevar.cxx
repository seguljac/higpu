/** DAVID CODE BEGIN **/

#include <assert.h>

#include "cmp_symtab.h"
#include "hc_dfa.h"
#include "hc_livevar.h"


/**
 * Simply add the given symbol to the KILL set.
 */
inline static void
update_kill(dfa_info *lvinfo, ST_IDX st_idx) {
    int idx = cmp_st_idx(st_idx) - (CMP_ST_IDX_ZERO+1);
    assert(idx >= 0);

    // printf("ADDED %s to the KILL set!\n", ST_name(st_idx));

    set_bit(lvinfo->kill, idx, true);
}

/**
 * Add the given symbol to the GEN set if it has not been defined before,
 * i.e. it is not in the current KILL set.
 */
inline static void
update_gen(dfa_info *lvinfo, ST_IDX st_idx) {
    int idx = cmp_st_idx(st_idx) - (CMP_ST_IDX_ZERO+1);
    assert(idx >= 0);

    if (!get_bit(lvinfo->kill, idx)) set_bit(lvinfo->gen, idx, true);
}

static void update_lv_genkill_set(dfa_info *lvinfo, WN *stmt);

/**
 * Add all symbols referenced in 'expr' to the GEN set.
 */
static void
update_gen(dfa_info *lvinfo, WN *expr) {
    OPERATOR opr = WN_operator(expr);

    assert(OPERATOR_is_expression(opr));

    /* For most expression, we just check if it has a symbol or pass
     * the task to its kids. COMMA and RCOMMA are special: they contain
     * BLOCKs. */

    if (opr == OPR_COMMA) {
        // execute kid0 block first.
        WN *node = WN_first(WN_kid0(expr));
        while (node != NULL) {
            update_lv_genkill_set(lvinfo, node);
            node = WN_next(node);
        }
        // evaluate kid1.
        update_gen(lvinfo, WN_kid1(expr));
    } else if (opr == OPR_RCOMMA) {
        // evaluate kid0 first.
        update_gen(lvinfo, WN_kid0(expr));
        // execute kid1 block.
        WN *node = WN_first(WN_kid1(expr));
        while (node != NULL) {
            update_lv_genkill_set(lvinfo, node);
            node = WN_next(node);
        }
    } else {
        if (OPERATOR_has_sym(opr)) {
            ST_IDX st_idx = WN_st_idx(expr);
            if (st_idx != ST_IDX_ZERO) update_gen(lvinfo, st_idx);
        }

        int nkids = WN_kid_count(expr);
        for (int i = 0; i < nkids; ++i) update_gen(lvinfo, WN_kid(expr,i));
    }
}

/**
 * Update the GEN and KILL set in 'lvinfo' based on 'stmt', assuming that
 * we are going through statements in the forward direction.
 */
static void
update_lv_genkill_set(dfa_info *lvinfo, WN *stmt) {
    OPERATOR opr = WN_operator(stmt);

    // It must be a statement node that is not structured control flow.
    assert(OPERATOR_is_stmt(opr) && !OPERATOR_is_scf(opr));

    switch (opr) {
        case OPR_FORWARD_BARRIER:
        case OPR_BACKWARD_BARRIER:
        case OPR_ASM_STMT:
        case OPR_OPT_CHI:
        case OPR_OPT_RESERVE2:
        case OPR_DEALLOCA:
            fprintf(stderr, "LIVEVAR: unsupported stmt %s\n",
                OPERATOR_name(opr));
            break;

        case OPR_TRAP:
        case OPR_AFFIRM:
        case OPR_PRAGMA:
        case OPR_XPRAGMA:
        case OPR_EXC_SCOPE_BEGIN:
        case OPR_EXC_SCOPE_END:
        case OPR_COMMENT: {
            // do nothing
            break;
        }
        case OPR_EVAL:
        case OPR_ASSERT: {
            // use kid0
            update_gen(lvinfo, WN_kid0(stmt));

            break;
        }
        case OPR_CALL: {
            // use the procedure symbol
            update_gen(lvinfo, WN_st_idx(stmt));
            // fall through
        }
        case OPR_ICALL:
        case OPR_VFCALL:
        case OPR_INTRINSIC_CALL:    // TODO: intrinsic field
        case OPR_IO: {
            // use all kids
            int nkids = WN_kid_count(stmt);
            for (int i = 0; i < nkids; ++i) update_gen(lvinfo, WN_kid(stmt,i));
            // We do not care about the dedicated preg for return value.
            // Check the spec for COMMA.
            
            break;
        }
        case OPR_ISTORE:
        case OPR_MSTORE:
        case OPR_ISTBITS: {
            /* let's be conservative here because we don't know where the
             * value is written to. */

            // use all the kids
            int nkids = WN_kid_count(stmt);
            for (int i = 0; i < nkids; ++i) update_gen(lvinfo, WN_kid(stmt,i));

            break;
        }
        case OPR_STID:
        case OPR_STBITS: {
            // use kid0
            update_gen(lvinfo, WN_kid0(stmt));
            // define st_idx
            update_kill(lvinfo, WN_st_idx(stmt));

            break;
        }

        default:
            fprintf(stderr, "LIVEVAR: not possible!\n");
            abort();
    }
}

void
compute_lv_genkill_sets(hc_bblist *list, int n_syms) {
    /* Here, we will find GEN and KILL set even for unreachable BBs.
     * However, we do skip entry and exit BBs. */

    hc_bb *bb = list->head;
    while (bb != NULL) {
        if (bb != list->entry && bb != list->exit) {
            // Allocate and init the GEN and KILL sets.
            dfa_info *di = &(bb->dfa[DFA_LIVE_VAR]);
            di->gen = new_bit_vector(n_syms);
            set_all_bits(di->gen, false);
            di->kill = new_bit_vector(n_syms);
            set_all_bits(di->kill, false);

            // Go through each stmt in the WN block in the forward order
            // to update the GEN/KILL set.
            WN *stmt = WN_first(bb->body_blk);
            while (stmt != NULL) {
                update_lv_genkill_set(di, stmt);
                stmt = WN_next(stmt);
            }
        }

        bb = bb->next;
    }
}

/*** DAVID CODE END ***/
