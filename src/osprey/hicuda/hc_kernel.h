/** DAVID CODE BEGIN **/

#ifndef _HC_KERNEL_H_
#define _HC_KERNEL_H_

#include "defs.h"
#include "wn.h"

class HC_LOOP_PART_INFO;
class HC_LOCAL_VAR_STORE;

/*****************************************************************************
 *
 * Information of a DO_LOOP needed to handle a LOOP_PARTITION directive
 *
 ****************************************************************************/

class HC_DOLOOP_INFO
{
private:

    ST_IDX _idxv_st_idx;    // loop index variable symbol

    WN *_init_wn;
    WN *_end_wn;    // inclusive
    INT _step;      // must be a constant for now
                    // <op> is LE if <_step> is +ive, and GE otherwise

    UINT _empty_loop;   // 0: NO, 1: YES, 2: NOT SURE

    BOOL is_ldid_scalar(WN *wn, ST_IDX var_st_idx)
    {
        return (WN_operator(wn) == OPR_LDID
                && WN_offset(wn) == 0 && WN_st_idx(wn) == var_st_idx);
    }

    /**
     * Reverse the comparison operator.
     */
    OPERATOR reverse_comp(OPERATOR opr)
    {
        switch (opr)
        {
            case OPR_GE: return OPR_LE;
            case OPR_GT: return OPR_LT;
            case OPR_LE: return OPR_GE;
            case OPR_LT: return OPR_GT;
            case OPR_EQ:
            case OPR_NE: return opr;
            default:     return OPERATOR_UNKNOWN;
        }
    }

public:

    /**
     * Fill the instance and make sure that the end-condition check and the
     * sign of the loop step are consistent.
     */
    HC_DOLOOP_INFO(const WN *doloop_wn);

    ~HC_DOLOOP_INFO();

    // return the original instances
    WN* get_init_expr() const { return _init_wn; }
    WN* get_end_expr() const { return _end_wn; }
    INT get_step() const { return _step; }
    ST_IDX get_idx_var() const { return _idxv_st_idx; }

    UINT is_empty_loop() const { return _empty_loop; }

    /**
     * If we know for sure whether or not this loop is empty, return the
     * expression for the tripcount.
     */
    WN* get_tripcount_expr() const;

    /**
     * Generate a BLOCK node that contains code that computes the tripcount of
     * this loop, stored in the given variable symbol (INT type).
     */
    WN* gen_tripcount(ST_IDX st_idx) const;
};


/*****************************************************************************
 *
 * This function modifies the DO_LOOP in the region body.
 *
 * Return TRUE if the loop is not eliminated and FALSE otherwise.
 *
 ****************************************************************************/

extern BOOL HC_lower_loop_part_region(WN *region, HC_LOOP_PART_INFO *lpi,
        HC_LOCAL_VAR_STORE *lvar_store, MEM_POOL *pool);

extern WN* HC_lower_barrier(WN *pramga_wn, WN *parent_wn, BOOL gen_code);

#endif  // _HC_KERNEL_H_

/*** DAVID CODE END ***/
