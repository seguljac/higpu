/** DAVID CODE BEGIN **/

#ifndef _HICUDA_SUBSCRIPT_H_
#define _HICUDA_SUBSCRIPT_H_

// use to hold error messages
extern char hc_subscript_errmsg[256];

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Routines dealing with array subscripts
 */

struct subscript_context {
    bool valid;

    // a list of LDIDs of variables we care about, i.e. we want their
    // coefficients
    WN **var_ldids;
    // list length
    UINT nvars;

    // storage for the coefficients of corresponding variable LDIDs
    // It must be of length 'nvars' + 1; the last element stores
    // the constant term.
    WN **var_coeffs;
};

extern struct subscript_context ssinfo;

/**
 * Reset 'ssinfo' to its initial state.
 * 
 * The client is responsible for deallocating var_ldids and var_coeffs.
 */
extern void reset_subscript_context();

/**
 * Normalize the given subscript expression w.r.t. a list of variables in
 * 'sinfo.var_ldids'.
 *
 * Return true if the expression is a linear combination of variables in
 * 'sinfo.var_ldids', and 'sinfo.var_coeffs' will stores the coefficients.
 * Return false otherwise.
 *
 * The client should first provide necessary info in 'ssinfo' and then
 * call this method.
 *
 * 'ssinfo.var_ldids' does not have to be provided, in which case
 * 'ssinfo.nvars' and 'ssinfo.var_coeffs' should be left as 0 and NULL.
 * All variables accessed in 'expr' will be considered.
 *
 * If the analysis is not successful, 'ssinfo.var_ldids' will still have the
 * variable list; 'ssinfo.var_coeffs' will be an array of NULLs.
 */
extern bool analyze_subscript(WN *expr);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

enum TRISTATE_BOOL {
    TB_TRUE,
    TB_FALSE,
    TB_NOTSURE
};

typedef enum TRISTATE_BOOL TBOOL;

/**
 * The triplet is (start_idx, stride, end_idx).
 *
 * Return an expression (end_idx - start_idx + 1).
 *
 * If start_idx and end_idx are linear expressions, smart algebraic
 * simplification will be performed.
 */
extern WN* idx_range_size(WN *hc_triplet);

/**
 * Return TB_TRUE if ss1 == ss2; TB_FALSE if ss1 != ss2; TB_NOTSURE otherwise.
 */
extern TBOOL are_subscripts_equal(WN *ss1, WN *ss2);

/**
 * Return TB_TRUE if ss1 <= ss2; TB_FALSE if ss1 > ss2; TB_NOTSURE otherwise.
 *
 * The loops are optionally provided to help determining variable signs.
 */
extern TBOOL compare_subscripts(WN *ss1, WN *ss2,
    struct loop_part_info **loops = NULL, UINT nloops = 0);

/**
 * Both 't1' and 't2' are TRIPLET nodes, with positive constant step.
 *
 * Return TB_TRUE if 't1' covers 't2'; TB_FALSE if 't1' does not cover 't2';
 * TB_NOTSURE otherwise.
 *
 * The loops are optionally provided to help determining variable signs.
 */
extern TBOOL is_triplet_covered(WN *t1, WN *t2,
    struct loop_part_info **loops = NULL, UINT nloops = 0);

/**
 * Both 'r1' and 'r2' are ARRSECTION nodes.
 * Return TB_TRUE if 'r1' covers 'r2'; TB_FALSE if 'r1' does not cover 'r2';
 * TB_NOTSURE otherwise.
 *
 * The loops are optionally provided to help determining variable signs.
 */
extern TBOOL is_region_covered(WN *r1, WN *r2,
    struct loop_part_info **loops = NULL, UINT nloops = 0);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Given an array region and a set of loops that we want to project the
 * region onto, return the projected region.
 *
 * For each loop, its index variable is in 'loop_idxvs' and its index range
 * is in 'loop_ranges'.
 */
extern WN* project_region(WN *arr_section,
    ST_IDX *loop_idxvs, WN **loop_ranges, UINT nloops);

#endif  // _HICUDA_SUBSCRIPT_H_

/*** DAVID CODE END ***/
