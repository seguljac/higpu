/** DAVID CODE BEGIN **/

#ifndef _HICUDA_HC_ALG_SIMP_H_
#define _HICUDA_HC_ALG_SIMP_H_

/*****************************************************************************
 *
 * Routines related to analyzing and simplifying expressions.
 *
 ****************************************************************************/

#include "defs.h"
#include "wn.h"
#include "cxx_template.h"

extern void HCWN_delete_simp_pool();

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Represent properties of an integer expression
 *
 * This is safe across procedure clones.
 *
 ****************************************************************************/

class HC_EXPR_PROP
{
private:

    WN *_wn;
    
    UINT _factor;   // <_wn> is a multiple of <factor>

    void init(WN *wn, UINT factor)
    {
        Is_True(wn != NULL && OPERATOR_is_expression(WN_operator(wn)), (""));
        Is_True(factor != 0, (""));

        _wn = wn;
        _factor = factor;
    }

public:

    // HC_EXPR_PROP() { _wn = NULL; _factor = 1; }
    HC_EXPR_PROP(WN *wn, UINT factor) { init(wn, factor); }
    ~HC_EXPR_PROP() {}

    WN* get_expr() const { return _wn; }
    UINT get_factor() const { return _factor; }
};

typedef DYN_ARRAY<HC_EXPR_PROP*> HC_EXPR_PROP_LIST;

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

// Simplifies the given WN expression <expr_wn>.
//
// It leaves <expr_wn> intact, and returns a new expression if simplified or
// NULL if <expr_wn> is too messy.
//
extern WN* HCWN_simplify_expr(WN *expr_wn);

// Simplifies the given WN expression *<expr_wn_p>.
//
// The original <expr_wn> is consumed by this function and cannot be used
// again. The simplified version replaces the original <expr_wn>.
//
extern void HCWN_simplify_expr(WN **expr_wn_p);

/*****************************************************************************
 *
 * Return TRUE if <expr_wn> references the given variable <st_idx> at offset
 * <ofst>, and FALSE otherwise.
 *
 ****************************************************************************/

extern BOOL HCWN_ref_var(WN *expr_wn, ST_IDX st_idx, WN_OFFSET ofst);

/*****************************************************************************
 *
 * Determine the minimum and maximum of <expr_wn>, given that the range of
 * variable, represented by <st_idx> at <ofst>, is between <var_lbnd_wn> and
 * <var_ubnd_wn>.
 *
 * All given WN nodes are left intact.
 *
 * Return NULL if <expr_wn> is too messy or has non-linear terms that contains
 * the given variable.
 *
 ****************************************************************************/

extern WN* HCWN_expr_min(WN *expr_wn,
        ST_IDX st_idx, WN_OFFSET ofst, WN *var_lbnd_wn, WN *var_ubnd_wn);

extern WN* HCWN_expr_max(WN *expr_wn,
        ST_IDX st_idx, WN_OFFSET ofst, WN *var_lbnd_wn, WN *var_ubnd_wn);

/*****************************************************************************
 *
 * Given a list of sub-expression properties <sub_expr_props> (with count
 * <n_sub_exprs>), determine the largest +ive integer that always divides the
 * given expression <expr_wn>.
 *
 * <sub_expr_props> is allowed to be NULL.
 *
 ****************************************************************************/

extern UINT HCWN_is_a_multiple_of(WN *expr_wn,
        HC_EXPR_PROP_LIST *sub_expr_props);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Give a relative measure of the complexity of the given expression,
 * based on the GPU architecture.
 *
 * The returned value is higher if the epxression is expected to take
 * longer to run.
 */
extern UINT expr_complexity(WN *expr);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

#include <map>

/* a map between LDID nodes and two 32-bit integers representing the lower
 * and upper bounds
 */
typedef std::map<WN*, UINT64> VAR_BOUND_MAP;
 
/**
 * Smart integer division: dividend / divisor
 *
 * 'vbounds' is optional and provides extra information about lower and
 * upper bound of variables, that may help the division.
 *
 * *quotient and *remainder will hold the quotient and remained respectively.
 */
extern void HCAS_IntDiv(WN *dividend, UINT divisor,
    const VAR_BOUND_MAP *vbounds, WN **quotient, WN **remainder);

#endif  // _HICUDA_HC_ALG_SIMP_H_

/*** DAVID CODE END ***/
