/** DAVID CODE BEGIN **/

#include "wn.h"
#include "wn_simp.h"
#include "wn_util.h"

#include "cxx_memory.h"
#include "cxx_base.h"

#include "hc_expr.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * For now, all classes here assume integer expression.
 *
 ****************************************************************************/

class HC_TERM : public SLIST_NODE
{
private:

    INT _coeff;
    WN *_var_term_wn;   // could be any non-const expression

#define HCT_TO_BE_REMOVED   0x02

    UINT _flags;

    DECLARE_SLIST_NODE_CLASS(HC_TERM);

public:

    HC_TERM(INT coeff, WN *var_term_wn)
    {
        Is_True(coeff != 0, (""));
        _coeff = coeff;
        Is_True(var_term_wn != NULL, (""));
        _var_term_wn = var_term_wn;

        _flags = 0;
    }
    ~HC_TERM() {}

    void incr_coeff(INT incr) { _coeff += incr; }

    INT get_coeff() const { return _coeff; }
    WN* get_var_term() const { return _var_term_wn; }

    BOOL to_be_removed() const { return _flags & HCT_TO_BE_REMOVED; }
    void set_to_be_removed() { _flags |= HCT_TO_BE_REMOVED; }
    void reset_to_be_removed() { _flags &= (~HCT_TO_BE_REMOVED); }

    // The given variable term is left intact.
    BOOL same_var_term(WN *var_term_wn)
    {
        return (WN_Simp_Compare_Trees(_var_term_wn, var_term_wn) == 0);
    }
};

class HC_TERM_LIST : public SLIST
{
private:

    HC_TERM_LIST(const HC_TERM_LIST&);
    HC_TERM_LIST& operator = (const HC_TERM_LIST&);

    DECLARE_SLIST_CLASS(HC_TERM_LIST, HC_TERM);

public:

    ~HC_TERM_LIST() {}
};

class HC_EXPR
{
private:

    BOOL _too_messy;

    INT _const_term;
    HC_TERM_LIST *_var_terms;

    MEM_POOL *_pool;

    void add_term(WN *var_term_wn, INT coeff);
    void add_term_walker(WN *wn, INT coeff);

public:

    HC_EXPR(WN *expr_wn, MEM_POOL *pool);

    BOOL is_too_messy() const { return _too_messy; }

    UINT num_var_terms() const { return _var_terms->Len(); }

    // Determine the largest +ive constant that divides this expression.
    UINT is_a_multiple_of() const;

    void add_const(INT c) { _const_term += c; }

    // If the given variable is linear in this expression, remove it from the
    // expression and return the coefficient through <coeff_ptr>. Otherwise,
    // return FALSE and leave the expression and <coeff_ptr> intact.
    //
    BOOL extract_linear_term(ST_IDX st_idx, WN_OFFSET ofst, INT *coeff);

    // Remove a multiple of the given sub-expression from this expression,
    // when the resulting expression does not share any non-const terms with
    // the sub-expression. Otherwise, return FALSE and leave this epxression
    // intact.
    //
    BOOL extract_linear_term(const HC_EXPR *sub_expr, INT *coeff);

    // NULL if too messy.
    WN* to_wn() const;
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void HC_EXPR::add_term(WN *var_term_wn, INT coeff)
{
    // Does this term exist in the list?
    HC_TERM *term = _var_terms->Head();
    for ( ; term != NULL; term = term->Next())
    {
        if (term->same_var_term(var_term_wn)) break;
    }

    if (term == NULL)
    {
        term = CXX_NEW(HC_TERM(coeff, WN_COPY_Tree(var_term_wn)), _pool);
        _var_terms->Append(term);
    }
    else
    {
        term->incr_coeff(coeff);
    }
}

// follows ACCESS_VECTOR::Add_Sum
void HC_EXPR::add_term_walker(WN *wn, INT coeff)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);
    Is_True(OPERATOR_is_expression(opr), (""));

    if (opr == OPR_ADD)
    {
        add_term_walker(WN_kid0(wn), coeff);
        add_term_walker(WN_kid1(wn), coeff);
    }
    else if (opr == OPR_SUB)
    {
        add_term_walker(WN_kid0(wn), coeff);
        add_term_walker(WN_kid1(wn), -coeff);
    }
    else if (opr == OPR_NEG)
    {
        add_term_walker(WN_kid0(wn), -coeff);
    }
    else if (opr == OPR_MPY)
    {
        WN *lopnd_wn = WN_kid0(wn), *ropnd_wn = WN_kid1(wn);
        if (WN_operator(lopnd_wn) == OPR_INTCONST)
        {
            add_term_walker(ropnd_wn, coeff*WN_const_val(lopnd_wn));
        }
        else if (WN_operator(ropnd_wn) == OPR_INTCONST)
        {
            add_term_walker(lopnd_wn, coeff*WN_const_val(ropnd_wn));
        }
        else
        {
            // This is a variable term we cannot currently split.
            add_term(wn, coeff);
        }
    }
    else if (opr == OPR_LDID)
    {
        add_term(wn, coeff);
    }
    else if (opr == OPR_INTCONST)
    {
        _const_term += (coeff * WN_const_val(wn));
    }
    else if (opr == OPR_PAREN)
    {
        add_term_walker(WN_kid0(wn), coeff);
    }
    else if (opr == OPR_CVT)
    {
        if (WN_opcode(wn) == OPC_I8I4CVT || WN_opcode(wn) == OPC_U8I4CVT)
        {
            add_term_walker(WN_kid0(wn), coeff);
        }
        else if (WN_opcode(wn) == OPC_I4U8CVT
                && WN_opcode(WN_kid0(wn)) == OPC_U8CVTL
                && WN_cvtl_bits(WN_kid0(wn)) == 32)
        {
            add_term_walker(WN_kid0(WN_kid0(wn)), coeff);
        }
        else
        {
            _too_messy = TRUE;
        }
    }
    else
    {
        _too_messy = TRUE;
    }
}

HC_EXPR::HC_EXPR(WN *expr_wn, MEM_POOL *pool)
{
    Is_True(pool != NULL, (""));
    _pool = pool;

    _too_messy = FALSE;
    _const_term = 0;

    _var_terms = CXX_NEW(HC_TERM_LIST(), pool);
    add_term_walker(expr_wn, 1);
}

static INT HC_compute_gcd(INT x, INT y)
{
    Is_True(x > 0 && y > 0, (""));

    if (x == y) return x;

    if (x < y)
    {
        // Swap x and y.
        INT tmp = x; x = y; y = tmp;
    }

    // Here, x >= y.
    do
    {
        INT r = x % y;
        x = y;
        y = r;
    } while (y != 0);

    return x;
}

UINT HC_EXPR::is_a_multiple_of() const
{
    INT factor = (_const_term >= 0) ? _const_term : -_const_term;
    if (factor == 1) return 1;

    // Go through the variable terms.
    for (HC_TERM *t = _var_terms->Head(); t != NULL; t = t->Next())
    {
        INT coeff = t->get_coeff();
        if (coeff < 0) coeff = -coeff;

        factor = (factor == 0) ? coeff : HC_compute_gcd(factor, coeff);
        if (factor == 1) return 1;
    }

    return (UINT)factor;
}

BOOL HC_EXPR::extract_linear_term(ST_IDX st_idx, WN_OFFSET ofst,
        INT *coeff_ptr)
{
    INT coeff = 0;

    // There should be at most one term with a LDID node of the given
    // variable.
    HC_TERM *curr_term = _var_terms->Head(), *prev_term = NULL;
    while (curr_term != NULL)
    {
        WN *var_term = curr_term->get_var_term();
        if (WN_operator(var_term) == OPR_LDID
                && WN_st_idx(var_term) == st_idx
                && WN_offset(var_term) == ofst)
        {
            coeff = curr_term->get_coeff();
            // Remove this node.
            HC_TERM *tmp_term = curr_term;
            curr_term = curr_term->Next();
            if (prev_term == NULL)
            {
                _var_terms->Set_Head(curr_term);
            }
            else
            {
                prev_term->Set_Next(curr_term);
            }
            CXX_DELETE(tmp_term, _pool);
        }
        else
        {
            // Does the variable appear in this non-linear term?
            if (HCWN_ref_var(var_term, st_idx, ofst)) return FALSE;
            prev_term = curr_term;
            curr_term = curr_term->Next();
        }
    }

    if (coeff_ptr != NULL) *coeff_ptr = coeff;
    return TRUE;
}

BOOL HC_EXPR::extract_linear_term(const HC_EXPR *sub_expr, INT *coeff_ptr)
{
    BOOL coeff_inited = FALSE;
    INT coeff = 0;

    // Go through each non-const term in the sub-expression.
    for (HC_TERM *curr_sub_term = sub_expr->_var_terms->Head();
            curr_sub_term != NULL; curr_sub_term = curr_sub_term->Next())
    {
        WN *sub_term_wn = curr_sub_term->get_var_term();
        INT sub_term_coeff = curr_sub_term->get_coeff();

        // Search for this term in this expression.
        // There should be at most one such term.
        HC_TERM *curr_term = _var_terms->Head(), *prev_term = NULL;
        while (curr_term != NULL
                && WN_Simp_Compare_Trees(sub_term_wn,
                    curr_term->get_var_term()) != 0)
        {
            prev_term = curr_term;
            curr_term = curr_term->Next();
        }

        INT l_coeff = 0;
        if (curr_term != NULL)
        {
            // The coefficient must divide.
            INT term_coeff = curr_term->get_coeff();
            l_coeff = term_coeff / sub_term_coeff;
            if (term_coeff != l_coeff * sub_term_coeff) return FALSE;
        }

        // The coefficients for all sub-terms must be the same.
        if (!coeff_inited)
        {
            coeff = l_coeff;
            coeff_inited = TRUE;
        }
        else if (coeff != l_coeff)
        {
            return FALSE;
        }

        // We cannot delete this term now because we have to look at other
        // sub-terms first. Mark it to be removed.
        if (curr_term != NULL) curr_term->set_to_be_removed();
    }

    if (coeff_ptr != NULL) *coeff_ptr = coeff;

    // Nothing more to do if the sub-expression does not exist in this one.
    if (coeff == 0) return TRUE;

    // The extraction process is successful, so remove the cached terms from
    // this expression.
    HC_TERM *curr_term = _var_terms->Head(), *prev_term = NULL;
    while (curr_term != NULL)
    {
        HC_TERM *next_term = curr_term->Next();

        if (curr_term->to_be_removed())
        {
            if (prev_term == NULL)
            {
                _var_terms->Set_Head(next_term);
            }
            else
            {
                prev_term->Set_Next(next_term);
            }
            CXX_DELETE(curr_term, _pool);
        }
        else
        {
            prev_term = curr_term;
        }

        curr_term = next_term;
    }

    // Update the constant term.
    _const_term -= (coeff * sub_expr->_const_term);

    return TRUE;
}


WN* HC_EXPR::to_wn() const
{
    if (_too_messy) return NULL;

    WN *expr_wn = WN_Intconst(Integer_type, _const_term);
    for (const HC_TERM *term = _var_terms->Head();
            term != NULL; term = term->Next())
    {
        INT coeff = term->get_coeff();
        if (coeff == 0) continue;

        if (coeff > 0)
        {
            expr_wn = WN_Add(Integer_type, expr_wn,
                    WN_Mpy(Integer_type,
                        WN_Intconst(Integer_type, coeff),
                        WN_COPY_Tree(term->get_var_term())));
        }
        else
        {
            expr_wn = WN_Sub(Integer_type, expr_wn,
                    WN_Mpy(Integer_type,
                        WN_Intconst(Integer_type, -coeff),
                        WN_COPY_Tree(term->get_var_term())));
        }
    }

    return expr_wn;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static MEM_POOL hc_simp_pool;
static BOOL hc_simp_pool_initialized = FALSE;

static void HCWN_init_simp_pool()
{
    if (!hc_simp_pool_initialized)
    {
        MEM_POOL_Initialize(&hc_simp_pool,
                "hiCUDA WN simplifier pool", FALSE);
        hc_simp_pool_initialized = TRUE;
    }
}

void HCWN_delete_simp_pool()
{
    if (hc_simp_pool_initialized)
    {
        MEM_POOL_Delete(&hc_simp_pool);
        hc_simp_pool_initialized = FALSE;
    }
}

WN* HCWN_simplify_expr(WN *expr_wn)
{
    WN *simp_expr_wn = NULL;

    HCWN_init_simp_pool();
    MEM_POOL *pool = &hc_simp_pool;
    MEM_POOL_Push(pool);
    {
        HC_EXPR expr(expr_wn, pool);
        simp_expr_wn = expr.to_wn();
    }
    MEM_POOL_Pop(pool);

    return simp_expr_wn;
}

// This is just a convenience method for the above.
//
void HCWN_simplify_expr(WN **expr_wn_p)
{
    Is_True(expr_wn_p != NULL, (""));

    WN *expr_wn = *expr_wn_p;
    WN *simp_expr_wn = HCWN_simplify_expr(expr_wn);
    if (simp_expr_wn != NULL)
    {
        WN_DELETE_Tree(expr_wn);
        *expr_wn_p = simp_expr_wn;
    }
}

BOOL HCWN_ref_var(WN *expr_wn, ST_IDX st_idx, WN_OFFSET ofst)
{
    if (expr_wn == NULL) return FALSE;

    OPERATOR opr = WN_operator(expr_wn);
    Is_True(OPERATOR_is_expression(opr), (""));

    if (opr == OPR_LDID
            && WN_st_idx(expr_wn) == st_idx && WN_offset(expr_wn) == ofst)
    {
        return TRUE;
    }

    INT nkids = WN_kid_count(expr_wn);
    for (INT i = 0; i < nkids; ++i)
    {
        if (HCWN_ref_var(WN_kid(expr_wn,i), st_idx, ofst)) return TRUE;
    }

    return FALSE;
}

static WN* HCWN_expr_bound(WN *expr_wn, BOOL is_min,
        ST_IDX st_idx, WN_OFFSET ofst, WN *var_lbnd_wn, WN *var_ubnd_wn)
{
    WN *expr_bnd_wn = NULL;

    HCWN_init_simp_pool();
    MEM_POOL *pool = &hc_simp_pool;
    MEM_POOL_Push(pool);
    {
        HC_EXPR expr(expr_wn, pool);

        if (!expr.is_too_messy())
        {
            INT coeff = 0;
            if (expr.extract_linear_term(st_idx, ofst, &coeff))
            {
                if (coeff == 0)
                {
                    expr_bnd_wn = WN_COPY_Tree(expr_wn);
                }
                else
                {
                    WN *bnd_wn = ((coeff > 0 && is_min)
                            || (coeff < 0 && !is_min)) ?
                        var_lbnd_wn : var_ubnd_wn;
                    expr_bnd_wn = WN_Add(Integer_type,
                            WN_Mpy(Integer_type,
                                WN_Intconst(Integer_type, coeff),
                                WN_COPY_Tree(bnd_wn)),
                            expr.to_wn());
                }
            }
        }
    }
    MEM_POOL_Pop(pool);

    return expr_bnd_wn;
}

WN* HCWN_expr_min(WN *expr_wn,
        ST_IDX st_idx, WN_OFFSET ofst, WN *var_lbnd_wn, WN *var_ubnd_wn)
{
    return HCWN_expr_bound(expr_wn, TRUE,
            st_idx, ofst, var_lbnd_wn, var_ubnd_wn);
}

WN* HCWN_expr_max(WN *expr_wn,
        ST_IDX st_idx, WN_OFFSET ofst, WN *var_lbnd_wn, WN *var_ubnd_wn)
{
    return HCWN_expr_bound(expr_wn, FALSE,
            st_idx, ofst, var_lbnd_wn, var_ubnd_wn);
}

UINT HCWN_is_a_multiple_of(WN *expr_wn, HC_EXPR_PROP_LIST *sub_expr_props)
{
    UINT factor = 1;

    HCWN_init_simp_pool();
    MEM_POOL *pool = &hc_simp_pool;
    MEM_POOL_Push(pool);
    {
        HC_EXPR expr(expr_wn, pool);
        if (!expr.is_too_messy())
        {
            factor = expr.is_a_multiple_of();

            if (sub_expr_props != NULL)
            {
                UINT n_sub_exprs = sub_expr_props->Elements();
                for (UINT i = 0; i < n_sub_exprs; ++i)
                {
                    HC_EXPR sub_expr((*sub_expr_props)[i]->get_expr(), pool);
                    if (!sub_expr.is_too_messy())
                    {
                        INT coeff = 0;
                        if (expr.extract_linear_term(&sub_expr, &coeff))
                        {
                            if (coeff != 0) expr.add_const(
                                    coeff * (*sub_expr_props)[i]->get_factor());
                            factor = expr.is_a_multiple_of();
                        }
                    }
                }
            }
        }
    }
    MEM_POOL_Pop(pool);

    return factor;
}

#if 0

void
HCAS_IntDiv(WN *dividend, UINT divisor, const VAR_BOUND_MAP *vbounds,
        WN **quotient_ptr, WN **remainder_ptr) {
    assert(dividend != NULL && MTYPE_is_integral(dividend));
    assert(divisor > 0);
    assert(quotient != NULL);
    assert(remainder != NULL);

    OPERATOR opr = WN_operator(dividend);
    assert(OPERATOR_is_expression(opr));

    WN *quotient = NULL, *remainder = NULL;

    switch (opr) {
        case OPR_ADD: {
            WN *q1, *r1, *q2, *r2;

            HCWN_IntDiv(dividend, divisor, vbounds, q1, r1);
            HCWN_IntDiv(dividend, divisor, vbounds, q2, r2);

            // Sum the quotients.
            q1 = WN_Add(Integer_type, q1, q2);

            // Sum the remainders.
            r1 = WN_Add(Integer_type, r1, r2);

            // Divide the remainder again.
            HCWN_IntDiv(r1, divisor, vbounds, q2, remainder);

            // Sum the quotients again.
            quotient = WN_Add(Integer_type, q1, q2);
        }
        break;

        case OPR_SUB: {
            WN *q1, *r1, *q2, *r2;

            HCWN_IntDiv(dividend, divisor, vbounds, q1, r1);
            HCWN_IntDiv(dividend, divisor, vbounds, q2, r2);

            // Diff the quotients.
            q1 = WN_Sub(Integer_type, q1, q2);

            // Diff the remainders.
            r1 = WN_Sub(Integer_type, r1, r2);
        }
        break;

        case OPR_MPY: {
        }
        break;

        case OPR_REM: {
        }
        break;

        case OPR_INTCONST: {
            INT dd = WN_const_val(dividend);
            quotient = WN_Intconst(Integer_type, dd / divisor);
            remainder = WN_Intconst(Integer_type, dd % divisor);
        }
        break;
 
        case OPR_LDID: {
            VAR_BOUND_MAP::iterator it = vbounds->begin();
            while (it != vbounds->end()
                && !HCWN_are_ldids_equal(it.first, dividend)) it++;
            if (it != vbound->end()) {
                INT32 lbnd = (INT32)(it.second >> 32);
                INT32 ubnd = (INT32)(it.second & 0xFFFFFFFF);

                if (lbnd >= 0 && ubnd < divisor) {
                    quotient = WN_Intconst(Integer_type, 0);
                    remainder = WN_COPY_Tree(dividend);
                    break;
                }
            }
        }
        // fall through
       
        default: {
            // Here, we cannot do anything smart.
            quotient = WN_Div(Integer_type,
                WN_COPY_Tree(dividend), WN_Intconst(Integer_type, divisor));
            remainder = WN_Binary(OPR_REM, Integer_type,
                WN_COPY_Tree(dividend), WN_Intconst(Integer_type, divisor));
        }
        break;
    }
}

#endif

/*** DAVID CODE END ***/
