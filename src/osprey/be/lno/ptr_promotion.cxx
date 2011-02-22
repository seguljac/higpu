/** DAVID CODE BEGIN **/

#include "defs.h"
#include "glob.h"
#include "cxx_template.h"
#include "cxx_memory.h"
#include "cxx_hash.h"

#include "tracing.h"        // for TDEBUG_HICUDA

#include "wn.h"
#include "wn_util.h"
#include "wn_simp.h"
#include "ir_reader.h"

#include "access_vector.h"
#include "lwn_util.h"
#include "cond.h"
#include "opt_du.h"

#include "ptr_promotion.h"


/* TODO: make it extern */
static MEM_POOL hc_local_pool;
static BOOL initialized_hc_local_pool = FALSE;


/*****************************************************************************
 *
 * Since the pointer promotion interface is external, and assumes no
 * initialization of LNO. We need to do some preliminary initialization before
 * using functions in LNO.
 *
 * These two functions are extracted from Lnoptimizer in lnopt_main.cxx.
 *
 ****************************************************************************/

static VINDEX16 save_graph_capacity;

static void init_lno_for_cond_info(WN *func_wn,
        DU_MANAGER *du_mgr, ALIAS_MANAGER *alias_mgr)
{
    Du_Mgr = du_mgr;
    Alias_Mgr = alias_mgr;

    // LWN_Delete_DU is defined in lwn_util.cxx.
    WN_Register_Delete_Cleanup_Function(LWN_Delete_DU);

    save_graph_capacity = GRAPH16_CAPACITY;
    GRAPH16_CAPACITY = LNO_Graph_Capacity;

    // Current_Func_Node = func_wn;

    /* Init and push the mempools. */
    MEM_POOL_Initialize(&LNO_default_pool, "LNO_default_pool", FALSE);
    MEM_POOL_Initialize(&LNO_local_pool, "LNO_local_pool", FALSE);
    MEM_POOL_Push(&LNO_local_pool);
    MEM_POOL_Push_Freeze(&LNO_default_pool);

    /* Set up the maps. */
    Parent_Map = WN_MAP_Create(&LNO_default_pool);
    WN_SimpParentMap = Parent_Map;  // Let the simplifier know about it
    FmtAssert(Parent_Map != -1,
            ("Ran out of mappings in init_lno_for_cond_info"));

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "init_lno_for_cond_info: Parent_Map = %d, &Parent_Map = %p\n",
                Parent_Map, &Parent_Map);
    }

    LNO_Info_Map = WN_MAP_Create(&LNO_default_pool);
    FmtAssert(LNO_Info_Map != -1,
            ("Ran out of mappings in init_lno_for_cond_info"));

    /* Walk through the function tree and fill the Parent_Map. */
    LWN_Parentize(func_wn);

    // We do not want to enable pointer promotion in <Mark_Code> because a
    // more powerful promotion pass is implemented here.
    Mark_Code(func_wn, FALSE, TRUE);

    // Fills the access arrays for each DO loop's bounds.
    LNO_Build_Access(func_wn, &LNO_default_pool);
}

static void fini_lno_for_cond_info()
{
    /* Free the maps. */
    WN_MAP_Delete(Parent_Map);
    WN_SimpParentMap = WN_MAP_UNDEFINED;

    WN_MAP_Delete(LNO_Info_Map);

    /* Pop and free the mempools. We have to free them because
     * lno_mempool_initialized is not set in lnopt_main.cxx and the mempools
     * may be allocated twice.
     */
    MEM_POOL_Pop(&LNO_local_pool);
    MEM_POOL_Pop_Unfreeze(&LNO_default_pool);
    MEM_POOL_Delete(&LNO_local_pool);
    MEM_POOL_Delete(&LNO_default_pool);

    GRAPH16_CAPACITY = save_graph_capacity;

    WN_Remove_Delete_Cleanup_Function(LWN_Delete_DU);
    Du_Mgr = NULL;    
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * <arr_st_idx> must be ST_IDX_ZERO before calling this recursive function.
 *
 * If the base symbol is found, <arr_st_idx> stores the symbol. This routine
 * also determines the leftover offset, which is passed out in two ways. If
 * <wn> should be replaced entirely to represent this offset, then the offset
 * is returned. Otherwise, NULL is returned and <wn> holds the offset.
 *
 * If the base symbol is not found, <arr_st_idx> stays ST_IDX_ZERO. NULL is
 * returned and <wn> is left intact.
 *
 * <addr_wn> holds the root WN for the expression, that is constant over the
 * recursive. It is used when walking through the parent chain of a child node
 * in the expression tree.
 *
 * NOTE: we use recursion here because it is easier to construct the offset.
 *
 ****************************************************************************/

static WN* HC_extract_arr_base_walker(WN *wn, WN *addr_wn, ST_IDX& arr_st_idx)
{
    if (wn == NULL) return NULL;
    if (arr_st_idx != ST_IDX_ZERO) return NULL;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_LDID || opr == OPR_LDA)
    {
        // The symbol must be an array or a pointer-to-array.
        ST_IDX st_idx = WN_st_idx(wn);
        Is_True(st_idx != ST_IDX_ZERO, (""));
        TY_IDX ty_idx = ST_type(st_idx);
        if (opr == OPR_LDID)
        {
            // pointer-to-array
            if (TY_kind(ty_idx) != KIND_POINTER
                    || TY_kind(TY_pointed(ty_idx)) != KIND_ARRAY) return NULL;
        }
        else
        {
            // array
            if (TY_kind(ty_idx) != KIND_ARRAY) return NULL;
        }

        // Make sure that multiplicative factor of this load is 1, by checking
        // if all parents in the middle of the chain up to the root node
        // <addr_wn> are ADD, SUB or CVT.
        if (wn != addr_wn)
        {
            WN *pwn = LWN_Get_Parent(wn);
            for ( ; pwn != addr_wn; pwn = LWN_Get_Parent(pwn))
            {
                OPERATOR popr = WN_operator(pwn);
                // TODO: we need a more strict check for SUB because it may
                // lead to a multiplicative factor of -1.
                if (popr != OPR_ADD && popr != OPR_SUB
                        && popr != OPR_CVT) return NULL;
            }
        }

        arr_st_idx = st_idx;

        // Replace the LDID/LDA node with an INTCONST node with the offset.
        return WN_Intconst(Integer_type, WN_offset(wn));
    }

    if (opr == OPR_BLOCK)
    {
        // Go through the enclosing stmts.
        for (WN *stmt_wn = WN_first(wn); stmt_wn != NULL;
                stmt_wn = WN_next(stmt_wn))
        {
            Is_True(HC_extract_arr_base_walker(stmt_wn,
                        addr_wn, arr_st_idx) == NULL, (""));
            if (arr_st_idx != ST_IDX_ZERO) break;
        }
    }
    else
    {
        // Go through the kids.
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            WN *kid_wn = WN_kid(wn,i);
            WN *new_kid_wn = HC_extract_arr_base_walker(kid_wn,
                    addr_wn, arr_st_idx);
            if (arr_st_idx != ST_IDX_ZERO)
            {
                if (new_kid_wn != NULL)
                {
                    WN_DELETE_Tree(kid_wn);
                    WN_kid(wn,i) = new_kid_wn;
                }
                break;
            }
        }
    }

    return NULL;
}

/*****************************************************************************
 *
 * Extracts the base array symbol from an address expression <addr_wn>. The
 * multiplier for this symbol's address in <wn> is guaranteed to be 1.
 *
 * If the base symbol is extracted successfully, it is stored in <arr_st_idx>,
 * and a fresh WN node is returned to represent the offset.
 *
 * Otherwise, <arr_st_idx> is ST_IDX_ZERO and NULL is returned.
 *
 ****************************************************************************/

WN* HC_extract_arr_base(WN *addr_wn, ST_IDX& arr_st_idx)
{
    // Create a copy of the address for later modification.
    WN *addr_copy_wn = WN_COPY_Tree(addr_wn);
    // map used later
    // Can we use LWN_Parentize?
    WN_Parentize(addr_copy_wn, Parent_Map, Current_Map_Tab);
    arr_st_idx = ST_IDX_ZERO;

    WN *ofst_wn = HC_extract_arr_base_walker(addr_copy_wn, addr_copy_wn,
            arr_st_idx);
    if (arr_st_idx == ST_IDX_ZERO)
    {
        WN_DELETE_Tree(addr_copy_wn);
        Is_True(ofst_wn == NULL, (""));
    }
    else if (ofst_wn != NULL)
    {
        WN_DELETE_Tree(addr_copy_wn);
    }
    else
    {
        // Simplify the offset.
        ofst_wn = WN_Simplify_Tree(addr_copy_wn);
    }

    return ofst_wn;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef STACK<WN*> DOLOOP_STACK;

/*****************************************************************************
 * 
 * Return TRUE if <wn> references symbol <st_idx> and FALSE otherwise.
 * TODO: this should be in <hc_utils.cxx>.
 *
 *****************************************************************************/

static BOOL HCWN_contains_sym(WN *wn, ST_IDX st_idx)
{
    if (wn == NULL) return FALSE;

    OPERATOR opr = WN_operator(wn);
    if (OPERATOR_has_sym(opr))
    {
        if (st_idx == WN_st_idx(wn)) return TRUE;
    }

    // Handle composite node.
    if (opr == OPR_BLOCK)
    {
        for (WN *stmt_wn = WN_first(wn); stmt_wn != NULL;
                stmt_wn = WN_next(stmt_wn))
        {
            if (HCWN_contains_sym(stmt_wn, st_idx)) return TRUE;
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            if (HCWN_contains_sym(WN_kid(wn,i), st_idx)) return TRUE;
        }
    }

    return FALSE;
}


/*****************************************************************************
 *
 * Expand the given (address) expression by copy propagation. Note that the
 * definition of a variable is recursively expanded.
 *
 * The variables we expand upon must be a local (not formal) variable that
 * 1) is not a loop variable (nor mentioned in loop bounds),
 * 2) has a single definition, and
 * 3) is not aliased
 *
 * The newly constructed WN node does not share any data structure with the
 * original node. Its Parent_Map is not constructed, but its Def-Use chains
 * are replicated from the original node.
 *
 ****************************************************************************/

static WN* HC_expand_expr(WN *wn, DOLOOP_STACK *stack)
{
    if (wn == NULL) return NULL;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_LDID)
    {
        // Is the symbol to be loaded a local scalar?
        ST_IDX st_idx = WN_st_idx(wn);
        Is_True(st_idx != ST_IDX_ZERO, (""));
        ST& st = St_Table[st_idx];
        if (TY_kind(ST_type(st)) == KIND_SCALAR
                && ST_sclass(st) == SCLASS_AUTO)
        {
            // Is this symbol a loop variable or mentioned in loop bounds?
            INT i;
            for (i = 0; i < stack->Elements(); ++i)
            {
                WN *loop_wn = stack->Bottom_nth(i);

                if (st_idx == WN_st_idx(WN_index(loop_wn))) break;
                if (HCWN_contains_sym(WN_start(loop_wn), st_idx)) break;
                if (HCWN_contains_sym(WN_end(loop_wn), st_idx)) break;
            }
            if (i == stack->Elements())
            {
                // This symbol is not a loop variable.
                // TODO: Is this symbol aliased?
                DEF_LIST *defs = Du_Mgr->Ud_Get_Def(wn);
                if (defs != NULL)
                {
                    DU_NODE *def = defs->Head();
                    if (def != NULL && def->Next() == NULL)
                    {
                        // This symbol has a single definition.
                        WN *def_wn = def->Wn();
                        if (def_wn != NULL && WN_operator(def_wn) == OPR_STID)
                        {
                            return HC_expand_expr(WN_kid0(def_wn), stack);
                        }
                    }
                }
            }
        }
    }

    // Copy the current node (not recursively).
    WN *new_wn = WN_CopyNode(wn);
    // IMPORTANT! Migrate the Def and Use info (shallow copy).
    DEF_LIST *defs = Du_Mgr->Ud_Get_Def(wn);
    Du_Mgr->Ud_Put_Def(new_wn, defs);
    USE_LIST *uses = Du_Mgr->Du_Get_Use(wn);
    Du_Mgr->Du_Put_Use(new_wn, uses);

    if (opr == OPR_BLOCK)
    {
        // Go through the enclosing stmts.
        for (WN *stmt_wn = WN_first(wn); stmt_wn != NULL;
                stmt_wn = WN_next(stmt_wn))
        {
            WN_INSERT_BlockLast(new_wn, HC_expand_expr(stmt_wn, stack));
        }
    }
    else
    {
        // Go through the kids.
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            WN_kid(new_wn, i) = HC_expand_expr(WN_kid(wn,i), stack);
        }
    }

    return new_wn;
}


/*****************************************************************************
 *
 * Recursively clear the Def-Use info in the given WN tree <wn>, that must be
 * constructed from HC_expand_expr. We must do this because the DEF_LIST's and
 * USE_LIST's for these nodes are shallow copies. Otherwise, calling
 * WN_DELETE_Tree on <wn> will crash.
 *
 ****************************************************************************/

static void HC_clear_du_info(WN *wn)
{
    Du_Mgr->Ud_Put_Def(wn, NULL);
    Du_Mgr->Du_Put_Use(wn, NULL);

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_BLOCK)
    {
        // Go through the enclosing stmts.
        for (WN *stmt_wn = WN_first(wn); stmt_wn != NULL;
                stmt_wn = WN_next(stmt_wn)) HC_clear_du_info(stmt_wn);
    }
    else
    {
        // Go through the kids.
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) HC_clear_du_info(WN_kid(wn,i));
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Analyze the given array type:
 * 1) Find the array element type (being returned).
 * 2) Find the dimensionality (stored in <ndims>).
 *
 ****************************************************************************/

static TY_IDX analyze_arr_ty(TY_IDX arr_ty_idx, UINT *ndims)
{
    UINT16 dimen = 0;
    TY_IDX ty_idx = arr_ty_idx;

    while (TY_kind(ty_idx) == KIND_ARRAY)
    {
        ARB_IDX arb_idx = Ty_Table[ty_idx].Arb();
        // Accumulate the dimensionality.
        dimen += ARB_dimension(ARB_HANDLE(arb_idx));
        // Move to the inner type.
        ty_idx = TY_etype(ty_idx);
    }

    if (ndims != NULL) *ndims = dimen;    
    return ty_idx;
}


/*****************************************************************************
 *
 * Parse each dimension of the given array type, and store the result in
 * <dim_sz> and <is_dim_sz_const>.
 *
 * Each element of <dim_sz> could be treated as a constant or an ST_IDX,
 * depending on the corresponding flag in <is_dim_sz_const>.
 *
 * A supported dimension is:
 * 1) unit stride, and
 * 2) [0,const ubnd] or [1, ST_IDX ubnd].
 *
 * Return TRUE if all dimensions meet the conditions and FALSE otherwise.
 *
 ****************************************************************************/

static BOOL fill_arr_dims(TY_IDX arr_ty_idx, UINT ndims,
        UINT *dim_sz, BOOL *is_dim_sz_const)
{
    UINT16 dimen = 0;
    TY_IDX ty_idx = arr_ty_idx;

    while (TY_kind(ty_idx) == KIND_ARRAY)
    {
        ARB_HANDLE ah_base = ARB_HANDLE(Ty_Table[ty_idx].Arb());

        UINT ldims = ARB_dimension(ah_base);
        for (UINT i = 0; i < ldims; ++i, ++dimen)
        {
            ARB_HANDLE ah = ah_base[i];

            // The stride must be constant. For now, we ignore what it is.
            if (!ARB_const_stride(ah)) return FALSE;

            // The lower bound must be constant.
            if (!ARB_const_lbnd(ah)) return FALSE;
            INT lbnd = ARB_lbnd_val(ah);

            if (ARB_const_ubnd(ah))
            {
                // [0, const ubnd]
                if (lbnd != 0) return FALSE;
                dim_sz[dimen] = ARB_ubnd_val(ah) + 1;
                is_dim_sz_const[dimen] = TRUE;
            }
            else
            {
                // [1, ST_IDX ubnd]
                if (lbnd != 1) return FALSE;
                dim_sz[dimen] = ARB_ubnd_var(ah);
                is_dim_sz_const[dimen] = FALSE;
            }
        }

        // Move to the inner type.
        ty_idx = TY_etype(ty_idx);
    }

    // Sanity check: just in case buffer overflow.
    Is_True(dimen == ndims, (""));

    return TRUE;
}


/*****************************************************************************
 *
 * Try to turn the address expression "wn + offset" into an ARRAY. Access to
 * this address happens within the given DO_LOOP stack.
 *
 * Return the ARRAY node if successful or NULL otherwise. The returned ARRAY
 * node is guaranteed not to shared any tree node with "wn", which can be
 * safely freed.
 *
 * NOTE: the given offset may not be completely "consumed" by the returned
 * ARRAY node if successful, so the client should include this updated offset
 * in the parent ILOAD/ISTORE node.
 *
 ****************************************************************************/

static WN* normalize_addr(WN *wn, INT& offset,
        WN *func_wn, DOLOOP_STACK *stack)
{
    // First, separate the base pointer address and the offset (in bytes).
    // NOTE: we have not included the passed-in "offset" yet.
    ST_IDX base_st_idx = ST_IDX_ZERO;
    WN *offset_wn = HC_extract_arr_base(wn, base_st_idx);
    if (base_st_idx == ST_IDX_ZERO) return NULL;

    /* Second, construct a 1D access array and delinearize it. The
     * delinearization process goes like this:
     *
     * At each step, avs[i] is split into (new_avs[i-1],new_avs[i]) so that 
     * new_avs[i-1] = avs[i] / arr_dims[i], new_avs[i] = avs[i] % arr_dims[i].
     *
     * We need to pass in the DO_LOOP stack surrounding the access because it
     * provides constraints on the index variables used in the access.
     */

    // Collect array dimension info.
    TY_IDX arr_ty_idx = TY_pointed(ST_type(base_st_idx));
    UINT ndims = 0;
    TY_IDX elem_ty_idx = analyze_arr_ty(arr_ty_idx, &ndims);
    Is_True(ndims > 0, (""));
    INT elem_sz = TY_size(elem_ty_idx);

    UINT dim_sz[ndims];
    BOOL is_dim_sz_const[ndims];
    if (! fill_arr_dims(arr_ty_idx, ndims, dim_sz, is_dim_sz_const))
    {
        return NULL;
    }

    // <offset_wn> is definitely a multiple of the array element size. We must
    // split the given extra offset into two components: a multiple of element
    // size and the remainder.
    BOOL is_ofst_neg = (offset < 0);
    INT ofst_abs = (is_ofst_neg) ? -offset : offset;
    INT ofst_main = ofst_abs / elem_sz;
    if (is_ofst_neg)
    {
        if (ofst_abs > ofst_main * elem_sz) ++ofst_main;
        ofst_main = -ofst_main;
    }
    ofst_main *= elem_sz;
    offset -= ofst_main;
    Is_True(offset >= 0, (""));

    MEM_POOL_Push(&hc_local_pool);

    ACCESS_VECTOR *avs = CXX_NEW_ARRAY(ACCESS_VECTOR, ndims, &hc_local_pool);
    for (UINT i = 0; i < ndims; ++i)
    {
        avs[i].Init(stack->Elements(), &hc_local_pool);
    }

    // The initial offset is put in the last slot.
    // NOTE: here we include the extra offset.
    avs[ndims-1].Set(offset_wn, stack, 1, ofst_main, TRUE);
    if (avs[ndims-1].Too_Messy)
    {
        MEM_POOL_Pop(&hc_local_pool);
        return NULL;
    }

    // Divide the offset by the array element size.
    Is_True(avs[ndims-1].perfect_divide_by_const(TY_size(elem_ty_idx)), (""));

    for (INT i = ndims-1; i > 0; --i)
    {
        ACCESS_VECTOR *q = NULL;
        if (is_dim_sz_const[i])
        {
            q = avs[i].divide_by_const(dim_sz[i], wn);
        }
        else
        {
            ST_IDX st_idx = (ST_IDX)dim_sz[i];

            // Check if it is a loop index variable.
            UINT iv_idx, n_ivs = stack->Elements();
            for (iv_idx = 0; iv_idx < n_ivs; ++iv_idx)
            {
                WN *doloop = stack->Bottom_nth(iv_idx);
                if (st_idx == WN_st_idx(WN_index(doloop))) break;
            }

            if (iv_idx < n_ivs)
            {
                q = avs[i].divide_by_iv(iv_idx, wn);
            }
            else
            {
                // Construct a SYMBOL and do division by symbol.
                // There are two ways to construct a SYMBOL depending on
                // whether or not this variable is a formal. Let's try both.

                // Is this a formal?
                UINT f, n_formals = WN_num_formals(func_wn);
                for (f = 0; f < n_formals; ++f)
                {
                    if (st_idx == WN_st_idx(WN_formal(func_wn,f))) break;
                }
                if (f < n_formals)
                {
                    // Try the formal SYMBOL approach.
                    SYMBOL sym(f, 0, TY_mtype(ST_type(st_idx)));
                    q = avs[i].divide_by_sym(sym, wn);
                }

                // Try the general approach.
                if (q == NULL)
                {
                    SYMBOL sym(ST_ptr(st_idx), 0, TY_mtype(ST_type(st_idx)));
                    q = avs[i].divide_by_sym(sym, wn);
                }
            }
        }

        if (q == NULL)
        {
            MEM_POOL_Pop(&hc_local_pool);
            return NULL;
        }
        // We could just do a shallow copy.
        avs[i-1].Init(q, &hc_local_pool);
    }

    /* Third, construct an ARRAY node out of the array of ACCESS_VECTORs. */

    WN *access_wn = WN_Create(OPR_ARRAY, Pointer_type, MTYPE_V, ndims*2+1);
    WN_element_size(access_wn) = TY_size(elem_ty_idx);
    WN_array_base(access_wn) = WN_LdidScalar(base_st_idx);
    for (UINT i = 0; i < ndims; ++i)
    {
        WN_array_dim(access_wn,i) = is_dim_sz_const[i] ?
            WN_Intconst(Integer_type, dim_sz[i]) : WN_LdidScalar(dim_sz[i]);
        WN_array_index(access_wn,i) = avs[i].to_wn(stack);
    }

    MEM_POOL_Pop(&hc_local_pool);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "CONSTRUCTED AN ARRAY ACCESS:\n");
        fdump_tree(TFile, access_wn);
        fprintf(TFile, "FROM:\n");
        fdump_tree(TFile, wn);
    }

    return access_wn;
}


/*****************************************************************************
 *
 * Recursively walk through <wn> and transform pointer access to ARRAYs.
 * A DO_LOOP stack is built along the way to facilitate access normalization.
 *
 ****************************************************************************/

static void transform_ptr_access_to_array_walker(WN *wn, WN *func_wn,
        DOLOOP_STACK *dl_stack)
{
    if (wn == NULL) return;

    Is_True(dl_stack != NULL, (""));

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_DO_LOOP) dl_stack->Push(wn);

    // Walk through the kids first.
    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            transform_ptr_access_to_array_walker(kid_wn, func_wn, dl_stack);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            transform_ptr_access_to_array_walker(WN_kid(wn,i),
                    func_wn, dl_stack);
        }
    }

    if (opr == OPR_ILOAD || opr == OPR_ISTORE)
    {
        INT addr_kid_idx = (opr == OPR_ILOAD) ? 0 : 1;
        WN *addr_wn = WN_kid(wn, addr_kid_idx);

        if (WN_operator(addr_wn) != OPR_ARRAY)
        {
            // Before address normalization, expand the expression as much as
            // possible using copy propagation.
            WN *exp_addr_wn = HC_expand_expr(addr_wn, dl_stack);
            // Temporarily replace the address with the expanded version.
            WN_kid(wn, addr_kid_idx) = exp_addr_wn;
            // Update Parent_Map.
            LWN_Parentize(exp_addr_wn);
            // NOTE: the Parent_Map info for <addr_wn> is still there.
            WN_MAP_Set(Parent_Map, exp_addr_wn, wn);

            // Normalize the expanded address.
            INT ofst = WN_offset(wn);
            WN *new_addr_wn = normalize_addr(exp_addr_wn,
                    ofst, func_wn, dl_stack);
            if (new_addr_wn != NULL)
            {
                // Replace <addr_wn> with <new_addr_wn>.
                WN_kid(wn, addr_kid_idx) = new_addr_wn;
                WN_offset(wn) = ofst;
                // Update Parent_Map.
                WN_MAP_Set(Parent_Map, new_addr_wn, wn);

                // This will automatically update Parent_Map if a proper
                // "cleanup" function is registered through
                // WN_Register_Delete_Cleanup_Function.
                WN_DELETE_Tree(addr_wn);
            }
            else
            {
                // Restore the original array access.
                WN_kid(wn, addr_kid_idx) = addr_wn;
                // No need to update Parent_Map.
            }

            // Clean up.
            HC_clear_du_info(exp_addr_wn);
            WN_DELETE_Tree(exp_addr_wn);
        }
    }

    if (opr == OPR_DO_LOOP) dl_stack->Push(wn);
}

/*****************************************************************************
 *
 * Transform accesses like a[1] where <a> is of type int(*)[5], into ARRAYs.
 * This allows:
 * 1) array access redirection to GPU variables, and
 * 2) more accurate IP array section analysis later on.
 *
 * <func_wn> is a FUNC_ENTRY node.
 *
 ****************************************************************************/

void transform_ptr_access_to_array(WN *func_wn,
        DU_MANAGER *du_mgr, ALIAS_MANAGER *alias_mgr)
{
    Is_True(func_wn != NULL && WN_operator(func_wn) == OPR_FUNC_ENTRY,
            ("transform_ptr_access_to_array: invalid <func_wn>!\n"));

    // Initialize the LNO environment.
    init_lno_for_cond_info(func_wn, du_mgr, alias_mgr);

    if (!initialized_hc_local_pool)
    {
        MEM_POOL_Initialize(&hc_local_pool, "hiCUDA local pool", FALSE);
        initialized_hc_local_pool = TRUE;
    }

    MEM_POOL_Push(&hc_local_pool);

    // Init a stack of DO_LOOPs (mimic IPL_Build_Access_Vectors).
    DOLOOP_STACK *dl_stack = CXX_NEW(
            DOLOOP_STACK(&hc_local_pool), &hc_local_pool);

    transform_ptr_access_to_array_walker(func_wn, func_wn, dl_stack);

    MEM_POOL_Pop(&hc_local_pool);

    fini_lno_for_cond_info();
}

/*** DAVID CODE END ***/
