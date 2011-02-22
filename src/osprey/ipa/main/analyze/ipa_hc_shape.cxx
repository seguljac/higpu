/** DAVID CODE BEGIN **/

#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "wn.h"
#include "wn_util.h"
#include "wn_simp.h"

#include "hc_utils.h"

#include "ipa_cg.h"
#include "ipa_hc_shape.h"
#include "ipa_hc_shape_prop.h"

#include "ipo_defs.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

HC_ARRAY_SHAPE_INFO* HC_ARRAY_SHAPE_INFO::create_shape(
        ST_IDX arr_st_idx, MEM_POOL *pool)
{
    TY_IDX arr_ty_idx = ST_type(arr_st_idx);
    Is_True(TY_kind(arr_ty_idx) == KIND_ARRAY, (""));

    // Determine the dimensionality and element type.
    UINT ndims = 0;
    TY_IDX elem_ty_idx = analyze_arr_ty(arr_ty_idx, &ndims);

    UINT dim_sz[ndims];
    BOOL is_dim_sz_const[ndims];

    // Go through each dimension.
    // This code mimics <fill_arr_dims> in <ptr_promotion.cxx>.
    UINT dimen = 0;
    TY_IDX ty_idx = arr_ty_idx;
    while (TY_kind(ty_idx) == KIND_ARRAY)
    {
        ARB_HANDLE ah = ARB_HANDLE(Ty_Table[ty_idx].Arb());

        // The stride must be constant. For now, we ignore what it is.
        if (!ARB_const_stride(ah)) return NULL;

        // The lower bound must be constant.
        if (!ARB_const_lbnd(ah)) return NULL;
        INT lbnd = ARB_lbnd_val(ah);

        if (ARB_const_ubnd(ah))
        {
            // [0, const ubnd]
            if (lbnd != 0) return NULL;
            dim_sz[dimen] = ARB_ubnd_val(ah) + 1;
            is_dim_sz_const[dimen] = TRUE;
        }
        else
        {
            // [1, ST_IDX ubnd]
            if (lbnd != 1) return NULL;
            dim_sz[dimen] = ARB_ubnd_var(ah);
            is_dim_sz_const[dimen] = FALSE;
        }

        // Move to the inner type.
        ++dimen;
        ty_idx = TY_etype(ty_idx);
    }

    Is_True(dimen == ndims, (""));

    // Construct a HC_ARRAY_SHAPE_INFO.
    return CXX_NEW(
            HC_ARRAY_SHAPE_INFO(arr_st_idx, elem_ty_idx,
                ndims, dim_sz, is_dim_sz_const, pool), pool);
}

HC_ARRAY_SHAPE_INFO::HC_ARRAY_SHAPE_INFO(ST_IDX st_idx, TY_IDX elem_ty_idx,
        UINT ndims, UINT *dim_sz, BOOL *is_dim_sz_const, MEM_POOL *pool)
{
    _pool = pool;

    _st_idx = st_idx;
    _elem_ty_idx = elem_ty_idx;

    _ndims = ndims;
    _dim_sz = CXX_NEW_ARRAY(UINT, ndims, pool);
    _is_dim_sz_const = CXX_NEW_ARRAY(BOOL, ndims, pool);
    for (UINT i = 0; i < ndims; ++i) {
        _dim_sz[i] = dim_sz[i];
        _is_dim_sz_const[i] = is_dim_sz_const[i];
    }
}

HC_ARRAY_SHAPE_INFO::HC_ARRAY_SHAPE_INFO(WN *shape_wn, MEM_POOL *pool)
{
    Is_True(pool != NULL, (""));
    _pool = pool;

    // Sanity check on the ARRSECTION node.
    Is_True(shape_wn != NULL
            && WN_operator(shape_wn) == OPR_ARRSECTION, (""));

    // Get the host variable symbol.
    WN *base_wn = WN_array_base(shape_wn);
    Is_True(base_wn != NULL && WN_operator(base_wn) == OPR_LDID
            && WN_offset(base_wn) == 0, (""));
    _st_idx = WN_st_idx(base_wn);

    // Get the element type.
    TY_IDX ty_idx = ST_type(_st_idx);
    Is_True(TY_kind(ty_idx) == KIND_POINTER, (""));
    _elem_ty_idx = TY_pointed(ty_idx);
    // Somehow WN_element_size gives 0.
    // Is_True(TY_size(_elem_ty_idx) == WN_element_size(shape_wn), (""));

    // Get each dimension size.
    _ndims = WN_num_dim(shape_wn);
    _dim_sz = CXX_NEW_ARRAY(UINT, _ndims, pool);
    _is_dim_sz_const = CXX_NEW_ARRAY(BOOL, _ndims, pool);
    for (UINT i = 0; i < _ndims; ++i)
    {
        WN *dim_sz_wn = WN_array_dim(shape_wn,i);

        // It must be a constant or an integer symbol.
        OPERATOR opr = WN_operator(dim_sz_wn);
        Is_True(opr == OPR_INTCONST || opr == OPR_LDID, (""));
        if (opr == OPR_INTCONST)
        {
            INT tmp = WN_const_val(dim_sz_wn);
            Is_True(tmp > 0, (""));
            _dim_sz[i] = (UINT)tmp;
            _is_dim_sz_const[i] = TRUE;
        }
        else
        {
            Is_True(WN_offset(dim_sz_wn) == 0, (""));
            _dim_sz[i] = (UINT)WN_st_idx(dim_sz_wn);
            _is_dim_sz_const[i] = FALSE;
        }
    }
}

HC_ARRAY_SHAPE_INFO::HC_ARRAY_SHAPE_INFO(const HC_ARRAY_SHAPE_INFO *orig,
        MEM_POOL *pool)
{
    Is_True(pool != NULL, (""));
    _pool = pool;

    _st_idx = orig->_st_idx;
    _elem_ty_idx = orig->_elem_ty_idx;
    
    _ndims = orig->_ndims;
    _dim_sz = CXX_NEW_ARRAY(UINT, _ndims, pool);
    _is_dim_sz_const = CXX_NEW_ARRAY(BOOL, _ndims, pool);

    for (UINT i = 0; i < _ndims; ++i) {
        _dim_sz[i] = orig->_dim_sz[i];
        _is_dim_sz_const[i] = orig->_is_dim_sz_const[i];
    }
}

BOOL HC_ARRAY_SHAPE_INFO::replace_syms(ST_IDX st_idx,
        ST_IDX *from_syms, ST_IDX *to_syms, UINT n_syms)
{
    BOOL fully_replaced = TRUE;

    _st_idx = st_idx;

    // Go through each dimension.
    for (UINT i = 0; i < _ndims; ++i)
    {
        if (_is_dim_sz_const[i]) continue;

        // Search through the symbol map.
        UINT j;
        for (j = 0; j < n_syms; ++j) {
            if (_dim_sz[i] == from_syms[j]) break;
        }

        if (j < n_syms) {
            _dim_sz[i] = to_syms[j];
        } else if (! HCST_is_global_symbol(_dim_sz[i])) {
            // This non-global symbol cannot be replaced.
            fully_replaced = FALSE;
        }
    }

    return fully_replaced;
}

TY_IDX HC_ARRAY_SHAPE_INFO::create_dyn_array_ptr_type()
{
    // Determine the total array size, if each dimension has a constant size.
    UINT type_sz = TY_size(_elem_ty_idx);
    for (UINT i = 0; i < _ndims; ++i)
    {
        if (_is_dim_sz_const[i]) {
            type_sz *= _dim_sz[i];
        } else {
            type_sz = 0;
            break;
        }
    }

    // Create the type.
    TY_IDX ty_idx;
    TY &ty = New_TY(ty_idx);
    TY_Init(ty, type_sz, KIND_ARRAY, MTYPE_M,
            gen_var_str(_st_idx, ".arr-type"));

    // Create array bound info.
    for (UINT i = 0; i < _ndims; ++i)
    {
        ARB_HANDLE ah = New_ARB();
        if (_is_dim_sz_const[i]) {
            ARB_Init(ah, 0, _dim_sz[i]-1, 1);
        } else {
            // The bound is [1:n]. TODO: is this OK?
            ARB_Init(ah, 1, 0, 1);
            Set_ARB_ubnd_var(ah, _dim_sz[i]);
            Set_ARB_flags(ah, (ARB_flags(ah) & (~ARB_CONST_UBND)));
        }
        Set_ARB_dimension(ah, _ndims-i);

        if (i == 0) {
            Set_TY_arb(ty, ah);
            Set_ARB_flags(ah, ARB_flags(ah) | ARB_FIRST_DIMEN);
        }
        if (i == _ndims-1) Set_ARB_flags(ah, ARB_flags(ah) | ARB_LAST_DIMEN);
    }

    // Set the element type.
    Set_TY_etype(ty, _elem_ty_idx);

    // Create a pointer to this array type.
    return Make_Pointer_Type(ty_idx);
}


BOOL HC_ARRAY_SHAPE_INFO::equals(const HC_ARRAY_SHAPE_INFO *other) const
{
    if (this == other) return TRUE;
    if (other == NULL) return FALSE;

    if (_ndims != other->_ndims) return FALSE;

    // The element type must be the same.
    if (_elem_ty_idx != other->_elem_ty_idx) return FALSE;

    for (UINT i = 0; i < _ndims; ++i)
    {
        // Regardless of what _dim_sz[i] represents, the value must be the
        // same for both shapes.
        if (_dim_sz[i] != other->_dim_sz[i]) return FALSE;

        // Here, it is not safe to compare the two values directly.
        if (_is_dim_sz_const[i]) {
            if (! other->_is_dim_sz_const[i]) return FALSE;
        } else {
            if (other->_is_dim_sz_const[i]) return FALSE;
        }
    }

    return TRUE;
}

void HC_ARRAY_SHAPE_INFO::print(FILE *fp)
{
    for (UINT i = 0; i < _ndims; ++i)
    {
        fprintf(fp, "[");
        if (_is_dim_sz_const[i]) {
            fprintf(fp, "%u", _dim_sz[i]);
        } else {
            fprintf(fp, "%s", ST_name(_dim_sz[i]));
        }
        fprintf(fp, "]");
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <class CONTENT_TYPE>
void HC_SHAPE_CONTEXT<CONTENT_TYPE>::push_shape_info(ST_IDX st_idx,
        CONTENT_TYPE *info)
{
    SHAPE_PER_BLOCK_TABLE *t = _table->Top();
    SHAPE_STACK *stack = t->Find(st_idx);
    if (stack == NULL) {
        stack = CXX_NEW(SHAPE_STACK(_pool), _pool);
        t->Enter(st_idx, stack);
    }
    stack->Push(info);
}

template <class CONTENT_TYPE>
CONTENT_TYPE* HC_SHAPE_CONTEXT<CONTENT_TYPE>::find_shape_info(ST_IDX st_idx)
{
    INT depth = _table->Elements();
    for (INT i = 0; i < depth; ++i) {
        SHAPE_STACK *stack = _table->Top_nth(i)->Find(st_idx);
        if (stack != NULL && stack->Elements() != 0) return stack->Top();
    }

    return NULL;
}

// Explicit instantiation appears to be mandatory.
template class HC_SHAPE_CONTEXT<HC_ARRAY_SHAPE_INFO>;
template class HC_SHAPE_CONTEXT<ST>;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * For each SHAPE directive, create a new symbol of pointer-to-array type
 * where the array shape is specified by the directive. Add this symbol to
 * <shape_context> so that it can replace any references to the old symbol
 * within the proper scope (from the SHAPE directive to the end of the current
 * lexical scope). Return an STID node that assigns the old symbol to the new
 * symbol, which is intended to be inserted before the SHAPE directive.
 *
 * NOTE: At this stage, we cannot remove SHAPE directives even if they appear
 * useless once we have captured the array shape in the new symbol. This is
 * because variables used in shape specification could become dead once SHAPE
 * directives are removed. Their assignments could be eliminated in
 * preoptimization before their usage appear in later analyses (e.g. reaching
 * directive analysis).
 *
 ****************************************************************************/

static WN* HC_apply_shape_annot_walker(WN *wn,
        HC_DYN_ARRAY_PTR_CONTEXT *shape_context,
        UINT& shape_dir_id, HC_SHAPE_INFO_LIST *shapes)
{
    if (wn == NULL) return NULL;

    OPERATOR opr = WN_operator(wn);

    BOOL is_shape_dir = (opr == OPR_XPRAGMA
            && (WN_PRAGMA_ID)WN_pragma(wn) == WN_PRAGMA_HC_SHAPE);

    if (is_shape_dir)
    {
        // Get the parsed result.
        HC_ARRAY_SHAPE_INFO *shape = (*shapes)[shape_dir_id++];

        ST_IDX old_st_idx = shape->get_sym();

        // Create a local variable of pointer-to-array type.
        TY_IDX new_ty_idx = shape->create_dyn_array_ptr_type();
        ST_IDX new_st_idx = new_local_var(
                gen_var_str("a_", old_st_idx), new_ty_idx);
        ST *new_st = &St_Table[new_st_idx];

        // Add the pair of variables to the context.
        shape_context->push_shape_info(old_st_idx, new_st);

        // Create a STID node: <new_st_idx> = <old_st_idx>.
        return WN_StidScalar(new_st, WN_LdidScalar(old_st_idx));
    }

    if (OPERATOR_has_sym(opr))
    {
        // Is there a new ST to replace?
        ST *new_st = shape_context->find_shape_info(WN_st_idx(wn));
        if (new_st != NULL) WN_st_idx(wn) = ST_st_idx(new_st);
    }

    // Handle the composite node.
    if (opr == OPR_BLOCK)
    {
        // Push a new shape table.
        shape_context->push_block();

        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))    
        {
            WN *new_wn = HC_apply_shape_annot_walker(kid_wn,
                    shape_context, shape_dir_id, shapes);
            if (new_wn != NULL)
            {
                // Insert the assignment node before the SHAPE directive.
                WN_INSERT_BlockBefore(wn, kid_wn, new_wn);
            }
        }

        // Pop the entire shape table created before.
        shape_context->pop_block();
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            WN *new_wn = HC_apply_shape_annot_walker(WN_kid(wn,i),
                    shape_context, shape_dir_id, shapes);
            Is_True(new_wn == NULL, (""));
        }
    }

    return NULL;
}

/*****************************************************************************
 *
 * Based on the propagated and local shape annotations, redirect uses of these
 * pointers to new pointer types that include shape information.
 *
 * For example, given a SHAPE directive that specifies int* a is of shape [5],
 * 1) a new variable <a_a> of type int (*)[5] is created, 2) all occurrences
 * of <a> (including those in directives) in the scope of this SHAPE directive
 * are replaced with the new variable.
 *
 * For a propagated shape annotation, the formal type is changed.
 *
 * After this call, <node> contains the modified WN tree.
 *
 ****************************************************************************/

void HC_apply_shape_annot(IPA_NODE *node, MEM_POOL *tmp_pool)
{
    // Create a shape stack.
    HC_DYN_ARRAY_PTR_CONTEXT *shape_context = CXX_NEW(
            HC_DYN_ARRAY_PTR_CONTEXT(tmp_pool), tmp_pool);

    // Switch to the node's context.
    IPA_NODE_CONTEXT context(node);

    WN *func_wn = node->Whirl_Tree();
    WN *func_body_wn = WN_func_body(func_wn);

    // Initialize the shape stack with propagated shape annotations, by
    // pushing a fake block.
    shape_context->push_block();

    HC_FORMAL_SHAPE_ARRAY *fsa = NULL;
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));
    IPA_HC_ANNOT *annot = annots->Head();
    if (annot != NULL) {
        fsa = (HC_FORMAL_SHAPE_ARRAY*)annot->get_annot_data();
    }

    // If a formal has a shape, change its type to match the shape.
    WN *stid_blk = WN_CreateBlock();
    if (fsa != NULL)
    {
        UINT n_formals = fsa->num_formals();
        for (UINT i = 0; i < n_formals; ++i)
        {
            HC_ARRAY_SHAPE_INFO *shape = fsa->get_formal_shape(i);
            if (shape == NULL) continue;

            ST_IDX formal_st_idx = shape->get_sym();
            Is_True(formal_st_idx == WN_st_idx(WN_formal(func_wn,i)), (""));

            Set_ST_type(St_Table[formal_st_idx],
                    shape->create_dyn_array_ptr_type());
        }
    }

    // Here, we only process the function body.
    HC_SHAPE_INFO_LIST *shape_list = node->get_shape_info_list();
    UINT shape_dir_id = 0;
    HC_apply_shape_annot_walker(func_body_wn,
            shape_context, shape_dir_id, shape_list);
    Is_True(shape_dir_id == shape_list->Elements(), (""));

    // Pop the procedure-wide block.
    shape_context->pop_block();

    // Now, insert the assignment nodes at the beginning of the function body.
    WN_INSERT_BlockFirst(func_body_wn, stid_blk);

    // Rebuild the parent map (stored in this node). Here, Parent_Map is
    // defined in IPO and contains this node's parent map once the IPA context
    // is built.
    WN_Parentize(func_body_wn, Parent_Map, Current_Map_Tab);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Return TRUE if the given <wn> is a SHAPE directive, and FALSE otherwise.
static BOOL HC_remove_shape_dir_walker(WN *wn)
{
    if (wn == NULL) return FALSE;

    OPERATOR opr = WN_operator(wn);

    BOOL is_shape_dir = (opr == OPR_XPRAGMA
            && (WN_PRAGMA_ID)WN_pragma(wn) == WN_PRAGMA_HC_SHAPE);
    if (is_shape_dir) return TRUE;

    // Handle composite node.
    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            WN *next_wn = WN_next(kid_wn);
            if (HC_remove_shape_dir_walker(kid_wn))
            {
                WN_DELETE_FromBlock(wn, kid_wn);
            }
            kid_wn = next_wn;
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            Is_True(!HC_remove_shape_dir_walker(WN_kid(wn,i)), (""));
        }
    }

    return FALSE;
}

/*****************************************************************************
 *
 * Remove SHAPE directives in the given procedure <node>.
 *
 ****************************************************************************/

void HC_remove_shape_dir(IPA_NODE *node)
{
    IPA_NODE_CONTEXT context(node);

    // We only need to process the function body.
    WN *func_body_wn = WN_func_body(node->Whirl_Tree());

    Is_True(!HC_remove_shape_dir_walker(func_body_wn), (""));

    // Rebuild the parent map (stored in this node). Here, Parent_Map is
    // defined in IPO and contains this node's parent map once the IPA context
    // is built.
    WN_Parentize(func_body_wn, Parent_Map, Current_Map_Tab);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// symbols with demoted types
typedef DYN_ARRAY<ST_IDX> HC_SYM_LIST;

static void HC_demote_dyn_array_type_walker(WN *wn, HC_SYM_LIST *updated_syms)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_LDID || opr == OPR_STID)
    {
        ST_IDX st_idx = WN_st_idx(wn);
        TY_IDX ty_idx = ST_type(st_idx);

        // Has this symbol's type been modified?
        BOOL found = FALSE;
        for (INT i = 0; i < updated_syms->Elements(); ++i)
        {
            if (st_idx == (*updated_syms)[i]) { found = TRUE; break; }
        }

        if (!found && TY_kind(ty_idx) == KIND_POINTER)
        {
            ty_idx = TY_pointed(ty_idx);

            // If this is a local ptr-to-ARRAY, change it to ptr-to-element.
            if (HCTY_is_dyn_array(ty_idx) && HCST_is_local_symbol(st_idx))
            {
                ty_idx = Make_Pointer_Type(arr_elem_ty(ty_idx));
                Set_ST_type(ST_ptr(st_idx), ty_idx);
                updated_syms->AddElement(st_idx);
                found = TRUE;
            }
        }

        if (found) WN_set_ty(wn, ty_idx);
    }

    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            HC_demote_dyn_array_type_walker(kid_wn, updated_syms);
            kid_wn = WN_next(kid_wn);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HC_demote_dyn_array_type_walker(WN_kid(wn,i), updated_syms);
        }
    }
}

void IPA_HC_demote_dyn_array_types(MEM_POOL *tmp_pool)
{
    MEM_POOL_Push(tmp_pool);

    // Keep a list of procedures whose prototype has been updated.
    UINT n_proc_nodes = IPA_Call_Graph->Node_Size();
    BOOL prototype_updated[n_proc_nodes];
    for (UINT i = 0; i < n_proc_nodes; ++i) prototype_updated[i] = FALSE;

    // Modify the type of formals and locals that are ptr-to-ARRAYs.
    IPA_NODE_ITER cg_iter1(IPA_Call_Graph, PREORDER);
    for (cg_iter1.First(); !cg_iter1.Is_Empty(); cg_iter1.Next())
    {
        IPA_NODE *node = cg_iter1.Current();
        if (node == NULL) continue;

        // Switch to the node's context.
        IPA_NODE_CONTEXT context(node);
        WN *func_wn = node->Whirl_Tree();

        // Keep a list of symbols whose type has been updated, so that the
        // type of the associated WN node can be updated too.
        HC_SYM_LIST *updated_syms = CXX_NEW(HC_SYM_LIST(tmp_pool), tmp_pool);

        // Go through the list of formals.
        UINT n_formals = WN_num_formals(func_wn);
        TY_IDX formal_types[n_formals];
        for (UINT i = 0; i < n_formals; ++i)
        {
            ST_IDX formal_st_idx = WN_st_idx(WN_formal(func_wn,i));
            TY_IDX formal_ty_idx = ST_type(formal_st_idx);

            if (TY_kind(formal_ty_idx) == KIND_POINTER
                    && TY_kind(TY_pointed(formal_ty_idx)) == KIND_ARRAY)
            {
                // This is a ptr-to-ARRAY formal. 
                // Change its type to ptr-to-element.
                TY_IDX elem_ty_idx = arr_elem_ty(TY_pointed(formal_ty_idx));
                formal_ty_idx = Make_Pointer_Type(elem_ty_idx);
                Set_ST_type(ST_ptr(formal_st_idx), formal_ty_idx);

                updated_syms->AddElement(formal_st_idx);
            }

            formal_types[i] = formal_ty_idx;
        }

        if (updated_syms->Elements() > 0)
        {
            // Construct a new function prototype.
            ST_IDX func_st_idx = WN_entry_name(func_wn);
            TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
            TY_IDX new_func_ty_idx = new_func_type(TY_name(func_ty_idx),
                    Tylist_Table[TY_tylist(func_ty_idx)],   // return type
                    n_formals, formal_types);
            Set_PU_prototype(Pu_Table[ST_pu(St_Table[func_st_idx])],
                    new_func_ty_idx);

            prototype_updated[node->Array_Index()] = TRUE;
        }

        // Update WN tree and demote local ptr-to-ARRAY variables.
        HC_demote_dyn_array_type_walker(node->Whirl_Tree(), updated_syms);

        // The Parent_Map and the (IPA_EDGE,WN) map are still valid.
    }

    // Update the parameters of calls to procedures whose prototype has been
    // modified.
    IPA_NODE_ITER cg_iter2(IPA_Call_Graph, PREORDER);
    for (cg_iter2.First(); !cg_iter2.Is_Empty(); cg_iter2.Next())
    {
        IPA_NODE *node = cg_iter2.Current();
        if (node == NULL) continue;

        // Switch to the node's context.
        IPA_NODE_CONTEXT context(node);
        WN *func_wn = node->Whirl_Tree();

        // Link IPA_EDGEs with WN nodes.
        IPA_Call_Graph->Map_Callsites(node);

        // Go through the outgoing edges.
        IPA_SUCC_ITER succ_iter(IPA_Call_Graph, node);
        for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
        {
            IPA_EDGE *e = succ_iter.Current_Edge();
            if (e == NULL) continue;
            
            // Does the edge lead to a node with modified prototype?
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            if (! prototype_updated[callee->Array_Index()]) continue;

            // Get the number of parameters.
            WN *call_wn = e->Whirl_Node();
            Is_True(call_wn != NULL, (""));
            UINT n_params = WN_kid_count(call_wn);

            // Get the list of callee's formal types.
            // We could have fetched them from the prototype.
            TY_IDX formal_types[n_params];
            {
                IPA_NODE_CONTEXT callee_context(callee);    // IMPORTANT!

                WN *callee_func_wn = callee->Whirl_Tree();
                Is_True(n_params == WN_num_formals(callee_func_wn), (""));

                for (UINT i = 0; i < n_params; ++i)
                {
                    formal_types[i] =
                        ST_type(WN_st(WN_formal(callee_func_wn,i)));
                }
            }

            // Update the type of each actual.
            for (UINT i = 0; i < n_params; ++i)
            {
                WN *param_wn = WN_actual(call_wn,i);
                WN_set_ty(param_wn, formal_types[i]);
                WN_set_rtype(param_wn, TY_mtype(formal_types[i]));
            }
        }

        // The Parent_Map and the (IPA_EDGE,WN) map are still valid.
    }

    MEM_POOL_Pop(tmp_pool);
}

/*** DAVID CODE END ***/
