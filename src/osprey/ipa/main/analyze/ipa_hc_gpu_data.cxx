/** DAVID CODE BEGIN **/

// needed by ipl_summary.h included in ipa_cg.h
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "tracing.h"        // for TDEBUG_HICUDA
#include "wn.h"
#include "wn_util.h"
#include "wn_simp.h"        // WN_Simp_Compare_Trees
#include "ir_reader.h"

#include "ipa_cg.h"
#include "ipa_hc_gpu_data.h"
#include "ipa_hc_gdata_alloc.h"
#include "ipa_hc_gpu_data_prop.h"

#ifdef IPA_HICUDA
#include "hc_utils.h"

#include "ipa_hc_kernel.h"

#include "ipo_defs.h"       // IPA_NODE_CONTEXT
#include "ipo_lwn_util.h"   // LWN_Get_Parent
#endif  // IPA_HICUDA

#include "hc_common.h"
#include "hc_gpu_data.h"
#include "hc_directives.h"
#include "hc_expr.h"
#include "cuda_utils.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * A convenience method to test if a lower/upper bound of a section dimension
 * is present or not.
 *
 ****************************************************************************/

inline static BOOL is_dim_bound_valid(const WN *wn)
{
    if (wn == NULL) return FALSE;
    return (WN_operator(wn) != OPR_INTCONST) || (WN_const_val(wn) != -1);
}

/*****************************************************************************
 *
 * Check if the index range <triplet> is full, i.e. identical to the array
 * dimension range. This method does not do fancy evaluation of the index
 * values.
 *
 * If <ah> is [1:n] where n is a symbol, the range is assumed to be [0:n-1].
 *
 * Return TRUE if <triplet> is full and FALSE otherwise.
 *
 ****************************************************************************/

static BOOL HC_compare_dim_range(const ARB_HANDLE &ah, const WN *triplet)
{
    WN *lbnd_wn = WN_kid0(triplet), *ubnd_wn = WN_kid2(triplet);

    // Handle for the special case: [1:n].
    if (ARB_const_lbnd(ah) && ARB_lbnd_val(ah) == 1 && !ARB_const_ubnd(ah))
    {
        return (WN_operator(lbnd_wn) == OPR_INTCONST
                && WN_const_val(lbnd_wn) == 0
                && WN_operator(ubnd_wn) == OPR_LDID
                && WN_offset(ubnd_wn) == 0
                && WN_st_idx(ubnd_wn) == ARB_ubnd_var(ah));
    }

    // Compare the lower bound.
    if (ARB_const_lbnd(ah)) {
        if (WN_operator(lbnd_wn) != OPR_INTCONST
                || WN_const_val(lbnd_wn) != ARB_lbnd_val(ah)) return FALSE;
    } else {
        if (WN_operator(lbnd_wn) != OPR_LDID
                || WN_offset(lbnd_wn) != 0
                || WN_st_idx(lbnd_wn) != ARB_lbnd_var(ah)) return FALSE;
    }

    // Compare the upper bound.
    if (ARB_const_ubnd(ah)) {
        if (WN_operator(ubnd_wn) != OPR_INTCONST
                || WN_const_val(ubnd_wn) != ARB_ubnd_val(ah)) return FALSE;
    } else {
        if (WN_operator(ubnd_wn) != OPR_LDID
                || WN_offset(ubnd_wn) != 0
                || WN_st_idx(ubnd_wn) != ARB_ubnd_var(ah)) return FALSE;
    }

    return TRUE;
}

HC_ARRSECTION_INFO::HC_ARRSECTION_INFO(WN *section_wn, MEM_POOL *pool)
{
    Is_True(pool != NULL, (""));
    _pool = pool;
    _section = NULL;

    /* Parse the XPRAGMA node. */

    OPERATOR opr = WN_operator(section_wn);
    Is_True(opr == OPR_ARRSECTION, (""));

    WN *base_addr_wn = WN_kid0(section_wn);
    opr = WN_operator(base_addr_wn);
    Is_True(opr == OPR_LDA || opr == OPR_LDID, (""));
    Is_True(WN_offset(base_addr_wn) == 0, (""));

    // Get the array symbol.
    _arr_st_idx = WN_st_idx(base_addr_wn);

    // Validate the symbol type.
    TY_IDX arr_ty_idx = ST_type(_arr_st_idx);
    if (TY_kind(arr_ty_idx) == KIND_POINTER)
    {
        arr_ty_idx = TY_pointed(arr_ty_idx);
    }
    HC_assert(TY_kind(arr_ty_idx) == KIND_ARRAY,
            ("Variable <%s> in an array section spec is not an array!\n",
             ST_name(_arr_st_idx)));

    // Do a sanity check on the section's dimensionality.
    _ndims = WN_num_dim(section_wn);
    UINT arr_ndims = num_array_dims(arr_ty_idx);
    HC_assert(_ndims == arr_ndims,
            ("%dD array <%s> appears in a %dD array section spec!\n",
             arr_ndims, ST_name(_arr_st_idx), _ndims));

    _dim_sz = CXX_NEW_ARRAY(WN*, _ndims, pool);
    _dim_lbnd = CXX_NEW_ARRAY(WN*, _ndims, pool);
    _dim_ubnd = CXX_NEW_ARRAY(WN*, _ndims, pool);
    _full_dim_range = CXX_NEW_ARRAY(BOOL, _ndims, pool);

    // Analyze each array section dimension.
    TY_IDX ty_idx = arr_ty_idx;
    INT dim_idx = 0;
    while (TY_kind(ty_idx) == KIND_ARRAY)
    {
        ARB_IDX arb_idx = Ty_Table[ty_idx].Arb();
        UINT16 dim = ARB_dimension(ARB_HANDLE(arb_idx));

        for (UINT16 i = 0; i < dim; ++i)
        {
            ARB_HANDLE ah = ARB_HANDLE(arb_idx + i);

            WN *triplet = WN_array_index(section_wn,dim_idx);

            // Get the original array's dimension size.
            // NOTE: the special case in shape propagation [1:ST_IDX] works
            // perfectly here.
            WN *arr_dim_sz = array_dim_size(ah);
            _dim_sz[dim_idx] = arr_dim_sz;

            // Expand the compact form of specification.
            BOOL is_lbnd_present = is_dim_bound_valid(WN_kid0(triplet));
            BOOL is_ubnd_present = is_dim_bound_valid(WN_kid2(triplet));
            if (is_ubnd_present)
            {
                // This is a regular range.
                // The end index is present, so should the start index.
                Is_True(is_lbnd_present, (""));

                _full_dim_range[dim_idx] = HC_compare_dim_range(ah, triplet);
            }
            else if (is_lbnd_present)
            {
                // This is a single-point range.
                WN_DELETE_Tree(WN_kid2(triplet));
                WN_kid2(triplet) = WN_COPY_Tree(WN_kid0(triplet));

                // Check if the array dimension size is 1.
                _full_dim_range[dim_idx] =
                    (WN_operator(arr_dim_sz) == OPR_INTCONST)
                    && (WN_const_val(arr_dim_sz) == 1);
            }
            else
            {
                // This is a full range.
                // Get start and end idx from the array type.
                WN_DELETE_Tree(WN_kid0(triplet));
                WN_DELETE_Tree(WN_kid2(triplet));

                // Handle the special case: [1:n].
                if (ARB_const_lbnd(ah) && ARB_lbnd_val(ah) == 1
                        && !ARB_const_ubnd(ah))
                {
                    // The range is [0:n-1].
                    WN_kid0(triplet) = WN_Intconst(Integer_type, 0);
                    WN_kid2(triplet) = WN_Sub(Integer_type,
                            WN_LdidScalar(ARB_ubnd_var(ah)),
                            WN_Intconst(Integer_type, 1));
                }
                else
                {
                    WN_kid0(triplet) = ARB_const_lbnd(ah) ?
                        WN_Intconst(Integer_type, ARB_lbnd_val(ah)) :
                        WN_LdidScalar(ARB_lbnd_var(ah));
                    WN_kid2(triplet) = ARB_const_ubnd(ah) ?
                        WN_Intconst(Integer_type, ARB_ubnd_val(ah)) :
                        WN_LdidScalar(ARB_ubnd_var(ah));
                }

                _full_dim_range[dim_idx] = TRUE;
            }

            // Fill the lower/upper bound's ARRAY.
            _dim_lbnd[dim_idx] = WN_COPY_Tree(WN_kid0(triplet));
            _dim_ubnd[dim_idx] = WN_COPY_Tree(WN_kid2(triplet));

            ++dim_idx;
        }

        ty_idx = TY_etype(ty_idx);
    }

    // Now, <ty_idx> holds the array element type.
    _elem_ty_idx = ty_idx;

#if 0
    // Do a sanity check on the array's element size.
    INT elem_sz = TY_size(_elem_ty_idx);
    Is_True(elem_sz == WN_element_size(section_wn),
            ("HC_ARRSECTION_INFO: mismatch in array element size for <%s>\n",
             ST_name(_arr_st_idx)));
#endif

    // Find the pivot dimension: the right-most non-full dimension index or
    // zero if all dimensions are full.
    INT i = _ndims - 1;
    while (i >= 0 && _full_dim_range[i]) --i;
    _pivot_dim_idx = (i < 0) ? 0 : (UINT)i;
}

HC_ARRSECTION_INFO::HC_ARRSECTION_INFO(const HC_ARRSECTION_INFO *orig,
        MEM_POOL *pool)
{
    Is_True(pool != NULL, (""));
    _pool = pool;

    _arr_st_idx = orig->_arr_st_idx;
    _elem_ty_idx = orig->_elem_ty_idx;

    _ndims = orig->_ndims;
    _dim_sz = CXX_NEW_ARRAY(WN*, _ndims, pool);
    _dim_lbnd = CXX_NEW_ARRAY(WN*, _ndims, pool);
    _dim_ubnd = CXX_NEW_ARRAY(WN*, _ndims, pool);
    _full_dim_range = CXX_NEW_ARRAY(BOOL, _ndims, pool);
    for (UINT i = 0; i < _ndims; ++i)
    {
        _dim_sz[i] = WN_COPY_Tree(orig->_dim_sz[i]);
        _dim_lbnd[i] = WN_COPY_Tree(orig->_dim_lbnd[i]);
        _dim_ubnd[i] = WN_COPY_Tree(orig->_dim_ubnd[i]);
        _full_dim_range[i] = orig->_full_dim_range[i];
    }
    _pivot_dim_idx = orig->_pivot_dim_idx;

    _section = NULL;
    if (orig->_section != NULL) {
        _section = orig->_section->create_deep_copy(pool);
    }
}

WN* HC_ARRSECTION_INFO::get_orig_dim_sz(UINT idx) const
{
    Is_True(idx < _ndims,
            ("HC_ARRSECTION_INFO::get_orig_dim_sz: "
             "idx %d out of range\n", idx));
    return _dim_sz[idx];
}

WN* HC_ARRSECTION_INFO::get_dim_sz(UINT idx) const
{
    Is_True(idx < _ndims, (""));

    WN *size_wn = WN_Sub(Integer_type,
            WN_COPY_Tree(_dim_ubnd[idx]), WN_COPY_Tree(_dim_lbnd[idx]));
    size_wn = WN_Add(Integer_type, size_wn, WN_Intconst(Integer_type, 1));

    // Simplify the expression.
    WN *simp_size_wn = HCWN_simplify_expr(size_wn);
    if (simp_size_wn != NULL)
    {
        WN_DELETE_Tree(size_wn);
        return simp_size_wn;
    }

    return size_wn;
}

WN* HC_ARRSECTION_INFO::get_section_sz() const
{
    // Start with the element size.
    WN *section_sz_wn = WN_Intconst(Integer_type, TY_size(_elem_ty_idx));

    // Multiply each dimension's size.
    for (UINT i = 0; i < _ndims; ++i)
    {
        section_sz_wn = WN_Mpy(Integer_type, section_sz_wn, get_dim_sz(i));
    }

    return section_sz_wn;
}

BOOL HC_ARRSECTION_INFO::project(ST_IDX st_idx, WN_OFFSET ofst,
        WN *var_lbnd_wn, WN *var_ubnd_wn)
{
    WN* dim_lbnd[_ndims];
    WN* dim_ubnd[_ndims];

    UINT i;
    for (i = 0; i < _ndims; ++i)
    {
        dim_lbnd[i] = HCWN_expr_min(_dim_lbnd[i],
                st_idx, ofst, var_lbnd_wn, var_ubnd_wn);
        dim_ubnd[i] = HCWN_expr_max(_dim_ubnd[i],
                st_idx, ofst, var_lbnd_wn, var_ubnd_wn);
        if (dim_lbnd[i] == NULL || dim_ubnd[i] == NULL) break;
    }

    if (i < _ndims)
    {
        // Dimension i cannot be projected. Clean up the previous projected
        // dimensions.
        for (UINT j = 0; j <= i; ++j)
        {
            WN_DELETE_Tree(dim_lbnd[j]); dim_lbnd[j] = NULL;
            WN_DELETE_Tree(dim_ubnd[j]); dim_ubnd[j] = NULL;
        }

        return FALSE;
    }

    // Replace the original dimension bounds with the projected ones.
    for (i = 0; i < _ndims; ++i)
    {
        WN_DELETE_Tree(_dim_lbnd[i]); _dim_lbnd[i] = dim_lbnd[i];
        WN_DELETE_Tree(_dim_ubnd[i]); _dim_ubnd[i] = dim_ubnd[i];
    }

    return TRUE;
}

WN* HC_ARRSECTION_INFO::get_bound(BOOL is_lower) const
{
    // Construct the base address.
    TY_IDX arr_ty_idx = ST_type(_arr_st_idx);
    // TODO: do we specify a different TY?
    WN *base_addr_wn = (TY_kind(arr_ty_idx) == KIND_ARRAY) ?
        WN_LdaZeroOffset(_arr_st_idx) : WN_LdidScalar(_arr_st_idx);

    // Construct the ARRAY node.
    WN *bnd_wn = HCWN_CreateArray(base_addr_wn, _ndims);
    WN_element_size(bnd_wn) = TY_size(_elem_ty_idx);

    // Fill the dimension sizes and indices.
    WN **dim_bnd = is_lower ? _dim_lbnd : _dim_ubnd;
    for (UINT i = 0; i < _ndims; ++i)
    {
        WN_array_dim(bnd_wn,i) = WN_COPY_Tree(_dim_sz[i]);
        WN_array_index(bnd_wn,i) = WN_COPY_Tree(dim_bnd[i]);
    }

    return bnd_wn;
}

WN* HC_ARRSECTION_INFO::get_lower_bound() const
{
    return get_bound(TRUE);
}

WN* HC_ARRSECTION_INFO::get_upper_bound() const
{
    return get_bound(FALSE);
}

void HC_ARRSECTION_INFO::set_section(ACCESS_ARRAY *lbnd, ACCESS_ARRAY *ubnd,
        mUINT8 depth)
{
    _section = CXX_NEW(PROJECTED_REGION(lbnd, ubnd, depth, _pool), _pool);
}

void HC_ARRSECTION_INFO::deep_copy_section(MEM_POOL *pool)
{
    if (_section != NULL) _section = _section->create_deep_copy(pool);
}

BOOL HC_ARRSECTION_INFO::replace_syms_walker(WN *wn,
        ST_IDX *from_syms, ST_IDX *to_syms, UINT n_syms)
{
    if (wn == NULL) return TRUE;

    BOOL fully_replaced = TRUE;

    OPERATOR opr = WN_operator(wn);
    if (OPERATOR_has_sym(opr))
    {
        ST_IDX st_idx = WN_st_idx(wn);
        Is_True(st_idx != ST_IDX_ZERO, (""));

        // Search in the symbol map.
        UINT i;
        for (i = 0; i < n_syms; ++i) {
            if (st_idx == from_syms[i]) break;
        }

        if (i < n_syms) {
            Is_True(to_syms[i] != ST_IDX_ZERO, (""));
            WN_st_idx(wn) = to_syms[i];
        } else if (! HCST_is_global_symbol(st_idx)) {
            // This non-global symbol cannot be replaced.
            fully_replaced = FALSE;
        }
    }
    
    // Handle composite node.
    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn)) {
            if (! HC_ARRSECTION_INFO::replace_syms_walker(kid_wn,
                        from_syms, to_syms, n_syms)) {
                fully_replaced = FALSE;
            }
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            if (! HC_ARRSECTION_INFO::replace_syms_walker(WN_kid(wn,i),
                        from_syms, to_syms, n_syms)) {
                fully_replaced = FALSE;
            }
        }
    }

    return fully_replaced;
}

BOOL HC_ARRSECTION_INFO::replace_syms(ST_IDX st_idx,
        ST_IDX *from_syms, ST_IDX *to_syms, UINT n_syms)
{
    _arr_st_idx = st_idx;

    BOOL fully_replaced = TRUE;

    // Go through each dimension.
    for (UINT i = 0; i < _ndims; ++i)
    {
        if (! HC_ARRSECTION_INFO::replace_syms_walker(_dim_sz[i],
                    from_syms, to_syms, n_syms)) fully_replaced = FALSE;
        if (! HC_ARRSECTION_INFO::replace_syms_walker(_dim_lbnd[i],
                    from_syms, to_syms, n_syms)) fully_replaced = FALSE;
        if (! HC_ARRSECTION_INFO::replace_syms_walker(_dim_ubnd[i],
                    from_syms, to_syms, n_syms)) fully_replaced = FALSE;
    }

    return fully_replaced;
}

#ifdef IPA_HICUDA

void HC_ARRSECTION_INFO::build_st_table(HASH_TABLE<ST_IDX,ST*> *st_tbl) const
{
    // the main symbol
    st_tbl->Enter_If_Unique(_arr_st_idx, &St_Table[_arr_st_idx]);

    // dimension sizes, lower/upper bounds
    for (UINT i = 0; i < _ndims; ++i)
    {
        HCWN_build_st_table(_dim_sz[i], st_tbl);
        HCWN_build_st_table(_dim_lbnd[i], st_tbl);
        HCWN_build_st_table(_dim_ubnd[i], st_tbl);
    }
}

void HC_ARRSECTION_INFO::replace_idxvs_with_formals(
        HC_SYM_MAP *new_formal_map, const HASH_TABLE<ST_IDX, ST*> *st_tbl)
{
    for (UINT i = 0; i < _ndims; ++i)
    {
        HCWN_replace_syms(_dim_sz[i], new_formal_map, st_tbl);
        HCWN_replace_syms(_dim_lbnd[i], new_formal_map, st_tbl);
        HCWN_replace_syms(_dim_ubnd[i], new_formal_map, st_tbl);
    }
}

UINT HC_ARRSECTION_INFO::compute_pivot_dim_idx(
        const HC_ARRSECTION_INFO *other) const
{
    // The given section must have the same dimensionality as this one.
    Is_True(other != NULL && other->_ndims == _ndims, (""));

    // Make sure we turn on the simplifier.
    BOOL old_simp_state = WN_Simplifier_Enable(TRUE);

    // Compare the lower and upper bounds of each dimension, starting from the
    // innermost/rightmost one.
    UINT p_idx = _ndims - 1;
    for ( ; p_idx > 0; --p_idx)
    {
        // First check method: compute the difference and compare it with 0.
        WN *lbnd_diff_wn = WN_Sub(Integer_type,
                WN_COPY_Tree(_dim_lbnd[p_idx]),
                WN_COPY_Tree(other->_dim_lbnd[p_idx]));
        if (WN_operator(lbnd_diff_wn) != OPR_INTCONST
                || WN_const_val(lbnd_diff_wn) == 0)
        {
            // Second check method: compare the trees structurally.
            if (WN_Simp_Compare_Trees(_dim_lbnd[p_idx],
                        other->_dim_lbnd[p_idx])) break;
        }

        // Do the same thing for upper bound.
        WN *ubnd_diff_wn = WN_Sub(Integer_type,
                WN_COPY_Tree(_dim_ubnd[p_idx]),
                WN_COPY_Tree(other->_dim_ubnd[p_idx]));
        if (WN_operator(ubnd_diff_wn) != OPR_INTCONST
                || WN_const_val(ubnd_diff_wn) == 0)
        {
            if (WN_Simp_Compare_Trees(_dim_ubnd[p_idx],
                        other->_dim_ubnd[p_idx])) break;
        }
    }

    // Restore the simplifier enablement.
    WN_Simplifier_Enable(old_simp_state);

    return p_idx;
}

#endif  // IPA_HICUDA

BOOL HC_ARRSECTION_INFO::equals(const HC_ARRSECTION_INFO *other) const
{
    if (this == other) return TRUE;
    if (other == NULL) return FALSE;

    // Compare the element type.
    if (_elem_ty_idx != other->_elem_ty_idx) return FALSE;
    // Compare the dimensionality.
    if (_ndims != other->_ndims) return FALSE;

    // Compare the array section. We need to make sure that the original array
    // shapes are the same and the section bounds are the same.
    for (UINT i = 0; i < _ndims; ++i)
    {
        // Compare the original array's dimension size.
        WN *dim_sz_a = _dim_sz[i], *dim_sz_b = other->_dim_sz[i];
        if (WN_Simp_Compare_Trees(dim_sz_a, dim_sz_b)) return FALSE;

        // Compare the array section's lower bound.
        WN *lbnd_a = _dim_lbnd[i], *lbnd_b = other->_dim_lbnd[i];
        if (WN_Simp_Compare_Trees(lbnd_a, lbnd_b)) return FALSE;

        // Compare the array section's upper bound.
        WN *ubnd_a = _dim_ubnd[i], *ubnd_b = other->_dim_ubnd[i];
        if (WN_Simp_Compare_Trees(ubnd_a, ubnd_b)) return FALSE;
    }

    return TRUE;
}

void HC_ARRSECTION_INFO::print(FILE *fp) const
{
    fprintf(fp, "==== ARRSECTION of %s (%s) ===\n",
            ST_name(_arr_st_idx), TY_name(_elem_ty_idx));

    for (UINT i = 0; i < _ndims; ++i) fdump_tree(fp, _dim_lbnd[i]);
    fprintf(fp, "     ||\n");
    fprintf(fp, "     \\/\n");
    for (UINT i = 0; i < _ndims; ++i) fdump_tree(fp, _dim_ubnd[i]);

    fprintf(fp, "==== END OF ARRSECTION ====\n");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

HC_GPU_DATA::HC_GPU_DATA(WN *pragma_wn, IPA_NODE *node, UINT dir_id,
        MEM_POOL *pool)
{
    Is_True(pool != NULL, (""));
    _pool = pool;

    Is_True(node != NULL, (""));
    _orig_proc_node = node;
    _dir_id = dir_id;

    _flags = 0;

    /* Parse the pragma (and the pragmas after it). */

    OPERATOR opr = WN_operator(pragma_wn);
    Is_True(opr == OPR_PRAGMA || opr == OPR_XPRAGMA,
            ("HC_GPU_DATA: non-pragma\n"));

    WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(pragma_wn);
    Is_True(pid == WN_PRAGMA_HC_GLOBAL_COPYIN
            || pid == WN_PRAGMA_HC_CONST_COPYIN
            || pid == WN_PRAGMA_HC_SHARED_COPYIN,
            ("HC_GPU_DATA: invalid pragma type\n"));

    WN *alloc_pragma_wn = NULL;
    if (pid == WN_PRAGMA_HC_CONST_COPYIN)
    {
        _type = HC_CONSTANT_DATA;

        // The current pragma stores the "copyin" field.
        Is_True(opr == OPR_XPRAGMA,
                ("HC_GPU_DATA: invalid CONSTANT pragma\n"));
        alloc_pragma_wn = pragma_wn;

        set_do_copyin();
    }
    else
    {
        // GLOBAL or SHARED
        _type = (pid == WN_PRAGMA_HC_GLOBAL_COPYIN) ?
            HC_GLOBAL_DATA : HC_SHARED_DATA;

        // Get the flags in the base pragma.
        Is_True(opr == OPR_PRAGMA,
                ("HC_GPU_DATA: invalid GLOBAL pragma\n"));
        UINT flags = WN_pragma_arg1(pragma_wn);
        
        if (flags & HC_DIR_COPYIN) set_do_copyin();
        if (flags & HC_DIR_COPYIN_NOBNDCHECK)
        {
            // For now, only a SHARED directive is allowed to set this flag.
            Is_True(_type == HC_SHARED_DATA, (""));
            set_do_copyin_nobndcheck();
        }
        if (flags & HC_DIR_CLEAR)
        {
            Is_True(!do_copyin(), (""));
            set_do_clear();
        }

        // The next pragma stores the "alloc" field.
        alloc_pragma_wn = WN_next(pragma_wn);
        Is_True(alloc_pragma_wn != NULL
                && WN_opcode(alloc_pragma_wn) == OPC_XPRAGMA
                && WN_pragma(alloc_pragma_wn) == WN_PRAGMA_HC_DECL,
                ("HC_GPU_DATA: invalid GLOBAL ALLOC pragma\n"));
    }

    WN *alloc_wn = WN_kid0(alloc_pragma_wn);
    Is_True(alloc_wn != NULL,
            ("HC_GPU_DATA::HC_GPU_DATA: null ALLOC WN\n"));
    OPERATOR alloc_opr = WN_operator(alloc_wn);
    Is_True(alloc_opr == OPR_ARRSECTION || alloc_opr == OPR_LDA,
            ("HC_GPU_DATA::HC_GPU_DATA: invalid ALLOC opcode\n"));

    if (alloc_opr == OPR_LDA)
    {
        // This is a scalar.
        _st_idx = WN_st_idx(alloc_wn);
        // TODO: include directive's source position.
        Is_True(HCST_is_scalar(_st_idx),
                ("The %s directive of scalar variable <%s> "
                 "has a section specification.\n",
                 _type == HC_GLOBAL_DATA ? "GLOBAL" : "CONSTANT",
                 ST_name(_st_idx)));

        _alloc_section = NULL;
    }
    else
    {
        // This is an array section.
        // TODO: do the check first.
        _alloc_section = CXX_NEW(HC_ARRSECTION_INFO(alloc_wn, pool), pool);
        _st_idx = _alloc_section->get_arr_sym();
    }

    // Fill the COPYIN field.
    // For CONSTANT directive, it is always NULL. For GLOBAL or SHARED
    // directive with array section, it could be non-NULL.
    _copyin_section = NULL;
    if (do_copyin() && (_type == HC_GLOBAL_DATA || _type == HC_SHARED_DATA))
    {
        // Check the next pragma.
        WN *copyin_pragma_wn = WN_next(alloc_pragma_wn);
        if (copyin_pragma_wn != NULL
                && WN_pragma(copyin_pragma_wn) == WN_PRAGMA_HC_COPY)
        {
            WN *copyin_wn = WN_kid0(copyin_pragma_wn);
            Is_True(copyin_wn != NULL,
                    ("HC_GPU_DATA: NULL COPYIN WN\n"));
            OPERATOR copyin_opr = WN_operator(copyin_wn);
            Is_True(copyin_opr == OPR_ARRSECTION || copyin_opr == OPR_LDA,
                    ("HC_GPU_DATA: invalid COPYIN opcode\n"));

            // Make sure that ALLOC and COPYIN field refer to the same symbol.
            ST_IDX copyin_st_idx = WN_st_idx(
                    (copyin_opr == OPR_LDA) ? copyin_wn : WN_kid0(copyin_wn));
            Is_True(_st_idx == copyin_st_idx,
                    ("HC_GPU_DATA: unmatched COPYIN symbol <%s>\n",
                     ST_name(copyin_st_idx)));

            if (copyin_opr == OPR_ARRSECTION)
            {
                _copyin_section = CXX_NEW(
                        HC_ARRSECTION_INFO(copyin_wn, pool), pool);
                // TODO: ensure that the copyin section is within the
                // allocated section.
            }
        }
    }

    _copyout_section = NULL;

    _partner_gdata = NULL;
    _lp_idxv_props = NULL;
    _kinfo = NULL;

    _gvar_info = NULL;
    _ig_node = NULL;
}

HC_GPU_DATA::HC_GPU_DATA(const HC_GPU_DATA *orig, MEM_POOL *pool)
{
    Is_True(pool != NULL, (""));
    _pool = pool;

    _orig_proc_node = orig->_orig_proc_node;
    _dir_id = orig->_dir_id;

    _type = orig->_type;
    _st_idx = orig->_st_idx;

    _flags = orig->_flags;

    _alloc_section = NULL;
    if (orig->_alloc_section != NULL)
    {
        _alloc_section = CXX_NEW(
                HC_ARRSECTION_INFO(orig->_alloc_section, pool), pool);
    }
    _copyin_section = NULL;
    if (orig->_copyin_section != NULL)
    {
        _copyin_section = CXX_NEW(
                HC_ARRSECTION_INFO(orig->_copyin_section, pool), pool);
    }
    _copyout_section = NULL;
    if (orig->_copyout_section != NULL)
    {
        _copyout_section = CXX_NEW(
                HC_ARRSECTION_INFO(orig->_copyout_section, pool), pool);
    }

    _gvar_info = NULL;
    if (orig->_gvar_info != NULL) create_gvar_info(orig->_gvar_info);

    Is_True(orig->_partner_gdata == NULL, (""));
    _partner_gdata = NULL;

    _lp_idxv_props = NULL;
    if (orig->_lp_idxv_props != NULL)
    {
        // Deep-copy the list but shallow-copy the list elements.
        _lp_idxv_props = CXX_NEW(HC_EXPR_PROP_LIST(_pool), _pool);
        UINT n_loops = orig->_lp_idxv_props->Elements();
        for (UINT i = 0; i < n_loops; ++i)
        {
            _lp_idxv_props->AddElement((*orig->_lp_idxv_props)[i]);
        }
    }

    _kinfo = orig->_kinfo;

    // TODO: these rules might not be true as the program evolves.
    Is_True(orig->_ig_node == NULL, (""));
    _ig_node = NULL;
}

void HC_GPU_DATA::parse_copyout_dir(WN *pragma_wn)
{
    Is_True(pragma_wn != NULL, (""));

    OPERATOR opr = WN_operator(pragma_wn);

    set_do_copyout();
    if (is_arr_section())
    {
        Is_True(opr == OPR_ARRSECTION, (""));
        _copyout_section =
            CXX_NEW(HC_ARRSECTION_INFO(pragma_wn,_pool), _pool);
    }
    else
    {
        Is_True(opr == OPR_LDA, (""));
        _copyout_section = NULL;
    }

    if (WN_st_idx(pragma_wn) == 1)
    {
        // For now, only a SHARED directive is allowed to set this flag.
        Is_True(_type == HC_SHARED_DATA, (""));
        set_do_copyout_nobndcheck();
    }
}

WN* HC_GPU_DATA::compute_size() const
{
    return (_alloc_section == NULL) ?
        WN_Intconst(Integer_type, TY_size(ST_type(_st_idx))) :
        _alloc_section->get_section_sz();
}

HC_EXPR_PROP_LIST* HC_GPU_DATA::create_lp_idxv_prop_list()
{
    Is_True(_lp_idxv_props == NULL, (""));
    _lp_idxv_props = CXX_NEW(HC_EXPR_PROP_LIST(_pool), _pool);
    return _lp_idxv_props;
}

/*****************************************************************************
 *
 * This method cannot be put in the header file as HC_GPU_VAR_INFO is not
 * declared in the header file.
 *
 ****************************************************************************/

HC_GPU_VAR_INFO* HC_GPU_DATA::create_gvar_info(const HC_GPU_VAR_INFO *orig)
{
    Is_True(_gvar_info == NULL, (""));

    if (orig != NULL)
    {
        // The default copy constructor will shallow-copy the <size_wn>.
        _gvar_info = CXX_NEW(HC_GPU_VAR_INFO(*orig), _pool);
    }
    else
    {
        _gvar_info = CXX_NEW(HC_GPU_VAR_INFO(), _pool);
    }

    return _gvar_info;
}

BOOL HC_GPU_DATA::replace_syms(ST_IDX st_idx,
        ST_IDX *from_syms, ST_IDX *to_syms, UINT n_syms)
{
    _st_idx = st_idx;

    if (_alloc_section == NULL) return TRUE;

    return _alloc_section->replace_syms(st_idx, from_syms, to_syms, n_syms);
}

#ifdef IPA_HICUDA

void HC_GPU_DATA::add_idxv_range(const HC_LOOP_PART_INFO *lpi)
{
    // If the LOOP_PARTITION directive does not have a OVER_THREAD clause,
    // all threads within a block work on the same loop index at a time,
    // so there is no need to do projection.
    if (lpi->get_thread_clause() == HC_LPI_DT_NONE) return;

    ST_IDX st_idx = lpi->get_idxv();
    WN *lbnd = lpi->get_idxv_lbnd(), *ubnd = lpi->get_idxv_ubnd();
    Is_True(lbnd != NULL && ubnd != NULL, (""));

    if (_alloc_section != NULL)
    {
        HC_assert(_alloc_section->project(st_idx, 0, lbnd, ubnd),
                ("The ALLOC section in a %s directive for <%s> cannot be "
                 "projected with respect to loop index variable <%s>!",
                 HC_gpu_data_type_name(_type), ST_name(_st_idx),
                 ST_name(st_idx)));
    }
    if (_copyin_section != NULL)
    {
        HC_assert(_copyin_section->project(st_idx, 0, lbnd, ubnd),
                ("The COPYIN section in a %s directive for <%s> cannot be "
                 "projected with respect to loop index variable <%s>!",
                 HC_gpu_data_type_name(_type), ST_name(_st_idx),
                 ST_name(st_idx)));
    }
    if (_copyout_section != NULL)
    {
        HC_assert(_copyout_section->project(st_idx, 0, lbnd, ubnd),
                ("The COPYOUT section in a %s directive for <%s> cannot be "
                 "projected with respect to loop index variable <%s>!",
                 HC_gpu_data_type_name(_type), ST_name(_st_idx),
                 ST_name(st_idx)));
    }
}

void HC_GPU_DATA::build_st_table(HASH_TABLE<ST_IDX,ST*> *st_tbl) const
{
    if (_alloc_section != NULL) {
        _alloc_section->build_st_table(st_tbl);
    }

    if (_gvar_info != NULL) {
        ST_IDX st_idx = _gvar_info->get_symbol();
        st_tbl->Enter(st_idx, &St_Table[st_idx]);
    }
}

void HC_GPU_DATA::replace_idxvs_with_formals(IPA_NODE *node,
        HC_SYM_MAP *new_formal_map, const HASH_TABLE<ST_IDX, ST*> *st_tbl)
{
    if (_alloc_section == NULL) return;

    // IMPORTANT!
    IPA_NODE_CONTEXT context(node);

    _alloc_section->replace_idxvs_with_formals(new_formal_map, st_tbl);
}

BOOL HC_GPU_DATA::have_same_origin(const HC_GPU_DATA *other) const
{
    if (this == other) return TRUE;
    if (other == NULL) return FALSE;

    return (_orig_proc_node == other->_orig_proc_node
            && _dir_id == other->_dir_id);
}

BOOL HC_GPU_DATA::equals(const HC_GPU_DATA *other) const
{
    if (this == other) return TRUE;
    if (other == NULL) return FALSE;

    // Compare the GPU data type.
    if (_type != other->_type) return FALSE;

    // Here, we do not care if the main symbols are the same.

    if (_alloc_section == NULL)
    {
        // This is a non-array variable.
        if (other->_alloc_section != NULL) return FALSE;
        // Just check the symbol type.
        // TODO: more accurate type check.
        if (ST_type(_st_idx) != ST_type(other->_st_idx)) return FALSE;
    }
    else
    {
        // Compare the ALLOC section.
        if (! _alloc_section->equals(other->_alloc_section)) return FALSE;
    }

    return TRUE;
}

void HC_GPU_DATA::print(FILE *fp) const
{
    if (_alloc_section != NULL) {
        fprintf(fp, "ARRAY_SECTION:\n");
        _alloc_section->print(fp);
    } else {
        fprintf(fp, "NON-ARRAY_SECTION: TYPE %s\n",
                TY_name(ST_type(_st_idx)));
    }
}

#endif  // IPA_HICUDA

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
#if 0
HC_SHARED_DATA::HC_SHARED_DATA(WN *pragma_wn, MEM_POOL *pool)
{
}
#endif
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void HC_GPU_DATA_STACK::push(HC_GPU_DATA *hgd)
{
    // Select based on the host variable symbol.
    ST_IDX st_idx = hgd->get_symbol();
    GPU_DATA_PER_SYMBOL_STACK *s = _stack->Find(st_idx);
    if (s == NULL) {
        s = CXX_NEW(GPU_DATA_PER_SYMBOL_STACK(_pool), _pool);
        _stack->Enter(st_idx, s);
    }

    // Push the new entry.
    s->Push(hgd);
}

// TODO: we should not have used STACK to implement this.
HC_GPU_DATA* HC_GPU_DATA_STACK::pop(HC_GPU_DATA_TYPE dtype, ST_IDX st_idx)
{
    GPU_DATA_PER_SYMBOL_STACK *s = _stack->Find(st_idx);
    if (s == NULL) return NULL;

    // Search from the stack top, until seeing a HC_GPU_DATA with the correct
    // data type.
    HC_GPU_DATA **tmp_list = (HC_GPU_DATA**)alloca(
            s->Elements() * sizeof(HC_GPU_DATA*));
    INT gdata_count = 0;

    HC_GPU_DATA *gdata = NULL;
    while ((gdata = s->Pop()) != NULL)
    {
        if (gdata->get_type() == dtype) break;
        // Store it in the tmp list for later restoration.
        tmp_list[gdata_count++] = gdata;
    }

    // Restore the stack.
    if (gdata == NULL) Is_True(gdata_count == s->Elements(), (""));
    // MUST USE "INT" HERE.
    for (INT i = gdata_count-1; i >= 0; --i) s->Push(tmp_list[i]);

    return gdata;
}

HC_GPU_DATA* HC_GPU_DATA_STACK::peek(ST_IDX st_idx,
        HC_GPU_DATA_TYPE dtype) const
{
    GPU_DATA_PER_SYMBOL_STACK *s = _stack->Find(st_idx);
    if (s == NULL) return NULL;

    // Search from the top until seeing an HC_GPU_DATA with the correct type.
    UINT n_gdata = s->Elements();
    for (UINT i = 0; i < n_gdata; ++i)
    {
        HC_GPU_DATA *gdata = s->Top_nth(i);
        if (gdata->get_type() == dtype) return gdata;
    }

    return NULL;
}

HC_GPU_DATA* HC_GPU_DATA_STACK::peek(ST_IDX st_idx) const
{
    GPU_DATA_PER_SYMBOL_STACK *s = _stack->Find(st_idx);
    // DAVID FIX: is this really a proper fix for crash on s->Top?
    return (s == NULL || s->Is_Empty()) ? NULL : s->Top();
}

typedef HASH_TABLE_ITER<ST_IDX, GPU_DATA_PER_SYMBOL_STACK*>
GPU_DATA_STACK_ITER;

HC_VISIBLE_GPU_DATA* HC_GPU_DATA_STACK::top(MEM_POOL *pool) const
{
    HC_VISIBLE_GPU_DATA *snapshot = CXX_NEW(
            HC_VISIBLE_GPU_DATA(41,pool), pool);

    ST_IDX st_idx;
    GPU_DATA_PER_SYMBOL_STACK *s = NULL;

    GPU_DATA_STACK_ITER it(_stack);
    while (it.Step(&st_idx, &s)) snapshot->Enter(st_idx, s->Top());

    return snapshot;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
 
#ifdef IPA_HICUDA

/*****************************************************************************
 *
 * This function translates each GLOBAL/CONSTANT directive in <node> to CUDA
 * code. The list of HC_GPU_DATA stored in <node> must be present.
 *
 * If <node> is an N-procedure, directives are simply removed and not
 * translated into CUDA code.
 *
 * Returns the next WN node to be processed if <parent_wn> is a BLOCK, and
 * NULL otherwise.
 *
 ****************************************************************************/

static WN* HC_lower_data_dir_walker(WN *wn, WN *parent_wn, IPA_NODE *node,
        UINT& gdata_dir_id, HC_GPU_DATA_STACK *stack)
{
    if (wn == NULL) return NULL;

    BOOL is_parent_block =
        (parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK);

    // Determine the next WN node.
    WN *next_wn = is_parent_block ? WN_next(wn) : NULL;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        Is_True(is_parent_block, (""));

        HC_GPU_DATA *gdata = NULL;
        HC_LOCAL_VAR_STORE *lvar_store = node->get_hc_lvar_store();
        BOOL gen_code = node->may_lead_to_kernel();

        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        switch (pid)
        {
            case WN_PRAGMA_HC_GLOBAL_COPYIN:
            case WN_PRAGMA_HC_CONST_COPYIN:
                // Retrieve HC_GPU_DATA from the list in the node.
                gdata = (*node->get_gpu_data_list())[gdata_dir_id++];
                // Update the stack.
                stack->push(gdata);
                // Lower the directive.
                next_wn = (pid == WN_PRAGMA_HC_GLOBAL_COPYIN) ?
                    HC_lower_global_copyin(wn, parent_wn,
                            gdata, lvar_store, gen_code) :
                    HC_lower_const_copyin(wn, parent_wn,
                            gdata, lvar_store, gen_code);
                break;

            case WN_PRAGMA_HC_GLOBAL_COPYOUT:
            {
                // Do a little parsing.
                Is_True(opr == OPR_XPRAGMA, (""));
                WN *copyout_wn = WN_kid0(wn);
                Is_True(copyout_wn != NULL, (""));
                OPERATOR copyout_opr = WN_operator(copyout_wn);
                Is_True(copyout_opr == OPR_ARRSECTION
                        || copyout_opr == OPR_LDA, (""));

                // Retrieve the matching HC_GPU_DATA from the stack.
                ST_IDX st_idx = (copyout_opr == OPR_ARRSECTION) ?
                    WN_st_idx(WN_kid0(copyout_wn)) : WN_st_idx(copyout_wn);
                gdata = stack->peek(st_idx, HC_GLOBAL_DATA);
                Is_True(gdata != NULL, (""));

                // Fully parse the directive.
                gdata->parse_copyout_dir(copyout_wn);

                // Lower the directive.
                next_wn = HC_lower_global_copyout(wn, parent_wn,
                        gdata, lvar_store, gen_code);

                break;
            }

            case WN_PRAGMA_HC_GLOBAL_FREE:
            case WN_PRAGMA_HC_CONST_REMOVE:
            {
                // Retrieve the matching HC_GPU_DATA from the stack.
                ST_IDX st_idx = WN_st_idx(wn);
                HC_GPU_DATA_TYPE dtype = (pid == WN_PRAGMA_HC_GLOBAL_FREE) ?
                    HC_GLOBAL_DATA : HC_CONSTANT_DATA;
                gdata = stack->pop(dtype, st_idx);
                Is_True(gdata != NULL, (""));

                // Lower the directive.
                next_wn = (pid == WN_PRAGMA_HC_GLOBAL_FREE) ?
                    HC_lower_global_free(wn, parent_wn, gdata, gen_code) :
                    HC_lower_const_remove(wn, parent_wn, gdata, gen_code);
                break;
            }
        }

        return next_wn;
    }

    WN *kid_wn;
    if (opr == OPR_BLOCK) {
        kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            kid_wn = HC_lower_data_dir_walker(kid_wn, wn, node,
                    gdata_dir_id, stack);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            Is_True(HC_lower_data_dir_walker(WN_kid(wn,i), wn, node,
                        gdata_dir_id, stack) == NULL, (""));
        }
    }

    return next_wn;
}

/*****************************************************************************
 *
 * ASSUME: CUDA runtime functions and other symbols have been declared
 * already.
 *
 ****************************************************************************/

void HC_handle_data_directives(IPA_NODE *node)
{
    // Check if this node does have GLOBAL/CONSTANT directives.
    if (!node->contains_global_dir() && !node->contains_const_dir()) return;

    // TODO: use another pool
    MEM_POOL *tmp_pool = node->Mem_Pool();
    
    // We need a stack because the list of HC_GPU_DATA in the node only
    // accounts for COPYIN directives.
    HC_GPU_DATA_STACK *stack = CXX_NEW(HC_GPU_DATA_STACK(tmp_pool), tmp_pool);

    IPA_NODE_CONTEXT context(node);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "In node <%s> ...\n", node->Name());
    }

    UINT gdata_dir_id = 0;
    HC_lower_data_dir_walker(node->Whirl_Tree(), NULL,
            node, gdata_dir_id, stack);
    Is_True(gdata_dir_id == node->get_gpu_data_list()->Elements(), (""));

    // Manual cleanup.
    CXX_DELETE(stack, tmp_pool);

    // Rebuild the Parent_Map.
    WN_Parentize(node->Whirl_Tree(), Parent_Map, Current_Map_Tab);

    // Verify the WHIRL tree.
    Is_True(WN_verifier(node->Whirl_Tree()), (""));

    // There is no need to reset the WN-to-EDGE map.
    // IPA_Call_Graph->Reset_Callsite_Map(node);
}


/*****************************************************************************
 *
 * <dir_id> only counts SHARED ALLOC directives.
 *
 ****************************************************************************/

static void HC_parse_shared_dir_walker(WN *wn, IPA_NODE *node,
        HC_GPU_DATA_LIST *sdata_list, UINT& dir_id, HC_GPU_DATA_STACK *stack,
        MEM_POOL *pool)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);
    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        HC_GPU_DATA *sdata = NULL;
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        if (pid == WN_PRAGMA_HC_SHARED_COPYIN)
        {
            sdata = CXX_NEW(HC_GPU_DATA(wn, node, dir_id++, pool), pool);
            sdata_list->AddElement(sdata);
            // Push it onto the stack.
            stack->push(sdata);
        }
        else if (pid == WN_PRAGMA_HC_SHARED_COPYOUT)
        {
            // Do a little parsing.
            Is_True(opr == OPR_XPRAGMA, (""));
            WN *copyout_wn = WN_kid0(wn);
            Is_True(copyout_wn != NULL, (""));
            OPERATOR copyout_opr = WN_operator(copyout_wn);
            Is_True(copyout_opr == OPR_ARRSECTION
                    || copyout_opr == OPR_LDA, (""));

            // Retrieve the matching HC_GPU_DATA from the stack.
            ST_IDX st_idx = (copyout_opr == OPR_ARRSECTION) ?
                WN_st_idx(WN_kid0(copyout_wn)) : WN_st_idx(copyout_wn);
            sdata = stack->peek(st_idx);
            Is_True(sdata != NULL, (""));

            // Fully parse the directive.
            sdata->parse_copyout_dir(copyout_wn);
        }
        else if (pid == WN_PRAGMA_HC_SHARED_REMOVE)
        {
            // Pop it off the stack.
            ST_IDX st_idx = WN_st_idx(wn);
            sdata = stack->pop(HC_SHARED_DATA, st_idx);
            Is_True(sdata != NULL, (""));
        }
    }

    // Handle composite node.
    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            HC_parse_shared_dir_walker(kid_wn, node,
                    sdata_list, dir_id, stack, pool);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            HC_parse_shared_dir_walker(WN_kid(wn,i), node,
                    sdata_list, dir_id, stack, pool);
        }
    }
}

void HC_parse_shared_directives(IPA_NODE *node, MEM_POOL *pool)
{
    // We only process K-/IK-procedures that contain at least one SHARED
    // directive.
    if (!node->contains_kernel() && !node->may_be_inside_kernel()) return;
    if (!node->contains_shared_dir()) return;

    IPA_NODE_CONTEXT context(node);

    HC_GPU_DATA_LIST *sdata_list = node->get_shared_data_list();

    HC_GPU_DATA_STACK *stack = CXX_NEW(HC_GPU_DATA_STACK(pool), pool);

    // The WN pool is the node's pool. DO NOT USE <pool>.
    UINT dir_id = 0;
    HC_parse_shared_dir_walker(node->Whirl_Tree(), node,
            sdata_list, dir_id, stack, pool);
    Is_True(dir_id == sdata_list->Elements(), (""));

    // Manual cleanup.
    CXX_DELETE(stack, pool);
}


static WN* HC_lower_shared_dir_walker(WN *wn, WN *parent_wn, IPA_NODE *node,
        UINT& sdata_dir_id, HC_GPU_DATA_STACK *stack)
{
    if (wn == NULL) return NULL;

    BOOL is_parent_block =
        (parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK);

    // Determine the next WN node.
    WN *next_wn = is_parent_block ? WN_next(wn) : NULL;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        Is_True(is_parent_block, (""));

        HC_GPU_DATA *sdata = NULL;
        HC_LOCAL_VAR_STORE *lvar_store = node->get_hc_lvar_store();
        BOOL gen_code = (node->contains_kernel()
                || node->may_be_inside_kernel());

        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        if (pid == WN_PRAGMA_HC_SHARED_COPYIN)
        {
            // Retrieve HC_GPU_DATA from the list in the node.
            sdata = (*node->get_shared_data_list())[sdata_dir_id++];
            // Update the stack.
            stack->push(sdata);
            // Lower the directive.
            next_wn = HC_lower_shared_copyin(wn, parent_wn,
                    sdata, lvar_store, gen_code);
        }
        else if (pid == WN_PRAGMA_HC_SHARED_COPYOUT)
        {
            // Do a little parsing.
            Is_True(opr == OPR_XPRAGMA, (""));
            WN *copyout_wn = WN_kid0(wn);
            Is_True(copyout_wn != NULL, (""));
            OPERATOR copyout_opr = WN_operator(copyout_wn);
            Is_True(copyout_opr == OPR_ARRSECTION, (""));

            // Retrieve the matching HC_GPU_DATA from the stack.
            ST_IDX st_idx = WN_st_idx(WN_kid0(copyout_wn));
            sdata = stack->peek(st_idx, HC_SHARED_DATA);
            Is_True(sdata != NULL, (""));

            // Lower the directive.
            next_wn = HC_lower_shared_copyout(wn, parent_wn,
                    sdata, lvar_store, gen_code);
        }
        else if (pid == WN_PRAGMA_HC_SHARED_REMOVE)
        {
            Is_True(opr == OPR_PRAGMA, (""));

            // Retrieve the matching HC_GPU_DATA from the stack.
            ST_IDX st_idx = WN_st_idx(wn);
            sdata = stack->pop(HC_SHARED_DATA, st_idx);
            Is_True(sdata != NULL, (""));

            // Lower the directive.
            next_wn = HC_lower_shared_remove(wn, parent_wn, sdata, gen_code);
        }

        return next_wn;
    }

    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            kid_wn = HC_lower_shared_dir_walker(kid_wn, wn, node,
                    sdata_dir_id, stack);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            Is_True(HC_lower_shared_dir_walker(WN_kid(wn,i), wn, node,
                        sdata_dir_id, stack) == NULL, (""));
        }
    }

    return next_wn;
}

void HC_handle_shared_directives(IPA_NODE *node)
{
    // Do nothing if the procedure has no SHARED directive.
    if (!node->contains_shared_dir()) return;

    // TODO: use another pool
    MEM_POOL *tmp_pool = node->Mem_Pool();
    
    // We need a stack because the list of HC_GPU_DATA in the node only
    // accounts for COPYIN directives.
    HC_GPU_DATA_STACK *stack = CXX_NEW(HC_GPU_DATA_STACK(tmp_pool), tmp_pool);

    IPA_NODE_CONTEXT context(node);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "In node <%s> ...\n", node->Name());
    }

    UINT sdata_dir_id = 0;
    HC_lower_shared_dir_walker(node->Whirl_Tree(), NULL,
            node, sdata_dir_id, stack);
    Is_True(sdata_dir_id == node->get_shared_data_list()->Elements(), (""));

    // Manual cleanup.
    CXX_DELETE(stack, tmp_pool);

    // Rebuild the Parent_Map.
    WN_Parentize(node->Whirl_Tree(), Parent_Map, Current_Map_Tab);

    // Verify the WHIRL tree.
    Is_True(WN_verifier(node->Whirl_Tree()), (""));

    // There is no need to reset the WN-to-EDGE map.
    // IPA_Call_Graph->Reset_Callsite_Map(node);
}

#endif  // IPA_HICUDA

/*** DAVID CODE END ***/
