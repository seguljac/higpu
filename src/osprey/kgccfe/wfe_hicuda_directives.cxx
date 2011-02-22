/** DAVID CODE BEGIN **/
#ifdef HICUDA

#include <stdio.h>
#include <assert.h>

#include "gnu_config.h"
#include "system.h"
#include "gnu/tree.h"

#include "wn.h"
#include "wn_util.h"
#include "wfe_misc.h"
#include "wfe_expr.h"
#include "wfe_stmt.h"

#include "wfe_hicuda_directives.h"
#include "hicuda_types.h"

// Definition of WN representation of hiCUDA directives
#include "hc_directives.h"      // in "hicuda" directory
#include "hc_common.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Create a hiCUDA REGION node.
 *
 * The REGION node is first pushed onto the stack, with type
 * wfe_stmk_hc_region.
 * Its body BLOCK node is then pushed onto the stack, with type
 * wfe_stmk_hc_region_body.
 * The pragma BLOCK node is then pushed onto the stack, with type
 * wfe_stmk_region_pragmas.
 *
 * We push the REGION node onto the stack for the purpose of error handling.
 * We WFE_Stmt_Pop finds an unexpected stmt kind wfe_stmk_hc_region_body, then
 * we can use the lower REGION node on the stack to give more hiCUDA-specific
 * information, like missing "kernel_end" for which kernel.
 *
 * Return the created WN node for the region.
 *
 ****************************************************************************/

static WN* WFE_hicuda_region()
{
    WN *body_wn, *pragmas_wn, *region_wn;

    body_wn = WN_CreateBlock();
    pragmas_wn = WN_CreateBlock();
    region_wn = WN_CreateRegion(REGION_KIND_HICUDA, body_wn, pragmas_wn,
            WN_CreateBlock(), -1, 0);

    // Can we reuse Get_Srcpos()?

    // Append the region to the current top statement.
    WFE_Stmt_Append(region_wn, Get_Srcpos());

    // Push new blocks to the statement stack.
    WFE_Stmt_Push(region_wn, wfe_stmk_hc_region, Get_Srcpos());
    WFE_Stmt_Push(body_wn, wfe_stmk_hc_region_body, Get_Srcpos());
    WFE_Stmt_Push(pragmas_wn, wfe_stmk_region_pragmas, Get_Srcpos());

    return region_wn;
}

inline void WFE_end_hicuda_region()
{
    // Pop off both the body BLOCK node and the REGION node.
    WFE_Stmt_Pop(wfe_stmk_hc_region_body);
    WFE_Stmt_Pop(wfe_stmk_hc_region);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/**
 * Convenience methods for creating and identifying invalid unsigned ints.
 */
inline static WN* new_invalid_uint()
{
    return WN_Intconst(Integer_type, -1);
}

inline static bool is_uint_valid(WN *wn)
{
    assert(wn != NULL);
    return (WN_operator(wn) != OPR_INTCONST) || (WN_const_val(wn) != -1);
}

/**
 * Add the given list of index ranges to 'wn', an ARRSECTION node.
 *
 * If any of the starting index, ending index or the size is NULL, put an
 * INTCONST node of -1 in the corresponding place in 'wn'.
 */
static void wfe_expand_idx_range_list(WN *wn,
        struct idx_range_list *irl, UINT ndims)
{
    UINT i = 0;
    for ( ; irl != NULL; irl = irl->next)
    {
        ++i;

        // Kid i stores the size for the ith dimension.
        WN_kid(wn,i) = new_invalid_uint();

        // Kid (ndims+i) stores the index range for the ith dimension.
        WN *start_idx_wn = (irl->start_idx != NULL) ?
            WFE_Expand_Expr(irl->start_idx) : new_invalid_uint();
        WN *end_idx_wn = (irl->end_idx != NULL) ?
            WFE_Expand_Expr(irl->end_idx) : new_invalid_uint();
        // not sure if the stride is correct.
        WN_kid(wn,ndims+i) = WN_CreateTriplet(
                start_idx_wn, WN_Intconst(Integer_type, 1), end_idx_wn); 
    }

    Is_True(i == ndims, (""));
}

/*****************************************************************************
 *
 * If the variable symbol in <ar> is a scalar or a struct, verify that <ar>
 * does not have a section specification and convert <ar> to an LDA node.
 *
 * If the variable symbol in <ar> is a pointer or an array, it must be one of
 * the following three cases:
 *
 * 1) it is a static array, which can be multi-dimensional (e.g., int[8][4])
 * 2) it is a one-dimensional dynamic array (e.g., int*)
 * 3) it is a pointer to a static array (e.g., int (*)[8])
 *
 * This routine verifies that <ar> has a section specification (of at least
 * one dimension) and converts <ar> to an ARRSECTION node, in which each
 * TRIPLET has the format (start_idx, 1, end_idx).
 *
 * TODO: what if the variable symbol has an offset?
 *
 ****************************************************************************/

static WN* wfe_expand_arr_region(struct arr_region *ar)
{
    if (ar == NULL) return NULL;

    // Retrieve the variable symbol.
    // <ar>->var is assumed to be a VAR_DECL.
    ST_IDX var_st_idx = WN_st_idx(WFE_Expand_Expr(ar->var));
    ST& var_st = St_Table[var_st_idx];

    // Mark this variable hiCUDA host data
    // so that it will not be copy-propagated in WOPT.
    set_st_attr_is_hicuda_data(var_st_idx);

    // Get the variable type.
    TY_IDX var_ty_idx = ST_type(var_st);
    TY_KIND var_ty_kind = TY_kind(var_ty_idx);

    WN *wn = NULL;
    SRCPOS srcpos = Get_Srcpos();

    if (var_ty_kind == KIND_SCALAR || var_ty_kind == KIND_STRUCT)
    {
        // There must not be a section specification.
        HC_assert(ar->irl == NULL,
                ("Non-array variable <%s> cannot have a section "
                 "specification (line %d of file %d)!",
                 ST_name(var_st_idx),
                 SRCPOS_linenum(srcpos), SRCPOS_filenum(srcpos)));

        // Create an LDA node.
        wn = WN_LdaZeroOffset(var_st_idx);
    }
    else
    {
        // The variable type must be a pointer or an array.
        HC_assert(var_ty_kind == KIND_POINTER || var_ty_kind == KIND_ARRAY,
                ("Variable <%s> has an invalid type (line %d of file %d)!",
                 ST_name(var_st_idx),
                 SRCPOS_linenum(srcpos), SRCPOS_filenum(srcpos)));

        // There must be a section specification.
        HC_assert(ar->irl != NULL,
                ("Missing section specification for array <%s> "
                 "(line %d of file %d)!",
                 ST_name(var_st_idx),
                 SRCPOS_linenum(srcpos), SRCPOS_filenum(srcpos)));

        // Count the number of index ranges.
        INT n = 0;
        struct idx_range_list *curr = ar->irl;
        while (curr != NULL) { ++n; curr = curr->next; }

        // For an array or a pointer-to-array, do a sanity check on
        // its dimensionality.
        TY_IDX ty_idx = (var_ty_kind == KIND_POINTER) ?
            TY_pointed(var_ty_idx) : var_ty_idx;
        if (TY_kind(ty_idx) == KIND_ARRAY)
        {
            // Determine its dimensionality and element type.
            UINT16 ndims = 0;
            do
            {
                ndims += ARB_dimension(TY_arb(ty_idx));
                ty_idx = TY_etype(ty_idx);
            } while (TY_kind(ty_idx) == KIND_ARRAY);

            HC_assert(ndims == n,
                    ("Mismatch in dimensionality of array <%s> "
                     "(line %d of file %d)!",
                     ST_name(var_st_idx),
                     SRCPOS_linenum(srcpos), SRCPOS_filenum(srcpos)));
        }
        // For all three cases, <ty_idx> now holds the element type.

        // Create the base address: an LDA for array, or an LDID otherwise.
        WN *base_addr_wn = (var_ty_kind == KIND_POINTER) ?
                WN_LdidScalar(var_st_idx) :
                WN_LdaZeroOffset(var_st_idx, Make_Pointer_Type(ty_idx));

        // Create an ARRSECTION node.
        wn = WN_CreateArrSection(base_addr_wn, n);
        WN_element_size(wn) = TY_size(ty_idx);
        wfe_expand_idx_range_list(wn, ar->irl, n);
    }

    return wn;
}

/*****************************************************************************
 *
 * Handle the following directives:
 * - GLOBAL ALLOC, GLOBAL COPYOUT
 * - SHARED ALLOC, SHARED COPYOUT
 *
 ****************************************************************************/

void wfe_expand_hc_global_shared(struct hicuda_global_shared *dir,
        bool is_global_dir)
{
    WN *wn = NULL;

    if (dir->decl != NULL)
    {
        /* This is an ALLOC directive. */

        ST_IDX var_st_idx = WN_st_idx(WFE_Expand_Expr(dir->decl->var));

        // Create the base pragma.
        WN_PRAGMA_ID pid = is_global_dir ?
            WN_PRAGMA_HC_GLOBAL_COPYIN : WN_PRAGMA_HC_SHARED_COPYIN;
        UINT flags = 0;
        if (dir->copy != NULL)
        {
            flags |= HC_DIR_COPYIN;
            if (!is_global_dir && dir->copy->nobndcheck)
            {
                flags |= HC_DIR_COPYIN_NOBNDCHECK;
            }
        }
        if (is_global_dir && dir->clear_region) flags |= HC_DIR_CLEAR;
        wn = WN_CreatePragma(pid, var_st_idx, flags, 0);
        WN_set_pragma_hicuda(wn);
        // Add it to the current block: pragma block.
        WFE_Stmt_Append(wn, Get_Srcpos());

        // Create an xpragma holding the ALLOC field.
        wn = WN_CreateXpragma(WN_PRAGMA_HC_DECL, ST_IDX_ZERO, 1);
        WN_kid0(wn) = wfe_expand_arr_region(dir->decl);
        WN_set_pragma_hicuda(wn);
        WFE_Stmt_Append(wn, Get_Srcpos());

        // Create an xpragma holding the COPYIN field if present.
        if (dir->copy != NULL && dir->copy->region != NULL)
        {
            wn = WN_CreateXpragma(WN_PRAGMA_HC_COPY, ST_IDX_ZERO, 1);
            WN_kid0(wn) = wfe_expand_arr_region(dir->copy->region);
            WN_set_pragma_hicuda(wn);
            WFE_Stmt_Append(wn, Get_Srcpos());
        }
    }
    else
    {
        /* This is a COPYOUT directive. */

        Is_True(dir->copy != NULL && dir->copy->region != NULL, (""));
        WN_PRAGMA_ID pid = is_global_dir ?
            WN_PRAGMA_HC_GLOBAL_COPYOUT : WN_PRAGMA_HC_SHARED_COPYOUT;
        bool nobndcheck = (!is_global_dir && dir->copy->nobndcheck);
        // We use the st_idx field to store nobndcheck flag.
        wn = WN_CreateXpragma(pid, (nobndcheck ? 1 : 0), 1);
        WN_kid0(wn) = wfe_expand_arr_region(dir->copy->region);
        WN_set_pragma_hicuda(wn);
        WFE_Stmt_Append(wn, Get_Srcpos());
    }
}

void wfe_expand_hc_const(struct hicuda_const *dir)
{
    struct arr_region *copyin_ar = dir->copyin;
    Is_True(copyin_ar != NULL, (""));

    // Create an XPRAGMA for each object copied in.
    while (copyin_ar != NULL)
    {
        WN *wn = WN_CreateXpragma(WN_PRAGMA_HC_CONST_COPYIN, ST_IDX_ZERO, 1);
        WN_kid0(wn) = wfe_expand_arr_region(copyin_ar);
        WN_set_pragma_hicuda(wn);
        WFE_Stmt_Append(wn, Get_Srcpos());

        copyin_ar = copyin_ar->next;
    }
}

static void wfe_expand_hc_data_free(struct free_data_list *dir,
        WN_PRAGMA_ID data_type)
{
    for ( ; dir != NULL; dir = dir->next)
    {
        // Retrieve the variable symbol.
        // var is assumed to be a VAR_DECL.
        ST_IDX var_st_idx = WN_st_idx(WFE_Expand_Expr(dir->var));

        // Mark this variable hiCUDA host data
        // so that it will not be copy-propagated in WOPT.
        set_st_attr_is_hicuda_data(var_st_idx);

        // Create a PRAGMA to hold the symbol.
        WN *wn = WN_CreatePragma(data_type, var_st_idx, 0, 0);
        WN_set_pragma_hicuda(wn);
        WFE_Stmt_Append(wn, Get_Srcpos());
    }
}

void wfe_expand_hc_global_free(struct free_data_list *dir)
{
    wfe_expand_hc_data_free(dir, WN_PRAGMA_HC_GLOBAL_FREE);
}

void wfe_expand_hc_shared_remove(struct free_data_list *dir)
{
    wfe_expand_hc_data_free(dir, WN_PRAGMA_HC_SHARED_REMOVE);
}

void wfe_expand_hc_const_remove(struct free_data_list *dir)
{
    wfe_expand_hc_data_free(dir, WN_PRAGMA_HC_CONST_REMOVE);
}

void wfe_expand_hc_shape(struct hicuda_shape *dir)
{
    for ( ; dir != NULL; dir = dir->next)
    {
        SRCPOS srcpos = Get_Srcpos();

        // Retrieve the variable symbol.
        // var is assumed to be a VAR_DECL.
        ST_IDX var_st_idx = WN_st_idx(WFE_Expand_Expr(dir->var));
        ST& var_st = St_Table[var_st_idx];

        // The variable must be of a pointer type.
        TY_IDX var_ty_idx = ST_type(var_st);
        HC_assert(TY_kind(var_ty_idx) == KIND_POINTER,
                ("SHAPE directive at line %d of file %d: "
                 "variable <%s> is not a pointer!",
                 SRCPOS_linenum(srcpos), SRCPOS_filenum(srcpos),
                 ST_name(var_st)));

        // Mark this variable hiCUDA host data
        // so that it will not be copy-propagated in WOPT.
        set_st_attr_is_hicuda_data(var_st_idx);

        TY_IDX elem_ty_idx = TY_pointed(var_ty_idx);

        // Determine the dimensionality of the shape.
        UINT ndims = 0;
        struct dim_sz_list *dsl = dir->shape;
        while (dsl != NULL) { ++ndims; dsl = dsl->next; }

        // Create an ARRSECTION node.
        WN *shape_wn = WN_CreateArrSection(WN_LdidScalar(var_st_idx), ndims);
        WN_element_size(shape_wn) = TY_size(elem_ty_idx);

        // Fill the array dimension kids of the ARRSECTION node.
        dsl = dir->shape;
        INT i = 0;
        while (dsl != NULL)
        {
            WN *dim_wn = WFE_Expand_Expr(dsl->dim_sz);

            // Make sure that it is either a constant or an integer symbol.
            OPERATOR opr = WN_operator(dim_wn);
            HC_assert(opr == OPR_INTCONST || opr == OPR_LDID,
                    ("SHAPE directive at line %d of file %d: "
                     "dimension #%d of array <%s> "
                     "must be a constant or a scalar variable!",
                     SRCPOS_linenum(srcpos), SRCPOS_filenum(srcpos),
                     i, ST_name(var_st)));
            if (opr == OPR_INTCONST)
            {
                HC_assert(WN_const_val(dim_wn) > 0,
                        ("SHAPE directive at line %d of file %d: "
                         "dimension #%d of array <%s> is not positive!",
                         SRCPOS_linenum(srcpos), SRCPOS_filenum(srcpos),
                         i, ST_name(var_st)));
            }
            else
            {
                // Check the variable type.
                TY_IDX ty_idx = ST_type(WN_st(dim_wn));
                HC_assert(TY_kind(ty_idx) == KIND_SCALAR
                        && WN_offset(dim_wn) == 0,
                        ("SHAPE directive at line %d of file %d: "
                         "dimension #%d of array <%s> "
                         "is not a scalar variable!",
                         SRCPOS_linenum(srcpos), SRCPOS_filenum(srcpos),
                         i, ST_name(var_st)));
                HC_assert(MTYPE_is_integral(TY_mtype(ty_idx)),
                        ("SHAPE directive at line %d of file %d: "
                         "dimension #%d of array <%s> "
                         "is not an integer variable!",
                         SRCPOS_linenum(srcpos), SRCPOS_filenum(srcpos),
                         i, ST_name(var_st)));
            }

            WN_array_dim(shape_wn,i) = WFE_Expand_Expr(dsl->dim_sz);

            // The index node is never used, it just makes the ARRSECTION a
            // valid node so that the existing passes will not complain.
            WN_array_index(shape_wn,i) = WN_CreateTriplet(
                    WN_Intconst(Integer_type,0),
                    WN_Intconst(Integer_type,0),
                    WN_Intconst(Integer_type,1));
            ++i;
            dsl = dsl->next;
        }

        // Create an XPRAGMA to hold the ARRSECTION.
        WN *wn = WN_CreateXpragma(WN_PRAGMA_HC_SHAPE, ST_IDX_ZERO, 1);
        WN_kid0(wn) = shape_wn;
        WN_set_pragma_hicuda(wn);

        WFE_Stmt_Append(wn, srcpos);
    }
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void
wfe_expand_hc_barrier() {
    // Create an emtpy PRAGMA.
    WN *wn = WN_CreatePragma(WN_PRAGMA_HC_BARRIER, ST_IDX_ZERO, 0, 0);
    WN_set_pragma_hicuda(wn);
    WFE_Stmt_Append(wn, Get_Srcpos());
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

static void
wfe_add_virtual_space_pragmas(WN_PRAGMA_ID id,
        struct virtual_space *vs) {
    WN *wn = NULL;

    while (vs != NULL) {
        // dim_size could be any expression
        wn = WN_CreateXpragma(id, ST_IDX_ZERO, 1);
        WN_kid0(wn) = WFE_Expand_Expr(vs->dim_size);
        // Get_Integer_Value(vs->dim_size)

        WN_set_pragma_hicuda(wn);
        WFE_Stmt_Append(wn, Get_Srcpos());

        vs = vs->next;
    }
}

void wfe_expand_hc_kernel_begin(struct hicuda_kernel *dir)
{
    // Create a region in the current block.
    WN *region = WFE_hicuda_region();

    // Get the kernel function name as a string.
    const char *kfunc_str = IDENTIFIER_POINTER(dir->kname);
    assert(kfunc_str != NULL);

    // Store it in the string table.
    bool is_new;
    STR_IDX kfunc_str_idx = Save_Str(kfunc_str, &is_new);
    HC_assert(is_new, ("Duplicate kernel name <%s>!", kfunc_str));

    // Build a CLASS_NAME symbol for this string.
    ST *kfunc_st = New_ST(GLOBAL_SYMTAB);
    ST_Init(kfunc_st, kfunc_str_idx, CLASS_NAME, SCLASS_UNKNOWN,
        EXPORT_LOCAL, 0);

    // Count the dimensionality of the block and the thread space.
    int n_blk_dims = 0, n_thr_dims = 0;
    struct virtual_space *curr = dir->block;
    while (curr != NULL) { ++n_blk_dims; curr = curr->next; }
    curr = dir->thread;
    while (curr != NULL) { ++n_thr_dims; curr = curr->next; }

    // Create the base pragma, which stores
    // 1) the kernel function symbol,
    // 2) the dimensionality of the block space, and
    // 3) the dimensionality of the thread space
    WN *wn = WN_CreatePragma(WN_PRAGMA_HC_KERNEL,
        ST_st_idx(kfunc_st), n_blk_dims, n_thr_dims);
    WN_set_pragma_hicuda(wn);
    // Add it to the current block: pragma block.
    WFE_Stmt_Append(wn, Get_Srcpos());
    
    wfe_add_virtual_space_pragmas(WN_PRAGMA_HC_KERNEL_BLK_SPACE, dir->block);
    wfe_add_virtual_space_pragmas(WN_PRAGMA_HC_KERNEL_THR_SPACE, dir->thread);

    // We are done with the pragma block. Move on to the body block.
    WFE_Stmt_Pop(wfe_stmk_region_pragmas);
}

void wfe_expand_hc_kernel_end()
{
    WFE_end_hicuda_region();
}

void wfe_expand_hc_kernel_part_begin(struct hicuda_kernel_part *dir)
{
    // Create a region in the current block.
    WN *region = WFE_hicuda_region();

    // Create the base pragma, which stores the distribution types of
    // the block space and the thread space.
    WN *wn = WN_CreatePragma(WN_PRAGMA_HC_KERNEL_PART,
        ST_IDX_ZERO, dir->block, dir->thread);
    WN_set_pragma_hicuda(wn);
    // Add it to the current block: pragma block.
    WFE_Stmt_Append(wn, Get_Srcpos());
 
    // We are done with the pragma block. Move on to the body block.
    WFE_Stmt_Pop(wfe_stmk_region_pragmas);
}

void wfe_expand_hc_kernel_part_end()
{
    WFE_end_hicuda_region();
}

void wfe_expand_hc_loopblock_begin(struct hicuda_loopblock *dir)
{
    // Create a region in the current block.
    WN *region = WFE_hicuda_region();

    // Create the base XPRAGMA, which contains the tile size node.
    WN *wn = WN_CreateXpragma(WN_PRAGMA_HC_LOOPBLOCK, ST_IDX_ZERO, 1);
    WN_kid0(wn) = WFE_Expand_Expr(dir->tile_sz);
    WN_set_pragma_hicuda(wn);
    WFE_Stmt_Append(wn, Get_Srcpos());

    // Create an XPRAGMA for each COPYIN array region.
    struct arr_region *copy_ar = dir->copyin;
    while (copy_ar != NULL) {
        wn = WN_CreateXpragma(WN_PRAGMA_HC_SHARED_COPYIN,
            ST_IDX_ZERO, 1);
        WN_kid0(wn) = wfe_expand_arr_region(copy_ar);
        WN_set_pragma_hicuda(wn);
        // Add it to the current block: pragma block of the new region.
        WFE_Stmt_Append(wn, Get_Srcpos());

        copy_ar = copy_ar->next;
    }

    // We are done with the pragma block. Move on to the body block.
    WFE_Stmt_Pop(wfe_stmk_region_pragmas);
}

void wfe_expand_hc_loopblock_end()
{ 
    WFE_end_hicuda_region();
}

#endif  // HICUDA
/*** DAVID CODE END ***/
