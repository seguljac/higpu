/** DAVID CODE BEGIN **/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#include "wn.h"
#include "wn_util.h"
#include "strtab.h"
#include "pu_info.h"
#include "ir_bread.h"
#include "ir_bwrite.h"

#include "driver.h"
#include "hc_handlers.h"
#include "hc_stack.h"
#include "hc_utils.h"
#include "hc_subscript.h"
#include "cuda_utils.h"

#include "cmp_symtab.h"
#include "hc_cfg.h"
#include "hc_dfa.h"
#include "hc_livevar.h"

// #define TEST

#ifdef TEST
#include "optimizer.h"
#include "opt_du.h"
#endif

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/* This is copied from hicuda_types.h. */

// The distribution strategy for kernel partitioning
enum kernel_part_distr_type {
    HC_KERNEL_PART_BLOCK,
    HC_KERNEL_PART_CYCLIC,
    HC_KERNEL_PART_NONE
};

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/**
 * Convenience methods for creating and identifying invalid unsigned ints.
 */
inline static WN* new_invalid_uint() {
    return WN_Intconst(Integer_type, -1);
}

inline static bool is_uint_valid(WN *wn) {
    assert(wn != NULL);
    return (WN_operator(wn) != OPR_INTCONST) || (WN_const_val(wn) != -1);
}

/**
 * Check if the index range 'triplet' is full, i.e. identical to the
 * array dimension range.
 *
 * This method does not do fancy evaluation of the index values.
 *
 * Return true if 'triplet' is full and false otherwise.
 */
static bool
compare_dim_range(const ARB_HANDLE &ah, const WN *triplet) {
    assert(triplet != NULL && WN_operator(triplet) == OPR_TRIPLET);

    WN *lbnd_wn = WN_kid0(triplet), *ubnd_wn = WN_kid2(triplet);

    // Compare the lower bound.
    if (ARB_const_lbnd(ah)) {
        if (WN_operator(lbnd_wn) != OPR_INTCONST
            || WN_const_val(lbnd_wn) != ARB_lbnd_val(ah)) {
            return false;
        }
    } else {
        if (WN_operator(lbnd_wn) != OPR_LDID
            || WN_st_idx(lbnd_wn) != ARB_lbnd_var(ah)) {
            return false;
        }
    }

    // Compare the upper bound.
    if (ARB_const_ubnd(ah)) {
        if (WN_operator(ubnd_wn) != OPR_INTCONST
            || WN_const_val(ubnd_wn) != ARB_ubnd_val(ah)) {
            return false;
        }
    } else {
        if (WN_operator(ubnd_wn) != OPR_LDID
            || WN_st_idx(ubnd_wn) != ARB_ubnd_var(ah)) {
            return false;
        }
    }

    return true;
}


/**
 * Assume 'ga' has the following information:
 * - 'gvar_info' with no dimension size kids
 *
 * 'var_ty_idx' could be an array type or a pointer type.
 *
 * This method produces the following:
 * - dimension sizes of 'gvar_info'
 * - var_dim_sz
 * - is_full_idx_range
 * - partial_dim_idx
 */
void
analyze_gvar_section(TY_IDX var_ty_idx, struct hc_gmem_alias *ga) {
    assert(ga != NULL && ga->gvar_info != NULL);

    INT ndims = WN_num_dim(ga->gvar_info);
    ga->var_dim_sz = (WN**)malloc(ndims * sizeof(WN*));
    ga->is_full_idx_range = (bool*)malloc(ndims * sizeof(bool));

    INT dim_idx = 0;

    if (TY_kind(var_ty_idx) == KIND_POINTER)
    {
        WN *triplet = WN_kid(ga->gvar_info,ndims+1);
        WN *gvar_dim_sz = NULL;

        // The first dimension's size must be provided.
        WN *var_dim_sz = WN_kid1(ga->gvar_info);
        assert(is_uint_valid(var_dim_sz));

        // Expand the compact form of specification.
        if (is_uint_valid(WN_kid2(triplet)))
        {
            // This is a regular range.
            // The end index is present, so should the start index.
            assert(is_uint_valid(WN_kid0(triplet)));

            gvar_dim_sz = idx_range_size(triplet);
            // Compare the gvar dim size to see if the range is full.
            ga->is_full_idx_range[0] =
                are_subscripts_equal(gvar_dim_sz, var_dim_sz);
        }
        else if (is_uint_valid(WN_kid0(triplet)))
        {
            // This is a single-point range.
            WN_DELETE_Tree(WN_kid2(triplet));
            WN_kid2(triplet) = WN_COPY_Tree(WN_kid0(triplet));

            gvar_dim_sz = WN_Intconst(Integer_type, 1);
            // Compare the gvar dim size to see if the range is full.
            ga->is_full_idx_range[0] = are_subscripts_equal(
                    gvar_dim_sz, var_dim_sz);
        }
        else
        {
            // This is a full range.
            WN_DELETE_Tree(WN_kid0(triplet));
            WN_kid0(triplet) = WN_Intconst(Integer_type, 0);
            WN_DELETE_Tree(WN_kid2(triplet));
            WN_kid2(triplet) = WN_Sub(Integer_type,
                    WN_COPY_Tree(var_dim_sz), WN_Intconst(Integer_type, 1));

            gvar_dim_sz = WN_COPY_Tree(var_dim_sz);
            ga->is_full_idx_range[0] = true;
        }

        ga->var_dim_sz[0] = var_dim_sz;
        WN_kid1(ga->gvar_info) = gvar_dim_sz;

        // Now we will work on the 2nd idx range.
        dim_idx++;
        var_ty_idx = TY_pointed(var_ty_idx);
    }
    
    TY_IDX ty_idx = var_ty_idx;
    while (TY_kind(ty_idx) == KIND_ARRAY) {
        ARB_IDX arb_idx = Ty_Table[ty_idx].Arb();
        UINT16 dim = ARB_dimension(ARB_HANDLE(arb_idx));

        for (UINT16 i = 0; i < dim; ++i) {
            ARB_HANDLE ah = ARB_HANDLE(arb_idx + i);

            WN *triplet = WN_kid(ga->gvar_info, ndims+dim_idx+1);
            // Get the original array's dimension size.
            WN *var_dim_sz = array_dim_size(ah);
            WN *gvar_dim_sz = NULL;

            // Expand the compact form of specification.
            if (is_uint_valid(WN_kid2(triplet)))
            {
                // This is a regular range.
                // The end index is present, so should the start index.
                assert(is_uint_valid(WN_kid0(triplet)));

                gvar_dim_sz = idx_range_size(triplet);
                ga->is_full_idx_range[dim_idx] =
                    compare_dim_range(ah, triplet);
            }
            else if (is_uint_valid(WN_kid0(triplet)))
            {
                // This is a single-point range.
                WN_DELETE_Tree(WN_kid2(triplet));
                WN_kid2(triplet) = WN_COPY_Tree(WN_kid0(triplet));

                gvar_dim_sz = WN_Intconst(Integer_type, 1);
                ga->is_full_idx_range[dim_idx] =
                    compare_dim_range(ah, triplet);
            }
            else
            {
                // This is a full range.
                // Get start and end idx from the array type.
                WN_DELETE_Tree(WN_kid0(triplet));
                WN_kid0(triplet) = ARB_const_lbnd(ah) ?
                    WN_Intconst(Integer_type, ARB_lbnd_val(ah)) :
                    WN_LdidScalar(ARB_lbnd_var(ah));
                WN_DELETE_Tree(WN_kid2(triplet));
                WN_kid2(triplet) = ARB_const_ubnd(ah) ?
                    WN_Intconst(Integer_type, ARB_ubnd_val(ah)) :
                    WN_LdidScalar(ARB_ubnd_var(ah));

                gvar_dim_sz = WN_COPY_Tree(var_dim_sz);
                ga->is_full_idx_range[0] = true;
            }

            ga->var_dim_sz[dim_idx] = var_dim_sz;
            WN_kid(ga->gvar_info, dim_idx+1) = gvar_dim_sz;

            ++dim_idx;
        }

        ty_idx = TY_etype(ty_idx);
    }

    // Check from the right for "full" dimensions.
    int i = ndims - 1;
    while (i >= 0 && ga->is_full_idx_range[i]) --i;

    ga->partial_dim_idx = i + 1;
}

/**
 * Construct a BLOCK node that contains code for copying between the host
 * and global memory, for a global or a const variable.
 *
 * If 'copyin' is true, the direction is from host to global, and
 * the other way if false.
 */
static WN* make_gvar_transfer_code(struct hc_gmem_alias *ga,
        WN *copy, bool copyin)
{
    Is_True(copy == NULL,
            ("Partial copy in a GLOBAL directive is not supported yet.\n"));

    ST_IDX var_st_idx = ga->ori_st_idx;
    ST_IDX gvar_st_idx = ga->gvar_st_idx;

    /* Is this a global variable or a const variable? */
    bool is_global_var = st_attr_is_global_var(gvar_st_idx);

    TY_IDX var_ty_idx = ST_type(var_st_idx);
    TY_KIND var_ty_kind = TY_kind(var_ty_idx);
    // For constant variable, the type of the gmem variable is not the type
    // of 'gvar_st_idx'. It must be determined explicitly.

    // Create an empty block.
    WN *memcpy_blk = WN_CreateBlock();

    /* Handle the non-array case first. */
    if (ga->gvar_info == NULL) {
        assert(var_ty_kind == KIND_SCALAR || var_ty_kind == KIND_STRUCT);

        WN *tcall = NULL;

        // Type of the original variable is: the object type
        WN *var_wn = WN_LdaZeroOffset(var_st_idx);

        if (is_global_var) {
            // Type of the global variable is a pointer to the object.
            // assert(TY_kind(gvar_ty_idx) == KIND_POINTER);
            WN *gvar_wn = WN_LdidScalar(gvar_st_idx);

            WN *src_wn = NULL, *dst_wn = NULL;
            if (copyin) {
                src_wn = var_wn; dst_wn = gvar_wn;
            } else {
                src_wn = gvar_wn; dst_wn = var_wn;
            }

            tcall = call_cudaMemcpy(dst_wn, src_wn, WN_COPY_Tree(ga->gvar_sz),
                (copyin ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));
        } else {
            // Here, we only have "copy in".
            // Type of the constant variable is the object type.
            tcall = call_cudaMemcpyToSymbol(gvar_st_idx, var_wn,
                WN_COPY_Tree(ga->gvar_sz), WN_Intconst(Integer_type, 0),
                cudaMemcpyHostToDevice
            );

            // We need to modify the offset parameter of this call later, so
            // keep a pointer in the context.
            ga->init_point = tcall;
        }

        WN_INSERT_BlockFirst(memcpy_blk, tcall);
        return memcpy_blk;
    }

    /* Gather array information. */
    INT ndims = WN_num_dim(ga->gvar_info);
    assert(ndims >= 1);

    INT64 elem_sz = WN_element_size(ga->gvar_info);

    int pdim_idx = ga->partial_dim_idx;

    WN **var_dim_sz = ga->var_dim_sz;
    WN *gvar_dim_sz[ndims];
    for (INT i = 0; i < ndims; ++i) {
        gvar_dim_sz[i] = WN_kid(ga->gvar_info,i+1);
    }

    /* Handle the simple case: the copied region is continuguous, so we
     * need only one transfer call.
     */
    if (pdim_idx <= 1) {
        WN *tcall = NULL;

        /* Get the first element's address. */
        WN *var_base_wn = (var_ty_kind == KIND_ARRAY) ?
            WN_LdaZeroOffset(var_st_idx) : WN_LdidScalar(var_st_idx);
        WN *var_wn = HCWN_CreateArray(var_base_wn, ndims);
        for (INT i = 0; i < ndims; ++i) {
            // dimension size of the original array type
            WN_kid(var_wn,i+1) = WN_COPY_Tree(ga->var_dim_sz[i]);
            // index in this dimension
            WN_kid(var_wn,ndims+i+1) = WN_COPY_Tree(
                WN_kid0(WN_kid(ga->gvar_info,ndims+i+1)));
        }
        WN_element_size(var_wn) = elem_sz;

        if (is_global_var) {
            // Type of the global variable: a pointer to the element type
            // assert(TY_kind(gvar_ty_idx) == KIND_POINTER);
            WN *gvar_wn = WN_LdidScalar(gvar_st_idx);

            WN *src_wn = NULL, *dst_wn = NULL;
            if (copyin) {
                src_wn = var_wn; dst_wn = gvar_wn;
            } else {
                src_wn = gvar_wn; dst_wn = var_wn;
            }

            tcall = call_cudaMemcpy(dst_wn, src_wn, WN_COPY_Tree(ga->gvar_sz),
                (copyin ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));
        } else {
            // Type of the constant variable: the array type
            tcall = call_cudaMemcpyToSymbol(gvar_st_idx, var_wn,
                WN_COPY_Tree(ga->gvar_sz), WN_Intconst(Integer_type, 0),
                cudaMemcpyHostToDevice
            );

            // We need to modify the offset parameter of this call later, so
            // keep a pointer in the context.
            ga->init_point = tcall;
        }

        // Insert it into the block.
        WN_INSERT_BlockFirst(memcpy_blk, tcall);
        return memcpy_blk;
    }

    // Now it must be +ive, so convert the index into zero-based.
    --pdim_idx;

    /* Compute the stride size, i.e. the number of elements copied at once.
     * It is the product of all dimension sizes starting from the first
     * partial dimension (inclusive).
     */
    WN *stride_sz = WN_COPY_Tree(gvar_dim_sz[pdim_idx]);
    for (INT i = pdim_idx+1; i < ndims; ++i) {
        stride_sz = WN_Mpy(WN_rtype(stride_sz),
            stride_sz, WN_COPY_Tree(var_dim_sz[i]));
    }

    /* Create a local variable to hold the stride size. */

    // declaration
    ST_IDX ss_st_idx = new_local_var(
        gen_var_str(ST_IDX_ZERO, "batsz_"),
        MTYPE_To_TY(WN_rtype(stride_sz))
    );
    TY_IDX ss_ty_idx = ST_type(ss_st_idx);
    ST *ss_st = ST_ptr(ss_st_idx);
    // initialization
    WN_INSERT_BlockLast(memcpy_blk, WN_StidScalar(ss_st, stride_sz));
    // NOTE: stride_sz is not fresh anymore.

    /* Create a local variable to hold the offset for accessing the gmem
     * variable in strides. It starts from 0 and advance by stride_sz.
     */

    // declaration
    ST_IDX gvar_ofst_st_idx = make_loop_idx(pdim_idx);
    TY_IDX gvar_ofst_ty_idx = ST_type(gvar_ofst_st_idx);
    ST *gvar_ofst_st = ST_ptr(gvar_ofst_st_idx);
    // initialization: 0
    WN_INSERT_BlockLast(memcpy_blk,
        WN_StidScalar(gvar_ofst_st,
            WN_Intconst(TY_mtype(gvar_ofst_ty_idx), 0))
    );
    // update statement: gvar_ofst = gvar_ofst + stride_sz
    WN *gvar_ofst_update_wn = WN_StidScalar(gvar_ofst_st,
        WN_Add(TY_mtype(gvar_ofst_ty_idx),
            WN_LdidScalar(gvar_ofst_st_idx),
            WN_LdidScalar(ss_st_idx)
        )
    );

    /* Generate a nest of DO_LOOPs. */

    WN *parent_loop = NULL;
    ST_IDX idxvs[pdim_idx];
    for (int i = 0; i < pdim_idx; ++i) {
        WN *triplet = WN_kid(ga->gvar_info, ndims+i+1);

        idxvs[i] = make_loop_idx(i);
        // The loop iteration range is the index range of the dimension.
        WN *loop = make_empty_doloop(idxvs[i],
            WN_kid0(triplet), WN_kid2(triplet));

        if (parent_loop == NULL) {
            // Insert it in memcpy_blk.
            WN_INSERT_BlockLast(memcpy_blk, loop);
        } else {
            // Insert it into the parent loop's body.
            WN_INSERT_BlockFirst(WN_kid(parent_loop,4), loop);
        }

        parent_loop = loop;
    }
    // Now, parent_loop stores the innermost loop.
    assert(parent_loop != NULL);
    WN *parent_blk = WN_kid(parent_loop,4);

    /* Construct the data transfer call:
     * - cudaMemcpy for global variable,
     * - cudaMemcpyToSymbol for constant variable.
     */

    /* Construct the access node for the original var:
     * &var[idx1][idx2]...[idx_(pdimIdx-1)][start_pdimIdx][0][0][0]...
     * This is common for global and constant variable.
     */
    WN *var_base_wn = (var_ty_kind == KIND_ARRAY) ?
        WN_LdaZeroOffset(var_st_idx) : WN_LdidScalar(var_st_idx);
    WN *var_access = HCWN_CreateArray(var_base_wn, ndims);
    for (INT i = 0; i < ndims; ++i) {
        // dimension size of the original array type
        WN_kid(var_access,i+1) = WN_COPY_Tree(var_dim_sz[i]);
        // index in this dimension
        WN_kid(var_access,ndims+i+1) = (i < pdim_idx) ?
            WN_LdidScalar(idxvs[i]) :
            WN_COPY_Tree(WN_kid0(WN_kid(ga->gvar_info,ndims+i+1)));
    }
    WN_element_size(var_access) = elem_sz;

    WN *tcall = NULL;

    // copy size: stride_sz * elem_sz
    WN *copy_sz = WN_Mpy(TY_mtype(ss_ty_idx),
        WN_LdidScalar(ss_st_idx),
        WN_Intconst(TY_mtype(ss_ty_idx), elem_sz)
    );

    if (is_global_var) {
        // Construct the gmem variable access: gvar + gvar_ofst
        WN *gvar_access = WN_Add(Pointer_type,
            WN_LdidScalar(gvar_st_idx),
            WN_Mpy(TY_mtype(gvar_ofst_ty_idx),
                WN_LdidScalar(gvar_ofst_st_idx),
                WN_Intconst(TY_mtype(gvar_ofst_ty_idx), elem_sz)
            )
        );

        WN *dst_wn = NULL, *src_wn = NULL;
        if (copyin) {
            dst_wn = gvar_access; src_wn = var_access;
        } else {
            dst_wn = var_access; src_wn = gvar_access;
        }

        // Create a cudaMemcpy call.
        tcall = call_cudaMemcpy(dst_wn, src_wn, copy_sz,
            (copyin ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost));

    } else {
        // gvar offset in bytes: gvar_ofst * elem_sz
        WN *gvar_ofst_in_bytes = WN_Mpy(TY_mtype(gvar_ofst_ty_idx),
            WN_LdidScalar(gvar_ofst_st_idx),
            WN_Intconst(TY_mtype(gvar_ofst_ty_idx), elem_sz)
        );

        // Create a cudaMemcpyToSymbol call.
        tcall = call_cudaMemcpyToSymbol(gvar_st_idx, var_access,
            copy_sz, gvar_ofst_in_bytes, cudaMemcpyHostToDevice);

        // We need to modify the offset parameter of this call later, so
        // keep a pointer in the context.
        ga->init_point = tcall;
    }

    // Insert the transfer call in the parent loop.
    WN_INSERT_BlockFirst(parent_blk, tcall);

    // Add an update statement for gvar_ofst.
    WN_INSERT_BlockAfter(parent_blk, tcall, gvar_ofst_update_wn);

    return memcpy_blk;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

WN* lower_hc_global_copyin(WN *parent, WN *pragma)
{
    printf("Start lowering a GLOBAL COPYIN directive ...\n");

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);

    // Make sure that we are not inside a kernel function.
    Is_True(!kinfo.valid,
        ("GLOBAL COPYIN directive inside a kernel!\n"));

    // Save the insertion point. All new code will be inserted before it.
    WN *insert_pt = pragma;

    /* Parse the pragma. */

    // base pragma
    assert(pragma != NULL && WN_opcode(pragma) == OPC_PRAGMA);
    bool docopy = WN_pragma_arg1(pragma);
    assert(WN_pragma_arg2(pragma) == 0);

    // Get the declare field.
    pragma = WN_next(pragma);
    assert(pragma != NULL && WN_opcode(pragma) == OPC_XPRAGMA);
    WN *decl = WN_kid0(pragma);
    assert(decl != NULL);
    OPERATOR opr = WN_operator(decl);
    assert(opr == OPR_ARRSECTION || opr == OPR_LDA);
    // Remove the declare field from the pragma, for arrays.
    if (opr == OPR_ARRSECTION) WN_kid0(pragma) = NULL;

    // Get the variable symbol and type.
    ST_IDX var_st_idx = (opr == OPR_ARRSECTION) ?
        WN_st_idx(WN_kid0(decl)) : WN_st_idx(decl);
    assert(var_st_idx != ST_IDX_ZERO);
    TY_IDX var_ty_idx = ST_type(var_st_idx);

    // Get the copy field.
    WN *copy = NULL;
    pragma = WN_next(pragma);
    if (pragma != NULL && WN_pragma(pragma) == WN_PRAGMA_HC_COPY) {
        copy = WN_kid0(pragma);
        assert(copy != NULL);
        opr = WN_operator(copy);
        assert(opr == OPR_ARRSECTION || opr == OPR_LDA);

        // Make sure that decl and copy refer to the same variable.
        if (opr == OPR_ARRSECTION) {
            assert(var_st_idx == WN_st_idx(WN_kid0(copy)));
        } else {
            assert(var_st_idx == WN_st_idx(copy));
        }

        pragma = WN_next(pragma);
    }
    // Now, 'pragma' points to the next pragma to be processed.

    /* Start processing. */

    // We only handle array, scalar and struct.
    TY_KIND kind = TY_kind(var_ty_idx);
    assert(kind == KIND_SCALAR || kind == KIND_STRUCT
        || kind == KIND_ARRAY || kind == KIND_POINTER);

    struct hc_gmem_alias *ga = NULL;

    if (WN_operator(decl) == OPR_ARRSECTION) {
        // array or pointer (treated as array)
        assert(kind == KIND_ARRAY || kind == KIND_POINTER);

        /* Declare a new global variable that is a pointer to the
         * element type of the original array variable. */

        // Get the array dimensionality from the pragma.
        UINT ndims = WN_num_dim(decl);
        /* printf("dimensionality of array %s is %u\n",
            ST_name(var_st_idx), ndims); */

        // Do a sanity check on the array dimensionality, and
        // get the element type.
        TY_IDX elem_ty_idx = TY_IDX_ZERO;
        if (kind == KIND_ARRAY) {
            assert(ndims == num_array_dims(var_ty_idx));
            elem_ty_idx = arr_elem_ty(var_ty_idx);
        } else {
            TY_IDX pty_idx = TY_pointed(var_ty_idx);
            assert(ndims == num_array_dims(pty_idx) + 1);
            elem_ty_idx = arr_elem_ty(pty_idx);
        }

        // Get the array element size.
        INT64 elem_sz = TY_size(elem_ty_idx);
        assert(elem_sz == WN_element_size(decl));

        // Create the global variable symbol.
        TY_IDX gvar_ty_idx = Make_Pointer_Type(elem_ty_idx);
        ST_IDX gvar_st_idx = new_local_var(
            gen_var_str("g_", var_st_idx),
            gvar_ty_idx
        );
        // Identify the symbol as a CUDA global variable.
        set_st_attr_is_global_var(gvar_st_idx);

        // Add the (var, glob_var) pair to the stack.
        ga = add_gvar_alias(gvar_st_idx, var_st_idx, parent);
        assert(ga != NULL);
        ga->gvar_info = decl;       // decl is not aliased anywhere else

        // Fill the rest of the data structure.
        analyze_gvar_section(var_ty_idx, ga);

        // Calculate the total size of the global variable, i.e.
        // num_arr_elems * elem_size
        ga->gvar_sz = WN_Intconst(Integer_type, elem_sz);
        for (UINT i = 0; i < ndims; ++i) {
            ga->gvar_sz = WN_Mpy(Integer_type,
                ga->gvar_sz, WN_COPY_Tree(WN_kid(decl,i+1)));
        }
    } else {
        // scalar or struct
        assert(kind == KIND_SCALAR || kind == KIND_STRUCT);

        // We can never have partial copies here.
        assert(copy == NULL);

        /* Declare the global variable. */

        UINT64 var_sz = TY_size(var_ty_idx);

        // The type can be retrieved from the LDA node.
        TY_IDX gvar_ty_idx = WN_ty(decl);
        ST_IDX gvar_st_idx = new_local_var(
            gen_var_str("g_", var_st_idx),
            gvar_ty_idx
        );
        // Identify the symbol as a CUDA global variable.
        set_st_attr_is_global_var(gvar_st_idx);

        // Add the pair of variables to the stack top.
        ga = add_gvar_alias(gvar_st_idx, var_st_idx, parent);
        assert(ga != NULL);

        // Determine the global variable's size.
        ga->gvar_sz = WN_Intconst(Integer_type, var_sz);
    }

    /* Construct a call that allocates the global variable, and insert it
     * to the parent block.
     */
    ga->init_point = call_cudaMalloc(
        WN_LdaZeroOffset(ST_ptr(ga->gvar_st_idx)),
        WN_COPY_Tree(ga->gvar_sz)
    );
    WN_INSERT_BlockBefore(parent, insert_pt, ga->init_point);

    /* Generate the code for copying the data from the host memory to
     * the global memory.
     */

    if (docopy) {
        WN *memcpy_blk = make_gvar_transfer_code(ga, copy, true);
        // Insert the resulting BLOCK node after cudaMalloc.
        WN_INSERT_BlockBefore(parent, insert_pt, memcpy_blk);
    }

    // Remove all pragmas from the insertion point (the 1st pragma)
    // to the current one (exclusive).
    WN *curr = insert_pt, *next = NULL;
    while (curr != pragma) {
        next = WN_next(curr);
        WN_DELETE_FromBlock(parent, curr);
        curr = next;
    }

    printf("Finished lowering the GLOBAL COPYIN directive.\n");

    return pragma;
}

WN* lower_hc_global_copyout(WN *parent, WN *pragma)
{
    printf("Start lowering a GLOBAL COPYOUT directive ...\n");

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);

    // Make sure that we are outside a kernel function.
    Is_True(!kinfo.valid, ("GLOBAL COPYOUT directive inside a kernel!\n"));

    // Get the copyout field.
    assert(pragma != NULL && WN_opcode(pragma) == OPC_XPRAGMA);
    WN *copy = WN_kid0(pragma);
    assert(copy != NULL);
    OPERATOR opr = WN_operator(copy);
    assert(opr == OPR_ARRSECTION || opr == OPR_LDA);

    // Get the variable symbol.
    ST_IDX var_st_idx = (opr == OPR_ARRSECTION) ?
        WN_st_idx(WN_kid0(copy)) : WN_st_idx(copy);

    // Get info about the corresponding global variable.
    struct hc_gmem_alias *ga = visible_global_var(var_st_idx);
    Is_True(ga != NULL,
        ("GLOBAL COPYOUT: unmatched variable %s\n", ST_name(var_st_idx)));

    /* Generate the data transfer code, and insert it before the pragma
     * so that it will not be processed. */
    WN *memcpy_blk = make_gvar_transfer_code(ga, NULL, false);
    WN_INSERT_BlockBefore(parent, pragma, memcpy_blk);

    // Get the next node to be processed.
    WN *next_wn = WN_next(pragma);
    // Remove the pragma node.
    WN_DELETE_FromBlock(parent, pragma);

    printf("Finished lowering the GLOBAL COPYOUT directive.\n");

    return next_wn;
}

WN* lower_hc_global_free(WN *parent, WN *pragma)
{
    printf("Start lowering a GLOBAL FREE directive ...\n");

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);

    // Make sure that we are outside a kernel function.
    Is_True(!kinfo.valid,
        ("GLOBAL FREE directive inside a kernel!\n"));

    // Get the variable whose corresponding global variable
    // should be freed.
    assert(pragma != NULL && WN_opcode(pragma) == OPC_PRAGMA);
    ST_IDX var_st_idx = WN_st_idx(pragma);

    // Remove the corresponding global variable.
    struct hc_gmem_alias *ga = remove_gvar_alias(var_st_idx, parent);
    Is_True(ga != NULL,
        ("GLOBAL FREE: unmatched variable %s\n", ST_name(var_st_idx)));

    // Create a call to cudaFree and add it before the pragma.
    ST_IDX gvar_st_idx = ga->gvar_st_idx;
    TY_IDX gvar_ty_idx = ST_type(gvar_st_idx);
    WN_INSERT_BlockBefore(parent, pragma,
        call_cudaFree(WN_Ldid(Pointer_type, 0, gvar_st_idx, gvar_ty_idx)));

    // Free the data structure.
    free_gmem_alias(ga);

    // Get the next node to be processed.
    WN *next_wn = WN_next(pragma);
    // Remove the pragma node.
    WN_DELETE_FromBlock(parent, pragma);

    printf("Finished lowering the GLOBAL FREE directive.\n");

    return next_wn;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

WN* lower_hc_const_copyin(WN *parent, WN *pragma)
{
    printf("Start lowering a CONST COPYIN directive ...\n");

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);

    // Make sure that we are not inside a kernel function.
    Is_True(!kinfo.valid,
        ("CONST COPYIN directive inside a kernel!\n"));

    // Get the copyin field.
    assert(pragma != NULL && WN_opcode(pragma) == OPC_XPRAGMA);
    WN *copyin_wn = WN_kid0(pragma);
    assert(copyin_wn != NULL);
    OPERATOR opr = WN_operator(copyin_wn);
    assert(opr == OPR_ARRSECTION || opr == OPR_LDA);
    // If it is an array, detach it from the pragma.
    if (opr == OPR_ARRSECTION) WN_kid0(pragma) = NULL;

    // Get the variable symbol and type.
    ST_IDX var_st_idx = (opr == OPR_ARRSECTION) ?
        WN_st_idx(WN_kid0(copyin_wn)) : WN_st_idx(copyin_wn);
    assert(var_st_idx != ST_IDX_ZERO);
    TY_IDX var_ty_idx = ST_type(var_st_idx);

    // We only handle array, scalar and struct.
    TY_KIND kind = TY_kind(var_ty_idx);
    assert(kind == KIND_SCALAR || kind == KIND_STRUCT
        || kind == KIND_ARRAY || kind == KIND_POINTER);

    // Add the (var, const_var) pair to the stack.
    struct hc_gmem_alias *ga = add_cvar_alias(var_st_idx, parent);
    assert(ga != NULL);

    if (WN_operator(copyin_wn) == OPR_ARRSECTION) {
        assert(kind == KIND_ARRAY || kind == KIND_POINTER);

        // Get the array dimensionality from the pragma.
        UINT ndims = WN_num_dim(copyin_wn);

        // Do a sanity check on the array dimensionality, and
        // get the element type.
        TY_IDX elem_ty_idx = TY_IDX_ZERO;
        if (kind == KIND_ARRAY) {
            assert(ndims == num_array_dims(var_ty_idx));
            elem_ty_idx = arr_elem_ty(var_ty_idx);
        } else {
            TY_IDX pty_idx = TY_pointed(var_ty_idx);
            assert(ndims == num_array_dims(pty_idx) + 1);
            elem_ty_idx = arr_elem_ty(pty_idx);
        }

        // Get the array element size.
        UINT64 elem_sz = TY_size(elem_ty_idx);

        ga->gvar_info = copyin_wn;  // it is not aliased anywhere else

        // Fill the rest of the data structure.
        analyze_gvar_section(var_ty_idx, ga);

        // Calculate the total size (in bytes) of the const variable, i.e.
        // num_arr_elems * elem_size
        ga->gvar_sz = WN_Intconst(Integer_type, elem_sz);
        for (UINT i = 0; i < ndims; ++i) {
            ga->gvar_sz = WN_Mpy(Integer_type,
                ga->gvar_sz, WN_COPY_Tree(WN_kid(copyin_wn,i+1)));
        }

        // Make sure that the size is a constant.
        Is_True(WN_operator(ga->gvar_sz) == OPR_INTCONST,
            ("The constant variable for %s does not have a constant size!\n",
            ST_name(var_st_idx)));
    } else {
        // scalar or struct
        assert(WN_operator(copyin_wn) == OPR_LDA);

        // Determine the constant variable's size.
        ga->gvar_sz = WN_Intconst(Integer_type, TY_size(var_ty_idx));
    }

    // Record that this const variable is alive now.
    append_cvar_life(ga, true);

    /* Generate the code for copying the data from the host memory to
     * the constant memory.
     */
    WN *memcpy_blk = make_gvar_transfer_code(ga, NULL, true);
    WN_INSERT_BlockBefore(parent, pragma, memcpy_blk);

    // Get the next node to be processed.
    WN *next_wn = WN_next(pragma);
    // Remove the pragma node.
    WN_DELETE_FromBlock(parent, pragma);

    printf("Finished lowering the CONST COPYIN directive.\n");

    return next_wn;
}

WN* lower_hc_const_remove(WN *parent, WN *pragma)
{
    printf("Start lowering a CONST REMOVE directive ...\n");

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);

    // Make sure that we are inside a kernel function.
    Is_True(!kinfo.valid,
        ("CONST REMOVE directive inside a kernel!\n"));

    // Get the variable whose corresponding const variable should be freed.
    assert(pragma != NULL && WN_opcode(pragma) == OPC_PRAGMA);
    ST_IDX var_st_idx = WN_st_idx(pragma);

    // Remove the corresponding shared variable.
    struct hc_gmem_alias *ga = remove_cvar_alias(var_st_idx, parent);
    Is_True(ga != NULL,
        ("CONST REMOVE: unmatched variable %s\n", ST_name(var_st_idx)));

    // Record, in the PU context, that the live range of this const
    // variable has ended.
    append_cvar_life(ga, false);

    // Get the next node to be processed.
    WN *next_wn = WN_next(pragma);
    // Remove the pragma node.
    WN_DELETE_FromBlock(parent, pragma);

    printf("Finished lowering the CONST REMOVE directive.\n");

    return next_wn;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/**
 * Expand the compact form of specification in 'arr_section'.
 * TODO: merge with analyze_gvar_section.
 */
static void complete_svar_section(TY_IDX var_ty_idx, WN *arr_section)
{
    UINT ndims = WN_num_dim(arr_section);

    UINT dim_idx = 0;
    if (TY_kind(var_ty_idx) == KIND_POINTER)
    {
        WN *triplet = WN_kid(arr_section,ndims+1);

        // The first dimension size must be provided.
        WN *var_dim_sz = WN_kid1(arr_section);
        assert(is_uint_valid(var_dim_sz));

        // Expand the compact form of specification.
        if (is_uint_valid(WN_kid2(triplet)))
        {
            // This is a regular range.
            // The end index is present, so should the start index.
            assert(is_uint_valid(WN_kid0(triplet)));
        }
        else if (is_uint_valid(WN_kid0(triplet)))
        {
            // This is a single-point range.
            WN_DELETE_Tree(WN_kid2(triplet));
            WN_kid2(triplet) = WN_COPY_Tree(WN_kid0(triplet));
        }
        else
        {
            // This is a full range.
            WN_DELETE_Tree(WN_kid0(triplet));
            WN_kid0(triplet) = WN_Intconst(Integer_type, 0);
            WN_DELETE_Tree(WN_kid2(triplet));
            WN_kid2(triplet) = WN_Sub(Integer_type,
                    WN_COPY_Tree(var_dim_sz), WN_Intconst(Integer_type, 1));
        }

        dim_idx++;
        var_ty_idx = TY_pointed(var_ty_idx);
    }

    TY_IDX ty_idx = var_ty_idx;
    while (TY_kind(ty_idx) == KIND_ARRAY) {
        ARB_IDX arb_idx = Ty_Table[ty_idx].Arb();
        UINT16 dim = ARB_dimension(ARB_HANDLE(arb_idx));

        for (UINT16 i = 0; i < dim; ++i) {
            ARB_HANDLE ah = ARB_HANDLE(arb_idx + i);
            WN *triplet = WN_kid(arr_section, ndims+dim_idx+1);

            // Expand the compact form of specification.
            if (is_uint_valid(WN_kid2(triplet)))
            {
                // This is a regular range.
                // The end index is present, so should the start index.
                assert(is_uint_valid(WN_kid0(triplet)));
            }
            else if (is_uint_valid(WN_kid0(triplet)))
            {
                // This is a single-point range.
                WN_DELETE_Tree(WN_kid2(triplet));
                WN_kid2(triplet) = WN_COPY_Tree(WN_kid0(triplet));
            }
            else
            {
                // This is a full range.
                // Get start and end idx from the array type.
                WN_DELETE_Tree(WN_kid0(triplet));
                WN_kid0(triplet) = ARB_const_lbnd(ah) ?
                    WN_Intconst(Integer_type, ARB_lbnd_val(ah)) :
                    WN_LdidScalar(ARB_lbnd_var(ah));
                WN_DELETE_Tree(WN_kid2(triplet));
                WN_kid2(triplet) = ARB_const_ubnd(ah) ?
                    WN_Intconst(Integer_type, ARB_ubnd_val(ah)) :
                    WN_LdidScalar(ARB_ubnd_var(ah));
            }

            ++dim_idx;
        }

        ty_idx = TY_etype(ty_idx);
    }
}

/**
 * Merge the given array section w.r.t. the enclosing loop index variables
 * that are block/thread-partitioned.
 *
 * Assume the given array section has complete start and end index.
 *
 * Return a new ARRSECTION node that represents the merged section, that
 * should be allocated in the shared memory. The size of each dimension is
 * guaranteed to be a constant.
 */
static WN* analyze_svar_section(WN *arr_section)
{
    /* Find the loops we want to project the region on. */

    UINT nloops = num_enclosing_loops();
    struct loop_part_info* ls_arr[nloops];
    ST_IDX loop_idxvs[nloops];
    WN* loop_ranges[nloops];

    // number of DO_LOOPs that are block/thread-partitioned
    UINT ls_sz = filter_doloops(ls_arr, NULL, false, true, NULL);
    for (UINT i = 0; i < ls_sz; ++i) {
        loop_idxvs[i] = ls_arr[i]->idxv_st_idx;
        loop_ranges[i] = ls_arr[i]->blk_range;
    }

    /* Project the given region on these loop index variables. */
    WN *merged_arr_section = project_region(arr_section,
        loop_idxvs, loop_ranges, ls_sz);
    Is_True(merged_arr_section != NULL, ("%s\n", hc_subscript_errmsg));

    /* Make sure that the size of dimension is a positive integer constant. */

    UINT ndims = WN_num_dim(merged_arr_section);
    for (UINT i = 0; i < ndims; ++i) {
        WN *size_wn = WN_kid(merged_arr_section,i+1);

        if (size_wn == NULL || WN_operator(size_wn) != OPR_INTCONST
            || WN_const_val(size_wn) <= 0) {
            WN_DELETE_Tree(merged_arr_section);

            Fail_FmtAssertion(
                "The size of dimension %u of the merged array section for "
                "variable %s is not a positive integer constant.\n",
                i, ST_name(WN_st(WN_kid0(arr_section))));
        }
    }

    return merged_arr_section;
}

/**
 * Return a BLOCK node that contains the data transfer code.
 *
 * Initialization statements for kernel-invariant variables are inserted
 * at the beginning of the kernel.
 *
 * TODO: handle copyout_wn != NULL.
 */
static WN* make_svar_transfer_code(struct hc_smem_alias *sa,
        bool copyin, const WN *copyout_wn = NULL)
{
    assert(copyout_wn == NULL);
    // assert(copyin || copyout_wn != NULL);

    /* Prepare information. */

    ST_IDX var_st_idx = sa->ori_st_idx;
    ST_IDX svar_st_idx = sa->svar_st_idx;

    TY_IDX var_ty_idx = ST_type(var_st_idx);
    TY_KIND var_ty_kind = TY_kind(var_ty_idx);
    assert(var_ty_kind == KIND_ARRAY || var_ty_kind == KIND_POINTER);
    TY_IDX svar_ty_idx = ST_type(svar_st_idx);
    assert(TY_kind(svar_ty_idx) == KIND_POINTER);

    // dimensionality
    UINT16 ndims = WN_num_dim(sa->svar_info);

    // size of each dimension
    UINT32 dim_sz[ndims];
    for (UINT16 i = 0; i < ndims; ++i) {
        dim_sz[i] = (UINT32)WN_const_val(WN_kid(sa->svar_info,i+1));
    }

    // element type and size
    TY_IDX elem_ty_idx = arr_elem_ty(var_ty_kind == KIND_ARRAY ?
        var_ty_idx : TY_pointed(var_ty_idx));
    UINT64 elem_sz = TY_size(elem_ty_idx);

    /* Determine which thread loads which array element(s). */

    // dimension sizes of the thread space
    UINT blk_dims[3];
    for (INT i = 0; i < 3; ++i) {
        Is_True(WN_operator(kinfo.blk_dims[i]) == OPR_INTCONST,
                ("SMEM DATA TRANSFER: the size of dimension %u of the thread "
                 "space is not constant.\n", i+1));
        blk_dims[i] = WN_const_val(kinfo.blk_dims[i]);
    }

    // The total number of elements must be a multiple of the number of
    // threads in a block.
    UINT n_elems = dim_sz[0] * TY_size(TY_pointed(svar_ty_idx)) / elem_sz;
    UINT n_threads = blk_dims[0] * blk_dims[1] * blk_dims[2];
    Is_True(n_elems % n_threads == 0,
            ("SMEM DATA TRANSFER: the number of threads in a block (%u) "
             "does not divide the total number of elements (%u).\n",
             n_threads, n_elems));

    // The innermost array dimension size must be a multiple of 16.
    assert(dim_sz[ndims-1] % 16 == 0);

    WN *svar_access_comp[ndims];
    for (UINT16 i = 0; i < ndims; ++i) svar_access_comp[i] = NULL;

    // We ignore unit block dimensions from z.
    UINT blk_dim_idx_max = 2;
    while (blk_dims[blk_dim_idx_max] == 1) --blk_dim_idx_max;

    // Work from the innermost array dimension and the 1st block dimension.
    UINT blk_dim_idx = 0;
    UINT blk_dim_gone = 1, blk_dim_left = blk_dims[0];
    UINT dim_gone, dim_left;
    INT16 i;
    for (i = ndims-1; i >= 0; --i) {
        WN *wn = WN_Intconst(MTYPE_U4, 0), *comp_wn = NULL;

        dim_gone = 1, dim_left = dim_sz[i];
        while (dim_left > 1 && blk_dim_idx <= blk_dim_idx_max) {
            if (dim_left < blk_dim_left) {
                assert(blk_dim_left % dim_left == 0);

                // ((threadIdx / blk_dim_gone) % dim_left) * dim_gone
                comp_wn = WN_Mpy(MTYPE_U4,
                    WN_Binary(OPR_REM, MTYPE_U4,
                        WN_Div(MTYPE_U4,
                            ldid_threadIdx(blk_dim_idx),
                            WN_Intconst(MTYPE_U4, blk_dim_gone)
                        ),
                        WN_Intconst(MTYPE_U4, dim_left)
                    ),
                    WN_Intconst(MTYPE_U4, dim_gone)
                );

                blk_dim_gone *= dim_left; blk_dim_left /= dim_left;
                dim_gone *= dim_left; dim_left = 1;
            } else {
                assert(dim_left % blk_dim_left == 0);

                // (threadIdx / blk_dim_gone) * dim_gone
                comp_wn = WN_Mpy(MTYPE_U4,
                    WN_Div(MTYPE_U4,
                        ldid_threadIdx(blk_dim_idx),
                        WN_Intconst(MTYPE_U4, blk_dim_gone)
                    ),
                    WN_Intconst(MTYPE_U4, dim_gone)
                );

                dim_gone *= blk_dim_left; dim_left /= blk_dim_left;
                // move to the next block dimension
                ++blk_dim_idx;
                if (blk_dim_idx <= blk_dim_idx_max) {
                    blk_dim_gone = 1; blk_dim_left = blk_dims[blk_dim_idx];
                }
            }

            wn = WN_Add(MTYPE_U4, wn, comp_wn);
        }

        // `wn' will never be NULL here.
        svar_access_comp[i] = wn;

        // We cannot use blk_dim_idx >= 3 as the end condition because
        // dim_left == 1 and blk_dim_idx == 3 can happen at the same time
        // and we only care about if dim_left == 1.
        if (dim_left > 1) break;
    }

    // This index is important!!
    INT16 pdim_idx = i;

    /* Generate the code that loads array elements from the global memory
     * to the shared memory. */

    WN *code_blk = WN_CreateBlock();

    /* Create a variable for each invariant subscript of the LHS access.
     * The subscript is invariant over the whole kernel function, so its
     * initialization should be placed at the beginning of the kernel. */
    WN *ss_init_blk = WN_CreateBlock();

    ST_IDX svar_ssbase[ndims];
    for (UINT16 i = 0; i < ndims; ++i) {
        if ((INT16)i < pdim_idx) {
            svar_ssbase[i] = ST_IDX_ZERO;
        } else {
            svar_ssbase[i] = new_local_var(
                gen_var_str(ST_IDX_ZERO, "ssbase"),
                MTYPE_To_TY(MTYPE_U4)
            );

            WN_INSERT_BlockLast(ss_init_blk,
                WN_StidScalar(ST_ptr(svar_ssbase[i]), svar_access_comp[i]));

            // Now the subscript is just a load of the variable.
            svar_access_comp[i] = WN_LdidScalar(svar_ssbase[i]);
        }
    }

    /* Construct the invariant part of the RHS subscript. */

    // The shared variable must have a corresponding global variable.
    struct hc_gmem_alias *ga = sa->ga;
    assert(ga != NULL);

    // Get the starting index of each dimension of the global and shared vars.
    WN* shared_start_idx[ndims];
    WN* global_start_idx[ndims];
    for (UINT16 i = 0; i < ndims; ++i) {
        shared_start_idx[i] = WN_COPY_Tree(
            WN_kid0(WN_kid(sa->svar_info,ndims+i+1)));
        global_start_idx[i] = WN_COPY_Tree(
            WN_kid0(WN_kid(ga->gvar_info,ndims+i+1)));
    }

    // Construct the constant part of the global variable subscript.
    WN *ss_const_wn = WN_Sub(MTYPE_U4,
        shared_start_idx[0],
        global_start_idx[0]
    );
    for (UINT16 i = 1; i < ndims; ++i) {
        ss_const_wn = WN_Add(MTYPE_U4,
            WN_Mpy(MTYPE_U4,
                ss_const_wn,
                WN_COPY_Tree(WN_kid(ga->gvar_info,i+1))
            ),
            WN_Sub(MTYPE_U4,
                shared_start_idx[i],
                global_start_idx[i]
            )
        );
    }

    // Create a variable to hold the constant part.
    // This initialization statement may not be kernel-invariant, so should be
    // put in code_blk.
    ST_IDX ss_const_st_idx = new_local_var(
        gen_var_str(ST_IDX_ZERO, "ssbase"),
        MTYPE_To_TY(MTYPE_U4)
    );
    WN_INSERT_BlockLast(code_blk,
        WN_StidScalar(ST_ptr(ss_const_st_idx), ss_const_wn));

    /* We may need a nest of DO_LOOP's around the store statement. */

    WN *parent_blk = code_blk;
    if (pdim_idx >= 0) {
        // cached zero constant
        WN *int_zero_wn = WN_Intconst(MTYPE_U4, 0);

        for (INT16 i = 0; i <= pdim_idx; ++i) {
            ST_IDX idxv_st_idx = make_loop_idx(i, true);

            // All loops except the innermost one iterate over the full range
            // of the corresponding array dimension.
            WN *end_wn = NULL, *step_wn = NULL;
            if (i < pdim_idx) {
                end_wn = WN_Intconst(MTYPE_U4, dim_sz[i]-1);
            } else {
                end_wn = WN_Intconst(MTYPE_U4, (dim_left-1)*dim_gone);
                step_wn = WN_Intconst(MTYPE_U4, dim_gone);
            }
            WN *loop = make_empty_doloop(idxv_st_idx,
                int_zero_wn, end_wn, step_wn);
            WN_DELETE_Tree(end_wn);
            if (step_wn != NULL) WN_DELETE_Tree(step_wn);

            // Add it to the parent block.
            WN_INSERT_BlockLast(parent_blk, loop);
            parent_blk = WN_kid(loop,4);

            // Complete svar_access_comp[i].
            if (svar_access_comp[i] == NULL) {
                svar_access_comp[i] = WN_LdidScalar(idxv_st_idx);
            } else {
                // svar_access_comp[i] (svar_ssbase[i]) + idxv_st_idx
                svar_access_comp[i] = WN_Add(MTYPE_U4,
                    svar_access_comp[i],
                    WN_LdidScalar(idxv_st_idx)
                );
            }
        }

        WN_DELETE_Tree(int_zero_wn);
    }

    // Construct the access to the shared variable.
    WN *svar_access = WN_Create(OPR_ARRAY, Pointer_type, MTYPE_V, 2*ndims+1);
    // WN_kid0(svar_access) = WN_LdaZeroOffset(svar_st_idx);
    WN_kid0(svar_access) = WN_LdidScalar(svar_st_idx);
    for (UINT16 i = 0; i < ndims; ++i) {
        // dimension size
        WN_kid(svar_access,i+1) = WN_Intconst(MTYPE_U4, dim_sz[i]);
        // subscript in this dimension
        WN_kid(svar_access,ndims+i+1) = svar_access_comp[i];
    }
    WN_element_size(svar_access) = elem_sz;

    /* Construct the access to the global variable. 
     *
     * The access is: glob_var[(shared_start_i - glob_start_i) + i]...
     * in the expanded form. We will extract the constant term first. */

    // Construct the variable part of the global variable subscript.
    WN *ss_var = WN_COPY_Tree(svar_access_comp[0]);
    for (UINT16 i = 1; i < ndims; ++i) {
        ss_var = WN_Add(MTYPE_U4,
            WN_Mpy(MTYPE_U4, ss_var, WN_COPY_Tree(WN_kid(ga->gvar_info,i+1))),
            WN_COPY_Tree(svar_access_comp[i])
        );
    }

    // Construct the RHS: global variable access.
    WN *offset = WN_Mpy(MTYPE_U4,
        WN_Add(MTYPE_U4, ss_var, WN_LdidScalar(ss_const_st_idx)),
        WN_Intconst(MTYPE_U4, elem_sz)
    );
    WN *gvar_access = WN_Add(Pointer_type,
        WN_LdidScalar(ga->gvar_st_idx),
        offset
    );
    
    // Determine which access goes to which side of the store statement
    // based on the copy direction.
    WN *lhs = NULL, *rhs = NULL;
    if (copyin) {
        // LHS is the shared var access; RHS is the global var access.
        lhs = svar_access;
        rhs = WN_Iload(TY_mtype(elem_ty_idx), 0, elem_ty_idx, gvar_access);
    } else {
        // LHS is the global var access; RHS is the shared var access.
        lhs = gvar_access;
        rhs = WN_Iload(TY_mtype(elem_ty_idx), 0, elem_ty_idx, svar_access);
    }

    // Construct and insert the ISTORE node in the original code block.
    WN_INSERT_BlockLast(parent_blk,
        WN_Istore(TY_mtype(elem_ty_idx), 0,
            Make_Pointer_Type(elem_ty_idx),
            lhs,    // address
            rhs     // value
        )
    );

    // Insert the initialization block to the beginning of the kernel.
    WN_INSERT_BlockFirst(kinfo.kernel_body, ss_init_blk);

    return code_blk;
}

/**
 * Put the code block that should be inserted into 'parent' in
 * 'svar_code_blk_ptr'. It is a fresh instance of BLOCK.
 *
 * Return the next pragma to be processed in the pragma block.
 */
static WN* lower_hc_shared_copyin(WN *parent, WN *pragma,
        WN **svar_code_blk_ptr)
{
    printf("Start lowering a SHARED COPYIN directive ...\n");

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);
    assert(svar_code_blk_ptr != NULL);

    // Make sure that we are inside a kernel function.
    Is_True(kinfo.valid, ("SHARED COPYIN directive outside a kernel!\n"));

    // Get the copy field.
    assert(pragma != NULL && WN_opcode(pragma) == OPC_XPRAGMA);
    WN *copy_wn = WN_kid0(pragma);
    assert(copy_wn != NULL);
    WN_kid0(pragma) = NULL;

    // Let 'pragma' point to the next pragma to be processed.
    pragma = WN_next(pragma);

    // We only work with array sections.
    assert(WN_operator(copy_wn) == OPR_ARRSECTION);

    /* Collect information about the shared variable and do sanity checks. */

    ST_IDX var_st_idx = WN_st_idx(WN_kid0(copy_wn));
    assert(var_st_idx != ST_IDX_ZERO);
    TY_IDX var_ty_idx = ST_type(var_st_idx);
    TY_KIND var_ty_kind = TY_kind(var_ty_idx);

    // Get the array dimensionality from the pragma.
    UINT ndims = WN_num_dim(copy_wn);
    // Do a sanity check on the array dimensionality.
    if (var_ty_kind == KIND_ARRAY) {
        assert(ndims == num_array_dims(var_ty_idx));
    } else {
        TY_IDX pty_idx = TY_pointed(var_ty_idx);
        assert(ndims == num_array_dims(pty_idx) + 1);
    }

    /* Complete the ranges in the array section by consulting the array type.
     */
    complete_svar_section(var_ty_idx, copy_wn);

    /* The given array section represents what each thread wants to copy in.
     * We need to figure out the merged array section.
     */
    copy_wn = analyze_svar_section(copy_wn);
    // TODO: error message inside
    assert(copy_wn != NULL);

    // size of each dimension
    UINT32 dim_sz[ndims];
    for (UINT16 i = 0; i < ndims; ++i) {
        dim_sz[i] = (UINT32)WN_const_val(WN_kid(copy_wn,i+1));
    }

    // element type
    TY_IDX elem_ty_idx = arr_elem_ty(var_ty_kind == KIND_ARRAY ?
            var_ty_idx : TY_pointed(var_ty_idx));

    /* Declare the shared variable. */

    // If the shared variable is an array of int[5][6], then its type is
    // int (*A)[6], i.e. a pointer to an array type int[6].
    TY_IDX svar_ty_idx = Make_Pointer_Type(
            (ndims == 1) ? elem_ty_idx :
            make_arr_type(gen_var_str(var_st_idx, ".svar.type"),
                ndims-1, &dim_sz[1], elem_ty_idx)
            );

    ST_IDX svar_st_idx = new_local_var(
            gen_var_str("s_", var_st_idx), svar_ty_idx);

    // Set the SHARED attribute of this symbol.
    // set_st_attr_is_shared_var(svar_st_idx);

    /* Package information we need to generate the data transfer code. */

    struct hc_smem_alias *sa =
        add_svar_alias(svar_st_idx, var_st_idx, parent);
    WN_element_size(copy_wn) = TY_size(elem_ty_idx);
    sa->svar_info = copy_wn;
    sa->svar_size = dim_sz[0] * TY_size(TY_pointed(svar_ty_idx));

    // Record, in the kernel context, about the start of life of this shared
    // variable.
    append_svar_life(sa, true);

    /* Generate the code that transfers data from global memory to
     * the new shared variable. */

    WN *svar_load_blk = make_svar_transfer_code(sa, true);
    *svar_code_blk_ptr = svar_load_blk;

    /* Make a copy of the data transfer code and cache it.
     * It is intended not to include initialization of the shared variable
     * in this copy because the shared variable will never be overwritten.
     */
    sa->copyin_wn = WN_COPY_Tree(svar_load_blk);

    /* Create an STID node for initialization of the shared variable, which
     * is incomplete for now but will be fixed later.
     *
     * For now, I do not put this initialization in copyout, but this will
     * increase the shared variable's live range, so register pressure.
     */
    WN *svar_stid_wn = WN_StidScalar(ST_ptr(svar_st_idx), 
            WN_Intconst(Integer_type, 0));
    sa->init_point = svar_stid_wn;

    WN_INSERT_BlockFirst(svar_load_blk, svar_stid_wn);
 
    printf("Finished lowering the SHARED COPYIN directive.\n");

    return pragma;
}

WN* lower_hc_shared_copyin_list(WN *parent, WN *pragma)
{
    /* Consume as many SHARED COPYIN directives as possible. */
    WN *svar_code_blk = NULL;
    bool at_least_one_dir = false;

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);

    // Any new code will be inserted before 'pragma'.
    WN *insert_pt = pragma;

    // Process the directive one at a time.
    while (pragma != NULL
            && WN_operator(pragma) == OPR_XPRAGMA
            && WN_pragma(pragma) == WN_PRAGMA_HC_SHARED_COPYIN) {
        at_least_one_dir = true;

        svar_code_blk = NULL;
        pragma = lower_hc_shared_copyin(parent, pragma, &svar_code_blk);
        assert(svar_code_blk != NULL);

        // Insert the new code block.
        WN_INSERT_BlockBefore(parent, insert_pt, svar_code_blk);
    }

    // Remove all pragmas from the insertion point (the 1st pragma)
    // to the current one (exclusive).
    WN *curr = insert_pt, *next = NULL;
    while (curr != pragma) {
        next = WN_next(curr);
        WN_DELETE_FromBlock(parent, curr);
        curr = next;
    }

    return pragma;
}

WN* lower_hc_shared_copyout(WN *parent, WN *pragma)
{
    printf("Start lowering a SHARED COPYOUT directive ...\n");

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);

    // Make sure that we are inside a kernel function.
    Is_True(kinfo.valid, ("SHARED COPYOUT directive outside a kernel!\n"));

    assert(pragma != NULL && WN_opcode(pragma) == OPC_XPRAGMA);
    WN *copy_wn = WN_kid0(pragma);
    assert(copy_wn != NULL);
    OPERATOR opr = WN_operator(copy_wn);
    assert(opr == OPR_ARRSECTION);

    // Get the variable symbol.
    ST_IDX var_st_idx = WN_st_idx(WN_kid0(copy_wn));

    // Get info about the corresponding shared variable.
    struct hc_smem_alias *sa = visible_shared_var(var_st_idx);
    assert(sa != NULL);

    // TODO: handle copy_wn != NULL
    copy_wn = NULL;

    // Construct the data transfer code.
    WN *svar_write_blk = make_svar_transfer_code(sa, false, copy_wn);
    // Insert it before the pragma node so that it will not be processed.
    WN_INSERT_BlockBefore(parent, pragma, svar_write_blk);

    // Get the next node to be processed.
    WN *next_wn = WN_next(pragma);
    // Remove the pragma node.
    WN_DELETE_FromBlock(parent, pragma);
  
    printf("Finished lowering the SHARED COPYOUT directive.\n");

    return next_wn;
}

WN* lower_hc_shared_remove(WN *parent, WN *pragma)
{
    printf("Start lowering a SHARED REMOVE directive ...\n");

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);

    // Make sure that we are inside a kernel function.
    Is_True(kinfo.valid,
        ("SHARED REMOVE directive outside a kernel!\n"));

    // Get the variable whose corresponding shared variable should be freed.
    assert(pragma != NULL && WN_opcode(pragma) == OPC_PRAGMA);
    ST_IDX var_st_idx = WN_st_idx(pragma);

    // Remove the corresponding shared variable.
    struct hc_smem_alias *sa = remove_svar_alias(var_st_idx, parent);
    Is_True(sa != NULL,
        ("Unmatched SHARED REMOVE directive for %s\n", ST_name(var_st_idx)));

    // Record, in the kernel context, that the live range of this shared
    // variable has ended.
    append_svar_life(sa, false);

    // Removing the shared variable implies a barrier.
    // Insert a call to __syncthreads before the pragma.
    // WN_INSERT_BlockBefore(parent, pragma, call_syncthreads());

    // Get the next node to be processed.
    WN *next_wn = WN_next(pragma);
    // Remove the pragma node.
    WN_DELETE_FromBlock(parent, pragma);
 
    printf("Finished lowering the SHARED REMOVE directive.\n");

    return next_wn;
}

void
allocate_svars() {
    if (kinfo.hsl_head != NULL) analyze_svar_live_ranges(kinfo.hsl_head);
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

WN* lower_hc_barrier(WN *parent, WN *pragma)
{
    printf("Start lowering a BARRIER directive ...\n");

    assert(parent != NULL && WN_opcode(parent) == OPC_BLOCK);

    // Make sure that we are inside a kernel function.
    Is_True(kinfo.valid,
        ("BARRIER directive outside a kernel!\n"));

    assert(pragma != NULL && WN_opcode(pragma) == OPC_PRAGMA);

    // Insert a call to __syncthreads before the pragma.
    WN_INSERT_BlockBefore(parent, pragma, call_syncthreads());

    // Get the next node to be processed.
    WN *next_wn = WN_next(pragma);
    // Remove the pragma node.
    WN_DELETE_FromBlock(parent, pragma);
 
    printf("Finished lowering the BARRIER directive.\n");

    return next_wn;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/* Use kinfo. */
void
outline_kernel(WN *region) {
    assert(kinfo.valid);

    WN *body_blk = WN_kid2(region);
    assert(kinfo.kernel_body == body_blk);

    ST_IDX kfunc_st_idx = kinfo.kfunc_st_idx;
    INT nparams = kinfo.nparams;

    // First, create a new scope nested in the current scope.
    // This does not mean that the function is nested because the
    // lexical level of the function symbol should be GLOBAL_SYMTAB+1.
    New_Scope(CURRENT_SYMTAB+1, Scope_tab[CURRENT_SYMTAB].pool, TRUE);
    // Now, the current scope is the kernel function's local scope.
    Scope_tab[CURRENT_SYMTAB].st = ST_ptr(kfunc_st_idx);

    // Transfer label table from the parent scope to the current one.
    transfer_labels(region, CURRENT_SYMTAB-1);

    /* Walk through the body block to transfer all symbols being accessed 
     * (global or local) to local symbols in the new PU. All global symbols
     * referenced in the body must appear as parameters. */
    ST_IDX params[nparams];
    transfer_symbols(body_blk, GLOBAL_SYMTAB+1, nparams, kinfo.params, params);

    /* Create an ENTRY node. */
    WN *kfunc_wn = WN_CreateEntry(nparams, kfunc_st_idx, body_blk, NULL, NULL);
    ST_TAB *curr_st_tab = Scope_tab[CURRENT_SYMTAB].st_tab;
    for (INT i = 0; i < nparams; ++i) {
        // The SYMTAB_LEVEL of the parameter symbol is wrong; it is actually
        // CURRENT_SYMTAB. We only rely on the symbol index.
        ST &st = curr_st_tab->Entry(ST_IDX_index(params[i]));

        // Fix the storage class and export field for each parameter.
        Set_ST_storage_class(st, SCLASS_FORMAL);
        Set_ST_export(st, EXPORT_LOCAL_INTERNAL);

        // Create a formal parameter node.
        WN_kid(kfunc_wn, i) = WN_CreateIdname(0, params[i]);
    }

    /* Set up PU_Info for the kernel function. */
    PU_Info *kfunc_pi = TYPE_MEM_POOL_ALLOC(PU_Info, Malloc_Mem_Pool);
    PU_Info_init(kfunc_pi);

    Set_PU_Info_tree_ptr(kfunc_pi, kfunc_wn);
    Set_PU_Info_state(kfunc_pi, WT_SYMTAB, Subsect_InMem);
    Set_PU_Info_state(kfunc_pi, WT_TREE, Subsect_InMem);
    Set_PU_Info_state(kfunc_pi, WT_PROC_SYM, Subsect_InMem);
    Set_PU_Info_flags(kfunc_pi, PU_IS_COMPILER_GENERATED);

    // Set the procedure symbol idx
    PU_Info_proc_sym(kfunc_pi) = kfunc_st_idx;

    // create a wn map table
    // PU_Info_maptab(kfunc_pi) = WN_MAP_TAB_Create(&_mem_pool);

    PU_Info_pu_dst(kfunc_pi) = DST_INVALID_IDX;
    PU_Info_cu_dst(kfunc_pi) = DST_INVALID_IDX;    

    // Save the local symbol table (just pointers) in the PU.
    SCOPE *saved_scope = TYPE_MEM_POOL_ALLOC(SCOPE, MEM_pu_nz_pool_ptr);
    *saved_scope = Scope_tab[CURRENT_SYMTAB];
    Set_PU_Info_symtab_ptr(kfunc_pi, saved_scope);
 
    // Set the language based on the current PU.
    PU &curr_pu = Current_PU_Info_pu();
    PU &kfunc_pu = PU_Info_pu(kfunc_pi);
    if(PU_c_lang(curr_pu)) Set_PU_c_lang(kfunc_pu);
    if(PU_cxx_lang(curr_pu)) Set_PU_cxx_lang(kfunc_pu);
    if(PU_f77_lang(curr_pu)) Set_PU_f77_lang(kfunc_pu);
    if(PU_f90_lang(curr_pu)) Set_PU_f90_lang(kfunc_pu);

#if 1
    // Insert the new PU before the current one so that it will
    // not be processed by this compiler.
    PU_Info *prev_pi = NULL, *curr_pi = pu_list;
    while (curr_pi != Current_PU_Info) {
        prev_pi = curr_pi;
        curr_pi = PU_Info_next(curr_pi);
    }
    PU_Info_next(kfunc_pi) = Current_PU_Info;
    if (prev_pi == NULL) {
        pu_list = kfunc_pi;
    } else {
        PU_Info_next(prev_pi) = kfunc_pi;
    }
#else
    // Insert the new PU after the current one because we
    // do want to traverse it once.
    PU_Info_next(kfunc_pi) = PU_Info_next(Current_PU_Info);
    PU_Info_next(Current_PU_Info) = kfunc_pi;
#endif

    // Fix the PU level now.
    Set_PU_lexical_level(kfunc_pu, GLOBAL_SYMTAB+1);

    /* No need to write the PU nor free the local symtab. */

    // Write the PU info to file.
    // fix_lexical_level(GLOBAL_SYMTAB+1);
    // Write_PU_Info(kfunc_pi);
    // Free_Local_Info(kfunc_pi);

    // IMPORTANT: restore the current scope.
    --CURRENT_SYMTAB;

    /* Put the replacement body in place.
     * 'body_blk' is taken by the new function, so no need to free it. */
    WN_kid2(region) = kinfo.replacement;

    /* Clean up work, very important!! */
    reset_kinfo();
}

static ST_IDX grid_dim_st_idx = ST_IDX_ZERO;
static ST_IDX blk_dim_st_idx = ST_IDX_ZERO;

WN* lower_hc_kernel(WN *region, WN *pragma)
{
    printf("Start lowering a KERNEL directive ...\n");

    assert(region != NULL && WN_operator(region) == OPR_REGION);

    // Make sure that we are not in a kernel.
    Is_True(!kinfo.valid, ("Found nested kernel directives.\n"));

#if 0

    WN *func_wn = PU_Info_tree_ptr(Current_PU_Info);

    ALIAS_MANAGER *alias_mgr = Create_Alias_Manager(MEM_pu_nz_pool_ptr);
    DU_MANAGER *du_mgr = Create_Du_Manager(MEM_pu_nz_pool_ptr);
    WN *opt_pu = Pre_Optimizer(PREOPT_PHASE, func_wn, du_mgr, alias_mgr);    

    du_mgr->Print_Du_Info();

    Delete_Du_Manager(du_mgr, MEM_pu_nz_pool_ptr);
    Delete_Alias_Manager(alias_mgr, MEM_pu_nz_pool_ptr);

#endif  // TEST

    /* Parse the pragmas. */

    // Get the kernel function symbol (CLASS_NAME).
    assert(pragma != NULL && WN_opcode(pragma) == OPC_PRAGMA);
    ST_IDX kfunc_st_idx = WN_st_idx(pragma);

    // Determine the dimensionality of the virtual block and thread space.
    kinfo.n_vgrid_dims = WN_pragma_arg1(pragma);
    kinfo.n_vblk_dims = WN_pragma_arg2(pragma);

    // Fill the dimension sizes of the block space.
    kinfo.vgrid_dims = (WN**)malloc(kinfo.n_vgrid_dims * sizeof(WN*));
    for (UINT i = 0; i < kinfo.n_vgrid_dims; ++i) {
        pragma = WN_next(pragma);
        assert(pragma != NULL && WN_opcode(pragma) == OPC_XPRAGMA);
        kinfo.vgrid_dims[i] = WN_kid0(pragma);
        // Detach it from the pragma.
        WN_kid0(pragma) = NULL;
    }

    // Fill the dimension sizes of the thread space.
    kinfo.vblk_dims = (WN**)malloc(kinfo.n_vblk_dims * sizeof(WN*));
    for (UINT i = 0; i < kinfo.n_vblk_dims; ++i) {
        pragma = WN_next(pragma);
        assert(pragma != NULL && WN_opcode(pragma) == OPC_XPRAGMA);
        kinfo.vblk_dims[i] = WN_kid0(pragma);
        // Detach it from the pragma.
        WN_kid0(pragma) = NULL;
    }

    // Point to the next pragma to be processed.
    pragma = WN_next(pragma);

    /* Verify whether or not the kernel region is OK in terms of jumps. */
    WN *parent_func_wn = PU_Info_tree_ptr(Current_PU_Info);
    Is_True(verify_region_labels(region, parent_func_wn),
            ("The code region for kernel %s has more than one entry or "
             "more than one exit.\n", ST_name(kfunc_st_idx)));

    /* Determine the physical block and thread space, and the mapping
     * from the physical space to the virtual space.
     */

    // The grid dimension is 2-D.
    kinfo.vblk_idx = (WN**)malloc(kinfo.n_vgrid_dims * sizeof(WN*));
    if (kinfo.n_vgrid_dims <= 2) {
        // The simple case
        UINT i = 0;
        for ( ; i < kinfo.n_vgrid_dims; ++i) {
            kinfo.grid_dims[kinfo.n_vgrid_dims-1-i] =
                WN_COPY_Tree(kinfo.vgrid_dims[i]);
            kinfo.vblk_idx[i] = ldid_blockIdx(kinfo.n_vgrid_dims-1-i);
        }
        for ( ; i < 3; ++i) {
            kinfo.grid_dims[i] = WN_Intconst(Integer_type, 1);
        }
    } else {
        WN* factor[kinfo.n_vgrid_dims];
        for (UINT i = 0; i < kinfo.n_vgrid_dims; ++i) {
            if (i <= 1) {
                kinfo.grid_dims[1-i] = WN_COPY_Tree(kinfo.vgrid_dims[i]);
            } else {
                kinfo.grid_dims[0] = WN_Mpy(Integer_type,
                    kinfo.grid_dims[0], WN_COPY_Tree(kinfo.vgrid_dims[i]));
            }

            if (i >= 1) {
                factor[i] = WN_Intconst(Integer_type, 1);
                for (UINT j = 1; j < i; ++j) {
                    factor[j] = WN_Mpy(Integer_type,
                        factor[j], WN_COPY_Tree(kinfo.vgrid_dims[i]));
                }
            }
        }
        kinfo.grid_dims[2] = WN_Intconst(Integer_type, 1);

        // Determine a mapping from the physical space to the virtual space.
        kinfo.vblk_idx[0] = ldid_blockIdx(1);
        kinfo.vblk_idx[1] = WN_Div(Integer_type, ldid_blockIdx(0), factor[1]);
        for (UINT i = 2; i < kinfo.n_vgrid_dims; ++i) {
            WN *wn = ldid_blockIdx(0);

            // Need to do MOD.
            wn = WN_Binary(OPR_REM, Integer_type,
                wn,
                WN_Mpy(Integer_type,
                    WN_COPY_Tree(factor[i]),
                    WN_COPY_Tree(kinfo.vgrid_dims[i])
                )
            );

            kinfo.vblk_idx[i] = WN_Div(Integer_type, wn, factor[i]);
        }
    }

    // The block dimension is 3-D.
    kinfo.vthr_idx = (WN**)malloc(kinfo.n_vblk_dims * sizeof(WN*));
    if (kinfo.n_vblk_dims <= 3) {
        // The simple case
        UINT i = 0;
        for ( ; i < kinfo.n_vblk_dims; ++i) {
            kinfo.blk_dims[kinfo.n_vblk_dims-1-i] =
                WN_COPY_Tree(kinfo.vblk_dims[i]);
            kinfo.vthr_idx[i] = ldid_threadIdx(kinfo.n_vblk_dims-1-i);
        }
        for ( ; i < 3; ++i) {
            kinfo.blk_dims[i] = WN_Intconst(Integer_type, 1);
        }
    } else {
        WN* factor[kinfo.n_vblk_dims];
        for (UINT i = 0; i < kinfo.n_vblk_dims; ++i) {
            if (i <= 2) {
                kinfo.blk_dims[2-i] = WN_COPY_Tree(kinfo.vblk_dims[i]);
            } else {
                kinfo.blk_dims[0] = WN_Mpy(Integer_type,
                    kinfo.blk_dims[0], WN_COPY_Tree(kinfo.vblk_dims[i]));
            }

            if (i >= 2) {
                factor[i] = WN_Intconst(Integer_type, 1);
                for (UINT j = 2; j < i; ++j) {
                    factor[j] = WN_Mpy(Integer_type,
                        factor[j], WN_COPY_Tree(kinfo.vblk_dims[i]));
                }
            }
        }

        // Determine a mapping from the physical space to the virtual space.
        kinfo.vthr_idx[0] = ldid_threadIdx(2);
        kinfo.vthr_idx[1] = ldid_threadIdx(1);
        kinfo.vthr_idx[2] = WN_Div(Integer_type, ldid_threadIdx(0), factor[2]);
        for (UINT i = 3; i < kinfo.n_vblk_dims; ++i) {
            WN *wn = ldid_threadIdx(0);

            // Need to do MOD.
            wn = WN_Binary(OPR_REM, Integer_type,
                wn,
                WN_Mpy(Integer_type,
                    WN_COPY_Tree(factor[i]),
                    WN_COPY_Tree(kinfo.vblk_dims[i])
                )
            );

            kinfo.vthr_idx[i] = WN_Div(Integer_type, wn, factor[i]);
        }
    }

    /* Perform data flow analysis on the region's body in order to determine
     * kernel parameters. At this point, HiCUDA directives inside the kernel
     * have not been expanded, but they will not generate any LIVE-IN
     * variables. */

    // Build a control flow graph of the region's body.
    hc_bblist *hbl = build_region_cfg(region);
    // print_all_hc_bbs(hbl->head);

    // Initialize the compact symbol index representation.
    init_cmp_st_idx();

    // Get the total number of symbols so far.
    int nsyms = total_num_symbols();

    // TODO: reset the DFA info field for live variable analysis.

    // Compute the GEN/KILL set of each BB.
    compute_lv_genkill_sets(hbl, nsyms);

    // Invoke the data flow framework.
    dfa_solver(hbl, DFA_LIVE_VAR, nsyms);

    // Get symbols selected in the entry's IN set. They are the ones the
    // kernel needs.
    ST_IDX used_syms[nsyms];
    int n_used_syms = bitvector_to_stlist(
        hbl->entry->dfa[DFA_LIVE_VAR].in, used_syms);

    // Deallocate the DFA info and the CFG.
    free_hc_bblist(hbl);

    reset_cmp_st_idx();

    // Go through each symbol to determine kernel parameters.
    int nparams = 0;
    ST_IDX params[nsyms];
    for (int i = 0; i < n_used_syms; ++i) {
        ST_IDX st_idx = used_syms[i];
        ST *st = ST_ptr(st_idx);

        printf("Kernel %s references symbol %s\n",
            ST_name(kfunc_st_idx), ST_name(st));

        // We only worry about variables (not costants).
        if (ST_sym_class(st) != CLASS_VAR) continue;

        // We skip CUDA built-in variables.
        if (st_attr_is_cuda_runtime(st_idx)) continue;

        /* Const variables take precedence over global variables.
         * We don't pass const variables as parameters because they are
         * declared globally.
         */
        if (visible_const_var(st_idx) != NULL) continue;

        // Check if it is a global variable.
        struct hc_gmem_alias *ga = visible_global_var(st_idx);
        if (ga != NULL) {
            // The global variable is the parameter.
            params[nparams++] = ga->gvar_st_idx;
        } else if (TY_kind(ST_type(st)) == KIND_SCALAR) {
            // This is a scalar, we can pass it in as a parameter.
            params[nparams++] = st_idx;
        } else {
            // We can not handle this. Give a warning.
            fprintf(stderr, 
                "Variable %s cannot be passed into kernel %s\n",
                ST_name(st), ST_name(kfunc_st_idx));
        }
    }

    /* Create the kernel function prototype. */

    // Get the parameter types.
    TY_IDX params_ty[nparams];
    for (int i = 0; i < nparams; ++i) params_ty[i] = ST_type(params[i]);

    declare_kernel(kfunc_st_idx,
        MTYPE_To_TY(MTYPE_V),       // void return type
        nparams, params_ty);

    /* Create a block that contains a call to the kernel function,
     * this block will be put in place of the kernel body after outlining. */

    WN *replacement = WN_CreateBlock();

    // Declare grid and block dimension variables.
    if (grid_dim_st_idx == ST_IDX_ZERO) {
        grid_dim_st_idx = new_local_var("dimGrid", dim3_ty_idx);
    }
    if (blk_dim_st_idx == ST_IDX_ZERO) {
        blk_dim_st_idx = new_local_var("dimBlock", dim3_ty_idx);
    }

    // Intialize the grid dimension variable first.
    for (int i = 0; i < 3; ++i) {
        WN_INSERT_BlockLast(replacement, 
            HCWN_StidStructField(grid_dim_st_idx, i+1,
                WN_COPY_Tree(kinfo.grid_dims[i]))
        );
    }
    // Intialize the grid dimension variable first.
    for (int i = 0; i < 3; ++i) {
        WN_INSERT_BlockLast(replacement,
            HCWN_StidStructField(blk_dim_st_idx, i+1,
                WN_COPY_Tree(kinfo.blk_dims[i]))
        );
    }

    // Create an execution configuration for the kernel.
    WN *kfunc_call = WN_Call(MTYPE_V, MTYPE_V, nparams+2, kfunc_st_idx);
    WN_Set_Call_Is_Kernel(kfunc_call);

    // The first two kids are IDNAMEs of grid and block dimension variables.
    WN_kid0(kfunc_call) = WN_CreateIdname(0, grid_dim_st_idx);
    WN_kid1(kfunc_call) = WN_CreateIdname(0, blk_dim_st_idx);

    // The remaining kids are arguments.
    for (int i = 0; i < nparams; ++i) {
        WN *ldid_wn = WN_LdidScalar(params[i]);
        WN_kid(kfunc_call, i+2) = HCWN_Parm(
                WN_desc(ldid_wn), ldid_wn, params_ty[i]);
    }

    WN_INSERT_BlockLast(replacement, kfunc_call);

    /* Save info in the kernel context. */

    kinfo.kfunc_st_idx = kfunc_st_idx;

    kinfo.nparams = nparams;
    kinfo.params = (ST_IDX*)malloc(nparams*sizeof(ST_IDX));
    for (int i = 0; i < nparams; ++i) kinfo.params[i] = params[i];

    kinfo.kernel_body = WN_kid2(region);
    kinfo.replacement = replacement;

    kinfo.valid = true;

    printf("Finished lowering the KERNEL directive.\n");

    return pragma;
}

WN* lower_hc_kernel_part(WN *region, WN *pragma)
{
    printf("Start lowering a LOOP_PARTITION directive ...\n");

    assert(region != NULL && WN_operator(region) == OPR_REGION);

    // Make sure that we are in a kernel.
    Is_True(kinfo.valid, ("LOOP_PARTITION directive outside a kernel!\n"));

    /* Parse the pragmas. */

    assert(pragma != NULL && WN_opcode(pragma) == OPC_PRAGMA);
    enum kernel_part_distr_type blk_distr =
        (enum kernel_part_distr_type)WN_pragma_arg1(pragma);
    enum kernel_part_distr_type thr_distr =
        (enum kernel_part_distr_type)WN_pragma_arg2(pragma);

    // Point to the next pragma to be processed.
    pragma = WN_next(pragma);

    /* There are in total 8 cases; we do not support 2 of them:
     * - over_block(CYCLIC) over_thread(BLOCK/CYCLIC)
     */
    Is_True(blk_distr != HC_KERNEL_PART_CYCLIC
            || thr_distr == HC_KERNEL_PART_NONE,
            ("over_block(CYCLIC) over_thread(*) is not supported.\n"));

    /* Analyze the loop we are about to partition. */

    // The loop must immediately follow the pragma.
    WN *parent_blk = WN_kid2(region);
    WN *loop = WN_first(parent_blk);
    assert(loop != NULL && WN_operator(loop) == OPR_DO_LOOP);

    // Normalize the loop.
    struct doloop_info li;
    assert(normalize_loop(loop, &li));

    // fields of the loop partition info
    WN *blk_range = NULL;
    INT vgrid_dim_idx = -1, vblk_dim_idx = -1;

    /* Handle case by case. */

    if (blk_distr == HC_KERNEL_PART_NONE)
    {
        /* over_thread only */
        assert(thr_distr != HC_KERNEL_PART_NONE);

        vblk_dim_idx = kinfo.curr_vblk_dim_idx;

        // Make sure that the virtual thread space has a dimension left.
        Is_True((UINT)vblk_dim_idx < kinfo.n_vblk_dims,
            ("The dimensionality of the thread space does not match"
             " the thread_partition directives specified.\n"));

        // The block range is the entire loop index range.
        blk_range = get_loop_idx_range(&li);

        if (thr_distr == HC_KERNEL_PART_CYCLIC)
        {
            /* new_init = init + threadIdx * step,
             * new_end = end,
             * new_step = n_threads * step */
            TYPE_ID mtype = WN_rtype(li.init_val);
            WN *new_init = WN_Add(mtype,
                WN_COPY_Tree(li.init_val),
                WN_Mpy(mtype,
                    WN_COPY_Tree(kinfo.vthr_idx[vblk_dim_idx]),
                    WN_COPY_Tree(li.step_val)
                )
            );
            WN_DELETE_Tree(WN_kid0(WN_kid1(loop)));
            WN_kid0(WN_kid1(loop)) = new_init;

            mtype = WN_rtype(li.step_val);
            WN *new_step = WN_Mpy(mtype,
                WN_COPY_Tree(kinfo.vblk_dims[vblk_dim_idx]),
                WN_COPY_Tree(li.step_val)
            );
            WN_DELETE_Tree(WN_kid1(WN_kid0(WN_kid(loop,3))));
            WN_kid1(WN_kid0(WN_kid(loop,3))) = new_step;

            // TODO: as an optimization, if we could figure out the loop's
            // tripcount, we could unroll the loop automatically.
        }
        else
        {
            /* new_init = init + threadIdx * thread_sz * step,
             * new_end = new_init + thread_sz * step,
             * new_step = step
             *
             * thread_sz = ceil(tripcount/n_threads)
             * If not perfect division, need to compare new_end with end.
             */

            /* Try our best to determine if the current dimension size of
             * the virtual thread space divides the loop's tripcount.
             */
            bool perfect_div;
            WN *thread_sz = HCWN_Ceil(li.tripcount,
                kinfo.vblk_dims[vblk_dim_idx], &perfect_div);

            // new_init
            TYPE_ID mtype = Integer_type;
            WN *new_init = WN_Add(mtype,
                WN_COPY_Tree(li.init_val),
                WN_Mpy(mtype,
                    WN_COPY_Tree(kinfo.vthr_idx[vblk_dim_idx]),
                    WN_Mpy(mtype,
                        WN_COPY_Tree(thread_sz),
                        WN_COPY_Tree(li.step_val)
                    )
                )
            );

            // Get the loop's index variable.
            ST_IDX idxv_st_idx = WN_st_idx(WN_kid0(loop));

            if (WN_operator(thread_sz) == OPR_INTCONST
                    && WN_const_val(thread_sz) == 1)
            {
                /* Replace the loop with a single index assignment. */

                WN *body_blk = WN_kid(loop,4);
                WN_kid(loop,4) = NULL;

                // For non-perfect division, add a guard for the loop body.
                if (!perfect_div) {
                    WN *guard = WN_CreateIf(WN_COPY_Tree(WN_kid2(loop)),
                        body_blk, WN_CreateBlock());
                    body_blk = WN_CreateBlock();
                    WN_INSERT_BlockFirst(body_blk, guard);
                }

                // Insert a STID of the actual index variable at the
                // beginning of the loop body.
                WN_INSERT_BlockFirst(body_blk,
                    WN_StidScalar(ST_ptr(idxv_st_idx), new_init)
                );

                // Insert the loop's body before the loop.
                WN_INSERT_BlockBefore(parent_blk, loop, body_blk);
                // Remove the current loop.
                WN_DELETE_FromBlock(parent_blk, loop);
            }
            else
            {
                /* Change the existing DO_LOOP's header. */

                // For non-perfect division, add a guard inside the loop.
                if (!perfect_div) {
                    WN *guarded_body = WN_CreateIf(WN_COPY_Tree(WN_kid2(loop)),
                        WN_kid(loop,4), WN_CreateBlock());
                    WN_kid(loop,4) = WN_CreateBlock();
                    WN_INSERT_BlockFirst(WN_kid(loop,4), guarded_body);
                }

                // Introduce a new local variable to hold the init value.
                ST_IDX init_var_st_idx = new_local_var(
                    gen_var_str(idxv_st_idx, "_init"),
                    MTYPE_To_TY(mtype)
                );
                ST *init_var_st = ST_ptr(init_var_st_idx);

                // Construct an assignment statement of the initial value.
                WN *init_var_stid = WN_StidScalar(init_var_st, new_init);
                // Insert it before the loop.
                WN_INSERT_BlockBefore(parent_blk, loop, init_var_stid);

                // Replace the original init expression in the loop.
                WN_DELETE_Tree(WN_kid0(WN_kid1(loop)));
                WN *init_var_ldid = WN_LdidScalar(init_var_st_idx);
                WN_kid0(WN_kid1(loop)) = init_var_ldid;

                // Construct an ending value expression.
                WN *new_end = WN_Add(mtype,
                    WN_COPY_Tree(init_var_ldid),
                    WN_Mpy(mtype,
                        WN_COPY_Tree(thread_sz),
                        WN_COPY_Tree(li.step_val)
                    )
                );

                // Introduce a new local variable to hold the end value.
                ST_IDX end_var_st_idx = new_local_var(
                    gen_var_str(idxv_st_idx, "_end"),
                    MTYPE_To_TY(mtype));
                ST *end_var_st = ST_ptr(end_var_st_idx);

                // Construct an assignment statement of the end value.
                WN *end_val_stid = WN_StidScalar(end_var_st, new_end);
                // Insert it before the loop.
                WN_INSERT_BlockBefore(parent_blk, loop, end_val_stid);

                // Replace the original end value with the new variable.
                WN_DELETE_Tree(WN_kid1(WN_kid2(loop)));
                WN_kid1(WN_kid2(loop)) = WN_LdidScalar(end_var_st_idx);
            }

            // Clean up.
            WN_DELETE_Tree(thread_sz);
        }

        // Update the current thread space dimension index.
        kinfo.curr_vblk_dim_idx += 1;
    }
    else if (thr_distr == HC_KERNEL_PART_NONE)
    {
        /* over_block only */
        vgrid_dim_idx = kinfo.curr_vgrid_dim_idx;

        // Make sure that the virtual block space has a dimension left.
        Is_True((UINT)vgrid_dim_idx < kinfo.n_vgrid_dims,
                ("The dimensionality of the block space does not match"
                 " the block_partition directives specified.\n"));

        if (blk_distr == HC_KERNEL_PART_CYCLIC)
        {
            /* One of the index expressions in the block range would be
             * complicated, so leave blk_range NULL.
             */

            /* new_init = init + blockIdx * step,
             * new_end = end,
             * new_step = n_blocks * step */
            TYPE_ID mtype = WN_rtype(li.init_val);
            WN *new_init = WN_Add(mtype,
                WN_COPY_Tree(li.init_val),
                WN_Mpy(mtype,
                    WN_COPY_Tree(kinfo.vblk_idx[vgrid_dim_idx]),
                    WN_COPY_Tree(li.step_val)
                )
            );
            WN_DELETE_Tree(WN_kid0(WN_kid1(loop)));
            WN_kid0(WN_kid1(loop)) = new_init;

            mtype = WN_rtype(li.step_val);
            WN *new_step = WN_Mpy(mtype,
                WN_COPY_Tree(kinfo.vgrid_dims[vgrid_dim_idx]),
                WN_COPY_Tree(li.step_val)
            );
            WN_DELETE_Tree(WN_kid1(WN_kid0(WN_kid(loop,3))));
            WN_kid1(WN_kid0(WN_kid(loop,3))) = new_step;

            // TODO: as an optimization, if we could figure out the loop's
            // tripcount, we could unroll the loop automatically.
        }
        else
        {
            /* new_init = init + blockIdx * block_sz * step,
             * new_end = new_init + block_sz * step,
             * new_step = step
             *
             * block_sz = ceil(tripcount/n_blocks)
             * If not perfect division, need to compare new_end with end.
             */

            /* Try our best to determine if the current dimension size of
             * the virtual block space divides the loop's tripcount.
             */
            bool perfect_div;
            WN *block_sz = HCWN_Ceil(li.tripcount,
                kinfo.vgrid_dims[vgrid_dim_idx], &perfect_div);

            /* Determine the block range. */
            if (WN_operator(li.step_val) == OPR_INTCONST)
            {
                INT64 step_val = WN_const_val(li.step_val);

                /* start_idx = init + blockIdx * block_sz * step
                 * stride = step
                 * end_idx = start_idx + (block_sz - 1) * step
                 */
                TYPE_ID mtype = Integer_type;
                WN *start_idx = WN_Add(mtype,
                    WN_COPY_Tree(li.init_val),
                    WN_Mpy(mtype,
                        WN_COPY_Tree(kinfo.vblk_idx[vgrid_dim_idx]),
                        WN_Mpy(mtype,
                            WN_COPY_Tree(block_sz),
                            WN_Intconst(mtype, step_val)
                        )
                    )
                );
                WN *end_idx = WN_Add(mtype,
                    WN_COPY_Tree(start_idx),
                    WN_Mpy(mtype,
                        WN_Sub(mtype,
                            WN_COPY_Tree(block_sz),
                            WN_Intconst(mtype, 1)
                        ),
                        WN_Intconst(mtype, step_val)
                    )
                );

                if (step_val > 0) {
                    blk_range = WN_CreateTriplet(
                        start_idx, WN_Intconst(mtype, step_val), end_idx);
                } else {
                    blk_range = WN_CreateTriplet(
                        end_idx, WN_Intconst(mtype, -step_val), start_idx);
                }
            }

            // new_init
            TYPE_ID mtype = Integer_type;
            WN *new_init = WN_Add(mtype,
                WN_COPY_Tree(li.init_val),
                WN_Mpy(mtype,
                    WN_COPY_Tree(kinfo.vblk_idx[vgrid_dim_idx]),
                    WN_Mpy(mtype,
                        WN_COPY_Tree(block_sz),
                        WN_COPY_Tree(li.step_val)
                    )
                )
            );

            // Get the loop's index variable.
            ST_IDX idxv_st_idx = WN_st_idx(WN_kid0(loop));

            if (WN_operator(block_sz) == OPR_INTCONST
                && WN_const_val(block_sz) == 1)
            {
                /* Replace the loop with a single index assignment. */

                WN *body_blk = WN_kid(loop,4);
                WN_kid(loop,4) = NULL;

                // For non-perfect division, add a guard for the loop body.
                if (!perfect_div) {
                    WN *guard = WN_CreateIf(WN_COPY_Tree(WN_kid2(loop)),
                        body_blk, WN_CreateBlock());
                    body_blk = WN_CreateBlock();
                    WN_INSERT_BlockFirst(body_blk, guard);
                }

                // Insert a STID of the actual index variable at the
                // beginning of the loop body.
                WN_INSERT_BlockFirst(body_blk,
                    WN_StidScalar(ST_ptr(idxv_st_idx), new_init)
                );

                // Insert the loop's body before the loop.
                WN_INSERT_BlockBefore(parent_blk, loop, body_blk);
                // Remove the current loop.
                WN_DELETE_FromBlock(parent_blk, loop);
            }
            else
            {
                /* Change the existing DO_LOOP's header. */

                // For non-perfect division, add a guard inside the loop.
                if (!perfect_div) {
                    WN *guarded_body = WN_CreateIf(WN_COPY_Tree(WN_kid2(loop)),
                        WN_kid(loop,4), WN_CreateBlock());
                    WN_kid(loop,4) = WN_CreateBlock();
                    WN_INSERT_BlockFirst(WN_kid(loop,4), guarded_body);
                }

                // Introduce a new local variable to hold the init value.
                ST_IDX init_var_st_idx = new_local_var(
                    gen_var_str(idxv_st_idx, "_init"),
                    MTYPE_To_TY(mtype)
                );
                ST *init_var_st = ST_ptr(init_var_st_idx);

                // Construct an assignment statement of the initial value.
                WN *init_var_stid = WN_StidScalar(init_var_st, new_init);
                // Insert it before the loop.
                WN_INSERT_BlockBefore(parent_blk, loop, init_var_stid);

                // Replace the original init expression in the loop.
                WN_DELETE_Tree(WN_kid0(WN_kid1(loop)));
                WN *init_var_ldid = WN_LdidScalar(init_var_st_idx);
                WN_kid0(WN_kid1(loop)) = init_var_ldid;

                // Construct an ending value expression.
                WN *new_end = WN_Add(mtype,
                    WN_COPY_Tree(init_var_ldid),
                    WN_Mpy(mtype,
                        WN_COPY_Tree(block_sz),
                        WN_COPY_Tree(li.step_val)
                    )
                );

                // Introduce a new local variable to hold the end value.
                ST_IDX end_var_st_idx = new_local_var(
                    gen_var_str(idxv_st_idx, "_end"),
                    MTYPE_To_TY(mtype));
                ST *end_var_st = ST_ptr(end_var_st_idx);

                // Construct an assignment statement of the end value.
                WN *end_val_stid = WN_StidScalar(end_var_st, new_end);
                // Insert it before the loop.
                WN_INSERT_BlockBefore(parent_blk, loop, end_val_stid);

                // Replace the original end value with the new variable.
                WN_DELETE_Tree(WN_kid1(WN_kid2(loop)));
                WN_kid1(WN_kid2(loop)) = WN_LdidScalar(end_var_st_idx);
            }

            // Clean up.
            WN_DELETE_Tree(block_sz);
        }

        // Update the current block space dimension index.
        kinfo.curr_vgrid_dim_idx += 1;
    }
    else
    {
        /* over_block(BLOCK) over_thread(*) */
        assert(blk_distr == HC_KERNEL_PART_BLOCK);

        vgrid_dim_idx = kinfo.curr_vgrid_dim_idx;
        vblk_dim_idx = kinfo.curr_vblk_dim_idx;

        // Make sure that the virtual block space has a dimension left.
        Is_True((UINT)vgrid_dim_idx < kinfo.n_vgrid_dims,
                ("The dimensionality of the block space does not match"
                 " the block_partition directives specified.\n"));

        // Make sure that the virtual thread space has a dimension left.
        Is_True((UINT)vblk_dim_idx < kinfo.n_vblk_dims,
                ("The dimensionality of the thread space does not match"
                 " the thread_partition directives specified.\n"));

        /* Try our best to determine if the current dimension size of
         * the virtual block space divides the loop's tripcount. */
        bool perfect_blk_div;
        WN *block_sz = HCWN_Ceil(li.tripcount,
                kinfo.vgrid_dims[vgrid_dim_idx], &perfect_blk_div);

        /* Determine the block range. */
        if (WN_operator(li.step_val) == OPR_INTCONST)
        {
            INT64 step_val = WN_const_val(li.step_val);

            /* start_idx = init + blockIdx * block_sz * step
             * stride = step
             * end_idx = start_idx + (block_sz - 1) * step
             */
            TYPE_ID mtype = Integer_type;
            WN *start_idx = WN_Add(mtype,
                    WN_COPY_Tree(li.init_val),
                    WN_Mpy(mtype,
                        WN_COPY_Tree(kinfo.vblk_idx[vgrid_dim_idx]),
                        WN_Mpy(mtype,
                            WN_COPY_Tree(block_sz),
                            WN_Intconst(mtype, step_val))));
            WN *end_idx = WN_Add(mtype,
                    WN_COPY_Tree(start_idx),
                    WN_Mpy(mtype,
                        WN_Sub(mtype,
                            WN_COPY_Tree(block_sz),
                            WN_Intconst(mtype, 1)),
                        WN_Intconst(mtype, step_val)));

            if (step_val > 0) {
                blk_range = WN_CreateTriplet(
                        start_idx, WN_Intconst(mtype, step_val), end_idx);
            } else {
                blk_range = WN_CreateTriplet(
                        end_idx, WN_Intconst(mtype, -step_val), start_idx);
            }
        }

        // Get the loop's index variable.
        ST_IDX idxv_st_idx = WN_st_idx(WN_kid0(loop));

        TYPE_ID mtype = WN_rtype(li.init_val);

        /* Introduce a local variable to hold `block_sz * step'. */

        // `<loop index>_blk_sz'
        ST_IDX blk_sz_st_idx = new_local_var(
                gen_var_str(idxv_st_idx, "_blk_sz"), MTYPE_To_TY(mtype));
        ST *blk_sz_st = ST_ptr(blk_sz_st_idx);

        WN_INSERT_BlockBefore(parent_blk, loop,
                WN_StidScalar(blk_sz_st,
                    WN_Mpy(mtype,
                        WN_COPY_Tree(block_sz),
                        WN_COPY_Tree(li.step_val))));

        /* Introduce a local variable to hold the block's starting index:
         * `init + blockIdx * block_sz * step'. */

        // `<loop_index>_blk_base'
        ST_IDX blk_base_st_idx = new_local_var(
                gen_var_str(idxv_st_idx, "_blk_base"), MTYPE_To_TY(mtype));
        ST *blk_base_st = ST_ptr(blk_base_st_idx);

        WN_INSERT_BlockBefore(parent_blk, loop,
                WN_StidScalar(blk_base_st,
                    WN_Add(mtype,
                        WN_COPY_Tree(li.init_val),
                        WN_Mpy(mtype,
                            WN_COPY_Tree(kinfo.vblk_idx[vgrid_dim_idx]),
                            WN_LdidScalar(blk_sz_st_idx)))));


        if (thr_distr == HC_KERNEL_PART_CYCLIC)
        {
            /* new_init = init + blockIdx * block_sz * step + threadIdx * step
             * new_end = init + blockIdx * block_sz * step + block_sz * step
             * new_step = step * n_threads
             *
             * If block_sz does not result from perfect division, add a guard
             * against the loop's end value inside the loop.
             */

            if (!perfect_blk_div) {
                WN *guarded_body = WN_CreateIf(WN_COPY_Tree(WN_kid2(loop)),
                        WN_kid(loop,4), WN_CreateBlock());
                WN_kid(loop,4) = WN_CreateBlock();
                WN_INSERT_BlockFirst(WN_kid(loop,4), guarded_body);
            }

            // Replace the original init expression in the loop.
            WN_DELETE_Tree(WN_kid0(WN_kid1(loop)));
            WN_kid0(WN_kid1(loop)) = WN_Add(mtype,
                    WN_LdidScalar(blk_base_st_idx),
                    WN_Mpy(mtype,
                        WN_COPY_Tree(kinfo.vthr_idx[vblk_dim_idx]),
                        WN_COPY_Tree(li.step_val)));

            // Introduce a local variable to hold the new past-end index.
            // `<loop_index>_blk_end'
            ST_IDX blk_end_st_idx = new_local_var(
                    gen_var_str(idxv_st_idx, "_blk_end"), MTYPE_To_TY(mtype));
            ST *blk_end_st = ST_ptr(blk_end_st_idx);
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(blk_end_st,
                        WN_Add(mtype,
                            WN_LdidScalar(blk_base_st_idx),
                            WN_LdidScalar(blk_sz_st_idx))));

            // Replace the end value in the loop.
            WN_DELETE_Tree(WN_kid1(WN_kid2(loop)));
            WN_kid1(WN_kid2(loop)) = WN_LdidScalar(blk_end_st_idx);

            // Replace the step value in the loop.
            WN_DELETE_Tree(WN_kid1(WN_kid0(WN_kid(loop,3))));
            WN_kid1(WN_kid0(WN_kid(loop,3))) = WN_Mpy(mtype,
                    WN_COPY_Tree(kinfo.vblk_dims[vblk_dim_idx]),
                    WN_COPY_Tree(li.step_val));

            // TODO: as an optimization, if we could figure out the loop's
            // tripcount, we could unroll the loop automatically.
        }
        else
        {
            /* new_init = init + (blockIdx*block_sz+threadIdx*thread_sz) * step
             * new_end = new_init + thread_sz * step
             * new_step = step
             */

            /* Try our best to determine if the current dimension size of
             * the virtual thread space divides the block size.
             */
            bool perfect_thr_div;
            WN *thread_sz = HCWN_Ceil(block_sz,
                    kinfo.vblk_dims[vblk_dim_idx], &perfect_thr_div);

            // Extract the loop body block.
            WN *body_blk = WN_kid(loop,4);
            WN_kid(loop,4) = NULL;

            /* Add a guard to the loop body.
             *
             * If block_sz does not result from perfect division, the guard
             * is against the loop tripcount; if thread_sz does not result
             * from perfect division, the guard is against block_end.
             */
            WN *guard = NULL;
            if (!perfect_blk_div) guard = WN_COPY_Tree(WN_kid2(loop));
            if (!perfect_thr_div) {
                // Introduce a local variable to hold the new past-end index.
                // `<loop_index>_blk_end'
                ST_IDX blk_end_st_idx = new_local_var(
                        gen_var_str(idxv_st_idx, "_blk_end"),
                        MTYPE_To_TY(mtype));
                ST *blk_end_st = ST_ptr(blk_end_st_idx);
                WN_INSERT_BlockBefore(parent_blk, loop,
                        WN_StidScalar(blk_end_st,
                            WN_Add(mtype,
                                WN_LdidScalar(blk_base_st_idx),
                                WN_LdidScalar(blk_sz_st_idx))));

                /* Generate the condition loop_idx < block_end.
                 * The sign of step_val is based on `flip_sign'. */
                WN *blk_guard = WN_Relational(
                        li.flip_sign ? OPR_GT : OPR_LT, mtype,
                        WN_LdidScalar(idxv_st_idx),
                        WN_LdidScalar(blk_end_st_idx));

                if (guard == NULL) {
                    guard = blk_guard;
                } else {
                    guard = WN_LAND(guard, blk_guard);
                }
            }

            if (guard != NULL) {
                WN *guarded_blk = WN_CreateIf(guard,
                        body_blk, WN_CreateBlock());
                body_blk = WN_CreateBlock();
                WN_INSERT_BlockFirst(body_blk, guarded_blk);
            }

            // Introduce a local variable to hold `thread_sz * step'.
            // `<loop index>_thr_sz'
            ST_IDX thr_sz_st_idx = new_local_var(
                    gen_var_str(idxv_st_idx, "_thr_sz"), MTYPE_To_TY(mtype));
            ST *thr_sz_st = ST_ptr(thr_sz_st_idx);
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(thr_sz_st,
                        WN_Mpy(mtype,
                            WN_COPY_Tree(thread_sz),
                            WN_COPY_Tree(li.step_val))));

            // Compute the new initial value, for later use.
            mtype = Integer_type;
            WN *new_init_wn = WN_Add(mtype,
                    WN_LdidScalar(blk_base_st_idx),
                    WN_Mpy(mtype,
                        WN_COPY_Tree(kinfo.vthr_idx[vblk_dim_idx]),
                        WN_LdidScalar(thr_sz_st_idx)));

            if (WN_operator(thread_sz) == OPR_INTCONST
                    && WN_const_val(thread_sz) == 1)
            {
                /* Replace the loop with a single index assignment. */

                // Insert a STID of the actual index variable at the
                // beginning of the loop body.
                WN_INSERT_BlockFirst(body_blk,
                        WN_StidScalar(ST_ptr(idxv_st_idx), new_init_wn));

                // Insert the loop's body before the loop.
                WN_INSERT_BlockBefore(parent_blk, loop, body_blk);
                // Remove the current loop.
                WN_DELETE_FromBlock(parent_blk, loop);
            }
            else
            {
                /* Change the existing DO_LOOP's header. */

                WN_kid(loop,4) = body_blk;

                // Introduce a new local variable to hold the init value.
                ST_IDX thr_base_st_idx = new_local_var(
                        gen_var_str(idxv_st_idx, "_thr_base"),
                        MTYPE_To_TY(mtype));
                ST *thr_base_st = ST_ptr(thr_base_st_idx);
                WN_INSERT_BlockBefore(parent_blk, loop,
                        WN_StidScalar(thr_base_st, new_init_wn));

                // Replace the init expression in the loop.
                WN_DELETE_Tree(WN_kid0(WN_kid1(loop)));
                WN_kid0(WN_kid1(loop)) = WN_LdidScalar(thr_base_st_idx);

                // Introduce a new local variable to hold the end value.
                ST_IDX thr_end_st_idx = new_local_var(
                        gen_var_str(idxv_st_idx, "_thr_end"),
                        MTYPE_To_TY(mtype));
                ST *thr_end_st = ST_ptr(thr_end_st_idx);
                WN_INSERT_BlockBefore(parent_blk, loop,
                        WN_StidScalar(thr_end_st,
                            WN_Add(mtype,
                                WN_LdidScalar(thr_base_st_idx),
                                WN_LdidScalar(thr_sz_st_idx))));

                // Replace the end value in the loop.
                WN_DELETE_Tree(WN_kid1(WN_kid2(loop)));
                WN_kid1(WN_kid2(loop)) = WN_LdidScalar(thr_end_st_idx);
            }

            // Clean up.
            WN_DELETE_Tree(thread_sz);
        }

        // Clean up.
        WN_DELETE_Tree(block_sz);

        // Update the current block space dimension index.
        kinfo.curr_vgrid_dim_idx += 1;
        // Update the current thread space dimension index.
        kinfo.curr_vblk_dim_idx += 1;
    }

    // Prepare the loop partition info.
    struct loop_part_info *lpi = (struct loop_part_info*)
        malloc(sizeof(struct loop_part_info));
    lpi->idxv_st_idx = WN_st_idx(WN_kid0(loop));
    lpi->vgrid_dim_idx = vgrid_dim_idx;
    lpi->vblk_dim_idx = vblk_dim_idx;
    lpi->blk_range = blk_range;
    lpi->full_range = get_loop_idx_range(&li);
    // Add it to the top of the stack.
    push_loop_info(lpi);

    // Clean up the loop info struct.
    clean_doloop_info(&li);

    printf("Finished lowering the LOOP_PARTITION directive.\n");

    return pragma;
}   /* lower_hc_kernel_part */

void
end_kernel_part_region() {
    // Pop the top element from the loop stack.
    struct loop_part_info *lpi = pop_loop_info();
    assert(lpi != NULL && lpi->idxv_st_idx != ST_IDX_ZERO);

    // Update the running context of kinfo.
    if (lpi->vgrid_dim_idx >= 0) {
        assert((UINT)lpi->vgrid_dim_idx == (--kinfo.curr_vgrid_dim_idx));
    }
    if (lpi->vblk_dim_idx >= 0) {
        assert((UINT)lpi->vblk_dim_idx == (--kinfo.curr_vblk_dim_idx));
    }

    // Clean up.
    free_loop_info(lpi);
}

void
end_loop_in_kernel(WN *loop) {
    // Do nothing if it's outside the kernel.
    if (!kinfo.valid) return;

    // Peek the top of the loop stack.
    struct loop_part_info *lpi = loop_stack_top();
    assert(lpi != NULL);

    if (lpi->vgrid_dim_idx < 0 && lpi->vblk_dim_idx < 0) {
        assert(lpi == pop_loop_info());
        // Clean up.
        free_loop_info(lpi);
    }
}

WN*
kernel_processing(WN *wn, WN *parent, bool *del_wn) {
    if (del_wn != NULL) *del_wn = false;

    if (parent == NULL) {
        assert(WN_operator(wn) == OPR_FUNC_ENTRY);
        return NULL;
    }

    WN *ret_wn = (WN_operator(parent) == OPR_BLOCK) ? WN_next(wn) : NULL;

    // Do nothing if we are not inside a kernel.
    if (!kinfo.valid) return ret_wn;

    OPERATOR opr = WN_operator(wn);
    switch (opr) {
        case OPR_DO_WHILE:
        case OPR_WHILE_DO: {
            // At this point, we cannot convert it to DO_LOOP.
            struct loop_part_info *lpi = (struct loop_part_info*)
                malloc(sizeof(struct loop_part_info));
            lpi->idxv_st_idx = ST_IDX_ZERO;
            lpi->vgrid_dim_idx = -1;
            lpi->vblk_dim_idx = -1;
            lpi->blk_range = NULL;
            lpi->full_range = NULL;

            // Add it to the top of the stack.
            push_loop_info(lpi);
        }
        break;
        
        case OPR_DO_LOOP: {
            // Normalize the loop.
            struct doloop_info li;
            assert(normalize_loop(wn, &li));

            struct loop_part_info *lpi = (struct loop_part_info*)
                malloc(sizeof(struct loop_part_info));
            lpi->idxv_st_idx = WN_st_idx(WN_kid0(wn));
            lpi->vgrid_dim_idx = -1;
            lpi->vblk_dim_idx = -1;
            // The block range is also the full loop index range.
            lpi->blk_range = get_loop_idx_range(&li);
            lpi->full_range = WN_COPY_Tree(lpi->blk_range);

            // It may have already been added to the loop stack,
            // due to KERNEL PART directive.
            struct loop_part_info *top = loop_stack_top();
            if (top == NULL || top->idxv_st_idx != lpi->idxv_st_idx
                || (top->vgrid_dim_idx < 0 && top->vblk_dim_idx < 0)) {
                // Add it to the top of the stack.
                push_loop_info(lpi);
            }

            // Clean up.
            clean_doloop_info(&li);
        }
        break;

        case OPR_LDID:
        case OPR_STID:
        case OPR_ARRAY: {
            WN *new_wn;
            replace_var_access(wn, &new_wn);
            if (new_wn != NULL) {
                if (WN_operator(parent) == OPR_BLOCK) {
                    WN_INSERT_BlockBefore(parent, wn, new_wn);
                    WN_DELETE_FromBlock(parent, wn);
                } else {
                    assert(replace_wn_kid(parent, wn, new_wn));
                    WN_DELETE_Tree(wn);
                    ret_wn = new_wn;
                }
                if (del_wn != NULL) *del_wn = true;
            }
        }
        break;

        default: break;
    }

    return ret_wn;
}   /* kernel_processing */

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

static bool referenced_by_ldid_only = true;

/**
 * Recursively replace all loads of a given variable with the given
 * expression.
 *
 * 'referenced_by_ldid_only' will be false if there is some reference to
 * the given variable which does not come from LDID.
 *
 * Return true if 'wn' should be replaced by 'expr_wn' and false otherwise.
 */
static bool
replace_ldid(WN *wn, ST_IDX st_idx, WN *expr_wn) {
    assert(wn != NULL && st_idx != ST_IDX_ZERO);

    OPERATOR opr = WN_operator(wn);

    // Skip pragmas: VERY IMPORTANT!.
    // Otherwise, $k may be replaced with $k + kt0.
    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA) return false;

    ST_IDX wn_st_idx = (OPERATOR_has_sym(opr) ? WN_st_idx(wn) : ST_IDX_ZERO);
    if (wn_st_idx == st_idx) {
        if (opr == OPR_LDID) return true;
        referenced_by_ldid_only = false;
    }

    // Handle non-leaf nodes.
    if (opr == OPR_BLOCK) {
        WN *node = WN_first(wn), *next = NULL;
        while (node != NULL) {
            next = WN_next(node);
            if (replace_ldid(node, st_idx, expr_wn)) {
                WN_INSERT_BlockBefore(wn, node, WN_COPY_Tree(expr_wn));
                WN_DELETE_FromBlock(wn, node);
            }
            node = next;
        }
    } else {
        int nkids = WN_kid_count(wn);
        for (int i = 0; i < nkids; ++i) {
            // Some of the kids might be NULL because the WN node the
            // front end generates might violate the WHIRL spec.
            WN *kid = WN_kid(wn,i);
            if (kid == NULL) continue;

            if (replace_ldid(kid, st_idx, WN_COPY_Tree(expr_wn))) {
                WN_DELETE_Tree(kid);
                WN_kid(wn,i) = expr_wn;
            }
        }
    }

    return false;
}

WN*
lower_hc_loopblock(WN *region, WN *pragma) {
    printf("Lowering a HiCUDA LOOPBLOCK directive ... ");

    // Sanity checks.
    assert(region != NULL && WN_opcode(region) == OPC_REGION);
    
    // The directive must be inside a kernel.
    Is_True(kinfo.valid,
        (("A LOOPBLOCK directive outside a kernel!\n")));

    /* Parse the pragmas. */

    assert(pragma != NULL && WN_opcode(pragma) == OPC_XPRAGMA);
    WN *tile_sz_wn = WN_kid0(pragma);
    assert(tile_sz_wn != NULL);

    // If there are follow-up pragmas, create an artifical REGION for
    // SHARED directives.
    WN *svar_region = NULL;
    pragma = WN_next(pragma);
    if (pragma != NULL && WN_pragma(pragma) == WN_PRAGMA_HC_SHARED_COPYIN) {
        svar_region = WN_CreateRegion(REGION_KIND_HICUDA,
            WN_CreateBlock(), WN_CreateBlock(), WN_CreateBlock(), -1, 0);

        WN *svar_pragma_blk = WN_kid1(svar_region);
        WN *pragma_blk = WN_kid1(region);
        WN *next_pragma = NULL;

        do {
            next_pragma = WN_next(pragma);
            // Must remove it from the current pragma block.
            WN_EXTRACT_FromBlock(pragma_blk, pragma);
            WN_INSERT_BlockLast(svar_pragma_blk, pragma);
            pragma = next_pragma;
        } while (pragma != NULL
            && WN_pragma(pragma) == WN_PRAGMA_HC_SHARED_COPYIN);
    }
    // Now, 'pragma' points to the next pragma to be processed.

    /* Although loop tiling size does not have to be known at compile time,
     * it does not make sense here because the purpose of loop tiling is
     * often for load data pieces into shared memory and we have to know the
     * size at compile time. Therefore, let's for now enforce that the tile
     * size must be a constant at this point. */
    Is_True(WN_operator(tile_sz_wn) == OPR_INTCONST,
        ("Non-constant tile size is not allowed in LOOPBLOCK directive.\n"));

    INT32 tile_sz = WN_const_val(tile_sz_wn);
    Is_True(tile_sz > 0,
        ("Invalid tile size in a LOOPBLOCK directive.\n"));

    /* Analyze the loop to which we apply tiling. */

    // Skip any non-executable nodes.
    WN *loop = WN_first(WN_kid2(region));
    assert(loop != NULL && WN_operator(loop) == OPR_DO_LOOP);

    // Normalize the loop.
    doloop_info li;
    assert(normalize_loop(loop, &li));

    // For now, the loop tripcount must be divisible by the tile size.
    Is_True(WN_operator(li.tripcount) == OPR_INTCONST
        && WN_const_val(li.tripcount) % tile_sz == 0,
        ("The tile size in a LOOPBLOCK directive does not divide "
         "the loop tripcount.\n"));

    /* Transform the loop:
     *
     * We keep the original loop as the iterator loop and we only adjust its
     * step size. A new inner loop is created whose tripcount is the tile
     * size. We do not actually normalize the iterator loop because it will
     * makes the generated loop and possibly shared memory loading more
     * efficient. */

    // Multiply step size in the iterator (current) loop by the tile size.
    WN *step_sz_wn = WN_kid1(WN_kid0(WN_kid3(loop)));
    assert(WN_operator(step_sz_wn) == OPR_INTCONST);
    int step_sz = WN_const_val(step_sz_wn);
    WN_const_val(step_sz_wn) = step_sz * tile_sz;

    ST_IDX idxv_st_idx = WN_st_idx(WN_kid0(loop));
    TY_IDX idxv_ty_idx = ST_type(idxv_st_idx);
    TYPE_ID idxv_mtype = TY_mtype(idxv_ty_idx);
    int abs_step_sz = (step_sz > 0) ? step_sz : -step_sz;

    if (tile_sz > 1) {
        /* Create an empty simple loop with the original step size.
         * Its tripcount is tile_sz. */
        ST_IDX inner_idxv_st_idx = new_local_var(
            gen_var_str(idxv_st_idx, "t"), idxv_ty_idx);
        WN *init_wn = WN_Intconst(idxv_mtype, 0);
        WN *end_wn = WN_Intconst(idxv_mtype, (tile_sz-1)*abs_step_sz);
        WN *step_wn = WN_Intconst(idxv_mtype, abs_step_sz);
        WN *inner_loop = make_empty_doloop(inner_idxv_st_idx,
            init_wn, end_wn, step_wn);
        WN_DELETE_Tree(init_wn);
        WN_DELETE_Tree(end_wn);
        WN_DELETE_Tree(step_wn);

        /* Move the original loop's body to the inner loop.
         * If there are SHARED COPYIN directives to process, insert
         * the REGION node between the two loops. */
        WN *inner_loop_body = WN_kid(inner_loop,4); // empty
        WN_kid(inner_loop,4) = WN_kid(loop,4);
        WN_kid(loop,4) = inner_loop_body;
        if (svar_region != NULL) {
            WN_INSERT_BlockLast(WN_kid2(svar_region), inner_loop);
            WN_INSERT_BlockLast(inner_loop_body, svar_region);
        } else {
            WN_INSERT_BlockLast(inner_loop_body, inner_loop);
        }

        /* Replace accesses to the original loop index with a new one:
         * <iterator loop index> +/- <inner loop index>
         * depending on the sign of step. */

        WN *loop_idx_ldid = WN_LdidScalar(idxv_st_idx);
        WN *inner_loop_idx_ldid = WN_LdidScalar(inner_idxv_st_idx);

        WN *replacement = (step_sz > 0) ?
            WN_Add(idxv_mtype, loop_idx_ldid, inner_loop_idx_ldid) :
            WN_Sub(idxv_mtype, loop_idx_ldid, inner_loop_idx_ldid);

        referenced_by_ldid_only = true;
        replace_ldid(WN_kid(inner_loop,4), idxv_st_idx, replacement);
        Is_True(referenced_by_ldid_only,
            ("Non-LDID access to the index variable of a loop "
             "tiled by a LOOPBLOCK directive.\n"));

        WN_DELETE_Tree(replacement);
    } else {
        /* If there are SHARED COPYIN directives to process, move
         * the original loop's body to the REGION node. */
        if (svar_region != NULL) {
            WN *region_body = WN_kid(svar_region,2);    // empty
            WN_kid(svar_region,2) = WN_kid(loop,4);
            WN_kid(loop,4) = region_body;
            WN_INSERT_BlockLast(region_body, svar_region);
        }
    }

    printf("done\n");

    return pragma;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

void
reset_handler_states_at_pu_end() {
    grid_dim_st_idx = ST_IDX_ZERO;
    blk_dim_st_idx = ST_IDX_ZERO;
}

/*** DAVID CODE END ***/
