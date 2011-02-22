/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_SHAPE_H_
#define _IPA_HC_SHAPE_H_

#include "defs.h"
#include "wn.h"

#include "cxx_memory.h"
#include "cxx_template.h"
#include "cxx_hash.h"


/*****************************************************************************
 *
 * Internal representation of a SHAPE directive
 *
 ****************************************************************************/

class HC_ARRAY_SHAPE_INFO
{
private:

    ST_IDX _st_idx;             // a pointer variable
    TY_IDX _elem_ty_idx;        // array element type

    UINT _ndims;                // dimensionality
    UINT *_dim_sz;              // each element is a ST_IDX or a constant
    BOOL *_is_dim_sz_const;     // decided by this boolean

    MEM_POOL *_pool;

    // used in <create_shape>.
    HC_ARRAY_SHAPE_INFO(ST_IDX st_idx, TY_IDX elem_ty_idx,
            UINT ndims, UINT *dim_sz, BOOL *is_dim_sz_const, MEM_POOL *pool);

public:

    /**
     * Create a shape based on the type of the given array variable.
     * Return NULL if not successful.
     */
    static HC_ARRAY_SHAPE_INFO* create_shape(ST_IDX arr_st_idx,
            MEM_POOL *pool);

    /**
     * Parse the given ARRSECTION node (in a SHAPE directive).
     */
    HC_ARRAY_SHAPE_INFO(WN *shape_wn, MEM_POOL *pool);

    /**
     * Deep-copy constructor using the given mem pool.
     */
    HC_ARRAY_SHAPE_INFO(const HC_ARRAY_SHAPE_INFO *orig, MEM_POOL *pool);

    ~HC_ARRAY_SHAPE_INFO() {}

    UINT get_ndims() { return _ndims; }
    TY_IDX get_elem_ty_idx() { return _elem_ty_idx; }
    ST_IDX get_sym() { return _st_idx; }

    /**
     * Construct an array type for this shape, and return a
     * pointer-to-this-array type.
     *
     * This function must be called in the procedure context.
     */
    TY_IDX create_dyn_array_ptr_type();

    /**
     * Replace the array symbol with <st_idx> and dimension size symbols from
     * <from_syms> to <to_syms>.
     *
     * Return TRUE if all non-global symbols are replaced and FALSE otherwise.
     *
     * This function is called by HC_FORMAL_SHAPE_ARRAY to map across calls.
     */
    BOOL replace_syms(ST_IDX st_idx,
            ST_IDX *from_syms, ST_IDX *to_syms, UINT n_syms);

    /**
     * ASSUME: both annotation data are in the same procedure context, which
     * must have been set up before calling this function.
     */
    BOOL equals(const HC_ARRAY_SHAPE_INFO *other) const;

    /**
     * ASSUME: the appropriate procedure context must be set up before calling
     * this function.
     */
    void print(FILE *fp);
};


/*****************************************************************************
 *
 * The shape context is organized by blocks. Each block is associated with a
 * hash table between symbols and a stack of shape information in this block.
 *
 * There are two types of shape information:
 * 1) HC_ARRAY_SHAPE_INFO*: actual array shape used in shape propagation
 * 2) ST*: symbol of the newly created dynamic array pointer, used in
 * transforming the procedure based on shape annotations
 *
 ****************************************************************************/

template <class CONTENT_TYPE>
class HC_SHAPE_CONTEXT
{
private:

    typedef STACK<CONTENT_TYPE*> SHAPE_STACK;
    typedef HASH_TABLE<ST_IDX, SHAPE_STACK*> SHAPE_PER_BLOCK_TABLE;
    typedef STACK<SHAPE_PER_BLOCK_TABLE*> SHAPE_TABLE;

    SHAPE_TABLE *_table;

    MEM_POOL *_pool;

public:

    HC_SHAPE_CONTEXT(MEM_POOL *pool)
    {
        _pool = pool;
        _table = CXX_NEW(SHAPE_TABLE(pool), pool);
    }

    ~HC_SHAPE_CONTEXT() {}

    // Called when a new BLOCK is met or has finished.
    void push_block() {
        _table->Push(CXX_NEW(SHAPE_PER_BLOCK_TABLE(307,_pool), _pool));
    }
    void pop_block() { _table->Pop(); }

    // Push the given shape info into the appropriate stack in the top block.
    void push_shape_info(ST_IDX st_idx, CONTENT_TYPE *info);

    // Search for the visible shape info for the given symbol.
    CONTENT_TYPE* find_shape_info(ST_IDX st_idx);
};

typedef HC_SHAPE_CONTEXT<HC_ARRAY_SHAPE_INFO> HC_ARRAY_SHAPE_CONTEXT;
typedef HC_SHAPE_CONTEXT<ST> HC_DYN_ARRAY_PTR_CONTEXT;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef DYN_ARRAY<HC_ARRAY_SHAPE_INFO*> HC_SHAPE_INFO_LIST;
typedef HASH_TABLE<ST_IDX, HC_ARRAY_SHAPE_INFO*> HC_SHAPE_INFO_MAP;

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

extern void HC_apply_shape_annot(IPA_NODE *node, MEM_POOL *tmp_pool);

extern void HC_remove_shape_dir(IPA_NODE *node);

/*****************************************************************************
 *
 * Invoke <transform_ptr_access_to_array> and then the pre-optimizer to update
 * the SUMMARY data structures.
 *
 ****************************************************************************/

extern void HC_promote_dynamic_arrays(IPA_NODE *node);

/*****************************************************************************
 *
 * For better CUDA code generation,
 *
 * 1) Demote each pointer-to-ARRAY formal to a pointer-to-element variable.
 *    Update the function prototype, the WHIRL tree, and the call WN nodes.
 *
 * 2) Demote each pointer-to-ARRAY local variable to a pointer-to-element
 *    variable.
 *
 * This function assumes that the call graph is correct and the
 * SUMMARY_CALLSITE IDs are still consistent, which is not the case after
 * kernel outlining.
 *
 ****************************************************************************/

extern void IPA_HC_demote_dyn_array_types(MEM_POOL *tmp_pool);

#endif  // _IPA_HC_SHAPE_H_

/*** DAVID CODE END ***/
