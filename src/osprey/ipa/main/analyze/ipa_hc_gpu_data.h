/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_GPU_DATA_H_
#define _IPA_HC_GPU_DATA_H_

#include "cxx_template.h"
#include "cxx_hash.h"

#include "ipa_hc_common.h"      // HC_SYM_MAP

class ACCESS_ARRAY;
class PROJECTED_REGION;
class IPA_EDGE;
class IPA_NODE;
class HC_GPU_VAR_INFO;
class HC_GDATA_IG_NODE;
class HC_LOOP_PART_INFO;
class HC_EXPR_PROP;
class HC_KERNEL_INFO;

/*****************************************************************************
 *
 * Information of an array section specified in the data directive
 *
 ****************************************************************************/

class HC_ARRSECTION_INFO
{
private:

    ST_IDX _arr_st_idx;     // either an array or a pointer-to-array
    TY_IDX _elem_ty_idx;    // array element type

    UINT _ndims;            // dimensionality
    WN **_dim_sz;           // original array's dimension sizes
    WN **_dim_lbnd;         // lower bounds of the section
    WN **_dim_ubnd;         // upper bounds of the section

    // temporary field for representing the section in a form that can be
    // compared with another section using existing code
    // It is explicitly set and cleared.
    PROJECTED_REGION *_section;

    // For each dimension, whether or not the section's range is the entire
    // array dimension
    BOOL *_full_dim_range;
    // the zero-base starting dimension index of the biggest contiguous
    // sub-section
    UINT _pivot_dim_idx;

    MEM_POOL *_pool;

    // used in <replace_syms>
    // The return value has the same meaning as <replace_syms>.
    BOOL replace_syms_walker(WN *wn,
            ST_IDX *from_syms, ST_IDX *to_syms, UINT n_syms);

    // used by <get_lower_bound> and <get_upper_bound>
    WN* get_bound(BOOL is_lower) const;

public:

    /**
     * This constructor parses the given ARRSECTION node, and fills all the
     * internal fields except <_section>.
     *
     * This function assumes that the parent node's IPA context is set up.
     */
    HC_ARRSECTION_INFO(WN *section_wn, MEM_POOL *pool);

    /**
     * Deep-copy constructor using the given mem pool.
     */
    HC_ARRSECTION_INFO(const HC_ARRSECTION_INFO* orig, MEM_POOL *pool);

    ST_IDX get_arr_sym() const { return _arr_st_idx; }
    TY_IDX get_elem_type() const { return _elem_ty_idx; }
    UINT get_num_dim() const { return _ndims; }
    UINT get_pivot_dim_idx() const { return _pivot_dim_idx; }

    BOOL is_section_full() const
    {
        for (UINT i = 0; i < _ndims; ++i)
        {
            if (! _full_dim_range[i]) return FALSE;
        }
        return TRUE;
    }

    WN* get_dim_lbnd(UINT idx) const
    {
        Is_True(idx < _ndims, (""));
        return _dim_lbnd[idx];
    }
    WN* get_dim_ubnd(UINT idx) const
    {
        Is_True(idx < _ndims, (""));
        return _dim_ubnd[idx];
    }

    /**
     * Construct a fresh ARRAY node to represent the lower and upper bound.
     * The dimension size kids hold the original array's dimension sizes.
     */
    WN* get_lower_bound() const;
    WN* get_upper_bound() const;

    /**
     * Given the ACCESS_ARRAY representation of the lower and the upper
     * bounds, create the internal representation of the array section in
     * PROJECTED_REGION.
     */
    void set_section(ACCESS_ARRAY *lbnd, ACCESS_ARRAY *ubnd, mUINT8 depth); 

    PROJECTED_REGION* get_section() const { return _section; }
    void clear_section() { _section = NULL; }

    void deep_copy_section(MEM_POOL *pool);

    /**
     * Return the size of the given dimension of the original array.
     * THIS IS A REFERENCE.
     */
    WN* get_orig_dim_sz(UINT idx) const;

    /**
     * Return the size of the given dimension, computed as ubnd - lbnd + 1.
     * THIS IS A NEW INSTANCE.
     */
    WN* get_dim_sz(UINT idx) const;

    /**
     * Return the size of this section in bytes, i.e.
     * num_elements * element_sz_in_bytes
     * THIS IS A NEW INSTANCE.
     */
    WN* get_section_sz() const;

    /**
     * Project this section with respect to the given variable, which is
     * between <var_lbnd_wn> and <var_ubnd_wn>.
     */
    BOOL project(ST_IDX st_idx, WN_OFFSET ofst,
            WN *var_lbnd_wn, WN *var_ubnd_wn);

    /**
     * Replace the main variable symbol with <st_idx> and symbols involved in
     * lower/upper bound and original dimension size, from <from_syms> to
     * <to_syms>.
     *
     * Return TRUE if all non-global symbols are replaced and FALSE otherwise.
     *
     * This function is called by HC_GPU_DATA::replace_syms.
     */
    BOOL replace_syms(ST_IDX st_idx,
            ST_IDX *from_syms, ST_IDX *to_syms, UINT n_syms);

    /**
     * Retrieve the actual ST object for each ST_IDX referenced in this
     * instance.
     */
    void build_st_table(HASH_TABLE<ST_IDX,ST*> *st_tbl) const;

    /**
     * Replace auxiliary variables referenced in the ALLOC section with new
     * formals.
     *
     * ASSUME: the appropriate procedure context has been set up.
     */
    void replace_idxvs_with_formals(HC_SYM_MAP *new_formal_map,
            const HASH_TABLE<ST_IDX, ST*> *st_tbl);

    // Compute the index of the leftmost dimension such that all dimensions to
    // the right are "full" w.r.t. the given section. This index is at least 0
    // even though D_0 might be "full" too.
    //
    UINT compute_pivot_dim_idx(const HC_ARRSECTION_INFO *other) const;

    /**
     * Compare everything except the main symbol.
     *
     * ASSUME: both annotation data are in the same procedure context, which
     * must have been set up before calling this function.
     */
    BOOL equals(const HC_ARRSECTION_INFO *other) const;

    /**
     * This routine can only be called by HC_GPU_DATA::print, which sets up
     * the IPA_NODE_CONTEXT properly.
     */
    void print(FILE *fp) const;
};


/*****************************************************************************
 *
 * Data structure for the data to be brought into the global/constant memory
 *
 ****************************************************************************/

typedef enum
{
    HC_GLOBAL_DATA,         // global memory variable
    HC_CONSTANT_DATA,       // constant memory variable
    HC_SHARED_DATA,         // shared memory variable
    HC_LAST_GPU_DATA_TYPE   // also the size of this enum
} HC_GPU_DATA_TYPE;

// a convenience function for debugging/error output
inline const char* HC_gpu_data_type_name(HC_GPU_DATA_TYPE dtype)
{
    switch (dtype)
    {
        case HC_GLOBAL_DATA: return "global";
        case HC_CONSTANT_DATA: return "constant";
        case HC_SHARED_DATA: return "shared";
        default: Is_True(FALSE, (""));
    }

    return NULL;    // should never reach here
}

typedef DYN_ARRAY<HC_EXPR_PROP*> HC_EXPR_PROP_LIST;

class HC_GPU_DATA
{
protected:

    // Two fields that are used to identify the original data directive this
    // instance is created from:
    //
    // - the procedure that contains the directive
    // - the pre-order traversal index in the group of ALLOC/COPYIN directives
    //   (GLOBAL/CONSTANT as one group, and SHARED as the other)
    //
    IPA_NODE *_orig_proc_node;
    UINT _dir_id;

    HC_GPU_DATA_TYPE _type;

    // the host variable symbol (array or non-array)
    ST_IDX _st_idx;

#define HC_GDATA_COPYIN             0x0001
#define HC_GDATA_COPYIN_NOBNDCHECK  0x0002
#define HC_GDATA_CLEAR              0x0004
#define HC_GDATA_COPYOUT            0x0008
#define HC_GDATA_COPYOUT_NOBNDCHECK 0x0010

    UINT _flags;

    // array section allocated in the GPU memory (or NULL)
    HC_ARRSECTION_INFO *_alloc_section;
    // array section copied from the host memory to the GPU memory
    HC_ARRSECTION_INFO *_copyin_section;
    // array section copied from the GPU memory to the host memory
    HC_ARRSECTION_INFO *_copyout_section;

    // For a SHARED directive, store the corresponding GLOBAL directive (local
    // or propagated). Otherwise NULL.
    HC_GPU_DATA *_partner_gdata;

    // For a SHARED directive, store a list of properties, one for each
    // enclosing loop.
    HC_EXPR_PROP_LIST *_lp_idxv_props;

    // For a SHARED directive, keep track of the kernel context.
    HC_KERNEL_INFO *_kinfo;

    // Data structure for code generation of the GPU variable
    HC_GPU_VAR_INFO *_gvar_info;

    // A transient field for constructing the interference graph if this is a
    // constant memory variable
    HC_GDATA_IG_NODE *_ig_node;

    MEM_POOL *_pool;        // used to allocate the array regions

    void set_do_copyin() { _flags |= HC_GDATA_COPYIN; }
    void set_do_copyin_nobndcheck() { _flags |= HC_GDATA_COPYIN_NOBNDCHECK; }
    void set_do_clear() { _flags |= HC_GDATA_CLEAR; }
    void set_do_copyout() { _flags |= HC_GDATA_COPYOUT; }
    void set_do_copyout_nobndcheck() { _flags |= HC_GDATA_COPYOUT_NOBNDCHECK; }

public:

    /**
     * Parses the given pragma (along with pragmas after it),
     * and fills all the fields.
     */
    HC_GPU_DATA(WN *pragma_wn, IPA_NODE *node, UINT dir_id, MEM_POOL *pool);

    /**
     * Deep-copy constructor using the given mem pool.
     */
    HC_GPU_DATA(const HC_GPU_DATA *orig, MEM_POOL *pool);

    ~HC_GPU_DATA()
    {
        // Do nothing. Popping the mem pool will do the work.
    }

    /**
     * The given WN node is the one within the copyout XPRAGMA.
     */
    void parse_copyout_dir(WN *wn);

    IPA_NODE* get_orig_proc() const { return _orig_proc_node; }
    UINT get_dir_id() const { return _dir_id; }

    HC_GPU_DATA_TYPE get_type() const { return _type; }
    const char* get_type_name() const { return HC_gpu_data_type_name(_type); }

    ST_IDX get_symbol() const { return _st_idx; }

    TY_IDX get_elem_type() const
    {
        return (_alloc_section == NULL) ?
            ST_type(_st_idx) : _alloc_section->get_elem_type();
    }

    // A convenience method that creates the type of the GPU memory variable.
    // NOTE: the type of a constant memory variable may be different from that
    // of <cmem>.
    // 
    TY_IDX create_gvar_type() const
    {
        TY_IDX ty_idx = get_elem_type();
        // Drop the "constant" attribute.
        Clear_TY_is_const(ty_idx);
        return Make_Pointer_Type(ty_idx);
    }

    /**
     * Compute the size of the GPU variable in bytes.
     * THIS IS A NEW INSTANCE.
     */
    WN* compute_size() const;

    BOOL is_arr_section() const { return (_alloc_section != NULL); }
    HC_ARRSECTION_INFO* get_alloc_section() const { return _alloc_section; }

    BOOL do_copyin() const { return _flags & HC_GDATA_COPYIN; }
    BOOL do_copyin_bndcheck() const
    {
        return ((_flags & HC_GDATA_COPYIN_NOBNDCHECK) == 0);
    }
    HC_ARRSECTION_INFO* get_copyin_section() const { return _copyin_section; }

    BOOL do_clear() const { return _flags & HC_GDATA_CLEAR; }

    BOOL do_copyout() const { return _flags & HC_GDATA_COPYOUT; }
    BOOL do_copyout_bndcheck() const
    {
        return ((_flags & HC_GDATA_COPYOUT_NOBNDCHECK) == 0);
    }
    HC_ARRSECTION_INFO* get_copyout_section() const
    {
        return _copyout_section;
    }
    void set_copyout_section(HC_ARRSECTION_INFO* hai)
    {
        _copyout_section = hai;
    }

    /**
     * Clear PROJECTED_REGION fields in the section specifications.
     */
    void clear_proj_regions()
    {
        if (_alloc_section != NULL) _alloc_section->clear_section();
        Is_True(_copyin_section == NULL
                || _copyin_section->get_section() == NULL, (""));
        Is_True(_copyout_section == NULL
                || _copyout_section->get_section() == NULL, (""));
    }

    BOOL is_alloc_section_full() const
    {
        Is_True(_alloc_section != NULL, (""));
        return _alloc_section->is_section_full();
    }

    /**
     * Project the ALLOC and COPYIN/COPYOUT sections based on the given range
     * of the loop index variable.
     * USE FOR SHARED MEMORY DATA
     */
    void add_idxv_range(const HC_LOOP_PART_INFO *lpi);

    HC_GPU_VAR_INFO* create_gvar_info(const HC_GPU_VAR_INFO *orig = NULL);
    HC_GPU_VAR_INFO* get_gvar_info() const { return _gvar_info; }

    HC_GDATA_IG_NODE* get_ig_node() const { return _ig_node; }
    void set_ig_node(HC_GDATA_IG_NODE *ig_node) { _ig_node = ig_node; }

    HC_GPU_DATA* get_partner_gdata() const { return _partner_gdata; }
    void set_partner_gdata(HC_GPU_DATA *gdata) { _partner_gdata = gdata; }

    HC_EXPR_PROP_LIST* create_lp_idxv_prop_list();
    HC_EXPR_PROP_LIST* get_lp_idxv_prop_list() const
    {
        return _lp_idxv_props;
    }
    void reset_lp_idxv_prop_list()
    {
        if (_lp_idxv_props == NULL) return;
        CXX_DELETE(_lp_idxv_props, _pool);
        _lp_idxv_props = NULL;
    }

    HC_KERNEL_INFO* get_kernel_info() const { return _kinfo; }
    void set_kernel_info(HC_KERNEL_INFO *kinfo)
    {
        Is_True(kinfo != NULL && _kinfo == NULL, (""));
        _kinfo = kinfo;
    }

    /**
     * Replace the main variable symbol with <st_idx> and symbols involved in
     * ALLOC section specification from <from_syms> to <to_syms>.
     *
     * Return TRUE if all non-global symbols are replaced and FALSE otherwise.
     *
     * This function is called by HC_FORMAL_GPU_DATA_ARRAY to map across calls.
     */
    BOOL replace_syms(ST_IDX st_idx,
            ST_IDX *from_syms, ST_IDX *to_syms, UINT n_syms);

    /**
     * Retrieve the actual ST object for each ST_IDX referenced in this
     * instance.
     */
    void build_st_table(HASH_TABLE<ST_IDX,ST*> *st_tbl) const;

    /**
     * Replace auxiliary variables referenced in the ALLOC section with new
     * formals in the given procedure.
     */
    void replace_idxvs_with_formals(IPA_NODE *node,
            HC_SYM_MAP *new_formal_map,
            const HASH_TABLE<ST_IDX, ST*> *st_tbl);

    // Are the two instances coming from the same directive?
    //
    // NOTE: if this function returns TRUE, <equals> must return TRUE too.
    //
    BOOL have_same_origin(const HC_GPU_DATA *other) const;

    /**
     * Compare the directive type, data type and the section specification.
     * We do not care whether or not they come from the same directive.
     *
     * ASSUME: both annotation data are in the same procedure context, which
     * must have been set up before calling this function.
     */
    BOOL equals(const HC_GPU_DATA *other) const;

    /**
     * ASSUME: the appropriate procedure context must be set up before calling
     * this function.
     */
    void print(FILE *fp) const;
};

#if 0
/*****************************************************************************
 *
 * Data structure for data to be brought into the shared memory
 *
 ****************************************************************************/

// a TRIPLET node for each visible loop index variable associated with a
// LOOP_PARTITION directive
typedef HASH_TABLE<ST_IDX, WN> HC_IDXV_RANGE_TABLE;

class HC_SHARED_DATA : public HC_GPU_DATA
{
private:

    // corresponding global memory data
    HC_GPU_DATA *_global_data;

    // for determining the actual allocation/copy section
    HC_IDXV_RANGE_TABLE *_idxv_range_table;

public:

    HC_SHARED_DATA(WN *pragma_wn, MEM_POOL *pool);
};
#endif

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Data structure for holding all visible GPU data, organized by host variable
 * symbol. It is essentially a snapshot of the stack (defined later) and is
 * used in HC_KERNEL_INFO.
 *
 ****************************************************************************/

typedef HASH_TABLE<ST_IDX, HC_GPU_DATA*> HC_VISIBLE_GPU_DATA;
typedef HASH_TABLE_ITER<ST_IDX, HC_GPU_DATA*> HC_VISIBLE_GPU_DATA_ITER;

/*****************************************************************************
 *
 * Data structure for a stack of GPU data, organized by host variable symbol.
 *
 ****************************************************************************/

typedef STACK<HC_GPU_DATA*> GPU_DATA_PER_SYMBOL_STACK;
typedef HASH_TABLE<ST_IDX, GPU_DATA_PER_SYMBOL_STACK*> GPU_DATA_STACK;

class HC_GPU_DATA_STACK
{
protected:

    GPU_DATA_STACK* _stack;

    MEM_POOL *_pool;

public:

    HC_GPU_DATA_STACK(MEM_POOL *pool)
    {
        Is_True(pool != NULL, (""));
        _pool = pool;
        _stack = CXX_NEW(GPU_DATA_STACK(41,pool), pool);
    }

    ~HC_GPU_DATA_STACK() {}

    /**
     * Push the given GPU data record onto the stack. It will be put in the
     * right stack according to its host variable symbol.
     */
    void push(HC_GPU_DATA *hgd);

    /**
     * Remove and return the top data record in the stack for the given
     * data type and the host variable symbol.
     */
    HC_GPU_DATA* pop(HC_GPU_DATA_TYPE dtype, ST_IDX st_idx);

    /**
     * Return the top data record in the stack for the given host variable
     * symbol.
     * This is used in collecting visible directive for a kernel region.
     */
    HC_GPU_DATA* peek(ST_IDX st_idx) const;

    /**
     * Return the top data record in the stack for the given host variable
     * symbol and the data type.
     * This is used to find matching HC_GPU_DATA for copyout.
     */
    HC_GPU_DATA* peek(ST_IDX st_idx, HC_GPU_DATA_TYPE dtype) const;

    /**
     * Use the given mempool to create a snapshot of the current stack.
     */
    HC_VISIBLE_GPU_DATA* top(MEM_POOL *pool) const;
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef DYN_ARRAY<HC_GPU_DATA*> HC_GPU_DATA_LIST;
typedef HASH_TABLE<HC_GPU_DATA*, HC_GPU_DATA*> HC_GPU_DATA_MAP;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Lower GLOBAL/CONSTANT directives to CUDA code.
 *
 ****************************************************************************/

extern void HC_handle_data_directives(IPA_NODE *node);

/*****************************************************************************
 *
 * Create a HC_GPU_DATA for each SHARED directive (ALLOC and COPYOUT), and
 * store it in the SHARED directive list in the given node, if the procedure
 * is K-/IK-.
 *
 ****************************************************************************/

extern void HC_parse_shared_directives(IPA_NODE *node, MEM_POOL *pool);

extern void HC_handle_shared_directives(IPA_NODE *node);

#endif  // _IPA_HC_GPU_DATA_H_

/*** DAVID CODE END ***/
