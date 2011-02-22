/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_KERNEL_H_
#define _IPA_HC_KERNEL_H_

#include "wn.h"

#include "cxx_hash.h"

#include "ipa_section.h"
#include "ipa_hc_annot.h"           // HC_ANNOT_DATA
#include "ipa_hc_gpu_data.h"

class IPA_NODE;
class IPA_HC_ANNOT;
class HC_FORMAL_GPU_DATA_ARRAY;     // ipa_hc_gpu_data_prop.h

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// used in cloning HC_KERNEL_INFO
typedef HASH_TABLE<WN*,WN*> HC_WN_MAP;

typedef HASH_TABLE<IPA_HC_ANNOT*, HC_GPU_DATA*> HC_MATCHED_GDATA_TABLE;
typedef HASH_TABLE_ITER<IPA_HC_ANNOT*, HC_GPU_DATA*> HC_MATCHED_GDATA_ITER;

class HC_ACCESS_INFO : public SLIST_NODE
{
protected:

// flags common to scalar and array accesses
#define HC_ACCESS_REF   0x01
#define HC_ACCESS_MOD   0x02
#define HC_REDIRECTION_TARGET_FOUND    0x04 // object internal consistency

    UINT _flags;

    // GPU variable to be redirected to
    // When HC_REDIRECTION_TARGET_FOUND is not set, <_temp> is used.
    // When HC_REDIRECTION_TARGET_FOUND is set, <_final> is used.
    union {
        HC_MATCHED_GDATA_TABLE *_temp;
        HC_GPU_DATA *_final;
    } _gpu_data;

    DECLARE_SLIST_NODE_CLASS(HC_ACCESS_INFO);

public:

    HC_ACCESS_INFO() { _flags = 0; _gpu_data._temp = NULL; }

    // Shallow-copy GPU data.
    HC_ACCESS_INFO(const HC_ACCESS_INFO *orig)
    {
        _flags = orig->_flags;
        _gpu_data = orig->_gpu_data;
    }

    BOOL is_ref() const { return (_flags & HC_ACCESS_REF); }
    BOOL is_mod() const { return (_flags & HC_ACCESS_MOD); }
    void set_access_type(BOOL is_ref) {
        _flags |= (is_ref ? HC_ACCESS_REF : HC_ACCESS_MOD);
    }

    // used to maintain internal consistency.
    BOOL is_redirection_target_found() const {
        return (_flags & HC_REDIRECTION_TARGET_FOUND);
    }
    void set_redirection_target_found() {
        _flags |= HC_REDIRECTION_TARGET_FOUND;
    }

    HC_GPU_DATA* get_gpu_data() const
    {
        Is_True(is_redirection_target_found(), (""));
        return _gpu_data._final;
    }
    BOOL replace_gpu_data(const HC_GPU_DATA_MAP *map)
    {
        Is_True(is_redirection_target_found(), (""));
        if (map == NULL || _gpu_data._final == NULL) return FALSE;

        HC_GPU_DATA *gdata = map->Find(_gpu_data._final);
        // It does not have to be in the map (e.g. a formal's GPU data).
        if (gdata != NULL) _gpu_data._final = gdata;

        return (gdata != NULL);
    }

    HC_MATCHED_GDATA_TABLE* get_gpu_data_table() const
    {
        Is_True(!is_redirection_target_found(), (""));
        return _gpu_data._temp;
    }

    void set_gpu_data_table(HC_MATCHED_GDATA_TABLE *gdata_tbl)
    {
        Is_True(!is_redirection_target_found(), (""));
        Is_True(gdata_tbl != NULL, (""));
        _gpu_data._temp = gdata_tbl;
    }
};

class HC_SCALAR_INFO : public HC_ACCESS_INFO
{
private:

    ST_IDX _st_idx;         // scalar symbol
    WN_OFFSET _offset;      // offset for aggregate symbols (struct)

    // flag attributes (must start from 0x100)
#define HSI_PASSED_AS_KERNEL_PARAM      0x100

public:

    HC_SCALAR_INFO(ST_IDX st_idx, WN_OFFSET offset, BOOL is_ref)
    {
        _st_idx = st_idx; _offset = offset;
        set_access_type(is_ref);
    }

    // shallow copy of GPU data
    HC_SCALAR_INFO(const HC_SCALAR_INFO *orig)
        : HC_ACCESS_INFO(orig)
    {
        _st_idx = orig->_st_idx;
        _offset = orig->_offset;
    }

    ~HC_SCALAR_INFO() {}

    // Select the GPU data for the given annotation.
    void finalize_gpu_data(IPA_HC_ANNOT *annot,
            const HC_GPU_DATA_MAP *gdata_map = NULL)
    {
        Is_True(!is_redirection_target_found(), (""));
        Is_True(_gpu_data._temp != NULL, (""));

        // If the selected GPU data is NULL, we don't know if it is due to
        // missing annotation or that the scalar is passed by parameter.
        // Let's live with this for now.
        HC_GPU_DATA *gdata = _gpu_data._temp->Find(annot);
        if (gdata == NULL) {
            set_passed_as_kernel_param();
        } else {
            set_gpu_data(gdata);
            replace_gpu_data(gdata_map);
        }
    }

    BOOL equals(const HC_SCALAR_INFO& other) const
    {
        return (_st_idx == other._st_idx) && (_offset == other._offset);
    }

    ST_IDX get_symbol() { return _st_idx; }
    WN_OFFSET get_offset() { return _offset; }
    
    BOOL passed_as_kernel_param()
    {
        Is_True(is_redirection_target_found(), (""));
        return (_flags & HSI_PASSED_AS_KERNEL_PARAM);
    }

    void set_passed_as_kernel_param()
    {
        _flags |= HSI_PASSED_AS_KERNEL_PARAM;
        _gpu_data._final = NULL;
        set_redirection_target_found();
    }

    void set_gpu_data(HC_GPU_DATA *gdata)
    {
        Is_True(gdata != NULL, (""));
        _gpu_data._final = gdata;
        _flags &= (~HSI_PASSED_AS_KERNEL_PARAM);
        set_redirection_target_found();
    }
};

class HC_SCALAR_INFO_LIST : public SLIST
{
private:

    HC_SCALAR_INFO_LIST(const HC_SCALAR_INFO_LIST&);
    HC_SCALAR_INFO_LIST& operator = (const HC_SCALAR_INFO_LIST&);

    DECLARE_SLIST_CLASS(HC_SCALAR_INFO_LIST, HC_SCALAR_INFO);

public:

    ~HC_SCALAR_INFO_LIST() {}
};

class HC_SCALAR_INFO_ITER : public SLIST_ITER
{
private:

    DECLARE_SLIST_ITER_CLASS(HC_SCALAR_INFO_ITER,
            HC_SCALAR_INFO, HC_SCALAR_INFO_LIST);
};


/*****************************************************************************
 *
 * Data structure for MOD/REF info of array sections in a kernel region
 *
 * The HC_ARRAY_INFOs are grouped into HC_ARRAY_SYM_INFOs based on the array
 * symbol.
 *
 ****************************************************************************/

class HC_ARRAY_INFO : public HC_ACCESS_INFO
{
private:

    // The WHIRL node for this array reference, which for now, can be an
    // ILOAD, ISTORE or CALL node.
    //
    // If it is an ILOAD/ISTORE, <actual_idx> must be -1.
    // If it is a CALL, <actual_idx> is a zero-based index of the actual
    // parameter or -1 for global variable accesses.
    //
    WN *_wn;
    INT _actual_idx;

    // projected array access, NULL if messy region
    // They are temporary fields, and will be set to NULL once the
    // corresponding GPU data is determined.
    PROJECTED_REGION *_ref_region;
    PROJECTED_REGION *_mod_region;

public:

    HC_ARRAY_INFO(WN *access,
            PROJECTED_REGION *ref_region, PROJECTED_REGION *mod_region)
    {
        Is_True(WN_operator(access) == OPR_ILOAD
                || WN_operator(access) == OPR_ISTORE, (""));
        _wn = access; _actual_idx = -1;
        _ref_region = ref_region; _mod_region = mod_region;
    }

    HC_ARRAY_INFO(WN *call, INT actual_idx,
            PROJECTED_REGION *ref_region, PROJECTED_REGION *mod_region)
    {
        Is_True(OPERATOR_is_call(WN_operator(call)), (""));
        _wn = call; _actual_idx = actual_idx;
        _ref_region = ref_region; _mod_region = mod_region;
    }

    /**
     * This copy constructor replaces the WN reference in <orig> with the new
     * one in <ww_map>. It shallow-copies the PROJECTED_REGION nodes.
     */
    HC_ARRAY_INFO(const HC_ARRAY_INFO *orig, const HC_WN_MAP *ww_map)
        : HC_ACCESS_INFO(orig)
    {
        _wn = ww_map->Find(orig->_wn);
        Is_True(_wn != NULL, (""));
        printf("HC_ARRAY_INFO: replaced %p with %p\n", orig->_wn, _wn);

        _actual_idx = orig->_actual_idx;

        _ref_region = orig->_ref_region;
        _mod_region = orig->_mod_region;
    }

    ~HC_ARRAY_INFO() {}

    WN* get_wn() const { return _wn; }
    INT get_actual_index() const { return _actual_idx; }

    PROJECTED_REGION* get_region(BOOL is_ref) const
    {
        return is_ref ? _ref_region : _mod_region;
    }

    void clear_region() { _ref_region = _mod_region = NULL; }

    //
    // Update the REF/MOD regions only when the given ones provide more
    // information than the existing ones.
    //
    // This is invoked by HC_ARRAY_SYM_INFO:add_access.
    //
    void update_regions(PROJECTED_REGION *ref_pr, PROJECTED_REGION *mod_pr)
    {
        if (_ref_region == NULL && ref_pr != NULL) _ref_region = ref_pr;
        if (_mod_region == NULL && mod_pr != NULL) _mod_region = mod_pr;
    }

    /**
     * Relace the REF and MOD sections with a new copy allocated using the
     * given mem pool.
     */
    void deep_copy_region(MEM_POOL *pool)
    {
        if (_ref_region != NULL) {
            _ref_region = _ref_region->create_deep_copy(pool);
        }
        if (_mod_region != NULL) {
            _mod_region = _mod_region->create_deep_copy(pool);
        }
    }

    void finalize_gpu_data(IPA_HC_ANNOT *annot,
            const HC_GPU_DATA_MAP *gdata_map = NULL)
    {
        Is_True(!is_redirection_target_found(), (""));
        Is_True(_gpu_data._temp != NULL, (""));

        HC_GPU_DATA *gdata = _gpu_data._temp->Find(annot);
        Is_True(gdata != NULL, (""));
        set_gpu_data(gdata);

        replace_gpu_data(gdata_map);
    }

    void set_gpu_data(HC_GPU_DATA *gdata)
    {
        Is_True(gdata != NULL, (""));
        _gpu_data._final = gdata;
        set_redirection_target_found();
    }

    /**
     * ASSUME: the appropriate procedure context must be set up before calling
     * this function.
     */
    void print(FILE *fp);
};

class HC_ARRAY_INFO_LIST : public SLIST
{
private:

    HC_ARRAY_INFO_LIST(const HC_ARRAY_INFO_LIST&);
    HC_ARRAY_INFO_LIST& operator = (const HC_ARRAY_INFO_LIST&);

    DECLARE_SLIST_CLASS(HC_ARRAY_INFO_LIST, HC_ARRAY_INFO);

public:

    ~HC_ARRAY_INFO_LIST() {}
};

class HC_ARRAY_INFO_ITER : public SLIST_ITER
{
private:

    DECLARE_SLIST_ITER_CLASS(HC_ARRAY_INFO_ITER,
            HC_ARRAY_INFO, HC_ARRAY_INFO_LIST);
};

class HC_ARRAY_SYM_INFO : public SLIST_NODE
{
private:

    ST_IDX _arr_st_idx;             // an array or ptr-to-array symbol
    HC_ARRAY_INFO_LIST _arr_info;   // list of sections of this symbol

    MEM_POOL *_mem_pool;

    DECLARE_SLIST_NODE_CLASS(HC_ARRAY_SYM_INFO);

public:

    HC_ARRAY_SYM_INFO(ST_IDX arr_st_idx, MEM_POOL *mem_pool)
    {
        _arr_st_idx = arr_st_idx;
        _mem_pool = mem_pool;
    }

    ~HC_ARRAY_SYM_INFO()
    {
        // Let the client pop the mem pool.
    }

    ST_IDX arr_sym() const { return _arr_st_idx; }
    HC_ARRAY_INFO_LIST* get_arr_info_list() { return &_arr_info; }

    /**
     * Set the PROJECTED_REGION field in all containing HC_ARRAY_INFO to NULL.
     */
    void clear_proj_regions()
    {
        HC_ARRAY_INFO_ITER haii(&_arr_info);
        for (HC_ARRAY_INFO *hai = haii.First(); !haii.Is_Empty();
                hai = haii.Next()) hai->clear_region();
    }

    void deep_copy_proj_regions(MEM_POOL *pool)
    {
        HC_ARRAY_INFO_ITER haii(&_arr_info);
        for (HC_ARRAY_INFO *hai = haii.First(); !haii.Is_Empty();
                hai = haii.Next()) hai->deep_copy_region(pool);
    }

    void add_access(WN *access_wn,
            PROJECTED_REGION *ref_pr, PROJECTED_REGION *mod_pr)
    {
        add_access(access_wn, -1, ref_pr, mod_pr);
    }

    // Here, if <access_wn> is an array access, <actual_idx> must be -1.
    // If <access_wn> is a function call, <actual_idx> must be the index
    // of the actual parameter or -1 for global variable accesses.
    //
    void add_access(WN *access_wn, INT actual_idx,
            PROJECTED_REGION *ref_pr, PROJECTED_REGION *mod_pr);

    /**
     * ASSUME: the appropriate procedure context must be set up before calling
     * this function.
     */
    void print(FILE *fp = stderr, INT indent = 0);
};

class HC_ARRAY_SYM_INFO_ITER;

/* This should really be a map <array symbol, HC_ARRAY_INFO>. */
class HC_ARRAY_SYM_INFO_LIST : public SLIST
{
private:

    HC_ARRAY_SYM_INFO_LIST(const HC_ARRAY_SYM_INFO_LIST&);
    HC_ARRAY_SYM_INFO_LIST& operator = (const HC_ARRAY_SYM_INFO_LIST&);

    DECLARE_SLIST_CLASS(HC_ARRAY_SYM_INFO_LIST, HC_ARRAY_SYM_INFO);

public:

    ~HC_ARRAY_SYM_INFO_LIST() {}
};

class HC_ARRAY_SYM_INFO_ITER : public SLIST_ITER
{
private:

    DECLARE_SLIST_ITER_CLASS(HC_ARRAY_SYM_INFO_ITER,
            HC_ARRAY_SYM_INFO, HC_ARRAY_SYM_INFO_LIST);
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

class HC_KERNEL_INFO
{
    // We actually schedule in the unit of half-warps, as opposed to warps.
    static const UINT _warp_sz = 16;

private:

    ST_IDX _kfunc_st_idx;   // kernel function symbol

    // flags
#define HC_KINFO_INCOMPLETE_ARRAY_INFO  0x02

    UINT _flags;

    IPA_NODE *_knode;       // kernel function node (once created)

    // virtual block space
    UINT _n_vgrid_dims;
    WN **_vgrid_dims;
    // index of a particular dimension of the virtual block space (for the
    // current thread), in terms of blockIdx and threadIdx
    WN **_vblk_idx;

    // virtual thread space
    UINT _n_vblk_dims;
    WN **_vblk_dims;
    // index of a particular dimension of the virtual thread space (for the
    // current thread), in terms of blockIdx and threadIdx
    WN **_vthr_idx;

    // physical tblock and thread space (in the order of x,y,z)
    WN* _grid_dims[3];      // TODO: isn't it 2-D?
    WN* _blk_dims[3];

    // The following fields are w.r.t. a thread block.
    UINT _n_warps;          // 0 if not computed yet
    WN *_warp_id_wn;        // NULL if not computed yet
    WN *_id_within_warp_wn; // NULL if not computed yet

    // MOD/REF info of data in the kernel region
    HC_SCALAR_INFO_LIST _scalar_access;
    HC_ARRAY_SYM_INFO_LIST _arr_access;

    // Whether or not each procedure is called inside this kernel region
    BOOL *_is_proc_called;
    UINT _n_procs;

    // a transient field that stores a list of HC_GPU_DATAs for all
    // SHARED directives reachable from this kernel region
    //
    // This facilitates shared memory allocation for the kernel region.
    //
    HC_GPU_DATA_LIST *_sdata_list;

    // the amount of (dynamically-allocated) shared memory needed by this
    // kernel region (in bytes)
    UINT _smem_size;

    // a list of variables accessed in the kernel region
    //
    // This list is generated by <get_kernel_params> during kernel outlining.
    // The variable symbols are those in the parent procedure of this kernel
    // region, and therefore represent the actuals to the kernel invocation.
    // The corresponding formals of the kernel function should have the same
    // ST_IDX's, except for global symbols, where new formals have been
    // created to replace them. This happens in <HC_outline_kernel>.
    //
    HC_SYM_LIST *_kparams;

    MEM_POOL *_pool;

    HC_ARRAY_SYM_INFO* get_arr_sym(ST_IDX arr_st_idx,
            BOOL create_if_not_existed);

    // Compute and validate <_n_warps>, and construct expressions for
    // <_warp_id_wn> and <id_within_warp_wn>.
    void gen_warp_info();

public:

    // Parse a KERNEL directive.
    HC_KERNEL_INFO(WN *kregion, MEM_POOL *pool);

    /**
     * Copy constructor
     * - makes a deep copy of <orig>
     * - replace old WN node reference with new ones, using <ww_map>
     */
    HC_KERNEL_INFO(HC_KERNEL_INFO *orig,
            const HC_WN_MAP *ww_map, MEM_POOL *pool);

    ~HC_KERNEL_INFO() {}


    void init_scalar_info_iter(HC_SCALAR_INFO_ITER *si_iter)
    {
        si_iter->Init(&_scalar_access);
    }

    void init_arr_region_info_iter(HC_ARRAY_SYM_INFO_ITER *asi_iter)
    {
        asi_iter->Init(&_arr_access);
    }

    /**
     * Select the GPU data for the given annotation, and replace any old
     * HC_GPU_DATA references with the new ones in the map.
     */
    void finalize_gpu_data(IPA_HC_ANNOT *annot,
            const HC_GPU_DATA_MAP *gdata_map = NULL);

    void process_grid_geometry();

    BOOL has_incomplete_array_info() const
    {
        return _flags & HC_KINFO_INCOMPLETE_ARRAY_INFO;
    }
    void set_has_incomplete_array_info()
    {
        _flags |= HC_KINFO_INCOMPLETE_ARRAY_INFO;
    }

    ST_IDX get_kernel_sym() const { return _kfunc_st_idx; }
    void set_kernel_sym(ST_IDX st_idx) { _kfunc_st_idx = st_idx; }

    UINT get_vgrid_dims() const { return _n_vgrid_dims; }
    WN* get_vgrid_dim(UINT idx) const
    {
        Is_True(idx < _n_vgrid_dims, (""));
        return _vgrid_dims[idx];
    }
    WN* get_vblk_idx(UINT idx) const
    {
        Is_True(idx < _n_vgrid_dims, (""));
        return _vblk_idx[idx];
    }

    UINT get_vblk_dims() const { return _n_vblk_dims; }
    WN* get_vblk_dim(UINT idx) const
    {
        Is_True(idx < _n_vblk_dims, (""));
        return _vblk_dims[idx];
    }
    WN* get_vthr_idx(UINT idx) const
    {
        Is_True(idx < _n_vblk_dims, (""));
        return _vthr_idx[idx];
    }

    WN* get_grid_dim(UINT idx) const
    {
        Is_True(idx < 3, (""));
        return _grid_dims[idx];
    }
    WN* get_blk_dim(UINT idx) const
    {
        Is_True(idx < 3, (""));
        return _blk_dims[idx];
    }

    // The following three functions return warp info w.r.t. a thread block.
    UINT get_num_warps() { gen_warp_info(); return _n_warps; }
    WN* get_warp_id() { gen_warp_info(); return _warp_id_wn; }
    WN* get_id_within_warp() { gen_warp_info(); return _id_within_warp_wn; }

    IPA_NODE* get_kernel_node() const { return _knode; }
    void set_kernel_node(IPA_NODE *knode) { _knode = knode; }

    /**
     * Indicate that the kernel region REF/MOD the given scalar variable.
     */
    void add_scalar(ST_IDX st_idx, WN_OFFSET offset, BOOL is_ref);

    /**
     * Indicate that the kernel region REF/MOD the given array region.
     */
    void add_arr_region(ST_IDX arr_st_idx, WN *access,
            PROJECTED_REGION *ref_region, PROJECTED_REGION *mod_region);
    void add_arr_region(ST_IDX arr_st_idx, WN *call, INT actual_idx,
            PROJECTED_REGION *ref_region, PROJECTED_REGION *mod_region);

    /**
     * Find the corresponding GPU data for a particular scalar access.
     */
    HC_GPU_DATA* find_gdata_for_scalar(ST_IDX st_idx, WN_OFFSET offset);
#if 0
    /**
     * Find the corresponding GPU data for a particular access if it is an
     * array section. The access is either in a call or an OPR_ARRAY.
     */
    HC_GPU_DATA* find_gdata_for_arr_region(ST_IDX st_idx,
            WN *call_wn, INT actual_idx);
    HC_GPU_DATA* find_gdata_for_arr_region(ST_IDX st_idx, WN *access_wn);
#else
    HC_GPU_DATA* find_gdata_for_arr_region(ST_IDX st_idx);
#endif

    /**
     * When called the first time, construct a list of kernel parameters based
     * on the scalar and array accesses in the region. Otherwise, returned the
     * constructed list.
     *
     * This routine is expected to be called during kernel outlining.
     */
    HC_SYM_LIST* get_kernel_params();

    /**
     * The PROJECTED_REGION objects cached in various fields use a different
     * mempool that may be popped before this object is freed. This function
     * set these fields to NULL to prevent invalid memory accesses.
     */
    void clear_proj_regions();

    void deep_copy_proj_regions();

    /**
     * Match kernel's DAS with the visible GPU data <vgdata>, which is the
     * result under (including) annotation <annot>.
     *
     * The GPU data in <annot> that are being used are marked.
     */
    void match_gpu_data_with_das(IPA_HC_ANNOT *annot,
            HC_VISIBLE_GPU_DATA *vgdata);

    void create_callee_proc_list(UINT n_procs)
    {
        Is_True(_is_proc_called == NULL, (""));
        _is_proc_called = CXX_NEW_ARRAY(BOOL, n_procs, _pool);
        for (UINT i = 0; i < n_procs; ++i) _is_proc_called[i] = FALSE;
        _n_procs = n_procs;
    }
    BOOL is_proc_called(UINT proc_idx) const
    {
        Is_True(_is_proc_called != NULL && proc_idx < _n_procs, (""));
        return _is_proc_called[proc_idx];
    }
    void set_proc_called(UINT proc_idx)
    {
        Is_True(_is_proc_called != NULL && proc_idx < _n_procs, (""));
        _is_proc_called[proc_idx] = TRUE;
    }
    void reset_callee_proc_list()
    {
        if (_is_proc_called != NULL)
        {
            CXX_DELETE_ARRAY(_is_proc_called, _pool);
            _is_proc_called = NULL;
        }
        _n_procs = 0;
    }

    HC_GPU_DATA_LIST* create_shared_data_list()
    {
        Is_True(_sdata_list == NULL, (""));
        _sdata_list = CXX_NEW(HC_GPU_DATA_LIST(_pool), _pool);
        return _sdata_list;
    }
    HC_GPU_DATA_LIST* get_shared_data_list() const
    {
        Is_True(_sdata_list != NULL, (""));
        return _sdata_list;
    }
    // NOTE: it does not free HC_GPU_DATA's in the list.
    void reset_shared_data_list()
    {
        if (_sdata_list != NULL)
        {
            CXX_DELETE(_sdata_list, _pool);
            _sdata_list = NULL;
        }
    }

    UINT get_smem_size() const { return _smem_size; }
    void set_smem_size(UINT size_in_bytes) { _smem_size = size_in_bytes; }

    /**
     * ASSUME: the appropriate procedure context must be set up before calling
     * this function.
     */
    void print(FILE *fp);
};

typedef DYN_ARRAY<HC_KERNEL_INFO*> HC_KERNEL_INFO_LIST;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// forward declaration
class HC_LOOP_PART_INFO;

class HC_KERNEL_CONTEXT : public HC_ANNOT_DATA
{
private:

    HC_KERNEL_INFO *_kernel_info;

    // Next available dimension index in the virtual block space
    UINT _vgrid_dim_idx;
    // Next available dimension index in the virtual thread space
    UINT _vblk_dim_idx;

public:

    HC_KERNEL_CONTEXT(HC_KERNEL_INFO *kinfo,
            UINT vgrid_dim_idx, UINT vblk_dim_idx)
    {
        Is_True(kinfo != NULL, (""));
        _kernel_info = kinfo;

        Is_True(vgrid_dim_idx <= kinfo->get_vgrid_dims(), (""));
        Is_True(vblk_dim_idx <= kinfo->get_vblk_dims(), (""));
        _vgrid_dim_idx = vgrid_dim_idx;
        _vblk_dim_idx = vblk_dim_idx;
    }

    // the "virtual" mode: used to get the local offset of grid/block
    // dimension index offset.
    HC_KERNEL_CONTEXT()
    {
        _kernel_info = NULL;
        _vgrid_dim_idx = _vblk_dim_idx = 0;
    }

    // copy constructor
    HC_KERNEL_CONTEXT(const HC_KERNEL_CONTEXT &other)
    {
        _kernel_info = other._kernel_info;
        _vgrid_dim_idx = other._vgrid_dim_idx;
        _vblk_dim_idx = other._vblk_dim_idx;
    }

    virtual ~HC_KERNEL_CONTEXT() {}

    HC_KERNEL_INFO* get_kernel_info() const { return _kernel_info; }

    UINT get_vgrid_dim_idx() const { return _vgrid_dim_idx; }
    UINT get_vblk_dim_idx() const { return _vblk_dim_idx; }

    void incr_vgrid_dim_idx(UINT ofst);
    void incr_vblk_dim_idx(UINT ofst);

    void consumed_by_loop_partition(const HC_LOOP_PART_INFO *lpi);
    void unconsumed_by_loop_partition(const HC_LOOP_PART_INFO *lpi);

    virtual BOOL is_dummy() const { return _kernel_info == NULL; }

    virtual BOOL equals(const HC_ANNOT_DATA* other) const;

    virtual void print(FILE *fp) const;
};

// distribution type for the clauses in a LOOP_PARTITION directive
// SAME ORDER AS <enum kernel_part_distr_type>.
typedef enum
{
    HC_LPI_DT_BLOCK,
    HC_LPI_DT_CYCLIC,
    HC_LPI_DT_NONE
} HC_LPI_DIST_TYPE;

/*****************************************************************************
 *
 * A LOOP_PARTITION directive is identified within a procedure node, using its
 * pre-order traversal index.
 *
 ****************************************************************************/

class HC_LOOP_PART_INFO
{
private:

    HC_KERNEL_CONTEXT *_kernel_context;

    HC_LPI_DIST_TYPE _blk_clause;
    HC_LPI_DIST_TYPE _thr_clause;

    ST_IDX _idxv_st_idx;    // loop index variable
    
    // range of the loop index variable concurrently executed by threads in a
    // thread block (for SHARED directive)
    //
    // They are constructed when lowering the LOOP_PARTITION directive to CUDA
    // code (HC_lower_loop_part_region). Note that they are left NULL if the
    // directive does not have an OVER_THREAD clause, i.e., _thr_clause is
    // HC_LPI_DT_NONE.
    //
    WN *_idxv_lbnd_wn;
    WN *_idxv_ubnd_wn;

    // We must use a pointer field here as we do not want to include hc_expr.h
    // in this header file.
    HC_EXPR_PROP *_idxv_prop;

public:

    // Parse the given OPR_REGION node.
    HC_LOOP_PART_INFO(WN *loop_region);

    // Copy constructor
    HC_LOOP_PART_INFO(const HC_LOOP_PART_INFO& orig);

    ~HC_LOOP_PART_INFO() {}

    HC_LPI_DIST_TYPE get_block_clause() const { return _blk_clause; }
    HC_LPI_DIST_TYPE get_thread_clause() const { return _thr_clause; }

    void set_kernel_context(HC_KERNEL_CONTEXT *kcontext)
    {
        Is_True(kcontext != NULL, (""));
        _kernel_context = kcontext;
    }
    HC_KERNEL_CONTEXT* get_kernel_context() const { return _kernel_context; }

    void set_idxv(ST_IDX st_idx)
    {
        Is_True(st_idx != ST_IDX_ZERO, (""));
        _idxv_st_idx = st_idx;
    }
    ST_IDX get_idxv() const { return _idxv_st_idx; }

    // No copying is done.
    void set_idxv_range(WN *lbnd_wn, WN* ubnd_wn)
    {
        Is_True(lbnd_wn != NULL && ubnd_wn != NULL, (""));
        _idxv_lbnd_wn = lbnd_wn;
        _idxv_ubnd_wn = ubnd_wn;
    }
    // Return the original reference.
    WN* get_idxv_lbnd() const { return _idxv_lbnd_wn; }
    WN* get_idxv_ubnd() const { return _idxv_ubnd_wn; }

    HC_EXPR_PROP* get_idxv_prop() const { return _idxv_prop; }
    void set_idxv_prop(HC_EXPR_PROP *prop) { _idxv_prop = prop; }
};

typedef DYN_ARRAY<HC_LOOP_PART_INFO*> HC_LOOP_PART_INFO_LIST;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/**
 * Go through each kernel region in the procedure <proc_node>, and determine
 * the data it needs and produces for each annotation.
 */
extern void HC_analyze_kernel_data(IPA_NODE *proc_node);

/*****************************************************************************
 *
 * Print out the data access summary of all kernels in each K-procedure.
 *
 ****************************************************************************/

extern void HC_print_kernel_das(IPA_NODE *node, FILE *fp);
extern void IPA_print_kernel_das(FILE *fp);

/**
 * Rebuild the scalar DAS of each kernel region in the given procedure.
 */
extern void HC_rebuild_kernel_scalar_das(IPA_NODE *proc_node);

/**
 * Rename each kernel region in the given clone node.
 * It replaces the symbols in the WN tree, updates the HC_KERNEL_INFO (if
 * existed), and updates the parent kernel symbol cached in each edge.
 */
extern void HC_rename_kernels(IPA_NODE *clone);

extern ST_IDX HC_get_kernel_sym(WN *kregion_wn);

extern BOOL is_loop_part_region(WN *wn);

extern void HC_parse_kernel_directives(IPA_NODE *node);

/*****************************************************************************
 *
 * - Translate each LOOP_PARTITION directive to CUDA code, and generate index
 *   variable range for the SHARED directives.
 * - Parse the SHARED directives and determine merged sections based on info
 *   passed from LOOP_PARTITION directives.
 *
 * The given mempool must be node-independent, and is for now used to allocate
 * properties of loop index variables, which will be used to translate SHARED
 * directives.
 *
 ****************************************************************************/

extern void HC_handle_in_kernel_directives(IPA_NODE *node, MEM_POOL *pool);

/*****************************************************************************
 *
 * The given node must not be an MK-procedure.
 *
 ****************************************************************************/

extern void HC_handle_misc_kernel_directives(IPA_NODE *node);

extern void HC_outline_kernels(IPA_NODE *node);

#endif  // _IPA_HC_KERNEL_H_

/*** DAVID CODE END ***/
