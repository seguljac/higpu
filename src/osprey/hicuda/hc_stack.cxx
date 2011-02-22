/** DAVID CODE BEGIN **/

#include <assert.h>
#include <stdarg.h>

#include <map>

#include "wn.h"
#include "wn_util.h"

#include "hc_stack.h"
#include "hc_utils.h"
#include "hc_subscript.h"


char hc_stack_errmsg[256];

static void
write_errmsg(const char *format, ...) {
    va_list args;
    va_start(args, format);

    vsnprintf(hc_stack_errmsg, sizeof(hc_stack_errmsg), format, args);

    va_end(args);
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/* The actual kernel outlining is delayed until the kernel body is processed
 * for other HiCUDA directives. This data structure holds necessary info to
 * perform outlining then. */
struct kernel_context kinfo = {
    false, ST_IDX_ZERO, 0, NULL,
    /* virtual block/thread space */
    NULL, 0, NULL,
    NULL, 0, NULL,
    /* physical block/thread space */
    {NULL, NULL, NULL}, {NULL, NULL, NULL},
    /* running context */
    0, 0, NULL, NULL,
    NULL, NULL
};

void
append_svar_life(struct hc_smem_alias *sa, bool start) {
    struct hc_svar_life *hsl = (struct hc_svar_life*)
        malloc(sizeof(struct hc_svar_life));
    hsl->start = start;
    hsl->sa = sa;
    hsl->next = NULL;
    if (kinfo.hsl_tail == NULL) {
        kinfo.hsl_head = hsl;
    } else {
        kinfo.hsl_tail->next = hsl;
    }
    kinfo.hsl_tail = hsl;
}

void
reset_kinfo() {
    kinfo.valid = false;

    kinfo.kfunc_st_idx = ST_IDX_ZERO;

    kinfo.nparams = 0;
    if (kinfo.params != NULL) {
        free(kinfo.params);
        kinfo.params = NULL;
    }

    /* Running context */
    assert(kinfo.curr_vgrid_dim_idx == 0);
    assert(kinfo.curr_vblk_dim_idx == 0);
    validate_at_kernel_end();

    struct hc_svar_life *hsl = kinfo.hsl_head, *tmp = NULL;
    while (hsl != NULL) {
        tmp = hsl;
        hsl = hsl->next;

        // We only free the struct once for start of the live range.
        if (tmp->start) {
            assert(tmp->sa != NULL);
            free_svar_alias(tmp->sa);
        }
        free(tmp);
    }
    kinfo.hsl_head = kinfo.hsl_tail = NULL;

    /* Virtual block and thread space */
    if (kinfo.vgrid_dims != NULL) {
        for (UINT i = 0; i < kinfo.n_vgrid_dims; ++i) {
            WN_DELETE_Tree(kinfo.vgrid_dims[i]);
        }
        free(kinfo.vgrid_dims);
        kinfo.vgrid_dims = NULL;
    }
    if (kinfo.vblk_idx != NULL) {
        for (UINT i = 0; i < kinfo.n_vgrid_dims; ++i) {
            WN_DELETE_Tree(kinfo.vblk_idx[i]);
        }
        free(kinfo.vblk_idx);
        kinfo.vblk_idx = NULL;
    }
    kinfo.n_vgrid_dims = 0;

    if (kinfo.vblk_dims != NULL) {
        for (UINT i = 0; i < kinfo.n_vblk_dims; ++i) {
            WN_DELETE_Tree(kinfo.vblk_dims[i]);
        }
        free(kinfo.vblk_dims);
        kinfo.vblk_dims = NULL;
    }
    if (kinfo.vthr_idx != NULL) {
        for (UINT i = 0; i < kinfo.n_vblk_dims; ++i) {
            WN_DELETE_Tree(kinfo.vthr_idx[i]);
        }
        free(kinfo.vthr_idx);
        kinfo.vthr_idx = NULL;
    }
    kinfo.vblk_dims = 0;

    /* Physical block and thread space */
    for (int i = 0; i < 3; ++i) {
        WN_DELETE_Tree(kinfo.grid_dims[i]);
        kinfo.grid_dims[i] = NULL;
        WN_DELETE_Tree(kinfo.blk_dims[i]);
        kinfo.blk_dims[i] = NULL;
    }

    kinfo.kernel_body = NULL;
    kinfo.replacement = NULL;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

static struct loop_part_info *lpi_top = NULL;
static UINT loop_stack_sz = 0;

struct loop_part_info*
loop_stack_top() {
    return lpi_top;
}

void
push_loop_info(struct loop_part_info *lpi) {
    // Add it to the beginning of the list.
    lpi->parent = lpi_top;
    lpi_top = lpi;

    loop_stack_sz++;
}

struct loop_part_info*
pop_loop_info() {
    if (lpi_top == NULL) return NULL;

    struct loop_part_info *lpi = lpi_top;
    lpi_top = lpi->parent;

    loop_stack_sz--;

    return lpi;
}

void
free_loop_info(struct loop_part_info *lpi) {
    WN_DELETE_Tree(lpi->blk_range);
    WN_DELETE_Tree(lpi->full_range);

    free(lpi);
}

UINT
num_enclosing_loops() {
    return loop_stack_sz;
}

UINT
filter_doloops(struct loop_part_info **lpi_arr,
        struct loop_part_info *loop, bool inner, bool only_thr_partitioned,
        bool *all_doloops) {
    UINT nloops = 0;
    if (all_doloops != NULL) *all_doloops = true;

    /* If 'inner' is true, we go from stack top to 'loop' (stack bottom if
     * NULL, exclusive). Otherwise, we go from 'loop' (stack top if NULL,
     * inclusive) to stack bottom. Therefore, when 'loop' is NULL, 'inner'
     * does not matter.
     */
    struct loop_part_info *lpi = ((inner || loop == NULL) ? lpi_top : loop);
    struct loop_part_info *lpi_end = (inner ? loop : NULL);

    while (lpi != lpi_end) {
        if (lpi->idxv_st_idx != ST_IDX_ZERO) {
            // This is a DO_LOOP.
            if (!only_thr_partitioned || lpi->vblk_dim_idx >= 0) {
                lpi_arr[nloops++] = lpi;
            } 
        } else if (all_doloops != NULL) {
            *all_doloops = false;
        }

        lpi = lpi->parent;
    }

    return nloops;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

typedef std::map<ST_IDX, struct hc_gmem_alias*> GMEM_ALIAS_MAP;

static GMEM_ALIAS_MAP gvar_aliases;
static GMEM_ALIAS_MAP cvar_aliases;

static struct hc_gmem_alias*
visible_gmem_alias(GMEM_ALIAS_MAP& map, ST_IDX ori_st_idx) {
    GMEM_ALIAS_MAP::iterator it = map.find(ori_st_idx);
    if (it == map.end()) return NULL;

    struct hc_gmem_alias *ga = it->second;
    assert(ga != NULL);

    return ga;
}

struct hc_gmem_alias*
visible_global_var(ST_IDX ori_st_idx) {
    return visible_gmem_alias(gvar_aliases, ori_st_idx);
}

struct hc_gmem_alias*
visible_const_var(ST_IDX ori_st_idx) {
    return visible_gmem_alias(cvar_aliases, ori_st_idx);
}

static struct hc_gmem_alias*
add_gmem_alias(GMEM_ALIAS_MAP& map,
        ST_IDX gvar_st_idx, ST_IDX ori_st_idx, WN *scope) {
    // Create a new node first.
    struct hc_gmem_alias *ga = (struct hc_gmem_alias*)
        malloc(sizeof(struct hc_gmem_alias));
    ga->ori_st_idx = ori_st_idx;
    ga->gvar_st_idx = gvar_st_idx;
    ga->scope = scope;
    ga->var_dim_sz = NULL;
    ga->gvar_info = NULL;
    ga->is_full_idx_range = NULL;
    ga->partial_dim_idx = -1;
    ga->gvar_sz = NULL;
    ga->alloc_offset = 0;
    ga->init_point = NULL;
    ga->cvar_refs = NULL;

    GMEM_ALIAS_MAP::iterator it = map.find(ori_st_idx);
    if (it == map.end()) {
        ga->next = NULL;
        map[ori_st_idx] = ga;
    } else {
        // Add the node to the front of the list.
        assert(it->second != NULL);
        ga->next = it->second;
        it->second = ga;
    }

    return ga;
}

struct hc_gmem_alias*
add_gvar_alias(ST_IDX gvar_st_idx, ST_IDX ori_st_idx, WN *scope) {
    return add_gmem_alias(gvar_aliases, gvar_st_idx, ori_st_idx, scope);
}

struct hc_gmem_alias*
add_cvar_alias(ST_IDX ori_st_idx, WN *scope) {
    /* The const variable is shared for all CONST COPYIN's in a PU.
     * Find out if it has been declared, otherwise declare it as a byte
     * array of unit length. Its actual length will be determined after the
     * whole PU is processed.
     */
    ST_IDX cvar_st_idx = ST_IDX_ZERO;

    GMEM_ALIAS_MAP::iterator it = cvar_aliases.begin();
    if (it != cvar_aliases.end()) {
        struct hc_gmem_alias *ga = it->second;
        assert(ga != NULL);
        cvar_st_idx = ga->gvar_st_idx;
    } else {
        UINT dims[1];
        dims[0] = 1;
        TY_IDX cvar_ty_idx = make_arr_type(Save_Str("cmem.ty"),
            1, dims, MTYPE_To_TY(MTYPE_I1));
        cvar_st_idx = new_global_var(
            gen_var_str(ST_IDX_ZERO, "cmem"), cvar_ty_idx, false);
        set_st_attr_is_const_var(cvar_st_idx);
    }
    assert(cvar_st_idx != ST_IDX_ZERO);

    struct hc_gmem_alias *ga = add_gmem_alias(
            cvar_aliases, cvar_st_idx, ori_st_idx, scope);

    ga->cvar_refs = new WN_SET();

    return ga;
}

static struct hc_gmem_alias*
remove_gmem_alias(GMEM_ALIAS_MAP& map, ST_IDX ori_st_idx, WN *scope) {
    GMEM_ALIAS_MAP::iterator it = map.find(ori_st_idx);
    if (it == map.end()) return NULL;

    struct hc_gmem_alias *ga = it->second;
    assert(ga != NULL);

    // Check if the scope matches.
    if (ga->scope != scope) return NULL;

    // Remove the node from the list.
    it->second = ga->next;
    ga->next = NULL;

    // If the list is empty, remove the map entry.
    if (it->second == NULL) map.erase(ori_st_idx);

    return ga;
}

struct hc_gmem_alias*
remove_gvar_alias(ST_IDX ori_st_idx, WN *scope) {
    return remove_gmem_alias(gvar_aliases, ori_st_idx, scope);
}

struct hc_gmem_alias*
remove_cvar_alias(ST_IDX ori_st_idx, WN *scope) {
    return remove_gmem_alias(cvar_aliases, ori_st_idx, scope);
}

void
free_gmem_alias(struct hc_gmem_alias *ga) {
    // Deallocate the node.
    if (ga->gvar_info != NULL) {
        assert(ga->var_dim_sz != NULL);
        assert(ga->is_full_idx_range != NULL);
        assert(ga->gvar_sz != NULL);

        // Get the array dimensionality.
        int ndims = WN_kid_count(ga->gvar_info) >> 1;

        for (int i = 0; i < ndims; ++i) {
            WN_DELETE_Tree(ga->var_dim_sz[i]);
        }
        free(ga->var_dim_sz);

        free(ga->is_full_idx_range);

        WN_DELETE_Tree(ga->gvar_info);
    } else {
        assert(ga->var_dim_sz == NULL);
        assert(ga->is_full_idx_range == NULL);
    }

    WN_DELETE_Tree(ga->gvar_sz);

    if (ga->cvar_refs != NULL) free(ga->cvar_refs);

    free(ga);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Given an ARRAY node, construct a new ARRAY that represents the same
 * access to the corresponding global variable.
 */
static WN*
global_access_for_array(const WN *access, const struct hc_gmem_alias *ga) {
    // Get array dimensionality.
    UINT16 ndims = WN_num_dim(access);

    // The new access treats the global variable as a 1-D array.
    WN *new_access = HCWN_CreateArray(WN_LdidScalar(ga->gvar_st_idx), 1);
    WN_element_size(new_access) = WN_element_size(access);

    // Compute the offset for the global array access.
    WN *offset = WN_Sub(Integer_type,
        WN_COPY_Tree(WN_kid(access,ndims+1)),
        WN_COPY_Tree(WN_kid0(WN_kid(ga->gvar_info,ndims+1)))
    );
    for (UINT16 i = 1; i < ndims; ++i) {
        // offset = offset * curr_dim_size + (curr_dim_idx - start_idx)
        offset = WN_Add(Integer_type,
            WN_Mpy(Integer_type,
                offset,
                WN_COPY_Tree(WN_kid(ga->gvar_info,i+1))
            ),
            WN_Sub(Integer_type,
                WN_COPY_Tree(WN_kid(access,ndims+i+1)),
                WN_COPY_Tree(WN_kid0(WN_kid(ga->gvar_info,ndims+i+1)))
            )
        );
    }

    // We provide a fake dimension size because it is not necessary in 1-D.
    WN_kid1(new_access) = WN_Intconst(Integer_type, 0);

    /* We do not need ILOAD because an ARRAY node only returns the access
     * address, so there should always be a parent ILOAD.
     */
    WN_kid2(new_access) = offset;

    return new_access;
}

/**
 * Given a LDID/STID node,
 * - for a scalar/struct variable, construct a ILOAD/ISTORE node that
 *   represents the same access to the corresponding global variable.
 * - for a pointer variable, just replace the symbol with the global variable
 */
static WN*
global_access_for_scalar(WN *access, const struct hc_gmem_alias *ga) {
    OPERATOR opr = WN_operator(access);

    TY_KIND var_ty_kind = TY_kind(ST_type(ga->ori_st_idx));

    if (var_ty_kind == KIND_POINTER) {
        assert(ST_type(ga->ori_st_idx) == ST_type(ga->gvar_st_idx));

        WN *result = WN_COPY_Tree(access);
        WN_st_idx(result) = ga->gvar_st_idx;
        return result;
    }

    assert(var_ty_kind == KIND_SCALAR || var_ty_kind == KIND_STRUCT);

    if (opr == OPR_LDID) {
        // Convert it to ILOAD.
        return WN_CreateIload(OPR_ILOAD,
            WN_rtype(access), WN_desc(access),
            WN_load_offset(access),
            WN_ty(access),
            ST_type(ga->gvar_st_idx),
            WN_LdidScalar(ga->gvar_st_idx),
            WN_field_id(access)
        );
    }
    
    if (opr == OPR_STID) {
        // Convert it to ISTORE.
        return WN_CreateIstore(OPR_ISTORE,
            WN_rtype(access), WN_desc(access),
            WN_load_offset(access),
            ST_type(ga->gvar_st_idx),
            WN_COPY_Tree(WN_kid0(access)),
            WN_LdidScalar(ga->gvar_st_idx),
            WN_field_id(access)
        );
    }

    return NULL;
}

/**
 * Given an ARRAY node, construct a ADD node that represents the same
 * access to the corresponding constant variable.
 *
 * Since the allocation offset of the constant variable has not been
 * determined when this routine is called, we assume it is zero. This
 * will be fixed in analyze_cvar_live_ranges.
 *
 * *((<cast_to>)(cmem + alloc_ofst + access_ofst))
 */
static WN* const_access_for_array(const WN *access,
        const struct hc_gmem_alias *ga)
{
    // Get array dimensionality.
    UINT16 ndims = WN_num_dim(access);

    // Compute the offset, in bytes, for the constant array access.
    WN *offset = WN_Sub(Integer_type,
        WN_COPY_Tree(WN_kid(access,ndims+1)),
        WN_COPY_Tree(WN_kid0(WN_kid(ga->gvar_info,ndims+1)))
    );
    for (UINT16 i = 1; i < ndims; ++i) {
        // offset = offset * curr_dim_size + (curr_dim_idx - start_idx)
        offset = WN_Add(Integer_type,
            WN_Mpy(Integer_type,
                offset,
                WN_COPY_Tree(WN_kid(ga->gvar_info,i+1))
            ),
            WN_Sub(Integer_type,
                WN_COPY_Tree(WN_kid(access,ndims+i+1)),
                WN_COPY_Tree(WN_kid0(WN_kid(ga->gvar_info,ndims+i+1)))
            )
        );
    }
    offset = WN_Mpy(Integer_type,
        offset, WN_Intconst(Integer_type, WN_element_size(access)));

    WN *base_addr_wn = WN_LdaZeroOffset(ga->gvar_st_idx,
            Make_Pointer_Type(MTYPE_To_TY(MTYPE_I1)));

    WN *new_access_wn = WN_Add(Pointer_type, base_addr_wn, offset);

    /* Keep track of this access as we need to update its offset later.
     * We cannot just store the base address because it may have been
     * absorbed in `new_access_wn'.
     */
    ga->cvar_refs->insert(new_access_wn);

    return new_access_wn;
}

/**
 * Given a LDID node,
 * - for a scalar/struct variable, construct a ILOAD node that represents
 *   the same access to the corresponding const variable.
 * - for a pointer variable, just replace the symbol with the const variable
 *   and update the offset
 *
 * Since the allocation offset of the constant variable has not been
 * determined when this routine is called, we assume it is zero. This
 * will be fixed in analyze_cvar_live_ranges.
 */
static WN* const_access_for_scalar(WN *access, const struct hc_gmem_alias *ga)
{
    WN *new_access_wn = NULL;

    TY_KIND var_ty_kind = TY_kind(ST_type(ga->ori_st_idx));

    if (var_ty_kind == KIND_POINTER)
    {
        hc_dev_warn("const_access_for_scalar: see pointer `%s'\n",
                ST_name(ga->ori_st_idx));

        new_access_wn = WN_CreateLda(OPR_LDA, Pointer_type, MTYPE_V,
                WN_load_offset(access), WN_ty(access), ga->gvar_st_idx);
    }
    else
    {
        // Convert it to ILOAD.
        TY_IDX char_ptr_ty_idx = Make_Pointer_Type(MTYPE_To_TY(MTYPE_I1));

        WN *base_addr_wn = WN_LdaZeroOffset(ga->gvar_st_idx, char_ptr_ty_idx);

        new_access_wn = WN_CreateIload(OPR_ILOAD,
                WN_rtype(access), WN_desc(access),
                WN_load_offset(access),
                WN_ty(access),
                Make_Pointer_Type(WN_ty(access)),
                base_addr_wn,
                WN_field_id(access));
    }

    // Keep track of the base access as we need to update its offset later.
    ga->cvar_refs->insert(new_access_wn);

    return new_access_wn;
}

/**
 * Given an ARRAY node, modify it so that it represents the same access to
 * the corresponding shared variable.
 *
 * We do not need to create a new node because the dimensionality does not
 * change.
 */
static void
shared_access_for_array(WN *access, const struct hc_smem_alias *sa) {
    /* Replace the ARRAY node with an access to the shared variable.
     * The access is shared_var[i - shared_start_i][j - shared_start_j] */

    // dimensionality
    UINT16 ndims = WN_num_dim(access);

    // base address
    WN_DELETE_Tree(WN_kid0(access));
    WN_kid0(access) = WN_LdidScalar(sa->svar_st_idx);

    // dimension size and subscript/index
    for (UINT16 i = 0; i < ndims; ++i) {
        // dimension size
        WN_DELETE_Tree(WN_kid(access,i+1));
        WN_kid(access,i+1) = WN_COPY_Tree(WN_kid(sa->svar_info,i+1));
        // subscript in this dimension
        // We incorporate the original subscript in the new one.
        WN_kid(access,ndims+i+1) = WN_Sub(Integer_type,
            WN_kid(access,ndims+i+1),
            WN_COPY_Tree(WN_kid0(WN_kid(sa->svar_info,ndims+i+1)))
        );
    }

    // The element size stays the same.
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * 'access_region' is an ARRSECTION node.
 *
 * If TB_NOTSURE is returned, the reason is stored in 'hc_stack_errmsg'.
 */
static TBOOL
is_covered_by_svar(WN *access_region, struct hc_smem_alias *sa) {
    TBOOL result = TB_NOTSURE;

    /* Project the access w.r.t. all loops between this access and the
     * declaration point of the shared variable. Among these loops, some
     * are block/thread-partitioned while others are not. For the former
     * group, we want to entire index range; for the latter group, we want
     * "block range". Luckily, the 'blk_range' field stores the correct
     * range for both types.
     */
    
    UINT nloops = num_enclosing_loops();
    struct loop_part_info* lpi_arr[nloops];
    ST_IDX loop_idxvs[nloops];
    WN* loop_ranges[nloops];

    /* Project the access region onto all loops between it and the declaration
     * of the corresponding shared variable.
     */
    bool all_doloops;
    UINT nloops_in_between = filter_doloops(lpi_arr,
        sa->loop_scope, true, false, &all_doloops);
    if (!all_doloops) {
        write_errmsg("Not all loops are DO_LOOPs");
        return result;
    }
    for (UINT i = 0; i < nloops_in_between; ++i) {
        loop_idxvs[i] = lpi_arr[i]->idxv_st_idx;
        loop_ranges[i] = lpi_arr[i]->blk_range;
    }

    WN *proj_region = project_region(access_region,
            loop_idxvs, loop_ranges, nloops_in_between);
    if (proj_region == NULL) {
        write_errmsg("Failed at stage 1 projection: %s", hc_subscript_errmsg);
        return result;
    }

    /* Do the same region projection as done on the user-specified shared
     * variable region. We care about all thread-partitioned loops enclosing
     * the shared variable declaration.
     */
    UINT nploops_outside_svar = filter_doloops(lpi_arr,
        sa->loop_scope, false, true, NULL);
    for (UINT i = 0; i < nploops_outside_svar; ++i) {
        loop_idxvs[i] = lpi_arr[i]->idxv_st_idx;
        loop_ranges[i] = lpi_arr[i]->blk_range;
    }

    WN *merged_region = project_region(proj_region,
        loop_idxvs, loop_ranges, nploops_outside_svar);
    WN_DELETE_Tree(proj_region);
    if (merged_region == NULL) {
        write_errmsg("Failed to do stage 2 projection: %s",
            hc_subscript_errmsg);
        return result;
    }

    /* Is this merged region covered by the shared variable? */

    result = is_region_covered(sa->svar_info, merged_region);

    // Clean up.
    WN_DELETE_Tree(merged_region);

    return result;
}

/**
 * 'access_region' is an ARRSECTION node.
 *
 * If TB_NOTSURE is returned, the reason is stored in 'hc_stack_errmsg'.
 */
static TBOOL
is_covered_by_gvar(WN *access_region, struct hc_gmem_alias *ga) {
    TBOOL result = TB_NOTSURE;

    /* Project the access region w.r.t. all loops in the stack (using
     * their full index range).
     */

    UINT nloops = num_enclosing_loops();
    struct loop_part_info* lpi_arr[nloops];
    ST_IDX loop_idxvs[nloops];
    WN* loop_ranges[nloops];

    bool all_doloops;
    UINT n_doloops = filter_doloops(lpi_arr, NULL, false, false, &all_doloops);
    if (!all_doloops) {
        write_errmsg("Not all loops are DO_LOOPs");
        return result;
    }

    for (UINT i = 0; i < n_doloops; ++i) {
        loop_idxvs[i] = lpi_arr[i]->idxv_st_idx;
        loop_ranges[i] = lpi_arr[i]->full_range;
    }

    // Do region projection on these loops.
    WN *merged_region = project_region(access_region,
        loop_idxvs, loop_ranges, n_doloops);
    if (merged_region == NULL) {
        write_errmsg("Failed to do projection: %s", hc_subscript_errmsg);
        return result;
    }

    /* Is this merged region covered by the global variable? */

    result = is_region_covered(ga->gvar_info, merged_region);

    // Clean up.
    WN_DELETE_Tree(merged_region);

    return result;
}

bool
replace_var_access(WN *wn, WN **new_wn) {
    assert(wn != NULL && new_wn != NULL);
    *new_wn = NULL;

    OPERATOR opr = WN_operator(wn);
    assert(opr == OPR_LDID || opr == OPR_STID || opr == OPR_ARRAY);

    if (opr == OPR_ARRAY) {
        // Get the original variable symbol.
        opr = WN_operator(WN_kid0(wn));
        if (opr != OPR_LDA && opr != OPR_LDID) return false;
        ST_IDX var_st_idx = WN_st_idx(WN_kid0(wn));

        // Check if it has a corresponding global or constant variable.
        struct hc_gmem_alias *ga = visible_global_var(var_st_idx);
        struct hc_gmem_alias *ca = visible_const_var(var_st_idx);
        if (ga == NULL && ca == NULL) return false;

        // Convert the access into an ARRSECTION.
        WN *access_region = WN_COPY_Tree(wn);
        UINT ndims = WN_num_dim(access_region);
        for (UINT i = 0; i < ndims; ++i) {
            WN *idx = WN_kid(access_region,ndims+i+1);
            WN_kid(access_region,ndims+i+1) = WN_CreateTriplet(
                idx, WN_Intconst(Integer_type, 1), WN_COPY_Tree(idx));
        }

        // Check if it is covered by a global variable.
        TBOOL covered_by_gvar = (ga == NULL) ? TB_FALSE :
            is_covered_by_gvar(access_region, ga);
        // Check if it is covered by a constant variable.
        TBOOL covered_by_cvar = (ca == NULL) ? TB_FALSE :
            is_covered_by_gvar(access_region, ca);

        // Check if it is covered by a shared variable.
        struct hc_smem_alias *sa = visible_shared_var(var_st_idx);
        TBOOL covered_by_svar = (sa == NULL) ? TB_FALSE :
            is_covered_by_svar(access_region, sa);

        WN_DELETE_Tree(access_region);

        // Consider the shared variable first.
        if (covered_by_svar != TB_FALSE) {
            if (covered_by_cvar != TB_FALSE) {
                fprintf(stderr, "Variable %s exist in both shared and "
                    "constant memory. Its access will be redirected to "
                    "the shared memory.\n", ST_name(var_st_idx));
            }

            if (covered_by_svar == TB_NOTSURE) {
                // TODO: display cause
                fprintf(stderr, "The access for variable %s may not entirely "
                    "exist in the shared memory. Please check yourself.\n",
                    ST_name(var_st_idx));
            }

            // Replace it with the shared variable access.
            shared_access_for_array(wn, sa);
            return true;
        }

        // Consider the constant memory.
        if (covered_by_cvar != TB_FALSE) {
            if (covered_by_gvar != TB_FALSE) {
                fprintf(stderr, "Variable %s exist in both constant and "
                    "global memory. Its access will be redirected to "
                    "the constant memory.\n", ST_name(var_st_idx));
            }

            if (covered_by_cvar == TB_NOTSURE) {
                // TODO: display cause
                fprintf(stderr, "The access for variable %s may not entirely "
                    "exist in the constant memory. Please check yourself.\n",
                    ST_name(var_st_idx));
            }

            *new_wn = const_access_for_array(wn, ca);
            return true;
        }

        // Now, the global variable must cover the access.
        Is_True(covered_by_gvar != TB_FALSE,
                ("The access to variable %s cannot be redirected to "
                 "any GPU memory (kernel %s)!\n",
                 ST_name(var_st_idx), ST_name(kinfo.kfunc_st_idx))
        );

        if (covered_by_gvar == TB_NOTSURE) {
            // TODO: display cause
            fprintf(stderr, "The access for variable %s may not entirely "
                "exist in the global memory. Please check yourself.\n",
                ST_name(var_st_idx));
        }

        *new_wn = global_access_for_array(wn, ga);
        return true;
    }

    if (opr == OPR_LDID) {
        // Get the original variable symbol.
        ST_IDX var_st_idx = WN_st_idx(wn);

        // Check if it has a corresponding global or constant variable.
        struct hc_gmem_alias *ga = visible_global_var(var_st_idx);
        struct hc_gmem_alias *ca = visible_const_var(var_st_idx);
        if (ga == NULL && ca == NULL) return false;

        if (ca != NULL) {
            if (ga != NULL) {
                fprintf(stderr, "Variable %s exist in both constant and "
                        "global memory. Its access will be redirected to "
                        "the constant memory.\n", ST_name(var_st_idx));
            }

            *new_wn = const_access_for_scalar(wn, ca);
            return true;
        }

        // It must be covered by the global variable.
        *new_wn = global_access_for_scalar(wn, ga);
        return true;
    }

    if (opr == OPR_STID) {
        // Get the original variable symbol.
        ST_IDX var_st_idx = WN_st_idx(wn);

        // We only consider the global variable because the constant variable
        // is read-only within a kernel.
        struct hc_gmem_alias *ga = visible_global_var(var_st_idx);
        if (ga == NULL) return false;

        *new_wn = global_access_for_scalar(wn, ga);
        return true;
    }

    return false;
}   /* replace_var_access */

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

typedef std::map<ST_IDX, struct hc_smem_alias*> SVAR_ALIAS_MAP;

static SVAR_ALIAS_MAP svar_aliases;

struct hc_smem_alias*
visible_shared_var(ST_IDX ori_st_idx) {
    SVAR_ALIAS_MAP::iterator it = svar_aliases.find(ori_st_idx);
    if (it == svar_aliases.end()) return NULL;

    struct hc_smem_alias *sa = it->second;
    assert(sa != NULL);

    return sa;
}

struct hc_smem_alias*
add_svar_alias(ST_IDX svar_st_idx, ST_IDX ori_st_idx, WN *scope) {
    SVAR_ALIAS_MAP::iterator it = svar_aliases.find(ori_st_idx);

    // Create a new node first.
    struct hc_smem_alias *sa = (struct hc_smem_alias*)
        malloc(sizeof(struct hc_smem_alias));
    sa->ori_st_idx = ori_st_idx;
    sa->svar_st_idx = svar_st_idx;
    sa->scope = scope;
    sa->loop_scope = loop_stack_top();
    sa->ga = NULL;
    sa->svar_info = NULL;
    sa->svar_size = 0;
    sa->alloc_offset = 0;
    sa->init_point = NULL;
    sa->copyin_wn = NULL;

    if (it == svar_aliases.end()) {
        sa->next = NULL;
        svar_aliases[ori_st_idx] = sa;
    } else {
        // Add the node to the front of the list.
        assert(it->second != NULL);
        sa->next = it->second;
        it->second = sa;
    }

    // The original variable must have a corresponding global variable already.
    sa->ga = visible_global_var(ori_st_idx);
    Is_True(sa->ga != NULL,
        ("Variable %s does not exist in the global memory while it is copied "
        "to the shared memory!\n", ST_name(ori_st_idx)));

    return sa;
}

struct hc_smem_alias*
remove_svar_alias(ST_IDX ori_st_idx, WN *scope) {
    SVAR_ALIAS_MAP::iterator it = svar_aliases.find(ori_st_idx);
    if (it == svar_aliases.end()) return NULL;

    struct hc_smem_alias *sa = it->second;
    assert(sa != NULL);

    // Check if the scope matches.
    if (sa->scope != scope) return NULL;

    // Remove the node from the list.
    it->second = sa->next;
    sa->next = NULL;

    // If the list is empty, remove the map entry.
    if (it->second == NULL) svar_aliases.erase(ori_st_idx);

    return sa;
}

void
free_svar_alias(struct hc_smem_alias *sa) {
    // Deallocate the node.
    if (sa->svar_info != NULL) WN_DELETE_Tree(sa->svar_info);
    if (sa->copyin_wn != NULL) WN_DELETE_Tree(sa->copyin_wn);

    free(sa);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/* The problem we are solving is as follows:
 *
 * Given a sequence of shared memory requests (of known sizes), find the
 * minimum amount of shared memory needed so that every request can be
 * satisfied. The byproduct of the solution is the offset of the allocated
 * memory for each request.
 */

/* Here we provide a fast algorithm that determines a sub-optimal solution:
 *
 * We simulate a memory of growing size, starting from 0. When each request
 * comes in, we search for fragments in the existing memory. If there is no
 * fragment (i.e. the entire memory is in use), we grow the memory by the
 * request size and let the request take this new chunk. If there is at least
 * one fragment, we let the request take the biggest fragment. We increase
 * the fragment size if necessary.
 */

/* A list of shared variables occupying memory, in increasing order of offset
 * in the memory */
static struct hc_smem_alias *svar_alive_list = NULL;

/* A list of shared variables whose live range is over, in undefined order */
static struct hc_smem_alias *svar_done_list = NULL;

static UINT smem_sz = 0;

ST_IDX
analyze_svar_live_ranges(struct hc_svar_life *hsls) {
    svar_alive_list = svar_done_list = NULL;
    smem_sz = 0;

    while (hsls != NULL) {
        if (hsls->start) {
            /* Search for fragments in the existing memory. */

            // the fragment size, if +ive; or no fragment, if 0
            UINT largest_fragment_sz = 0;
            // the shared variable before and after the largest fragment
            struct hc_smem_alias *svar_before = NULL, *svar_after = NULL;

            struct hc_smem_alias *sa = svar_alive_list;
            if (sa != NULL) {
                // There may be a fragment at the beginning.
                UINT fragment_sz = sa->alloc_offset;
                if (fragment_sz > 0) {
                    largest_fragment_sz = fragment_sz;
                    svar_before = NULL; svar_after = sa;
                }

                while (sa->next != NULL) {
                    // Check the fragment between this node and the next node.
                    fragment_sz = sa->next->alloc_offset
                        - sa->alloc_offset - sa->svar_size;
                    if (largest_fragment_sz < fragment_sz) {
                        largest_fragment_sz = fragment_sz;
                        svar_before = sa; svar_after = sa->next;
                    }

                    sa = sa->next;
                }

                // There may be a fragment at the end.
                fragment_sz = smem_sz - sa->alloc_offset - sa->svar_size;
                if (largest_fragment_sz < fragment_sz) {
                    largest_fragment_sz = fragment_sz;
                    svar_before = sa; svar_after = NULL;
                }
            }

            if (largest_fragment_sz > 0) {
                /* We found the largest fragment, is it big enough? */
                INT size_incr = hsls->sa->svar_size - largest_fragment_sz;

                if (size_incr > 0) {
                    /* Increase the fragment size to fit the shared variable.
                     *
                     * For each live shared variable allocated after
                     * 'svar_after', we need to increase its offset. For each
                     * done shared variable, we must also increase its offset
                     * if it is allocated after 'svar_after'.
                     */
                    smem_sz += size_incr;

                    if (svar_after != NULL) {
                        // NOTE: 'sa' is not useful anymore.
                        sa = svar_after;
                        do {
                            sa->alloc_offset += size_incr;
                            sa = sa->next;
                        } while (sa != NULL);

                        UINT svar_after_past_end = svar_after->alloc_offset +
                            svar_after->svar_size;

                        sa = svar_done_list;
                        while (sa != NULL) {
                            if (sa->alloc_offset >= svar_after_past_end) {
                                sa->alloc_offset += size_incr;
                            }
                            sa = sa->next;
                        }
                    }
                }
            } else {
                /* Grow the memory at the end, by the size the shared variable
                 * needs. Note that 'sa' holds the last shared variable.
                 */
                smem_sz += hsls->sa->svar_size;

                svar_before = sa; svar_after = NULL;
            }

            /* Allocate the shared variable. If the fragment is bigger
             * than what the variable needs, the allocation can take the
             * beginning or the end of the fragment, or less likely
             * anywhere in between. For now, we always allocate at the
             * beginning.
             */
            if (svar_before == NULL) {
                svar_alive_list = hsls->sa;
                hsls->sa->alloc_offset = 0;
            } else {
                svar_before->next = hsls->sa;
                hsls->sa->alloc_offset = svar_before->alloc_offset +
                    svar_before->svar_size;
            }
            hsls->sa->next = svar_after;

        } else {
            /* Move the shared variable from alive list to done list. */

            struct hc_smem_alias *curr_sa = svar_alive_list, *prev_sa = NULL;
            while (curr_sa != NULL) {
                if (curr_sa == hsls->sa) break;
                prev_sa = curr_sa;
                curr_sa = curr_sa->next;
            }
            assert(curr_sa != NULL);

            if (prev_sa != NULL) {
                prev_sa->next = curr_sa->next;
            } else {
                svar_alive_list = curr_sa->next;
            }

            curr_sa->next = svar_done_list;
            svar_done_list = curr_sa;
        }

        hsls = hsls->next;
    }

    assert(svar_alive_list == NULL);

    /* Declare a real shared variable: a 1-D array of size 'smem_sz'.
     *
     * The array's element type is arbitrary, as long as it's 4-byte, because
     * otherwise it may cause bank conflicts. For now, we use MTYPE_I4.
     */
    TY_IDX elem_ty_idx = MTYPE_To_TY(MTYPE_I4);
    UINT64 elem_sz = TY_size(elem_ty_idx);
    assert(smem_sz % elem_sz == 0);

    UINT32 dim_sz[1];
    dim_sz[0] = smem_sz / elem_sz;

    TY_IDX ty_idx = make_arr_type(Save_Str("svar.type"),
        1, dim_sz, elem_ty_idx);
    ST_IDX st_idx = new_local_var(Save_Str("smem"), ty_idx);

    // Set the SHARED attribute of this symbol.
    set_st_attr_is_shared_var(st_idx);

    /* For each shared variable, construct its initialization expression.
     */
    struct hc_smem_alias *sa = svar_done_list;
    while (sa != NULL) {
        assert(sa->init_point != NULL
            && WN_operator(sa->init_point) == OPR_STID);

        assert(sa->svar_size % elem_sz == 0);

        WN *init_val = WN_kid0(sa->init_point);
        if (init_val != NULL) WN_DELETE_Tree(init_val);
        WN_kid0(sa->init_point) = WN_Add(Pointer_type,
            WN_CreateLda(OPR_LDA, Pointer_type, MTYPE_V, 0,
                Make_Pointer_Type(elem_ty_idx), st_idx, 0),
            WN_Intconst(Integer_type, sa->alloc_offset)
        );

        sa = sa->next;
    }

    // Clean up.
    svar_done_list = NULL;  // no need to free them
    smem_sz = 0;

    return st_idx;
}

static struct hc_cvar_life *cvar_live_range_head = NULL;
static struct hc_cvar_life *cvar_live_range_tail = NULL;

void
append_cvar_life(struct hc_gmem_alias *ga, bool start) {
    struct hc_cvar_life *hcl = (struct hc_cvar_life*)
        malloc(sizeof(struct hc_cvar_life));
    hcl->start = start;
    hcl->ga = ga;
    hcl->next = NULL;
    if (cvar_live_range_tail == NULL) {
        cvar_live_range_head = hcl;
    } else {
        cvar_live_range_tail->next = hcl;
    }
    cvar_live_range_tail = hcl;
}

struct hc_cvar_life*
get_cvar_live_ranges() {
    return cvar_live_range_head;
}

/**
 * Search for an LDA/LDID node of `cmem_st_idx' in `access_wn', and
 * increment its offset by `offset'.
 *
 * Return true if the replacement occurs.
 */
static bool update_cvar_alloc_offset(WN *access_wn,
        ST_IDX cmem_st_idx, UINT offset)
{
    OPERATOR opr = WN_operator(access_wn);

    assert(OPERATOR_is_expression(opr));

    if (opr == OPR_LDID || opr == OPR_LDA) {
        if (WN_st_idx(access_wn) == cmem_st_idx) {
            WN_load_offset(access_wn) += offset;
            return true;
        }
        return false;
    }

    INT nkids = WN_kid_count(access_wn);
    for (INT i = 0; i < nkids; ++i) {
        WN *kid_wn = WN_kid(access_wn,i);
        assert(kid_wn != NULL);
        if (update_cvar_alloc_offset(kid_wn, cmem_st_idx, offset)) return true;
    }
#if 0
    if (opr == OPR_ILOAD) {
        return update_cvar_alloc_offset(
                WN_kid0(access_wn), cmem_st_idx, offset);
    }

    if (opr == OPR_ADD) {
        if (update_cvar_alloc_offset(
                    WN_kid0(access_wn), cmem_st_idx, offset)) return true;
        return update_cvar_alloc_offset(
                WN_kid1(access_wn), cmem_st_idx, offset);
    }

    hc_dev_warn("update_cvar_alloc_offset: meet a %s node.\n",
            OPERATOR_name(opr));
#endif
    return false;
}

/*
 * Why do we have these variables global??
 */

/* a list of const variables occupying memory, in increasing order of offset
 * in the memory */
static struct hc_gmem_alias *cvar_alive_list = NULL;

/* a list of const variables whose live range is over, in undefined order */
static struct hc_gmem_alias *cvar_done_list = NULL;

static UINT cmem_sz = 0;

/**
 * TODO: reuse of the algorithm code with analyze_svar_live_ranges
 */
void
analyze_cvar_live_ranges(struct hc_cvar_life *hcls) {
    assert(hcls != NULL);

    cvar_alive_list = cvar_done_list = NULL;
    cmem_sz = 0;

    while (hcls != NULL) {
        if (hcls->start) {
            /* Search for fragments in the existing memory. */

            // the fragment size, if +ive; or no fragment, if 0
            UINT largest_fragment_sz = 0;
            // the const variable before and after the largest fragment
            struct hc_gmem_alias *cvar_before = NULL, *cvar_after = NULL;

            struct hc_gmem_alias *ga = cvar_alive_list;
            if (ga != NULL) {
                // There may be a fragment at the beginning.
                UINT fragment_sz = ga->alloc_offset;
                if (fragment_sz > 0) {
                    largest_fragment_sz = fragment_sz;
                    cvar_before = NULL; cvar_after = ga;
                }

                while (ga->next != NULL) {
                    // Check the fragment between this node and the next node.
                    assert(WN_operator(ga->gvar_sz) == OPR_INTCONST);
                    fragment_sz = ga->next->alloc_offset
                        - ga->alloc_offset - WN_const_val(ga->gvar_sz);
                    if (largest_fragment_sz < fragment_sz) {
                        largest_fragment_sz = fragment_sz;
                        cvar_before = ga; cvar_after = ga->next;
                    }

                    ga = ga->next;
                }

                // There may be a fragment at the end.
                assert(WN_operator(ga->gvar_sz) == OPR_INTCONST);
                fragment_sz = cmem_sz
                    - ga->alloc_offset - WN_const_val(ga->gvar_sz);
                if (largest_fragment_sz < fragment_sz) {
                    largest_fragment_sz = fragment_sz;
                    cvar_before = ga; cvar_after = NULL;
                }
            }

            if (largest_fragment_sz > 0) {
                /* We found the largest fragment, is it big enough? */
                assert(WN_operator(hcls->ga->gvar_sz) == OPR_INTCONST);
                INT size_incr = WN_const_val(hcls->ga->gvar_sz)
                    - largest_fragment_sz;

                if (size_incr > 0) {
                    /* Increase the fragment size to fit the const variable.
                     *
                     * For each live const variable allocated after
                     * 'cvar_after', we need to increase its offset. For each
                     * done const variable, we must also increase its offset
                     * if it is allocated after 'cvar_after'.
                     */
                    cmem_sz += size_incr;

                    if (cvar_after != NULL) {
                        // NOTE: 'ga' is not useful anymore.
                        ga = cvar_after;
                        do {
                            ga->alloc_offset += size_incr;
                            ga = ga->next;
                        } while (ga != NULL);

                        UINT cvar_after_past_end = cvar_after->alloc_offset
                            + WN_const_val(cvar_after->gvar_sz);

                        ga = cvar_done_list;
                        while (ga != NULL) {
                            if (ga->alloc_offset >= cvar_after_past_end) {
                                ga->alloc_offset += size_incr;
                            }
                            ga = ga->next;
                        }
                    }
                }
            } else {
                /* Grow the memory at the end, by the size the const variable
                 * needs. Note that 'ga' holds the last const variable.
                 */
                assert(WN_operator(hcls->ga->gvar_sz) == OPR_INTCONST);
                cmem_sz += WN_const_val(hcls->ga->gvar_sz);

                cvar_before = ga; cvar_after = NULL;
            }

            /* Allocate the const variable. If the fragment is bigger
             * than what the variable needs, the allocation can take the
             * beginning or the end of the fragment, or less likely
             * anywhere in between. For now, we always allocate at the
             * beginning.
             */
            if (cvar_before == NULL) {
                cvar_alive_list = hcls->ga;
                hcls->ga->alloc_offset = 0;
            } else {
                cvar_before->next = hcls->ga;
                hcls->ga->alloc_offset = cvar_before->alloc_offset +
                    WN_const_val(cvar_before->gvar_sz);
            }
            hcls->ga->next = cvar_after;

        } else {
            /* Move the const variable from alive list to done list. */

            struct hc_gmem_alias *curr_ga = cvar_alive_list, *prev_ga = NULL;
            while (curr_ga != NULL) {
                if (curr_ga == hcls->ga) break;
                prev_ga = curr_ga;
                curr_ga = curr_ga->next;
            }
            assert(curr_ga != NULL);

            if (prev_ga != NULL) {
                prev_ga->next = curr_ga->next;
            } else {
                cvar_alive_list = curr_ga->next;
            }

            curr_ga->next = cvar_done_list;
            cvar_done_list = curr_ga;
        }

        hcls = hcls->next;
    }

    assert(cvar_alive_list == NULL);

    /* Modify the array bound of the real const variable. */

    assert(cvar_done_list != NULL);
    TY_IDX cvar_ty_idx = ST_type(cvar_done_list->gvar_st_idx);
    assert(set_arr_dim_sz(cvar_ty_idx, 0, cmem_sz));

    /* For each constant variable, update the offset parameter of the call
     * to cudaMemcpyToSymbol, and the offset field in references.
     */
    struct hc_gmem_alias *ga = cvar_done_list;
    while (ga != NULL) {
        printf("Variable %s is allocated in cmem at offset %u.\n",
                ST_name(ga->ori_st_idx), ga->alloc_offset);

        // Update the offset parameter in the call to cudaMemcpyToSymbol.
        WN *tcall = ga->init_point;
        assert(tcall != NULL && WN_operator(tcall) == OPR_CALL);
        WN *ofst_wn = WN_kid0(WN_kid(tcall,3));
        WN_kid0(WN_kid(tcall,3)) = WN_Add(WN_rtype(ofst_wn),
                ofst_wn, WN_Intconst(WN_rtype(ofst_wn), ga->alloc_offset));

        // For each access to the constant variable, search for the LDA node
        // in it and increment the offset field by `alloc_offset'.
        WN_SET::iterator it = ga->cvar_refs->begin();
        while (it != ga->cvar_refs->end()) {
            assert(update_cvar_alloc_offset(*it,
                        ga->gvar_st_idx, ga->alloc_offset));
            ++it;
        }

        ga = ga->next;
    }

    // Clean up.
    cvar_done_list = NULL;  // no need to free them
    cmem_sz = 0;
}

void
reset_cvar_live_ranges(struct hc_cvar_life *hcls) {
    struct hc_cvar_life *hcl = cvar_live_range_head, *tmp = NULL;
    while (hcl != NULL) {
        tmp = hcl;
        hcl = hcl->next;

        // We only free the struct once for start of the live range.
        if (tmp->start) {
            assert(tmp->ga != NULL);
            free_gmem_alias(tmp->ga);
        }
        free(tmp);
    }

    cvar_live_range_head = cvar_live_range_tail = NULL;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

void
validate_hc_data_context(WN *scope) {
    assert(scope != NULL && WN_operator(scope) == OPR_BLOCK);

    // Check the global variable list.
    GMEM_ALIAS_MAP::iterator ga_it = gvar_aliases.begin();
    while (ga_it != gvar_aliases.end()) {
        struct hc_gmem_alias *ga = ga_it->second;
        assert(ga != NULL);

        // The front one must be in a scope same or outside 'scope'.
        Is_True(ga->scope != scope,
            ("The global variable for %s is not deallocated.\n",
                ST_name(ga->ori_st_idx)));

        ga_it++;
    }

    // Check the const variable list.
    ga_it = cvar_aliases.begin();
    while (ga_it != cvar_aliases.end()) {
        struct hc_gmem_alias *ga = ga_it->second;
        assert(ga != NULL);

        // The front one must be in a scope same or outside 'scope'.
        Is_True(ga->scope != scope,
            ("The constant variable for %s is not deallocated.\n",
                ST_name(ga->ori_st_idx)));

        ga_it++;
    }

    // Check the shared variable list.
    SVAR_ALIAS_MAP::iterator sa_it = svar_aliases.begin();
    while (sa_it != svar_aliases.end()) {
        struct hc_smem_alias *sa = sa_it->second;
        assert(sa != NULL);

        // The front one must be in a scope same or outside 'scope'.
        Is_True(sa->scope != scope,
            ("The shared variable for %s is not removed.\n",
                ST_name(sa->ori_st_idx)));

        sa_it++;
    }
}

void
validate_at_kernel_end() {
    assert(lpi_top == NULL);
    assert(svar_aliases.size() == 0);
}

void init_hc_data_context() {
}

void finish_hc_data_context() {
    assert(gvar_aliases.size() == 0);
    assert(cvar_aliases.size() == 0);
    assert(svar_aliases.size() == 0);
}

/*** DAVID CODE END ***/
