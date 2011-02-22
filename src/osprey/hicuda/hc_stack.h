/** DAVID CODE BEGIN **/

#ifndef _HICUDA_HC_STACK_H_
#define _HICUDA_HC_STACK_H_

#include <set>

#include "wn.h"

/* NOTE:
 *
 * All WN instances in hc_gmem_alias or hc_smem_alias must be copied
 * before use, whether or not they should be freed here.
 */

/**
 * all information about a kernel
 */
struct kernel_context {
    // Whether the info is valid.
    bool valid;

    ST_IDX kfunc_st_idx;

    // Parameter info: global variable symbols and others in the original PU
    INT nparams;
    ST_IDX *params;

    /* Virtual block and thread space */

    WN **vgrid_dims;
    UINT n_vgrid_dims;
    // An array of length n_vgrid_dims (fresh WN)
    WN **vblk_idx;

    WN **vblk_dims;
    UINT n_vblk_dims;
    // An array of length n_vblk_dims (fresh WN)
    WN **vthr_idx;

    /* Physical block and thread space */

    // grid and block dimensions
    WN* grid_dims[3];
    WN* blk_dims[3];

    /* Running context for pragmas inside a kernel */

    // Next available dimension index in the virtual block space
    UINT curr_vgrid_dim_idx;
    // Next available dimension index in the virtual thread space
    UINT curr_vblk_dim_idx;

    // A list of shared variable live status (alloc/dealloc)
    struct hc_svar_life *hsl_head;
    struct hc_svar_life *hsl_tail;

    // A reference to the kernel body (a BLOCK node)
    // This is used if an inner directive wants to add code to the kernel.
    WN *kernel_body;

    // the body that should replace the kernel once it is outlined
    WN *replacement;
};

// Kernels cannot be nested, so one context is enough.
extern struct kernel_context kinfo;

/**
 * Record the start/end of a shared variable's live range
 * in the kernel context.
 */
extern void append_svar_life(struct hc_smem_alias *sa, bool start);

/**
 * Called when the processing of a kernel ends.
 */
extern void reset_kinfo();

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * information about a loop (DO_LOOP or WHILE_DO) in the kernel:
 * - whether it is block/thread-partitioned (could be neither)
 * - block range (the whole loop iteration for a non-partitioned loop)
 */
struct loop_part_info {
    // loop index variable, ST_IDX_ZERO if this is not a DO_LOOP
    ST_IDX idxv_st_idx;

    // dimension index of the virtual block space, -1 if not partitioned
    INT vgrid_dim_idx;
    // dimension index of the virtual thread space, -1 if not partitioned
    INT vblk_dim_idx;

    /* a TRIPLET that specifies the loop index range assigned to each block
     * based on the index in the dimension of the virtual block space, that
     * is assigned to this loop. The range stride is always a positive
     * constant.
     *
     * For example, loop I is block-partitioned, which takes the 2nd dimension
     * of the virtual block space. After calculation, the loop's index space
     * is partitioned into chunks of size 5. Then this triplet will be
     * [vblk[1]*5, (vblk[1]+1)*5), where vblk[1] is the index in the 2nd
     * dimension of the virtual block space.
     *
     * The range could be NULL if it is too hard to determine.
     */
    WN *blk_range;

    /* a TRIPLET that specifies the full loop index range. */
    WN *full_range;

    struct loop_part_info *parent;
};

extern struct loop_part_info* loop_stack_top();

extern void push_loop_info(struct loop_part_info *lpi);

extern struct loop_part_info* pop_loop_info();

extern void free_loop_info(struct loop_part_info *lpi);

/**
 * Return the loop stack's size.
 */
extern UINT num_enclosing_loops();

/**
 * Filter DO_LOOPs in the loop stack with the following criteria:
 * - range of loops
 * - only need thread-partitioned loops?
 *
 * The loop range is specified as follows:
 *
 * If 'inner' is true, we go from stack top to 'loop' (stack bottom if
 * NULL, exclusive). Otherwise, we go from 'loop' (stack top if NULL,
 * inclusive) to stack bottom. Therefore, when 'loop' is NULL, 'inner'
 * does not matter.
 *
 * If 'all_doloops' is not NULL, it will store if there is any non-DO_LOOPs
 * in the range of loops specified.
 *
 * Put all DO_LOOPs that meet the criteria to 'lpi_arr' and return the number
 * of such loops.
 */
extern UINT filter_doloops(struct loop_part_info **lpi_arr,
    struct loop_part_info *loop, bool inner, bool only_thr_partitioned,
    bool *all_doloops);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

typedef std::set<WN*> WN_SET;

/**
 * information about an alias in the GPU's global memory, for an array
 * variable in the host memory. The GPU variable could be a global variable
 * or a constant variable.
 *
 * The global and constant variables only share the data structure, because
 * the code for data transfer can be reused. They have different stacks.
 */
struct hc_gmem_alias {
    // the original variable
    ST_IDX ori_st_idx;

    /* corresponding variable in the global memory (including const mem)
     * The variable type (global/const) can be determined by ST attributes.
     * All const variable aliases in a PU refer to a single const variable.
     */
    ST_IDX gvar_st_idx;

    // the block in which the gmem variable is declared
    // THIS IS A REFERENCE, DO NOT FREE IT.
    WN *scope;

    // An array of dimension sizes of the original variable
    // THIS SHOULD BE FREED HERE.
    WN **var_dim_sz;

    // An ARRSECTION node that stores the starting/ending index,
    // and size of each dimension of the gmem variable
    // THIS SHOULD BE FREED HERE.
    WN *gvar_info;

    // For each dimension of the gmem variable, is it identical
    // to the corresponding dimension of the original variable?
    // THIS SHOULD BE FREED HERE.
    bool *is_full_idx_range;

    // One-based index of the first partial dimension from the right
    // or zero if it does not exist
    int partial_dim_idx;

    /* size of the gmem variable in bytes
     * This can be easily computed from 'gvar_info', but let's cache it.
     * THIS SHOULD BE FREED HERE.
     */
    WN *gvar_sz;

    /* For a constant variable, this stores where the variable is allocated
     * in the per-PU constant variable.
     * This will be determined after the whole PU is processed.
     */
    UINT alloc_offset;

    /* For a global variable, this points to the call to cudaMalloc.
     *
     * For a constant variable, this points to the call to cudaMemcpyToSymbol,
     * whether or not it is surrounded by a loop nest. The offset parameter
     * of this call should be incremented by 'alloc_offset', after the whole
     * PU is processed.
     *
     * THIS IS A REFERENCE, DO NOT FREE IT.
     */
    WN *init_point;

    /* Used exclusively by a constant variable, to store a list of WN nodes
     * that reference this variable. The offset parameter in these nodes must
     * be incremented by 'alloc_offset' after the whole PU is processed.
     */
    WN_SET *cvar_refs;

    struct hc_gmem_alias *next;
};

/**
 * Find the visible global variable corresponding to 'ori_st_idx'.
 *
 * Return the data structure that holds all info about this variable.
 */
extern struct hc_gmem_alias* visible_global_var(ST_IDX ori_st_idx);

/**
 * Find the visible const variable corresponding to 'ori_st_idx'.
 *
 * Return the data structure that holds all info about this variable.
 */
extern struct hc_gmem_alias* visible_const_var(ST_IDX ori_st_idx);

/**
 * Add a variable and its corresponding global variable.
 */
extern struct hc_gmem_alias* add_gvar_alias(
    ST_IDX gvar_st_idx, ST_IDX ori_st_idx, WN *scope);

/**
 * Add a variable and its corresponding const variable.
 */
extern struct hc_gmem_alias* add_cvar_alias(
    ST_IDX ori_st_idx, WN *scope);

/**
 * Remove the global variable for 'ori_st_idx' declared in the given scope.
 *
 * Return the removed global variable or NULL if there is no matched one.
 */
extern struct hc_gmem_alias* remove_gvar_alias(ST_IDX ori_st_idx, WN *scope);

/**
 * Remove the const variable for 'ori_st_idx' declared in the given scope.
 *
 * Return the removed const variable or NULL if there is no matched one.
 */
extern struct hc_gmem_alias* remove_cvar_alias(ST_IDX ori_st_idx, WN *scope);

extern void free_gmem_alias(struct hc_gmem_alias *ga);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

struct hc_smem_alias {
    // original variable
    ST_IDX ori_st_idx;
    // corresponding shared variable
    ST_IDX svar_st_idx;

    // the block in which the global variable is declared
    // THIS IS A REFERENCE, DO NOT FREE IT.
    WN *scope;

    /* the innermost loop inside the kernel that contains the initialization
     * of this shared variable
     * THIS IS A REFERENCE, DO NOT FREE IT.
     */
    struct loop_part_info *loop_scope;

    // global memory alias info
    // THIS IS A REFERENCE, DO NOT FREE IT.
    struct hc_gmem_alias *ga;

    // An ARRSECTION node that stores size, starting/ending indices
    // of each dimension of the shared variable.
    // THIS SHOULD BE FREED HERE.
    WN *svar_info;

    /* The size of the shared variable in bytes. This can be easily computed
     * from 'svar_info', but let's just cache it here for easy access.
     */
    UINT svar_size;

    /* Where the shared variable is allocated in the per-block shared memory.
     * This is most likely determined after the whole kernel is processed.
     */
    UINT alloc_offset;

    /* the STID node that computes the shared variable's starting address
     * The expression should be filled after the whole kernel is processed.
     * THIS IS A REFERENCE, DO NOT FREE IT.
     */
    WN *init_point;

    /* Information cached when COPYIN is lowered, which can be reused
     * when lowering COPYOUT.
     */

    // copyin code block (not including initialization statements
    // of kernel invariant subscripts)
    // THIS SHOULD BE FREED HERE.
    WN *copyin_wn;

    struct hc_smem_alias *next;
};

/**
 * Find the visible shared variable corresponding to 'ori_st_idx'.
 *
 * Return the data structure that holds all info about this variable.
 */
extern struct hc_smem_alias* visible_shared_var(ST_IDX ori_st_idx);

/**
 * Add a variable and its corresponding shared variable to
 * the top stack element. It also finds the corresponding global
 * variable info and store it in field 'ga'.
 */
extern struct hc_smem_alias* add_svar_alias(
    ST_IDX svar_st_idx, ST_IDX ori_st_idx, WN *scope);

/**
 * Remove the shared variable for 'ori_st_idx' declared in the given scope.
 *
 * Return the removed shared variable or NULL if there is no matched one.
 */
extern struct hc_smem_alias* remove_svar_alias(ST_IDX ori_st_idx, WN *scope);

extern void free_svar_alias(struct hc_smem_alias *sa);

/**
 * If 'wn' accesses a variable that has corresponding global or shared
 * variable, it will be replaced with an access to the latter.
 *
 * For now, 'wn' must be OPR_ARRAY, OPR_LDID, or OPR_STID.
 *
 * Return true if the replacement occurs, and false otherwise.
 *
 * If '*new_wn' is NULL, 'wn' has experience internal changes and the client
 * can use it normally. If '*new_wn' is not NULL, 'wn' should completely
 * replaced with '*new_wn' and can not be used anymore. However, the client
 * is responsible for deallocating 'wn'.
 */
extern bool replace_var_access(WN *wn, WN **new_wn);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Form a list of svar life range records. The list is kept in the kernel
 * context. After the kernel is processed, some algorithm is applied on this
 * list to figure out the minimal amount of shared memory needed by these
 * requests.
 */
struct hc_svar_life {
    // true for start of life, false for end of life
    bool start;

    // should NOT exist in the global stack of svars
    struct hc_smem_alias *sa;

    struct hc_svar_life *next;
};

/**
 * Declare a shared variable in the current scope, that holds the minimal
 * amount of memory needed by all memory requests in the given list.
 *
 * Each shared variable's offset w.r.t. to created single shared variable
 * is set in sa->init_point.
 *
 * Return the newly declared shared variable symbol.
 */
extern ST_IDX analyze_svar_live_ranges(struct hc_svar_life *hsls);

/**
 * Form a list of const variable life range records. The list is kept
 * internally. After a PU is processed, some algorithm is applied on this
 * list to figure out the minimal amount of constant memory needed by these
 * requests.
 */
struct hc_cvar_life {
    // true for start of life, false for end of life
    bool start;

    // should NOT exist in the global stack of cvars
    struct hc_gmem_alias *ga;

    struct hc_cvar_life *next;
};

/**
 * Record the start/end of a const variable's live range.
 */
extern void append_cvar_life(struct hc_gmem_alias *ga, bool start);

extern struct hc_cvar_life* get_cvar_live_ranges();

/**
 * Determine the minimal amount of constant memory needed by all requests
 * from the given list of const variables.
 *
 * Each const variable's offset w.r.t. to the single const variable will
 * be used to increment the offset field in ga->init_point.
 *
 * The single const variable's array bound will also be updated.
 */
extern void analyze_cvar_live_ranges(struct hc_cvar_life *hcls);

/**
 * Deallocate the internal list of const variable live ranges.
 */
extern void reset_cvar_live_ranges(struct hc_cvar_life *hcls);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Make sure there is no global or shared variable left in the list, that is
 * declared in the given scope, which is about to end.
 */
extern void validate_hc_data_context(WN *scope);

/**
 * Make sure that the loop stack and the shared variable stack are clear
 * after the whole kernel is processed.
 */
extern void validate_at_kernel_end();

extern void init_hc_data_context();

extern void finish_hc_data_context();

#endif  // _HICUDA_HC_STACK_H_

/*** DAVID CODE END ***/
