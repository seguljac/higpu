/** DAVID CODE BEGIN **/

// needed by ipl_summary.h included in ipa_cg.h
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "wn.h"
#include "tracing.h"
#include "ir_reader.h"

#include "ipa_hc_common.h"
#include "ipa_hc_kernel.h"

#include "ipa_cg.h"
#include "ipa_preopt.h"
#include "ipa_section_annot.h"

#include "ipo_defs.h"

#include "opt_du.h"

#if 0

/*****************************************************************************
 *
 * Check if the given WHIRL node is a kernel region.
 *
 ****************************************************************************/

static inline BOOL is_kernel_region(WN *wn)
{
    OPERATOR opr = WN_operator(wn);

    if (opr != OPR_REGION
            || WN_region_kind(wn) != REGION_KIND_HICUDA) return FALSE;

    // Get the first pragma in the pragma block.
    WN *pragma = WN_first(WN_kid1(wn));
    return (pragma != NULL
            && (WN_PRAGMA_ID)WN_pragma(pragma) == WN_PRAGMA_HC_KERNEL);
}

/*****************************************************************************
 *
 * Walk through the given procedure (FUNC_ENTRY), and do the following:
 * - Make sure that kernel regions are not nested.
 * - Make sure indirect loads/stores have ARRAY as the address.
 * - For each DEF/USE in the DU-chain, mark it inside/outside a kernel region
 *   (if inside, which kernel region).
 *
 * This is a recursive function. Make sure the global states are set up
 * properly before calling it. After the function finishes,
 * <g_inside_kernel_map> is filled.
 *
 ****************************************************************************/

/* Context set up before the call */
static IPA_NODE *g_proc_node = NULL;
static DU_MANAGER *g_du_mgr = NULL;

/* Keep track of the kernel context during the call.
 * 
 * This variable is overwritten and reused in the <construct_das> to store the
 * kernel region currently being processed.
 */
static WN *g_curr_kernel_region = NULL;

/* A map from each WN node to its parent kernel region (or NULL), to be used
 * in <construct_das>
 */
static WN_MAP g_inside_kernel_map = WN_MAP_UNDEFINED;

static void IPA_HC_kernel_preprocess(WN *wn)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    BOOL is_kregion = is_kernel_region(wn);
   
    if (is_kregion)
    {
        // Make sure there is no nested kernel regions.
        Is_True(g_curr_kernel_region == NULL,
                ("IPA_HC_kernel_preprocess: nested kernel in %s\n",
                 ST_name(g_proc_node->Func_ST())));
        g_curr_kernel_region = wn;

    }

    if (opr == OPR_ILOAD)
    {
        // Make sure the address (kid 0) is ARRAY.
        WN *addr_wn = WN_kid0(wn);
        Is_True(WN_operator(addr_wn) == OPR_ARRAY,
                ("IPA_HC_kernel_preprocess: invalid ILOAD\n"));

        // Make sure that the address's base is an array variable (or a
        // pointer to an array variable).
        WN *addr_base_wn = WN_array_base(addr_wn);
        OPERATOR addr_base_opr = WN_operator(addr_base_wn);
        Is_True(addr_base_opr == OPR_LDID || addr_base_opr == OPR_LDA,
                ("IPA_HC_kernel_preprocess: invalid ILOAD ARRAY\n"));
        // TODO: check further.
    }
    else if (opr == OPR_ISTORE)
    {
        // Make sure the address (kid 1) is ARRAY.
        WN *addr_wn = WN_kid1(wn);
        Is_True(WN_operator(addr_wn) == OPR_ARRAY,
                ("IPA_HC_kernel_preprocess: invalid ISTORE\n"));

        // Make sure that the address's base is an array variable (or a
        // pointer to an array variable).
        WN *addr_base_wn = WN_array_base(addr_wn);
        OPERATOR addr_base_opr = WN_operator(addr_base_wn);
        Is_True(addr_base_opr == OPR_LDID || addr_base_opr == OPR_LDA,
                ("IPA_HC_kernel_preprocess: invalid ILOAD ARRAY\n"));
        // TODO: check further.
    }

    // Record whether the current node is inside/outside any kernel region.
    WN_MAP_Set(g_inside_kernel_map, wn, g_curr_kernel_region);

    /* Handle composite node. */
    if (opr == OPR_BLOCK)
    {
        WN *node = WN_first(wn);
        while (node != NULL) {
            IPA_HC_kernel_preprocess(node);
            node = WN_next(node);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            IPA_HC_kernel_preprocess(WN_kid(wn,i));
        }
    }

    if (is_kregion) g_curr_kernel_region = NULL;
}


/*****************************************************************************
 *
 * A pointer parameter of a call within a kernel region must be one of the
 * following two:
 * 1) LDID of an array pointer variable
 * 2) LDA of an array variable
 *
 * This function checks the given parameter node (the one within PARM, not the
 * PARM itself). It return the array symbol or NULL if failed.
 *
 ****************************************************************************/

static ST* verify_pointer_param(WN *param_wn)
{
    OPERATOR opr = WN_operator(param_wn);

    if (opr == OPR_LDID)
    {
        ST *arr_st = WN_st(param_wn);
        TY_IDX ty_idx = ST_type(arr_st);
        return (TY_kind(ty_idx) == KIND_POINTER
            && TY_kind(TY_pointed(ty_idx)) == KIND_ARRAY) ? arr_st : NULL;
    }
    else if (opr == OPR_LDA)
    {
        ST *arr_st = WN_st(param_wn);
        return (TY_kind(ST_type(arr_st)) == KIND_ARRAY) ? arr_st : NULL;
    }

    return NULL;
}


typedef STACK<LOOPINFO*> LOOPINFO_STACK;

/*****************************************************************************
 *
 * Project the given array region w.r.t. to the given loop stack. We will
 * traverse the stack from top to bottom (i.e. from the innermost loop to the
 * outermost).
 *
 ****************************************************************************/

static inline void project_arr_region(PROJECTED_REGION *pr,
        LOOPINFO_STACK *stack)
{
    INT nlevels = stack->Elements();
    for (INT i = 0; i < nlevels; ++i) {
        LOOPINFO *li = stack->Top_nth(i);
        pr->Project(li->Get_nest_level(), li);
    }
}

/*****************************************************************************
 *
 * Recursively go through the procedure node tree. For each kernel, create and
 * construct its data access summary and store it in the corresponding
 * IPA_NODE.
 *
 * Before calling this function:
 * - <g_proc_node> is the procedure node to be processed,
 * - <g_curr_kernel_region> is NULL,
 * - <g_process_scalar_only> is FALSE
 * - <g_doloop_stack> and <g_doloop_wn_stack> allocated
 *
 ****************************************************************************/

/* A stack of DO_LOOP info maintained during the construction of data access
 * summary (DAS).
 */
static LOOPINFO_STACK *g_doloop_stack = NULL;
static DOLOOP_STACK *g_doloop_wn_stack = NULL;

static BOOL g_process_scalar_only = FALSE;

static HC_KERNEL_DATA *g_kernel_data = NULL;

static void construct_das(WN *wn)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);
    // Skip pragmas.
    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA) return;

    // Whether or not to continue handling child nodes
    BOOL handle_composite = TRUE;

    /* Prepare kernel context when meeting a kernel region. */
    BOOL is_kregion = is_kernel_region(wn);
    if (is_kregion)
    {
        // Create a HC_KERNEL_DATA for this kernel and add it to the IPA node.
        g_curr_kernel_region = wn;
        g_kernel_data = g_proc_node->add_kernel_data(wn);
    }

    /* Update the DO_LOOP stack when meeting a DO_LOOP. */
    if (opr == OPR_DO_LOOP)
    {
        // <IPA_get_procedure_array> only returns a per-file handle.
        SUMMARY_PROCEDURE *sp = IPA_get_procedure_array(g_proc_node)
            + g_proc_node->Summary_Proc_Index();

        // I do not know how to obtain the SUMMARY_CONTROL_DEPENDENCE
        // corresponding to this WN node, so I will just do a brute-force
        // search.
        SUMMARY_CONTROL_DEPENDENCE *scd = IPA_get_ctrl_dep_array(g_proc_node);
        INT scd_idx_start = sp->Get_ctrl_dep_index();
        INT scd_idx_end = scd_idx_start + sp->Get_ctrl_dep_count();
        INT scd_idx;
        for (scd_idx = scd_idx_start; scd_idx < scd_idx_end; ++scd_idx) {
            if (scd[scd_idx].Get_map_id() == WN_map_id(wn)) break;
        }
        Is_True(scd_idx < scd_idx_end,
                ("construct_das (%s): no SUMMARY_CD for loop %s",
                 ST_name(g_proc_node->Func_ST()),
                 ST_name(WN_st(WN_index(wn)))));

        // Now, I do not know how to obtain the CFG_NODE_INFO from SUMMARY_CD,
        // so a brute-force search.
        CFG_NODE_INFO *cni = IPA_get_cfg_node_array(g_proc_node);
        INT cni_idx_start = sp->Get_array_section_index();
        INT cni_idx_end = cni_idx_start + sp->Get_array_section_count();
        INT cni_idx;
        for (cni_idx = cni_idx_start; cni_idx < cni_idx_end; ++cni_idx) {
            // The SUMMARY_CD index is per-file.
            if (cni[cni_idx].Is_do_loop()
                && cni[cni_idx].Get_cd_index() == scd_idx) break;
        }
        Is_True(cni_idx < cni_idx_end,
                ("construct_das (%s): no CFG_NODE_INFO for loop %s",
                 ST_name(g_proc_node->Func_ST()),
                 ST_name(WN_st(WN_index(wn)))));

        // IMPORTANT: need -1, see Map_loop_info
        // We use the second struct in the union in LOOPINFO.
        LOOPINFO *li = IPA_get_loopinfo_array(g_proc_node)
            + cni[cni_idx].Get_loop_index() - 1;
        Is_True(li != NULL,
                ("construct_das (%s): no LOOPINFO for loop %s\n",
                 ST_name(g_proc_node->Func_ST()),
                 ST_name(WN_st(WN_index(wn)))));

        // Map the LOOPINFO structure to memory, i.e. using the 1st struct.
        // TODO: is it ok to use MEM_local_pool?
        LOOPINFO* copy_li = CXX_NEW(
                LOOPINFO(&MEM_local_pool, li->Get_cd_idx()), &MEM_local_pool);
        copy_li->Set_nest_level(li->Get_nest_level());
        copy_li->Set_flags(li->Get_flags());
        copy_li->Set_ub_term_index(li->Get_ub_term_index());
        copy_li->Set_lb_term_index(li->Get_lb_term_index());
        copy_li->Set_step_term_index(li->Get_step_term_index());
        copy_li->Set_ub_term_count(li->Get_ub_term_index());
        copy_li->Set_lb_term_count(li->Get_lb_term_index());
        copy_li->Set_step_term_count(li->Get_step_term_index());
        copy_li->Create_linex(IPA_get_term_array(g_proc_node));

        // Push the LOOPINFO object onto the stack.
        g_doloop_stack->Push(copy_li);
        g_doloop_wn_stack->Push(wn);
    }

    if (g_curr_kernel_region != NULL)
    {

    /* We only care about the following nodes:
     * - LDID, ILOAD, STID, ISTORE
     * - bit counterparts of the above four nodes
     * - LDA, LDMA
     *
     * TODO: CONST?
     */
    if (opr == OPR_LDID || opr == OPR_LDBITS)
    {
        ST *st = WN_st(wn);

        if (g_process_scalar_only
            && TY_kind(ST_type(st)) != KIND_SCALAR) return;

        /* If at least one of its DEFs is outside the kernel region, add this
         * USE to the data access summary.
         */
        DEF_LIST_ITER defs_iter(g_du_mgr->Ud_Get_Def(wn));

        BOOL def_outside = FALSE;
        for (DU_NODE *def = defs_iter.First();
                !defs_iter.Is_Empty(); def = defs_iter.Next()) {
            // either outside any kernel region, or inside some different
            // kernel region
            if (WN_MAP_Get(g_inside_kernel_map, def->Wn())
                        != g_curr_kernel_region) {
                def_outside = TRUE;
                break;
            }
        }

        if (def_outside) g_kernel_data->add_scalar(st, WN_offset(wn), TRUE);
    }
    else if (opr == OPR_STID || opr == OPR_STBITS)
    {
        ST *st = WN_st(wn);

        /* If at least one of its USEs is outside the kernel region, add this
         * DEF to the data access summary.
         */
        USE_LIST_ITER uses_iter(g_du_mgr->Du_Get_Use(wn));

        BOOL use_outside = FALSE;
        for (DU_NODE *use = uses_iter.First();
                !uses_iter.Is_Empty(); use = uses_iter.Next()) {
            // either outside any kernel region, or inside some different
            // kernel region
            if (WN_MAP_Get(g_inside_kernel_map, use->Wn())
                        != g_curr_kernel_region) {
                use_outside = TRUE;
                break;
            }
        }

        if (use_outside) g_kernel_data->add_scalar(st, WN_offset(wn), FALSE);
    }
    else if (opr == OPR_CALL)
    {
        // A call needs special handling: we need to process the call's side
        // effects as opposed to its literal arguments.
        IPA_EDGE *e = IPA_get_ipa_edge(g_proc_node, wn);
        Is_True(e != NULL,
                ("construct_das (%s): no IPA_EDGE for call %s\n",
                 ST_name(g_proc_node->Func_ST()), ST_name(WN_st(wn))));

        SUMMARY_CALLSITE *call = e->Summary_Callsite();

        IPA_NODE *callee = IPA_Call_Graph->Callee(e);
        IPA_NODE_SECTION_INFO *callee_info = callee->Section_Annot();

        SUMMARY_FORMAL *callee_formals = IPA_get_formal_array(callee);
        SUMMARY_ACTUAL *caller_actuals = IPA_get_actual_array(g_proc_node);
        SUMMARY_ACTUAL *actuals = caller_actuals + call->Get_actual_index();

        MEM_POOL *pool = g_proc_node->Mem_Pool();

        /* Go through each formal of the callee:
         * - If the formal is a scalar, process the corresponding actual as
         *   normal.
         * - If the formal is an array, map the MOD/REF regions to the
         *   caller's space and process them.
         */
        INT formal_count = callee->Summary_Proc()->Get_formal_count();
        INT actual_count = e->Num_Actuals();
        Is_True(formal_count == actual_count,
                ("constrct_das (%s): unequal formal/actual count for call %s",
                 ST_name(g_proc_node->Func_ST()), ST_name(WN_st(wn))));

        for (INT i = 0; i < formal_count; ++i)
        {
            STATE *callee_annot = callee_info->Get_formal(i);
            if (callee_annot->Is_scalar()) {
                construct_das(WN_actual(wn,i));
            } else {
                // Sanity check: a pointer argument of the call must be an
                // LDID of an array pointer, or an LDA of an array.
                WN *param_wn = WN_kid(wn,i);
                ST *arr_st = verify_pointer_param(WN_kid0(param_wn));
                Is_True(TY_kind(WN_ty(param_wn)) == KIND_POINTER
                        && arr_st != NULL,
                        ("construct_das (%s): parameter #%d of call %s "
                         "is invalid\n",
                         ST_name(g_proc_node->Func_ST()), i,
                         ST_name(WN_st(wn))));

                // MOD info
                PROJECTED_REGION *callee_region = 
                    callee_annot->Get_projected_mod_region();
                PROJECTED_REGION *caller_region = CXX_NEW(
                        PROJECTED_REGION(callee_region->Get_type(),
                                callee_region->Get_depth(),
                                callee_region->Get_num_dims(), pool), pool);
                Map_callee_region_to_caller(g_proc_node, callee, call,
                        caller_region, callee_region);
                project_arr_region(caller_region, g_doloop_stack);
                g_kernel_data->add_arr_region(arr_st, caller_region,
                        FALSE, wn, i);

                // REF info
                callee_region = callee_annot->Get_projected_ref_region();
                caller_region = CXX_NEW(
                        PROJECTED_REGION(callee_region->Get_type(),
                                callee_region->Get_depth(),
                                callee_region->Get_num_dims(), pool), pool);
                Map_callee_region_to_caller(g_proc_node, callee, call,
                        caller_region, callee_region);
                project_arr_region(caller_region, g_doloop_stack);
                g_kernel_data->add_arr_region(arr_st, caller_region,
                        TRUE, wn, i);
            }
        }

        /* Go through each global variable accessed by the call, and do the
         * same thing as actuals.
         */
        GLOBAL_ARRAY_TABLE *callee_tbl = callee_info->Global_Array_Table();

        // Walk through the commons.
        ST_IDX st;
        GLOBAL_ARRAY_LIST *callee_list = NULL;
        GLOBAL_ARRAY_TABLE_ITER callee_tbl_iter(callee_tbl);
        while (callee_tbl_iter.Step(&st, &callee_list))
        {
            // Since we don't have common block in C, we skip messy ones.
            if (callee_list->Is_messy()) continue;

            // Walk through all common elements and merge mod/ref regions.
            GLOBAL_ARRAY_LIST_ITER iter(callee_list);
            for (iter.First(); !iter.Is_Empty(); iter.Next())
            {
                GLOBAL_ARRAY_INFO *callee_info = iter.Cur();
                ST *arr_st = &St_Table[callee_info->St_Idx()];
                STATE *callee_annot = callee_info->Get_state();

                // MOD info
                PROJECTED_REGION *callee_region = 
                    callee_annot->Get_projected_mod_region();
                PROJECTED_REGION *caller_region = CXX_NEW(
                        PROJECTED_REGION(callee_region->Get_type(),
                                callee_region->Get_depth(),
                                callee_region->Get_num_dims(), pool), pool);
                Map_callee_region_to_caller(g_proc_node, callee, call,
                        caller_region, callee_region);
                project_arr_region(caller_region, g_doloop_stack);
                g_kernel_data->add_arr_region(arr_st, caller_region,
                        FALSE, wn, -1);

                // REF info
                callee_region = callee_annot->Get_projected_ref_region();
                caller_region = CXX_NEW(
                        PROJECTED_REGION(callee_region->Get_type(),
                                callee_region->Get_depth(),
                                callee_region->Get_num_dims(), pool), pool);
                Map_callee_region_to_caller(g_proc_node, callee, call,
                        caller_region, callee_region);
                project_arr_region(caller_region, g_doloop_stack);
                g_kernel_data->add_arr_region(arr_st, caller_region,
                        TRUE, wn, -1);
            }
        }

        handle_composite = FALSE;
    }
    else if (opr == OPR_ILOAD || opr == OPR_ILDBITS)
    {
        WN *addr_wn = WN_kid0(wn);
        MEM_POOL *pool = g_proc_node->Mem_Pool();

        /* Turn the WHIRL node into a PROJECT_REGION. */

        // TODO: use a temp mempool
        ACCESS_ARRAY *array = CXX_NEW(
                ACCESS_ARRAY(WN_num_dim(addr_wn),
                    g_doloop_wn_stack->Elements(), pool), pool);
        array->Set_Array(addr_wn, g_doloop_wn_stack);

        PROJECTED_REGION *pr = CXX_NEW(PROJECTED_REGION(array, pool,
                    g_doloop_stack->Top(), FALSE, NULL), pool);

        // REF info
        project_arr_region(pr, g_doloop_stack);
        // assuming LDID or LDA
        g_kernel_data->add_arr_region(WN_st(WN_array_base(addr_wn)), pr,
                TRUE, addr_wn);

        g_process_scalar_only = TRUE;
    }
    else if (opr == OPR_ISTORE || opr == OPR_ISTBITS)
    {
        WN *addr_wn = WN_kid1(wn);
        MEM_POOL *pool = g_proc_node->Mem_Pool();

        /* Turn the WHIRL node into a PROJECT_REGION. */

        // TODO: use a temp mempool
        ACCESS_ARRAY *array = CXX_NEW(
                ACCESS_ARRAY(WN_num_dim(addr_wn),
                    g_doloop_wn_stack->Elements(), pool), pool);
        array->Set_Array(wn, g_doloop_wn_stack);

        PROJECTED_REGION *pr = CXX_NEW(
                PROJECTED_REGION(array, pool, NULL, FALSE, NULL), pool);

        // MOD info
        project_arr_region(pr, g_doloop_stack);
        // assuming LDID or LDA
        g_kernel_data->add_arr_region(WN_st(WN_array_base(addr_wn)), pr,
                FALSE, addr_wn);

        g_process_scalar_only = TRUE;
    }

    }

    /* Handle composite nodes. */
    if (handle_composite)
    {
        if (opr == OPR_BLOCK) {
            WN *node = WN_first(wn);
            while (node != NULL) {
                construct_das(node);
                node = WN_next(node);
            }
        } else {
            INT nkids = WN_kid_count(wn);
            for (INT i = 0; i < nkids; ++i) construct_das(WN_kid(wn,i));
        }
    }

    if (opr == OPR_ILOAD || opr == OPR_ILDBITS
            || opr == OPR_ISTORE || opr == OPR_ISTBITS) {
        g_process_scalar_only = FALSE;
    }

    /* Pop the DO_LOOP from the stack. */
    if (opr == OPR_DO_LOOP) {
        g_doloop_stack->Pop();
        g_doloop_wn_stack->Pop();
    }

    /* Reset the kernel context. */
    if (is_kregion) {
        g_curr_kernel_region = NULL;
        g_kernel_data = NULL;
    }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

extern ARRAY_SUMMARY Array_Summary;
extern IVAR *Ivar;

// calls in access_vector.cxx
extern void Initialize_Access_Vals(DU_MANAGER*, FILE*);
extern void Finalize_Access_Vals();


void IPA_HC_collect_kernel_data(IPA_NODE *proc_node)
{ 
    fprintf(stderr, "\n!!! Collecting kernel data for <%s>\n",
            ST_name(proc_node->Func_ST()));

    /* This is for local allocation. Permanent data structure (like kernel
     * DAS) is allocated using IPA node's mempool.
     */
    MEM_POOL_Push(MEM_local_nz_pool_ptr);

    IPA_NODE_CONTEXT context(proc_node);

    BOOL save_IR_dmi = IR_dump_map_info;
    BOOL save_IR_dwa = IR_dump_wn_addr;
    IR_dump_map_info = TRUE;
    IR_dump_wn_addr = TRUE;
    // dump_tree(proc_node->Whirl_Tree());

    /* Obtain the DU chains for the procedure. */

    g_du_mgr = NULL;
    ALIAS_MANAGER *alias_mgr = NULL;
    IPA_get_du_info(proc_node, MEM_local_nz_pool_ptr,
            &g_du_mgr, &alias_mgr);

    // dump_tree(proc_node->Whirl_Tree());

    IR_dump_map_info = save_IR_dmi;
    IR_dump_wn_addr = save_IR_dwa;

    /* Pre-process the procedure. */

    g_inside_kernel_map = WN_MAP_Create(MEM_local_nz_pool_ptr);
    g_proc_node = proc_node;

    g_curr_kernel_region = NULL;
    IPA_HC_kernel_preprocess(proc_node->Whirl_Tree());

    /* Construct data access summary for each kernel region. */

    // We will use ACCESS_VECTOR later, so init LNO from outside.
    Initialize_Access_Vals(g_du_mgr, NULL);

    g_curr_kernel_region = NULL;
    g_process_scalar_only = FALSE;
    g_doloop_stack = CXX_NEW(LOOPINFO_STACK(MEM_local_nz_pool_ptr),
            MEM_local_nz_pool_ptr);
    g_doloop_wn_stack = CXX_NEW(DOLOOP_STACK(MEM_local_nz_pool_ptr),
            MEM_local_nz_pool_ptr);

    // TODO: we can rely on the callsite ID (the i_th call in this procedure)
    // to link WHIRL node with IPA_EDGE.
    IPA_Call_Graph->Map_Callsites(proc_node);

#if 0
    Ipl_Init_From_Ipa(MEM_local_nz_pool_ptr);
    Array_Summary.init_pools();
    INT32 size;
    Ivar = IPA_get_ivar_array(g_proc_node, size);
#endif
    construct_das(proc_node->Whirl_Tree());

    Array_Summary.Finalize();

    Finalize_Access_Vals();

    /* Clean up. */

    WN_MAP_Delete(g_inside_kernel_map);
    g_inside_kernel_map = WN_MAP_UNDEFINED;

    Delete_Du_Manager(g_du_mgr, MEM_local_nz_pool_ptr); g_du_mgr = NULL;
    Delete_Alias_Manager(alias_mgr, MEM_local_nz_pool_ptr);

    MEM_POOL_Pop(MEM_local_nz_pool_ptr);

    g_doloop_stack = NULL;
    g_doloop_wn_stack = NULL;

    /* Debugging output */

    HC_KERNEL_DATA_ITER hkdi(proc_node->get_kernel_data_list());
    for (HC_KERNEL_DATA *hkd = hkdi.First(); !hkdi.Is_Empty();
            hkd = hkdi.Next()) {
        hkd->print(stderr);
    }
}

#endif

void IPA_Call_Graph_print(FILE *fp)
{
    fprintf(fp, "\n%s%s", DBar, SBar);

    IPA_NODE_ITER cg_iter(IPA_Call_Graph, POSTORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL) continue;

        IPA_NODE_CONTEXT context(node);
        fdump_tree(fp, node->Whirl_Tree());
        fprintf(fp, "%s", SBar);
    }

    fprintf(fp, "%s\n", DBar);
}

#if 0

extern ARRAY_SUMMARY Array_Summary;

void HC_match_gpu_data(IPA_NODE *node)
{
    Array_Summary.init_pools();

    HC_KERNEL_DATA_ITER hkdi(node->get_kernel_data_list());
    for (HC_KERNEL_DATA *hkd = hkdi.First(); !hkdi.Is_Empty();
            hkd = hkdi.Next())
    {
        hkd->match_gpu_data_with_das();
    }

    Array_Summary.Finalize();
}

#endif

/*** DAVID CODE END ***/
