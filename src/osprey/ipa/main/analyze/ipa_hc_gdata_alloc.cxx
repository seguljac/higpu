/** DAVID CODE BEGIN **/

// needed by ipl_summary.h included in ipa_cg.h
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "tracing.h"            // for TDEBUG_HICUDA
#include "wn.h"

#include "ipa_cg.h"
#include "ipo_defs.h"           // IPA_NODE_CONTEXT
#include "ipa_hc_gpu_data.h"
#include "ipa_hc_gdata_alloc.h"

#include "hc_gpu_data.h"

extern BOOL flag_opencl;

/*****************************************************************************
 *
 * Determine the largest scalar type size in the given type, which must be
 * either KIND_SCALAR or KIND_STRUCT.  If the given type is KIND_STRUCT, it
 * must consist of KIND_SCALAR fields only.
 *
 ****************************************************************************/

static UINT HC_max_scalar_ty_size(TY_IDX ty_idx)
{
    TY_KIND kind = TY_kind(ty_idx);
    Is_True(kind == KIND_SCALAR || kind == KIND_STRUCT, (""));

    if (kind == KIND_SCALAR) return TY_size(ty_idx);

    UINT max_ty_sz = 0;

    // Go through each struct field.
    FLD_ITER fld_iter = Make_fld_iter(TY_fld(ty_idx));
    do
    {
        FLD_HANDLE fld(fld_iter);

        TY_IDX fld_ty_idx = FLD_type(fld);
        Is_True(TY_kind(fld_ty_idx) == KIND_SCALAR, (""));

        UINT fld_ty_sz = TY_size(fld_ty_idx);
        if (max_ty_sz < fld_ty_sz) max_ty_sz = fld_ty_sz;
    } while (!FLD_last_field(fld_iter++));

    return max_ty_sz;
}

/*****************************************************************************
 *
 * An enhanced stack for statically-allocated HC_GPU_DATA's, with additional
 * functions that operate on each HC_GPU_DATA in the stack (instead of just
 * the top one).
 *
 ****************************************************************************/

typedef HASH_TABLE_ITER<ST_IDX, GPU_DATA_PER_SYMBOL_STACK*> GDATA_STACK_ITER;

class HC_STATIC_DATA_STACK : public HC_GPU_DATA_STACK
{
public:

    HC_STATIC_DATA_STACK(MEM_POOL *pool) : HC_GPU_DATA_STACK(pool) {}

    // For each HC_GPU_DATA, add the given node to its HC_GDATA_IG_NODE,
    // indicating that it is called within the live range of this directive.
    //
    void add_proc_call(IPA_NODE *node);

    // For each HC_GPU_DATA, add an interference edge to the given IC solver
    // between its HC_DATA_IG_NODE and the given one.
    //
    void add_interference_edge(HC_IC_SOLVER *ic_solver,
            HC_GDATA_IG_NODE *gin);
};

void HC_STATIC_DATA_STACK::add_proc_call(IPA_NODE *node)
{
    GDATA_STACK_ITER gs_iter(_stack);
    ST_IDX st_idx;
    GPU_DATA_PER_SYMBOL_STACK* gdata_stack;
    while (gs_iter.Step(&st_idx, &gdata_stack))
    {
        INT n_elems = gdata_stack->Elements();
        for (INT i = 0; i < n_elems; ++i)
        {
            HC_GDATA_IG_NODE *gin = gdata_stack->Bottom_nth(i)->get_ig_node();
            Is_True(gin != NULL, (""));
            gin->add_proc_call(node);
        }
    }
}

void HC_STATIC_DATA_STACK::add_interference_edge(HC_IC_SOLVER *ic_solver,
        HC_GDATA_IG_NODE *gin)
{
    GDATA_STACK_ITER gs_iter(_stack);
    ST_IDX st_idx;
    GPU_DATA_PER_SYMBOL_STACK* gdata_stack;
    while (gs_iter.Step(&st_idx, &gdata_stack))
    {
        INT n_elems = gdata_stack->Elements();
        for (INT i = 0; i < n_elems; ++i)
        {
            HC_GDATA_IG_NODE *other_gin =
                gdata_stack->Bottom_nth(i)->get_ig_node();
            Is_True(other_gin != NULL, (""));
            ic_solver->connect(gin, other_gin);
        }
    }
}

/*****************************************************************************
 *
 * Process CONSTANT/SHARED directives local in the given procedure <node>:
 *
 * - Make sure that each GPU memory variable has a constant size.
 * - Update the max element size in <max_elem_sz>.
 * - Create HC_GDATA_IG_NODE for each HC_GPU_DATA.
 * - Construct local interference graph in <ic_solver>.
 * - Determine calls made in the live range of each HC_GPU_DATA, preparing for
 *   constructing the global interference graph.
 *
 ****************************************************************************/

static void HC_local_process_data_dir_walker(WN *wn, IPA_NODE *node,
        BOOL process_const_dir, UINT& gdata_dir_id,
        HC_STATIC_DATA_STACK *stack,
        HC_IC_SOLVER *ic_solver, UINT *max_elem_sz, MEM_POOL *pool)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        HC_GPU_DATA_LIST *gdata_list;
        WN_PRAGMA_ID copyin_pragma_id, remove_pragma_id;
        HC_GPU_DATA_TYPE gdata_type;
        if (process_const_dir)
        {
            gdata_list = node->get_gpu_data_list();
            copyin_pragma_id = WN_PRAGMA_HC_CONST_COPYIN;
            remove_pragma_id = WN_PRAGMA_HC_CONST_REMOVE;
            gdata_type = HC_CONSTANT_DATA;
        }
        else
        {
            gdata_list = node->get_shared_data_list();
            copyin_pragma_id = WN_PRAGMA_HC_SHARED_COPYIN;
            remove_pragma_id = WN_PRAGMA_HC_SHARED_REMOVE;
            gdata_type = HC_SHARED_DATA;
        }

        HC_GPU_DATA *gdata = NULL;
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        if (pid == WN_PRAGMA_HC_GLOBAL_COPYIN)
        {
            // We do not care about GLOBAL directives, but we have to
            // update <gdata_dir_id>.
            // This will never happen when processing shared directives.
            gdata = (*gdata_list)[gdata_dir_id++];
            Is_True(gdata->get_type() == HC_GLOBAL_DATA, (""));
        }
        else if (pid == copyin_pragma_id)
        {
            gdata = (*gdata_list)[gdata_dir_id++];
            Is_True(gdata->get_type() == gdata_type, (""));

            // Determine the size of the GPU memory variable in bytes.
            WN *size_wn = gdata->compute_size();
            Is_True(WN_operator(size_wn) == OPR_INTCONST,
                    ("The %s memory variable for <%s> in procedure <%s> "
                     "does not have a constant size.",
                     HC_gpu_data_type_name(gdata_type),
                     ST_name(gdata->get_symbol()), node->Name()));
            UINT size = WN_const_val(size_wn);

            // Create the HC_GPU_VAR_INFO. 
            HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();
            Is_True(gvi != NULL, (""));
            gvi->set_size(size_wn);
            // Create the HC_GDATA_IG_NODE and add it to the IC solver.
            HC_GDATA_IG_NODE *gin = CXX_NEW(
                    HC_GDATA_IG_NODE(size, pool), pool);
            gdata->set_ig_node(gin);
            ic_solver->add_node(gin);

            // This HC_GDATA_IG_NODE interferes with every HC_GDATA_IG_NODE in
            // the stack.
            stack->add_interference_edge(ic_solver, gin);

            // Determine the largest scalar element size in this variable for
            // alignment purpose.
            TY_IDX elem_ty_idx = gdata->get_elem_type();
            UINT elem_sz = HC_max_scalar_ty_size(elem_ty_idx);
            if (*max_elem_sz < elem_sz) *max_elem_sz = elem_sz;

            // Push this GPU data onto the stack.
            stack->push(gdata);
        }
        else if (pid == remove_pragma_id)
        {
            // Pop the matching HC_GPU_DATA from the stack.
            ST_IDX st_idx = WN_st_idx(wn);
            gdata = stack->pop(gdata_type, st_idx);
            Is_True(gdata != NULL, (""));
        }
    }
    else if (OPERATOR_is_call(opr))
    {
        // Get the callee procedure if possible.
        WN_TO_EDGE_MAP *wte_map = node->get_wn_to_edge_map();
        Is_True(wte_map != NULL, (""));
        IPA_EDGE *e = wte_map->Find(wn);
        if (e != NULL)
        {
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            Is_True(callee != NULL, (""));
            // Add it to each HC_GPU_DATA in the stack.
            stack->add_proc_call(callee);
        }
    }

    // Handle composite node.
    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            HC_local_process_data_dir_walker(kid_wn, node,
                    process_const_dir, gdata_dir_id,
                    stack, ic_solver, max_elem_sz, pool);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            HC_local_process_data_dir_walker(WN_kid(wn,i), node,
                    process_const_dir, gdata_dir_id,
                    stack, ic_solver, max_elem_sz, pool);
        }
    }
}

/*****************************************************************************
 *
 * If <process_const_dir> is FALSE, this procedure processes SHARED
 * directives.
 *
 ****************************************************************************/

static void IPA_HC_local_process_data_dir(BOOL process_const_dir,
        HC_IC_SOLVER *ic_solver, UINT *max_elem_sz, MEM_POOL *pool)
{
    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL) continue;
        
        if (process_const_dir)
        {
            // The procedure must be K/MK and contain CONSTANT directives.
            if (!node->may_lead_to_kernel()
                    || !node->contains_const_dir()) continue;
        }
        else
        {
            // The procedure must be K/IK and contain SHARED directives.
            if (!node->contains_kernel()
                    && !node->may_be_inside_kernel()) continue;
            if (!node->contains_shared_dir()) continue;
        }

        IPA_NODE_CONTEXT context(node);

        HC_STATIC_DATA_STACK *stack =
            CXX_NEW(HC_STATIC_DATA_STACK(pool), pool);

        UINT gdata_dir_id = 0;
        HC_local_process_data_dir_walker(node->Whirl_Tree(), node,
                process_const_dir, gdata_dir_id, stack,
                ic_solver, max_elem_sz, pool);
        // Sanity check.
        HC_GPU_DATA_LIST *gdata_list = process_const_dir ?
            node->get_gpu_data_list() : node->get_shared_data_list();
        Is_True(gdata_dir_id == gdata_list->Elements(), (""));
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Determine all procedures that could be invoked in each procedure.
 * This propagation happens among ALL procedures.
 *
 ****************************************************************************/

class IPA_HC_CALLEE_PROP_DF : public IPA_DATA_FLOW
{
protected:

    virtual void* Meet(void* in, void* vertex, INT *change);
    virtual void* Trans(void* in, void* out, void* vertex, INT *change);

public:

    IPA_HC_CALLEE_PROP_DF(MEM_POOL *pool)
        : IPA_DATA_FLOW(BACKWARD, pool) {}

    virtual void InitializeNode(void *n);

    void Cleanup();
};

void IPA_HC_CALLEE_PROP_DF::InitializeNode(void *vertex)
{
    IPA_NODE *node = (IPA_NODE*)vertex;

    Is_True(node->get_hc_callee_list() == NULL, (""));

    // Allocate the flag array.
    UINT n_procs = IPA_Call_Graph->Node_Size();
    BOOL *is_proc_called = CXX_NEW_ARRAY(BOOL, n_procs, m);
    for (UINT i = 0; i < n_procs; ++i) is_proc_called[i] = FALSE;

    // Mark procedures directly called within this one.
    IPA_SUCC_ITER succ_iter(node);
    for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
    {
        IPA_EDGE *e = succ_iter.Current_Edge();
        if (e == NULL) continue;

        IPA_NODE *callee = IPA_Call_Graph->Callee(e);
        Is_True(callee != NULL, (""));
        is_proc_called[callee->Array_Index()] = TRUE;    
    }

    node->set_hc_callee_list(is_proc_called);
}

void* IPA_HC_CALLEE_PROP_DF::Meet(void* in, void* vertex, INT *change)
{
    // All the work is done in TRANS operation.
    return NULL;
}

void* IPA_HC_CALLEE_PROP_DF::Trans(void* in, void* out, void* vertex,
        INT *change)
{
    if (vertex == NULL) return NULL;

    IPA_NODE *node = (IPA_NODE*)vertex;

    UINT n_procs = IPA_Call_Graph->Node_Size();
    BOOL *is_proc_called = node->get_hc_callee_list();
    Is_True(is_proc_called != NULL, (""));

    // Iterate through its predecessors, and add to each one any new callees
    // of this procedure.
    IPA_PRED_ITER pred_iter(node);
    for (pred_iter.First(); !pred_iter.Is_Empty(); pred_iter.Next())
    {
        IPA_EDGE *e = pred_iter.Current_Edge();
        if (e == NULL) continue;

        IPA_NODE *caller = IPA_Call_Graph->Caller(e);
        Is_True(caller != NULL, (""));

        BOOL *caller_is_proc_called = caller->get_hc_callee_list();
        Is_True(caller_is_proc_called != NULL, (""));

        for (UINT i = 0; i < n_procs; ++i)
        {
            if (is_proc_called[i] && !caller_is_proc_called[i])
            {
                caller_is_proc_called[i] = TRUE;
                *change = TRUE;
            }
        }
    }

    return NULL;
}

void IPA_HC_CALLEE_PROP_DF::Cleanup()
{
    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL) continue;

        // Manually de-allocate the flag array.
        BOOL *is_proc_called = node->get_hc_callee_list();
        Is_True(is_proc_called != NULL, (""));
        CXX_DELETE_ARRAY(is_proc_called, m);
        node->set_hc_callee_list(NULL);
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Add inter-procedural interference edges to the IC solver.
 * Normalize the weight of each HC_GDATA_IG_NODE.
 *
 ****************************************************************************/

static void HC_global_process_data_dir(IPA_NODE *node,
        BOOL process_const_dir,
        HC_IC_SOLVER *ic_solver, UINT max_elem_sz)
{
    // Used to store all procedures called within the scope of a CONSTANT (or
    // SHARED) directive.
    UINT n_procs = IPA_Call_Graph->Node_Size();
    BOOL *is_proc_called = (BOOL*)alloca(n_procs * sizeof(BOOL));

    // Go through each CONSTANT (or SHARED) directive.
    HC_GPU_DATA_LIST *gdata_list = process_const_dir ?
        node->get_gpu_data_list() : node->get_shared_data_list();
    Is_True(gdata_list != NULL, (""));
    UINT n_gdata = gdata_list->Elements();
    for (UINT i = 0; i < n_gdata; ++i)
    {
        HC_GPU_DATA *gdata = (*gdata_list)[i];
        HC_GPU_DATA_TYPE gdata_type = gdata->get_type();
        if (process_const_dir)
        {
            if (gdata_type != HC_CONSTANT_DATA) continue;
        }
        else
        {
            Is_True(gdata_type == HC_SHARED_DATA, (""));
        }

        HC_GDATA_IG_NODE *gin = gdata->get_ig_node();
        Is_True(gin != NULL, (""));

        // Normalize the GPU memory variable size.
        gin->normalize_size(max_elem_sz);

        // Amalgamate all relevant procedures called within the scope.
        for (UINT j = 0; j < n_procs; ++j) is_proc_called[j] = FALSE;
        UINT n_direct_procs = gin->num_proc_calls();
        for (UINT j = 0; j < n_direct_procs; ++j)
        {
            IPA_NODE *callee = gin->get_proc_call(j);
            if (process_const_dir)
            {
                // For a CONSTANT directive, we only care about
                // K-/MK-procedures.
                if (!callee->may_lead_to_kernel()) continue;
            }
            else
            {
                // For a SHARED directive, we only care about IK procedures.
                if (!callee->may_be_inside_kernel()) continue;
            }

            // First, add this procedure to the list.
            is_proc_called[callee->Array_Index()] = TRUE;

            // Then add all relevant callees of this procedure to the list.
            BOOL *callee_is_proc_called = callee->get_hc_callee_list();
            Is_True(callee_is_proc_called != NULL, (""));
            for (UINT k = 0; k < n_procs; ++k)
            {
                if (!callee_is_proc_called[k]) continue;

                IPA_NODE *callee_callee = IPA_Call_Graph->Node(k);
                if (process_const_dir)
                {
                    // For a CONSTANT directive, we only care about
                    // K-/MK-procedures.
                    if (!callee_callee->may_lead_to_kernel()) continue;
                }
                else
                {
                    // For a SHARED directive, we only care about IK
                    // procedures.
                    if (!callee_callee->may_be_inside_kernel()) continue;
                }

                is_proc_called[k] = TRUE;
            }
        }

        // For each procedure marked above that contains CONSTANT (or SHARED)
        // directives, add an interference edge between the current one with
        // each directive in it.
        for (UINT j = 0; j < n_procs; ++j)
        {
            if (! is_proc_called[j]) continue;

            IPA_NODE *callee = IPA_Call_Graph->Node(j);
            HC_GPU_DATA_LIST *callee_gdata_list;
            if (process_const_dir)
            {
                if (!callee->contains_const_dir()) continue;
                callee_gdata_list = callee->get_gpu_data_list();
            }
            else
            {
                if (!callee->contains_shared_dir()) continue;
                callee_gdata_list = callee->get_shared_data_list();
            }
            Is_True(callee_gdata_list != NULL, (""));

            UINT n_callee_gdata = callee_gdata_list->Elements();
            for (UINT k = 0; k < n_callee_gdata; ++k)
            {
                HC_GPU_DATA *callee_gdata = (*callee_gdata_list)[k];
                HC_GPU_DATA_TYPE callee_gdata_type = callee_gdata->get_type();
                if (process_const_dir)
                {
                    if (callee_gdata_type != HC_CONSTANT_DATA) continue;
                }
                else
                {
                    Is_True(callee_gdata_type == HC_SHARED_DATA, (""));
                }

                HC_GDATA_IG_NODE *callee_gin = callee_gdata->get_ig_node();
                Is_True(callee_gin != NULL, (""));

                ic_solver->connect(gin, callee_gin);
            }
        }
    }
}

static void IPA_HC_global_process_data_dir(BOOL process_const_dir,
        HC_IC_SOLVER *ic_solver, UINT max_elem_sz)
{
    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL) continue;

        if (process_const_dir)
        {
            // The procedure must be K/MK and contain CONSTANT directives.
            if (!node->may_lead_to_kernel()
                    || !node->contains_const_dir()) continue;
        }
        else
        {
            // The procedure must be K/IK and contain SHARED directives.
            if (!node->contains_kernel()
                    && !node->may_be_inside_kernel()) continue;
            if (!node->contains_shared_dir()) continue;
        }

        HC_global_process_data_dir(node, process_const_dir,
                ic_solver, max_elem_sz);
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Go through each kernel region in the given procedure, and interval-color
 * the SHARED directives within this region.
 *
 * Return FALSE if some SHARE directive gets two different offsets.
 *
 ****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Set allocation offset of <cmem> (or <smem>) in each HC_GPU_DATA, based on
 * the result of the given interval-coloring solver. Remove HC_GDATA_IG_NODE
 * from each HC_GPU_DATA.
 *
 ****************************************************************************/

static void IPA_HC_set_alloc_offset(BOOL process_const_dir,
        HC_IC_SOLVER *ic_solver, UINT elem_sz)
{
    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL) continue;
        
        if (process_const_dir)
        {
            // This must be an K-/MK-procedure that contains CONSTANT
            // directives.
            if (!node->may_lead_to_kernel()
                    || !node->contains_const_dir()) continue;
        }
        else
        {
            // This must be an K-/IK-procedure that contains SHARED
            // directives.
            if (!node->contains_shared_dir()) continue;
            if (!node->contains_kernel()
                    && !node->may_be_inside_kernel()) continue;
        }

        IPA_NODE_CONTEXT context(node);

        // Go through each CONSTANT (or SHARED) directive.
        HC_GPU_DATA_LIST *gdata_list = process_const_dir ?
            node->get_gpu_data_list() : node->get_shared_data_list();
        Is_True(gdata_list != NULL, (""));
        UINT n_gdata = gdata_list->Elements();
        for (UINT i = 0; i < n_gdata; ++i)
        {
            HC_GPU_DATA *gdata = (*gdata_list)[i];
            HC_GPU_DATA_TYPE gdata_type = gdata->get_type();
            if (process_const_dir)
            {
                if (gdata_type != HC_CONSTANT_DATA) continue;
            }
            else
            {
                Is_True(gdata_type == HC_SHARED_DATA, (""));
            }

            HC_GDATA_IG_NODE *gin = gdata->get_ig_node();
            Is_True(gin != NULL, (""));
            HC_GPU_VAR_INFO *gvar_info = gdata->get_gvar_info();
            Is_True(gvar_info != NULL, (""));

            // Convert the offset back to bytes.
            INT ofst = ic_solver->get_offset(gin) * elem_sz;
            gvar_info->set_offset(ofst);
            if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
            {
                fprintf(TFile,
                        "Allocation offset of %s memory variable for <%s> "
                        "is %d\n",
                        HC_gpu_data_type_name(gdata_type),
                        ST_name(gdata->get_symbol()), ofst);
            }

            // IMPORTANT!!
            gdata->set_ig_node(NULL);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void IPA_HC_alloc_const_mem()
{
    // We must use a dedicated mempool just in case an existing one conflicts
    // with the pool used to create HC_GPU_VAR_INFO in HC_GPU_DATAs.
    MEM_POOL tmp_pool;
    MEM_POOL_Initialize(&tmp_pool,
            "hiCUDA GPU memory allocation pool", FALSE);

    MEM_POOL_Push(&tmp_pool);
    {
        // Init an interval-coloring solver.
        HC_IC_SOLVER ic_solver(&tmp_pool);

        // We need this to determine the data type for the global <cmem>.
        UINT max_elem_sz = 1;   // starting from 1-byte

        // Create HC_GDATA_IG_NODE in each procedure, and add local
        // interference edges.
        IPA_HC_local_process_data_dir(TRUE,
                &ic_solver, &max_elem_sz, &tmp_pool);

        // Determine all possible callees of each procedure.
        IPA_HC_CALLEE_PROP_DF df(&tmp_pool);
        df.Init();
        df.Solve();

        // Add inter-procedural interference edges to the IC solver.
        // Normalize the weight of each HC_GDATA_IG_NODE.
        IPA_HC_global_process_data_dir(TRUE, &ic_solver, max_elem_sz);

        df.Cleanup();

        // Run the IC solver.
        UINT n_cmem_elems = ic_solver.solve();

        if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
        {
            fprintf(TFile, "cmem: %d x %d-byte\n", n_cmem_elems, max_elem_sz);
        }
        if (n_cmem_elems > 0)
        {
	  if (flag_opencl){ 
	    // Do nothing
	  } else {
            // Allocate a global constant memory variable "cmem".
            hc_glob_var_store.create_cmem_sym(n_cmem_elems, max_elem_sz);
	  }
	  // Save the allocation offset for each constant directive.
	  IPA_HC_set_alloc_offset(TRUE, &ic_solver, max_elem_sz);
        }
    }
    MEM_POOL_Pop(&tmp_pool);

    MEM_POOL_Delete(&tmp_pool);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * For each kernel region in the given procedure, collect all procedures
 * called within it (all along the call chains) and SHARED directives directly
 * within it.
 *
 ****************************************************************************/

static void HC_collect_sdata_walker(WN *wn, IPA_NODE *node,
        UINT& sdata_dir_id, HC_KERNEL_INFO *kinfo)
{
    if (wn == NULL) return;

    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        // This is a kernel region. Init the list to hold all reachable SHARED
        // directives.
        Is_True(kinfo == NULL, (""));
        kinfo = node->get_kernel_info_by_sym(kfunc_st_idx);
        Is_True(kinfo != NULL, (""));
        kinfo->create_callee_proc_list(IPA_Call_Graph->Node_Size());
        kinfo->create_shared_data_list();
    }

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        if (pid == WN_PRAGMA_HC_SHARED_COPYIN)
        {
            // This is a SHARED ALLOC directive (directly inside a kernel
            // region). Add it to the list.
            HC_GPU_DATA *sdata =
                (*node->get_shared_data_list())[sdata_dir_id++];

            Is_True(kinfo != NULL, (""));
            kinfo->get_shared_data_list()->AddElement(sdata);
        }
    }
    else if (OPERATOR_is_call(opr) && kinfo != NULL)
    {
        // Get the callee procedure if possible.
        WN_TO_EDGE_MAP *wte_map = node->get_wn_to_edge_map();
        Is_True(wte_map != NULL, (""));
        IPA_EDGE *e = wte_map->Find(wn);
        if (e != NULL)
        {
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            Is_True(callee != NULL && callee->may_be_inside_kernel(), (""));

            // Add this callee in the list.
            kinfo->set_proc_called(callee->Array_Index());

            // Add all procedures called in <callee> to the list.
            BOOL *callee_is_proc_called = callee->get_hc_callee_list();
            Is_True(callee_is_proc_called != NULL, (""));
            // There should not be any inconsistency in the array length.
            UINT n_procs = IPA_Call_Graph->Node_Size();
            for (UINT i = 0; i < n_procs; ++i)
            {
                if (callee_is_proc_called[i]) kinfo->set_proc_called(i);
            }
        }
    }

    // Handle composite node.
    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            HC_collect_sdata_walker(kid_wn, node, sdata_dir_id, kinfo);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            HC_collect_sdata_walker(WN_kid(wn,i), node, sdata_dir_id, kinfo);
        }
    }
}

/*****************************************************************************
 *
 * Assuming that IPA_HC_CALLEE_PROP_DF has been invoked, collect all SHARED
 * directives reachable from each kernel region and store the list in the
 * corresponding HC_KERNEL_INFO.
 *
 ****************************************************************************/

static void IPA_HC_collect_sdata_per_kernel()
{
    // Go through each K-procedure.
    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL || !node->contains_kernel()) continue;

        IPA_NODE_CONTEXT context(node);

        // Link EDGEs with WNs.
        IPA_Call_Graph->Map_Callsites(node);

        // Collect direct SHARED directives and all callee procedures.
        UINT sdata_dir_id = 0;
        HC_collect_sdata_walker(node->Whirl_Tree(), node, sdata_dir_id, NULL);
        Is_True(sdata_dir_id
                == node->get_shared_data_list()->Elements(), (""));

        // For each kernel region, add indirect SHARED directives (in callee
        // procedures).
        HC_KERNEL_INFO_LIST *kinfo_list = node->get_kernel_info_list();
        UINT n_kregions = kinfo_list->Elements();
        for (UINT i = 0; i < n_kregions; ++i)
        {
            HC_KERNEL_INFO *kinfo = (*kinfo_list)[i];
            HC_GPU_DATA_LIST *sdata_list = kinfo->get_shared_data_list();

            UINT n_procs = IPA_Call_Graph->Node_Size();
            for (UINT p = 0; p < n_procs; ++p)
            {
                if (!kinfo->is_proc_called(p)) continue;

                IPA_NODE *callee = IPA_Call_Graph->Node(p);
                HC_GPU_DATA_LIST *callee_sdata_list =
                    callee->get_shared_data_list();
                // No need for uniqueness check.
                UINT n_sdata = callee_sdata_list->Elements();
                for (UINT s = 0; s < n_sdata; ++s)
                {
                    sdata_list->AddElement((*callee_sdata_list)[s]);
                }
            }
        }
    }
}

/*****************************************************************************
 *
 * Construct a new IC solver on those SHARED directive nodes within the given
 * kernel region. Solve this IC problem nad set smem offsets in each directive
 * in the kernel region. Set the total amount of shared memory needed by this
 * kernel region in HC_KERNEL_INFO.
 *
 * Return TRUE if all offsets are set with no conflict with those previously
 * set by other kernel regions, and FALSE otherwise.
 *
 ****************************************************************************/

static BOOL HC_set_smem_alloc_offset_per_kernel(HC_KERNEL_INFO *kinfo,
        HC_IC_SOLVER *ic_solver, UINT elem_sz)
{
    // Construct a list of HC_GDATA_IG_NODE's.
    HC_GPU_DATA_LIST *sdata_list = kinfo->get_shared_data_list();
    UINT n_sdata = sdata_list->Elements();
    HC_IG_NODE_INFO **sdata_nodes = (HC_IG_NODE_INFO**)
        alloca(n_sdata * sizeof(HC_IG_NODE_INFO*));
    for (UINT i = 0; i < n_sdata; ++i)
    {
        sdata_nodes[i] = (*sdata_list)[i]->get_ig_node();
    }

    // Create a sub interference graph.
    HC_IC_SOLVER *sub_ics = ic_solver->sub_ic_solver(sdata_nodes, n_sdata);

    // Solve the interval-coloring problem.
    INT n_smem_elems = sub_ics->solve();

    // Save the allocation offset for each shared memory variable.
    for (UINT i = 0; i < n_sdata; ++i)
    {
        HC_GPU_DATA *sdata = (*sdata_list)[i];
        HC_GPU_VAR_INFO *gvi = sdata->get_gvar_info();
        Is_True(gvi != NULL, (""));

        // The offset is in bytes.
        INT sdata_ofst = sub_ics->get_offset(sdata->get_ig_node()) * elem_sz;
        INT old_sdata_ofst = gvi->get_offset();
        if (old_sdata_ofst < 0)
        {
            gvi->set_offset(sdata_ofst);
        }
        else if (sdata_ofst != old_sdata_ofst)
        {
            return FALSE;
        }
    }

    // Save the total per-kernel shared memory usage in HC_KERNEL_INFO.
    kinfo->set_smem_size(n_smem_elems * elem_sz);

    return TRUE;
}

static void HC_compute_smem_usage_per_kernel(HC_KERNEL_INFO *kinfo,
        HC_IC_SOLVER *ic_solver, UINT elem_sz)
{
    // Construct a list of HC_GDATA_IG_NODE's.
    HC_GPU_DATA_LIST *sdata_list = kinfo->get_shared_data_list();
    UINT n_sdata = sdata_list->Elements();
    HC_IG_NODE_INFO **sdata_nodes = (HC_IG_NODE_INFO**)
        alloca(n_sdata * sizeof(HC_IG_NODE_INFO*));
    for (UINT i = 0; i < n_sdata; ++i)
    {
        sdata_nodes[i] = (*sdata_list)[i]->get_ig_node();
    }

    INT n = ic_solver->compute_subgraph_offset(sdata_nodes, n_sdata);
    kinfo->set_smem_size(n * elem_sz);
}

static void IPA_HC_clean_up_smem_lists()
{
    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL || !node->contains_kernel()) continue;

        IPA_NODE_CONTEXT context(node);

        // Go through each kernel region in this node.
        HC_KERNEL_INFO_LIST *kinfo_list = node->get_kernel_info_list();
        UINT n_kinfo = kinfo_list->Elements(), i;
        for (i = 0; i < n_kinfo; ++i)
        {
            HC_KERNEL_INFO *kinfo = (*kinfo_list)[i];
            kinfo->reset_callee_proc_list();
            kinfo->reset_shared_data_list();
        }
    }
}


void IPA_HC_alloc_shared_mem()
{
    // We must use a dedicated mempool just in case an existing one conflicts
    // with the pool used to create HC_GPU_VAR_INFO in HC_GPU_DATAs.
    MEM_POOL tmp_pool;
    MEM_POOL_Initialize(&tmp_pool,
            "hiCUDA GPU memory allocation pool", FALSE);

    MEM_POOL_Push(&tmp_pool);
    {
        // Init an interval-coloring solver.
        HC_IC_SOLVER ic_solver(&tmp_pool);

        // We need this to determine the data type for the global <smem>.
        UINT max_elem_sz = 1;   // starting from 1-byte

        // Create HC_GDATA_IG_NODE in each procedure, and add local
        // interference edges.
        IPA_HC_local_process_data_dir(FALSE,
                &ic_solver, &max_elem_sz, &tmp_pool);

        // Determine all possible callees of each procedure.
        IPA_HC_CALLEE_PROP_DF df(&tmp_pool);
        df.Init();
        df.Solve();

        // Add inter-procedural interference edges to the IC solver.
        // Normalize the weight of each HC_GDATA_IG_NODE.
        IPA_HC_global_process_data_dir(FALSE, &ic_solver, max_elem_sz);

        // For each kernel region, determine all reachable SHARED directives
        // and store the list in HC_KERNEL_INFO.
        IPA_HC_collect_sdata_per_kernel();

        df.Cleanup();

        // Do we really need to declare the global <smem> variable?
        BOOL need_smem = FALSE;

        // For each kernel region, solve an interval-coloring problem on the
        // sub-graph of the base interference graph constructed before.
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL || !node->contains_kernel()) continue;

            IPA_NODE_CONTEXT context(node);

            // Go through each kernel region in this node.
            HC_KERNEL_INFO_LIST *kinfo_list = node->get_kernel_info_list();
            UINT n_kinfo = kinfo_list->Elements(), i;
            for (i = 0; i < n_kinfo; ++i)
            {
                HC_KERNEL_INFO *kinfo = (*kinfo_list)[i];
                if (!HC_set_smem_alloc_offset_per_kernel(kinfo,
                            &ic_solver, max_elem_sz)) break;
                if (kinfo->get_smem_size() > 0) need_smem = TRUE;
            }
            if (i < n_kinfo) break;
        }

        if (!cg_iter.Is_Empty())
        {
            // There are conflicts in setting alloc offsets for each kernel.
            //
            // We now go for another solution: solve the entire IC problem and
            // calculate the amount of smem used for each kernel based on that
            // result.
            //
            // This solution may be sub-optimal for some kernels, but
            // guarantees that no cloning is required due to offset conflicts
            // from different kernels, which could be messy.

            // Run the IC solver.
            UINT n_smem_elems = ic_solver.solve();
            if (n_smem_elems > 0)
            {
                need_smem = TRUE;

                // For each kernel region, determine the amount of shared
                // memory needed in bytes.
                IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
                for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
                {
                    IPA_NODE *node = cg_iter.Current();
                    if (node == NULL || !node->contains_kernel()) continue;

                    IPA_NODE_CONTEXT context(node);

                    // Go through each kernel region in this node.
                    HC_KERNEL_INFO_LIST *kinfo_list =
                        node->get_kernel_info_list();
                    UINT n_kinfo = kinfo_list->Elements(), i;
                    for (i = 0; i < n_kinfo; ++i)
                    {
                        HC_KERNEL_INFO *kinfo = (*kinfo_list)[i];
                        HC_compute_smem_usage_per_kernel(kinfo,
                                &ic_solver, max_elem_sz);
                    }
                }

                // Save the allocation offset for each SHARED directive.
                // Also clean up the HC_GDATA_IG_NODE field.
                IPA_HC_set_alloc_offset(FALSE, &ic_solver, max_elem_sz);
            }
        }

        // Clean up the list of callee proc and shared data.
        IPA_HC_clean_up_smem_lists();

        if (need_smem)
        {
	  if (flag_opencl){
	    // Do nothing
	  } else {
            // Allocate an external shared memory variable <smem>.
            hc_glob_var_store.create_smem_sym(max_elem_sz);
	  }
        }
    }
    MEM_POOL_Pop(&tmp_pool);

    MEM_POOL_Delete(&tmp_pool);
}

/*** DAVID CODE END ***/
