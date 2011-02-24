/** DAVID CODE BEGIN **/

#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "tracing.h"                    // for TDEBUG_HICUDA
#include "wn.h"
#include "wn_simp.h"
#include "wn_util.h"

#include "cxx_template.h"
#include "cxx_hash.h"

#include "ipa_cg.h"
#include "ipa_option.h"                 // trace options
#include "ipa_summary.h"
#include "ipa_section_annot.h"

#include "ipa_hc_common.h"
#include "ipa_hc_kernel.h"
#include "ipa_hc_preprocess.h"
#include "ipa_hc_access_redirection.h"

#include "hc_gpu_data.h"
#include "hc_common.h"
#include "hc_utils.h"

#include "ipo_defs.h"
#include "ipo_lwn_util.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

SDATA_TABLE* HC_FORMAL_GPU_VAR_ARRAY::get_sdata_table()
{
    if (_gsdata_map == NULL)
    {
        _gsdata_map = CXX_NEW(SDATA_TABLE(41,_pool), _pool);
    }
    return _gsdata_map;
}

void HC_FORMAL_GPU_VAR_ARRAY::set_sdata_table(SDATA_TABLE *map)
{
    Is_True(map != NULL, (""));
    _gsdata_map = map;
}

HC_GPU_DATA* HC_FORMAL_GPU_VAR_ARRAY::get_sdata_alias(HC_GPU_DATA *gdata) const
{
    return (_gsdata_map == NULL) ? NULL : _gsdata_map->Find(gdata);
}

void HC_FORMAL_GPU_VAR_ARRAY::add_gsdata_alias(
        HC_GPU_DATA *gdata, HC_GPU_DATA *sdata)
{
    if (_gsdata_map == NULL)
    {
        get_sdata_table();
    }
    else
    {
        Is_True(_gsdata_map->Find(gdata) == NULL, (""));
    }
    _gsdata_map->Enter(gdata, sdata);
    _gsdata_map_changed = TRUE;
}

// TODO: to be reused.
HC_GPU_DATA* HC_FORMAL_GPU_VAR_ARRAY::search(WN *func_wn, ST_IDX st_idx) const
{
    // Go through the formals.
    for (UINT i = 0; i < _n_formals; ++i)
    {
        if (st_idx == WN_st_idx(WN_formal(func_wn,i))) {
            return _formal_data[i];
        }
    }

    // Go through the globals.
    return _global_data->Find(st_idx);
}

BOOL HC_FORMAL_GPU_VAR_ARRAY::is_dummy() const
{
    // The annotation is dummy if no formal or global has HC_GPU_DATA.
    for (UINT i = 0; i < _n_formals; ++i) {
        if (_formal_data[i] != NULL) return FALSE;
    }

    return (_global_data->Num_Entries() == 0);
}

BOOL HC_FORMAL_GPU_VAR_ARRAY::equals(const HC_ANNOT_DATA *o) const
{
    if (this == o) return TRUE;
    if (o == NULL) return FALSE;

    HC_FORMAL_GPU_VAR_ARRAY *other = (HC_FORMAL_GPU_VAR_ARRAY*)o;

    // The two annotation are assumed to be for the same procedure.
    Is_True(_n_formals == other->_n_formals, (""));

    // Compare the formals' GPU data.
    for (UINT i = 0; i < _n_formals; ++i)
    {
        HC_GPU_DATA *s1 = _formal_data[i];
        HC_GPU_DATA *s2 = other->_formal_data[i];

        // Here, we just need a very shallow comparison.
        if (s1 != s2) return FALSE;
    }

    // Compare the globals' GPU data.
    ST_IDX st_idx;
    HC_GPU_DATA *gdata = NULL;
    GLOBAL_GPU_DATA_ITER gga_iter(_global_data);
    while (gga_iter.Step(&st_idx, &gdata))
    {
        HC_GPU_DATA *other_gdata = other->_global_data->Find(st_idx);
        if (other_gdata == NULL) return FALSE;
        if (gdata != other_gdata) return FALSE;
    }

    return TRUE;
}

void HC_FORMAL_GPU_VAR_ARRAY::print(FILE *fp) const
{
    fprintf(fp, "(");
    for (UINT i = 0; i < _n_formals; ++i)
    {
        if (_formal_data[i] == NULL) {
            fprintf(fp, "null");
        } else {
            fprintf(fp, "%p", _formal_data[i]);
        }
        fprintf(fp, ", ");
    }
    fprintf(fp, ")");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void HC_build_edge_kernel_annot_walker(WN *wn, IPA_NODE *node,
        WN_TO_EDGE_MAP *wte_map, ST_IDX kfunc_st_idx)
{
    if (wn == NULL) return;

    // Check if it is a kernel region.
    ST_IDX st_idx = HC_get_kernel_sym(wn);
    if (st_idx != ST_IDX_ZERO) kfunc_st_idx = st_idx;

    // Check if it is a call with an outgoing edge.
    OPERATOR opr = WN_operator(wn);
    if (OPERATOR_is_call(opr))
    {
        IPA_EDGE *e = wte_map->Find(wn);
        if (e != NULL) e->set_parent_kernel_sym(kfunc_st_idx);
    }

    // Handle composite node.
    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            HC_build_edge_kernel_annot_walker(kid_wn, node,
                    wte_map, kfunc_st_idx);
            kid_wn = WN_next(kid_wn);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HC_build_edge_kernel_annot_walker(WN_kid(wn,i), node,
                    wte_map, kfunc_st_idx);
        }
    }
}

void HC_build_edge_kernel_annot(IPA_NODE *node, MEM_POOL *tmp_pool)
{
    // Switch to this node's context.
    IPA_NODE_CONTEXT context(node);

    WN *func_wn = node->Whirl_Tree();

    // Link IPA_EDGEs with WN nodes.
    IPA_Call_Graph->Map_Callsites(node);
    WN_TO_EDGE_MAP *wte_map = node->get_wn_to_edge_map();
    
    HC_build_edge_kernel_annot_walker(func_wn, node, wte_map, ST_IDX_ZERO);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IPA_HC_GPU_VAR_PROP_DF::IPA_HC_GPU_VAR_PROP_DF(MEM_POOL *m)
: IPA_DATA_FLOW(FORWARD, m)
{
}

/*****************************************************************************
 *
 * Construct a GPU variable annotation for the given call edge at the caller
 * space (i.e. not mapped to the callee space). The edge does not cache this
 * annotation.
 *
 * ASSUME:
 * 1) in the caller's context
 * 2) the given edge is linked with WN node.
 *
 ****************************************************************************/

HC_FORMAL_GPU_VAR_ARRAY*
IPA_HC_GPU_VAR_PROP_DF::construct_callee_gpu_var_annot(IPA_EDGE *e)
{
    IPA_NODE *callee = IPA_Call_Graph->Callee(e);
    IPA_NODE *caller = IPA_Call_Graph->Caller(e);
    WN *call_wn = e->Whirl_Node();
    Is_True(call_wn != NULL, (""));
    WN *caller_func_wn = caller->Whirl_Tree();

    // Using SUMMARY_CALLSITE here should still be accurate.
    UINT n_params = e->Summary_Callsite()->Get_param_count();
    Is_True(n_params == WN_kid_count(call_wn), (""));
    {
        IPA_NODE_CONTEXT callee_context(callee);
        Is_True(n_params == WN_num_formals(callee->Whirl_Tree()), (""));
    }

    // Create a new annotation.
    HC_FORMAL_GPU_VAR_ARRAY *fgva = CXX_NEW(
            HC_FORMAL_GPU_VAR_ARRAY(n_params,m), m);

    // Get the parent kernel info.
    ST_IDX kfunc_st_idx = e->get_parent_kernel_sym();
    Is_True(kfunc_st_idx != ST_IDX_ZERO, (""));
    HC_KERNEL_INFO *ki = caller->get_kernel_info_by_sym(kfunc_st_idx);
    Is_True(ki != NULL, (""));

    // Create a small table for symbol to be referenced in the annotation.
    GDATA_ST_TABLE *st_table = CXX_NEW(GDATA_ST_TABLE(41,m), m);

    // Go through each actual symbol and see if it matches any symbol in the
    // kernel's DAS that has a corresponding GPU variable.
    for (UINT i = 0; i < n_params; ++i)
    {
        WN *actual_wn = WN_kid0(WN_actual(call_wn,i));

        // Propagate LDA of ARRAY or LDID of pointer-to-ARRAY.
        // NOTE: LDA of scalar does not need to be propagated.
        ST_IDX actual_st_idx = HCWN_verify_pointer_param(actual_wn);
        if (actual_st_idx == ST_IDX_ZERO) continue;

        // Use the call's WN node and the actual index to search for array
        // section info.
#if 0
        HC_GPU_DATA *gdata = ki->find_gdata_for_arr_region(actual_st_idx,
                call_wn, i);
#else
        HC_GPU_DATA *gdata = ki->find_gdata_for_arr_region(actual_st_idx);
#endif
        Is_True(gdata != NULL, (""));

        // Add ST_IDX <-> ST* referenced in this HC_GPU_DATA in the table.
        gdata->build_st_table(st_table);

        fgva->set_formal_data(i, gdata);
    }

    // Get the global variable access summary of the callee.
    IPA_NODE_SECTION_INFO *callee_info = callee->Section_Annot();
    ST_IDX st_idx;
    GLOBAL_ARRAY_LIST *gal = NULL;
    GLOBAL_ARRAY_TABLE_ITER gat_iter(callee_info->Global_Array_Table());
    while (gat_iter.Step(&st_idx, &gal))
    {
        // TODO: handle messy region
        if (gal->Is_messy()) {
            DevWarn("messy region is not supported yet.");
            continue;
        }

        // Walk through elements in the common block.
        GLOBAL_ARRAY_LIST_ITER gal_iter(gal);
        for (gal_iter.First(); !gal_iter.Is_Empty(); gal_iter.Next())
        {
            GLOBAL_ARRAY_INFO *gai = gal_iter.Cur();
            STATE *state = gai->Get_state();
            ST_IDX glob_st_idx = gai->St_Idx();

            HC_GPU_DATA *gdata = NULL;
            if (HCST_is_scalar(glob_st_idx))
            {
                // scalar access
                Is_True(state->Is_scalar(), (""));
                // TODO: offset?
                gdata = ki->find_gdata_for_scalar(glob_st_idx, 0);

                // FIXME: <gdata> could be NULL, when the scalar variable is
                // passed as a kernel parameter.
            }
            else
            {
                // array section access (cannot use Is_scalar() to verify)
#if 0
                gdata = ki->find_gdata_for_arr_region(glob_st_idx,
                        call_wn, -1);
#else
                gdata = ki->find_gdata_for_arr_region(glob_st_idx);
#endif
            }
            Is_True(gdata != NULL, (""));

            gdata->build_st_table(st_table);
            fgva->set_global_data(glob_st_idx, gdata);
        }
    }

    fgva->set_st_table(st_table);

    return fgva;
}

/*****************************************************************************
 *
 * Nodes involved in this propagation (i.e. have annotation list) are K- and
 * IK-procedures. Predecessors of IK-procedures may not be involved in this
 * propagation, but this case will be handled during cloning and edge fixing.
 *
 * The procedure only works on K-procedures and initializes IK-procedures that
 * are direct successors of K-procedures.
 *
 ****************************************************************************/

void IPA_HC_GPU_VAR_PROP_DF::InitializeNode(void *vertex)
{
    IPA_NODE *node = (IPA_NODE*)vertex;

    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    if (! node->may_be_inside_kernel()) Is_True(annots == NULL, (""));

    // Only works on IK- or K-procedures.
    if (!node->contains_kernel() && !node->may_be_inside_kernel()) return;

    // Create a GPU variable annotation list for this node (if not done).
    if (annots == NULL)
    {
        annots = CXX_NEW(IPA_HC_ANNOT_LIST(m), m);
        node->set_hc_annots(annots);
    }

    // All actual init work is done from K-procedures.
    if (node->may_be_inside_kernel()) return;

    // A K-procedure has only a dummy annotation.
    annots->add_dummy();
    IPA_CALL_CONTEXT *cc = annots->Head()->get_call_context();

    // Switch to this node's context.
    IPA_NODE_CONTEXT context(node);

    // Link IPA_EDGEs with WN nodes.
    IPA_Call_Graph->Map_Callsites(node);

    // For each outgoing edge to a IK-procedure (i.e. inside a kernel region),
    // construct the GPU variable annotation in the callee.
    IPA_SUCC_ITER succ_iter(node);
    for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
    {
        IPA_EDGE *e = succ_iter.Current_Edge();
        if (e == NULL) continue;

        // Does this edge come from (within) a kernel region?
        //
        // NOTE: we cannot answer this question by checking if the callee is
        // an IK-procedure because an IK-procedure (at this stage) can still
        // be called outside kernel regions.
        //
        if (e->get_parent_kernel_sym() == ST_IDX_ZERO) continue;

        // Sanity check: the callee must be an IK-procedure.
        IPA_NODE *callee = IPA_Call_Graph->Callee(e);
        Is_True(callee->may_be_inside_kernel(), (""));

        // Create an annotation list in the callee (if not done before).
        IPA_HC_ANNOT_LIST *callee_annots = callee->get_hc_annots();
        if (callee_annots == NULL)
        {
            callee_annots = CXX_NEW(IPA_HC_ANNOT_LIST(m), m);
            callee->set_hc_annots(callee_annots);
        }

        // Construct and add an annotation to the callee.
        callee_annots->add(e, cc, construct_callee_gpu_var_annot(e));
    }
}

void* IPA_HC_GPU_VAR_PROP_DF::Meet(void* in, void* vertex, INT *change)
{
    return NULL;
}

/*****************************************************************************
 *
 * Propagate the given GPU variable annotation in the caller through the given
 * call edge. Return the new annotation for the callee.
 *
 * ASSUME:
 * 1) in the caller's context
 * 2) the given edge is linked with WN node.
 *
 ****************************************************************************/

HC_FORMAL_GPU_VAR_ARRAY* IPA_HC_GPU_VAR_PROP_DF::propagate_gpu_var_annot(
        HC_FORMAL_GPU_VAR_ARRAY *caller_fgva, IPA_EDGE *e)
{
    IPA_NODE *caller = IPA_Call_Graph->Caller(e);
    IPA_NODE *callee = IPA_Call_Graph->Callee(e);
    WN *call_wn = e->Whirl_Node();
    Is_True(call_wn != NULL, (""));
    WN *caller_func_wn = caller->Whirl_Tree();

    // Using SUMMARY_CALLSITE here should still be accurate.
    UINT n_params = e->Summary_Callsite()->Get_param_count();
    Is_True(n_params == WN_kid_count(call_wn), (""));
    {
        IPA_NODE_CONTEXT callee_context(callee);
        Is_True(n_params == WN_num_formals(callee->Whirl_Tree()), (""));
    }

    // Create the annotation for the callee.
    HC_FORMAL_GPU_VAR_ARRAY *callee_fgva = CXX_NEW(
            HC_FORMAL_GPU_VAR_ARRAY(n_params,m), m);

    UINT n_caller_formals = WN_num_formals(caller_func_wn);

    // Go through each actual symbol and see if it matches any symbol in the
    // kernel's DAS that has a corresponding GPU variable.
    for (UINT i = 0; i < n_params; ++i)
    {
        // Propagate LDA of ARRAY or LDID of pointer-to-ARRAY.
        // NOTE: LDA of scalar does not need to be propagated.
        WN *param_wn = WN_actual(call_wn,i);
        if (WN_rtype(param_wn) != Pointer_type) continue;

        // If this parameter contains a symbol that has corresponding GPU
        // variable, it must be an LDA of ARRAY or LDID of pointer-to-ARRAY.
        // TODO: do this validation
        WN *actual_wn = WN_kid0(param_wn);
        ST_IDX actual_st_idx = HCWN_verify_pointer_param(actual_wn);
        if (actual_st_idx == ST_IDX_ZERO) continue;

        // Go through the caller's formals to see if it has a corresponding
        // HC_GPU_DATA.
        HC_GPU_DATA *gdata = NULL;
        for (UINT j = 0; j < n_caller_formals; ++j)
        {
            if (actual_st_idx == WN_st_idx(WN_formal(caller_func_wn,j))) {
                gdata = caller_fgva->get_formal_data(j);
                break;
            }
        }
        if (gdata == NULL) {
            // Go through the caller's globals.
            gdata = caller_fgva->get_global_data_table()->Find(actual_st_idx);
        }

        // No need to map to the callee space.
        if (gdata != NULL) callee_fgva->set_formal_data(i, gdata);
    }

    // Get the global variable access summary of the callee.
    IPA_NODE_SECTION_INFO *callee_info = callee->Section_Annot();
    ST_IDX st_idx;
    GLOBAL_ARRAY_LIST *gal = NULL;
    GLOBAL_ARRAY_TABLE_ITER gat_iter(callee_info->Global_Array_Table());
    while (gat_iter.Step(&st_idx, &gal))
    {
        // TODO: handle messy region
        if (gal->Is_messy()) {
            DevWarn("messy region is not supported yet.");
            continue;
        }

        // Walk through elements in the common block.
        GLOBAL_ARRAY_LIST_ITER gal_iter(gal);
        for (gal_iter.First(); !gal_iter.Is_Empty(); gal_iter.Next())
        {
            GLOBAL_ARRAY_INFO *gai = gal_iter.Cur();
            ST_IDX glob_st_idx = gai->St_Idx();

            // Go through the caller's globals, which must exist.
            HC_GPU_DATA *gdata =
                caller_fgva->get_global_data_table()->Find(glob_st_idx);
            Is_True(gdata != NULL, (""));

            callee_fgva->set_global_data(glob_st_idx, gdata);
        }
    }

    // Inherit the symbol table.
    callee_fgva->set_st_table(caller_fgva->get_st_table());

    return callee_fgva;
}

void* IPA_HC_GPU_VAR_PROP_DF::Trans(void* in, void* out, void* vertex,
        INT *change)
{
    if (vertex == NULL) return NULL;

    IPA_NODE *node = (IPA_NODE*)vertex;

    // Only work on IK-procedures.
    if (! node->may_be_inside_kernel()) return NULL;

    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));

    // Switch to this node's context.
    IPA_NODE_CONTEXT nc(node);

    // Link IPA_EDGEs with WN nodes (needed in annotation propagation).
    IPA_Call_Graph->Map_Callsites(node);

    // For each NEW GPU variable annotation in this node, iterate through each
    // outgoing edge and propagate the annotation to the callee.
    IPA_HC_ANNOT_ITER annot_iter(annots);
    for (IPA_HC_ANNOT *annot = annot_iter.First(); !annot_iter.Is_Empty();
            annot = annot_iter.Next())
    {
        if (annot->is_processed()) continue;

        HC_FORMAL_GPU_VAR_ARRAY *fgva =
            (HC_FORMAL_GPU_VAR_ARRAY*)annot->get_annot_data();
        // TODO: <fgva> could be NULL.

        IPA_SUCC_ITER succ_iter(node);
        for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
        {
            IPA_EDGE *e = succ_iter.Current_Edge();
            // Somehow this could be NULL.
            if (e == NULL) continue;

            // The callee must be an IK-procedure.
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            Is_True(callee->may_be_inside_kernel(), (""));
            IPA_HC_ANNOT_LIST *callee_annots = callee->get_hc_annots();
            Is_True(callee_annots != NULL, (""));

            HC_FORMAL_GPU_VAR_ARRAY *callee_fgva =
                propagate_gpu_var_annot(fgva, e);
            IPA_CALL_CONTEXT *caller_cc = annot->get_call_context();
            if (callee_annots->add(e, caller_cc, callee_fgva)) *change = TRUE;
        }

        annot->set_processed();
    }

    return NULL;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * For now, we do not perform cloning when propagating SHARED data among K-
 * and IK-procedures. This allows us to embed this propagation as a part of
 * the GPU variable propagation and only needs to extend
 * HC_FORMAL_GPU_VAR_ARRAY so that it caches a mapping from GLOBAL data to
 * SHARED data.
 *
 ****************************************************************************/

class IPA_HC_SVAR_PROP_DF : public IPA_DATA_FLOW
{
protected:

    virtual void* Meet(void* in, void* vertex, INT *change);
    virtual void* Trans(void* in, void* out, void* vertex, INT *change);

public:

    IPA_HC_SVAR_PROP_DF(MEM_POOL *pool);

    virtual void InitializeNode(void *n);
    virtual void Print_entry(FILE *fp, void* out, void* n);
    virtual void PostProcessIO(void *);
};

IPA_HC_SVAR_PROP_DF::IPA_HC_SVAR_PROP_DF(MEM_POOL *m)
: IPA_DATA_FLOW(FORWARD, m)
{
}

// <callee> must be an IK-procedure.
//
static void HC_init_prop_svar_to_callee(IPA_NODE *callee,
        HC_GPU_DATA_STACK *sdata_stack, MEM_POOL *tmp_pool)
{
    IPA_HC_ANNOT *callee_annot = callee->get_hc_annots()->Head();
    if (callee_annot == NULL || callee_annot->is_dummy()) return;

    HC_FORMAL_GPU_VAR_ARRAY *fgva = (HC_FORMAL_GPU_VAR_ARRAY*)
        callee_annot->get_annot_data();
    Is_True(fgva != NULL, (""));

    IPA_NODE_CONTEXT callee_nc(callee);
    WN *callee_wn = callee->Whirl_Tree();

    // Go through the visible SHARED data.
    HC_VISIBLE_GPU_DATA_ITER v_sdata_iter(sdata_stack->top(tmp_pool));
    ST_IDX st_idx;
    HC_GPU_DATA *sdata;
    while (v_sdata_iter.Step(&st_idx, &sdata))
    {
        // Search for the GLOBAL data first.
        HC_GPU_DATA *gdata = fgva->search(callee_wn, st_idx);
        if (gdata != NULL)
        {
            // Search for the existing SHARED data in the annotation.
            HC_GPU_DATA *old_sdata = fgva->get_sdata_alias(gdata);
            if (old_sdata != sdata)
            {
                // NOTE: cannot print out the symbol name.
                HC_assert(old_sdata == NULL,
                        ("Multiple SHARED directives for the same variable "
                         "reach procedure <%s>!", callee->Name()));

                fgva->add_gsdata_alias(gdata, sdata);

                // Cache the symbol table.
                sdata->build_st_table(fgva->get_st_table());
            }
        }
    }
}

static void HC_init_prop_svar_walker(WN *wn, IPA_NODE *node,
        UINT& shared_dir_id,
        HC_GPU_DATA_STACK *sdata_stack, MEM_POOL *tmp_pool)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        HC_GPU_DATA *sdata;
        if (pid == WN_PRAGMA_HC_SHARED_COPYIN)
        {
            sdata = (*node->get_shared_data_list())[shared_dir_id++];
            Is_True(sdata != NULL, (""));

            // Here, we forbid SHARED directives (for the same host variable)
            // to be nested.
            HC_assert(sdata_stack->peek(sdata->get_symbol()) == NULL,
                    ("SHARED directives for <%s> are nested in "
                     "procedure <%s>!",
                     ST_name(sdata->get_symbol()), node->Name()));

            // Push it onto the stack.
            sdata_stack->push(sdata);
        }
        else if (pid == WN_PRAGMA_HC_SHARED_REMOVE)
        {
            // Retrieve the matching HC_GPU_DATA from the stack.
            ST_IDX st_idx = WN_st_idx(wn);
            sdata = sdata_stack->pop(HC_SHARED_DATA, st_idx);
            Is_True(sdata != NULL, (""));
        }
    }
    else if (OPERATOR_is_call(opr))
    {
        IPA_EDGE *e = node->get_wn_to_edge_map()->Find(wn);
        if (e != NULL)
        {
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            if (callee->may_be_inside_kernel())
            {
                HC_init_prop_svar_to_callee(callee, sdata_stack, tmp_pool);
            }
        }
    }

    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            HC_init_prop_svar_walker(kid_wn, node,
                    shared_dir_id, sdata_stack, tmp_pool);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            HC_init_prop_svar_walker(WN_kid(wn,i), node,
                    shared_dir_id, sdata_stack, tmp_pool);
        }
    }
}

void IPA_HC_SVAR_PROP_DF::InitializeNode(void *vertex)
{
    IPA_NODE *node = (IPA_NODE*)vertex;

    // Only works on IK- or K-procedures.
    if (!node->contains_kernel() && !node->may_be_inside_kernel()) return;

    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));

    // Do nothing if this node has no SHARED directives.
    if (!node->contains_shared_dir()) return;

    // Switch to this node's context.
    IPA_NODE_CONTEXT context(node);

    // Link IPA_EDGEs with WN nodes.
    IPA_Call_Graph->Map_Callsites(node);

    // Start a empty stack of SHARED data.
    HC_GPU_DATA_STACK *sdata_stack = CXX_NEW(HC_GPU_DATA_STACK(m), m);

    UINT shared_dir_id = 0;
    HC_init_prop_svar_walker(node->Whirl_Tree(), node,
            shared_dir_id, sdata_stack, m);
    Is_True(shared_dir_id == node->get_shared_data_list()->Elements(), (""));
}

void* IPA_HC_SVAR_PROP_DF::Meet(void* in, void* vertex, INT *change)
{
    return NULL;
}

void* IPA_HC_SVAR_PROP_DF::Trans(void* in, void* out, void* vertex,
        INT *change)
{
    if (vertex == NULL) return NULL;

    IPA_NODE *node = (IPA_NODE*)vertex;

    // Only work on IK-procedures.
    if (! node->may_be_inside_kernel()) return NULL;

    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));
    IPA_HC_ANNOT *annot = annots->Head();
    if (annot == NULL || annot->is_dummy()) return NULL;
    HC_FORMAL_GPU_VAR_ARRAY *fgva =
        (HC_FORMAL_GPU_VAR_ARRAY*)annot->get_annot_data();
    Is_True(fgva != NULL, (""));

    // Switch to this node's context.
    IPA_NODE_CONTEXT nc(node);

    // Link IPA_EDGEs with WN nodes (needed in annotation propagation).
    IPA_Call_Graph->Map_Callsites(node);

    // We do not keep track what SHARED data are new, so let's try everything
    // again.
    SDATA_TABLE_ITER s_tbl_iter(fgva->get_sdata_table());
    HC_GPU_DATA *gdata, *sdata;
    while (s_tbl_iter.Step(&gdata, &sdata))
    {
        ST_IDX st_idx = sdata->get_symbol();

        // Go through the outgoing edges.
        IPA_SUCC_ITER succ_iter(node);
        for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
        {
            IPA_EDGE *e = succ_iter.Current_Edge();
            // Somehow this could be NULL.
            if (e == NULL) continue;

            // The callee must be an IK-procedure.
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            Is_True(callee->may_be_inside_kernel(), (""));
            IPA_HC_ANNOT_LIST *callee_annots = callee->get_hc_annots();
            Is_True(callee_annots != NULL, (""));

            IPA_HC_ANNOT *callee_annot = callee_annots->Head();
            if (callee_annot == NULL || callee_annot->is_dummy()) continue;

            HC_FORMAL_GPU_VAR_ARRAY *callee_fgva = 
                (HC_FORMAL_GPU_VAR_ARRAY*)callee_annot->get_annot_data();
            Is_True(callee_fgva != NULL, (""));

            // Switch to callee space.
            IPA_NODE_CONTEXT callee_nc(callee);
            WN *callee_wn = callee->Whirl_Tree();

            // Search for the GLOBAL data first.
            HC_GPU_DATA *callee_gdata =
                callee_fgva->search(callee_wn, st_idx);
            if (callee_gdata != NULL)
            {
                // Search for the existing SHARED data in the annotation.
                HC_GPU_DATA *old_callee_sdata =
                    callee_fgva->get_sdata_alias(callee_gdata);
                if (old_callee_sdata != sdata)
                {
                    // NOTE: cannot print out the symbol name.
                    HC_assert(old_callee_sdata == NULL,
                            ("Multiple SHARED directives for the same "
                             "variable reach procedure <%s>!",
                             callee->Name()));

                    callee_fgva->add_gsdata_alias(callee_gdata, sdata);
                }
            }

            // Here, we do not need to update callee's st_table as it aliases
            // with the caller's.
            Is_True(fgva->get_st_table() == callee_fgva->get_st_table(), (""));
        }
    }

    // IMPORTANT!
    fgva->reset_gsdata_map_changed();

    return NULL;
}

void IPA_HC_SVAR_PROP_DF::Print_entry(FILE *fp, void *, void *vertex)
{
}

void IPA_HC_SVAR_PROP_DF::PostProcessIO(void *vertex)
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// For a GLOBAL or SHARED directive, the GPU memory variable is a formal. For
// a CONSTANT directive, it is a local variable. Although we do not know if it
// is necessary until we figure out whether or not references are made to it,
// it does not hurt to create it now.
//
// Called for both formals and global variables accessed.
//
static void HC_create_local_gvar(ST_IDX var_st_idx, HC_GPU_DATA *gdata,
        GDATA_ST_TABLE *st_table)
{
    // The type of the local GPU memory variable should be the same as that of
    // the one created in the original procedure, where the corresponding
    // directive is declared.
    HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();
    Is_True(gvi != NULL, (""));
    ST_IDX orig_gvar_st_idx = gvi->get_symbol();
    Is_True(orig_gvar_st_idx != ST_IDX_ZERO, (""));
    ST *orig_gvar_st = st_table->Find(orig_gvar_st_idx);
    Is_True(orig_gvar_st != NULL, (""));
    TY_IDX gvar_ty_idx = ST_type(orig_gvar_st);

    // The name follows the given host variable with the usual prefix.
    ST_IDX gvar_st_idx;
    switch (gdata->get_type())
    {
        case HC_GLOBAL_DATA:
        {
            STR_IDX gvar_str_idx = gen_var_str("g_", var_st_idx);
            gvar_st_idx = new_formal_var(gvar_str_idx, gvar_ty_idx);
            break;
        }

        case HC_SHARED_DATA:
        {
            STR_IDX gvar_str_idx = gen_var_str("s_", var_st_idx);
            gvar_st_idx = new_formal_var(gvar_str_idx, gvar_ty_idx);
            break;
        }

        case HC_CONSTANT_DATA:
        {
            STR_IDX gvar_str_idx = gen_var_str("c_", var_st_idx);
            gvar_st_idx = new_local_var(gvar_str_idx, gvar_ty_idx);
        }
    }

    // Store it in HC_GPU_VAR_INFO.
    gvi->set_symbol(gvar_st_idx);
}

void IPA_HC_GPU_VAR_PROP_DF::map_gpu_var_annot_to_callee(IPA_NODE *callee)
{
    // Get the GPU variable annotation.
    IPA_HC_ANNOT_LIST *annots = callee->get_hc_annots();
    Is_True(annots != NULL, (""));
    IPA_HC_ANNOT *annot = annots->Head();
    if (annot == NULL) return;  // unreachable
    HC_FORMAL_GPU_VAR_ARRAY *fgva =
        (HC_FORMAL_GPU_VAR_ARRAY*)annot->get_annot_data();
    if (fgva == NULL) return;   // dummy annotation

    IPA_NODE_CONTEXT context(callee);
    WN *callee_wn = callee->Whirl_Tree();

    // Map HC_GPU_DATA of each formal and global to the callee space, and
    // create a list of extra formal/actual pairs we need.
    //
    // TODO: reuse existing formal/actual pairs, but may be disturbed by the
    // next step.
    
    // Step 1: for each auxilliary variable (or index variables used in
    // bounds) in each HC_GPU_DATA, add a new formal and record the
    // correspondence.
    //
    HC_SYM_MAP *sym_map = CXX_NEW(HC_SYM_MAP(41,m), m);
    fgva->set_idxv_sym_map(sym_map);

    // Get the ST_IDX to ST* map because the symbols are in the original
    // procedure's context.
    GDATA_ST_TABLE *st_table = fgva->get_st_table();
    Is_True(st_table != NULL, (""));

    SDATA_TABLE *new_sdata_tbl = CXX_NEW(SDATA_TABLE(41,m), m);

    // Go through each formal.
    UINT n_formals = fgva->num_formals();
    for (UINT i = 0; i < n_formals; ++i)
    {
        HC_GPU_DATA *gdata = fgva->get_formal_data(i);
        if (gdata == NULL) continue;
        HC_GPU_DATA *sdata = fgva->get_sdata_alias(gdata);

        // IMPORTANT: create a deep copy first.
        gdata = CXX_NEW(HC_GPU_DATA(gdata,m), m);
        fgva->set_formal_data(i, gdata);
        gdata->replace_idxvs_with_formals(callee, sym_map, st_table);

        if (sdata == NULL) continue;
        sdata = CXX_NEW(HC_GPU_DATA(sdata,m), m);
        sdata->replace_idxvs_with_formals(callee, sym_map, st_table);

        new_sdata_tbl->Enter(gdata, sdata);
    }

    // Go through each global.
    GLOBAL_GPU_DATA_TABLE *gat = fgva->get_global_data_table();
    Is_True(gat != NULL, (""));
    ST_IDX st_idx;
    HC_GPU_DATA *gdata;
    // Create a temp array for storing new HC_GPU_DATA.
    HC_GPU_DATA **new_gat = (HC_GPU_DATA**)alloca(
            gat->Num_Entries() * sizeof(HC_GPU_DATA*));
    UINT gdata_idx = 0;
    {
        GLOBAL_GPU_DATA_ITER gdata_iter(gat);
        while (gdata_iter.Step(&st_idx, &gdata))
        {
            HC_GPU_DATA *sdata = fgva->get_sdata_alias(gdata);

            // Create a deep copy first.
            gdata = CXX_NEW(HC_GPU_DATA(gdata,m), m);
            new_gat[gdata_idx++] = gdata;
            gdata->replace_idxvs_with_formals(callee, sym_map, st_table);

            if (sdata == NULL) continue;
            sdata = CXX_NEW(HC_GPU_DATA(sdata,m), m);
            sdata->replace_idxvs_with_formals(callee, sym_map, st_table);

            new_sdata_tbl->Enter(gdata, sdata);
        }
    }
    for (UINT i = 0; i < gdata_idx; ++i)
    {
        gdata = new_gat[i];
        st_idx = gdata->get_symbol();
        gat->Remove(st_idx);
        gat->Enter(st_idx, gdata);
    }

    // Store the new SHARED data list.
    fgva->set_sdata_table(new_sdata_tbl);

    // Step 2: create the main GPU variable for each HC_GPU_DATA and store it
    // in HC_GPU_VAR_INFO, which is a new instance for every mapped
    // HC_GPU_DATA.
    //
    // For a GLOBAL or SHARED directive, this variable is a formal. For a
    // constant memory variable, it is a local variable. Although it is not
    // necessary to create it here because it will be useless if no reference
    // is made, it does not hurt to create it now.

    // Go through each formal.
    for (UINT i = 0; i < n_formals; ++i)
    {
        HC_GPU_DATA *gdata = fgva->get_formal_data(i);
        if (gdata == NULL) continue;

        // Get the formal symbol.
        ST_IDX formal_st_idx = WN_st_idx(WN_formal(callee_wn,i));

        HC_create_local_gvar(formal_st_idx, gdata, st_table);

        HC_GPU_DATA *sdata = fgva->get_sdata_alias(gdata);
        if (sdata != NULL)
        {
            HC_create_local_gvar(formal_st_idx, sdata, st_table);
        }
    }

    // Go through each global.
    {
        GLOBAL_GPU_DATA_ITER gdata_iter(gat);
        ST_IDX st_idx;
        HC_GPU_DATA *gdata;
        while (gdata_iter.Step(&st_idx, &gdata))
        {
            HC_create_local_gvar(st_idx, gdata, st_table);

            HC_GPU_DATA *sdata = fgva->get_sdata_alias(gdata);
            if (sdata != NULL) 
            {
                HC_create_local_gvar(st_idx, sdata, st_table);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Replace ILOAD and ISTORE nodes in an IK-procedure.
 * Link local SHARED directives with corresponding GLOBAL data.
 *
 * Return a WN node to replace <wn>.
 *
 * NOTE: need local context of <node>.
 *
 ****************************************************************************/

static WN* HC_replace_array_access_walker(WN *wn,
        IPA_NODE *node, HC_FORMAL_GPU_VAR_ARRAY *fgva,
        UINT& shared_dir_id, HC_GPU_DATA_STACK *sdata_stack)
{
    if (wn == NULL) return NULL;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        HC_GPU_DATA *sdata;
        if (pid == WN_PRAGMA_HC_SHARED_COPYIN)
        {
            sdata = (*node->get_shared_data_list())[shared_dir_id++];
            Is_True(sdata != NULL, (""));

            // Push it onto the stack.
            sdata_stack->push(sdata);

            // Search for the corresponding GLOBAL data in the annotation.
            ST_IDX st_idx = sdata->get_symbol();
            HC_GPU_DATA *gdata = fgva->search(node->Whirl_Tree(), st_idx);
            HC_assert(gdata != NULL,
                    ("The SHARED directive for <%s> in procedure <%s> "
                    "does not have a corresponding GLOBAL directive!",
                    ST_name(st_idx), node->Name()));
            sdata->set_partner_gdata(gdata);
        }
        else if (pid == WN_PRAGMA_HC_SHARED_REMOVE)
        {
            // Retrieve the matching HC_GPU_DATA from the stack.
            ST_IDX st_idx = WN_st_idx(wn);
            sdata = sdata_stack->pop(HC_SHARED_DATA, st_idx);
            Is_True(sdata != NULL, (""));
        }

        // We must not further process hiCUDA directives as it may affect the
        // translation of SHARED directives.
        if (WN_pragma_hicuda(wn)) return NULL;
    }

    // IMPORTANT: handle the kids first.
    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            WN *next_wn = WN_next(kid_wn);
            WN *new_wn = HC_replace_array_access_walker(kid_wn,
                    node, fgva, shared_dir_id, sdata_stack);
            if (new_wn != NULL)
            {
                // Insert it before the current kid.
                WN_INSERT_BlockBefore(wn, kid_wn, new_wn);
                WN_DELETE_FromBlock(wn, kid_wn);
            }
            kid_wn = next_wn;
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            WN *kid_wn = WN_kid(wn,i);
            WN *new_wn = HC_replace_array_access_walker(kid_wn,
                    node, fgva, shared_dir_id, sdata_stack);
            if (new_wn != NULL)
            {
                // Replace the kid with the new one.
                WN_DELETE_Tree(kid_wn);
                WN_kid(wn,i) = new_wn;
            }
        }
    }

    if (opr == OPR_ILOAD || opr == OPR_ISTORE)
    {
        WN *addr_wn = (opr == OPR_ILOAD) ? WN_kid0(wn) : WN_kid1(wn);

        if (WN_operator(addr_wn) == OPR_ARRAY)
        {
            // Find the corresponding HC_GPU_DATA.
            ST_IDX st_idx = WN_st_idx(WN_array_base(addr_wn));

            // TODO: nesting detection

            HC_GPU_DATA *gdata, *sdata;

            // First, search the SHARED data stack.
            sdata = sdata_stack->peek(st_idx);
            if (sdata == NULL)
            {
                // Second, search the annotation for SHARED data.
                gdata = fgva->search(node->Whirl_Tree(), st_idx);
                if (gdata != NULL) sdata = fgva->get_sdata_alias(gdata);
            }

            // The SHARED data take precedence over GLOBAL data.
            if (sdata != NULL) gdata = sdata;

            if (gdata != NULL)
            {
                Is_True(gdata->is_arr_section(), (""));
                WN *new_addr_wn =
                    HC_create_gvar_access_for_array(addr_wn, gdata);
                WN_kid(wn, opr == OPR_ILOAD ? 0 : 1) = new_addr_wn;
                WN_DELETE_Tree(addr_wn); addr_wn = NULL;
            }
        }
        else
        {
            // TODO: handle general indirect accesses.
        }
    }

    return NULL;
}

/*****************************************************************************
 *
 * Replace ILOAD and ISTORE nodes in a K-procedure.
 * Link local SHARED directives with corresponding GLOBAL data.
 *
 * Return a WN node to replace <wn>.
 *
 * NOTE: we must put <kinfo> parameter before <node> in order to distinguish
 * from the other HC_replace_array_access_walker when <kinfo> is NULL.
 *
 ****************************************************************************/

static WN* HC_replace_array_access_walker(WN *wn,
        HC_KERNEL_INFO *kinfo, IPA_NODE *node,
        UINT& shared_dir_id, HC_GPU_DATA_STACK *sdata_stack)
{
    if (wn == NULL) return NULL;

    // Establish the kernel context.
    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        kinfo = node->get_kernel_info_by_sym(kfunc_st_idx);
    }

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        HC_GPU_DATA *sdata;
        if (pid == WN_PRAGMA_HC_SHARED_COPYIN)
        {
            sdata = (*node->get_shared_data_list())[shared_dir_id++];
            Is_True(sdata != NULL, (""));

            // Push it onto the stack.
            sdata_stack->push(sdata);

            // Search for the corresponding GLOBAL data in the kernel's DAS.
            ST_IDX st_idx = sdata->get_symbol();
            Is_True(kinfo != NULL, (""));
            HC_GPU_DATA *gdata = kinfo->find_gdata_for_arr_region(st_idx);
            HC_assert(gdata != NULL,
                    ("The SHARED directive for <%s> in kernel <%s> "
                     "does not have a corresponding GLOBAL directive!",
                     ST_name(st_idx), ST_name(kinfo->get_kernel_sym())));
            sdata->set_partner_gdata(gdata);
        }
        else if (pid == WN_PRAGMA_HC_SHARED_REMOVE)
        {
            // Retrieve the matching HC_GPU_DATA from the stack.
            ST_IDX st_idx = WN_st_idx(wn);
            sdata = sdata_stack->pop(HC_SHARED_DATA, st_idx);
            Is_True(sdata != NULL, (""));
        }

        // We must not further process hiCUDA directives as it may affect the
        // translation of SHARED directives.
        if (WN_pragma_hicuda(wn)) return NULL;
    }

    // IMPORTANT: handle the kids first.
    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            WN *next_wn = WN_next(kid_wn);
            WN *new_wn = HC_replace_array_access_walker(kid_wn,
                    kinfo, node, shared_dir_id, sdata_stack);
            if (new_wn != NULL)
            {
                // Insert it before the current kid.
                WN_INSERT_BlockBefore(wn, kid_wn, new_wn);
                WN_DELETE_FromBlock(wn, kid_wn);
                kid_wn = new_wn;
            } else kid_wn = next_wn;
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            WN *kid_wn = WN_kid(wn,i);
            WN *new_wn = HC_replace_array_access_walker(kid_wn,
                    kinfo, node, shared_dir_id, sdata_stack);
            if (new_wn != NULL)
            {
                // Replace the kid with the new one.
                WN_DELETE_Tree(kid_wn);
                WN_kid(wn,i) = new_wn;
            }
        }
    }

    if (kinfo == NULL) return NULL;

    if (opr == OPR_ILOAD || opr == OPR_ISTORE)
    {
        WN *addr_wn = (opr == OPR_ILOAD) ? WN_kid0(wn) : WN_kid1(wn);

        if (WN_operator(addr_wn) == OPR_ARRAY)
        {
            // Find the corresponding HC_GPU_DATA.
            ST_IDX st_idx = WN_st_idx(WN_array_base(addr_wn));

            // First, search the SHARED data stack.
            HC_GPU_DATA *gdata = sdata_stack->peek(st_idx);
            if (gdata == NULL)
            {
                // Second, search the kernel DAS.
                // Use the ILOAD/ISTORE node to probe, not ARRAY node.
                gdata = kinfo->find_gdata_for_arr_region(st_idx);
            }

            if (gdata != NULL)
            {
                Is_True(gdata->is_arr_section(), (""));

                WN *new_addr_wn =
                    HC_create_gvar_access_for_array(addr_wn, gdata);
                WN_kid(wn, opr == OPR_ILOAD ? 0 : 1) = new_addr_wn;
                WN_DELETE_Tree(addr_wn); addr_wn = NULL;
            }
        }
        else
        {
            // TODO: handle general indirect accesses.
        }
    }

    return NULL;
}

/*****************************************************************************
 *
 * Replace LDID/STID/LDA nodes in an IK-procedure.
 * Return a WN node to replace <wn>.
 *
 ****************************************************************************/

static WN* HC_replace_scalar_access_walker(WN *wn,
        IPA_NODE *node, HC_FORMAL_GPU_VAR_ARRAY *fgva)
{
    if (wn == NULL) return NULL;

    OPERATOR opr = WN_operator(wn);

    // We must not further process hiCUDA directives as it may affect the
    // translation of SHARED directives.
    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        if (WN_pragma_hicuda(wn)) return NULL;
    }

    // IMPORTANT: handle the kids first.
    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            WN *next_wn = WN_next(kid_wn);
            WN *new_wn = HC_replace_scalar_access_walker(kid_wn, node, fgva);
            if (new_wn != NULL) {
                // Insert it before the current kid.
                WN_INSERT_BlockBefore(wn, kid_wn, new_wn);
                WN_DELETE_FromBlock(wn, kid_wn);
            }
            kid_wn = next_wn;
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            WN *kid_wn = WN_kid(wn,i);
            WN *new_wn = HC_replace_scalar_access_walker(kid_wn, node, fgva);
            if (new_wn != NULL) {
                // Replace the kid with the new one.
                WN_DELETE_Tree(kid_wn);
                WN_kid(wn,i) = new_wn;
            }
        }
    }

    if (opr == OPR_LDID || opr == OPR_STID || opr == OPR_LDA)
    {
        // Get the HC_GPU_DATA and the GPU variable.
        ST_IDX st_idx = WN_st_idx(wn);
        HC_GPU_DATA *gdata = fgva->search(node->Whirl_Tree(), st_idx);
        if (gdata != NULL)
        {
            // It could be a scalar variable,
            // or LDID of pointer-to-ARRAY, or LDA of ARRAY.
            return HC_create_gvar_access_for_scalar(wn, gdata);
        }
    }

    return NULL;
}

/*****************************************************************************
 *
 * Replace LDID/STID/LDA nodes in a K-procedure.
 * Return a WN node to replace <wn>.
 *
 ****************************************************************************/

static WN* HC_replace_scalar_access_walker(WN *wn, HC_KERNEL_INFO *kinfo,
        IPA_NODE *node)
{
    if (wn == NULL) return NULL;

    // Establish the kernel context.
    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        kinfo = node->get_kernel_info_by_sym(kfunc_st_idx);
    }

    OPERATOR opr = WN_operator(wn);

    // We must not further process hiCUDA directives as it may affect the
    // translation of SHARED directives.
    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        if (WN_pragma_hicuda(wn)) return NULL;
    }

    // IMPORTANT: handle the kids first.
    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            WN *next_wn = WN_next(kid_wn);
            WN *new_wn = HC_replace_scalar_access_walker(kid_wn, kinfo, node);
            if (new_wn != NULL)
            {
                // Insert it before the current kid.
                WN_INSERT_BlockBefore(wn, kid_wn, new_wn);
                WN_DELETE_FromBlock(wn, kid_wn);
                kid_wn = new_wn;
            } else kid_wn = next_wn;
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            WN *kid_wn = WN_kid(wn,i);
            WN *new_wn = HC_replace_scalar_access_walker(kid_wn, kinfo, node);
            if (new_wn != NULL)
            {
                // Replace the kid with the new one.
                WN_DELETE_Tree(kid_wn);
                WN_kid(wn,i) = new_wn;
            }
        }
    }

    if (kinfo == NULL) return NULL;

    if (opr == OPR_LDID || opr == OPR_STID || opr == OPR_LDA)
    {
        // Get the HC_GPU_DATA and the GPU variable.
        HC_GPU_DATA *gdata = NULL;
        ST_IDX st_idx = WN_st_idx(wn);
        if (HCST_is_scalar(st_idx))
        {
            gdata = kinfo->find_gdata_for_scalar(st_idx, WN_offset(wn));
        }
        else if (HCST_is_array(st_idx))
        {
            gdata = kinfo->find_gdata_for_arr_region(st_idx);
        }
        if (gdata != NULL)
        {
            // It could be a scalar variable,
            // or LDID of pointer-to-ARRAY, or LDA of ARRAY.
            return HC_create_gvar_access_for_scalar(wn, gdata);
        }
    }

    return NULL;
}

static void HC_add_cvar_init_stmt(WN *blk_wn, HC_GPU_DATA *gdata)
{
    Is_True(gdata != NULL, (""));
    if (gdata->get_type() != HC_CONSTANT_DATA) return;

    if (flag_opencl){ 
      // Do nothing
      return;
    }

    HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();
    if (gvi->has_local_ref())
    {
        WN_INSERT_BlockLast(blk_wn, HC_create_cvar_init_stmt(gvi));
    }
}

// ASSUME: local procedure context is established.
static void HC_init_cvars(IPA_NODE *node, HC_FORMAL_GPU_VAR_ARRAY *fgva)
{
    WN *init_blk_wn = WN_CreateBlock();

    // Go through each formal.
    UINT n_formals = fgva->num_formals();
    for (UINT i = 0; i < n_formals; ++i)
    {
        HC_GPU_DATA *gdata = fgva->get_formal_data(i);
        if (gdata == NULL) continue;
        HC_add_cvar_init_stmt(init_blk_wn, gdata);
    }

    // Go through each global.
    {
        GLOBAL_GPU_DATA_TABLE *gat = fgva->get_global_data_table();
        Is_True(gat != NULL, (""));
        GLOBAL_GPU_DATA_ITER gdata_iter(gat);
        ST_IDX st_idx;
        HC_GPU_DATA *gdata;
        while (gdata_iter.Step(&st_idx, &gdata))
        {
            HC_add_cvar_init_stmt(init_blk_wn, gdata);
        }
    }

    // Insert the init statements at the beginning of the procedure body.
    WN *func_body_wn = WN_func_body(node->Whirl_Tree());
    WN_INSERT_BlockFirst(func_body_wn, init_blk_wn);
}

// ASSUME: local procedure context is established.
static void HC_init_cvars(WN *wn, IPA_NODE *node)
{
    if (wn == NULL) return;

    // Establish the kernel context.
    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        HC_KERNEL_INFO *kinfo = node->get_kernel_info_by_sym(kfunc_st_idx);
        Is_True(kinfo != NULL, (""));

        // Go through the kernel's data access summary.
        WN *init_blk_wn = WN_CreateBlock();
        
        // scalar accesses
        HC_SCALAR_INFO_ITER si_iter;
        kinfo->init_scalar_info_iter(&si_iter);
        for (HC_SCALAR_INFO *si = si_iter.First();
                !si_iter.Is_Empty(); si = si_iter.Next())
        {
            HC_GPU_DATA *gdata = si->get_gpu_data();
            if (gdata == NULL) continue;
            HC_add_cvar_init_stmt(init_blk_wn, gdata);
        }

        // array section accesses
        HC_ARRAY_SYM_INFO_ITER asi_iter;
        kinfo->init_arr_region_info_iter(&asi_iter);
        for (HC_ARRAY_SYM_INFO *asi = asi_iter.First();
                !asi_iter.Is_Empty(); asi = asi_iter.Next())
        {
            HC_ARRAY_INFO_ITER ai_iter(asi->get_arr_info_list());
            for (HC_ARRAY_INFO *ai = ai_iter.First();
                    !ai_iter.Is_Empty(); ai = ai_iter.Next())
            {
                HC_GPU_DATA *gdata = ai->get_gpu_data();
                if (gdata == NULL) continue;
                HC_add_cvar_init_stmt(init_blk_wn, gdata);
            }
        }

        // Add the block to the beginning of the kernel region.
        WN *kregion_body_wn = WN_region_body(wn);
        WN_INSERT_BlockFirst(kregion_body_wn, init_blk_wn);
    }

    OPERATOR opr = WN_operator(wn);

    // Handle composite node.
    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            HC_init_cvars(kid_wn, node);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            HC_init_cvars(WN_kid(wn,i), node);
        }
    }
}

/*****************************************************************************
 *
 * Apart from access redirection, this routine also links SHARED data with
 * GLOBAL data.
 *
 ****************************************************************************/

static void HC_redirect_accesses(IPA_NODE *node)
{
    // Access redirection can happen in both K- and IK-procedures.
    if (!node->contains_kernel() && !node->may_be_inside_kernel()) return;

    // Get the GPU variable annotation.
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));
    IPA_HC_ANNOT *annot = annots->Head();
    // Each K-procedure has a dummy annotation.
    if (annot == NULL) return;

    // Switch to this node's context.
    IPA_NODE_CONTEXT context(node);
    WN *func_wn = node->Whirl_Tree();

    // TODO: Use another pool.
    MEM_POOL *tmp_pool = node->Mem_Pool();

    if (node->may_be_inside_kernel())
    {
        // IK-procedure
        if (annot->is_dummy()) return;
        HC_FORMAL_GPU_VAR_ARRAY *fgva = 
            (HC_FORMAL_GPU_VAR_ARRAY*)annot->get_annot_data();
        Is_True(fgva != NULL, (""));

        // ILOAD/ISTORE
        HC_GPU_DATA_STACK *sdata_stack = CXX_NEW(
                HC_GPU_DATA_STACK(tmp_pool), tmp_pool);
        UINT shared_dir_id = 0;
        Is_True(HC_replace_array_access_walker(func_wn,
                    node, fgva, shared_dir_id, sdata_stack) == NULL, (""));
        Is_True(shared_dir_id
                == node->get_shared_data_list()->Elements(), (""));
        CXX_DELETE(sdata_stack, tmp_pool); sdata_stack = NULL;

        // LDID/STID/LDA
        Is_True(HC_replace_scalar_access_walker(func_wn, node, fgva)
                == NULL, (""));

        // Add initialization statements for constant memory variables
        // referenced in this procedure.
        HC_init_cvars(node, fgva);
    }
    else
    {
        // K-procedure
        Is_True(annot->is_dummy(), (""));

        // ILOAD/ISTORE
        HC_GPU_DATA_STACK *sdata_stack = CXX_NEW(
                HC_GPU_DATA_STACK(tmp_pool), tmp_pool);
        UINT shared_dir_id = 0;
        Is_True(HC_replace_array_access_walker(func_wn,
                    NULL, node, shared_dir_id, sdata_stack) == NULL, (""));
        Is_True(shared_dir_id
                == node->get_shared_data_list()->Elements(), (""));
        CXX_DELETE(sdata_stack, tmp_pool); sdata_stack = NULL;

        // LDID/STID/LDA
        Is_True(HC_replace_scalar_access_walker(func_wn, NULL, node)
                == NULL, (""));

        // Add initialization statements for constant memory variables
        // referenced in each kernel region.
        HC_init_cvars(node->Whirl_Tree(), node);
    }

    // Rebuild the Parent_Map.
    WN_Parentize(node->Whirl_Tree(), Parent_Map, Current_Map_Tab);

    // Verify the WHIRL tree.
    Is_True(WN_verifier(node->Whirl_Tree()), (""));

    // No need to reset the WN-to-IPA_EDGE map as the call nodes are intact
    // (although the paramters are not).
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * TODO: factor out code common in IPA_HC_GPU_DATA_PROP_DF::expand_formals.
 *
 ****************************************************************************/

void IPA_HC_GPU_VAR_PROP_DF::expand_formals(IPA_NODE *node)
{
    // Formal expansion only occurs in IK-procedures.
    if (! node->may_be_inside_kernel()) return;

    // Get the GPU variable annotation.
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));
    IPA_HC_ANNOT *annot = annots->Head();
    if (annot == NULL) return;
    Is_True(annot->Next() == NULL, (""));
    // No need to expand formals if the annotation has no useful info.
    if (annot->is_dummy()) return;
    HC_FORMAL_GPU_VAR_ARRAY *fgva = 
        (HC_FORMAL_GPU_VAR_ARRAY*)annot->get_annot_data();
    Is_True(fgva != NULL, (""));

    // Switch to this node's context.
    IPA_NODE_CONTEXT context(node);
    WN *func_wn = node->Whirl_Tree();

    // Construct a list of new formals for this procedure in this order:
    //      existing formals,
    //      extra globals,
    //      extra SHARED variables,
    //      extra index variables
    //
    UINT n_formals = WN_num_formals(func_wn);
    UINT n_global_gvars = fgva->get_global_data_table()->Num_Entries();
    UINT n_svars = fgva->get_sdata_table()->Num_Entries();
    UINT n_idxvs = fgva->get_idxv_sym_map()->Num_Entries();

    // This is an upper bound (not exact).
    UINT n_new_formals = n_formals + n_global_gvars + n_svars + n_idxvs;
    ST_IDX new_formals[n_new_formals];
    UINT f = 0;

    // Go through the list of existing formals. Not all of them will be
    // included in the new list, particularly those with a constant memory
    // variable.
    for (UINT i = 0; i < n_formals; ++i)
    {
        HC_GPU_DATA *gdata = fgva->get_formal_data(i);
        if (gdata == NULL)
        {
            // The original formal.
            new_formals[f++] = WN_st_idx(WN_formal(func_wn,i));
        }
        else if (gdata->get_type() == HC_GLOBAL_DATA)
        {
            new_formals[f++] = gdata->get_gvar_info()->get_symbol();
        }
    }

    // Go through the list of globals. Not all of them will be included in the
    // new list, particularly those with a constant memory variable.
    GLOBAL_GPU_DATA_ITER gdata_iter(fgva->get_global_data_table());
    ST_IDX st_idx;
    HC_GPU_DATA *gdata;
    while (gdata_iter.Step(&st_idx, &gdata))
    {
        if (gdata->get_type() == HC_GLOBAL_DATA)
        {
            new_formals[f++] = gdata->get_gvar_info()->get_symbol();
        }
    }

    // Go through the list of SHARED data.
    SDATA_TABLE_ITER stbl_iter(fgva->get_sdata_table());
    HC_GPU_DATA *sdata;
    while (stbl_iter.Step(&gdata, &sdata))
    {
        new_formals[f++] = sdata->get_gvar_info()->get_symbol();
    }

    // Go through the list of index variables.
    ST_IDX formal_st_idx;
    HC_SYM_MAP_ITER sm_iter(fgva->get_idxv_sym_map());
    while (sm_iter.Step(&st_idx, &formal_st_idx))
    {
        new_formals[f++] = formal_st_idx;
    }

    // Update the actual number of new formals.
    n_new_formals = f;

    //----------- COMMON CODE ------------------

    // Create a new function type.
    ST_IDX func_st_idx = WN_entry_name(func_wn);
    TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
    TY_IDX new_formal_types[n_new_formals];
    for (UINT i = 0; i < n_new_formals; ++i) {
        new_formal_types[i] = ST_type(new_formals[i]);
    }
    TY_IDX new_func_ty_idx = new_func_type(
            TY_name(func_ty_idx),
            Tylist_Table[TY_tylist(func_ty_idx)],
            n_new_formals, new_formal_types);

    // Update the PU table.
    Set_PU_prototype(Pu_Table[ST_pu(St_Table[func_st_idx])], new_func_ty_idx);

    // Create a new FUNC_ENTRY node.
    // We must preserve original instances of function body as there are WN
    // references in kernels' DAS.
    WN *new_func_wn = WN_CreateEntry(n_new_formals, func_st_idx,
            WN_func_body(func_wn),
            WN_func_pragmas(func_wn),
            WN_func_varrefs(func_wn));
    // IMPORTANT!!
    WN_func_body(func_wn) = NULL;
    WN_func_pragmas(func_wn) = NULL;
    WN_func_varrefs(func_wn) = NULL;
    WN_DELETE_Tree(func_wn);

    // Fill in the new formal list.
    for (UINT i = 0; i < n_new_formals; ++i) {
        WN_formal(new_func_wn,i) = WN_CreateIdname(0, new_formals[i]);
    }

    // Save the new WHIRL tree in <node> and update its Parent_Map.
    node->Set_Whirl_Tree(new_func_wn);
    WN_Parentize(node->Whirl_Tree(), Parent_Map, Current_Map_Tab);

    // Verify the WHIRL tree.
    Is_True(WN_verifier(node->Whirl_Tree()), (""));

    // The WN-to-IPA_EDGE map should not be affected.
}

// Return the next node after the given call node.
//
static WN* HC_expand_actuals_for_call(WN *wn, WN *parent_wn,
        IPA_EDGE *e, IPA_NODE *node,
        HC_FORMAL_GPU_VAR_ARRAY *fgva, HC_GPU_DATA_STACK *sdata_stack)
{
    Is_True(parent_wn != NULL && WN_operator(parent_wn) == OPR_BLOCK, (""));

    WN *next_wn = WN_next(wn);

    // For a K-procedure, the edge must be within a kernel region.
    HC_KERNEL_INFO *kinfo = NULL;
    if (node->contains_kernel())
    {
        ST_IDX kfunc_st_idx = e->get_parent_kernel_sym();
        if (kfunc_st_idx == ST_IDX_ZERO) return next_wn;
        kinfo = node->get_kernel_info_by_sym(kfunc_st_idx);
        Is_True(kinfo != NULL, (""));
    }

    // Sanity check: the callee must be an IK-procedure.
    IPA_NODE *callee = IPA_Call_Graph->Callee(e);
    Is_True(callee->may_be_inside_kernel(), ("")); 

    // Do nothing if the callee does not have useful annotation.
    IPA_HC_ANNOT_LIST *callee_annots = callee->get_hc_annots();
    Is_True(callee_annots != NULL, (""));
    IPA_HC_ANNOT *callee_annot = callee_annots->Head();
    if (callee_annot == NULL) return next_wn;
    Is_True(callee_annot->Next() == NULL, (""));
    if (callee_annot->is_dummy()) return next_wn;
    HC_FORMAL_GPU_VAR_ARRAY *callee_fgva =
        (HC_FORMAL_GPU_VAR_ARRAY*)callee_annot->get_annot_data();
    Is_True(callee_fgva != NULL, (""));

    // Get the call's WN node.
    WN *call_wn = e->Whirl_Node();
    Is_True(call_wn == wn, (""));

    // Sanity check on the number of parameters.
    UINT n_actuals = WN_kid_count(call_wn);
    Is_True(n_actuals == callee_fgva->num_formals(), (""));

    // The new list of actuals is a concatenation of four groups:
    // - existing actuals, which have been redirected to GPU variables
    // - actuals for global variables accessed by the callee
    // - actuals for SHARED variables
    // - auxiliary index variables
    // We cnstruct this new list IN THE SAME ORDER AS FORMAL EXPANSION.
    //
    UINT n_callee_global_gvars =
        callee_fgva->get_global_data_table()->Num_Entries();
    UINT n_callee_svars = callee_fgva->get_sdata_table()->Num_Entries();
    UINT n_callee_idxvs = callee_fgva->get_idxv_sym_map()->Num_Entries();

    // This is an upper bound (not exact).
    UINT n_new_actuals = n_actuals
        + n_callee_global_gvars + n_callee_svars + n_callee_idxvs;
    WN* new_actuals[n_new_actuals];
    UINT a_idx = 0;

    // Go through the existing actuals.
    for (UINT i = 0; i < n_actuals; ++i)
    {
        HC_GPU_DATA *gdata = callee_fgva->get_formal_data(i);
        if (gdata == NULL || gdata->get_type() == HC_GLOBAL_DATA)
        {
            // Migrate the original actual.
            new_actuals[a_idx++] = WN_kid(call_wn,i);
            WN_kid(call_wn,i) = NULL;   // IMPORTANT!
        }
    }

    // Go through each global of the callee.
    GLOBAL_GPU_DATA_ITER gdata_iter(callee_fgva->get_global_data_table());
    ST_IDX st_idx;
    HC_GPU_DATA *callee_gdata;
    while (gdata_iter.Step(&st_idx, &callee_gdata))
    {
        ST_IDX gvar_st_idx;
        if (node->may_be_inside_kernel())
        {
            // Find the GPU memory variable in the caller's annotation.
            Is_True(fgva != NULL, (""));
            HC_GPU_DATA *gdata =
                fgva->get_global_data_table()->Find(st_idx);
            Is_True(gdata != NULL, (""));

            gvar_st_idx = gdata->get_gvar_info()->get_symbol();
        }
        else
        {
            // Find the GPU memory variable in the kernel's DAS.
            Is_True(kinfo != NULL, (""));
            HC_GPU_DATA *gdata = kinfo->find_gdata_for_scalar(st_idx, 0);
            if (gdata == NULL)
            {
                gdata = kinfo->find_gdata_for_arr_region(st_idx);
            }
            Is_True(gdata != NULL, (""));
            // FIXME: again, <gdata> could be NULL, when a global scalar
            // variable is passed as a kernel parameter.

            gvar_st_idx = gdata->get_gvar_info()->get_symbol();
        }
        Is_True(gvar_st_idx != ST_IDX_ZERO, (""));

        TY_IDX gvar_ty_idx = ST_type(gvar_st_idx);
        new_actuals[a_idx++] = HCWN_Parm(TY_mtype(gvar_ty_idx),
                WN_LdidScalar(gvar_st_idx), gvar_ty_idx);
    }

    // Go through the SHARED variables.
    SDATA_TABLE_ITER stbl_iter(callee_fgva->get_sdata_table());
    HC_GPU_DATA *callee_sdata;
    while (stbl_iter.Step(&callee_gdata, &callee_sdata))
    {
        ST_IDX st_idx = callee_sdata->get_symbol(); // original host variable

        // First, search the local SHARED data stack.
        HC_GPU_DATA *sdata = sdata_stack->peek(st_idx);
        if (sdata == NULL)
        {
            Is_True(fgva != NULL, (""));
            HC_GPU_DATA *gdata = fgva->search(node->Whirl_Tree(), st_idx);
            Is_True(gdata != NULL, (""));
            sdata = fgva->get_sdata_alias(gdata);
            Is_True(sdata != NULL, (""));
        }

        ST_IDX svar_st_idx = sdata->get_gvar_info()->get_symbol();
        Is_True(svar_st_idx != ST_IDX_ZERO, (""));

        TY_IDX svar_ty_idx = ST_type(svar_st_idx);
        new_actuals[a_idx++] = HCWN_Parm(TY_mtype(svar_ty_idx),
                WN_LdidScalar(svar_st_idx), svar_ty_idx);
    }

    // Go through the extra index variables.
    HC_SYM_MAP_ITER sm_iter(callee_fgva->get_idxv_sym_map());
    ST_IDX formal_st_idx;
    while (sm_iter.Step(&st_idx, &formal_st_idx))
    {
        ST_IDX idxv_st_idx;
        if (node->may_be_inside_kernel())
        {
            // Find it in the caller's annotation.
            Is_True(fgva != NULL, (""));
            idxv_st_idx = fgva->get_idxv_sym_map()->Find(st_idx);
        }
        else
        {
            // This is just the original variable.
            idxv_st_idx = st_idx;
        }
        Is_True(idxv_st_idx != ST_IDX_ZERO, (""));

        TY_IDX idxv_ty_idx = ST_type(idxv_st_idx);
        new_actuals[a_idx++] = HCWN_Parm(TY_mtype(idxv_ty_idx),
                WN_LdidScalar(idxv_st_idx), idxv_ty_idx);
    }

    // Create a new call ndoe with an expanded list of actuals.
    n_new_actuals = a_idx;
    WN *new_call_wn = WN_Create(WN_opcode(call_wn), n_new_actuals);
    WN_Copy_u1u2(new_call_wn, call_wn);
    WN_Copy_u3(new_call_wn, call_wn);

    // We must preserve the original instances of PARM nodes because they
    // may contain calls with IPA_EDGEs that we need to fix.
    for (UINT i = 0; i < n_new_actuals; ++i)
    {
        WN_kid(new_call_wn,i) = new_actuals[i];
    }

    // Replace the old call with the new one.
    // TODO: is it always true that a call's parent is a BLOCK?
    WN_INSERT_BlockBefore(parent_wn, call_wn, new_call_wn);
    WN_DELETE_FromBlock(parent_wn, call_wn);

    // Rebuild the Parent_Map.
    WN_Parentize(parent_wn, Parent_Map, Current_Map_Tab);

    // There is no need to update the edge's WN node reference as we will
    // reset the entire map anyways.

    // Now the WN references in the kernel's DAS may be outdated.

    return next_wn;
}

/*****************************************************************************
 *
 * For a K-procedure, <fgva> stays NULL.
 * For an IK-procedure, <fgva> is non-NULL.
 *
 * Return the next WN node to be processed if <parent_wn> is a BLOCK, or NULL
 * otherwise.
 *
 ****************************************************************************/

static WN* HC_expand_actuals_walker(WN *wn, WN *parent_wn, IPA_NODE *node,
        HC_FORMAL_GPU_VAR_ARRAY *fgva,
        UINT& shared_dir_id, HC_GPU_DATA_STACK *sdata_stack)
{
    WN *next_wn = (parent_wn != NULL && WN_operator(parent_wn) == OPR_BLOCK) ?
        WN_next(wn) : NULL;

    if (wn == NULL) return next_wn;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        HC_GPU_DATA *sdata;
        if (pid == WN_PRAGMA_HC_SHARED_COPYIN)
        {
            sdata = (*node->get_shared_data_list())[shared_dir_id++];
            Is_True(sdata != NULL, (""));

            // Push it onto the stack.
            sdata_stack->push(sdata);
        }
        else if (pid == WN_PRAGMA_HC_SHARED_REMOVE)
        {
            // Retrieve the matching HC_GPU_DATA from the stack.
            ST_IDX st_idx = WN_st_idx(wn);
            sdata = sdata_stack->pop(HC_SHARED_DATA, st_idx);
            Is_True(sdata != NULL, (""));
        }

        return next_wn;
    }

    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            kid_wn = HC_expand_actuals_walker(kid_wn, wn, node,
                    fgva, shared_dir_id, sdata_stack);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            Is_True(HC_expand_actuals_walker(WN_kid(wn,i), wn, node,
                        fgva, shared_dir_id, sdata_stack) == NULL, (""));
        }
    }

    // We must process call argments BEFORE processing the call because these
    // arguments may be calls too.
    if (OPERATOR_is_call(opr))
    {
        IPA_EDGE *e = node->get_wn_to_edge_map()->Find(wn);
        if (e != NULL)
        {
            next_wn = HC_expand_actuals_for_call(wn, parent_wn, e,
                    node, fgva, sdata_stack);
        }
    }

    return next_wn;
}

void IPA_HC_GPU_VAR_PROP_DF::expand_actuals(IPA_NODE *node)
{
    // Actual expansion can happen in both K- and IK-procedures.
    if (!node->contains_kernel() && !node->may_be_inside_kernel()) return;

    // Get the GPU variable annotation.
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));
    HC_FORMAL_GPU_VAR_ARRAY *fgva = NULL;
    IPA_HC_ANNOT *annot = annots->Head();
    // If no annotation, this node must be unreachable, but we will handle it.
    if (annot != NULL)
    {
        Is_True(annot->Next() == NULL, (""));
        fgva = (HC_FORMAL_GPU_VAR_ARRAY*)annot->get_annot_data();
        // Here, fgva could still be NULL.
    }

    IPA_NODE_CONTEXT context(node);

    // Link IPA_EDGE with WN.
    IPA_Call_Graph->Map_Callsites(node);

    MEM_POOL *tmp_pool = node->Mem_Pool();

    HC_GPU_DATA_STACK *sdata_stack =
        CXX_NEW(HC_GPU_DATA_STACK(tmp_pool), tmp_pool);
    UINT shared_dir_id = 0;
    HC_expand_actuals_walker(node->Whirl_Tree(), NULL, node,
            fgva, shared_dir_id, sdata_stack);
    Is_True(shared_dir_id
            == node->get_shared_data_list()->Elements(), (""));
    CXX_DELETE(sdata_stack, tmp_pool);

    // Verify the WHIRL tree.
    Is_True(WN_verifier(node->Whirl_Tree()), (""));

    // The WN-to-IPA_EDGE map is out-dated.
    IPA_Call_Graph->Reset_Callsite_Map(node);
}

void IPA_HC_GPU_VAR_PROP_DF::PostProcess()
{
    // Print the annotations of each node.
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_print_hc_annots(TFile);

    // Create clones and fix call edges.
    IPA_HC_clone_and_fix_call_edges(m);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_Call_Graph->Print(TFile);

    // IMPORTANT: redo kernel classification.
    IPA_HC_classify_procedure();

    // Migrate the list of SHARED data across clones. Although they are safe
    // across clones, we need separate instance to store different SVARs.
    //
    // Since there is no edge annotation, this process is much simpler than
    // the one in reaching directives analysis, and is similar to the
    // migration of LOOP_PARTITION info.
    //
    {
        // Go through each new clone.
        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL || !node->Is_New_Clone()) continue;

            // The new clone must be an IK-procedure (even after
            // reclassification).
            Is_True(node->may_be_inside_kernel(), (""));

            // This node must have exactly one non-dummy annotation.
            IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
            Is_True(annots != NULL, (""));
            IPA_HC_ANNOT *annot = annots->Head();
            Is_True(annot != NULL && !annot->is_dummy(), (""));
            Is_True(annot->Next() == NULL, (""));

            // Get the original node.
            IPA_NODE *orig = IPA_Call_Graph->Clone_Origin(node);
            Is_True(orig != NULL, (""));
            // This original node could be an N-procedure (after
            // reclassification), but it must have exactly one annotation.

            HC_GPU_DATA_LIST *orig_sl = orig->get_shared_data_list();
            UINT n_sdata = orig_sl->Elements();
            HC_GPU_DATA_LIST *node_sl = node->get_shared_data_list();
            for (UINT i = 0; i < n_sdata; ++i)
            {
                HC_GPU_DATA *sdata = (*orig_sl)[i];
                node_sl->AddElement(CXX_NEW(HC_GPU_DATA(sdata,m), m));
            }
        }
    }

    // For each SHARED directive, create its HC_GPU_VAR_INFO and declare a
    // local shared memory variable. This is required in access redirection.
    {
        // Go through each K- and IK-procedure.
        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL) continue;

            if (!node->contains_kernel()
                    && !node->may_be_inside_kernel()) continue;
            if (!node->contains_shared_dir()) continue;

            // Needed by the declaration of shared memory variables.
            IPA_NODE_CONTEXT context(node);

            HC_GPU_DATA_LIST *sdl = node->get_shared_data_list();
            UINT n_sdata = sdl->Elements();
            for (UINT i = 0; i < n_sdata; ++i)
            {
                HC_GPU_DATA *sdata = (*sdl)[i];
                sdata->create_gvar_info();
                HC_declare_svar(sdata);
            }
        }
    }

    // Propagate SHARED data on top of the GLOBAL data.
    IPA_HC_SVAR_PROP_DF df(m);
    df.Init();
    df.Solve();

    // Some node type verification.
    // Map each node's annotation to this node's space.
    {
        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL) continue;

            // If this node has annotation, it must be an IK-,K-,or N-
            // procedure. Also, if it is not an IK-procedure, its annotation
            // must be either empty or dummy.
            IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
            if (annots == NULL) continue;
            Is_True(node->contains_kernel()
                    || !node->may_lead_to_kernel(), (""));
            IPA_HC_ANNOT *annot = annots->Head();
            if (! node->may_be_inside_kernel())
            {
                Is_True(annot == NULL || annot->is_dummy(), (""));
                // Do nothing for non-IK procedure.
                continue;
            }

            map_gpu_var_annot_to_callee(node);
        }
    }

    // Perform access redirection and clean up.
    {
        Temporary_Error_Phase ephase("Access Redirection ... ");
        if (Verbose) {
            fprintf(stderr, "Access Redirection ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile,"\t<<<Access Redirection begins>>>\n");
        }

        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL) continue;

            // Also, link SHARED directives with GLOBAL data.
            HC_redirect_accesses(node);

            // Expand formals and actuals (at each callsite).
            expand_formals(node);
            expand_actuals(node);
        }

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Access Redirection ends>>>\n");
        }
    }

    // Clear temporary states stored in each procedure node and each edge.
    {
        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL) continue;

            // Clear the New_Clone flag.
            node->Clear_New_Clone();

            node->set_hc_annots(NULL);
        }
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_Call_Graph_print(TFile);
}

void IPA_HC_GPU_VAR_PROP_DF::Print_entry(FILE *fp, void *, void *vertex)
{
    if (vertex == NULL) return;

    IPA_NODE *node = (IPA_NODE*)vertex;
    IPA_NODE_CONTEXT context(node);

    fprintf(fp, "===> NODE %s:\n", node->Name());
    node->get_hc_annots()->print(fp);
}

void IPA_HC_GPU_VAR_PROP_DF::PostProcessIO(void *vertex)
{
}

/*** DAVID CODE END ***/
