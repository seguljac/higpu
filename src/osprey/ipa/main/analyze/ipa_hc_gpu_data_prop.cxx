/** DAVID CODE BEGIN **/

#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "tracing.h"                // for TDEBUG_HICUDA
#include "wn.h"
#include "wn_simp.h"
#include "wn_util.h"

#include "cxx_template.h"
#include "cxx_hash.h"

#include "ipa_cg.h"
#include "ipa_option.h"             // trace options
#include "ipa_summary.h"
#include "ipa_hc_common.h"
#include "ipa_hc_gpu_data_prop.h"
#include "ipa_hc_gdata_alloc.h"

#include "ipo_defs.h"
#include "ipo_lwn_util.h"

#include "cuda_utils.h"
#include "hc_gpu_data.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

BOOL HC_FORMAL_GPU_DATA_ARRAY::set_formal_data_used(UINT idx)
{
    Is_True(idx < _n_formals, (""));

    // Make sure the corresponding formal has GPU data.
    Is_True(_formal_data[idx] != NULL,
            ("HC_FORMAL_GPU_DATA_ARRAY::set_formal_data_used: "
             "formal #%d is set used, but it has no data\n", idx));

    if (is_formal_data_used(idx)) return FALSE;
    _formal_flags[idx] |= HC_FGDA_DATA_USED;
    return TRUE;
}

INT HC_FORMAL_GPU_DATA_ARRAY::set_formal_data_used(const HC_GPU_DATA *gdata)
{
    Is_True(gdata != NULL, (""));

    for (UINT i = 0; i < _n_formals; ++i)
    {
        if (_formal_data[i] == gdata) {
            set_formal_data_used(i);
            return i;
        }
    }

    return -1;
}

void HC_FORMAL_GPU_DATA_ARRAY::map_to_callee(IPA_EDGE *e)
{
    // Build a list of actual symbols that are directly passed.
    WN *call_wn = e->Whirl_Node();
    Is_True(call_wn != NULL, (""));
    UINT n_actuals = WN_kid_count(call_wn);
    Is_True(n_actuals == _n_formals, (""));
    ST_IDX *actuals = (ST_IDX*)alloca(sizeof(ST_IDX) * n_actuals);
    for (UINT i = 0; i < n_actuals; ++i)
    {
        WN *actual_wn = WN_kid0(WN_kid(call_wn,i));
        if (WN_operator(actual_wn) == OPR_LDID
                && WN_offset(actual_wn) == 0) {
            actuals[i] = WN_st_idx(actual_wn);
        } else {
            actuals[i] = ST_IDX_ZERO;
        }
    }

    // Build a list of formal symbols.
    IPA_NODE *callee = IPA_Call_Graph->Callee(e);
    UINT n_formals;
    ST_IDX *formals = NULL;
    {
        IPA_NODE_CONTEXT callee_context(callee);

        WN *callee_wn = callee->Whirl_Tree();
        n_formals = WN_num_formals(callee_wn);
        Is_True(n_formals == _n_formals, (""));
        formals = (ST_IDX*)alloca(sizeof(ST_IDX) * n_formals);
        for (UINT i = 0; i < n_formals; ++i) {
            formals[i] = WN_st_idx(WN_formal(callee_wn,i));
        }
    }

    IPA_NODE *caller = IPA_Call_Graph->Caller(e);

    // Go through each HC_GPU_DATA and replace actual symbols with
    // corresponding formal symbols.
    for (UINT i = 0; i < _n_formals; ++i)
    {
        HC_GPU_DATA *gdata = _formal_data[i];
        if (gdata == NULL) continue;

        // Here, the corresponding actual symbol may not be in the list
        // because it could be an LDA node.
        if (actuals[i] != ST_IDX_ZERO) {
            Is_True(gdata->get_symbol() == actuals[i], (""));
        }

        if (! gdata->replace_syms(formals[i],
                    actuals, formals, _n_formals))
        {
            // This GPU data is not propagatable to callee.
            printf("The GPU data of parameter #%d cannot be propagated "
                    "from <%s> to <%s>\n",
                    i, caller->Name(), callee->Name());

            _formal_data[i] = NULL;
        }
    }
}

BOOL HC_FORMAL_GPU_DATA_ARRAY::is_dummy() const
{
    // The annotation is dummy if no HC_GPU_DATA for any formal.
    for (UINT i = 0; i < _n_formals; ++i)
    {
        if (_formal_data[i] != NULL) return FALSE;
    }

    return TRUE;
}

BOOL HC_FORMAL_GPU_DATA_ARRAY::equals(const HC_ANNOT_DATA *o) const
{
    if (this == o) return TRUE;
    if (o == NULL) return FALSE;

    HC_FORMAL_GPU_DATA_ARRAY *other = (HC_FORMAL_GPU_DATA_ARRAY*)o;

    // This is an assertion because both instances are assumed to be in the
    // same procedure context.
    Is_True(_n_formals == other->_n_formals, (""));

    for (UINT i = 0; i < _n_formals; ++i)
    {
        HC_GPU_DATA *s1 = _formal_data[i];
        HC_GPU_DATA *s2 = other->_formal_data[i];

        if (s1 == NULL) {
            if (s2 != NULL) return FALSE;
        } else {
            if (! s1->equals(s2)) return FALSE;
        }
    }

    return TRUE;
}

void HC_FORMAL_GPU_DATA_ARRAY::print(FILE *fp) const
{
    fprintf(fp, "(");
    for (UINT i = 0; i < _n_formals; ++i)
    {
        fprintf(fp, "(");
        if (_formal_data[i] == NULL) {
            Is_True(!is_formal_data_used(i), (""));
            fprintf(fp, "null),");
        } else {
            _formal_data[i]->print(fp);
            fprintf(fp, "%s),", is_formal_data_used(i) ? "USED" : "UNUSED");
        }
    }
    fprintf(fp, ")");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

MEM_POOL Ipa_gpu_data_prop_pool;


IPA_HC_GPU_DATA_PROP_DF::IPA_HC_GPU_DATA_PROP_DF(MEM_POOL *m)
: IPA_DATA_FLOW(FORWARD, m)
{
}

/*****************************************************************************
 *
 * The GPU data annotation in the edge is in the caller space (not callee).
 *
 * This function is used by <construct_local_gpu_data_annot> and
 * <rebuild_edge_gdata_walker>.
 *
 ****************************************************************************/

void IPA_HC_GPU_DATA_PROP_DF::set_edge_gdata(IPA_EDGE *e,
        HC_GPU_DATA_STACK *stack)
{
    // Make sure the edge does not have GPU data annotation.
    Is_True(e->get_gpu_data_annot() == NULL, (""));

    WN *call_wn = e->Whirl_Node();
    Is_True(call_wn != NULL, (""));

    // Create a GPU data annotation for the edge.
    UINT n_actuals = WN_kid_count(call_wn);
    HC_FORMAL_GPU_DATA_ARRAY *fgda = CXX_NEW(
            HC_FORMAL_GPU_DATA_ARRAY(n_actuals,m), m);
    e->set_gpu_data_annot(fgda);

    // For each pointer actual, try to obtain its GPU data.
    for (UINT i = 0; i < n_actuals; ++i)
    {
        WN *param_wn = WN_kid(call_wn,i);
        if (WN_rtype(param_wn) != Pointer_type) continue;

        // The actual must be an LDID or an LDA.
        WN *actual_wn = WN_kid0(param_wn);
        if (WN_operator(actual_wn) != OPR_LDID
                && WN_operator(actual_wn) != OPR_LDA) continue;
        if (WN_offset(actual_wn) != 0) continue;

        ST_IDX actual_st_idx = WN_st_idx(actual_wn);

        // Look for the actual symbol in the data context.
        HC_GPU_DATA *gdata = stack->peek(actual_st_idx);
        if (gdata != NULL) fgda->set_formal_data(i, gdata);
    }
}


/*****************************************************************************
 *
 * Recursively walk through the WHIRL tree and determine the visible GLOBAL or
 * CONSTANT directive for each out-going IPA_EDGE.
 *
 * <dir_id> only counts GLOBAL ALLOC and CONSTANT COPYIN directives.
 *
 ****************************************************************************/

void IPA_HC_GPU_DATA_PROP_DF::construct_local_gpu_data_annot(WN *wn,
        IPA_NODE *node, WN_TO_EDGE_MAP *wte_map,
        HC_GPU_DATA_LIST *gdata_list, UINT& dir_id, HC_GPU_DATA_STACK *stack)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        switch (pid)
        {
            case WN_PRAGMA_HC_GLOBAL_COPYIN:
            case WN_PRAGMA_HC_CONST_COPYIN:
            {
                // Parse the directive.
                HC_GPU_DATA *gdata = 
                    CXX_NEW(HC_GPU_DATA(wn, node, dir_id++, m), m);
                // Cache it in the node and push it onto the data stack.
                gdata_list->AddElement(gdata);
                stack->push(gdata);
                break;
            }

            case WN_PRAGMA_HC_GLOBAL_FREE:
            case WN_PRAGMA_HC_CONST_REMOVE:
            {
                // Pop it from the data stack.
                ST_IDX st_idx = WN_st_idx(wn);
                HC_GPU_DATA_TYPE dtype = (pid == WN_PRAGMA_HC_GLOBAL_FREE) ?
                    HC_GLOBAL_DATA : HC_CONSTANT_DATA;
                Is_True(stack->pop(dtype, st_idx) != NULL, (""));
                break;
            }
        }
    }

    // For a call (with an outgoing edge) to a MAY_LEAD_TO_KERNEL function,
    // we will cache in the IPA_EDGE an annotation of visible GPU data.
    if (OPERATOR_is_call(opr))
    {
        IPA_EDGE *e = wte_map->Find(wn);
        if (e != NULL)
        {
            // NOTE: we cannot use NULL-annot check here because not every
            // node has been initialized yet.
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            if (callee->may_lead_to_kernel())
            {
                // For now, we only handle OPR_CALL's.
                Is_True(opr == OPR_CALL,
                        ("IPA_HC_GPU_DATA_PROP_DF:: "
                         "construct_local_gpu_data_annot: "
                         "meet an outgoing call edge "
                         "that is not OPR_CALL!\n"));
                set_edge_gdata(e, stack);
            }
        }
    }

    // Handle the composite node.
    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            construct_local_gpu_data_annot(kid_wn, node, wte_map,
                    gdata_list, dir_id, stack);
            kid_wn = WN_next(kid_wn);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            construct_local_gpu_data_annot(WN_kid(wn,i), node, wte_map,
                    gdata_list, dir_id, stack);
        }
    }
}

/*****************************************************************************
 *
 * For each procedure,
 * 1) initialize the GPU data annotation,
 * 2) initialize the local GPU data annotation for each outgoing edge.
 *
 ****************************************************************************/

void IPA_HC_GPU_DATA_PROP_DF::InitializeNode(void *vertex)
{
    IPA_NODE *node = (IPA_NODE*)vertex;

    // Make sure its annotation is NULL.
    Is_True(node->get_hc_annots() == NULL, (""));

    // THIS IS IMPORTANT!!
    IPA_NODE_CONTEXT ipa_context(node);

    // Do nothing if this node is not MAY_LEAD_TO_KERNEL type.
    if (! node->may_lead_to_kernel()) return;

    // Create a GPU data annotation list for this node.
    IPA_HC_ANNOT_LIST *annots = CXX_NEW(IPA_HC_ANNOT_LIST(m), m);
    node->set_hc_annots(annots);

    // Start with a dummy annotation for the root node.
    // This annotation has no GPU data info.
    if (Pred_Is_Root(node))
    {
        annots->add_dummy();
        if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
        {
            fprintf(TFile, "<%s> is connected to the root.\n", node->Name());
        }
    }

    // Link IPA_EDGEs with WN nodes.
    IPA_Call_Graph->Map_Callsites(node);

    // Create a running GPU data context.
    HC_GPU_DATA_STACK *context = CXX_NEW(HC_GPU_DATA_STACK(m), m);

    // Create a list to store the parsed COPYIN data directives.
    HC_GPU_DATA_LIST *gdata_list = node->get_gpu_data_list(); 

    // Make sure the WN nodes in the GPU data are allocated using this
    // dedicated mempool as opposed to the node's pool.
    MEM_POOL *save_wn_mempool = WN_mem_pool_ptr;
    WN_mem_pool_ptr = m;
    UINT dir_id = 0;
    construct_local_gpu_data_annot(node->Whirl_Tree(),
            node, node->get_wn_to_edge_map(), gdata_list, dir_id, context);
    Is_True(dir_id == gdata_list->Elements(), (""));
    WN_mem_pool_ptr = save_wn_mempool;
}

void* IPA_HC_GPU_DATA_PROP_DF::Meet(void* in, void* vertex, INT *change)
{
    // All the work is done in TRANS operation.
    return NULL;
}

void* IPA_HC_GPU_DATA_PROP_DF::Trans(void* in, void* out, void* vertex,
        INT *change)
{
    if (vertex == NULL) return NULL;

    IPA_NODE *node = (IPA_NODE*)vertex;

    // Do nothing for nodes that are not involved in propagation (i.e. not
    // MAY_LEAD_TO_KERNEL type).
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    if (annots == NULL) return NULL;

    // Switch to this node's context.
    IPA_NODE_CONTEXT nc(node);

    UINT n_caller_formals = WN_num_formals(node->Whirl_Tree());

    // For each NEW GPU data annotation in this node, iterate through each
    // outgoing edge and propagate the info to the callee.
    IPA_HC_ANNOT_ITER it(annots);
    for (IPA_HC_ANNOT *annot = it.First(); !it.Is_Empty(); annot = it.Next())
    {
        if (annot->is_processed()) continue;

        HC_FORMAL_GPU_DATA_ARRAY *fgda =
            (HC_FORMAL_GPU_DATA_ARRAY*)annot->get_annot_data();
        // TODO: fgda could be NULL.

        IPA_SUCC_ITER succ_iter(node);
        for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
        {
            IPA_EDGE *e = succ_iter.Current_Edge();
            // Somehow this could be NULL.
            if (e == NULL) continue;

            // Only propagate to a MAY_LEAD_TO_KERNEL node.
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            IPA_HC_ANNOT_LIST *callee_annots = callee->get_hc_annots();
            if (callee_annots == NULL) continue;

            // Create a new annotation for the callee.
            WN *call_wn = e->Whirl_Node();
            Is_True(call_wn != NULL, (""));
            UINT n_actuals = WN_kid_count(call_wn);
            HC_FORMAL_GPU_DATA_ARRAY *callee_fgda = CXX_NEW(
                    HC_FORMAL_GPU_DATA_ARRAY(n_actuals,m), m);

            HC_FORMAL_GPU_DATA_ARRAY *e_fgda = e->get_gpu_data_annot();

            for (UINT i = 0; i < n_actuals; ++i)
            {
                WN *param_wn = WN_kid(call_wn,i);
                if (WN_rtype(param_wn) != Pointer_type) continue;

                // First find the local GPU data cached in the edge.
                HC_GPU_DATA *gdata = e_fgda->get_formal_data(i);
                if (gdata == NULL && fgda != NULL)
                {
                    // The actual must be an LDID of a pointer.
                    WN *actual_wn = WN_kid0(param_wn);
                    if (WN_operator(actual_wn) != OPR_LDID
                            || WN_offset(actual_wn) != 0) continue;

                    // Go through the caller's formal to see if this
                    // actual has a corresponding HC_GPU_DATA.
                    ST_IDX actual_st_idx = WN_st_idx(actual_wn);
                    for (UINT j = 0; j < n_caller_formals; ++j)
                    {
                        HC_GPU_DATA *f_gdata = fgda->get_formal_data(j);
                        if (f_gdata == NULL) continue;

                        if (actual_st_idx == f_gdata->get_symbol()) {
                            gdata = f_gdata;
                            break;
                        }
                    }
                }

                if (gdata != NULL) {
                    // We must create a copy.
                    callee_fgda->set_formal_data(i,
                            CXX_NEW(HC_GPU_DATA(gdata,m), m));
                }
            }

            // Map the annotation to the callee space.
            callee_fgda->map_to_callee(e);

            IPA_CALL_CONTEXT *caller_cc = annot->get_call_context();
            if (callee_annots->add(e, caller_cc, callee_fgda)) *change = TRUE;
        }

        annot->set_processed();
    }

    return NULL;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * This function is very similar to <construct_local_gpu_data_annot>.
 *
 ****************************************************************************/

void IPA_HC_GPU_DATA_PROP_DF::rebuild_edge_gdata_walker(WN *wn,
        IPA_NODE *node, WN_TO_EDGE_MAP *wte_map,
        HC_GPU_DATA_LIST *gdata_list, UINT& gdata_dir_id,   // REFERENCE!
        HC_GPU_DATA_STACK *stack)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        if (pid == WN_PRAGMA_HC_GLOBAL_COPYIN
                || pid == WN_PRAGMA_HC_CONST_COPYIN)
        {
            HC_GPU_DATA *gdata = (*gdata_list)[gdata_dir_id++];
            Is_True(gdata != NULL, (""));
            stack->push(gdata);
        }
        else if (pid == WN_PRAGMA_HC_GLOBAL_FREE
                || pid == WN_PRAGMA_HC_CONST_REMOVE)
        {
            HC_GPU_DATA_TYPE type = (pid == WN_PRAGMA_HC_GLOBAL_FREE) ?
                HC_GLOBAL_DATA : HC_CONSTANT_DATA;
            ST_IDX st_idx = WN_st_idx(wn);

            Is_True(stack->pop(type, st_idx) != NULL, (""));
        }
    }

    if (OPERATOR_is_call(opr))
    {
        IPA_EDGE *e = wte_map->Find(wn);
        if (e != NULL)
        {
            // We only handle edges that lead to MAY_LEAD_TO_KERNEL functions.
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            if (callee->may_lead_to_kernel())
            {
                // For now, we only handle OPR_CALL.
                Is_True(opr == OPR_CALL,
                        ("HC_build_edge_gdata_annot_walker: meet an outgoing "
                         "call edge that is not OPR_CALL in procedure <%s>\n",
                         node->Name()));

                set_edge_gdata(e, stack);
            }
        }
    }

    // Handle composite node.
    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            WN *next_wn = WN_next(kid_wn);
            rebuild_edge_gdata_walker(kid_wn,
                    node, wte_map, gdata_list, gdata_dir_id, stack);
            kid_wn = next_wn;
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            rebuild_edge_gdata_walker(WN_kid(wn,i),
                    node, wte_map, gdata_list, gdata_dir_id, stack);
        }
    }
}

BOOL IPA_HC_GPU_DATA_PROP_DF::backprop_used_flag(IPA_NODE *node)
{
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    if (annots == NULL) return FALSE;

    BOOL changed = FALSE;

    // Go through each annotation.
    IPA_HC_ANNOT_ITER annot_iter(annots);
    for (IPA_HC_ANNOT *annot = annot_iter.First(); !annot_iter.Is_Empty();
            annot = annot_iter.Next())
    {
        HC_FORMAL_GPU_DATA_ARRAY *fgda =
            (HC_FORMAL_GPU_DATA_ARRAY*)annot->get_annot_data();
        if (fgda == NULL) continue;

        // Go through each predecessor in the call context.
        IPA_CALLER_TABLE_ITER ct_iter(
                annot->get_call_context()->get_caller_table());
        IPA_EDGE *e = NULL;
        IPA_CALL_CONTEXT_LIST *ccl = NULL;
        while (ct_iter.Step(&e, &ccl))
        {
            IPA_NODE *caller = IPA_Call_Graph->Caller(e);
            if (caller == NULL) continue;

            // Make sure we have some annotations in the caller to backprop.
            IPA_HC_ANNOT_LIST *caller_annots = caller->get_hc_annots();
            if (caller_annots == NULL) continue;

            // Get the local actual annotation in IPA_EDGE.
            HC_FORMAL_GPU_DATA_ARRAY *e_fgda = e->get_gpu_data_annot();

            // Switch to the caller context.
            IPA_NODE_CONTEXT context(caller);
            WN *caller_wn = caller->Whirl_Tree();

            // Used to get the list of actuals at the callsite.
            IPA_Call_Graph->Map_Callsites(caller);
            WN *call_wn = e->Whirl_Node();

            UINT n_formals = fgda->num_formals();
            for (UINT i = 0; i < n_formals; ++i)
            {
                // The annotation of an actual must come from the caller's
                // formal annotation if the callee's formal has GPU data but
                // the edge's formal does not.
                if (! fgda->is_formal_data_used(i)) continue;
                Is_True(fgda->get_formal_data(i) != NULL, (""));
                if (e_fgda->get_formal_data(i) != NULL) continue;

                // Get the actual symbol.
                WN *actual_wn = WN_kid0(WN_actual(call_wn,i));
                OPERATOR actual_opr = WN_operator(actual_wn);
                Is_True((actual_opr == OPR_LDID || actual_opr == OPR_LDA)
                        && WN_offset(actual_wn) == 0, (""));
                ST_IDX actual_st_idx = WN_st_idx(actual_wn);

                // This actual must be a caller's formal; find its index.
                INT n_caller_formals = WN_num_formals(caller_wn);
                INT f_idx = 0;
                for ( ; f_idx < n_caller_formals; ++f_idx) {
                    if (WN_st_idx(WN_formal(caller_wn,f_idx))
                            == actual_st_idx) break;
                }

                // Search this actual symbol in each of the caller's contexts,
                // and mark the corresponding formal data used.
                IPA_CALL_CONTEXT_ITER cci(ccl); 
                for (IPA_CALL_CONTEXT *cc = cci.First(); !cci.Is_Empty();
                        cc = cci.Next())
                {
                    HC_FORMAL_GPU_DATA_ARRAY *caller_fgda =
                        (HC_FORMAL_GPU_DATA_ARRAY*)
                        caller_annots->find_annot_data(cc);
                    Is_True(caller_fgda != NULL, (""));

                    if (caller_fgda->set_formal_data_used(f_idx)) {
                        changed = TRUE;
                    }
                }
            }
        }
    }

    return changed;
}

/*****************************************************************************
 *
 * <from_node> and <to_node> must have cloned WN trees. This function
 * constructs a map from WN nodes in <from_node> to those in <to_node>. The WN
 * nodes are call nodes and ARRAY nodes.
 *
 * This map is used to fix WN references in a cloned HC_KERNEL_INFO.
 *
 ****************************************************************************/

void IPA_HC_GPU_DATA_PROP_DF::build_wn_map(HC_WN_MAP *ww_map,
        IPA_NODE *from_node, IPA_NODE *to_node)
{
    // Get the WN trees of both nodes.
    WN *from_node_wn = NULL, *to_node_wn = NULL;
    {
        IPA_NODE_CONTEXT context(from_node);
        from_node_wn = from_node->Whirl_Tree();
    }
    {
        IPA_NODE_CONTEXT context(to_node);
        to_node_wn = to_node->Whirl_Tree();
    }

    // Since we do not need to use symbol table in this function, it does not
    // matter which procedure context we are in.
    WN_ITER *from_wni = WN_WALK_TreeIter(from_node_wn);
    WN_ITER *to_wni = WN_WALK_TreeIter(to_node_wn);
    while (from_wni != NULL)
    {
        Is_True(to_wni != NULL, (""));

        WN *from_wn = WN_ITER_wn(from_wni);
        OPERATOR from_opr = WN_operator(from_wn);
        if (OPERATOR_is_call(from_opr)
                || from_opr == OPR_ILOAD || from_opr == OPR_ISTORE)
        {
            WN *to_wn = WN_ITER_wn(to_wni);
            OPERATOR to_opr = WN_operator(to_wn);
            Is_True(OPERATOR_is_call(to_opr)
                    || to_opr == OPR_ILOAD || to_opr == OPR_ISTORE, (""));
            ww_map->Enter(from_wn, to_wn);
        }

        from_wni = WN_WALK_TreeNext(from_wni);
        to_wni = WN_WALK_TreeNext(to_wni);
    }

    Is_True(to_wni == NULL, (""));
}

/*****************************************************************************
 *
 * Clone hiCUDA data structures after cloning the node.
 *
 * - Clone the list of local HC_GPU_DATAs.
 * - Clone the HC_FORMAL_GPU_DATA_ARRAY annotation of each edge and fix the
 *   references to HC_GPU_DATAs.
 * - Clone the list of HC_KERNEL_INFOs, and fix the references to WN nodes and
 *   HC_GPU_DATAs.
 * - Rename each kernel region, and update the symbol in HC_KERNEL_INFO and
 *   the edge annotations.
 *
 ****************************************************************************/

void IPA_HC_GPU_DATA_PROP_DF::clone_hc_info(IPA_NODE *orig, IPA_NODE *clone)
{
    Is_True(!orig->Is_New_Clone() && clone->Is_New_Clone(), (""));

    // Switch to the clone's context so that WN origs constructed when cloning
    // HC_KERNEL_INFO are allocated using the clone's mempool.
    IPA_NODE_CONTEXT context(clone);

    // Although HC_GPU_DATA is safe across clones, we need to have separate
    // instances to store different GPU variables, so let's make a deep copy.
    // We maintain a map from old HC_GPU_DATA to the new HC_GPU_DATA, which
    // will be used to update HC_GPU_DATAs in edge annotation and
    // HC_KERNEL_INFO.
    HC_GPU_DATA_MAP *gdata_map = CXX_NEW(HC_GPU_DATA_MAP(41,m), m);
    HC_GPU_DATA_LIST *gl = orig->get_gpu_data_list();
    UINT n_gdata = gl->Elements();
    HC_GPU_DATA_LIST *clone_gl = clone->get_gpu_data_list();
    for (UINT i = 0; i < n_gdata; ++i)
    {
        HC_GPU_DATA *gdata = (*gl)[i];
        // All HC_GPU_DATA are allocated using the mempool in the data flow
        // framework.
        HC_GPU_DATA *gdata_clone = CXX_NEW(HC_GPU_DATA(gdata,m), m);
        clone_gl->AddElement(gdata_clone);
        gdata_map->Enter(gdata, gdata_clone);
    }

    // The edge annotation needs to refer to the HC_GPU_DATA clones, so it
    // needs to be deep-cloned too.
    IPA_SUCC_ITER succ_iter(clone);
    for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
    {
        IPA_EDGE *e = succ_iter.Current_Edge();
        if (e == NULL) continue;
        HC_FORMAL_GPU_DATA_ARRAY *e_fgda = e->get_gpu_data_annot();
        if (e_fgda == NULL) continue;

        // We must create a new HC_FORMAL_GPU_DATA_ARRAY.
        UINT n_params = e_fgda->num_formals();
        HC_FORMAL_GPU_DATA_ARRAY *new_e_fgda = CXX_NEW(
                HC_FORMAL_GPU_DATA_ARRAY(n_params,m), m);

        for (UINT i = 0; i < n_params; ++i)
        {
            HC_GPU_DATA *gdata = e_fgda->get_formal_data(i);
            if (gdata == NULL) continue;

            // Find the HC_GPU_DATA clone and put in the new annotation.
            gdata = gdata_map->Find(gdata);
            Is_True(gdata != NULL, (""));
            new_e_fgda->set_formal_data(i, gdata);
        }

        e->set_gpu_data_annot(new_e_fgda);
    }

    // Optimization: no need to clone kernel info if it has no kernel regions.
    if (! orig->contains_kernel())
    {
        // Manual clean up.
        CXX_DELETE(gdata_map, m);
        return;
    }

    // HC_KERNEL_INFO is not safe across clones, do a deep copy.
    UINT n_kinfo = orig->num_kregions();
    Is_True(n_kinfo > 0, (""));

    // First, build a WN-to-WN map from <orig> to <clone>.
    HC_WN_MAP *ww_map = CXX_NEW(HC_WN_MAP(307,m), m);
    build_wn_map(ww_map, orig, clone);

    // During the deep copy of a HC_KERNEL_INFO, we also finalize the GPU data
    // to be redirected to, so we need to clone's annotation.
    IPA_HC_ANNOT_LIST *clone_annots = clone->get_hc_annots();
    Is_True(clone_annots != NULL, (""));
    IPA_HC_ANNOT *clone_annot = clone_annots->Head();
    Is_True(clone_annot != NULL && clone_annot->Next() == NULL, (""));

#if 0
    printf("Finalizing GPU data for procedure <%s> ...\n", clone->Name());
#endif
    HC_KERNEL_INFO_LIST *kil = orig->get_kernel_info_list();
    HC_KERNEL_INFO_LIST *clone_kil = clone->get_kernel_info_list();
    MEM_POOL *cpool = clone->Mem_Pool();
    for (UINT i = 0; i < n_kinfo; ++i)
    {
        HC_KERNEL_INFO *clone_ki = CXX_NEW(
                HC_KERNEL_INFO((*kil)[i],ww_map,cpool), cpool);
        clone_ki->finalize_gpu_data(clone_annot, gdata_map);
        clone_kil->AddElement(clone_ki);
    }

    // Manual clean up.
    CXX_DELETE(gdata_map, m);
    CXX_DELETE(ww_map, m);

    // Rename kernel regions in the clone.
    HC_rename_kernels(clone);
}

/*****************************************************************************
 *
 * Expand the list of formals of <node> to pass GPU variables that correspond
 * to the formal that have *used* GPU data.
 *
 ****************************************************************************/

void IPA_HC_GPU_DATA_PROP_DF::expand_formals(IPA_NODE *node)
{
    // Only a K-/MK-procedure needs to have the formal list expanded.
    if (! node->may_lead_to_kernel()) return;

    // Get the GPU data annotation.
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));
    IPA_HC_ANNOT *annot = annots->Head();
    if (annot == NULL) return;
    Is_True(annot->Next() == NULL, (""));
    // No need to expand formals if the annotation has no useful info.
    if (annot->is_dummy()) return;
    HC_FORMAL_GPU_DATA_ARRAY *fgda =
        (HC_FORMAL_GPU_DATA_ARRAY*)annot->get_annot_data();
    Is_True(fgda != NULL, (""));

    // Switch to this node's context.
    IPA_NODE_CONTEXT context(node);
    WN *func_wn = node->Whirl_Tree();

    // Get the number of formals.
    UINT n_formals = fgda->num_formals();
    Is_True(WN_num_formals(func_wn) == n_formals, (""));

    // Construct the new list of formals.
    ST_IDX new_formals[2*n_formals];    // at most double the count
    // Start from the original list.
    for (UINT i = 0; i < n_formals; ++i)
    {
        new_formals[i] = WN_st_idx(WN_formal(func_wn,i));
    }
    UINT n_new_formals = n_formals;

    for (UINT i = 0; i < n_formals; ++i)
    {
        // The GPU data must be used.
        if (! fgda->is_formal_data_used(i)) continue;

        HC_GPU_DATA *gdata = fgda->get_formal_data(i);
        Is_True(gdata != NULL, (""));

        // Create a corresponding GPU memory variable in the local context.
        ST_IDX var_st_idx = gdata->get_symbol();
        TY_IDX gvar_ty_idx = gdata->create_gvar_type();
        ST_IDX gvar_st_idx = ST_IDX_ZERO;

        // For constant memory data, its local variable should have been
        // created by now.
        if (gdata->get_type() == HC_CONSTANT_DATA)
        {
            Is_True(gdata->get_gvar_info()->get_symbol() != ST_IDX_ZERO, (""));
            continue;
        }

        // Create a formal variable for GLOBAL.
        gvar_st_idx = new_formal_var(gen_var_str("g_", var_st_idx),
                gvar_ty_idx);

        // Store this variable in HC_GPU_DATA.
        gdata->create_gvar_info()->set_symbol(gvar_st_idx);

        // Add it to the list of formal variables.
        new_formals[n_new_formals++] = gvar_st_idx;
    }

    // Early exit: no GPU data used.
    if (n_new_formals == n_formals) return;

    // Create a new function prototype.
    ST_IDX func_st_idx = WN_entry_name(func_wn);
    TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
    TY_IDX new_formal_types[n_new_formals];
    for (UINT i = 0; i < n_new_formals; ++i) {
        new_formal_types[i] = ST_type(new_formals[i]);
    }
    TY_IDX new_func_ty_idx = new_func_type(TY_name(func_ty_idx),
            Tylist_Table[TY_tylist(func_ty_idx)],   // return type
            n_new_formals, new_formal_types);

    // Update the PU table.
    Set_PU_prototype(Pu_Table[ST_pu(St_Table[func_st_idx])], new_func_ty_idx);

    // Create a new FUNC_ENTRY node.
    // We must preserve original instances of function body as there are WN
    // references in kernels' DAS.
    WN *new_func_wn = WN_CreateEntry(n_new_formals, func_st_idx,
            WN_func_body(func_wn), WN_func_pragmas(func_wn),
            WN_func_varrefs(func_wn));
    // IMPORTANT!!
    WN_func_body(func_wn) = NULL;
    WN_func_pragmas(func_wn) = NULL;
    WN_func_varrefs(func_wn) = NULL;
    WN_DELETE_Tree(func_wn);

    // Fill in the new formal list.
    for (UINT i = 0; i < n_new_formals; ++i)
    {
        WN_formal(new_func_wn,i) = WN_CreateIdname(0, new_formals[i]);
    }

    // Save the new WHIRL tree in <node> and update its Parent_Map.
    node->Set_Whirl_Tree(new_func_wn);
    WN_Parentize(node->Whirl_Tree(), Parent_Map, Current_Map_Tab);

    // Verify the WHIRL tree.
    Is_True(WN_verifier(node->Whirl_Tree()), (""));

    // The WN-to-IPA_EDGE map should not be affected.
}

/*****************************************************************************
 *
 * At each callsite in this node, expand the list of actuals to pass GPU
 * variables that correspond to the actual that have *used* GPU data.
 *
 * ASSUME: each edge has a GPU data annotation based on local directives.
 *
 ****************************************************************************/

void IPA_HC_GPU_DATA_PROP_DF::expand_actuals(IPA_NODE *node)
{
    // Only MK-procedure needs to have the actual lists expanded.
    if (! node->may_lead_to_kernel()) return;

    // Get the GPU data annotation.
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));
    HC_FORMAL_GPU_DATA_ARRAY *fgda = NULL;
    IPA_HC_ANNOT *annot = annots->Head();
    // If no annotation, this node must be unreachable, but we will handle it.
    if (annot != NULL) {
        Is_True(annot->Next() == NULL, (""));
        fgda = (HC_FORMAL_GPU_DATA_ARRAY*)annot->get_annot_data();
        // Here, fgda could still be NULL.
    }

    IPA_NODE_CONTEXT context(node);

    // Link IPA_EDGEs with WNs.
    IPA_Call_Graph->Map_Callsites(node);

    // Go through each outgoing edge.
    IPA_SUCC_ITER succ_iter(node);
    for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
    {
        IPA_EDGE *e = succ_iter.Current_Edge();
        if (e == NULL) continue;

        // Do nothing for non-MK callee.
        IPA_NODE *callee = IPA_Call_Graph->Callee(e);
        if (! callee->may_lead_to_kernel()) continue;
        
        // Do nothing if the callee does not have useful annotation.
        IPA_HC_ANNOT_LIST *callee_annots = callee->get_hc_annots();
        Is_True(callee_annots != NULL, (""));
        IPA_HC_ANNOT *callee_annot = callee_annots->Head();
        if (callee_annot == NULL) continue;
        Is_True(callee_annot->Next() == NULL, (""));
        if (callee_annot->is_dummy()) continue;
        HC_FORMAL_GPU_DATA_ARRAY *callee_fgda =
            (HC_FORMAL_GPU_DATA_ARRAY*)callee_annot->get_annot_data();
        Is_True(callee_fgda != NULL, (""));

        // Get the corresponding annotation in the caller space, stored in the
        // edge. This annotation does not include the propagated data in the
        // caller.
        HC_FORMAL_GPU_DATA_ARRAY *e_fgda = e->get_gpu_data_annot();
        Is_True(e_fgda != NULL, (""));

        // Get the call's WN node.
        WN *call_wn = e->Whirl_Node();
        Is_True(call_wn != NULL, (""));

        // Sanity check on the number of parameters.
        UINT n_actuals = WN_kid_count(call_wn);
        Is_True(n_actuals == callee_fgda->num_formals(), (""));
        Is_True(n_actuals == e_fgda->num_formals(), (""));

        // Construct the list of extra actuals for GPU variables.
        ST_IDX gpu_actuals[n_actuals];
        UINT n_gpu_actuals = 0;

        // Go through each actual, IN THE SAME ORDER AS IN FORMAL EXPANSION.
        for (UINT i = 0; i < n_actuals; ++i)
        {
            // Do nothing if the callee's formal does not have GPU data or the
            // data is not used.
            if (!callee_fgda->is_formal_data_used(i)) continue;

            // Do nothing for a constant memory variable as we must refer to
            // <cmem> directly in access redirection.
            HC_GPU_DATA *callee_gdata = callee_fgda->get_formal_data(i);
            Is_True(callee_gdata != NULL, (""));
            if (callee_gdata->get_type() == HC_CONSTANT_DATA) continue;

            // Get the actual's GPU data.
            HC_GPU_DATA *gdata = e_fgda->get_formal_data(i);
            if (gdata == NULL)
            {
                // The GPU data must come from the caller's annotation.
                Is_True(fgda != NULL, (""));

                // Get the actual symbol.
                WN *param_wn = WN_kid(call_wn,i);
                Is_True(WN_rtype(param_wn) == Pointer_type, (""));
                WN *actual_wn = WN_kid0(param_wn);
                Is_True(WN_operator(actual_wn) == OPR_LDID
                        && WN_offset(actual_wn) == 0, (""));
                ST_IDX actual_st_idx = WN_st_idx(actual_wn);

                // Search the symbol among the caller's formals.
                UINT n_caller_formals = fgda->num_formals();
                for (UINT j = 0; j < n_caller_formals; ++j)
                {
                    HC_GPU_DATA *f_gdata = fgda->get_formal_data(j);
                    if (f_gdata == NULL) continue;

                    if (actual_st_idx == f_gdata->get_symbol()) {
                        gdata = f_gdata;
                        break;
                    }
                }
                Is_True(gdata != NULL, (""));
            }

            // The annotation must have a corresponding GPU variable.
            HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();
            Is_True(gvi != NULL, (""));
            gpu_actuals[n_gpu_actuals++] = gvi->get_symbol();
        }

        // Early exit: no used actual GPU data.
        if (n_gpu_actuals == 0) continue;

        // Create a new call node with an expanded list of actuals.
        UINT n_new_actuals = n_actuals + n_gpu_actuals;
        WN *new_call_wn = WN_Create(WN_opcode(call_wn), n_new_actuals);
        WN_Copy_u1u2(new_call_wn, call_wn);
        WN_Copy_u3(new_call_wn, call_wn);

        // We must preserve the original instances of PARM nodes because they
        // may contain calls with IPA_EDGEs that we need to fix.
        UINT a_idx = 0;
        for (UINT i = 0; i < n_actuals; ++i) {
            WN_kid(new_call_wn,a_idx++) = WN_kid(call_wn,i);
            WN_kid(call_wn,i) = NULL;   // IMPORTANT!
        }
        for (UINT i = 0; i < n_gpu_actuals; ++i) {
            WN_kid(new_call_wn,a_idx++) = HCWN_Parm(Pointer_type,
                    WN_LdidScalar(gpu_actuals[i]), ST_type(gpu_actuals[i]));
        }

        // Replace the old call with the new one.
        // TODO: is it always true that a call's parent is a BLOCK?
        WN *parent_wn = LWN_Get_Parent(call_wn);
        WN_INSERT_BlockBefore(parent_wn, call_wn, new_call_wn);
        WN_DELETE_FromBlock(parent_wn, call_wn);

        // Rebuild the Parent_Map.
        WN_Parentize(parent_wn, Parent_Map, Current_Map_Tab);

        // There is no need to update the edge's WN node reference as we will
        // reset the entire map anyways.

        // There is no need to update WN references in any kernel's DAS
        // because the call must be outside any kernel region.
    }

    // Verify the WHIRL tree.
    Is_True(WN_verifier(node->Whirl_Tree()), (""));

    // The WN-to-IPA_EDGE map is out-dated.
    IPA_Call_Graph->Reset_Callsite_Map(node);
}

void IPA_HC_GPU_DATA_PROP_DF::PostProcess()
{
    // Print the annotations of each node.
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_print_hc_annots(TFile);

    // Do kernel data analysis and match data accesses with visible GPU data.
    // The visible GPU data are marked USED in HC_FORMAL_GPU_DATA_ARRAY.
    {
        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            // We only work on K-procedures.
            if (node == NULL || !node->contains_kernel()) continue;

            HC_analyze_kernel_data(node);

            // In this case, the edges are NOT re-generated after IPA
            // preoptimization, so we do not have to rebuild the GPU data
            // (or parent kernel) annotation in each edge.
#if 0
            IPA_NODE_CONTEXT context(node);

            // Link IPA_EDGEs with WN nodes.
            IPA_Call_Graph->Map_Callsites(node);

            // Create a running GPU data context.
            HC_GPU_DATA_STACK *stack = CXX_NEW(HC_GPU_DATA_STACK(m), m);

            HC_GPU_DATA_LIST *gdata_list = node->get_gpu_data_list();
            UINT gdata_dir_id = 0;
            rebuild_edge_gdata_walker(node->Whirl_Tree(), node,
                    node->get_wn_to_edge_map(),
                    gdata_list, gdata_dir_id, stack);
            Is_True(gdata_dir_id == gdata_list->Elements(), (""));
#endif
        }
    }

    // Back-propagate the USED flags in HC_FORMAL_GPU_DATA_ARRAY.
    {
        BOOL changed;

        if (Verbose) {
            fprintf(stderr, "Back-propagation of GPU data ... ");
            fflush(stderr);
        }

        do {
            changed = FALSE;

            IPA_NODE_ITER cg_it(IPA_Call_Graph, POSTORDER);
            for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
            {
                IPA_NODE *node = cg_it.Current();
                if (node == NULL) continue;
                if (backprop_used_flag(node)) changed = TRUE;
            }
        } while (changed);
    }

    // Print the updated annotations of each node.
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_print_hc_annots(TFile);

    // Create clones and fix call edges.
    IPA_HC_clone_and_fix_call_edges(m);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_Call_Graph->Print(TFile);

    // Migrate the list of HC_GPU_DATA and HC_KERNEL_INFO to clones.
    {
        // Go through each original node.
        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL || node->Is_New_Clone()) continue;

            // This node must have at most one annotation.
            IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
            if (annots == NULL) continue;
            IPA_HC_ANNOT *annot = annots->Head();
            if (annot == NULL) continue;
            Is_True(annot->Next() == NULL, (""));

            // Go through each NEW clone of this node.
            IPA_CLONE_ARRAY *clones = IPA_Call_Graph->Clone_Array(node);
            if (clones != NULL)
            {
                UINT n_clones = clones->Elements();
                for (UINT c = 0; c < n_clones; ++c)
                {
                    IPA_NODE *clone = (*clones)[c];
                    if (clone->Is_New_Clone()) clone_hc_info(node, clone);
                }
            }

#if 0
            printf("Finalizing GPU data for procedure <%s> ...\n",
                    node->Name());
#endif
            // Finalize the GPU data in the original node's kernel info.
            // IMPORTANT: this must be done even if there is no clones.
            // This work even for nodes that do not have any kernel regions.
            UINT n_kinfo = node->num_kregions();
            for (UINT i = 0; i < n_kinfo; ++i)
            {
                node->get_kernel_info(i)->finalize_gpu_data(annot);
            }
        }
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_print_kernel_das(TFile);

    // Create GPU memory variable info structure in each HC_GPU_DATA.
    {
        // Go through all K- and MK-procedures.
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            if (!node->may_lead_to_kernel()) continue;

            HC_GPU_DATA_LIST *gdl = node->get_gpu_data_list();
            UINT n_gdata = gdl->Elements();
            for (UINT i = 0; i < n_gdata; ++i) (*gdl)[i]->create_gvar_info();
        }
    }

    // Do constant memory allocation before translating data directives.
    IPA_HC_alloc_const_mem();

    // For each propagated constant memory HC_GPU_DATA,
    //
    // 1) set its allocation offset (based on the original HC_GPU_DATA),
    // 2) create a local constant memory variable for access redirection.
    //
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            // We only work on K-/MK-procedures.
            if (node == NULL || !node->may_lead_to_kernel()) continue;

            IPA_NODE_CONTEXT context(node);

            // This node must have at most one annotation.
            IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
            Is_True(annots != NULL, (""));
            IPA_HC_ANNOT *annot = annots->Head();
            if (annot == NULL) continue;
            Is_True(annot->Next() == NULL, (""));
            if (annot->is_dummy()) continue;
            
            HC_FORMAL_GPU_DATA_ARRAY *fgda = (HC_FORMAL_GPU_DATA_ARRAY*)
                annot->get_annot_data();
            Is_True(fgda != NULL, (""));

            UINT n_formals = fgda->num_formals();
            for (UINT i = 0; i < n_formals; ++i)
            {
                // The GPU data must be used.
                if (! fgda->is_formal_data_used(i)) continue;

                HC_GPU_DATA *gdata = fgda->get_formal_data(i);
                Is_True(gdata != NULL, (""));
                // We only care about constant memory variables.
                if (gdata->get_type() != HC_CONSTANT_DATA) continue;

                // Get the original HC_GPU_DATA.
                IPA_NODE *orig_node = gdata->get_orig_proc();
                HC_GPU_DATA *orig_gdata = 
                    (*orig_node->get_gpu_data_list())[gdata->get_dir_id()];
                Is_True(gdata->have_same_origin(orig_gdata), (""));

                // Migrate the HC_GVAR_INFO.
                HC_GPU_VAR_INFO *orig_gvi = orig_gdata->get_gvar_info();
                Is_True(orig_gvi != NULL, (""));
                gdata->create_gvar_info(orig_gvi);

                // Declare a local constant memory variable.
                HC_create_local_cvar(gdata);
            }
        }
    }

    // We do not need to re-classify procedures because a clone of a
    // MAY_LEAD_TO_KERNEL node may still lead to a kernel region (i.e.
    // outgoing edges are cloned).

    {
        Temporary_Error_Phase ephase(
                "Translation of GLOBAL/CONSTANT Directives ... ");
        if (Verbose) {
            fprintf(stderr, "Translation of GLOBAL/CONSTANT Directives ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile,"\t<<<Translation of GLOBAL/CONSTANT Directives "
                    "begins>>>\n");
        }

        // Declare CUDA symbols and types that will be used when lowering
        // hiCUDA directives.
        // NOTE: it only needs to be done once globally.
        init_cuda_includes();

        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            // We only work on K-/MK-procedures.
            if (node == NULL || !node->may_lead_to_kernel()) continue;

            // Lower directives.
            HC_handle_data_directives(node);

            // Expand formals and actuals (at each callsite).
            expand_formals(node);
            expand_actuals(node);
        }

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Translation of GLOBAL/CONSTATNT Directives "
                    "ends>>>\n");
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

            // NOTE: this does not free HC_GPU_DATAs.
            node->reset_gpu_data_list();

            // Clear the annotation in each outgoing edge.
            IPA_SUCC_ITER succ_iter(node);
            for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
            {
                IPA_EDGE *e = succ_iter.Current_Edge();
                if (e != NULL) e->set_gpu_data_annot(NULL);
            }
        }
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_Call_Graph_print(TFile);
}

void IPA_HC_GPU_DATA_PROP_DF::PostProcessIO(void *vertex)
{
    if (vertex == NULL) return;

    IPA_NODE *node = (IPA_NODE*)vertex;
    // Print_entry(stderr, NULL, vertex);
}

void IPA_HC_GPU_DATA_PROP_DF::Print_entry(FILE *fp, void *, void *vertex)
{
    if (vertex == NULL) return;

    IPA_NODE *node = (IPA_NODE*)vertex;
    IPA_NODE_CONTEXT context(node);

    fprintf(fp, "===> NODE %s:\n", node->Name());
    node->get_hc_annots()->print(fp);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * We determine the local GPU data that are visible to each kernel region.
 * Then for each annotation, we augment this visible data set to include those
 * propagated GPU data and then do data matching.
 *
 * The result is cached in each HC_ACCESS_INFO.
 *
 ****************************************************************************/

static void HC_process_gpu_data_walker(WN *wn,
        IPA_NODE *node, UINT& gdata_dir_id, // REFERENCE!
        HC_GPU_DATA_STACK *stack, MEM_POOL *tmp_pool)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        // Handle the GLOBAL/CONSTANT directive here, by updating the
        // stack of visible global/constant variables.
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        if (pid == WN_PRAGMA_HC_GLOBAL_COPYIN
                || pid == WN_PRAGMA_HC_CONST_COPYIN)
        {
            HC_GPU_DATA *gdata = (*node->get_gpu_data_list())[gdata_dir_id++];
            Is_True(gdata != NULL, (""));

            // Push it onto the stack.
            stack->push(gdata);
        }
        else if (pid == WN_PRAGMA_HC_GLOBAL_FREE
                || pid == WN_PRAGMA_HC_CONST_REMOVE)
        {
            // Parse the pragma to get the GPU data type and host variable
            // symbol, so that we can find the right stack to pop entries.
            HC_GPU_DATA_TYPE type = (pid == WN_PRAGMA_HC_GLOBAL_FREE) ?
                HC_GLOBAL_DATA : HC_CONSTANT_DATA;
            ST_IDX st_idx = WN_st_idx(wn);

            HC_GPU_DATA *gdata = stack->pop(type, st_idx);
            Is_True(gdata != NULL, (""));
        }

        // No need to process further.
        return;
    }

    // For a KERNEL region, determine the visible GPU data considering
    // the propagated directives.
    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        // Get the HC_KERNEL_INFO.
        HC_KERNEL_INFO *ki = node->get_kernel_info_by_sym(kfunc_st_idx);
        Is_True(ki != NULL, (""));

        // Go through each GPU data annotation.
        IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
        Is_True(annots != NULL, (""));
        if (annots->Head() == NULL) {
            // This procedure is unreachable, so there is no annotation.
            // Let's add a dummy one.
            annots->add_dummy();
        }

        IPA_HC_ANNOT_ITER annot_iter(annots);
        for (IPA_HC_ANNOT *annot = annot_iter.First(); !annot_iter.Is_Empty();
                annot = annot_iter.Next())
        {
            // Retrieve the local visible GPU data from the stack.
            HC_VISIBLE_GPU_DATA *vgdata = stack->top(tmp_pool);

            // Add in the HC_GPU_DATAs in the annotation.
            HC_FORMAL_GPU_DATA_ARRAY *fgda =
                (HC_FORMAL_GPU_DATA_ARRAY*)annot->get_annot_data();
            if (fgda != NULL)
            {
                UINT n_formals = fgda->num_formals();
                for (UINT i = 0; i < n_formals; ++i)
                {
                    HC_GPU_DATA *gdata = fgda->get_formal_data(i);
                    if (gdata == NULL) continue;
                    // Do not add it in if the symbol already exists.
                    vgdata->Enter_If_Unique(gdata->get_symbol(), gdata);
                }
            }

            // Do data matching.
            // The matched data tables are allocated using the pool in
            // HC_KERNEL_INFO.
            ki->match_gpu_data_with_das(annot, vgdata);
        }
    }
    
    // Handle composite node.
    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            HC_process_gpu_data_walker(kid_wn,
                    node, gdata_dir_id, stack, tmp_pool);
            kid_wn = WN_next(kid_wn);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HC_process_gpu_data_walker(WN_kid(wn,i),
                    node, gdata_dir_id, stack, tmp_pool);
        }
    }
}

void IPA_HC_match_gpu_data_with_kernel_das(WN *func_wn, IPA_NODE *node,
        MEM_POOL *tmp_pool)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        HC_print_kernel_das(node, TFile);
    }

    // Assume that <tmp_pool> has been initialized.
    MEM_POOL_Push(tmp_pool);

    IPA_NODE_CONTEXT context(node);

    HC_GPU_DATA_STACK *gdata_stack = CXX_NEW(
            HC_GPU_DATA_STACK(tmp_pool), tmp_pool);

    UINT gdata_dir_id = 0;
    HC_process_gpu_data_walker(func_wn, node,
            gdata_dir_id, gdata_stack, tmp_pool);
    Is_True(gdata_dir_id == node->get_gpu_data_list()->Elements(), (""));

    MEM_POOL_Pop(tmp_pool);
}

/*** DAVID CODE END ***/
