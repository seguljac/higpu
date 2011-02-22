/** DAVID CODE BEGIN **/

#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "tracing.h"        // for TDEBUG_HICUDA
#include "wn.h"
#include "wn_simp.h"
#include "wn_util.h"

#include "ipc_file.h"       // SECTION_FILE_ANNOT

#include "ipa_cg.h"
#include "ipa_summary.h"
#include "ipa_section_annot.h"
#include "ipa_hc_common.h"
#include "ipa_hc_shape_prop.h"

#include "hc_common.h"
#include "hc_utils.h"

#include "ipo_defs.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void HC_FORMAL_SHAPE_ARRAY::map_to_callee(IPA_EDGE *e)
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

    // Go through each shape and replace actual symbols with corresponding
    // formal symbols.
    for (UINT i = 0; i < _n_formals; ++i)
    {
        HC_ARRAY_SHAPE_INFO *shape = _formal_shapes[i];
        if (shape == NULL) continue;

        // Here, the corresponding actual symbol may not be in the list
        // because it could be an LDA node.
        if (actuals[i] != ST_IDX_ZERO) {
            Is_True(shape->get_sym() == actuals[i], (""));
        }

        if (! shape->replace_syms(formals[i], actuals, formals, _n_formals))
        {
            // This shape is not propagatable to callee.
            HC_warn("The shape of parameter #%d cannot be propagated "
                    "from <%s> to <%s>!",
                    i, caller->Name(), callee->Name());

            _formal_shapes[i] = NULL;
        }
    }
}

BOOL HC_FORMAL_SHAPE_ARRAY::is_dummy() const
{
    // The annotation is dummy if no HC_ARRAY_SHAPE_INFO for any formal.
    for (UINT i = 0; i < _n_formals; ++i) {
        if (_formal_shapes[i] != NULL) return FALSE;
    }

    return TRUE;
}

BOOL HC_FORMAL_SHAPE_ARRAY::equals(const HC_ANNOT_DATA *o) const
{
    if (this == o) return TRUE;
    if (o == NULL) return FALSE; 

    HC_FORMAL_SHAPE_ARRAY *other = (HC_FORMAL_SHAPE_ARRAY*)o;

    if (_n_formals != other->_n_formals) return FALSE;

    for (UINT i = 0; i < _n_formals; ++i)
    {
        HC_ARRAY_SHAPE_INFO *s1 = _formal_shapes[i];
        HC_ARRAY_SHAPE_INFO *s2 = other->_formal_shapes[i];

        if (s1 == NULL) {
            if (s2 != NULL) return FALSE;
        } else {
            if (! s1->equals(s2)) return FALSE;
        }
    }

    return TRUE;
}

void HC_FORMAL_SHAPE_ARRAY::print(FILE *fp) const
{
    fprintf(fp, "(");
    for (UINT i = 0; i < _n_formals; ++i)
    {
        if (_formal_shapes[i] == NULL) {
            fprintf(fp, "null");
        } else {
            _formal_shapes[i]->print(fp);
        }
        fprintf(fp, ",");
    }
    fprintf(fp, ")");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

MEM_POOL Ipa_shape_prop_pool;


IPA_HC_SHAPE_PROP_DF::IPA_HC_SHAPE_PROP_DF(MEM_POOL* m)
: IPA_DATA_FLOW(FORWARD, m)
{
}

/*****************************************************************************
 *
 * Recursively walk through the WHIRL tree and determine the visible SHAPE
 * directive for each out-going IPA_EDGE.
 *
 ****************************************************************************/

void IPA_HC_SHAPE_PROP_DF::construct_local_shape_annot(WN *wn,
        WN_TO_EDGE_MAP *wte_map,
        HC_SHAPE_INFO_LIST *shapes, HC_ARRAY_SHAPE_CONTEXT *context,
        HC_SHAPE_INFO_MAP *arr_shape_map)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    BOOL is_shape_dir = (opr == OPR_XPRAGMA
            && (WN_PRAGMA_ID)WN_pragma(wn) == WN_PRAGMA_HC_SHAPE);

    if (is_shape_dir)
    {
        // Parse the SHAPE directive.
        HC_ARRAY_SHAPE_INFO *shape = CXX_NEW(
                HC_ARRAY_SHAPE_INFO(WN_kid0(wn),m), m);

        // Append it to the shape list (to follow pre-order traversal). 
        shapes->AddElement(shape);
        // Push it onto the appropriate stack.
        context->push_shape_info(shape->get_sym(), shape);
    }

    // Check if it is a call, i.e. with an outgoing IPA_EDGE.
    if (OPERATOR_is_call(opr))
    {
        IPA_EDGE *e = wte_map->Find(wn);
        if (e != NULL)
        {
            // For now, we only handle OPR_CALL's.
            Is_True(opr == OPR_CALL,
                    ("IPA_HC_SHAPE_PROP_DF::construct_local_shape_annot: "
                     "meet an outgoing call edge that is not OPR_CALL!\n"));

            // Create a shape annotation for the edge.
            UINT n_actuals = WN_kid_count(wn);
            HC_FORMAL_SHAPE_ARRAY *fsa = CXX_NEW(
                    HC_FORMAL_SHAPE_ARRAY(n_actuals,m), m);
            e->set_shape_annot(fsa);

            // For each pointer actual, try to obtain its shape.
            for (UINT i = 0; i < n_actuals; ++i)
            {
                WN *param_wn = WN_kid(wn,i);
                if (WN_rtype(param_wn) != Pointer_type) continue;

                HC_ARRAY_SHAPE_INFO *shape = NULL;

                WN *actual_wn = WN_kid0(param_wn);
                switch (WN_operator(actual_wn))
                {
                    case OPR_LDID:
                    {
                        if (WN_offset(actual_wn) != 0) break;
                        // Retrieve it from the shape stack.
                        ST_IDX actual_st_idx = WN_st_idx(actual_wn);
                        shape = context->find_shape_info(actual_st_idx);
                    }
                    break;

                    case OPR_LDA:
                    {
                        if (WN_offset(actual_wn) != 0) break;
                        // The variable must be an array.
                        ST_IDX actual_st_idx = WN_st_idx(actual_wn);
                        TY_IDX actual_ty_idx = ST_type(actual_st_idx);
                        if (TY_kind(actual_ty_idx) != KIND_ARRAY) break;
                        // Search it in the implicit shape map.
                        shape = arr_shape_map->Find(actual_st_idx);
                        if (shape != NULL) break;
                        // Create a shape based on the variable's type.
                        shape = HC_ARRAY_SHAPE_INFO::create_shape(
                                actual_st_idx, m);
                        // Cache the shape in the map, even if it is NULL.
                        arr_shape_map->Enter(actual_st_idx, shape);
                    }
                    break;
                }

                if (shape != NULL) fsa->set_formal_shape(i, shape);
            }
        }
    }

    // Handle the composite node.
    if (opr == OPR_BLOCK)
    {
        // Push a new shape table.
        context->push_block();

        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            construct_local_shape_annot(kid_wn,
                    wte_map, shapes, context, arr_shape_map);
            kid_wn = WN_next(kid_wn);
        }

        // Pop the entire shape table created before.
        context->pop_block();
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            construct_local_shape_annot(WN_kid(wn,i),
                    wte_map, shapes, context, arr_shape_map);
        }
    }
}

/*****************************************************************************
 *
 * For each procedure,
 * 1) initialize the shape annotation,
 * 2) initialize the local shape annotation for each outgoing edge.
 *
 ****************************************************************************/

void IPA_HC_SHAPE_PROP_DF::InitializeNode(void *vertex)
{
    IPA_NODE *node = (IPA_NODE*)vertex;

    // Make sure its annotation is NULL.
    Is_True(node->get_hc_annots() == NULL, (""));

    // Every node is involved in this propagation.
    IPA_HC_ANNOT_LIST *annots = CXX_NEW(IPA_HC_ANNOT_LIST(m), m);
    node->set_hc_annots(annots);

    // THIS IS IMPORTANT!!
    IPA_NODE_CONTEXT ipa_context(node);

    // Start with a dummy annotation in the root node.
    // This annotation has no shape info.
    if (Pred_Is_Root(node))
    {
        annots->add_dummy();
        if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
        {
            fprintf(TFile, "<%s> is connected to the root.\n", node->Name());
        }
    }

    // TODO: use a temp mempool for the context.

    // Link IPA_EDGEs with WN nodes.
    IPA_Call_Graph->Map_Callsites(node);

    // Create a running shape context.
    HC_ARRAY_SHAPE_CONTEXT *context = CXX_NEW(HC_ARRAY_SHAPE_CONTEXT(m), m);

    // Make sure the WN nodes in the shapes are allocated using this dedicated
    // mempool as opposed to the node's pool.
    MEM_POOL *save_wn_mempool = WN_mem_pool_ptr;
    WN_mem_pool_ptr = m;
    construct_local_shape_annot(node->Whirl_Tree(),
            node->get_wn_to_edge_map(),
            node->get_shape_info_list(), context,
            node->get_arr_var_shape_info_map());
    WN_mem_pool_ptr = save_wn_mempool;
}

void* IPA_HC_SHAPE_PROP_DF::Meet(void* in, void* vertex, INT *change)
{
    // All the work is done in TRANS operation.
    return NULL;
}

void* IPA_HC_SHAPE_PROP_DF::Trans(void* in, void* out, void* vertex,
        INT *change)
{
    if (vertex == NULL) return NULL;

    IPA_NODE *node = (IPA_NODE*)vertex;

    // Every node is involved in the propagation.
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    Is_True(annots != NULL, (""));

    // Switch to this node's context.
    IPA_NODE_CONTEXT nc(node);

    UINT n_caller_formals = WN_num_formals(node->Whirl_Tree());

    // For each NEW shape annotation in this node, iterate through each
    // outgoing edge and propagate the shape info to the callee.
    IPA_HC_ANNOT_ITER it(annots);
    for (IPA_HC_ANNOT *annot = it.First(); !it.Is_Empty(); annot = it.Next())
    {
        if (annot->is_processed()) continue;

        HC_FORMAL_SHAPE_ARRAY *fsa =
            (HC_FORMAL_SHAPE_ARRAY*)annot->get_annot_data();

        IPA_SUCC_ITER succ_iter(node);
        for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
        {
            IPA_EDGE *e = succ_iter.Current_Edge();
            // Somehow this could be NULL.
            if (e == NULL) continue;

            IPA_NODE *callee = IPA_Call_Graph->Callee(e);

            // Create a new annotation for the callee.
            WN *call_wn = e->Whirl_Node();
            Is_True(call_wn != NULL, (""));
            UINT n_actuals = WN_kid_count(call_wn);
            HC_FORMAL_SHAPE_ARRAY *callee_fsa = CXX_NEW(
                    HC_FORMAL_SHAPE_ARRAY(n_actuals,m), m);

            HC_FORMAL_SHAPE_ARRAY *e_fsa = e->get_shape_annot();

            for (UINT i = 0; i < n_actuals; ++i)
            {
                WN *param_wn = WN_kid(call_wn,i);
                if (WN_rtype(param_wn) != Pointer_type) continue;

                // First find the local shape cached in the edge.
                HC_ARRAY_SHAPE_INFO *shape = e_fsa->get_formal_shape(i);
                if (shape == NULL && fsa != NULL)
                {
                    // The actual must be an LDID of a pointer.
                    WN *actual_wn = WN_kid0(param_wn);
                    if (WN_operator(actual_wn) != OPR_LDID
                            || WN_offset(actual_wn) != 0) continue;

                    // Go through the caller's formal to see if this actual
                    // has a corresponding HC_ARRAY_SHAPE_INFO.
                    ST_IDX actual_st_idx = WN_st_idx(actual_wn);
                    for (UINT j = 0; j < n_caller_formals; ++j)
                    {
                        HC_ARRAY_SHAPE_INFO *f_shape =
                            fsa->get_formal_shape(j);
                        if (f_shape == NULL) continue;

                        if (actual_st_idx == f_shape->get_sym()) {
                            shape = f_shape;
                            break;
                        }
                    }
                }

                if (shape != NULL) {
                    // We must create a copy.
                    callee_fsa->set_formal_shape(i,
                            CXX_NEW(HC_ARRAY_SHAPE_INFO(shape,m), m));
                }
            }

            // Map the annotation to the callee space.
            callee_fsa->map_to_callee(e);

            IPA_HC_ANNOT_LIST *callee_annots = callee->get_hc_annots();
            Is_True(callee_annots != NULL, (""));
            IPA_CALL_CONTEXT *caller_cc = annot->get_call_context();
            if (callee_annots->add(e, caller_cc, callee_fsa)) *change = TRUE;
        }

        annot->set_processed();
    }

    return NULL;
}

void IPA_HC_SHAPE_PROP_DF::PostProcess()
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_print_hc_annots(TFile);

    // Clone nodes based on annotations and fix call edges.
    IPA_HC_clone_and_fix_call_edges(m);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_Call_Graph->Print(TFile);

    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL) continue;

        // Rename kernel regions in cloned nodes.
        if (node->Is_New_Clone()) HC_rename_kernels(node);

        // Clear the New_Clone flag.
        node->Clear_New_Clone();

        // Apply propagated and local shape annotations.
        HC_apply_shape_annot(node, m);

        // Since each node will have a new array summary after pointer
        // promotion, we need to make sure the IVAR array is updated properly
        // in SECTION_FILE_ANNOT after IPA preoptimization.
        // THIS IS COPIED FROM IPA_ARRAY_DF_FLOW::InitializeNode.
        IP_FILE_HDR& file_hdr = node->File_Header();
        if (IP_FILE_HDR_section_annot(file_hdr) == NULL)
        {
            INT32 size;
            IVAR *ivar = IPA_get_ivar_array(node, size);
            // TODO: is this mempool permanent?
            SECTION_FILE_ANNOT *section_annot = CXX_NEW(
                    SECTION_FILE_ANNOT(ivar, node->Mem_Pool()),
                    node->Mem_Pool());
            Set_IP_FILE_HDR_section_annot(file_hdr, section_annot);
        }

        // Perform pointer promotion.
        HC_promote_dynamic_arrays(node);

        // IMPORTANT! Reset states used during propagation.
        node->reset_shape_info_list();
        node->reset_arr_var_shape_info_map();
        node->set_hc_annots(NULL);

        // NOTE: at this stage, SHAPE directives are not removed yet. They
        // will be removed at the very end of hiCUDA processing. See
        // <HC_apply_shape_annot> for the reason we need them there.

        IPA_SUCC_ITER succ_iter(node);
        for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
        {
            IPA_EDGE *e = succ_iter.Current_Edge();
            if (e == NULL) continue;
            e->set_shape_annot(NULL);
        }

        // Since the node's WN tree has changed completely after pointer
        // promotion, we need to reset WN-to-IPA_EDGE mapping.
        IPA_Call_Graph->Reset_Callsite_Map(node);
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_Call_Graph_print(TFile);
}

void IPA_HC_SHAPE_PROP_DF::PostProcessIO(void *vertex)
{
    if (vertex == NULL) return;

    IPA_NODE *node = (IPA_NODE*)vertex;
    // Print_entry(stderr, NULL, vertex);
}

void IPA_HC_SHAPE_PROP_DF::Print_entry(FILE *fp, void *, void *vertex)
{
    if (vertex == NULL) return;

    IPA_NODE *node = (IPA_NODE*)vertex;

    fprintf(fp, "===> NODE %s:\n", node->Name());
    node->get_hc_annots()->print(fp);
}

/*** DAVID CODE END ***/
