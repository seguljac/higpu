/** DAVID CODE BEGIN **/

#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "wn.h"

#include "ipa_cg.h"

#include "ipo_defs.h"   // IPA_NODE_CONTEXT

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

UINT IPA_CALL_CONTEXT::_internal_id = 0;

IPA_CALL_CONTEXT::IPA_CALL_CONTEXT(const IPA_CALL_CONTEXT *orig)
{
    Is_True(orig != NULL,
            ("IPA_CALL_CONTEXT COPY CONSTRUCTOR: orig == NULL\n"));

    _pool = orig->_pool;

    // The clone should have the same ID as the original one.
    _id = orig->_id;

    // Make a shallow copy of the table.
    _table = orig->_table;
}

void IPA_CALL_CONTEXT::add(IPA_EDGE *e, IPA_CALL_CONTEXT *caller_context)
{
    Is_True(e != NULL, ("IPA_CALL_CONTEXT::add: NULL edge\n"));

    IPA_CALL_CONTEXT_LIST *l = _table->Find(e);
    if (l == NULL) {
        l = CXX_NEW(IPA_CALL_CONTEXT_LIST, _pool);
        _table->Enter(e, l);
    }
    l->Append(caller_context);
}

void IPA_CALL_CONTEXT::mark_edges_to_be_deleted()
{
    IPA_CALLER_TABLE_ITER ct_iter(_table);
    IPA_EDGE *e;
    IPA_CALL_CONTEXT_LIST *ccl;
    while (ct_iter.Step(&e, &ccl)) {
        e->set_to_be_deleted();
#if 0
        printf("Init Edge %d (%p) to be deleted\n", e->Edge_Index(), e);
#endif
    }
}

BOOL IPA_CALL_CONTEXT::equals(const IPA_CALL_CONTEXT *other) const
{
    if (this == other) return TRUE;
    if (other == NULL) return FALSE;

    // A call context is uniquely identified by its ID.
    if (_id != other->_id) return FALSE;

    Is_True(_table == other->_table,
            ("IPA_CALL_CONTEXT::equals: invalid internal state\n"));
    return TRUE;
}

void IPA_CALL_CONTEXT::print(FILE *fp) const
{
    fprintf(fp, "[CC%d", _id);
    
    IPA_EDGE *e;
    IPA_CALL_CONTEXT_LIST *cl;
    IPA_CALLER_TABLE_ITER ti(_table);
    while (ti.Step(&e, &cl)) {
        fprintf(fp, " {E%d", e->Edge_Index());
        IPA_CALL_CONTEXT_ITER ci(cl);
        for (IPA_CALL_CONTEXT *cc = ci.First(); !ci.Is_Empty();
                cc = ci.Next()) fprintf(fp, " CC%d", cc->_id);
        fprintf(fp, "}");
    }

    fprintf(fp, "]");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

BOOL IPA_HC_ANNOT::equals(const IPA_HC_ANNOT *other) const
{
    if (this == other) return TRUE;
    if (other == NULL) return FALSE;

    if (is_dummy() && other->is_dummy()) return TRUE;

    HC_ANNOT_DATA *adata1 = _annot, *adata2 = other->_annot;
    Is_True(adata1 != NULL && adata2 != NULL, (""));
    return adata1->equals(adata2);
}

void IPA_HC_ANNOT::print(FILE *fp) const
{
    fprintf(fp, "{");

    if (is_dummy()) {
        fprintf(fp, "dummy");
    } else {
        Is_True(_annot != NULL, (""));
        _annot->print(fp);
    }

    fprintf(fp, ", ");

    if (_context == NULL) {
        fprintf(fp, "null");
    } else {
        _context->print(fp);
    }

    fprintf(fp, ", %s}\n", (is_processed() ? "PROCESSED" : "NEW"));
}

BOOL IPA_HC_ANNOT_LIST::add(IPA_EDGE *e, IPA_CALL_CONTEXT *caller_context,
        HC_ANNOT_DATA *adata)
{
    Is_True(adata != NULL && caller_context != NULL, (""));

    IPA_HC_ANNOT *annot = NULL;
    BOOL changed = FALSE;

    // Check if this annotation data is dummy.
    if (adata->is_dummy())
    {
        // Find the existing dummy annotation.
        annot = Head();
        // Create a dummy annotation if not done before.
        if (annot == NULL || !annot->is_dummy())
        {
            // Here, we use NULL as opposed to the given data.
            annot = CXX_NEW(IPA_HC_ANNOT(NULL,TRUE,_pool), _pool);
            // Insert it to the beginning of the list.
            Prepend(annot);
            changed = TRUE;
        }
    }
    else
    {
        // IMPORTANT! We must be in the callee space to do equivalence check.
        IPA_NODE_CONTEXT context(IPA_Call_Graph->Callee(e));

        // Find the existing equivalent annotation.
        IPA_HC_ANNOT_ITER annot_iter(this);
        for (annot = annot_iter.First(); !annot_iter.Is_Empty();
                annot = annot_iter.Next())
        {
            // Skip the first dummy annotation.
            if (annot->is_dummy()) continue;

            HC_ANNOT_DATA *adata2 = annot->get_annot_data();
            Is_True(adata2 != NULL, (""));
            if (adata->equals(adata2)) break;
        }

        // I am not sure if checking <annot> for NULL is safe.
        if (annot_iter.Is_Empty())
        {
            Is_True(annot == NULL, (""));
            annot = CXX_NEW(IPA_HC_ANNOT(adata,FALSE,_pool), _pool);
            Append(annot);
            changed = TRUE;
        }
    }

    Is_True(annot != NULL, (""));
    // Insert a shallow copy of the call context.
    annot->add_call_context(e,
            CXX_NEW(IPA_CALL_CONTEXT(caller_context), _pool));

    return changed;
}

void IPA_HC_ANNOT_LIST::add_dummy()
{
    // Make sure that there is no existing dummy annotation.
    IPA_HC_ANNOT *annot = Head();
    Is_True(annot == NULL || !annot->is_dummy(), 
            ("IPA_HC_ANNOT_LIST::add_dummy: duplicate dummy annotation!\n"));

    Prepend(CXX_NEW(IPA_HC_ANNOT(NULL,TRUE,_pool), _pool));
}

HC_ANNOT_DATA* IPA_HC_ANNOT_LIST::find_annot_data(
        const IPA_CALL_CONTEXT *context)
{
    Is_True(context != NULL, (""));

    IPA_HC_ANNOT_ITER it(this);
    for (IPA_HC_ANNOT *annot = it.First(); !it.Is_Empty(); annot = it.Next())
    {
        if (context->equals(annot->get_call_context())) {
            return annot->get_annot_data();
        }
    }
    return NULL;
}

void IPA_HC_ANNOT_LIST::print(FILE *fp)
{
    IPA_HC_ANNOT_ITER it(this);
    for (IPA_HC_ANNOT *annot = it.First(); !it.Is_Empty();
            annot = it.Next()) annot->print(fp);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * For each "clonable" node, i.e. with a non-NULL annotation list, create a
 * clone for each annotation.
 *
 * If a clonable node has at least one incoming edge that does not appear in
 * any call context, this edge is not involved in the propagation and the
 * original node is preserved, by associating it with a dummy annotation
 * (create one if not existed). Note that we can not use the check of whether
 * or not a node has a non-clonable predecessor to decide if this original
 * node needs to be preserved because there can be multiple edges from a
 * cloneable node to another clonable node, some of which can be involved in
 * the propagation while others of which are not.
 *
 * For a clonable but unreachable node (i.e. with no annotation), it outgoing
 * edges will not change (i.e. connect to the original nodes). However, the
 * successors could have a non-dummy annotation.
 *
 * The edges to be deleted must meet the following three conditions:
 * 1) connected to an original node,
 * 2) used in this node's annotations, and
 * 3) not used in orig-orig connection.
 * The 2nd condition is especially important.
 *
 * After calling this function, new clones have New_Clone set to TRUE. They
 * must be cleared before the next invocation of this function.
 *
 * Also, the call context of each annotation is cleared.
 *
 ****************************************************************************/

// The signature is the call context ID. We cannot use IPA_CALL_CONTEXT* here
// because there could be multiple such instances with the same ID.
typedef HASH_TABLE<UINT, IPA_NODE*> CALL_CONTEXT_CLONE_MAP;

void IPA_HC_clone_and_fix_call_edges(MEM_POOL *m)
{
    // Sanity checks,
    // Also, reset the WN-to-IPA_EDGE mapping as the edges will be changed.
    {
        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL) continue;

            // The node must not have NEW_CLONE flag set.
            Is_True(!node->Is_New_Clone(), (""));

            // No edge has TO_BE_DELETED set.
            IPA_SUCC_ITER succ_iter(node);
            for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
            {
                IPA_EDGE *e = succ_iter.Current_Edge();
                if (e == NULL) continue;
                Is_True(!e->is_to_be_deleted(), (""));
            }

            IPA_Call_Graph->Reset_Callsite_Map(node);
        }
    }

    CALL_CONTEXT_CLONE_MAP *clone_map =
        CXX_NEW(CALL_CONTEXT_CLONE_MAP(307,m), m);

    // For each clonable node, create a clone for each of its annotations, and
    // cache the correspondence in the above map.
    IPA_NODE_ITER cg_it1(IPA_Call_Graph, PREORDER);
    for (cg_it1.First(); !cg_it1.Is_Empty(); cg_it1.Next())
    {
        IPA_NODE *node = cg_it1.Current();
        if (node == NULL) continue;

        IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
        if (annots == NULL) continue;
        IPA_HC_ANNOT *annot = annots->Head();
        if (annot == NULL) continue;
#if 0
        printf("NODE %s:\n", node->Name());
#endif
        // Mark all incoming edges that appear in at least one annotation
        // TO_BE_DELETED.
        IPA_HC_ANNOT_ITER annot_iter(annots);
        for (IPA_HC_ANNOT *a = annot_iter.First(); !annot_iter.Is_Empty();
                a = annot_iter.Next())
        {
            a->get_call_context()->mark_edges_to_be_deleted();
        }

        // A node with at least one incoming edge that is not involved in the
        // propagation must has a dummy annotation.
        BOOL must_have_dummy = FALSE;
        IPA_PRED_ITER pred_iter(node);
        for (pred_iter.First(); !pred_iter.Is_Empty(); pred_iter.Next())
        {
            IPA_EDGE *e = pred_iter.Current_Edge();
            if (e == NULL) continue;
#if 0
            printf("PRED: edge %d (%p)\n", e->Edge_Index(), e);
#endif
            // Does it appear in any annotation's call context? i.e. is it
            // marked TO_BE_DELETED?
            if (! e->is_to_be_deleted()) { must_have_dummy = TRUE; break; }
        }
        if (must_have_dummy && !annot->is_dummy())
        {
            annots->add_dummy();
            annot = annots->Head();
            Is_True(annot != NULL, (""));
        }

        // The first annotation stays with the original node.
        clone_map->Enter(annot->get_call_context()->get_id(), node);

        annot = (IPA_HC_ANNOT*)annot->Next();
        while (annot != NULL)
        {
            IPA_HC_ANNOT *next_annot = (IPA_HC_ANNOT*)annot->Next();

            // Remove this annotation from the original node.
            annots->Remove_node(annot);
            // Create the clone and move this annotation into the clone.
            // TODO: is this move always safe? i.e. symbol references, etc.
            IPA_NODE *clone =
                IPA_Call_Graph->Simple_Create_Clone(node, annot, m);

            // Clone all outgoing edges to non-clonable nodes.
            IPA_SUCC_ITER succ_iter(node);
            for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
            {
                IPA_EDGE *e = succ_iter.Current_Edge();
                if (e == NULL) continue;

                // Ignore edges to clonable nodes.
                IPA_NODE *callee = IPA_Call_Graph->Callee(e);
                if (callee->get_hc_annots() != NULL) continue;

                // Add a new edge from the clone to the callee.
                IPA_EDGE *ne = IPA_Call_Graph->Add_New_Edge(
                        e->Summary_Callsite(),
                        clone->Node_Index(), callee->Node_Index());

                // IMPORTANT: migrate hiCUDA annotations.
                ne->copy_hc_annots(e);
#if 0
                printf("Added new edge %d (%p) from <%s> to <%s>\n",
                        ne->Edge_Index(), ne, clone->Name(), callee->Name());
#endif
            }

            // Record the association between call context and clone.
            clone_map->Enter(annot->get_call_context()->get_id(), clone);

            annot = next_annot;
        }
    }

    // For each node (including the clones), go through the callers in the
    // call context and add edges. For each original node, we need to clear
    // TO_BE_DELETE flag of edges that are actually being used.
    IPA_NODE_ITER cg_it2(IPA_Call_Graph, POSTORDER);
    for (cg_it2.First(); !cg_it2.Is_Empty(); cg_it2.Next())
    {
        IPA_NODE *callee = cg_it2.Current();
        if (callee == NULL) continue;

        IPA_HC_ANNOT_LIST *annots = callee->get_hc_annots();
        if (annots == NULL) continue;
        IPA_HC_ANNOT *annot = annots->Head();
        if (annot == NULL) continue;

        IPA_CALLER_TABLE *caller_table =
            annot->get_call_context()->get_caller_table();

        IPA_CALLER_TABLE_ITER ct_iter(caller_table);
        IPA_EDGE *e = NULL;
        IPA_CALL_CONTEXT_LIST *ccl = NULL;
        while (ct_iter.Step(&e, &ccl))
        {
            IPA_CALL_CONTEXT_ITER cc_iter(ccl);
            for (IPA_CALL_CONTEXT *caller_context = cc_iter.First();
                    !cc_iter.Is_Empty(); caller_context = cc_iter.Next())
            {
                // Get the caller's node.
                IPA_NODE *caller = clone_map->Find(caller_context->get_id());
                Is_True(caller != NULL, (""));

                if (callee->Is_New_Clone() || caller->Is_New_Clone()) {
                    // Add a new edge between the caller and this node.
                    IPA_EDGE *ne = IPA_Call_Graph->Add_New_Edge(
                            e->Summary_Callsite(),
                            caller->Node_Index(), callee->Node_Index());

                    // IMPORTANT: migrate hiCUDA annotations.
                    ne->copy_hc_annots(e);
#if 0
                    printf("Added new edge %d (%p) from <%s> to <%s>\n",
                            ne->Edge_Index(), ne,
                            caller->Name(), callee->Name());
#endif
                } else {
                    // This existing edge is used.
                    e->clear_to_be_deleted();
#if 0
                    printf("Edge %d (%p) is marked used\n",
                            e->Edge_Index(), e);
#endif
                }
            }
        }
    }

    // Remove edges that are still TO_BE_DELETED, and
    // clean up call context in the annotation as it is useless now.
    {
        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL) continue;

            IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
            if (annots != NULL)
            {
                IPA_HC_ANNOT *annot = annots->Head();
                if (annot != NULL) annot->clear_call_context();
            }

            // As an optimization, skip clone nodes as they have no incoming
            // edges TO_BE_DELETED.
            if (node->Is_New_Clone()) continue;

            // Get the number of incoming edges, or predecessors.
            UINT n_preds = IPA_Call_Graph->Graph()->Num_Preds(
                    node->Node_Index());
            EDGE_INDEX *useless_edges = (EDGE_INDEX*)alloca(
                    sizeof(EDGE_INDEX) * n_preds);
            UINT ue_count = 0;
#if 0
            printf("NODE %s:\n", node->Name());
#endif
            // Go through the predecessors.
            UINT pred_count = 0;
            IPA_PRED_ITER pred_iter(node);
            for (pred_iter.First(); !pred_iter.Is_Empty(); pred_iter.Next())
            {
                IPA_EDGE *e = pred_iter.Current_Edge();
                pred_count++;
                if (e != NULL) {
#if 0
                    printf("Edge %d (%p) is %s\n",
                            e->Edge_Index(), e,
                            e->is_to_be_deleted() ? "USELESS" : "USEFUL");
#endif
                    if (e->is_to_be_deleted()) {
                        useless_edges[ue_count++] = e->Edge_Index();
                    }
                }
            }

            Is_True(pred_count == n_preds, (""));

            for (UINT i = 0; i < ue_count; ++i) {
                IPA_Call_Graph->Graph()->Delete_Edge(useless_edges[i]);
#if 0
                printf("Deleting edge %d\n", useless_edges[i]);
#endif
            }
        }
    }

    // The clone map is useless at this stage.
    CXX_DELETE(clone_map, m); clone_map = NULL;

    // Go through each procedure, and update each call's symbol with the
    // updated IPA_EDGEs.
    IPA_NODE_ITER cg_it3(IPA_Call_Graph, PREORDER);
    for (cg_it3.First(); !cg_it3.Is_Empty(); cg_it3.Next())
    {
        IPA_NODE *caller = cg_it3.Current();
        if (caller == NULL) continue;

        // Switch to this node's context.
        IPA_NODE_CONTEXT context(caller);

        // Map each IPA_EDGE to the WHIRL node.
        IPA_Call_Graph->Map_Callsites(caller);

        // Go through each successor and fix its corresponding WN node.
        // NOTE: WN-to-IPA_EDGE mapping does not need to be reset.
        IPA_SUCC_ITER succ_it(caller);
        for (succ_it.First(); !succ_it.Is_Empty(); succ_it.Next())
        {
            IPA_EDGE *e = succ_it.Current_Edge();
            if (e == NULL) continue;

            WN *call_wn = e->Whirl_Node();
            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            WN_st_idx(call_wn) = ST_st_idx(callee->Func_ST());
        }
    }

    // Validate that, for each node, the number of outgoing edges is no less
    // than the number of callsites (which also includes library calls) in
    // this node.  Also fill in Total_Succ field.
    IPA_NODE_ITER cg_it4(IPA_Call_Graph, POSTORDER);
    for (cg_it4.First(); !cg_it4.Is_Empty(); cg_it4.Next())
    {
        IPA_NODE *node = cg_it4.Current();
        if (node == NULL) continue;

        // Get the number of successors.
        INT n_succs = IPA_Call_Graph->Graph()->Num_Succs(node->Node_Index());
        // Get the number of callsites.
        // NOTE: this number should be accurate no matter what.
        INT n_callsites = node->Summary_Proc()->Get_callsite_count();

        Is_True(n_succs <= n_callsites,
                ("IPA_HC_clone_and_fix_call_edges: procedure <%s> has "
                 "%d successors but %d callsites.\n",
                 node->Name(), n_succs, n_callsites));

        // We must use the callsite count.
        node->Set_Total_Succ(n_callsites);
    }

    // Go through the successors of the root node, remove the link with each
    // clone node.
    NODE_INDEX root = IPA_Call_Graph->Root();
    EDGE_INDEX *useless_edges = (EDGE_INDEX*)alloca(sizeof(EDGE_INDEX) *
            IPA_Call_Graph->Graph()->Num_Succs(root));
    INT edge_count = 0;

    IPA_SUCC_ITER succ_it(root);
    for (succ_it.First(); !succ_it.Is_Empty(); succ_it.Next())
    {
        EDGE_INDEX ei = succ_it.Current_Edge_Index();
        IPA_NODE *callee = IPA_Call_Graph->Callee(ei);
        if (callee != NULL && callee->Is_New_Clone()) {
            useless_edges[edge_count++] = ei;
        }
    }

    for (INT i = 0; i < edge_count; ++i) {
        IPA_Call_Graph->Graph()->Delete_Edge(useless_edges[i]);
    }
}

void IPA_print_hc_annots(FILE *fp)
{
    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL) continue;

        // THIS IS IMPORTANT!
        IPA_NODE_CONTEXT context(node);

        fprintf(fp, "\nAnnotation(s) of <%s>:\n", node->Name());
        IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
        if (annots != NULL) annots->print(fp);
    }
}

/*** DAVID CODE END ***/
