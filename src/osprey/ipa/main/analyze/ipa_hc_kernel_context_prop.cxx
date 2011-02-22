/** DAVID CODE BEGIN **/

#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "tracing.h"       // for TDEBUG_HICUDA
#include "wn.h"
#include "wn_simp.h"
#include "wn_util.h"

#include "cxx_template.h"
#include "cxx_hash.h"

#include "ipa_cg.h"
#include "ipa_option.h"                 // trace options
#include "ipa_summary.h"
#include "ipa_section_annot.h"

#include "ipa_hc_kernel.h"
#include "ipa_hc_preprocess.h"
#include "ipa_hc_kernel_context_prop.h"

#include "hc_utils.h"

#include "ipo_defs.h"
#include "ipo_lwn_util.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

BOOL HC_KERNEL_CONTEXT::equals(const HC_ANNOT_DATA *o) const
{
    if (this == o) return TRUE;
    if (o == NULL) return FALSE;

    HC_KERNEL_CONTEXT *other = (HC_KERNEL_CONTEXT*)o;

    return (_kernel_info == other->_kernel_info)
        && (_vgrid_dim_idx == other->_vgrid_dim_idx)
        && (_vblk_dim_idx == other->_vblk_dim_idx);
}

void HC_KERNEL_CONTEXT::print(FILE *fp) const
{
    fprintf(fp, "(%p, %u, %u)",
            _kernel_info, _vgrid_dim_idx, _vblk_dim_idx);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

MEM_POOL Ipa_kernel_prop_pool;

IPA_HC_KERNEL_CONTEXT_PROP_DF::IPA_HC_KERNEL_CONTEXT_PROP_DF(MEM_POOL *m)
: IPA_DATA_FLOW(FORWARD, m)
{
}

void IPA_HC_KERNEL_CONTEXT_PROP_DF::construct_kernel_context_annot(WN *wn,
        IPA_NODE *node, WN_TO_EDGE_MAP *wte_map, HC_KERNEL_CONTEXT *kcontext)
{
    if (wn == NULL) return;

    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        Is_True(kcontext == NULL, (""));
        // Start a kernel context.
        HC_KERNEL_INFO *kinfo = node->get_kernel_info_by_sym(kfunc_st_idx);
        Is_True(kinfo != NULL, (""));
        kcontext = CXX_NEW(HC_KERNEL_CONTEXT(kinfo,0,0), m);
    }

    HC_LOOP_PART_INFO *lpi = NULL;
    if (is_loop_part_region(wn))
    {
        Is_True(kcontext != NULL, (""));
        // Parse the directive.
        lpi = CXX_NEW(HC_LOOP_PART_INFO(wn), m);
        node->get_loop_part_info_list()->AddElement(lpi);
        // Update the kernel context.
        kcontext->consumed_by_loop_partition(lpi);
    }

    OPERATOR opr = WN_operator(wn);

    // We only care about calls within kernel regions.
    if (opr == OPR_CALL && kcontext != NULL)
    {
        IPA_EDGE *e = wte_map->Find(wn);
        if (e != NULL)
        {
            // Annotate the edge with vgrid and vblk dim index offsets.
            e->set_kernel_dim_idx_offsets(kcontext->get_vgrid_dim_idx(),
                    kcontext->get_vblk_dim_idx());

            if (node->contains_kernel())
            {
                // Construct the annotation list for the callee if not done.
                IPA_NODE *callee = IPA_Call_Graph->Callee(e);
                IPA_HC_ANNOT_LIST *callee_annots = callee->get_hc_annots();
                if (callee_annots == NULL) {
                    callee_annots = CXX_NEW(IPA_HC_ANNOT_LIST(m), m);
                    callee->set_hc_annots(callee_annots);
                }

                // Add a copy of the current kernel context to the callee.
                callee_annots->add(e,
                        node->get_hc_annots()->Head()->get_call_context(),
                        CXX_NEW(HC_KERNEL_CONTEXT(*kcontext), m));
            }
        }
    }

    // Handle composite node.
    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            construct_kernel_context_annot(kid_wn, node, wte_map, kcontext);
            kid_wn = WN_next(kid_wn);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            construct_kernel_context_annot(WN_kid(wn,i),
                    node, wte_map, kcontext);
        }
    }

    // Revert back the kernel context.
    if (lpi != NULL) kcontext->unconsumed_by_loop_partition(lpi);
}

/*****************************************************************************
 *
 * K- and IK-procedures are involved in the propagation of kernel context, so
 * they will have a non-NULL annotation list.
 *
 * N-procedures may have LOOP_PARTITION directives, but they are not parsed.
 *
 * A K-procedure will have a single dummy annotation; no IK-procedures have
 * dummy annotations.
 *
 ****************************************************************************/

void IPA_HC_KERNEL_CONTEXT_PROP_DF::InitializeNode(void *vertex)
{
    IPA_NODE *node = (IPA_NODE*)vertex;

    // The nodes involved in kernel context propagation are
    // MAY_BE_INSIDE_KERNEL or CONTAINS_KERNEL.
    if (!node->contains_kernel() && !node->may_be_inside_kernel()) return;

    // Create the annotation list for this node.
    IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
    if (node->contains_kernel()) {
        // Make sure that the node has no annotation list.
        Is_True(annots == NULL, (""));
    }
    if (annots == NULL) {
        annots = CXX_NEW(IPA_HC_ANNOT_LIST(m), m);
        node->set_hc_annots(annots);
    }

    IPA_NODE_CONTEXT ipa_context(node);
    WN *func_wn = node->Whirl_Tree();

    // Link IPA_EDGEs with WN nodes.
    IPA_Call_Graph->Map_Callsites(node);
    WN_TO_EDGE_MAP *wte_map = node->get_wn_to_edge_map();

    if (node->contains_kernel())
    {
        // Construct a dummy annotation.
        annots->add_dummy();
        // Construct edge annotations and successors' annotations.
        construct_kernel_context_annot(func_wn, node, wte_map, NULL);
    }
    else
    {
        // Construct edge annotations using a virtual kernel context.
        construct_kernel_context_annot(func_wn, node, wte_map,
                CXX_NEW(HC_KERNEL_CONTEXT(), m));
    }
}

void* IPA_HC_KERNEL_CONTEXT_PROP_DF::Meet(void* in, void* vertex, INT *change)
{
    return NULL;
}

void* IPA_HC_KERNEL_CONTEXT_PROP_DF::Trans(void* in, void* out, void* vertex,
        INT *change)
{
    if (vertex == NULL) return NULL;

    IPA_NODE *node = (IPA_NODE*)vertex;

    // Only propagate if the node is MAY_BE_INSIDE_KERNEL.
    if (!node->may_be_inside_kernel()) return NULL;

    // Switch to this node's context.
    IPA_NODE_CONTEXT context(node);

    // For each NEW annotation in this node, iterate through each outgoing
    // edge and propagate the kernel context to the callee.
    IPA_HC_ANNOT_ITER annot_iter(node->get_hc_annots());
    for (IPA_HC_ANNOT *annot = annot_iter.First(); !annot_iter.Is_Empty();
            annot = annot_iter.Next())
    {
        if (annot->is_processed()) continue;

        // The annotation is never dummy.
        Is_True(! annot->is_dummy(), (""));
        HC_KERNEL_CONTEXT *kc = (HC_KERNEL_CONTEXT*)annot->get_annot_data();
        Is_True(kc != NULL, (""));

        IPA_SUCC_ITER succ_iter(node);
        for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
        {
            IPA_EDGE *e = succ_iter.Current_Edge();
            if (e == NULL) continue;

            // Add the dimension index offsets stored in the edge to the
            // new annotation.
            HC_KERNEL_CONTEXT *callee_kc = CXX_NEW(HC_KERNEL_CONTEXT(*kc), m);
            callee_kc->incr_vgrid_dim_idx(e->get_vgrid_dim_idx_ofst());
            callee_kc->incr_vblk_dim_idx(e->get_vblk_dim_idx_ofst());

            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            IPA_HC_ANNOT_LIST *callee_annots = callee->get_hc_annots();
            Is_True(callee_annots != NULL, (""));

            callee_annots->add(e, annot->get_call_context(), callee_kc);
        }

        annot->set_processed();
    }
}

void IPA_HC_KERNEL_CONTEXT_PROP_DF::PostProcess()
{
    // Print the annotations of each node.
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_print_hc_annots(TFile);

    // Create clones and fix call edges.
    IPA_HC_clone_and_fix_call_edges(m);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_Call_Graph->Print(TFile);

    // IMPORTANT: redo kernel classification.
    IPA_HC_classify_procedure();

    // Migrate the HC_LOOP_PART_INFO list and the array section information to
    // clones. The latter will be used in access redirection.
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
            // The original node could be a N-procedure now, but it must have
            // exactly one annotation.

            // For now, just shallow-copy the array section info.
            node->Set_Section_Annot(orig->Section_Annot());

            // Although HC_LOOP_PART_INFO is safe across clones, we need to
            // have separate instances to store different HC_KERNEL_CONTEXT.
            HC_LOOP_PART_INFO_LIST *lpil = node->get_loop_part_info_list();
            Is_True(lpil->Elements() == 0, (""));
            HC_LOOP_PART_INFO_LIST *orig_lpil =
                orig->get_loop_part_info_list();
            UINT n_lp_info = orig_lpil->Elements();
            for (UINT i = 0; i < n_lp_info; ++i)
            {
                lpil->AddElement(
                        CXX_NEW(HC_LOOP_PART_INFO(*(*orig_lpil)[i]), m));
            }
        }
    }

    // Some IK-procedures may have become N-procedures, so remove their
    // LOOP_PARTITION directive list to maintain consistency.
    // THIS MUST BE DONE SEPARATELY.
    {
        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL) continue;
            if (node->may_be_inside_kernel()
                    || node->may_lead_to_kernel()) continue;

            node->reset_loop_part_info_list();
        }
    }

    // Handle LOOP_PARTITION directives.
    // Parse SHARED directives.
    {
        Temporary_Error_Phase ephase(
                "Translation of LOOP_PARTITION Directives ... ");
        if (Verbose) {
            fprintf(stderr, "Translation of LOOP_PARTITION Directives ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile,"\t<<<Translation of LOOP_PARTITION Directives "
                    "begins>>>\n");
        }

        IPA_NODE_ITER cg_it(IPA_Call_Graph, PREORDER);
        for (cg_it.First(); !cg_it.Is_Empty(); cg_it.Next())
        {
            IPA_NODE *node = cg_it.Current();
            if (node == NULL) continue;

            HC_handle_in_kernel_directives(node, m);
        }

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Translation of LOOP_PARTITION Directives "
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

            // NOTE: this does not free HC_LOOP_PART_INFOs.
            // This is intended because HC_LOOP_PART_INFO s may be cached in
            // HC_GPU_DATAs for local SHARED directives.
            node->reset_loop_part_info_list();
            node->set_hc_annots(NULL);

            // We can reset the two offsets stored in each edge here.
        }
    }
}

void IPA_HC_KERNEL_CONTEXT_PROP_DF::Print_entry(FILE *fp, void *, void *vertex)
{
    if (vertex == NULL) return;

    IPA_NODE *node = (IPA_NODE*)vertex;
    IPA_NODE_CONTEXT context(node);

    fprintf(fp, "===> NODE %s:\n", node->Name());
    node->get_hc_annots()->print(fp);
}

void IPA_HC_KERNEL_CONTEXT_PROP_DF::PostProcessIO(void *vertex)
{
}

/*** DAVID CODE END ***/
