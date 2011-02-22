/** DAVID CODE BEGIN **/

// needed by ipl_summary.h included in ipa_cg.h
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "tracing.h"       // for TDEBUG_HICUDA
#include "wn.h"

#include "ipo_defs.h"

#include "ipa_trace.h"
#include "ipa_option.h"
#include "ipa_hc_kernel.h"
#include "ipa_hc_preprocess.h"

#include "hc_common.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static MEM_POOL Ipa_hc_preprocess_pool;
static BOOL Ipa_hc_pool_initialized = FALSE;

static void HC_init_mempool()
{
    if (Ipa_hc_pool_initialized) return;
    MEM_POOL_Initialize(&Ipa_hc_preprocess_pool, "hiCUDA preprocess pool", 0);
    Ipa_hc_pool_initialized = TRUE;
}

void HC_finalize_mempool()
{
    if (!Ipa_hc_pool_initialized) return;
    MEM_POOL_Delete(&Ipa_hc_preprocess_pool);
    Ipa_hc_pool_initialized = FALSE;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * - Check for kernel nesting and invalid placement of data directives.
 *
 * - Mark the procedure node if it contains at least one kernel region.
 * - For each outgoing edge, store the parent kernel region that contains the
 *   callsite.
 *
 * - Mark the procedure node if it contains at least one GLOBAL/CONSTANT dir.
 *
 * - Mark the procedure node if it contains at least one LOOP_PARTITION dir.
 *   The placement validation must be done AFTER classification.
 *
 * This function assumes that IPA_EDGEs are linked with WN nodes already.
 *
 ****************************************************************************/

static void HC_check_placement_walker(WN *wn, IPA_NODE *node,
        ST_IDX parent_kernel_sym)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    ST_IDX kernel_sym = HC_get_kernel_sym(wn);
    if (kernel_sym != ST_IDX_ZERO)
    {
        // This is a kernel region, so check for kernel nesting.
        HC_assert(parent_kernel_sym == ST_IDX_ZERO,
                ("Nested KERNEL region <%s> in procedure <%s>!",
                 ST_name(kernel_sym), node->Name()));

        // This procedure has at least one kernel.
        node->set_contains_kernel();

        parent_kernel_sym = kernel_sym;
    }
    else if (is_loop_part_region(wn))
    {
        // Meet a LOOP_PARTITION directive.
        node->set_contains_loop_part_dir();
    }
    else if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        WN_PRAGMA_ID ptype = (WN_PRAGMA_ID)WN_pragma(wn);
        switch (ptype)
        {
            case WN_PRAGMA_HC_GLOBAL_COPYIN:
            case WN_PRAGMA_HC_GLOBAL_COPYOUT:
            case WN_PRAGMA_HC_GLOBAL_FREE:
                node->set_contains_global_dir();
                HC_assert(parent_kernel_sym == ST_IDX_ZERO,
                        ("GLOBAL directive inside "
                         "KERNEL region <%s> in procedure <%s>!",
                         ST_name(parent_kernel_sym), node->Name()));
                break;

            case WN_PRAGMA_HC_CONST_COPYIN:
            case WN_PRAGMA_HC_CONST_REMOVE:
                node->set_contains_const_dir();
                HC_assert(parent_kernel_sym == ST_IDX_ZERO,
                        ("CONSTANT directive inside "
                         "KERNEL region <%s> in procedure <%s>!",
                         ST_name(parent_kernel_sym), node->Name()));
                break;

            case WN_PRAGMA_HC_SHARED_COPYIN:
            case WN_PRAGMA_HC_SHARED_COPYOUT:
            case WN_PRAGMA_HC_SHARED_REMOVE:
                node->set_contains_shared_dir();
                break;
        }
    }
    else if (OPERATOR_is_call(opr))
    {
        // Tag the edge with the parent kernel symbol.
        WN_TO_EDGE_MAP *wte_map = node->get_wn_to_edge_map();
        Is_True(wte_map != NULL, (""));
        IPA_EDGE *e = wte_map->Find(wn);
        if (e != NULL) e->set_parent_kernel_sym(parent_kernel_sym);
    }

    // Handle composite node.
    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            HC_check_placement_walker(kid_wn, node, parent_kernel_sym);
            kid_wn = WN_next(kid_wn);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HC_check_placement_walker(WN_kid(wn,i), node, parent_kernel_sym);
        }
    }
}

// a map between LABEL and its enclosing kernel's symbol (or ST_IDX_ZERO)
typedef HASH_TABLE<LABEL_IDX, ST_IDX> LABEL_DEST_INFO;
// a map between GOTO and its enclosing kernel's symbol (or ST_IDX_ZERO)
typedef HASH_TABLE<WN*, ST_IDX> LABEL_SRC_INFO;
typedef HASH_TABLE_ITER<WN*, ST_IDX> LABEL_SRC_INFO_ITER;

/*****************************************************************************
 *
 * Recursively walk through the WN tree to store each LABEL and label source
 * (i.e. GOTO etc.) in two hash tables.
 *
 ****************************************************************************/

static void HC_collect_label_info_walker(WN *wn, ST_IDX parent_kernel_sym,
        LABEL_DEST_INFO *ldest, LABEL_SRC_INFO *lsrc)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    ST_IDX kernel_sym = HC_get_kernel_sym(wn);
    if (kernel_sym != ST_IDX_ZERO) {
        // This is a kernel region, so check for kernel nesting.
        parent_kernel_sym = kernel_sym;
    } else if (opr == OPR_LABEL) {
        ldest->Enter(WN_label_number(wn), parent_kernel_sym);
    } else if (OPERATOR_has_label(opr) || OPERATOR_has_last_label(opr)) {
        lsrc->Enter(wn, parent_kernel_sym);
    }

    // Handle composite node.
    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            HC_collect_label_info_walker(kid_wn,
                    parent_kernel_sym, ldest, lsrc);
            kid_wn = WN_next(kid_wn);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HC_collect_label_info_walker(WN_kid(wn,i),
                    parent_kernel_sym, ldest, lsrc);
        }
    }
}

/*****************************************************************************
 *
 * Make sure that:
 * - each kernel region has a single entry and a single exit
 * - kernel regions are not nested
 * - no GLOBAL/CONSTANT directives are within any kernel regions
 *
 * This routine does not parse the KERNEL directives and is called before
 * procedure classification.
 *
 ****************************************************************************/

void HC_preprocess(IPA_NODE *node)
{
    HC_init_mempool();

    IPA_NODE_CONTEXT context(node);

    MEM_POOL_Push(&Ipa_hc_preprocess_pool);

    WN *func_wn = node->Whirl_Tree();

    // Link callsites with WN nodes.
    IPA_Call_Graph->Map_Callsites(node);

    HC_check_placement_walker(func_wn, node, ST_IDX_ZERO);

    // Determine where each LABEL and GOTO is located (i.e. which kernel).
    LABEL_DEST_INFO *ldest = CXX_NEW(
            LABEL_DEST_INFO(307, &Ipa_hc_preprocess_pool),
            &Ipa_hc_preprocess_pool);
    LABEL_SRC_INFO *lsrc = CXX_NEW(
            LABEL_SRC_INFO(307, &Ipa_hc_preprocess_pool),
            &Ipa_hc_preprocess_pool);
    HC_collect_label_info_walker(func_wn, ST_IDX_ZERO, ldest, lsrc);

    // Go through each label src and make sure its corresponding label dest is
    // in the same kernel context.
    WN *lsrc_wn = NULL;
    ST_IDX lsrc_kernel;
    LABEL_SRC_INFO_ITER lsrc_iter(lsrc); 
    while (lsrc_iter.Step(&lsrc_wn, &lsrc_kernel))
    {
        ST_IDX ldest_kernel = ldest->Find(WN_label_number(lsrc_wn));
        HC_assert(lsrc_kernel == ldest_kernel,
                ("GOTO statement in <%s> jumps to LABEL in <%s>, "
                 "crossing kernel boundaries!",
                 lsrc_kernel == ST_IDX_ZERO ? "" : ST_name(lsrc_kernel),
                 ldest_kernel == ST_IDX_ZERO ? "" : ST_name(ldest_kernel)));
    }

    MEM_POOL_Pop(&Ipa_hc_preprocess_pool);
}


/*****************************************************************************
 *
 * For a procedure that contains at least one KERNEL region, make sure that
 * each LOOP_PARTITION or SHARED directive is inside a KERNEL region.
 *
 * To be more efficient, we never walk into a KERNEL region.
 *
 ****************************************************************************/

static void HC_check_placement_within_kernel_walker(WN *wn, IPA_NODE *node)
{
    if (wn == NULL || HC_get_kernel_sym(wn) != ST_IDX_ZERO) return;

    HC_assert(!is_loop_part_region(wn),
            ("LOOP_PARTITION directive outside KERNEL regions "
             "in procedure <%s>!", node->Name()));

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        WN_PRAGMA_ID ptype = (WN_PRAGMA_ID)WN_pragma(wn);
        HC_assert(ptype != WN_PRAGMA_HC_SHARED_COPYIN
                && ptype != WN_PRAGMA_HC_SHARED_COPYOUT
                && ptype != WN_PRAGMA_HC_SHARED_REMOVE,
                ("SHARED directive outside KERNEL regions "
                 "in procedure <%s>!", node->Name()));
    }

    // Handle composite node.
    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            HC_check_placement_within_kernel_walker(kid_wn, node);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            HC_check_placement_within_kernel_walker(WN_kid(wn,i), node);
        }
    }
}

/*****************************************************************************
 *
 * - Make sure that no GLOBAL/CONSTANT directives in IK-procedures, and warn
 *   if there are these directives in N-procedures.
 *
 * - Make sure that no LOOP_PARTITION directives in MK-procedures. Directives
 *   in N-procedures are ignored.
 *   For K-procedures, we need to validate by another tree traversal.
 *
 * This is called after procedure classification.
 *
 ****************************************************************************/

void HC_post_validate(IPA_NODE *node)
{
    IPA_NODE_CONTEXT context(node);

    if (node->may_be_inside_kernel())
    {
        // An IK-procedure must not contain GLOBAL/CONSTANT directives.
        HC_assert(!node->contains_global_dir(),
                ("Procedure <%s>, which may be inside a kernel, "
                 "contains GLOBAL directive!", node->Name()));
        HC_assert(!node->contains_const_dir(),
                ("Procedure <%s>, which may be inside a kernel, "
                 "contains CONST directive!", node->Name()));
    }
    else if (!node->may_lead_to_kernel())
    {
        // The GLOBAL/CONSTANT directives in an N-procedure are ignored.
        if (node->contains_global_dir())
        {
            HC_warn("The GLOBAL directive(s) in procedure <%s> "
                    "are never used.", node->Name());
        }
        if (node->contains_const_dir())
        {
            HC_warn("The CONSTANT directive(s) in procedure <%s> "
                    "are never used.", node->Name());
        }
    }

    if (node->contains_loop_part_dir() || node->contains_shared_dir())
    {
        // Only K-/IK-procedures can have LOOP_PARTITION or SHARED directives.
        HC_assert(!node->may_lead_to_kernel() || node->contains_kernel(),
                ("Procedure <%s>, which is outside any KERNEL region, "
                 "contains LOOP_PARTITION directive!", node->Name()));

        if (node->contains_kernel())
        {
            HC_check_placement_within_kernel_walker(node->Whirl_Tree(), node);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Mark each procedure whether or not it may lead to a kernel. The initial set
 * is those procedures that directly contain kernels, and this information
 * will be propagated backwards.
 *
 ****************************************************************************/

class IPA_HC_LEAD_TO_KERNEL_PROP_DF : public IPA_DATA_FLOW
{
protected:

    virtual void* Meet(void* in, void* vertex, INT *change);
    virtual void* Trans(void* in, void* out, void* vertex, INT *change);

public:

    IPA_HC_LEAD_TO_KERNEL_PROP_DF(MEM_POOL *pool)
        : IPA_DATA_FLOW(BACKWARD, pool) {}

    virtual void InitializeNode(void *n);
};

void IPA_HC_LEAD_TO_KERNEL_PROP_DF::InitializeNode(void *vertex)
{
    IPA_NODE *node = (IPA_NODE*)vertex;
    // Start with nodes that directly contain kernel regions.
    if (node->contains_kernel()) node->set_may_lead_to_kernel();
}

void* IPA_HC_LEAD_TO_KERNEL_PROP_DF::Meet(void* in, void* vertex, INT *change)
{
    // All the work is done in TRANS operation.
    return NULL;
}

void* IPA_HC_LEAD_TO_KERNEL_PROP_DF::Trans(void* in, void* out, void* vertex,
        INT *change)
{
    if (vertex == NULL) return NULL;

    IPA_NODE *node = (IPA_NODE*)vertex;

    // Iterate through its predecessors and mark each one "may lead to kernel"
    // if this one is "may lead to kernel".
    if (node->may_lead_to_kernel())
    {
        IPA_PRED_ITER pred_iter(node);
        for (pred_iter.First(); !pred_iter.Is_Empty(); pred_iter.Next())
        {
            IPA_EDGE *e = pred_iter.Current_Edge();
            if (e == NULL) continue;

            IPA_NODE *caller = IPA_Call_Graph->Caller(e);
            Is_True(caller != NULL, (""));
            if (! caller->may_lead_to_kernel())
            {
                caller->set_may_lead_to_kernel();
                *change = TRUE;
            }
        }
    }

    return NULL;
}


/*****************************************************************************
 *
 * Mark each procedure whether or not it may be inside a kernel. The initial
 * set is those procedures that are called within kernels, and this
 * information will be propagated forward.
 *
 ****************************************************************************/

class IPA_HC_INSIDE_KERNEL_PROP_DF : public IPA_DATA_FLOW
{
protected:

    virtual void* Meet(void* in, void* vertex, INT *change);
    virtual void* Trans(void* in, void* out, void* vertex, INT *change);

public:

    IPA_HC_INSIDE_KERNEL_PROP_DF(MEM_POOL *pool)
        : IPA_DATA_FLOW(FORWARD, pool) {}

    virtual void InitializeNode(void *n);
};

void IPA_HC_INSIDE_KERNEL_PROP_DF::InitializeNode(void *vertex)
{
    IPA_NODE *node = (IPA_NODE*)vertex;

    // Check each incoming IPA_EDGE.
    IPA_PRED_ITER pred_iter(node);
    for (pred_iter.First(); !pred_iter.Is_Empty(); pred_iter.Next()) {
        IPA_EDGE *e = pred_iter.Current_Edge();
        if (e != NULL && e->get_parent_kernel_sym() != ST_IDX_ZERO) {
            node->set_may_be_inside_kernel();
            break;
        }
    }
}

void* IPA_HC_INSIDE_KERNEL_PROP_DF::Meet(void* in, void* vertex, INT *change)
{
    // All the work is done in TRANS operation.
    return NULL;
}

void* IPA_HC_INSIDE_KERNEL_PROP_DF::Trans(void* in, void* out, void* vertex,
        INT *change)
{
    if (vertex == NULL) return NULL;

    IPA_NODE *node = (IPA_NODE*)vertex;

    // Iterate through its successors and mark each one "may be inside kernel"
    // if this one is "may be inside kernel".
    if (node->may_be_inside_kernel())
    {
        IPA_SUCC_ITER succ_iter(node);
        for (succ_iter.First(); !succ_iter.Is_Empty(); succ_iter.Next())
        {
            IPA_EDGE *e = succ_iter.Current_Edge();
            if (e == NULL) continue;

            IPA_NODE *callee = IPA_Call_Graph->Callee(e);
            Is_True(callee != NULL, (""));
            if (! callee->may_be_inside_kernel())
            {
                callee->set_may_be_inside_kernel();
                *change = TRUE;
            }
        }
    }

    return NULL;
}


/*****************************************************************************
 *
 * Classify each procedure into three DISJOINT groups:
 * - TYPE-1: may_lead_to_kernel (sub-type: contains_kernel),
 * - TYPE-2: may_be_inside_kernel,
 * - TYPE-3: others
 *
 * Works on IPA_Call_Graph
 *
 ****************************************************************************/

void IPA_HC_classify_procedure()
{
    HC_init_mempool();

    Temporary_Error_Phase ephase("hiCUDA Kernel Classification");

    if (Verbose) {
        fprintf(stderr, "Kernel classification ... ");
        fflush(stderr);
    }

    if (Trace_IPA || Trace_Perf
            || Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "\t<<<Kernel Classification begins>>>\n");
    }

    // First, reset flags of both MAY_LEAD_TO_KERNEL and MAY_BE_INSIDE_KERNEL.
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;
            node->reset_may_lead_to_kernel();
            node->reset_may_be_inside_kernel();
        }
    }

    // "may lead to kernel" analysis
    IPA_HC_LEAD_TO_KERNEL_PROP_DF df1(&Ipa_hc_preprocess_pool);
    df1.Init();
    df1.Solve();

    // "may be inside kernel" analysis
    IPA_HC_INSIDE_KERNEL_PROP_DF df2(&Ipa_hc_preprocess_pool);
    df2.Init();
    df2.Solve();

    // Go through each procedure to make sure that it does not have both
    // "may lead to kernel" and "may be inside kernel" on.
    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL) continue;

        IPA_NODE_CONTEXT context(node);
        HC_assert(!node->may_be_inside_kernel()
                || !node->may_lead_to_kernel(),
                ("Potential kernel nesting in procedure <%s>!",
                 node->Name()));
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            IPA_NODE_CONTEXT context(node);
            fprintf(TFile, "NODE %s: ", node->Name());
            if (node->may_be_inside_kernel()) {
                fprintf(TFile, "MAY_BE_INSIDE_KERNEL");
            } else if (node->contains_kernel()) {
                fprintf(TFile, "CONTAINS_KERNEL");
            } else if (node->may_lead_to_kernel()) {
                fprintf(TFile, "MAY_LEAD_TO_KERNEL");
            } else {
                fprintf(TFile, "REGULAR");
            }
            fprintf(TFile, "\n");
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*** DAVID CODE END ***/
