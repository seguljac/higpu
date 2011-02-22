/** DAVID CODE BEGIN **/
#ifdef HICUDA

#include "defs.h"
#include "wn_pragmas.h"
#include "wn.h"

#include "opt_hc_kernel_das.h"


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Return the symbol for the given KERNEL region, or ST_IDX_ZERO if it is not.
 *
 * THIS FUNCTION IS COPIED FROM <ipa/main/analyze/ipa_hc_kernel.cxx>.
 *
 ****************************************************************************/

static ST_IDX WOPT_HC_get_kernel_sym(WN *kregion_wn)
{
    if (WN_opcode(kregion_wn) != OPC_REGION
            || WN_region_kind(kregion_wn) != REGION_KIND_HICUDA) {
        return ST_IDX_ZERO;
    }

    // Get the first pragma in the pragma block.
    WN *pragma_wn = WN_first(WN_kid1(kregion_wn));
    if (pragma_wn == NULL || WN_opcode(pragma_wn) != OPC_PRAGMA
            || (WN_PRAGMA_ID)WN_pragma(pragma_wn) != WN_PRAGMA_HC_KERNEL) {
        return ST_IDX_ZERO;
    }

    return WN_st_idx(pragma_wn);
}

/*****************************************************************************
 *
 * <func_st>: the function's symbol
 * <parent_kernel_sym>: symbol of the parent kernel region during WN traversal
 *
 * <parent_kernel_map> will store the parent kernel symbol for each WN node.
 *
 ****************************************************************************/

static void HC_kernel_preprocess_walker(WN *wn,
        ST *func_st, ST_IDX parent_kernel_sym, WN_MAP parent_kernel_map)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    ST_IDX kfunc_st_idx = WOPT_HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        // Make sure there is no nested kernel regions.
        Is_True(parent_kernel_sym == ST_IDX_ZERO,
                ("HC_kernel_preprocess: nested kernel in %s\n",
                 ST_name(func_st)));
        parent_kernel_sym = kfunc_st_idx;
    }
#if 0
    if (opr == OPR_ILOAD)
    {
        // Make sure the address (kid 0) is ARRAY.
        WN *addr_wn = WN_kid0(wn);
        Is_True(WN_operator(addr_wn) == OPR_ARRAY,
                ("HC_kernel_preprocess: invalid ILOAD\n"));

        // Make sure that the address's base is an array variable (or a
        // pointer to an array variable).
        WN *addr_base_wn = WN_array_base(addr_wn);
        OPERATOR addr_base_opr = WN_operator(addr_base_wn);
        Is_True(addr_base_opr == OPR_LDID || addr_base_opr == OPR_LDA,
                ("HC_kernel_preprocess: invalid ILOAD ARRAY\n"));
        // TODO: check further.
    }
    else if (opr == OPR_ISTORE)
    {
        // Make sure the address (kid 1) is ARRAY.
        WN *addr_wn = WN_kid1(wn);
        Is_True(WN_operator(addr_wn) == OPR_ARRAY,
                ("HC_kernel_preprocess: invalid ISTORE\n"));

        // Make sure that the address's base is an array variable (or a
        // pointer to an array variable).
        WN *addr_base_wn = WN_array_base(addr_wn);
        OPERATOR addr_base_opr = WN_operator(addr_base_wn);
        Is_True(addr_base_opr == OPR_LDID || addr_base_opr == OPR_LDA,
                ("HC_kernel_preprocess: invalid ILOAD ARRAY\n"));
        // TODO: check further.
    }
#endif
    // Record whether the current node is inside/outside any kernel region.
    WN_MAP32_Set(parent_kernel_map, wn, parent_kernel_sym);

    // Handle the composite node.
    if (opr == OPR_BLOCK)
    {
        WN *node = WN_first(wn);
        while (node != NULL) {
            HC_kernel_preprocess_walker(node,
                    func_st, parent_kernel_sym, parent_kernel_map);
            node = WN_next(node);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HC_kernel_preprocess_walker(WN_kid(wn,i),
                    func_st, parent_kernel_sym, parent_kernel_map);
        }
    }
}

/*****************************************************************************
 *
 * Walk through the given procedure (FUNC_ENTRY), and do the following:
 * - Make sure that kernel regions are not nested.
 * - Make sure indirect loads/stores have ARRAY as the address.
 * - For each DEF/USE in the DU-chain, mark it inside/outside a kernel region
 *   (if inside, which kernel region).
 *
 * <parent_kernel_map> is a map from each WN node to the symbol of its parent
 * kernel region (or ST_IDX_ZERO). It should be empty before the initial call
 * and gets filled at the end.
 *
 ****************************************************************************/

void HC_kernel_preprocess(WN *wn, WN_MAP parent_kernel_map)
{
    HC_kernel_preprocess_walker(wn,
            WN_st(wn), ST_IDX_ZERO, parent_kernel_map);
}

#endif  // HICUDA
/*** DAVID CODE END ***/
