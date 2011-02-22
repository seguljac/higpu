/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_GPU_DATA_PROP_H_
#define _IPA_HC_GPU_DATA_PROP_H_

/*****************************************************************************
 *
 * Interprocedural hiCUDA Data Directive Propagation
 *
 * It uses the existing IPA_DATA_FLOW framework.
 *
 * During the initialization stage:
 * - Each call in each node is tagged with locally visible data directives.
 * - An initial list of IPA_HC_ANNOT is added to each successor of the
 *   root node.
 *
 * During the MEET stage:
 * - Do nothing
 *
 * During the TRANSFER stage:
 * - Go through each IPA_HC_ANNOT in each node, combine it with each
 *   call's local info and update the callee's IPA_HC_ANNOT
 *
 * The annotation of a procedure is changed iff there is a new
 * HC_FORMAL_GPU_DATA_ARRAY. Two HC_FORMAL_GPU_DATA_ARRAYs are the same iff
 * the GPU-allocated region for each formal matches.
 *
 * TODO: we could limit the number of clones per procedure.
 *
 * Each GPU data annotation of a procedure consists of the GPU data info for
 * each formal. We need to create a procedure clone for each annotation. In
 * order to link these clone properly, we also keep track of the calling
 * context in the annotation.
 *
 ****************************************************************************/

#include "defs.h"
#include "wn.h"

#include "cxx_base.h"
#include "cxx_template.h"
#include "cxx_hash.h"

#include "ip_graph.h"
#include "ipa_df.h"
#include "ipa_cg.h"
#include "ipa_hc_gpu_data.h"
#include "ipa_hc_annot.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

class HC_FORMAL_GPU_DATA_ARRAY : public HC_ANNOT_DATA
{
private:

    UINT _n_formals;

    HC_GPU_DATA **_formal_data;

    // formal flags
#define HC_FGDA_DATA_USED   0x02    // it this formal's data used?
                                    // set during back-propagation

    UINT *_formal_flags;

    MEM_POOL *_pool;

public:

    HC_FORMAL_GPU_DATA_ARRAY(UINT n_formals, MEM_POOL *pool)
    {
        Is_True(pool != NULL, (""));
        _pool = pool;
        _n_formals = n_formals;

        _formal_data = CXX_NEW_ARRAY(HC_GPU_DATA*, n_formals, pool);
        _formal_flags = CXX_NEW_ARRAY(UINT, n_formals, pool);
        for (UINT i = 0; i < n_formals; ++i) {
            _formal_data[i] = NULL;
            _formal_flags[i] = 0;
        }
    }

    virtual ~HC_FORMAL_GPU_DATA_ARRAY() {}

    UINT num_formals() const { return _n_formals; }

    HC_GPU_DATA* get_formal_data(UINT idx) const {
        Is_True(idx < _n_formals, (""));
        return _formal_data[idx];
    }
    void set_formal_data(UINT idx, HC_GPU_DATA *gdata) {
        Is_True(idx < _n_formals, (""));
        _formal_data[idx] = gdata;
    }

    BOOL is_formal_data_used(UINT idx) const {
        Is_True(idx < _n_formals, (""));
        return (_formal_flags[idx] & HC_FGDA_DATA_USED);
    }

    /**
     * Return whether or not the USED flag has been changed.
     */
    BOOL set_formal_data_used(UINT idx);

    /**
     * Set the USED flag for the formal that is associated with the given
     * GPU data, and return the formal index (or -1 if not found).
     */
    INT set_formal_data_used(const HC_GPU_DATA *gdata);

    void map_to_callee(IPA_EDGE *e);

    virtual BOOL is_dummy() const;

    /**
     * ASSUME: both annotation data are in the same procedure context, which
     * must have been set up before calling this function.
     */
    virtual BOOL equals(const HC_ANNOT_DATA *other) const;

    /**
     * ASSUME: the appropriate procedure context must be set up before calling
     * this function.
     */
    virtual void print(FILE *fp) const;
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

extern MEM_POOL Ipa_gpu_data_prop_pool;

class IPA_HC_GPU_DATA_PROP_DF : public IPA_DATA_FLOW
{
private:

    void set_edge_gdata(IPA_EDGE *e, HC_GPU_DATA_STACK *stack);

    void construct_local_gpu_data_annot(WN *wn,
            IPA_NODE *node, WN_TO_EDGE_MAP *wte_map,
            HC_GPU_DATA_LIST *gdata_list, UINT& dir_id,
            HC_GPU_DATA_STACK *stack);

    void rebuild_edge_gdata_walker(WN *wn,
            IPA_NODE *node, WN_TO_EDGE_MAP *wte_map,
            HC_GPU_DATA_LIST *gdata_list, UINT& gdata_dir_id,
            HC_GPU_DATA_STACK *stack);

    BOOL backprop_used_flag(IPA_NODE *node);

    static void build_wn_map(HC_WN_MAP *ww_map,
            IPA_NODE *from_node, IPA_NODE *to_node);

    void clone_hc_info(IPA_NODE *orig, IPA_NODE *clone);

    void expand_formals(IPA_NODE *node);
    void expand_actuals(IPA_NODE *node);

protected:

    virtual void* Meet(void* in, void* vertex, INT *change);
    virtual void* Trans(void* in, void* out, void* vertex, INT *change);

public:

    IPA_HC_GPU_DATA_PROP_DF(MEM_POOL *pool);

    virtual void InitializeNode(void *n);
    virtual void Print_entry(FILE *fp, void* out, void* n);
    virtual void PostProcessIO(void *);

    /**
     * - Do a simple match between visible GPU data and kernel regions within
     *   each procedure, and mark "used" flags in each
     *   HC_FORMAL_GPU_DATA_ARRAY.
     *
     * - Back-propagate the "used" flags and simplify annotations.
     *
     * - Create clones and fix the edges in the call graph.
     */
    void PostProcess();
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/* external interface for IPL */

/*****************************************************************************
 *
 * For each GPU data annotation in the node, match the visible GPU data with
 * each kernel's DAS and report any coverage problems.
 *
 * After calling this function, the data accesses in each kernel have the
 * corresponding HC_GPU_DATA cached in HC_ACCESS_INFO (for each annotation).
 *
 * Th HC_GPU_DATA used are marked properly in the HC_FORMAL_GPU_DATA_ARRAY.
 *
 * This function must be called inside IPL, so it DOES NOT establish the
 * node's context again.
 *
 * This function needs the procedure WN node separately because
 * node->Whirl_Tree() could be outdated.
 *
 ****************************************************************************/

extern void IPA_HC_match_gpu_data_with_kernel_das(WN *func_wn, IPA_NODE *node,
        MEM_POOL *tmp_pool);

#endif  // _IPA_HC_GPU_DATA_PROP_H_

/*** DAVID CODE END ***/
