/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_KERNEL_CONTEXT_PROP_H_
#define _IPA_HC_KERNEL_CONTEXT_PROP_H_

/*****************************************************************************
 *
 * Interprocedural Kernel Context Propagation
 *
 * The procedure nodes involved are CONTAINS_KERNEL nodes and their successors
 * (which must be MAY_BE_INSIDE_KERNEL).
 *
 * During initialization, each procedure's LOOP_PARTITION directives are
 * parsed, and initial kernel context annotation is constructed for all direct
 * successors of CONTAINS_KERNEL nodes (with kernel regions).
 *
 * During TRANSFER operation, the kernel context is propagated among
 * MAY_BE_INSIDE_KERNEL nodes. Note that the kernel context for each
 * LOOP_PARTITION directive is NOT filled.
 *
 * TODO: support non-const grid/block dimensions
 *
 ****************************************************************************/

extern MEM_POOL Ipa_kernel_prop_pool;

class IPA_HC_KERNEL_CONTEXT_PROP_DF : public IPA_DATA_FLOW
{
private:

    void construct_kernel_context_annot(WN *wn, IPA_NODE *node,
            WN_TO_EDGE_MAP *wte_map, HC_KERNEL_CONTEXT *kcontext);

protected: 
    virtual void* Meet(void* in, void* vertex, INT *change);
    virtual void* Trans(void* in, void* out, void* vertex, INT *change);

public:

    IPA_HC_KERNEL_CONTEXT_PROP_DF(MEM_POOL *pool);

    virtual void InitializeNode(void *n);
    virtual void Print_entry(FILE *fp, void* out, void* n);
    virtual void PostProcessIO(void *);

    void PostProcess();
};

#endif  // _IPA_HC_KERNEL_CONTEXT_PROP_H_

/*** DAVID CODE END ***/
