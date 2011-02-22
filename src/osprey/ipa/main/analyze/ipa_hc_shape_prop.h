/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_SHAPE_PROP_H_
#define _IPA_HC_SHAPE_PROP_H_

/*****************************************************************************
 *
 * Interprocedural Array Shape Propagation
 *
 * It uses the existing IPA_DATA_FLOW framework. The annotation is
 * IPA_HC_ANNOT, with data type HC_FORMAL_SHAPE_ARRAY.
 *
 * During the initialization stage:
 * - Each call in each node is tagged with locally visible shapes.
 * - A hashtable for <HC_FORMAL_SHAPE_ARRAY, IPA_CALL_CONTEXT> is alloated in
 *   each node.
 *
 * During the MEET stage:
 * - Do nothing
 *
 * During the TRANSFER stage:
 * - Go through each IPA_HC_ANNOT in each node, combine it with each call's
 *   local info and update the callee's IPA_HC_ANNOT
 *
 * The IPA_HC_ANNOT is changed iff there is a new HC_FORMAL_SHAPE_ARRAY. Two
 * HC_FORMAL_SHAPE_ARRAYs are the same iff the shape for each formal matches.
 *
 * TODO: we could reduce the number of clones by giving a more relaxed defn of
 * "equivalent" HC_FORMAL_SHAPE_ARRAYs: matched formal shapes for those that
 * will affect the pointer promotion result.
 *
 * TODO: we could limit the number of clones per procedure.
 *
 * Each shape annotation of a procedure consists of the shape info for each
 * formal. We need to create a procedure clone for each shape annotation. In
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
#include "ipa_hc_shape.h"
#include "ipa_hc_annot.h"


class HC_FORMAL_SHAPE_ARRAY : public HC_ANNOT_DATA
{
private:

    HC_ARRAY_SHAPE_INFO **_formal_shapes;
    UINT _n_formals;

    MEM_POOL *_pool;

public:

    HC_FORMAL_SHAPE_ARRAY(UINT n_formals, MEM_POOL *pool)
    {
        _pool = pool;
        _n_formals = n_formals;
        _formal_shapes = CXX_NEW_ARRAY(HC_ARRAY_SHAPE_INFO*, n_formals, pool);
        for (UINT i = 0; i < n_formals; ++i) _formal_shapes[i] = NULL;
    }

    ~HC_FORMAL_SHAPE_ARRAY() {}

    UINT num_formals() { return _n_formals; }

    void set_formal_shape(UINT idx, HC_ARRAY_SHAPE_INFO *shape) {
        Is_True(idx < _n_formals, (""));
        _formal_shapes[idx] = shape;
    }
    HC_ARRAY_SHAPE_INFO* get_formal_shape(UINT idx) {
        Is_True(idx < _n_formals, (""));
        return _formal_shapes[idx];
    }

    void map_to_callee(IPA_EDGE *e);

    virtual BOOL is_dummy() const;

    virtual BOOL equals(const HC_ANNOT_DATA *other) const;

    virtual void print(FILE *fp) const;
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

extern MEM_POOL Ipa_shape_prop_pool;

class IPA_HC_SHAPE_PROP_DF : public IPA_DATA_FLOW
{
private:

    void construct_local_shape_annot(WN *wn, WN_TO_EDGE_MAP *wte_map,
            HC_SHAPE_INFO_LIST *shapes, HC_ARRAY_SHAPE_CONTEXT *context,
            HC_SHAPE_INFO_MAP *arr_shape_map);

protected:

    virtual void* Meet(void* in, void* vertex, INT *change);
    virtual void* Trans(void* in, void* out, void* vertex, INT *change);

public:

    IPA_HC_SHAPE_PROP_DF(MEM_POOL *pool);

    virtual void InitializeNode(void *n);
    virtual void Print_entry(FILE *fp, void* out, void* n);
    virtual void PostProcessIO(void *);

    void PostProcess();
};

#endif  // _IPA_HC_SHAPE_PROP_H_

/*** DAVID CODE END ***/
