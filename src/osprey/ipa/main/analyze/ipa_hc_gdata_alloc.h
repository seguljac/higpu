/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_GDATA_ALLOC_H_
#define _IPA_HC_GDATA_ALLOC_H_

/*****************************************************************************
 *
 * Data structure and routines that are used to determine an optimal scheme
 * for allocating constant and shared memory
 *
 ****************************************************************************/

#include "defs.h"
#include "mempool.h"
#include "cxx_template.h"

#include "hc_ic_solver.h"   // HC_IG_NODE_INFO

class IPA_NODE;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Each HC_GPU_DATA has such an instance.
 *
 ****************************************************************************/

typedef DYN_ARRAY<IPA_NODE*> HC_PROC_CALL_LIST;

class HC_GDATA_IG_NODE : public HC_IG_NODE_INFO
{
private:

    HC_PROC_CALL_LIST *_calls;

    MEM_POOL *_pool;

public:

    HC_GDATA_IG_NODE(UINT size_in_bytes, MEM_POOL *pool)
        : HC_IG_NODE_INFO(size_in_bytes)
    {
        Is_True(pool != NULL, (""));
        _pool = pool;

        _calls = NULL;
    }
    ~HC_GDATA_IG_NODE() {}

    // Turn the GPU data size into a multiple of the largest element size, so
    // that it can be used as a "weight" in interval coloring.
    void normalize_size(UINT elem_sz)
    {
        _weight = (_weight + elem_sz - 1)/elem_sz;
    }

    void add_proc_call(IPA_NODE *node)
    {
        if (_calls == NULL) _calls = CXX_NEW(HC_PROC_CALL_LIST(_pool), _pool);
        _calls->AddElement(node);
    }
    UINT num_proc_calls()
    {
        return (_calls == NULL) ? 0 : _calls->Elements();
    }
    IPA_NODE* get_proc_call(UINT idx)
    {
        Is_True(idx < num_proc_calls(), (""));
        return (*_calls)[idx];
    }

    virtual BOOL equals(const HC_IG_NODE_INFO& other) const
    {
        return (this == (const HC_GDATA_IG_NODE*)&other);
    }
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Allocate the global <cmem> variable and determine the offset of each
 * CONSTANT directive, which is stored in the corresponding HC_GPU_VAR_INFO.
 *
 * NOTE: HC_GPU_VAR_INFO is assumed to be created already.
 *
 ****************************************************************************/

extern void IPA_HC_alloc_const_mem();

/*****************************************************************************
 *
 * Allocate the global <smem> variable and determine the offset of each
 * SHARED directive, which is stored in the corresponding HC_GPU_VAR_INFO.
 *
 * NOTE: HC_GPU_VAR_INFO is assumed to be created already.
 *
 ****************************************************************************/

extern void IPA_HC_alloc_shared_mem();

#endif  // _IPA_HC_GDATA_ALLOC_H_

/*** DAVID CODE END ***/
