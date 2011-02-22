/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_ACCESS_REDIRECTION_H_
#define _IPA_HC_ACCESS_REDIRECTION_H_

/*****************************************************************************
 *
 * Inter-procedural Access Redirection
 *
 * The GPU data to be propagated is HC_GPU_DATA (ipa_hc_gpu_data.h). The
 * propagation is among IK- and K-procedures.
 *
 * ASSUME: list of HC_KERNEL_INFOs and updated parent kernel symbol annotation
 * in each edge.
 *
 * The initialization of each MAY_BE_INSIDE_KERNEL node involves go through
 * each incoming edge that is within a kernel region and construct visible GPU
 * data annotation.
 *
 * The TRANSFER operation simply involves propagating each NEW annotation to
 * every outgoing call edge that reaches a MAY_BE_INSIDE_KERNEL node.
 *
 ****************************************************************************/

#include "defs.h"
#include "wn.h"

#include "cxx_base.h"
#include "cxx_template.h"
#include "cxx_hash.h"

#include "ip_graph.h"
#include "ipa_df.h"

#include "ipa_hc_common.h"      // HC_SYM_MAP
#include "ipa_hc_annot.h"
#include "ipa_hc_gpu_data.h"

/*****************************************************************************
 *
 * This is different from HC_FORMAL_GPU_DATA_ARRAY, in that:
 * - It has one HC_GPU_DATA per global symbol accessed.
 *
 ****************************************************************************/

typedef HASH_TABLE<ST_IDX, HC_GPU_DATA*> GLOBAL_GPU_DATA_TABLE;
typedef HASH_TABLE_ITER<ST_IDX, HC_GPU_DATA*> GLOBAL_GPU_DATA_ITER;
typedef HASH_TABLE<ST_IDX,ST*> GDATA_ST_TABLE;

typedef HASH_TABLE<HC_GPU_DATA*,HC_GPU_DATA*> SDATA_TABLE;
typedef HASH_TABLE_ITER<HC_GPU_DATA*,HC_GPU_DATA*> SDATA_TABLE_ITER;

class HC_FORMAL_GPU_VAR_ARRAY : public HC_ANNOT_DATA
{
private:

    HC_GPU_DATA **_formal_data;
    UINT _n_formals;

    GLOBAL_GPU_DATA_TABLE *_global_data;

    // symbol referenced in HC_GPU_DATAs
    GDATA_ST_TABLE *_st_table;

    // symbol in HC_GPU_DATA's alloc sections to newly created formals
    HC_SYM_MAP *_idxv_sym_map;

    // a mapping from GLOBAL data to SHARED data
    // For now, we have no way of telling which entries are new.
    SDATA_TABLE *_gsdata_map;
    BOOL _gsdata_map_changed;

    MEM_POOL *_pool;

public:

    HC_FORMAL_GPU_VAR_ARRAY(UINT n_formals, MEM_POOL *pool)
    {
        _pool = pool;

        _n_formals = n_formals;
        _formal_data = CXX_NEW_ARRAY(HC_GPU_DATA*, n_formals, pool);
        for (UINT i = 0; i < n_formals; ++i) _formal_data[i] = NULL;

        _global_data = CXX_NEW(GLOBAL_GPU_DATA_TABLE(41,_pool), _pool);

        _st_table = NULL;
        _idxv_sym_map = NULL;

        _gsdata_map = NULL;
        _gsdata_map_changed = FALSE;
    }

    ~HC_FORMAL_GPU_VAR_ARRAY() {}

    HC_GPU_DATA* get_formal_data(UINT idx) const
    {
        Is_True(idx < _n_formals, (""));
        return _formal_data[idx];
    }
    void set_formal_data(UINT idx, HC_GPU_DATA *gdata)
    {
        Is_True(idx < _n_formals, (""));
        _formal_data[idx] = gdata;
    }

    UINT num_formals() const { return _n_formals; }

    GLOBAL_GPU_DATA_TABLE* get_global_data_table() const
    {
        return _global_data;
    }
    HC_GPU_DATA* get_global_data(ST_IDX st_idx) const
    {
        return _global_data->Find(st_idx);
    }
    void set_global_data(ST_IDX st_idx, HC_GPU_DATA *gdata)
    {
        _global_data->Enter(st_idx, gdata);
    }

    void set_st_table(GDATA_ST_TABLE *tbl) { _st_table = tbl; }
    GDATA_ST_TABLE* get_st_table() { return _st_table; }

    void set_idxv_sym_map(HC_SYM_MAP *map) { _idxv_sym_map = map; }
    HC_SYM_MAP* get_idxv_sym_map() { return _idxv_sym_map; }

    SDATA_TABLE* get_sdata_table();
    void set_sdata_table(SDATA_TABLE *map);
    // Return the SHARED data corresponding to the given GLOBAL data.
    HC_GPU_DATA* get_sdata_alias(HC_GPU_DATA *gdata) const;
    // The given GLOBAL data must not have been associated with a SHARED data.
    void add_gsdata_alias(HC_GPU_DATA *gdata, HC_GPU_DATA *sdata);

    BOOL has_gsdata_map_changed() const { return _gsdata_map_changed; }
    void reset_gsdata_map_changed() { _gsdata_map_changed = FALSE; }

    HC_GPU_DATA* search(WN *func_wn, ST_IDX st_idx) const;

    virtual BOOL is_dummy() const;

    virtual BOOL equals(const HC_ANNOT_DATA *other) const;

    virtual void print(FILE *fp) const;
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

class IPA_HC_GPU_VAR_PROP_DF : public IPA_DATA_FLOW
{
private:

    HC_FORMAL_GPU_VAR_ARRAY* construct_callee_gpu_var_annot(IPA_EDGE *e);

    HC_FORMAL_GPU_VAR_ARRAY* propagate_gpu_var_annot(
            HC_FORMAL_GPU_VAR_ARRAY *caller_fgva, IPA_EDGE *e);

    void map_gpu_var_annot_to_callee(IPA_NODE *callee);

    void expand_formals(IPA_NODE *node);
    void expand_actuals(IPA_NODE *node);

protected:

    virtual void* Meet(void* in, void* vertex, INT *change);
    virtual void* Trans(void* in, void* out, void* vertex, INT *change);

public:

    IPA_HC_GPU_VAR_PROP_DF(MEM_POOL *pool);

    virtual void InitializeNode(void *n);
    virtual void Print_entry(FILE *fp, void* out, void* n);
    virtual void PostProcessIO(void *);

    /**
     * Create clones and fix call edges.
     */
    void PostProcess();
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

extern void HC_build_edge_kernel_annot(IPA_NODE *node, MEM_POOL *tmp_pool);

#endif  // _IPA_HC_ACCESS_REDIRECTION_H_

/*** DAVID CODE END ***/
