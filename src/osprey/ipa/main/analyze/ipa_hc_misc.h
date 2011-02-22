/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_MISC_H_
#define _IPA_HC_MISC_H_

/*****************************************************************************
 *
 * Miscellaneous features in the hiCUDA phases
 *
 ****************************************************************************/

#include "wn.h"

class IPA_NODE;

/*****************************************************************************
 *
 * Local variables are created in CUDA code generation. Some of these
 * variables can be reused in multiple generation sites. For example, the
 * <batsz>, <g_stride> and <h_stride> variables used in generating data
 * transfer code can be used in both copyin and copyout.
 *
 * We cache these variable symbols in this per-procedure instance.
 *
 ****************************************************************************/

class HC_LOCAL_VAR_STORE
{
private:

    IPA_NODE *_proc_node;

    // a list of loop index variables (ordered by nesting level)
    ST_IDX *_loop_idx_vars;
    UINT _n_nesting_levels;

    // variables used in handling loop_partition directives
    ST_IDX _tripcount_st_idx;
    ST_IDX _tblk_stride_st_idx;

    // variables used in data transfer code
    ST_IDX _batsz_st_idx;
    ST_IDX _stride_st_idx;
    ST_IDX _goffset_st_idx;
    ST_IDX _hoffset_st_idx;

    // variables used in kernel execution configuration
    ST_IDX _grid_dim_st_idx;
    ST_IDX _tblk_dim_st_idx;

    // variables used in smem data transfer
    ST_IDX _cs_sz_st_idx;
    ST_IDX _gcs_sz_st_idx;
    ST_IDX _gcs_ofst_st_idx;
    ST_IDX _scs_sz_st_idx;
    ST_IDX _scs_ofst_st_idx;
    ST_IDX _n_segs_per_cs_st_idx;
    ST_IDX _n_segs_st_idx;
    ST_IDX _warp_id_st_idx;
    ST_IDX _id_within_warp_st_idx;
    // more (used within the loop)
    ST_IDX _cs_id_st_idx;
    ST_IDX _seg_id_in_cs_st_idx;
    ST_IDX _g_cs_ofst_st_idx;
    ST_IDX _s_cs_ofst_st_idx;
    ST_IDX _thr_ofst_st_idx;

    MEM_POOL *_pool;

    ST_IDX get_sym(ST_IDX& st_idx, const char *st_name, TY_IDX ty_idx);

public:

    HC_LOCAL_VAR_STORE(IPA_NODE *node, MEM_POOL *pool)
    {
        Is_True(pool != NULL, (""));
        _pool = pool;
        Is_True(node != NULL, (""));
        _proc_node = node;

        _loop_idx_vars = NULL;
        _n_nesting_levels = 0;

        _tripcount_st_idx = ST_IDX_ZERO;
        _tblk_stride_st_idx = ST_IDX_ZERO;

        _batsz_st_idx = ST_IDX_ZERO;
        _stride_st_idx = ST_IDX_ZERO;
        _goffset_st_idx = ST_IDX_ZERO;
        _hoffset_st_idx = ST_IDX_ZERO;

        _grid_dim_st_idx = ST_IDX_ZERO;
        _tblk_dim_st_idx = ST_IDX_ZERO;

        _cs_sz_st_idx = ST_IDX_ZERO;
        _gcs_sz_st_idx = ST_IDX_ZERO;
        _gcs_ofst_st_idx = ST_IDX_ZERO;
        _scs_sz_st_idx = ST_IDX_ZERO;
        _scs_ofst_st_idx = ST_IDX_ZERO;
        _n_segs_per_cs_st_idx = ST_IDX_ZERO;
        _n_segs_st_idx = ST_IDX_ZERO;
        _warp_id_st_idx = ST_IDX_ZERO;
        _id_within_warp_st_idx = ST_IDX_ZERO;

        _cs_id_st_idx = ST_IDX_ZERO;
        _seg_id_in_cs_st_idx = ST_IDX_ZERO;
        _g_cs_ofst_st_idx = ST_IDX_ZERO;
        _s_cs_ofst_st_idx = ST_IDX_ZERO;
        _thr_ofst_st_idx = ST_IDX_ZERO;
    }

    ~HC_LOCAL_VAR_STORE() {}

    // zero-based nesting level
    ST_IDX get_loop_idx_var(UINT nesting_level);

    ST_IDX get_tripcount()
    {
        return get_sym(_tripcount_st_idx,
                "tripcount", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_tblock_stride()
    {
        return get_sym(_tblk_stride_st_idx,
                "tblk_stride", MTYPE_To_TY(Integer_type));
    }

    ST_IDX get_batsz_sym()
    {
        return get_sym(_batsz_st_idx, "batsz", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_stride_sym()
    {
        return get_sym(_stride_st_idx, "stride", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_goffset_sym()
    {
        return get_sym(_goffset_st_idx, "goffset", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_hoffset_sym()
    {
        return get_sym(_hoffset_st_idx, "hoffset", MTYPE_To_TY(Integer_type));
    }

    ST_IDX get_grid_dim_sym();
    ST_IDX get_tblk_dim_sym();

    ST_IDX get_cs_sz_sym()
    {
        return get_sym(_cs_sz_st_idx, "cs_sz", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_gcs_sz_sym()
    {
        return get_sym(_gcs_sz_st_idx, "gcs_sz", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_gcs_ofst_sym()
    {
        return get_sym(_gcs_ofst_st_idx, "gcs_ofst",
                MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_scs_sz_sym()
    {
        return get_sym(_scs_sz_st_idx, "scs_sz", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_scs_ofst_sym()
    {
        return get_sym(_scs_ofst_st_idx, "scs_ofst",
                MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_n_segs_per_cs_sym()
    {
        return get_sym(_n_segs_per_cs_st_idx, "n_segs_per_cs",
                MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_n_segs_sym()
    {
        return get_sym(_n_segs_st_idx, "n_segs", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_warp_id_sym()
    {
        return get_sym(_warp_id_st_idx, "warp_id", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_id_within_warp_sym()
    {
        return get_sym(_id_within_warp_st_idx, "id_within_warp",
                MTYPE_To_TY(Integer_type));
    }

    ST_IDX get_cs_id_sym()
    {
        return get_sym(_cs_id_st_idx, "cs_id", MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_seg_id_in_cs_sym()
    {
        return get_sym(_seg_id_in_cs_st_idx, "seg_id_in_cs",
                MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_g_cs_ofst_sym()
    {
        return get_sym(_g_cs_ofst_st_idx, "g_cs_ofst",
                MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_s_cs_ofst_sym()
    {
        return get_sym(_s_cs_ofst_st_idx, "s_cs_ofst",
                MTYPE_To_TY(Integer_type));
    }
    ST_IDX get_thr_ofst_sym()
    {
        return get_sym(_thr_ofst_st_idx, "thr_ofst",
                MTYPE_To_TY(Integer_type));
    }
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Global variables that are created in CUDA code generation.
 *
 ****************************************************************************/

class HC_GLOBAL_VAR_STORE
{
private:

    // the single global constant memory variable
    ST_IDX _cmem_st_idx;

    // the single global shared memory variable
    ST_IDX _smem_st_idx;

public:

    HC_GLOBAL_VAR_STORE()
    {
        _cmem_st_idx = ST_IDX_ZERO;
    }
    ~HC_GLOBAL_VAR_STORE() {}

    ST_IDX create_cmem_sym(UINT n_elems, UINT elem_sz);

    ST_IDX get_cmem_sym() const
    {
        Is_True(_cmem_st_idx != ST_IDX_ZERO, (""));
        return _cmem_st_idx;
    }

    ST_IDX create_smem_sym(UINT elem_sz);

    ST_IDX get_smem_sym() const
    {
        Is_True(_smem_st_idx != ST_IDX_ZERO, (""));
        return _smem_st_idx;
    }
};

extern HC_GLOBAL_VAR_STORE hc_glob_var_store;

#endif  // _IPA_HC_MISC_H_

/*** DAVID CODE END ***/
