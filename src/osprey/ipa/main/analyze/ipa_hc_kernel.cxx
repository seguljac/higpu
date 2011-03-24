/** DAVID CODE BEGIN **/

// needed by ipl_summary.h included in ipa_cg.h
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "tracing.h"        // TDEBUG_HICUDA
#include "config_ipa.h"     // IPA_Disable_Statics_Promotion
#include "wn.h"
#include "wn_util.h"
#include "ir_reader.h"

#include "ipa_cg.h"
#include "ipa_hc_common.h"  // HC_SYM_MAP
#include "ipa_hc_kernel.h"
#include "ipa_hc_gpu_data_prop.h"

#include "hc_common.h"
#include "hc_utils.h"
#include "hc_kernel.h"
#include "hc_expr.h"

#ifdef IPA_HICUDA
#include "cuda_utils.h"
#include "hc_gpu_data.h"

#include "ipo_defs.h"       // IPA_NODE_CONTEXT
#include "ipo_lwn_util.h"   // LWN_Get_Parent
#include "ipo_clone.h"      // IPO_Clone
#endif  // IPA_HICUDA

extern BOOL flag_opencl;
// Needed by OpenCl, New_Const_Sym
#include "const.h"
// Needed for OpenCL global variables
#include "ipa_hc_misc.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void HC_ARRAY_INFO::print(FILE *fp)
{
    fprintf(fp, "(REF: ");
    if (_ref_region == NULL) {
        fprintf(fp, "<null> ");
    } else {
        _ref_region->Print(fp);
    }
    fprintf(fp, ", MOD: ");
    if (_mod_region == NULL) {
        fprintf(fp, "<null> ");
    } else {
        _mod_region->Print(fp);
    }
    fprintf(fp, ")\n");
}

void HC_ARRAY_SYM_INFO::add_access(WN *access_wn, INT actual_idx,
        PROJECTED_REGION *ref_pr, PROJECTED_REGION *mod_pr)
{
    HC_ARRAY_INFO_ITER haii(&_arr_info);
    HC_ARRAY_INFO *hai;

    // Is there such an access in the record already?
    for (hai = haii.First(); hai != NULL; hai = haii.Next())
    {
        if (hai->get_wn() == access_wn
                && hai->get_actual_index() == actual_idx) break;
    }

    if (hai != NULL)
    {
        // Update the ref/mod regions.
        hai->update_regions(ref_pr, mod_pr);
    }
    else
    {
        // Add a new entry to the record.
        OPERATOR opr = WN_operator(access_wn);
        hai = (opr == OPR_ILOAD || opr == OPR_ISTORE) ?
            CXX_NEW(HC_ARRAY_INFO(access_wn, ref_pr, mod_pr), _mem_pool) :
            CXX_NEW(HC_ARRAY_INFO(access_wn, actual_idx, ref_pr, mod_pr),
                    _mem_pool);
        _arr_info.Append(hai);
    }
}

void HC_ARRAY_SYM_INFO::print(FILE *fp, INT indent)
{
    const char *arr_st_name = ST_name(_arr_st_idx);

    // Go through each HC_ARRAY_INFO.
    HC_ARRAY_INFO_ITER haii(&_arr_info);
    for (HC_ARRAY_INFO *hai = haii.First(); !haii.Is_Empty();
            hai = haii.Next())
    {
        fprintf(fp, "%s", arr_st_name);
        hai->print(fp);

        WN *wn = hai->get_wn();
        INT actual_idx = hai->get_actual_index();

        fprintf(fp, " FROM (%p) ", wn);
        fdump_wn(fp, wn);

        if (WN_operator(wn) == OPR_CALL) {
            if (actual_idx < 0) {
                fprintf(fp, " GLOBAL");
            } else {
                fprintf(fp, " PARM #%d", actual_idx);
            }
        }
        fprintf(fp, "\n");
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

HC_KERNEL_INFO::HC_KERNEL_INFO(WN *kregion, MEM_POOL *pool)
{
    Is_True(pool != NULL, (""));
    _pool = pool;

    Is_True(kregion != NULL && WN_operator(kregion) == OPR_REGION
            && WN_region_kind(kregion) == REGION_KIND_HICUDA,
            ("HC_KERNEL_INFO:: not a hiCUDA REGION\n"));

    // Get the kernel pragma.
    WN *pragma_wn = WN_first(WN_kid1(kregion));
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_PRAGMA
            && (WN_PRAGMA_ID)WN_pragma(pragma_wn) == WN_PRAGMA_HC_KERNEL,
            ("HC_KERNEL_INFO: invalid KERNEL pragma\n"));

    // Get the kernel function symbol.
    _kfunc_st_idx = WN_st_idx(pragma_wn);

    // Determine the dimensionality of the virtual tblock and thread space.
    _n_vgrid_dims = WN_pragma_arg1(pragma_wn);
    _n_vblk_dims = WN_pragma_arg2(pragma_wn);

    // Fill the dimension sizes of the tblock space.
    _vgrid_dims = CXX_NEW_ARRAY(WN*, _n_vgrid_dims, pool);
    for (UINT i = 0; i < _n_vgrid_dims; ++i)
    {
        pragma_wn = WN_next(pragma_wn);
        Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_XPRAGMA,
                ("HC_KERNEL_INFO: invalid KERNEL thread block space\n"));
        _vgrid_dims[i] = WN_COPY_Tree(WN_kid0(pragma_wn));
    }

    // Fill the dimension sizes of the thread space.
    _vblk_dims = CXX_NEW_ARRAY(WN*, _n_vblk_dims, pool);
    for (UINT i = 0; i < _n_vblk_dims; ++i)
    {
        pragma_wn = WN_next(pragma_wn);
        Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_XPRAGMA,
                ("HC_KERNEL_INFO: invalid KERNEL thread space\n"));
        _vblk_dims[i] = WN_COPY_Tree(WN_kid0(pragma_wn));
    }

    Is_True(WN_next(pragma_wn) == NULL,
            ("HC_KERNEL_INFO: more KERNEL pragmas??\n"));

    _is_proc_called = NULL; _n_procs = 0;
    _sdata_list = NULL;
    _smem_size = 0;

    _kparams = NULL;

    _vblk_idx = _vthr_idx = NULL;
    _flags = 0;
    _knode = NULL;
}

HC_KERNEL_INFO::HC_KERNEL_INFO(HC_KERNEL_INFO *orig,
        const HC_WN_MAP *ww_map, MEM_POOL *pool)
{
    Is_True(pool != NULL, (""));
    _pool = pool;

    _kfunc_st_idx = orig->_kfunc_st_idx;
    _flags = orig->_flags;
    _knode = orig->_knode;

    _is_proc_called = NULL; _n_procs = 0;
    Is_True(orig->_is_proc_called == NULL && orig->_n_procs == 0, (""));
    _sdata_list = NULL;
    Is_True(orig->_sdata_list == NULL, (""));
    _smem_size = 0;
    Is_True(orig->_smem_size == 0, (""));
    _kparams = NULL;
    Is_True(orig->_kparams == NULL, (""));

    // virtual thread block space
    _n_vgrid_dims = orig->_n_vgrid_dims;
    _vgrid_dims = CXX_NEW_ARRAY(WN*, _n_vgrid_dims, pool);
    for (UINT i = 0; i < _n_vgrid_dims; ++i) {
        _vgrid_dims[i] = WN_COPY_Tree(orig->_vgrid_dims[i]);
    }
    if (orig->_vblk_idx != NULL)
    {
        _vblk_idx = CXX_NEW_ARRAY(WN*, _n_vgrid_dims, pool);
        for (UINT i = 0; i < _n_vgrid_dims; ++i) {
            _vblk_idx[i] = WN_COPY_Tree(orig->_vblk_idx[i]);
        }
    }

    // virtual thread space
    _n_vblk_dims = orig->_n_vblk_dims;
    _vblk_dims = CXX_NEW_ARRAY(WN*, _n_vblk_dims, pool);
    for (UINT i = 0; i < _n_vblk_dims; ++i) {
        _vblk_dims[i] = WN_COPY_Tree(orig->_vblk_dims[i]);
    }
    if (orig->_vthr_idx != NULL)
    {
        _vthr_idx = CXX_NEW_ARRAY(WN*, _n_vblk_dims, pool);
        for (UINT i = 0; i < _n_vblk_dims; ++i) {
            _vthr_idx[i] = WN_COPY_Tree(orig->_vthr_idx[i]);
        }
    }

    // physical thread block space
    for (UINT i = 0; i < 2; ++i) {
        _grid_dims[i] = WN_COPY_Tree(orig->_grid_dims[i]);
    }

    // physical thread space
    for (UINT i = 0; i < 3; ++i) {
        _blk_dims[i] = WN_COPY_Tree(orig->_blk_dims[i]);
    }

    // scalar accesses
    HC_SCALAR_INFO_ITER orig_si_iter(&(orig->_scalar_access));
    for (HC_SCALAR_INFO *orig_si = orig_si_iter.First();
            !orig_si_iter.Is_Empty(); orig_si = orig_si_iter.Next())
    {
        _scalar_access.Append(CXX_NEW(HC_SCALAR_INFO(orig_si), pool));
    }

    // array section accesses
    HC_ARRAY_SYM_INFO_ITER orig_asi_iter(&(orig->_arr_access));
    for (HC_ARRAY_SYM_INFO *orig_asi = orig_asi_iter.First();
            !orig_asi_iter.Is_Empty(); orig_asi = orig_asi_iter.Next())
    {
        HC_ARRAY_SYM_INFO *asi = CXX_NEW(
                HC_ARRAY_SYM_INFO(orig_asi->arr_sym(),pool), pool);
        _arr_access.Append(asi);

        HC_ARRAY_INFO_LIST *ail = asi->get_arr_info_list();
        HC_ARRAY_INFO_ITER orig_ai_iter(orig_asi->get_arr_info_list());
        for (HC_ARRAY_INFO *orig_ai = orig_ai_iter.First();
                !orig_ai_iter.Is_Empty(); orig_ai = orig_ai_iter.Next())
        {
            ail->Append(CXX_NEW(HC_ARRAY_INFO(orig_ai,ww_map), pool));
        }
    }
}

void HC_KERNEL_INFO::finalize_gpu_data(IPA_HC_ANNOT *annot,
        const HC_GPU_DATA_MAP *gdata_map)
{
    // scalar accesses
    HC_SCALAR_INFO_ITER si_iter(&_scalar_access);
    for (HC_SCALAR_INFO *si = si_iter.First();
            !si_iter.Is_Empty(); si = si_iter.Next())
    {
        si->finalize_gpu_data(annot, gdata_map);
    }

    // array section accesses
    HC_ARRAY_SYM_INFO_ITER asi_iter(&_arr_access);
    for (HC_ARRAY_SYM_INFO *asi = asi_iter.First();
            !asi_iter.Is_Empty(); asi = asi_iter.Next())
    {
        HC_ARRAY_INFO_ITER ai_iter(asi->get_arr_info_list());
        for (HC_ARRAY_INFO *ai = ai_iter.First();
                !ai_iter.Is_Empty(); ai = ai_iter.Next())
        {
            ai->finalize_gpu_data(annot, gdata_map);
#if 0
            printf("Finalized HC_ARRAY_INFO %p\n", ai);
#endif
        }
    }
}

#ifdef IPA_HICUDA

void HC_KERNEL_INFO::process_grid_geometry()
{
    // Fill physical tblock and thread space dimensions, and construct
    // expressions for dimension index of virtual tblock/thread spaces in
    // terms of blockIdx and threadIdx.

    // The grid dimension is 2-D.
    _vblk_idx = CXX_NEW_ARRAY(WN*, _n_vgrid_dims, _pool);
    if (_n_vgrid_dims <= 2)
    {
        // The simple case
        UINT i = 0;
        for ( ; i < _n_vgrid_dims; ++i)
        {
            _grid_dims[_n_vgrid_dims-1-i] = WN_COPY_Tree(_vgrid_dims[i]);
            _vblk_idx[i] = ldid_blockIdx(_n_vgrid_dims-1-i);
        }
        for ( ; i < 3; ++i) _grid_dims[i] = WN_Intconst(Integer_type, 1);
    }
    else
    {
        WN* factor[_n_vgrid_dims];
        for (UINT i = 0; i < _n_vgrid_dims; ++i)
        {
            if (i <= 1) {
                _grid_dims[1-i] = WN_COPY_Tree(_vgrid_dims[i]);
            } else {
                _grid_dims[0] = WN_Mpy(Integer_type,
                    _grid_dims[0], WN_COPY_Tree(_vgrid_dims[i]));
            }

            if (i >= 1) {
                factor[i] = WN_Intconst(Integer_type, 1);
                for (UINT j = 1; j < i; ++j) {
                    factor[j] = WN_Mpy(Integer_type,
                            factor[j], WN_COPY_Tree(_vgrid_dims[i]));
                }
            }
        }
        _grid_dims[2] = WN_Intconst(Integer_type, 1);

        // Determine a mapping from the physical space to the virtual space.
        _vblk_idx[0] = ldid_blockIdx(1);
        _vblk_idx[1] = WN_Div(Integer_type, ldid_blockIdx(0), factor[1]);
        for (UINT i = 2; i < _n_vgrid_dims; ++i)
        {
            WN *wn = ldid_blockIdx(0);

            // Need to do MOD.
            wn = WN_Binary(OPR_REM, Integer_type,
                wn,
                WN_Mpy(Integer_type,
                    WN_COPY_Tree(factor[i]),
                    WN_COPY_Tree(_vgrid_dims[i])
                )
            );

            _vblk_idx[i] = WN_Div(Integer_type, wn, factor[i]);
        }
    }

    // The block dimension is 3-D.
    _vthr_idx = CXX_NEW_ARRAY(WN*, _n_vblk_dims, _pool);
    if (_n_vblk_dims <= 3)
    {
        // The simple case
        UINT i = 0;
        for ( ; i < _n_vblk_dims; ++i) {
            _blk_dims[_n_vblk_dims-1-i] = WN_COPY_Tree(_vblk_dims[i]);
            _vthr_idx[i] = ldid_threadIdx(_n_vblk_dims-1-i);
        }
        for ( ; i < 3; ++i) _blk_dims[i] = WN_Intconst(Integer_type, 1);
    }
    else
    {
        WN* factor[_n_vblk_dims];
        for (UINT i = 0; i < _n_vblk_dims; ++i)
        {
            if (i <= 2) {
                _blk_dims[2-i] = WN_COPY_Tree(_vblk_dims[i]);
            } else {
                _blk_dims[0] = WN_Mpy(Integer_type,
                    _blk_dims[0], WN_COPY_Tree(_vblk_dims[i]));
            }

            if (i >= 2) {
                factor[i] = WN_Intconst(Integer_type, 1);
                for (UINT j = 2; j < i; ++j) {
                    factor[j] = WN_Mpy(Integer_type,
                        factor[j], WN_COPY_Tree(_vblk_dims[i]));
                }
            }
        }

        // Determine a mapping from the physical space to the virtual space.
        _vthr_idx[0] = ldid_threadIdx(2);
        _vthr_idx[1] = ldid_threadIdx(1);
        _vthr_idx[2] = WN_Div(Integer_type, ldid_threadIdx(0), factor[2]);
        for (UINT i = 3; i < _n_vblk_dims; ++i)
        {
            WN *wn = ldid_threadIdx(0);

            // Need to do MOD.
            wn = WN_Binary(OPR_REM, Integer_type,
                wn,
                WN_Mpy(Integer_type,
                    WN_COPY_Tree(factor[i]),
                    WN_COPY_Tree(_vblk_dims[i])
                )
            );

            _vthr_idx[i] = WN_Div(Integer_type, wn, factor[i]);
        }
    }
}

/*****************************************************************************
 *
 * We assume that the (physical) block dimension sizes are constant.
 *
 * <warp_id> and <id_within_warp> are simply the quotient and remainder when
 * (threadIdx.x + Dx * threadIdx.y + * Dx * Dy * threadIdx.z) divided by
 * <warp_sz>.
 *
 * For each product term, we figure out its quotient and remainder by dividing
 * the coefficient by <warp_sz>. If this is not successful, we try a special
 * case matching: the coefficient divides <warp_sz>,
 *
 *      threadIdx.y * 8 => (threadIdx.y/4) * 32 + (threadIdx.y%4)*8
 *
 * This only simplifies the quotient term, not the remainder term. If this is
 * not successful either, the entire optimization process stops.
 *
 * Once the remainders for all product terms are collected and their ranges
 * are determined, we test if it remains to be a valid remainder, i.e. less
 * than <warp_sz>. If so, the optimized expression is generated; otherwise, we
 * fall back to the naive expressions.
 *
 ****************************************************************************/

void HC_KERNEL_INFO::gen_warp_info()
{
    if (_n_warps > 0) return;
    Is_True(_warp_id_wn == NULL && _id_within_warp_wn == NULL, (""));

    // For each physical dimension size of thread block, convert WN into UINT.
    UINT blk_dim_sz[3]; // x,y,z
    for (UINT i = 0; i < 3; ++i)
    {
        HC_assert(WN_operator(_blk_dims[i]) == OPR_INTCONST,
                ("Dimension #%u of the thread block space in kernel <%s> "
                 "does not have a constant size!",
                 i, ST_name(get_kernel_sym())));
        blk_dim_sz[i] = WN_const_val(_blk_dims[i]);
    }

    // Try generating an optimized version first.
    _warp_id_wn = WN_Intconst(Integer_type, 0);
    _id_within_warp_wn = WN_Intconst(Integer_type, 0);
    UINT coeff = 1;     // running coefficient for the current product term
    UINT rem_max = 0;   // max of the running sum of remainder terms
    BOOL opt_expr = TRUE;
    for (UINT i = 0; i < 3 && opt_expr; ++i)
    {
        // Do nothing for a unit dimension.
        if (blk_dim_sz[i] > 1)
        {
            // Divide the coefficient by the warp size.
            UINT q = coeff / _warp_sz;
            UINT r = coeff - q * _warp_sz;
            UINT r_max = r * (blk_dim_sz[i]-1);
            if (rem_max + r_max < _warp_sz)
            {
                // The remainder of this term looks good.
                rem_max += r_max;
                _warp_id_wn = WN_Add(Integer_type, _warp_id_wn,
                        WN_Mpy(Integer_type, ldid_threadIdx(i),
                            WN_Intconst(Integer_type, q)));
                _id_within_warp_wn = WN_Add(Integer_type, _id_within_warp_wn,
                        WN_Mpy(Integer_type, ldid_threadIdx(i),
                            WN_Intconst(Integer_type, r)));
            }
            else if (_warp_sz % coeff == 0)
            {
                // Special case: the coeff divides the warp size.
                UINT q = _warp_sz / coeff;
                UINT r_max = _warp_sz - coeff;
                if (rem_max + r_max < _warp_sz)
                {
                    // The remainder of this term looks good.
                    rem_max += r_max;
                    _warp_id_wn = WN_Add(Integer_type, _warp_id_wn,
                            WN_Div(Integer_type, ldid_threadIdx(i),
                                WN_Intconst(Integer_type, q)));
                    _id_within_warp_wn = WN_Add(Integer_type,
                            _id_within_warp_wn,
                            WN_Mpy(Integer_type,
                                WN_Binary(OPR_REM, Integer_type,
                                    ldid_threadIdx(i),
                                    WN_Intconst(Integer_type, q)),
                                WN_Intconst(Integer_type, coeff)));
                }
                else
                {
                    // Stop the optimization.
                    opt_expr = FALSE;
                }
            }
            else
            {
                // Stop the optimization.
                opt_expr = FALSE;
            }
        }

        coeff *= blk_dim_sz[i];
    }

    // Now <coeff> holds the total number of threads. Ensure that it is a
    // multiple of the warp size.
    HC_assert(coeff % _warp_sz == 0,
            ("The number of threads (in a thread block) that execute "
             "kernel <%s> is not a multiple of the warp size (%u)!",
             ST_name(get_kernel_sym()), _warp_sz));

    _n_warps = coeff/_warp_sz;

    if (!opt_expr)
    {
        // Generate the naive expressions.
        coeff = blk_dim_sz[0];
        WN *tid_wn = ldid_threadIdx(0);
        for (UINT i = 1; i < 3; ++i)
        {
            if (blk_dim_sz[i] > 1)
            {
                tid_wn = WN_Add(Integer_type, tid_wn,
                        WN_Mpy(Integer_type, ldid_threadIdx(i),
                            WN_Intconst(Integer_type, coeff)));
            }
            coeff *= blk_dim_sz[i];
        }

        WN_DELETE_Tree(_warp_id_wn);
        _warp_id_wn = WN_Div(Integer_type, WN_COPY_Tree(tid_wn),
                WN_Intconst(Integer_type, _warp_sz));

        WN_DELETE_Tree(_id_within_warp_wn);
        _id_within_warp_wn = WN_Binary(OPR_REM, Integer_type, tid_wn,
                WN_Intconst(Integer_type, _warp_sz));
    }

#if 0
    // Start from the rightmost (innermost) dimension and stop at where the
    // number of threads is a multiple of <warp_sz>.
    INT i = 2;
    UINT n_in_warps = vblk_dim_sz[i];
    while (i > 0 && n_in_warps % warp_sz != 0) n_in_warps *= vblk_dim_sz[--i];
    Is_True(n_in_warps % warp_sz == 0,
            ("The number of threads (in a thread block) that execute "
             "kernel <%s> is not a multiple of the warp size (%d)!\n",
             ST_name(get_kernel_sym()), warp_sz));

    UINT p = i;

    // Compute the number of warps.
    n_in_warps /= warp_sz;
    _n_warps = n_in_warps;
    while (i > 0) _n_warps *= vblk_dim_sz[--i];

    // Generate an index expression from dimension p to the end.
    WN *in_idx_wn = ldid_threadIdx(2-p);
    for (i = p+1; i < 3; ++i)
    {
        in_idx_wn = WN_Add(Integer_type,
                WN_Mpy(Integer_type, in_idx_wn,
                    WN_Intconst(Integer_type, vblk_dim_sz[i])),
                ldid_threadIdx(2-i));
    }

    // Generate an index expression from dimension 0 to dimension (p-1).
    WN *out_idx_wn = WN_Intconst(Integer_type, 0);
    for (i = 0; i < p; ++i)
    {
        out_idx_wn = WN_Add(Integer_type,
                WN_Mpy(Integer_type, out_idx_wn,
                    WN_Intconst(Integer_type, vblk_dim_sz[i])),
                ldid_threadIdx(2-i));
    }

    // ID within a warp is <in_idx_wn> % <warp_sz>.
    _id_within_warp_wn = WN_Binary(OPR_REM, Integer_type,
            WN_COPY_Tree(in_idx_wn), WN_Intconst(Integer_type, warp_sz));

    // The warp ID is <out_idx_wn> * <n_in_warps> + <in_idx_wn> / <warp_sz>.
    _warp_id_wn = WN_Add(Integer_type,
            WN_Mpy(Integer_type, out_idx_wn,
                WN_Intconst(Integer_type, n_in_warps)),
            WN_Div(Integer_type, in_idx_wn,
                WN_Intconst(Integer_type, warp_sz)));
#endif
}

#endif  // IPA_HICUDA

/*****************************************************************************
 *
 * Find the data structure that stores sections for the given array symbol
 * <arr_st_idx>, if it exists. Otherwise, optionally create a new one.
 *
 ****************************************************************************/

HC_ARRAY_SYM_INFO* HC_KERNEL_INFO::get_arr_sym(ST_IDX arr_st_idx,
        BOOL create_if_not_existed)
{
    HC_ARRAY_SYM_INFO_ITER asi_iter(&_arr_access);
    HC_ARRAY_SYM_INFO *asi = asi_iter.First();
    for ( ; !asi_iter.Is_Empty(); asi = asi_iter.Next()) {
        if (asi->arr_sym() == arr_st_idx) return asi;
    }

    if (! create_if_not_existed) return NULL;

    asi = CXX_NEW(HC_ARRAY_SYM_INFO(arr_st_idx, _pool), _pool);
    _arr_access.Append(asi);
    return asi;
}


void HC_KERNEL_INFO::add_scalar(ST_IDX st_idx, WN_OFFSET offset, BOOL is_ref)
{
    // If already in the list, adjust its MOD/REF info.
    HC_SCALAR_INFO_ITER si_iter(&_scalar_access);
    for (HC_SCALAR_INFO *si = si_iter.First();
            !si_iter.Is_Empty(); si = si_iter.Next()) {
        if (si->get_symbol() == st_idx && si->get_offset() == offset) {
            si->set_access_type(is_ref);
            return;
        }
    }

    _scalar_access.Append(
            CXX_NEW(HC_SCALAR_INFO(st_idx, offset, is_ref), _pool));
}

void HC_KERNEL_INFO::add_arr_region(ST_IDX arr_st_idx, WN *access,
        PROJECTED_REGION *ref_region, PROJECTED_REGION *mod_region)
{
    HC_ARRAY_SYM_INFO *asi = get_arr_sym(arr_st_idx, TRUE);
    asi->add_access(access, ref_region, mod_region);
}

void HC_KERNEL_INFO::add_arr_region(ST_IDX arr_st_idx,
        WN *call, INT actual_idx,
        PROJECTED_REGION *ref_region, PROJECTED_REGION *mod_region)
{
    HC_ARRAY_SYM_INFO *asi = get_arr_sym(arr_st_idx, TRUE);
    asi->add_access(call, actual_idx, ref_region, mod_region);
}

HC_GPU_DATA* HC_KERNEL_INFO::find_gdata_for_scalar(ST_IDX st_idx,
        WN_OFFSET offset)
{
    HC_SCALAR_INFO_ITER si_iter(&_scalar_access);
    for (HC_SCALAR_INFO *si = si_iter.First();
            !si_iter.Is_Empty(); si = si_iter.Next())
    {
        if (si->get_symbol() == st_idx && si->get_offset() == offset)
        {
            return si->passed_as_kernel_param() ? NULL : si->get_gpu_data();
        }
    }

    return NULL;
}

#if 0

HC_GPU_DATA* HC_KERNEL_INFO::find_gdata_for_arr_region(ST_IDX st_idx,
        WN *call_wn, INT actual_idx)
{
    Is_True(call_wn != NULL && OPERATOR_is_call(WN_operator(call_wn)), (""));

    HC_ARRAY_SYM_INFO *asi = get_arr_sym(st_idx);
    HC_ARRAY_INFO_ITER ai_iter(asi->get_arr_info_list());
    for (HC_ARRAY_INFO *ai = ai_iter.First(); !ai_iter.Is_Empty();
            ai = ai_iter.Next()) {
        if (ai->get_wn() == call_wn && ai->get_actual_index() == actual_idx) {
            return ai->get_gpu_data();
        }
    }

    return NULL;
}

HC_GPU_DATA* HC_KERNEL_INFO::find_gdata_for_arr_region(ST_IDX st_idx,
        WN *access_wn)
{
    Is_True(access_wn != NULL, (""));
    OPERATOR opr = WN_operator(access_wn);
    Is_True(opr == OPR_ILOAD || opr == OPR_ISTORE, (""));

    HC_ARRAY_SYM_INFO *asi = get_arr_sym(st_idx);
    HC_ARRAY_INFO_ITER ai_iter(asi->get_arr_info_list());
    for (HC_ARRAY_INFO *ai = ai_iter.First(); !ai_iter.Is_Empty();
            ai = ai_iter.Next()) {
        if (ai->get_wn() == access_wn && ai->get_actual_index() == -1) {
            return ai->get_gpu_data();
        }
    }

    return NULL;
}

#else

HC_GPU_DATA* HC_KERNEL_INFO::find_gdata_for_arr_region(ST_IDX st_idx)
{
    HC_ARRAY_SYM_INFO *asi = get_arr_sym(st_idx, FALSE);
    if (asi == NULL) return NULL;

    // Return the first one in the list.
    HC_ARRAY_INFO *ai = asi->get_arr_info_list()->Head();
    Is_True(ai != NULL, (""));

    return ai->get_gpu_data();
}

#endif

#ifdef IPA_HICUDA

HC_SYM_LIST* HC_KERNEL_INFO::get_kernel_params()
{
    if (_kparams != NULL) return _kparams;

    _kparams = CXX_NEW(HC_SYM_LIST(_pool), _pool);

    // Go through array accesses.
    HC_ARRAY_SYM_INFO_ITER asi_iter(&_arr_access);
    for (HC_ARRAY_SYM_INFO *asi = asi_iter.First(); !asi_iter.Is_Empty();
            asi = asi_iter.Next())
    {
        HC_ARRAY_INFO_ITER ai_iter(asi->get_arr_info_list());
        for (HC_ARRAY_INFO *ai = ai_iter.First(); !ai_iter.Is_Empty();
                ai = ai_iter.Next())
        {
            HC_GPU_DATA *gdata = ai->get_gpu_data();
            Is_True(gdata != NULL, (""));
	    if (flag_opencl){
	      // For OpenCL do not skip constant memory data when seting up
	      // kernel parameters
	    } else {
	      // Skip constant memory data because any reference is made
	      // directly to <cmem> in the local procedure.
	      if (gdata->get_type() == HC_CONSTANT_DATA) continue;
	    }

            HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();
            Is_True(gvi != NULL, (""));
            ST_IDX arr_st_idx = gvi->get_symbol();
            Is_True(arr_st_idx != ST_IDX_ZERO, (""));

            // Check if it exists in the list.
            BOOL found = FALSE;
            for (UINT i = 0; i < _kparams->Elements(); ++i)
            {
                if ((*_kparams)[i] == arr_st_idx) { found = TRUE; break; }
            }

            if (!found)
            {
                _kparams->AddElement(arr_st_idx);

                if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
                {
                    // This is safe in the clone's context.
                    fprintf(TFile, "KERNEL <%s> ARRAY PARAM: %d %s\n",
                            ST_name(_kfunc_st_idx),
                            arr_st_idx, ST_name(arr_st_idx));
                }
            }
        }
    }

    // Go through scalar accesses.
    HC_SCALAR_INFO_ITER si_iter(&_scalar_access);
    for (HC_SCALAR_INFO *si = si_iter.First(); !si_iter.Is_Empty();
            si = si_iter.Next())
    {
        // Some accesses may not have a redirection target, it must be the
        // result of 2nd scalar DAS. Pass it as a parameter.
        if (! si->is_redirection_target_found())
        {
            Is_True(! si->is_mod(), (""));
            si->set_passed_as_kernel_param();
        }

        ST_IDX st_idx = ST_IDX_ZERO;
        if (si->passed_as_kernel_param())
        {
            st_idx = si->get_symbol();
        }
        else
        {
            HC_GPU_DATA *gdata = si->get_gpu_data();
            Is_True(gdata != NULL, (""));
            // Skip constant memory data because any reference is made
            // directly to <cmem> in the local procedure.
            if (gdata->get_type() == HC_CONSTANT_DATA) continue;

            HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();
            Is_True(gvi != NULL, (""));
            st_idx = gvi->get_symbol();
        }
        Is_True(st_idx != ST_IDX_ZERO, (""));

        // IMPORTANT: skip CUDA runtime symbols: blockIdx and thrreadIdx.
        if (st_attr_is_cuda_runtime(st_idx)) continue;

        // Does this scalar variable exist in the current parameter list?
        BOOL found = FALSE;
        for (UINT i = 0; i < _kparams->Elements(); ++i)
        {
            if ((*_kparams)[i] == st_idx) { found = TRUE; break; }
        }
        // Sanity check: if the scalar is not a struct's field, it must not be
        // in the current parameter list.
        if (TY_kind(ST_type(st_idx)) != KIND_STRUCT) Is_True(!found, (""));

        if (!found)
        {
            _kparams->AddElement(st_idx);

            if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
            {
                // This is safe in the clone's context.
                fprintf(TFile, "KERNEL <%s> SCALAR PARAM: %d %s\n",
                        ST_name(_kfunc_st_idx), st_idx, ST_name(st_idx));
            }
        }
    }

    return _kparams;
}

#endif  // IPA_HICUDA

void HC_KERNEL_INFO::clear_proj_regions()
{
    // the PROJECTED_REGIONs in <_arr_access>
    HC_ARRAY_SYM_INFO_ITER hasii_ref(&_arr_access);
    for (HC_ARRAY_SYM_INFO *hasi = hasii_ref.First(); !hasii_ref.Is_Empty();
            hasi = hasii_ref.Next()) hasi->clear_proj_regions();
}

void HC_KERNEL_INFO::deep_copy_proj_regions()
{
    // the PROJECTED_REGIONs in <_arr_access>
    HC_ARRAY_SYM_INFO_ITER hasii_ref(&_arr_access);
    for (HC_ARRAY_SYM_INFO *hasi = hasii_ref.First(); !hasii_ref.Is_Empty();
            hasi = hasii_ref.Next()) hasi->deep_copy_proj_regions(_pool);
}

void HC_KERNEL_INFO::match_gpu_data_with_das(IPA_HC_ANNOT *annot,
        HC_VISIBLE_GPU_DATA *vgdata)
{
    Is_True(annot != NULL, (""));
    HC_FORMAL_GPU_DATA_ARRAY *fgda =
        (HC_FORMAL_GPU_DATA_ARRAY*)annot->get_annot_data();
    // This could be NULL.

    // Go through the scalar accesses.
    HC_SCALAR_INFO_ITER si_iter(&_scalar_access);
    for (HC_SCALAR_INFO *si = si_iter.First(); !si_iter.Is_Empty();
            si = si_iter.Next())
    {
        ST_IDX st_idx = si->get_symbol();
        // TODO: do we care about the offset here?

        // Make sure the access has a transient GPU data table, which will
        // contain one entry for each annotation.
        HC_MATCHED_GDATA_TABLE *mgt = si->get_gpu_data_table();
        if (mgt == NULL) {
            mgt = CXX_NEW(HC_MATCHED_GDATA_TABLE(5,_pool), _pool);
            si->set_gpu_data_table(mgt);
        }

        HC_GPU_DATA *gdata = vgdata->Find(st_idx);
        if (gdata == NULL)
        {
            // This access must be read-only.
            HC_assert(!si->is_mod(),
                    ("Scalar <%s> is modified by kernel <%s> but "
                     "does not exist in any GPU memories!",
                     ST_name(st_idx), ST_name(get_kernel_sym())));
        }
        else
        {
            // The GPU data for a scalar can never have section spec.
            Is_True(!gdata->is_arr_section(), (""));

            // Make sure the access type and the GPU data type are compatible.
            HC_assert(!(si->is_mod()
                        && gdata->get_type() == HC_CONSTANT_DATA),
                    ("Scalar <%s> is modified by kernel <%s> but "
                     "only exists in the constant memory!",
                     ST_name(st_idx), ST_name(get_kernel_sym())));
        }

        // gdata == NULL indicates passed as a parameter.
        mgt->Enter(annot, gdata);
        
        // If the GPU data is in the annotation, mark it USED.
        if (gdata != NULL && fgda != NULL) fgda->set_formal_data_used(gdata);
    }

    // Go through the array section accesses.
    HC_ARRAY_SYM_INFO_ITER asi_iter(&_arr_access);
    for (HC_ARRAY_SYM_INFO *asi = asi_iter.First(); !asi_iter.Is_Empty();
            asi = asi_iter.Next())
    {
        ST_IDX st_idx = asi->arr_sym();

        HC_GPU_DATA *gdata = vgdata->Find(st_idx);
        // The GPU data for an array variable must have section spec.
        HC_assert(gdata != NULL && gdata->is_arr_section(),
                ("Array <%s> is accessed by kernel <%s> but "
                 "does not exist in any GPU memories!",
                 ST_name(st_idx), ST_name(get_kernel_sym())));

        PROJECTED_REGION *gdata_pr =
            gdata->get_alloc_section()->get_section();
        Is_True(gdata_pr != NULL, (""));

        // We only compare PROJECTED_REGIONs if the kernel has complete array
        // info and not the entire array is brought into the GPU memory.
        BOOL compare_pr = (!has_incomplete_array_info() &&
                !gdata->is_alloc_section_full());

        // Make sure it covers each access region.
        HC_ARRAY_INFO_ITER ai_iter(asi->get_arr_info_list());
        for (HC_ARRAY_INFO *ai = ai_iter.First(); !ai_iter.Is_Empty();
                ai = ai_iter.Next())
        {
            // Make sure the access has a transient GPU data table, which will
            // contain one entry for each annotation.
            HC_MATCHED_GDATA_TABLE *mgt = ai->get_gpu_data_table();
            if (mgt == NULL) {
                mgt = CXX_NEW(HC_MATCHED_GDATA_TABLE(5,_pool), _pool);
                ai->set_gpu_data_table(mgt);
            }

            PROJECTED_REGION *ref_pr = ai->get_region(TRUE);
            if (compare_pr && ref_pr != NULL)
            {
                // Check the coverage of the REF region.
                INT result = gdata_pr->Contains(ref_pr);
                HC_assert(result != 0,
                        ("The section of array <%s> in the %s memory "
                         "cannot cover the sections referenced by "
                         "kernel <%s>!",
                         ST_name(st_idx), gdata->get_type_name(),
                         ST_name(get_kernel_sym())));

                if (result == -1)
                {
                    // We cannot do the check accurately, so give a warning.
                    HC_warn("The section of array <%s> in the %s memory "
                            "may not cover the sections referenced by "
                            "kernel <%s>.",
                            ST_name(st_idx), gdata->get_type_name(),
                            ST_name(get_kernel_sym()));

                    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
                    {
                        gdata_pr->Print(TFile);
                        ref_pr->Print(TFile);
                    }
                }
            }

            PROJECTED_REGION *mod_pr = ai->get_region(FALSE);
            if (mod_pr != NULL)
            {
                // The GPU data cannot be CONSTANT.
                HC_assert(gdata->get_type() != HC_CONSTANT_DATA,
                        ("Array <%s> is modified by kernel <%s> but "
                         "only exists in the constant memory!",
                         ST_name(st_idx), ST_name(get_kernel_sym())));

                if (compare_pr)
                {
                    // Check the coverage of the MOD region.
                    INT result = gdata_pr->Contains(mod_pr);
                    HC_assert(result != 0,
                            ("The section of array <%s> in the %s memory "
                             "cannot cover the sections modified by "
                             "kernel <%s>!",
                             ST_name(st_idx), gdata->get_type_name(),
                             ST_name(get_kernel_sym())));

                    if (result == -1)
                    {
                        // We cannot do the check accurately, so give a warning.
                        HC_warn("The section of array <%s> in the %s memory "
                                "may not cover the sections modified by "
                                "kernel <%s>.",
                                ST_name(st_idx), gdata->get_type_name(),
                                ST_name(get_kernel_sym()));

                        if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
                        {
                            gdata_pr->Print(TFile);
                            mod_pr->Print(TFile);
                        }
                    }
                }
            }

            mgt->Enter(annot, gdata);
        }

        // If the GPU data is in the annotation, mark it USED.
        if (fgda != NULL) fgda->set_formal_data_used(gdata);
    }

    if (has_incomplete_array_info() && !_arr_access.Is_Empty())
    {
        HC_warn("You are responsible for ensuring that array sections "
                "accessed by kernel <%s> are covered by data directives.",
                ST_name(get_kernel_sym()));
    }
}

HC_KERNEL_INFO* IPA_NODE::get_kernel_info_by_sym(ST_IDX kfunc_st_idx) const
{
    Is_True(_kernel_info_list != NULL, (""));

    UINT n_kinfo = _kernel_info_list->Elements();
    for (UINT i = 0; i < n_kinfo; ++i)
    {
        HC_KERNEL_INFO *kinfo = (*_kernel_info_list)[i];
        if (kfunc_st_idx == kinfo->get_kernel_sym()) return kinfo;
    }

    return NULL;
}

void HC_KERNEL_INFO::print(FILE *fp)
{
    fprintf(fp, "====> [DAS OF KERNEL %s]\n", ST_name(get_kernel_sym()));

    if (has_incomplete_array_info())
    {
        fprintf(fp, "INCOMPLETE ARRAY INFO\n");
    }

    // scalar MOD/REF info
    fprintf(fp, "SCALAR ACCESS:\n");
    HC_SCALAR_INFO_ITER si_iter(&_scalar_access);
    for (HC_SCALAR_INFO *si = si_iter.First(); !si_iter.Is_Empty();
            si = si_iter.Next())
    {
        fprintf(fp, "\t%s (offset %d): %s %s\n",
                ST_name(si->get_symbol()), si->get_offset(),
                si->is_ref() ? "REF" : "", si->is_mod() ? "MOD" : "");
    }

    // Array MOD info
    fprintf(fp, "ARRAY ACCESS:\n");
    HC_ARRAY_SYM_INFO_ITER asi_iter(&_arr_access);
    for (HC_ARRAY_SYM_INFO *asi = asi_iter.First(); !asi_iter.Is_Empty();
            asi = asi_iter.Next()) {
        asi->print(fp, 1);
    }

    fprintf(fp, "====> [END OF DAS OF KERNEL %s]\n",
            ST_name(get_kernel_sym()));
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void HC_KERNEL_CONTEXT::consumed_by_loop_partition(
        const HC_LOOP_PART_INFO *lpi)
{
    Is_True(lpi != NULL, (""));

    if (lpi->get_block_clause() != HC_LPI_DT_NONE)
    {
        _vgrid_dim_idx++;
        if (_kernel_info != NULL)
        {
            HC_assert(_vgrid_dim_idx <= _kernel_info->get_vgrid_dims(),
                    ("The number of over_tblock clauses in kernel <%s> "
                     "exceeds the dimensionality (%u) of its thread block "
                     "space!", ST_name(_kernel_info->get_kernel_sym()),
                     _kernel_info->get_vgrid_dims()));
        }
    }

    if (lpi->get_thread_clause() != HC_LPI_DT_NONE)
    {
        _vblk_dim_idx++;
        if (_kernel_info != NULL)
        {
            HC_assert(_vblk_dim_idx <= _kernel_info->get_vblk_dims(),
                    ("The number of over_thread clauses in kernel <%s> "
                     "exceeds the dimensionality (%u) of its thread space!",
                     ST_name(_kernel_info->get_kernel_sym()),
                     _kernel_info->get_vblk_dims()));
        }
    }
}

void HC_KERNEL_CONTEXT::unconsumed_by_loop_partition(
        const HC_LOOP_PART_INFO *lpi)
{
    Is_True(lpi != NULL, (""));

    if (lpi->get_block_clause() != HC_LPI_DT_NONE)
    {
        Is_True(_vgrid_dim_idx > 0, (""));
        _vgrid_dim_idx--;
    }

    if (lpi->get_thread_clause() != HC_LPI_DT_NONE)
    {
        Is_True(_vblk_dim_idx > 0, (""));
        _vblk_dim_idx--;
    }
}

void HC_KERNEL_CONTEXT::incr_vgrid_dim_idx(UINT ofst)
{
    _vgrid_dim_idx += ofst;
    Is_True(_kernel_info == NULL
            || _vgrid_dim_idx <= _kernel_info->get_vgrid_dims(), (""));
}

void HC_KERNEL_CONTEXT::incr_vblk_dim_idx(UINT ofst)
{
    _vblk_dim_idx += ofst;
    Is_True(_kernel_info == NULL
            || _vblk_dim_idx <= _kernel_info->get_vblk_dims(), (""));
}

HC_LOOP_PART_INFO::HC_LOOP_PART_INFO(WN *region)
{
    // Parse the pragma.
    Is_True(region != NULL && WN_operator(region) == OPR_REGION, (""));
    WN *pragma_blk = WN_region_pragmas(region);
    Is_True(pragma_blk != NULL && WN_operator(pragma_blk) == OPR_BLOCK, (""));
    WN *pragma = WN_first(pragma_blk);
    Is_True(pragma != NULL && WN_operator(pragma) == OPR_PRAGMA
            && WN_pragma(pragma) == WN_PRAGMA_HC_KERNEL_PART
            && WN_next(pragma) == NULL, (""));

    _blk_clause = (HC_LPI_DIST_TYPE)WN_pragma_arg1(pragma);
    _thr_clause = (HC_LPI_DIST_TYPE)WN_pragma_arg2(pragma);

    _kernel_context = NULL;

    _idxv_st_idx = ST_IDX_ZERO;
    _idxv_lbnd_wn = _idxv_ubnd_wn = NULL;

    _idxv_prop = NULL;
}

HC_LOOP_PART_INFO::HC_LOOP_PART_INFO(const HC_LOOP_PART_INFO& orig)
{
    // Do not migrate kernel context.
    _kernel_context = NULL;

    _blk_clause = orig._blk_clause;
    _thr_clause = orig._thr_clause;

    _idxv_st_idx = orig._idxv_st_idx;

    _idxv_lbnd_wn = (orig._idxv_lbnd_wn == NULL) ?
        NULL : WN_COPY_Tree(orig._idxv_lbnd_wn);
    _idxv_ubnd_wn = (orig._idxv_ubnd_wn == NULL) ?
        NULL : WN_COPY_Tree(orig._idxv_ubnd_wn);

    // shallow-copy
    _idxv_prop = orig._idxv_prop;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Return the symbol for the given KERNEL region, or ST_IDX_ZERO if it is not.
 *
 ****************************************************************************/

ST_IDX HC_get_kernel_sym(WN *kregion_wn)
{
    if (WN_opcode(kregion_wn) != OPC_REGION
            || WN_region_kind(kregion_wn) != REGION_KIND_HICUDA) {
        return ST_IDX_ZERO;
    }

    // Get the first pragma in the pragma block.
    WN *pragma_wn = WN_first(WN_region_pragmas(kregion_wn));
    if (pragma_wn == NULL || WN_opcode(pragma_wn) != OPC_PRAGMA
            || (WN_PRAGMA_ID)WN_pragma(pragma_wn) != WN_PRAGMA_HC_KERNEL) {
        return ST_IDX_ZERO;
    }

    return WN_st_idx(pragma_wn);
}

BOOL is_loop_part_region(WN *wn)
{
    OPERATOR opr = WN_operator(wn);

    if (opr != OPR_REGION
            || WN_region_kind(wn) != REGION_KIND_HICUDA) return FALSE;

    // Get the first pragma in the pragma block.
    WN *pragma = WN_first(WN_kid1(wn));
    return (pragma != NULL
            && (WN_PRAGMA_ID)WN_pragma(pragma) == WN_PRAGMA_HC_KERNEL_PART);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifdef IPA_HICUDA

void HC_print_kernel_das(IPA_NODE *node, FILE *fp)
{
    // ASSUME: the kernel-related flags have been determined.
    if (! node->contains_kernel()) return;

    fprintf(fp, "%s\tKernel regions in <%s>\n%s", DBar, node->Name(), DBar);

    // IMPORTANT!
    IPA_NODE_CONTEXT context(node);

    UINT n_kernels = node->num_kregions();
    Is_True(n_kernels > 0, (""));
    HC_KERNEL_INFO_LIST *kil = node->get_kernel_info_list();
    for (UINT i = 0; i < n_kernels; ++i) (*kil)[i]->print(fp);
}

void IPA_print_kernel_das(FILE *fp)
{
    IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *node = cg_iter.Current();
        if (node == NULL) continue;
        HC_print_kernel_das(node, fp);
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void HC_parse_kernel_dir_walker(WN *wn,
        HC_KERNEL_INFO_LIST *ki_list, MEM_POOL *pool)
{
    if (wn == NULL) return;

    // Check if it is a kernel region.
    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        ki_list->AddElement(CXX_NEW(HC_KERNEL_INFO(wn,pool), pool));
        // No need to process further as kernels are not nested.
        return;
    }

    OPERATOR opr = WN_operator(wn);

    // Handle composite node.
    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            HC_parse_kernel_dir_walker(kid_wn, ki_list, pool);
            kid_wn = WN_next(kid_wn);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HC_parse_kernel_dir_walker(WN_kid(wn,i), ki_list, pool);
        }
    }
}

void HC_parse_kernel_directives(IPA_NODE *node)
{
    node->reset_kernel_info_list();

    // Do nothing if it does not contain any KERNEL regions.
    if (! node->contains_kernel()) return;

    // IMPORTANT!
    IPA_NODE_CONTEXT context(node);

    // Use the node's own mempool to allocate HC_KERNEL_INFO.
    HC_parse_kernel_dir_walker(node->Whirl_Tree(),
            node->get_kernel_info_list(), node->Mem_Pool());
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void HC_rename_kernel_walker(WN *wn, IPA_NODE *node,
        ST_IDX parent_kernel_sym, UINT& kregion_id)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        // Generate a new kernel symbol.
        INT clone_num = IPA_Call_Graph->get_clone_num(node);
        Is_True(clone_num >= 0, (""));
        const char *kfunc_name_str = ST_name(kfunc_st_idx);
        char *kfunc_clone_name_str = (char*)alloca(strlen(kfunc_name_str)+15);
        sprintf(kfunc_clone_name_str,
                "%s_clone%d", kfunc_name_str, clone_num);

        ST *kfunc_clone_st = New_ST(ST_IDX_level(kfunc_st_idx));
        // Copied from <wfe_expand_hc_kernel_begin>.
        ST_Init(kfunc_clone_st, Save_Str(kfunc_clone_name_str),
                CLASS_NAME, SCLASS_UNKNOWN, EXPORT_LOCAL, 0);
        ST_IDX kfunc_clone_st_idx = ST_st_idx(kfunc_clone_st);

        // Update the WN tree.
        WN_st_idx(WN_first(WN_region_pragmas(wn))) = kfunc_clone_st_idx;

        // Update HC_KERNEL_INFO.
        if (node->num_kregions() > 0)
        {
            HC_KERNEL_INFO *ki = node->get_kernel_info(kregion_id++);
            Is_True(ki != NULL, (""));
            ki->set_kernel_sym(kfunc_clone_st_idx);
        }

        // Establish the kernel context for updating edge annotation.
        parent_kernel_sym = kfunc_clone_st_idx;
    }

    if (OPERATOR_is_call(opr))
    {
        IPA_EDGE *e = node->get_wn_to_edge_map()->Find(wn);
        // If not in a kernel region, <parent_kernel_sym> is ST_IDX_ZERO.
        if (e != NULL) e->set_parent_kernel_sym(parent_kernel_sym);
    }

    // Handle composite node.
    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            HC_rename_kernel_walker(kid_wn,
                    node, parent_kernel_sym, kregion_id);
            kid_wn = WN_next(kid_wn);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HC_rename_kernel_walker(WN_kid(wn,i),
                    node, parent_kernel_sym, kregion_id);
        }
    }
}

void HC_rename_kernels(IPA_NODE *clone)
{
    Is_True(clone != NULL && clone->Is_New_Clone(), (""));

    // Skip the node that does not contain any kernel regions.
    // We cannot use this check because it may not have this flag yet.
    // if (! clone->contains_kernel()) return;

    IPA_NODE_CONTEXT context(clone);

    // Link IPA_EDGEs with WN nodes.
    IPA_Call_Graph->Map_Callsites(clone);

    UINT kregion_id = 0;
    HC_rename_kernel_walker(clone->Whirl_Tree(),
            clone, ST_IDX_ZERO, kregion_id);
    Is_True(kregion_id == clone->num_kregions(), (""));

    // No need to rebuild Parent_Map as only symbols are replaced.
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef STACK<HC_LOOP_PART_INFO*> HC_LOOP_PART_STACK;
typedef STACK<HC_EXPR_PROP*> HC_EXPR_PROP_STACK;

/*****************************************************************************
 *
 * The structure of this function is indentical to
 * <construct_kernel_context_annot>.
 *
 * NOTE: LOOP_PARTITION directives are lowered from inside out; otherwise, the
 * Parent_Map needs to be rebuilt many more times.
 *
 * The given mempool is used to allocate HC_EXPR_PROP.
 *
 ****************************************************************************/

#if 0

static void HC_handle_in_kernel_dir_walker(WN *wn, IPA_NODE *node,
        HC_KERNEL_CONTEXT *kcontext,
        UINT& lp_dir_id /* REFERENCE! */, HC_LOOP_PART_STACK *lp_stack,
        HC_EXPR_PROP_STACK *prop_stack,
        UINT& shared_dir_id, MEM_POOL *pool)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        // Start a kernel context.
        HC_KERNEL_INFO *kinfo = node->get_kernel_info_by_sym(kfunc_st_idx);
        Is_True(kcontext == NULL && kinfo != NULL, (""));
        // TODO: use a temp mempool
        kcontext = CXX_NEW(HC_KERNEL_CONTEXT(kinfo,0,0), node->Mem_Pool());
    }

    BOOL is_loop_part = is_loop_part_region(wn);
    HC_LOOP_PART_INFO *lpi = NULL;

    HC_EXPR_PROP *prop = NULL;

    // HC_LOOP_PART_INFO only exists for K- and IK-procedures.
    if (node->contains_kernel() || node->may_be_inside_kernel())
    {
        if (is_loop_part)
        {
            lpi = (*node->get_loop_part_info_list())[lp_dir_id++];
            Is_True(kcontext != NULL && lpi != NULL, (""));

            // Save a copy of the current kernel context.
            lpi->set_kernel_context(
                    CXX_NEW(HC_KERNEL_CONTEXT(*kcontext), node->Mem_Pool()));
            // Update the kernel context.
            kcontext->consumed_by_loop_partition(lpi);

            // Push the directive onto the stack.
            lp_stack->Push(lpi);
        }

        if (opr == OPR_DO_LOOP)
        {
            // This must be a regular DO LOOP. Find the property of its loop
            // index variable.
            HC_DOLOOP_INFO li(wn);

            INT factor = li.get_step();
            if (factor < 0) factor = -factor;
            if (factor > 1)
            {
                WN *expr_wn = WN_Sub(Integer_type,
                        WN_LdidScalar(li.get_idx_var()),
                        WN_COPY_Tree(li.get_init_expr()));
                prop = CXX_NEW(HC_EXPR_PROP(expr_wn, factor), pool);

                prop_stack->push(prop);
            }
        }

        if (opr == OPR_PRAGMA)
        {
            WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
            if (pid == WN_PRAGMA_HC_SHARED_COPYIN)
            {
                // Add the range of each visible loop index variable to this
                // SHARED directive.
                HC_GPU_DATA *sdata =
                    (*node->get_shared_data_list())[shared_dir_id++];
                Is_True(sdata != NULL, (""));

                UINT n_lp_dirs = lp_stack->Elements();
                for (UINT i = 0; i < n_lp_dirs; ++i)
                {
                    sdata->add_idxv_range(lp_stack->Bottom_nth(i));
                }
            }
        }
    }

    // Handle composite node.
    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            WN *next_wn = WN_next(kid_wn);
            HC_handle_in_kernel_dir_walker(kid_wn, node, kcontext,
                    lp_dir_id, lp_stack, shared_dir_id, pool);
            kid_wn = next_wn;
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            // Here, we know for sure the kid is not a loop region.
            HC_handle_in_kernel_dir_walker(WN_kid(wn,i), node, kcontext,
                    lp_dir_id, lp_stack, shared_dir_id, pool);
        }
    }

    if (lpi != NULL)
    {
        // Pop it off the stack.
        Is_True(lp_stack->Pop() == lpi, (""));

        // Revert back the kernel context.
        kcontext->unconsumed_by_loop_partition(lpi);

        // Lower the LOOP_PARTITION directive to CUDA code.
        HC_lower_loop_part_region(wn, lpi, node->get_hc_lvar_store(), pool);
    }

    if (prop != NULL)
    {
        // Pop the property off the stack.
        Is_True(prop_stack->Pop() == prop, (""));
    }

    // This will remove LOOP_PARTITION directives in an N-procedure.
    if (is_loop_part)
    {
        // Now, the entire loop region has been processed (including nested
        // regions), we need to pull the region body out and insert it before
        // the region.
        WN *parent_wn = LWN_Get_Parent(wn);
        WN *region_body_wn = WN_region_body(wn);
        WN_region_body(wn) = NULL;
        WN_INSERT_BlockBefore(parent_wn, wn, region_body_wn);
        WN_DELETE_FromBlock(parent_wn, wn);

        // Rebuild the Parent_Map locally.
        WN_Parentize(parent_wn, Parent_Map, Current_Map_Tab);
    }
}

#else

// Remove LOOP_PARTITION directives in an N-procedure.
//
// Return the next node to be processed in <parent_wn> if it is a BLOCK, or
// NULL otherwise.
//
// The caller must rebuild the Parent map after this call.
//
static WN* HC_remove_lp_dir_walker(WN *parent_wn, WN *wn)
{
    WN *next_wn = (parent_wn != NULL && WN_operator(parent_wn) == OPR_BLOCK) ?
        WN_next(wn) : NULL;

    if (wn == NULL) return next_wn;

    if (is_loop_part_region(wn))
    {
        // The parent must be a BLOCK node.
        Is_True(parent_wn != NULL
                && WN_operator(parent_wn) == OPR_BLOCK, (""));

        WN *region_body_wn = WN_region_body(wn);
        WN_region_body(wn) = NULL;      // IMPORTANT!
        WN_INSERT_BlockAfter(parent_wn, wn, region_body_wn);
        next_wn = WN_next(wn);
        WN_DELETE_FromBlock(parent_wn, wn);

        return next_wn;
    }

    if (WN_operator(wn) == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            kid_wn = HC_remove_lp_dir_walker(wn, kid_wn);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            Is_True(HC_remove_lp_dir_walker(wn, WN_kid(wn,i)) == NULL, (""));
        }
    }

    return next_wn;
}

// Translate LOOP_PARTITION directives and prepare for SHARED directives.
//
// Return the next node to be processed in <parent_wn> if it is a BLOCK, or
// NULL otherwise.
//
// The caller must rebuild the Parent map after this call.
//
static WN* HC_handle_in_kernel_dir_walker(WN *parent_wn, WN *wn,
        IPA_NODE *node, HC_KERNEL_CONTEXT *kcontext,
        UINT& lp_dir_id /* REFERENCE! */, HC_LOOP_PART_STACK *lp_stack,
        HC_EXPR_PROP_STACK *prop_stack,
        UINT& shared_dir_id, MEM_POOL *pool)
{
    WN *next_wn = (parent_wn != NULL && WN_operator(parent_wn) == OPR_BLOCK) ?
        WN_next(wn) : NULL;

    if (wn == NULL) return next_wn;

    if (is_loop_part_region(wn))
    {
        // The parent must be a BLOCK node.
        Is_True(parent_wn != NULL
                && WN_operator(parent_wn) == OPR_BLOCK, (""));

        HC_LOOP_PART_INFO *lpi =
            (*node->get_loop_part_info_list())[lp_dir_id++];
        Is_True(kcontext != NULL && lpi != NULL, (""));

        // Save a copy of the current kernel context.
        lpi->set_kernel_context(
                CXX_NEW(HC_KERNEL_CONTEXT(*kcontext), node->Mem_Pool()));
        // Lower the LOOP_PARTITION directive to CUDA code.
        BOOL keep_loop = HC_lower_loop_part_region(wn, lpi,
                node->get_hc_lvar_store(), pool);

        // Update the kernel context.
        kcontext->consumed_by_loop_partition(lpi);
        // Push the directive onto the stack.
        lp_stack->Push(lpi);

        // Add the index variable property to the stack.
        HC_EXPR_PROP *prop = lpi->get_idxv_prop();
        if (prop != NULL) prop_stack->Push(prop);

        // Process the region body.
        WN *region_body_wn = WN_region_body(wn);
        if (keep_loop)
        {
            // We must skip the DO loop and start processing its body;
            // otherwise another property will be generated when handling
            // regular DO loops.
            WN *loop_wn = WN_first(region_body_wn);
            while (loop_wn != NULL && WN_operator(loop_wn) != OPR_DO_LOOP) 
            {
                loop_wn = WN_next(loop_wn);
            }
            Is_True(loop_wn != NULL, (""));

            HC_handle_in_kernel_dir_walker(loop_wn, WN_kid(loop_wn,4),
                    node, kcontext, lp_dir_id, lp_stack,
                    prop_stack, shared_dir_id, pool);

            // There should be no nodes after the loop.
            Is_True(WN_next(loop_wn) == NULL, (""));
        }
        else
        {
            HC_handle_in_kernel_dir_walker(wn, region_body_wn,
                    node, kcontext, lp_dir_id, lp_stack,
                    prop_stack, shared_dir_id, pool);
        }

        // Pull the region body out of the region node.
        WN_region_body(wn) = NULL;
        WN_INSERT_BlockBefore(parent_wn, wn, region_body_wn);
        WN_DELETE_FromBlock(parent_wn, wn);

        // Pop it off the stack.
        Is_True(lp_stack->Pop() == lpi, (""));
        // Revert back the kernel context.
        kcontext->unconsumed_by_loop_partition(lpi);

        // Pop the loop index var property off the stack.
        if (prop != NULL) Is_True(prop_stack->Pop() == prop, (""));

        // Now we will process from <node_wn> as normal upon return.
        return next_wn;
    }

    ST_IDX kfunc_st_idx = HC_get_kernel_sym(wn);
    if (kfunc_st_idx != ST_IDX_ZERO)
    {
        // Start a kernel context.
        HC_KERNEL_INFO *kinfo = node->get_kernel_info_by_sym(kfunc_st_idx);
        Is_True(kcontext == NULL && kinfo != NULL, (""));
        // TODO: use a temp mempool
        kcontext = CXX_NEW(HC_KERNEL_CONTEXT(kinfo,0,0), node->Mem_Pool());

        // To speed up the traversal, go directly to the kernel body.
        HC_handle_in_kernel_dir_walker(wn, WN_region_body(wn),
                node, kcontext, lp_dir_id, lp_stack,
                prop_stack, shared_dir_id, pool);

        return next_wn;
    }

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_DO_LOOP)
    {
        // This must be a regular DO LOOP. Find the property of its loop
        // index variable.
        HC_DOLOOP_INFO li(wn);

        HC_EXPR_PROP *prop = NULL;
        INT factor = li.get_step();
        if (factor < 0) factor = -factor;
        if (factor > 1)
        {
            WN *expr_wn = WN_Sub(Integer_type,
                    WN_LdidScalar(li.get_idx_var()),
                    WN_COPY_Tree(li.get_init_expr()));
            prop = CXX_NEW(HC_EXPR_PROP(expr_wn, factor), pool);

            // Push the property.
            prop_stack->Push(prop);
        }

        // Process the loop body directly.
        HC_handle_in_kernel_dir_walker(wn, WN_kid(wn,4),
                node, kcontext, lp_dir_id, lp_stack,
                prop_stack, shared_dir_id, pool);

        // Pop the property off the stack.
        if (prop != NULL) Is_True(prop_stack->Pop() == prop, (""));
    }
    else if (opr == OPR_PRAGMA)
    {
        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        if (pid == WN_PRAGMA_HC_SHARED_COPYIN)
        {
            HC_GPU_DATA *sdata =
                (*node->get_shared_data_list())[shared_dir_id++];
            Is_True(sdata != NULL, (""));

            // Save the kernel context.
            sdata->set_kernel_info(kcontext->get_kernel_info());

            // Add the range of each visible loop index variable to this
            // SHARED directive.
            UINT n_lp_dirs = lp_stack->Elements();
            for (UINT i = 0; i < n_lp_dirs; ++i)
            {
                sdata->add_idxv_range(lp_stack->Bottom_nth(i));
            }

            // Add the index variable properties to this HC_GPU_DATA.
            HC_EXPR_PROP_LIST *props = sdata->create_lp_idxv_prop_list();
            UINT n_props = prop_stack->Elements();
            for (UINT i = 0; i < n_props; ++i)
            {
                props->AddElement(prop_stack->Bottom_nth(i));
            }
        }

        // No need to process inside.
    }
    // Handle composite node.
    else if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            kid_wn = HC_handle_in_kernel_dir_walker(wn, kid_wn,
                    node, kcontext, lp_dir_id, lp_stack,
                    prop_stack, shared_dir_id, pool);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            Is_True(HC_handle_in_kernel_dir_walker(wn, WN_kid(wn,i),
                        node, kcontext, lp_dir_id, lp_stack,
                        prop_stack, shared_dir_id, pool) == NULL, (""));
        }
    }

    return next_wn;
}

#endif

/*****************************************************************************
 *
 * Determine the kernel context for each LOOP_PARTITION directive and lower it
 * into CUDA code. For an N-procedure, these directives are simply removed.
 *
 * Here, we assume CUDA runtime functions and other symbols have been declared
 * already.
 *
 ****************************************************************************/

void HC_handle_in_kernel_directives(IPA_NODE *node, MEM_POOL *pool)
{
    // K-, IK-, and N-procedures can have LOOP_PARTITION or SHARED directives.
    if (!node->contains_loop_part_dir()
            && !node->contains_shared_dir()) return;
    Is_True(node->contains_kernel() || !node->may_lead_to_kernel(), (""));

    HC_KERNEL_CONTEXT *kcontext = NULL;
    if (node->may_be_inside_kernel())
    {
        // For a IK-procedure, we have a single non-dummy kernel context in
        // the propagated annotation.
        IPA_HC_ANNOT_LIST *annots = node->get_hc_annots();
        Is_True(annots != NULL, (""));
        IPA_HC_ANNOT *annot = annots->Head();
        Is_True(annot != NULL && !annot->is_dummy(), (""));
        kcontext = (HC_KERNEL_CONTEXT*)annot->get_annot_data();
        Is_True(kcontext != NULL, (""));
    }

    // Switch to this node's context.
    IPA_NODE_CONTEXT ipa_context(node);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "In node <%s> ...\n", node->Name());
    }

    if (node->contains_kernel() || node->may_be_inside_kernel())
    {
        // TODO: use some other pool
        MEM_POOL *tmp_pool = node->Mem_Pool();
        HC_LOOP_PART_STACK *lp_stack =
            CXX_NEW(HC_LOOP_PART_STACK(tmp_pool), tmp_pool);
        HC_EXPR_PROP_STACK *prop_stack =
            CXX_NEW(HC_EXPR_PROP_STACK(tmp_pool), tmp_pool);

        // The root function node should not change.
        // <kcontext> is NULL for K- and N-procedures.
        UINT lp_dir_id = 0, shared_dir_id = 0;
        HC_handle_in_kernel_dir_walker(NULL, node->Whirl_Tree(),
                node, kcontext, lp_dir_id, lp_stack,
                prop_stack, shared_dir_id, pool);
        Is_True(lp_dir_id
                == node->get_loop_part_info_list()->Elements(), (""));
        Is_True(shared_dir_id
                == node->get_shared_data_list()->Elements(), (""));

        // Manual cleanup.
        CXX_DELETE(lp_stack, tmp_pool);
        CXX_DELETE(prop_stack, tmp_pool);
    }
    else
    {
        HC_remove_lp_dir_walker(NULL, node->Whirl_Tree());
    }

    // Rebuild the Parent map.
    WN_Parentize(node->Whirl_Tree(), Parent_Map, Current_Map_Tab);
    // Verify WN node.
    Is_True(WN_verifier(node->Whirl_Tree()), (""));
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static WN* HC_handle_misc_kernel_dir_walker(WN *wn, WN *parent_wn,
        IPA_NODE *node)
{
    WN *next_wn = (parent_wn != NULL && WN_operator(parent_wn) == OPR_BLOCK) ?
        WN_next(wn) : NULL;

    if (wn == NULL) return next_wn;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA)
    {
        BOOL gen_code = node->contains_kernel()
            || node->may_be_inside_kernel();

        WN_PRAGMA_ID pid = (WN_PRAGMA_ID)WN_pragma(wn);
        if (pid == WN_PRAGMA_HC_BARRIER)
        {
            next_wn = HC_lower_barrier(wn, parent_wn, gen_code);
        }

        // No need to process further.
    }
    // Handle composite node.
    else if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL)
        {
            kid_wn = HC_handle_misc_kernel_dir_walker(kid_wn, wn, node);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            Is_True(HC_handle_misc_kernel_dir_walker(WN_kid(wn,i), wn, node)
                    == NULL, (""));
        }
    }

    return next_wn;
}

void HC_handle_misc_kernel_directives(IPA_NODE *node)
{
    Is_True(node->contains_kernel() || !node->may_lead_to_kernel(), (""));

    IPA_NODE_CONTEXT context(node);

    Is_True(HC_handle_misc_kernel_dir_walker(node->Whirl_Tree(), NULL, node)
            == NULL, (""));

    // Rebuild the Parent map.
    WN_Parentize(node->Whirl_Tree(), Parent_Map, Current_Map_Tab);
    // Verify WN node.
    Is_True(WN_verifier(node->Whirl_Tree()), (""));
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * ASSUME: the context of <node> must be established.
 *
 ****************************************************************************/

static void HC_outline_kernel(IPA_NODE *node, HC_KERNEL_INFO *kinfo)
{
    // Create a new kernel function node.
    IPA_NODE *knode = IPA_Call_Graph->Add_New_Node(node->Func_ST(),
            node->File_Index(), node->Proc_Info_Index(),
            node->Summary_Proc_Index());

    // Update the orig<->clone(s) maps for IPO_Clone.
    // update_clone_orig_maps(node, clone);

    // Initialize the kernel's mempool.
    MEM_POOL_Initialize(knode->Mem_Pool(), node->Name(), 1);
    MEM_POOL_Push(knode->Mem_Pool());
    knode->Set_Mempool_Initialized();

    // Connect it to the root node.
    IPA_Call_Graph->Graph()->Add_Edge(
            IPA_Call_Graph->Root(), knode->Node_Index(), NULL);

    {
        // Swith to the original node's context.
        IPA_NODE_CONTEXT context(node);

        // The kernel function starts with a clone of the original node.
        // NOTE: PU-static variables will be kept as they are, even though
        // this is incorrect from the point of view of procedure cloning.
        BOOL saved_opt = IPA_Disable_Statics_Promotion;
        IPA_Disable_Statics_Promotion = TRUE;
        IPO_Clone(node, knode,
                ST_name_idx(St_Table[kinfo->get_kernel_sym()]));
        IPA_Disable_Statics_Promotion = saved_opt;
    }

    // Switch to the kernel function's context.
    IPA_NODE_CONTEXT context(knode);

    ST_IDX kfunc_st_idx = ST_st_idx(knode->Func_ST());

    // Construct the list of kernel parameters from its updated DAS.
    // Note that these ST_IDX's are for the parent procedure, but they should
    // be the same in the cloned procedure.
    HC_SYM_LIST *kparams_tmp = kinfo->get_kernel_params();

    // Turn these parameter symbols into formals.
    //
    // For each local variable, we just modify its SCLASS and EXPORT fields
    // directly. For each global variable, we need to create a corresponding
    // formal variable and replace all occurrences in the kernel region with
    // the new formal variable.
    //
    UINT n_kparams = kparams_tmp->Elements();
    ST_IDX kparams[n_kparams];
    TY_IDX kparam_types[n_kparams];
    // global => formal
    MEM_POOL *kmpool = knode->Mem_Pool();
    HC_SYM_MAP *formals_for_globals = CXX_NEW(HC_SYM_MAP(41,kmpool), kmpool);
    for (UINT i = 0; i < n_kparams; ++i)
    {
        ST_IDX st_idx = (*kparams_tmp)[i];
        ST &st = St_Table[st_idx];

        switch (ST_sclass(st))
        {
            case SCLASS_AUTO:
            case SCLASS_FORMAL:
            case SCLASS_FORMAL_REF:
            case SCLASS_PSTATIC:
                // a local variable (which may have a global scope)
                Is_True(ST_IDX_level(st_idx) == CURRENT_SYMTAB, (""));
                Set_ST_sclass(st, SCLASS_FORMAL);
                Set_ST_export(st, EXPORT_LOCAL_INTERNAL);
                break;

            case SCLASS_COMMON:
            case SCLASS_FSTATIC:
            case SCLASS_EXTERN:
            case SCLASS_UGLOBAL:
            {
                // a global variable
                Is_True(ST_IDX_level(st_idx) == GLOBAL_SYMTAB, (""));
                // Create a new formal variable.
                ST_IDX formal_st_idx = new_formal_var(
                        gen_var_str("f_", st_idx), ST_type(st));
                // Update the symbol map (should be unique).
                formals_for_globals->Enter(st_idx, formal_st_idx);
                st_idx = formal_st_idx;
                break;
            }

            default:
                // Do not know what to know.
                HC_warn("Do not support the type of parameter <%s> "
                        "of kernel <%s>.",
                        ST_name(st), ST_name(kfunc_st_idx));
                // For now, do the same thing as for local variables.
                Set_ST_sclass(st, SCLASS_FORMAL);
                Set_ST_export(st, EXPORT_LOCAL_INTERNAL);
                break;
        }

        kparams[i] = st_idx;
        kparam_types[i] = ST_type(st_idx);
    }

    // Create the kernel function prototype.
    TY_IDX kfunc_ty_idx = new_func_type(ST_name(kfunc_st_idx),
            MTYPE_To_TY(MTYPE_V), n_kparams, kparam_types);

    PU &kpu = Pu_Table[ST_pu(St_Table[kfunc_st_idx])];
    Set_PU_prototype(kpu, kfunc_ty_idx);

    // Mark the function as a CUDA kernel.
    Set_PU_is_kernel(kpu);
    // Mark the function symbol too.
    set_st_attr_is_kernel(kfunc_st_idx);

    if (flag_opencl){  
      // Do not output definition of kernel function
      set_st_attr_is_cuda_runtime(kfunc_st_idx);

      // Mark all types used in the kernel
      WN *k_wn = knode->Whirl_Tree();
      for (WN_ITER *wni = WN_WALK_TreeIter(k_wn); wni != NULL;
	   wni = WN_WALK_TreeNext(wni)){
        WN *wn = WN_ITER_wn(wni);
   	OPERATOR opr = WN_operator(wn);

   	//printf("ALL\n");
   	//dump_tree(wn);
   	if (opr == OPR_LDID || opr == OPR_STID){
   	  ST *st = WN_st(wn);
   	  TY_IDX ty_idx = ST_type(st);
   	  //dump_tree(wn);
   	  if (TY_kind(ty_idx) == KIND_STRUCT){
   	    Set_TY_is_used_in_kernel(ty_idx);
   	    //printf(" Type: %d\n", ty_idx);
   	  }
   	  if (TY_kind(ty_idx) == KIND_POINTER || TY_kind(ty_idx) == KIND_POINTER){
   	    TY_IDX base_ty_idx = TY_pointed(Ty_Table[ty_idx]);	
   	    if (TY_kind(base_ty_idx) == KIND_STRUCT){
   	      Set_TY_is_used_in_kernel(base_ty_idx);
   	      //printf("   Base type: %d\n", base_ty_idx);
   	    }
   	  }
   	}	
      }
    }

    // Extract the kernel body.
    WN *kernel_body = NULL;
    WN *kfunc_wn = knode->Whirl_Tree();
    for (WN_ITER *wni = WN_WALK_TreeIter(kfunc_wn); wni != NULL;
            wni = WN_WALK_TreeNext(wni))
    {
        WN *wn = WN_ITER_wn(wni);
        ST_IDX st_idx = HC_get_kernel_sym(wn);
        if (st_idx == kinfo->get_kernel_sym())
        {
            kernel_body = WN_region_body(wn);
            break;
        }
    }
    Is_True(kernel_body != NULL, (""));

    // Replace global symbol references in the kernel region with
    // corresponding formals.
    HCWN_replace_syms(kernel_body, formals_for_globals);

    // Create a new kernel function WN node.
    WN *new_kfunc_wn = WN_CreateEntry(n_kparams, kfunc_st_idx, kernel_body,
            WN_func_pragmas(kfunc_wn), WN_func_varrefs(kfunc_wn));
    // WE DO NOT DELETE THE OLD NODE.

    // Fill in the new formal list.
    for (UINT i = 0; i < n_kparams; ++i)
    {
        WN_formal(new_kfunc_wn,i) = WN_CreateIdname(0, kparams[i]);
    }

    // Save the new WHIRL tree in <knode> and update its Parent_Map.
    knode->Set_Whirl_Tree(new_kfunc_wn);
    WN_Parentize(knode->Whirl_Tree(), Parent_Map, Current_Map_Tab);

    // Save the node in the kernel info.
    kinfo->set_kernel_node(knode);
}

static void HC_insert_kernel_calls(WN *wn, IPA_NODE *node)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    ST_IDX st_idx = HC_get_kernel_sym(wn);
    if (st_idx != ST_IDX_ZERO)
    {
        // Find the kernel info.
        HC_KERNEL_INFO *kinfo = node->get_kernel_info_by_sym(st_idx);

        IPA_NODE *knode = kinfo->get_kernel_node();
        ST_IDX kfunc_st_idx = ST_st_idx(knode->Func_ST());

        // Create a block that contains the kernel execution configuration.
        WN *replacement = WN_CreateBlock();

        // Declare/Get the local grid and block dimension variables.
        HC_LOCAL_VAR_STORE *lvar_store = node->get_hc_lvar_store();
        ST_IDX grid_dim_st_idx = lvar_store->get_grid_dim_sym();
        ST_IDX tblk_dim_st_idx = lvar_store->get_tblk_dim_sym();

        // Intialize the grid dimension variable first.
        for (int i = 0; i < 3; ++i)
        {
	  if (flag_opencl){ 
	    // For open cl grid (global) size is the total number of threads, not blocks
	    // hence gridDimx = grid_dim_x X blk_dim_x
	    WN_INSERT_BlockLast(replacement, 
				HCWN_StidStructField(grid_dim_st_idx, i+1,
						     WN_Mpy(Integer_type,
							    WN_COPY_Tree(kinfo->get_grid_dim(i)),
							    WN_COPY_Tree(kinfo->get_blk_dim(i)))));
	  } else {
            WN_INSERT_BlockLast(replacement, 
                    HCWN_StidStructField(grid_dim_st_idx, i+1,
                        WN_COPY_Tree(kinfo->get_grid_dim(i))));
	  }
        }
        // Intialize the block dimension variable second.
        for (int i = 0; i < 3; ++i)
        {
            WN_INSERT_BlockLast(replacement,
                    HCWN_StidStructField(tblk_dim_st_idx, i+1,
                        WN_COPY_Tree(kinfo->get_blk_dim(i))));
        }

        // Get the parameters.
        HC_SYM_LIST *kactuals = kinfo->get_kernel_params();
        UINT n_actuals = kactuals->Elements();

	if (flag_opencl){  

	  // Generate string expression for the funtion name in name_wn
	  char *st_name = ST_name(kfunc_st_idx);
	  TCON tcon = Host_To_Targ_String(MTYPE_STRING, st_name, strlen(st_name));
	  TY_IDX cc_ty_idx = MTYPE_To_TY(MTYPE_I1);
	  Set_TY_is_const(cc_ty_idx);
	  TY_IDX ccs_ty_idx = Make_Pointer_Type(cc_ty_idx);
	  ST *name_st = New_Const_Sym(Enter_tcon(tcon), ccs_ty_idx);
	  WN *name_wn = WN_LdaZeroOffset(ST_st_idx(name_st), ccs_ty_idx);
	  
	  // Insert clCreateKernelRet call
	  WN *clCreateKernel_wn = call_clCreateKernelRet(WN_LdaZeroOffset(lvar_store->get_cl_kernel_sym()),
							 WN_LdidScalar(hc_glob_var_store.get_cl_program_sym()), 
							 name_wn,
							 WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()));
	  
	  WN_INSERT_BlockLast(replacement, clCreateKernel_wn);
	  
	  // Set kernel arguments
	  for (int i = 0; i < n_actuals; ++i){
	    ST_IDX actual_st_idx = (*kactuals)[i];
	    WN *ldid_wn = WN_LdaZeroOffset(actual_st_idx);
	    WN *clSetKernelArg_wn = call_clSetKernelArg(WN_LdidScalar(lvar_store->get_cl_kernel_sym()), 
							WN_Intconst(Integer_type, i),
							WN_Intconst(Integer_type,  TY_size(ST_type(actual_st_idx))),
							ldid_wn);
	    WN_INSERT_BlockLast(replacement, clSetKernelArg_wn);
	  }

	  // Insert clEnqueueNDRangeKernel call
	  WN *clEnqueueNDRangeKernel_wn = call_clEnqueueNDRangeKernel(WN_LdidScalar(hc_glob_var_store.get_cl_command_queue_sym()),
								      WN_LdidScalar(lvar_store->get_cl_kernel_sym()),
								      WN_Intconst(Integer_type, 3),
								      WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()),
								      WN_LdaZeroOffset(grid_dim_st_idx),
								      WN_LdaZeroOffset(tblk_dim_st_idx),
								      WN_Zerocon(Integer_type),
								      WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()),
								      WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()));
	  
	  WN_INSERT_BlockLast(replacement, clEnqueueNDRangeKernel_wn);
	} else {
	
        // The kernel execution configuration is stored in the first few
        // regular parameters of the kernel function call.
        //
        // They are regular PARM nodes 
        //
        // The first two are IDNAMEs of grid and block dimension variables.
        // The third is a INTCONST for the amount of shared memory allocated
        // for this kernel.
        //
        WN *kcall_wn = WN_Call(MTYPE_V, MTYPE_V, n_actuals+3, kfunc_st_idx);
        WN_Set_Call_Is_Kernel(kcall_wn);

        WN *actual_wn = WN_LdidScalar(grid_dim_st_idx);
        WN_kid0(kcall_wn) = HCWN_Parm(WN_desc(actual_wn), actual_wn,
                ST_type(grid_dim_st_idx));
        actual_wn = WN_LdidScalar(tblk_dim_st_idx);
        WN_kid1(kcall_wn) = HCWN_Parm(WN_desc(actual_wn), actual_wn,
                ST_type(tblk_dim_st_idx));
        actual_wn = WN_Intconst(Integer_type, kinfo->get_smem_size());
        WN_kid2(kcall_wn) = HCWN_Parm(Integer_type, actual_wn, 
                MTYPE_To_TY(Integer_type));
#if 0
        WN_kid0(kcall_wn) = HCWN_Parm( WN_CreateIdname(0, grid_dim_st_idx);
        WN_kid1(kcall_wn) = WN_CreateIdname(0, tblk_dim_st_idx);
        WN_kid2(kcall_wn) =
            WN_Intconst(Integer_type, kinfo->get_smem_size());
#endif
        // The remaining kids are arguments.
        for (int i = 0; i < n_actuals; ++i)
        {
            ST_IDX actual_st_idx = (*kactuals)[i];
            WN *ldid_wn = WN_LdidScalar(actual_st_idx);
            WN_kid(kcall_wn, i+3) =
                HCWN_Parm(WN_desc(ldid_wn), ldid_wn, ST_type(actual_st_idx));
        }
				       
        WN_INSERT_BlockLast(replacement, kcall_wn);
	}			       
        // Insert the block before the kernel region.
        WN *parent_wn = LWN_Get_Parent(wn);
        WN_INSERT_BlockBefore(parent_wn, wn, replacement);
        WN_DELETE_FromBlock(parent_wn, wn);
        WN_Parentize(parent_wn, Parent_Map, Current_Map_Tab);

        return;
    }

    // Handle composite node.
    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            WN *next_wn = WN_next(kid_wn);
            HC_insert_kernel_calls(kid_wn, node);
            kid_wn = next_wn;
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HC_insert_kernel_calls(WN_kid(wn,i), node);
        }
    }
}

void HC_outline_kernels(IPA_NODE *node)
{
    if (!node->contains_kernel()) return;

    // Rebuild the kernels' scalar DAS.
    HC_rebuild_kernel_scalar_das(node);

    // THIS IS IMPORTANT!!
    IPA_NODE_CONTEXT context(node);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "%s", SBar);
        fdump_tree(TFile, node->Whirl_Tree());
        fprintf(TFile, "%s", SBar);
    }

    // Create kernel functions.
    UINT n_kregions = node->num_kregions();
    Is_True(n_kregions > 0, (""));
    HC_KERNEL_INFO_LIST *kil = node->get_kernel_info_list();
    for (UINT i = 0; i < n_kregions; ++i) HC_outline_kernel(node, (*kil)[i]);

    // Verify WN node.
    HCWN_check_map_id(node->Whirl_Tree());
    Is_True(WN_verifier(node->Whirl_Tree()), (""));
    HCWN_check_parentize(node->Whirl_Tree(), Parent_Map);

    // Replace original kernel regions with execution configurations.
    HC_insert_kernel_calls(node->Whirl_Tree(), node);

    // Verify WN node.
    HCWN_check_map_id(node->Whirl_Tree());
    Is_True(WN_verifier(node->Whirl_Tree()), (""));

    // Rebuild the Parent_Map.
    WN_Parentize(node->Whirl_Tree(), Parent_Map, Current_Map_Tab);
    HCWN_check_parentize(node->Whirl_Tree(), Parent_Map);

    // Reset WN-to-IPA_EDGE map.
    IPA_Call_Graph->Reset_Callsite_Map(node);
}

#endif  // IPA_HICUDA

/*** DAVID CODE END ***/
