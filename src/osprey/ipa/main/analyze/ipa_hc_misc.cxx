/** DAVID CODE BEGIN **/

// needed by ipl_summary.h included in ipa_cg.h
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "wn.h"

#include "ipa_hc_misc.h"

#include "ipo_defs.h"       // IPA_NODE_CONTEXT

#include "hc_utils.h"
#include "cuda_utils.h"


ST_IDX HC_LOCAL_VAR_STORE::get_loop_idx_var(UINT nesting_level)
{
    // Create/resize the list of loop index variables if necessary.
    if (_loop_idx_vars == NULL)
    {
        _n_nesting_levels = nesting_level*2;
        _loop_idx_vars = CXX_NEW_ARRAY(ST_IDX, _n_nesting_levels, _pool);
        for (UINT i = 0; i < _n_nesting_levels; ++i) {
            _loop_idx_vars[i] = ST_IDX_ZERO;
        }
    }
    else if (nesting_level >= _n_nesting_levels)
    {
        UINT new_n_nesting_levels = nesting_level*2;
        ST_IDX *new_loop_idx_vars = CXX_NEW_ARRAY(ST_IDX,
                new_n_nesting_levels, _pool);
        UINT i = 0;
        for ( ; i < _n_nesting_levels; ++i) {
            new_loop_idx_vars[i] = _loop_idx_vars[i];
        }
        for ( ; i < new_n_nesting_levels; ++i) {
            new_loop_idx_vars[i] = ST_IDX_ZERO;
        }

        _n_nesting_levels = new_n_nesting_levels;
        CXX_DELETE(_loop_idx_vars, _pool);
        _loop_idx_vars = new_loop_idx_vars;
    }

    ST_IDX idxv_st_idx = _loop_idx_vars[nesting_level];
    if (idxv_st_idx == ST_IDX_ZERO)
    {
        IPA_NODE_CONTEXT context(_proc_node);   // IMPORTANT!
        idxv_st_idx = make_loop_idx(nesting_level, false);
        _loop_idx_vars[nesting_level] = idxv_st_idx;
    }

    return idxv_st_idx;
}

ST_IDX HC_LOCAL_VAR_STORE::get_sym(ST_IDX& st_idx,
        const char *st_name, TY_IDX ty_idx)
{
    if (st_idx == ST_IDX_ZERO)
    {
        // Create a new local variable.
        IPA_NODE_CONTEXT context(_proc_node);   // IMPORTANT!
        st_idx = new_local_var(st_name, ty_idx);
    }

    return st_idx;
}

ST_IDX HC_LOCAL_VAR_STORE::get_grid_dim_sym()
{
    Is_True(dim3_ty_idx != TY_IDX_ZERO, (""));
    return get_sym(_grid_dim_st_idx, "dimGrid", dim3_ty_idx);
}

ST_IDX HC_LOCAL_VAR_STORE::get_tblk_dim_sym()
{
    Is_True(dim3_ty_idx != TY_IDX_ZERO, (""));
    return get_sym(_tblk_dim_st_idx, "dimBlock", dim3_ty_idx);
}

// OpenCL local variables
ST_IDX HC_LOCAL_VAR_STORE::get_cl_kernel_sym()
{
    Is_True(cl_kernel_ty_idx != TY_IDX_ZERO, (""));
    return get_sym(_cl_kernel_st_idx, "__cl_kernel", cl_kernel_ty_idx);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

HC_GLOBAL_VAR_STORE hc_glob_var_store;

static TY_IDX HC_gen_default_elem_type(UINT elem_sz)
{
    TYPE_ID mtype;
    switch (elem_sz)
    {
        case 1: mtype = MTYPE_U1; break;
        case 2: mtype = MTYPE_U2; break;
        case 4: mtype = MTYPE_U4; break;
        case 8: mtype = MTYPE_U8; break;
        default:
                Is_True(FALSE,
                        ("Invalid element size: %d bytes\n", elem_sz));
    }

    return MTYPE_To_TY(mtype);
}

ST_IDX HC_GLOBAL_VAR_STORE::create_cmem_sym(UINT n_elems, UINT elem_sz)
{
    Is_True(_cmem_st_idx == ST_IDX_ZERO, (""));

    // Use a default element type for the given size.
    TY_IDX elem_ty_idx = HC_gen_default_elem_type(elem_sz);

    // Create the type of <cmem>.
    TY_IDX cmem_ty_idx = make_arr_type(Save_Str(".cmem.type"),
            1, &n_elems, elem_ty_idx);

    _cmem_st_idx = new_global_var("cmem", cmem_ty_idx);
    // Mark it with a special flag for CUDA code generation.
    set_st_attr_is_const_var(_cmem_st_idx);

    return _cmem_st_idx;
}

ST_IDX HC_GLOBAL_VAR_STORE::create_smem_sym(UINT elem_sz)
{
    Is_True(_smem_st_idx == ST_IDX_ZERO, (""));

    // Use a default element type for the given size.
    TY_IDX elem_ty_idx = HC_gen_default_elem_type(elem_sz);

    // Create the type of <smem>.
    TY_IDX smem_ty_idx = make_incomplete_arr_type(Save_Str(".smem.type"),
            elem_ty_idx);

    _smem_st_idx = new_extern_var("smem", smem_ty_idx);
    // Mark it with a special flag for CUDA code generation.
    set_st_attr_is_shared_var(_smem_st_idx);

    return _smem_st_idx;
}

// OpenCL global variables

ST_IDX HC_GLOBAL_VAR_STORE::create_cl_program_sym()
{
    Is_True(_cl_program_st_idx == ST_IDX_ZERO, (""));
    _cl_program_st_idx = new_global_var("__cl_program", cl_program_ty_idx);
    return _cl_program_st_idx;
}


ST_IDX HC_GLOBAL_VAR_STORE::create_cl_command_queue_sym()
{
    Is_True(_cl_command_queue_st_idx == ST_IDX_ZERO, (""));
    _cl_command_queue_st_idx = new_global_var("__cl_queue", cl_command_queue_ty_idx);
    return _cl_command_queue_st_idx;
}

ST_IDX HC_GLOBAL_VAR_STORE::create_cl_context_sym()
{
    Is_True(_cl_command_queue_st_idx == ST_IDX_ZERO, (""));
    _cl_context_st_idx = new_global_var("__cl_context", cl_context_ty_idx);
    return _cl_context_st_idx;
}

// OpenCL constants

ST_IDX HC_GLOBAL_VAR_STORE::create_cl_null_sym()
{
    Is_True(_cl_null_st_idx == ST_IDX_ZERO, (""));
    _cl_null_st_idx = new_extern_var("NULL", Make_Pointer_Type(MTYPE_To_TY(MTYPE_U4)));	
    set_st_attr_is_cuda_runtime(_cl_null_st_idx);
    return _cl_null_st_idx;
}

ST_IDX HC_GLOBAL_VAR_STORE::create_cl_local_mem_fence_sym()
{
    Is_True(_cl_local_mem_fence_st_idx == ST_IDX_ZERO, (""));
    _cl_local_mem_fence_st_idx = new_extern_var("CLK_LOCAL_MEM_FENCE", MTYPE_To_TY(MTYPE_U4));	
    set_st_attr_is_cuda_runtime(_cl_local_mem_fence_st_idx);
    return _cl_local_mem_fence_st_idx;
}

ST_IDX HC_GLOBAL_VAR_STORE::create_cl_false_sym()
{
    Is_True(_cl_false_st_idx == ST_IDX_ZERO, (""));
    _cl_false_st_idx = new_extern_var("CL_FALSE", MTYPE_To_TY(MTYPE_U4));	
    set_st_attr_is_cuda_runtime(_cl_false_st_idx);
    return _cl_false_st_idx;
}

ST_IDX HC_GLOBAL_VAR_STORE::create_cl_true_sym()
{
    Is_True(_cl_true_st_idx == ST_IDX_ZERO, (""));
    _cl_true_st_idx = new_extern_var("CL_TRUE",MTYPE_To_TY(MTYPE_U4));	
    set_st_attr_is_cuda_runtime(_cl_true_st_idx);
    return _cl_true_st_idx;
}

ST_IDX HC_GLOBAL_VAR_STORE::create_cl_mem_read_write_sym()
{
    Is_True(_cl_mem_read_write_st_idx == ST_IDX_ZERO, (""));
    _cl_mem_read_write_st_idx = new_extern_var("CL_MEM_READ_WRITE", MTYPE_To_TY(MTYPE_U4));	
    set_st_attr_is_cuda_runtime(_cl_mem_read_write_st_idx);
    return _cl_mem_read_write_st_idx;
}

ST_IDX HC_GLOBAL_VAR_STORE::create_cl_mem_read_only_sym()
{
    Is_True(_cl_mem_read_only_st_idx == ST_IDX_ZERO, (""));
    _cl_mem_read_only_st_idx = new_extern_var("CL_MEM_READ_ONLY", MTYPE_To_TY(MTYPE_U4));	
    set_st_attr_is_cuda_runtime(_cl_mem_read_only_st_idx);
    return _cl_mem_read_only_st_idx;
}

ST_IDX HC_GLOBAL_VAR_STORE::create_cl_mem_copy_host_ptr_sym()
{
    Is_True(_cl_mem_copy_host_ptr_st_idx == ST_IDX_ZERO, (""));
    _cl_mem_copy_host_ptr_st_idx  = new_extern_var("CL_MEM_COPY_HOST_PTR", MTYPE_To_TY(MTYPE_U4));   
    set_st_attr_is_cuda_runtime(_cl_mem_copy_host_ptr_st_idx);
    return _cl_mem_copy_host_ptr_st_idx;
}


/*** DAVID CODE END ***/
