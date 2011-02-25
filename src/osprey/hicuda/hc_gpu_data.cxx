/** DAVID CODE BEGIN **/

#include "defs.h"
#include "tracing.h"            // for TDEBUG_HICUDA
#include "wn.h"
#include "wn_util.h"

#include "hc_gpu_data.h"
#include "ipa_hc_gpu_data.h"
#include "ipa_hc_kernel.h"
#include "ipa_hc_misc.h"

#include "hc_utils.h"
#include "hc_expr.h"
#include "cuda_utils.h"

#include "assert.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Construct a BLOCK node that contains code for copying between the host and
 * global memory, for a global/constant memory variable.
 *
 * If <copyin> is true, the direction is from host to global/constant, and the
 * other way if false.
 *
 ****************************************************************************/

static WN* HC_make_gvar_transfer_code(HC_GPU_DATA *gdata, BOOL copyin,
        HC_LOCAL_VAR_STORE *lvar_store)
{
    ST_IDX var_st_idx = gdata->get_symbol();
    HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();
    Is_True(gvi != NULL, (""));
    ST_IDX gvar_st_idx = gvi->get_symbol();

    // Is this a global memory variable or a const memory variable?
    BOOL is_global_var = st_attr_is_global_var(gvar_st_idx);
    if (is_global_var) Is_True(gdata->get_type() == HC_GLOBAL_DATA, (""));

    TY_IDX var_ty_idx = ST_type(var_st_idx);
    TY_KIND var_ty_kind = TY_kind(var_ty_idx);

    // Create an empty block.
    WN *memcpy_blk = WN_CreateBlock();

    // copy direction
    enum cudaMemcpyKind copy_direction = (copyin ?
            cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);
    // We only support copyin for a constant memory variable.
    Is_True(is_global_var || copyin, (""));

    /* Handle the non-array case first. */
    if (! gdata->is_arr_section())
    {
        WN *tcall_wn = NULL;

        // Type of the original variable is: the object type
        WN *var_wn = WN_LdaZeroOffset(var_st_idx);

        if (is_global_var)
        {
            // Type of the global variable is a pointer to the object.
            // assert(TY_kind(gvar_ty_idx) == KIND_POINTER);
            WN *gvar_wn = WN_LdidScalar(gvar_st_idx);

            WN *src_wn = NULL, *dst_wn = NULL;
            if (copyin) {
                src_wn = var_wn; dst_wn = gvar_wn;
            } else {
                src_wn = gvar_wn; dst_wn = var_wn;
            }

	    if (flag_opencl){
	      if (copyin) {
		tcall_wn = call_clEnqueueWriteBuffer(WN_LdidScalar((hc_glob_var_store.get_cl_command_queue_sym())),
						     gvar_wn, 
						     WN_LdidScalar(hc_glob_var_store.get_cl_false_sym()),
						     WN_Zerocon(Integer_type),
						     WN_COPY_Tree(gvi->get_size()),
						     var_wn,
						     WN_Zerocon(Integer_type),
						     WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()),
						     WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()));
	      } else {
		tcall_wn = call_clEnqueueReadBuffer(WN_LdidScalar(hc_glob_var_store.get_cl_command_queue_sym()), 
						    gvar_wn, 
						    WN_LdidScalar(hc_glob_var_store.get_cl_true_sym()),
						    WN_Zerocon(Integer_type),
						    WN_COPY_Tree(gvi->get_size()),
						    var_wn,
						    WN_Zerocon(Integer_type),
						    WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()),
						    WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()));
	      }
	    } else {
	      tcall_wn = call_cudaMemcpy(dst_wn, src_wn,
					 WN_COPY_Tree(gvi->get_size()), copy_direction);
	    }
        }
        else
        {
	    if (flag_opencl){
	      // OpenCL not implemented here.
	      assert(0);
	    } else {
	      // We must use the global <cmem> variable here, instead of the
	      // local constant memory pointer variable.
	      ST_IDX cmem_st_idx = hc_glob_var_store.get_cmem_sym();
	      WN *gvar_ofst_wn = WN_Intconst(Integer_type, gvi->get_offset());
	      // Create a cudaMemcpyToSymbol call.
	      tcall_wn = call_cudaMemcpyToSymbol(cmem_st_idx, var_wn,
						 WN_COPY_Tree(gvi->get_size()),
						 gvar_ofst_wn, copy_direction);
	    }
        }

        WN_INSERT_BlockFirst(memcpy_blk, tcall_wn);
        return memcpy_blk;
    }

    /* Gather info about the original, allocated and copy sections. */

    HC_ARRSECTION_INFO *alloc_section = gdata->get_alloc_section();
    Is_True(alloc_section != NULL, (""));
    HC_ARRSECTION_INFO *copy_section = copyin ?
        gdata->get_copyin_section() : gdata->get_copyout_section();
    // If the copy section is omitted, it is the allocated section.
    if (copy_section == NULL) copy_section = alloc_section;

    UINT ndims = alloc_section->get_num_dim();
    INT64 elem_sz = TY_size(alloc_section->get_elem_type());

    WN* var_dim_sz[ndims];
    WN* gvar_dim_sz[ndims];
    WN* copy_dim_sz[ndims];
    for (UINT i = 0; i < ndims; ++i)
    {
        var_dim_sz[i] = alloc_section->get_orig_dim_sz(i);  // reference
        gvar_dim_sz[i] = alloc_section->get_dim_sz(i);      // fresh instance
        copy_dim_sz[i] = copy_section->get_dim_sz(i);       // fresh instance
    }

    // Given that the copy section is within the allocated section, checking
    // that a dimension in the copy section is "full" w.r.t. the original
    // array, it must be "full" w.r.t. the allocated array.
    UINT pdim_idx = copy_section->get_pivot_dim_idx();

    /* Construct the local variable <stride>. */

    WN *stride_wn = WN_Intconst(Integer_type, 1);
    for (UINT i = pdim_idx+1; i < ndims; ++i)
    {
        stride_wn = WN_Mpy(Integer_type,
                stride_wn, WN_COPY_Tree(var_dim_sz[i]));
    }

    // Create the initialization stmt.
    ST_IDX stride_st_idx = lvar_store->get_stride_sym();
    WN_INSERT_BlockLast(memcpy_blk,
            WN_StidScalar(ST_ptr(stride_st_idx), stride_wn));

    /* Construct the local variable <batsz> that stores the batch size.
     * We can reuse <stride> in the expression.
     */

    // The batch size is in bytes.
    WN *batsz_wn = WN_Mpy(Integer_type,
            WN_Mpy(Integer_type,
                WN_COPY_Tree(copy_dim_sz[pdim_idx]),
                WN_LdidScalar(stride_st_idx)),
            WN_Intconst(Integer_type, elem_sz));

    // Create the initialization stmt.
    ST_IDX batsz_st_idx = lvar_store->get_batsz_sym();
    WN_INSERT_BlockLast(memcpy_blk,
            WN_StidScalar(ST_ptr(batsz_st_idx), batsz_wn));

    /* Construct a loop nest (dimension 0 to <pdim_idx-1>). */

    WN *parent_loop_wn = NULL;
    ST_IDX idxvs[pdim_idx];
    for (INT i = 0; i < pdim_idx; ++i)
    {
        // Create a loop index variable.
        idxvs[i] = lvar_store->get_loop_idx_var(i);

        // The loop iteration space is the index range of the copy section's
        // corresponding dimension.
        WN *loop_wn = make_empty_doloop(idxvs[i],
                copy_section->get_dim_lbnd(i), copy_section->get_dim_ubnd(i));

        if (parent_loop_wn == NULL) {
            // outer-most loop.
            WN_INSERT_BlockLast(memcpy_blk, loop_wn);
        } else {
            // Insert it into the parent loop's body.
            WN_INSERT_BlockLast(WN_kid(parent_loop_wn,4), loop_wn);
        }

        parent_loop_wn = loop_wn;
    }
    // Now, <parent_loop_wn> stores the innermost loop, which could be NULL.
    WN *parent_blk = (parent_loop_wn == NULL) ?
        memcpy_blk : WN_kid(parent_loop_wn,4);

    /* Construct the local variable <goffset> and <hoffset>, which store the
     * offset for the GPU variable and the host variable respectively.
     */

    WN *goffset_wn = WN_Intconst(Integer_type, 0);
    WN *hoffset_wn = WN_Intconst(Integer_type, 0);
    for (UINT i = 0; i < pdim_idx; ++i)
    {
        goffset_wn = WN_Mpy(Integer_type,
                WN_Add(Integer_type, goffset_wn,
                    WN_Sub(Integer_type, WN_LdidScalar(idxvs[i]),
                        WN_COPY_Tree(alloc_section->get_dim_lbnd(i)))),
                WN_COPY_Tree(gvar_dim_sz[i+1]));
        hoffset_wn = WN_Mpy(Integer_type,
                WN_Add(Integer_type, hoffset_wn, WN_LdidScalar(idxvs[i])),
                WN_COPY_Tree(var_dim_sz[i+1]));
    }
    goffset_wn = WN_Add(Integer_type, goffset_wn,
            WN_Sub(Integer_type,
                WN_COPY_Tree(copy_section->get_dim_lbnd(pdim_idx)),
                WN_COPY_Tree(alloc_section->get_dim_lbnd(pdim_idx))));
    hoffset_wn = WN_Add(Integer_type, hoffset_wn,
            WN_COPY_Tree(copy_section->get_dim_lbnd(pdim_idx)));

    /* Construct the data transfer call:
     * - cudaMemcpy for global memory variable,
     * - cudaMemcpyToSymbol for constant memory variable.
     */

    // Construct the access to the host variable:
    //      var + <hoffset_wn> * <hstride>
    WN *var_base_wn = (var_ty_kind == KIND_ARRAY) ?
        WN_LdaZeroOffset(var_st_idx) : WN_LdidScalar(var_st_idx);
    // Here, the offset must be in bytes.
    WN *var_ofst_wn = WN_Mpy(Integer_type,
            WN_Mpy(Integer_type, hoffset_wn, WN_LdidScalar(stride_st_idx)),
            WN_Intconst(Integer_type, elem_sz));
    WN *var_access_wn = WN_Add(Pointer_type, var_base_wn, var_ofst_wn);

    // Compute the access offset (in bytes) to the GPU memory variable.
    WN *gvar_ofst_wn = WN_Mpy(Integer_type,
            WN_Mpy(Integer_type, goffset_wn, WN_LdidScalar(stride_st_idx)),
            WN_Intconst(Integer_type, elem_sz));

    WN *tcall_wn = NULL;
    if (is_global_var)
    {
        // Construct the GPU variable access: gvar + <goffset> * <gstride>
        WN *gvar_access_wn = WN_Add(Pointer_type,
                WN_LdidScalar(gvar_st_idx), gvar_ofst_wn);

        WN *dst_wn = NULL, *src_wn = NULL;
        if (copyin) {
            dst_wn = gvar_access_wn; src_wn = var_access_wn;
        } else {
            dst_wn = var_access_wn; src_wn = gvar_access_wn;
        }

	if (flag_opencl){
	  if (copyin) {
	    tcall_wn = call_clEnqueueWriteBuffer(WN_LdidScalar((hc_glob_var_store.get_cl_command_queue_sym())),
						 gvar_access_wn, 
						 WN_LdidScalar(hc_glob_var_store.get_cl_false_sym()),
						 WN_Zerocon(Integer_type),
						 WN_LdidScalar(batsz_st_idx),
						 var_access_wn,
						 WN_Zerocon(Integer_type),
						 WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()),
						 WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()));
	  } else {
	    tcall_wn = call_clEnqueueReadBuffer(WN_LdidScalar(hc_glob_var_store.get_cl_command_queue_sym()), 
						gvar_access_wn, 
						WN_LdidScalar(hc_glob_var_store.get_cl_true_sym()),
						WN_Zerocon(Integer_type),
						WN_LdidScalar(batsz_st_idx),
						var_access_wn,
						WN_Zerocon(Integer_type),
						WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()),
						WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()));
	  }
	} else {
	  // Create a cudaMemcpy call.
	  tcall_wn = call_cudaMemcpy(dst_wn, src_wn,
				     WN_LdidScalar(batsz_st_idx), copy_direction);
	}
    }
    else
    {
	if (flag_opencl){
	  tcall_wn = call_clCreateBufferRet(WN_LdaZeroOffset(gvar_st_idx),
					    WN_LdidScalar(hc_glob_var_store.get_cl_context_sym()),
					    WN_Bior(MTYPE_U4, 
						    WN_LdidScalar(hc_glob_var_store.get_cl_mem_read_only_sym()),
						    WN_LdidScalar(hc_glob_var_store.get_cl_mem_copy_host_ptr_sym())
						    ),
					    WN_COPY_Tree(gvi->get_size()),
					    var_access_wn,
					    WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()));
	} else {
	  // We must use the global <cmem> variable here, instead of the
	  // local constant memory pointer variable.
	  ST_IDX cmem_st_idx = hc_glob_var_store.get_cmem_sym();
	  gvar_ofst_wn = WN_Add(Integer_type, gvar_ofst_wn,
				WN_Intconst(Integer_type, gvi->get_offset()));
	  // Create a cudaMemcpyToSymbol call.
	  tcall_wn = call_cudaMemcpyToSymbol(cmem_st_idx, var_access_wn,
					     WN_LdidScalar(batsz_st_idx),
					     gvar_ofst_wn, cudaMemcpyHostToDevice);
	}
    }

    // Insert the transfer call in the parent loop.
    WN_INSERT_BlockLast(parent_blk, tcall_wn);

    return memcpy_blk;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

WN* HC_lower_global_copyin(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "Start lowering a GLOBAL COPYIN directive (%p) ...\n", gdata);
    }

    Is_True(gdata != NULL, (""));
    Is_True(parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK,
            ("HC_lower_global_copyin: NULL parent block\n"));

    // Save the insertion point.
    WN *insert_pt = pragma_wn;

    /* Advance the pragmas and remove them. */

    // base pragma
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_PRAGMA,
            ("HC_lower_global_copyin: invalid base pragma\n"));
    // the ALLOC clause
    pragma_wn = WN_next(pragma_wn);
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_XPRAGMA
            && WN_kid0(pragma_wn) != NULL,
            ("HC_lower_global_copyin: invalid ALLOC pragma\n"));
    // the COPYIN clause
    pragma_wn = WN_next(pragma_wn);
    if (pragma_wn != NULL && WN_pragma(pragma_wn) == WN_PRAGMA_HC_COPY)
    {
        Is_True(gdata->do_copyin(), (""));
        pragma_wn = WN_next(pragma_wn);
    }
    // Now, <pragma_wn> points to the WN node after the last pragma.

    // Remove all pragmas from the insertion point (the 1st pragma)
    // to the current one (exclusive).
    WN *next_wn = NULL;
    while (insert_pt != pragma_wn)
    {
        next_wn = WN_next(insert_pt);
        WN_DELETE_FromBlock(parent_wn, insert_pt);
        insert_pt = next_wn;
    }
    // <insert_pt> == <pragma_wn>. All new code will be inserted before it.

    if (! gen_code)
    {
        printf("Finished lowering the GLOBAL COPYIN directive "
                "(no code generation).\n");
        return pragma_wn;
    }

    /* Start code generation. */

    // Get the variable symbol and type.
    ST_IDX var_st_idx = gdata->get_symbol();
    TY_IDX var_ty_idx = ST_type(var_st_idx);

    // We only handle array, scalar and struct.
    TY_KIND kind = TY_kind(var_ty_idx);
    Is_True(kind == KIND_SCALAR || kind == KIND_STRUCT
            || kind == KIND_ARRAY || kind == KIND_POINTER,
            ("HC_lower_global_copyin: invalid type (%s) of symbol <%s>\n",
             TY_name(var_ty_idx), ST_name(var_st_idx)));

    // Declare a new global memory variable and determine its size in bytes.
    WN *gvar_sz_wn = NULL;
    if (gdata->is_arr_section())
    {
        // array or pointer (treated as array)
        Is_True(kind == KIND_ARRAY || kind == KIND_POINTER,
                ("HC_lower_global_copyin: "
                 "invalid type for array section symbol <%s>\n",
                 ST_name(var_st_idx)));

        HC_ARRSECTION_INFO *alloc_section = gdata->get_alloc_section();

        gvar_sz_wn = alloc_section->get_section_sz();
    }
    else
    {
        // scalar or struct
        Is_True(kind == KIND_SCALAR || kind == KIND_STRUCT,
                ("HC_lower_global_copyin: "
                 "invalid type for scalar variable <%s>\n",
                 ST_name(var_st_idx)));

        gvar_sz_wn = WN_Intconst(Integer_type, TY_size(var_ty_idx));
    }

    // The global memory variable is a pointer to the element type of the
    // original array variable.
    ST_IDX gvar_st_idx = new_local_var(gen_var_str("g_", var_st_idx),
            gdata->create_gvar_type());

    // The global memory variable must be marked with a special flag.
    set_st_attr_is_global_var(gvar_st_idx);

    // Create the GPU variable record.
    HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();
    Is_True(gvi != NULL, (""));
    gvi->set_symbol(gvar_st_idx);
    gvi->set_size(gvar_sz_wn);

    if (flag_opencl){
      WN* wcall = call_clCreateBufferRet(WN_LdaZeroOffset(gvar_st_idx),
					 WN_LdidScalar(hc_glob_var_store.get_cl_context_sym()),
					 WN_LdidScalar(hc_glob_var_store.get_cl_mem_read_write_sym()),
					 WN_COPY_Tree(gvar_sz_wn),
					 WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()),
					 WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()));
      
      WN_INSERT_BlockBefore(parent_wn,insert_pt, wcall);
    } else {
      // Construct a call that allocates the global variable, and insert it
      // to the parent block.
      WN_INSERT_BlockBefore(parent_wn, insert_pt,
			    call_cudaMalloc(WN_LdaZeroOffset(ST_ptr(gvar_st_idx)),
					    WN_COPY_Tree(gvar_sz_wn)));
    }      

    // Generate the code for copying the data from the host memory to
    // the global memory.
    if (gdata->do_copyin())
    {
        WN *memcpy_blk = HC_make_gvar_transfer_code(gdata, TRUE, lvar_store);
        // Insert the resulting BLOCK node after cudaMalloc.
        WN_INSERT_BlockBefore(parent_wn, insert_pt, memcpy_blk);
    }
    else if (gdata->do_clear())
    {
      if (flag_opencl){
	WN_INSERT_BlockBefore(parent_wn, insert_pt,
			      call_clEnqueueWriteCleanBuffer(WN_LdidScalar(hc_glob_var_store.get_cl_command_queue_sym()),
							     WN_LdidScalar(gvar_st_idx), 
							     WN_LdidScalar(hc_glob_var_store.get_cl_false_sym()),
							     WN_Zerocon(Integer_type),
							     WN_COPY_Tree(gvar_sz_wn),
							     WN_LdidScalar(var_st_idx),
							     WN_Zerocon(Integer_type),
							     WN_LdidScalar(hc_glob_var_store.get_cl_null_sym()),
							     WN_LdidScalar(hc_glob_var_store.get_cl_null_sym())));
      } else {
	WN_INSERT_BlockBefore(parent_wn, insert_pt,
                call_cudaMemset(WN_LdidScalar(gvar_st_idx),
                    WN_Intconst(Integer_type, 0), WN_COPY_Tree(gvar_sz_wn)));
      }
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the GLOBAL COPYIN directive.\n");
    }

    return pragma_wn;
}

WN* HC_lower_global_copyout(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "Start lowering a GLOBAL COPYOUT directive (%p) ...\n",
                gdata);
    }

    Is_True(gdata != NULL, (""));
    Is_True(gdata->do_copyout(), (""));
    Is_True(parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK, (""));

    // Validate the COPYOUT field.
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_XPRAGMA, (""));
    WN *copy_wn = WN_kid0(pragma_wn);
    Is_True(copy_wn != NULL, (""));

    OPERATOR opr = WN_operator(copy_wn);
    Is_True(opr == OPR_ARRSECTION || opr == OPR_LDA, (""));
    if (opr == OPR_ARRSECTION) Is_True(gdata->is_arr_section(), (""));

    // Remove this pragma.
    WN *next_wn = WN_next(pragma_wn);
    WN_DELETE_FromBlock(parent_wn, pragma_wn);

    // Early exit.
    if (! gen_code)
    {
        if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
        {
            fprintf(TFile, "Finished lowering the GLOBAL COPYOUT directive "
                    "(no code generation).\n");
        }
        return next_wn;
    }

    /* Start code generation. */

    // Get the variable symbol and type.
    ST_IDX var_st_idx = gdata->get_symbol();
    TY_IDX var_ty_idx = ST_type(var_st_idx);

    // We only handle array, scalar and struct.
    TY_KIND kind = TY_kind(var_ty_idx);
    Is_True(kind == KIND_SCALAR || kind == KIND_STRUCT
            || kind == KIND_ARRAY || kind == KIND_POINTER,
            ("HC_lower_global_copyout: invalid type (%s) of symbol <%s>\n",
             TY_name(var_ty_idx), ST_name(var_st_idx)));

    // Generate the data transfer code.
    WN *memcpy_blk = HC_make_gvar_transfer_code(gdata, FALSE, lvar_store);
    // Insert the resulting BLOCK node at the place of the pragma.
    WN_INSERT_BlockBefore(parent_wn, next_wn, memcpy_blk);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the GLOBAL COPYOUT directive.\n");
    }

    return next_wn;
}

WN* HC_lower_global_free(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, BOOL gen_code)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "Start lowering a GLOBAL FREE directive (%p) ...\n", gdata);
    }

    Is_True(gdata != NULL, (""));
    Is_True(parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK,
            ("HC_lower_global_free: NULL parent block\n"));

    // Save the insertion point (next node).
    WN *insert_pt = WN_next(pragma_wn);

    // Validate and remove the pragma.
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_PRAGMA, (""));
    ST_IDX var_st_idx = WN_st_idx(pragma_wn);
    Is_True(var_st_idx == gdata->get_symbol(), (""));
    WN_DELETE_FromBlock(parent_wn, pragma_wn);

    if (! gen_code)
    {
        if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
        {
            fprintf(TFile, "Finished lowering the GLOBAL COPYIN directive "
                    "(no code generation).\n");
        }
        return insert_pt;
    }

    /* Start code generation. */

    // Create a call to cudaFree and add it before the pragma.
    ST_IDX gvar_st_idx = gdata->get_gvar_info()->get_symbol();
    TY_IDX gvar_ty_idx = ST_type(gvar_st_idx);

    if (flag_opencl){
      WN_INSERT_BlockBefore(parent_wn, insert_pt,
			    call_clReleaseMemObj(WN_LdidScalar(gvar_st_idx)));
    } else {
      WN_INSERT_BlockBefore(parent_wn, insert_pt,
			    call_cudaFree(WN_Ldid(Pointer_type, 0, gvar_st_idx, gvar_ty_idx)));
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the GLOBAL FREE directive.\n");
    }

    return insert_pt;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * At this stage, HC_GPU_VAR_INFO must have been created. The size of the
 * constant memory variable and its offset at the global <cmem> have been
 * determined.
 *
 * We create a local constant memory pointer variable here, but do not
 * initialize it because it must be initialized in every kernel region or
 * local procedure that needs to redirect to this variable.
 *
 * Return a BLOCK node that contains the code to be inserted before the
 * pragma.
 *
 ****************************************************************************/

static WN* HC_gen_code_for_const_copyin(WN *pragma_wn, HC_GPU_DATA *gdata,
        HC_LOCAL_VAR_STORE *lvar_store)
{
    HC_create_local_cvar(gdata);

    Is_True(gdata->do_copyin(), (""));
    return HC_make_gvar_transfer_code(gdata, TRUE, lvar_store);
}

// Assume that the local procedure context is set up.
//
void HC_create_local_cvar(HC_GPU_DATA *gdata)
{
    HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();
    Is_True(gvi != NULL && gvi->get_symbol() == ST_IDX_ZERO, (""));

    // Declare the constant memory variable (local).
    ST_IDX var_st_idx = gdata->get_symbol();
    ST_IDX cvar_st_idx = new_local_var(gen_var_str("c_", var_st_idx),
            gdata->create_gvar_type());
    if (flag_opencl){ 
      set_st_attr_is_const_var(cvar_st_idx);
    } else {
      // It is just a regular variable (no need for a special flag).
      // set_st_attr_is_const_var(cvar_st_idx);
    }
    // Save it in the GPU variable record.
    gvi->set_symbol(cvar_st_idx);
}

WN* HC_create_cvar_init_stmt(const HC_GPU_VAR_INFO *gvi)
{
    ST_IDX cvar_st_idx = gvi->get_symbol();
    Is_True(cvar_st_idx != ST_IDX_ZERO, (""));
    TY_IDX cvar_ty_idx = ST_type(cvar_st_idx);

    // Generate an assignment that initializes the constant memory variable
    // with <cmem> + <offset>.
    ST_IDX cmem_st_idx = hc_glob_var_store.get_cmem_sym();
    WN *cmem_wn = WN_CreateLda(OPR_LDA, Pointer_type, MTYPE_V, 0,
            cvar_ty_idx, cmem_st_idx, 0);
    return WN_StidScalar(ST_ptr(cvar_st_idx),
            WN_Add(Pointer_type, cmem_wn,
                WN_Intconst(Integer_type, gvi->get_offset())));
}

WN* HC_lower_const_copyin(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "Start lowering a CONST COPYIN directive (%p) ...\n", gdata);
    }

    Is_True(gdata != NULL, (""));
    Is_True(parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK,
            ("HC_lower_const_copyin: NULL parent block\n"));

    // Validate the pragma.
    Is_True(pragma_wn != NULL && WN_operator(pragma_wn) == OPR_XPRAGMA, (""));

    if (gen_code)
    {
        // Generate the code block and insert it before the pragma.
        WN_INSERT_BlockBefore(parent_wn, pragma_wn,
                HC_gen_code_for_const_copyin(pragma_wn, gdata, lvar_store));
    }

    // Save the next node and remove the pragma.
    WN *next_wn = WN_next(pragma_wn);
    WN_DELETE_FromBlock(parent_wn, pragma_wn);

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the CONST COPYIN directive%s.\n",
                gen_code ? " (omitted)" : "");
    }

    return next_wn;
}

WN* HC_lower_const_remove(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, BOOL gen_code)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "Start lowering a CONSTANT REMOVE directive (%p) ...\n",
                gdata);
    }

    Is_True(gdata != NULL, (""));
    Is_True(parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK,
            ("HC_lower_const_remove: NULL parent block\n"));

    // Save the insertion point (next node).
    WN *insert_pt = WN_next(pragma_wn);

    // Validate and remove the pragma.
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_PRAGMA, (""));
    ST_IDX var_st_idx = WN_st_idx(pragma_wn);
    Is_True(var_st_idx == gdata->get_symbol(), (""));
    WN_DELETE_FromBlock(parent_wn, pragma_wn);

    // No code generated.

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the CONSTANT REMOVE directive.\n");
    }

    return insert_pt;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void HC_declare_svar(HC_GPU_DATA *sdata)
{
    HC_GPU_VAR_INFO *svi = sdata->get_gvar_info();
    Is_True(svi != NULL, (""));

    // Get the variable symbol and type.
    ST_IDX var_st_idx = sdata->get_symbol();
    TY_IDX var_ty_idx = ST_type(var_st_idx);

    // The SHARED directive must have an array section.
    TY_KIND var_ty_kind = TY_kind(var_ty_idx);
    Is_True(var_ty_kind == KIND_ARRAY || var_ty_kind == KIND_POINTER, (""));

    // Declare the shared memory variable (local), a pointer to the element
    // type of the original array variable.
    ST_IDX svar_st_idx = new_local_var(gen_var_str("s_", var_st_idx),
            sdata->create_gvar_type());
    // It is just a regular variable (no need for a special flag).
    // set_st_attr_is_shared_var(svar_st_idx);
    // Save it in the GPU variable record.
    svi->set_symbol(svar_st_idx);
}

typedef enum
{
    HC_KEEP_LOOP,
    HC_REMOVE_LOOP,
    HC_GEN_BRANCH       // generate a branch instead of a full loop
} HC_GEN_LOOP_STATE;

static WN* HC_make_svar_transfer_code(HC_GPU_DATA *sdata,
        BOOL copyin, HC_LOCAL_VAR_STORE *lvar_store)
{
    // TODO: use the one defined in HC_KERNEL_INFO.
    const UINT warp_sz = 16;

    ST_IDX var_st_idx = sdata->get_symbol();
    HC_GPU_VAR_INFO *svi = sdata->get_gvar_info();
    Is_True(svi != NULL, (""));
    ST_IDX svar_st_idx = svi->get_symbol();
    TY_IDX svar_ty_idx = ST_type(svar_st_idx);

    // Get the corresponding GLOBAL directive (fully mapped to local proc).
    HC_GPU_DATA *gdata = sdata->get_partner_gdata();
    Is_True(gdata != NULL, (""));
    // THIS IS NOT TRUE!!
    // Is_True(var_st_idx == gdata->get_symbol(), (""));
    ST_IDX gvar_st_idx = gdata->get_gvar_info()->get_symbol();

    HC_ARRSECTION_INFO *gdata_section = gdata->get_alloc_section();
    HC_ARRSECTION_INFO *sdata_section = sdata->get_alloc_section();
    HC_ARRSECTION_INFO *copy_section = copyin ?
        sdata->get_copyin_section() : sdata->get_copyout_section();
    if (copy_section == NULL) copy_section = sdata_section;

    UINT ndims = copy_section->get_num_dim();

    // Determine the biggest sub-section to be transferred that is contiguous
    // in both shared and global memory.
    UINT sdata_p_idx = copy_section->compute_pivot_dim_idx(sdata_section);
    UINT gdata_p_idx = copy_section->compute_pivot_dim_idx(gdata_section);
    UINT p_idx = (sdata_p_idx > gdata_p_idx) ? sdata_p_idx : gdata_p_idx;

    // Compute the total number of elements in all "full" dimensions to the
    // right of the pivot dimension.
    WN *full_dim_prod_wn = WN_Intconst(Integer_type, 1);
    for (UINT i = p_idx + 1; i < ndims; ++i)
    {
        full_dim_prod_wn = WN_Mpy(Integer_type, full_dim_prod_wn,
                copy_section->get_dim_sz(i));
    }

    // Compute <gcs_size>.
    WN *gcs_size_wn = WN_Mpy(Integer_type, gdata_section->get_dim_sz(p_idx),
            WN_COPY_Tree(full_dim_prod_wn));
    // Compute <gcs_offset>.
    WN *gcs_offset_wn = WN_Sub(Integer_type,
            WN_COPY_Tree(copy_section->get_dim_lbnd(0)),
            WN_COPY_Tree(gdata_section->get_dim_lbnd(0)));
    for (UINT i = 1; i <= p_idx; ++i)
    {
        gcs_offset_wn = WN_Add(Integer_type,
                WN_Mpy(Integer_type, gcs_offset_wn,
                    WN_COPY_Tree(gdata_section->get_dim_sz(i))),
                WN_Sub(Integer_type,
                    WN_COPY_Tree(copy_section->get_dim_lbnd(i)),
                    WN_COPY_Tree(gdata_section->get_dim_lbnd(i))));
    }
    HCWN_simplify_expr(&gcs_offset_wn);
    gcs_offset_wn = WN_Mpy(Integer_type, gcs_offset_wn,
            WN_COPY_Tree(full_dim_prod_wn));
    // Compute <scs_size>.
    WN *scs_size_wn = WN_Mpy(Integer_type, sdata_section->get_dim_sz(p_idx),
            WN_COPY_Tree(full_dim_prod_wn));
    // Compute <scs_offset>.
    WN *scs_offset_wn = WN_Sub(Integer_type,
            WN_COPY_Tree(copy_section->get_dim_lbnd(0)),
            WN_COPY_Tree(sdata_section->get_dim_lbnd(0)));
    for (UINT i = 1; i <= p_idx; ++i)
    {
        scs_offset_wn = WN_Add(Integer_type,
                WN_Mpy(Integer_type, scs_offset_wn,
                    WN_COPY_Tree(sdata_section->get_dim_sz(i))),
                WN_Sub(Integer_type,
                    WN_COPY_Tree(copy_section->get_dim_lbnd(i)),
                    WN_COPY_Tree(sdata_section->get_dim_lbnd(i))));
    }
    HCWN_simplify_expr(&scs_offset_wn);
    scs_offset_wn = WN_Mpy(Integer_type, scs_offset_wn,
            WN_COPY_Tree(full_dim_prod_wn));
    // Compute <cs_size>. Last one, so use <full_dim_prod_wn> directly.
    WN *cs_size_wn = WN_Mpy(Integer_type,
            copy_section->get_dim_sz(p_idx), full_dim_prod_wn);

    // Compute the number of contiguous segments (CSs): <n_cs>.
    WN *n_cs_wn = WN_Intconst(Integer_type, 1);
    for (UINT i = 0; i < p_idx; ++i)
    {
        n_cs_wn = WN_Mpy(Integer_type, n_cs_wn, copy_section->get_dim_sz(i));
    }

    // Knowing this can eliminate the generation of <cs_id> later on.
    BOOL single_cs = (WN_operator(n_cs_wn) == OPR_INTCONST
            && WN_const_val(n_cs_wn) == 1);

    // Check if all CS's start with an aligned offset, i.e. if both
    // <gcs_offset> and <gcs_size> are multiples of the warp size.
    // This is used in optimizing <n_segments_per_cs> and <thr_ofst>.
    BOOL perfect_align = 
        (HCWN_is_a_multiple_of(gcs_size_wn, NULL) % warp_sz == 0)
        && (HCWN_is_a_multiple_of(gcs_offset_wn,
                    sdata->get_lp_idxv_prop_list()) % warp_sz == 0);

    // Whether or not we need to generate the bound check.
    BOOL add_bndcheck = (copyin ?
            sdata->do_copyin_bndcheck() : sdata->do_copyout_bndcheck());

    // Compute <n_segments_per_cs>.
    WN *n_segs_per_cs_wn = NULL;
    if (WN_operator(cs_size_wn) == OPR_INTCONST)
    {
        // If <cs_size> is a constant, divide it by the warp size.
        UINT cs_size = WN_const_val(cs_size_wn);
        UINT n_segs_per_cs = cs_size / warp_sz;

        if (cs_size % warp_sz == 0)
        {
            if (perfect_align)
            {
                add_bndcheck = FALSE;
            }
            else
            {
                ++n_segs_per_cs;
            }
        }
        else if (cs_size % warp_sz == 1)
        {
            ++n_segs_per_cs;
        }
        else
        {
            n_segs_per_cs += 2;
        }
        n_segs_per_cs_wn = WN_Intconst(Integer_type, n_segs_per_cs);
    }
    else
    {
        n_segs_per_cs_wn = WN_Add(Integer_type,
                WN_Div(Integer_type, WN_COPY_Tree(n_cs_wn),
                    WN_Intconst(Integer_type, warp_sz)),
                WN_Intconst(Integer_type, 2));
    }

    HC_KERNEL_INFO *kinfo = sdata->get_kernel_info();
    Is_True(kinfo != NULL, (""));
    UINT n_warps = kinfo->get_num_warps();

    // Compare <n_warps> and <n_segments> if possible.
    HC_GEN_LOOP_STATE gen_loop = HC_KEEP_LOOP;   
    if (WN_operator(n_segs_per_cs_wn) == OPR_INTCONST
            && WN_operator(n_cs_wn) == OPR_INTCONST)
    {
        INT n_segs = WN_const_val(n_segs_per_cs_wn) * WN_const_val(n_cs_wn);
        if (n_segs == n_warps)
            gen_loop = HC_REMOVE_LOOP;
        else if (n_segs < n_warps)
            gen_loop = HC_GEN_BRANCH;
    }

    // Create an empty block.
    WN *blk_wn = WN_CreateBlock();

    // Initialize variable cs_sz (only used in bound check).
    ST_IDX cs_sz_st_idx = ST_IDX_ZERO;
    if (add_bndcheck)
    {
        cs_sz_st_idx = lvar_store->get_cs_sz_sym();
        WN_INSERT_BlockLast(blk_wn,
                WN_StidScalar(ST_ptr(cs_sz_st_idx), cs_size_wn));
    }
    // Initialize variable gcs_sz.
    ST_IDX gcs_sz_st_idx = ST_IDX_ZERO;
    if (!single_cs)
    {
        gcs_sz_st_idx = lvar_store->get_gcs_sz_sym();
        WN_INSERT_BlockLast(blk_wn,
                WN_StidScalar(ST_ptr(gcs_sz_st_idx), gcs_size_wn));
    }
    // Initialize variable gcs_ofst.
    ST_IDX gcs_ofst_st_idx = lvar_store->get_gcs_ofst_sym();
    WN_INSERT_BlockLast(blk_wn,
            WN_StidScalar(ST_ptr(gcs_ofst_st_idx), gcs_offset_wn));
    // Initialize variable scs_sz.
    ST_IDX scs_sz_st_idx = ST_IDX_ZERO;
    if (!single_cs)
    {
        scs_sz_st_idx = lvar_store->get_scs_sz_sym();
        WN_INSERT_BlockLast(blk_wn,
                WN_StidScalar(ST_ptr(scs_sz_st_idx), scs_size_wn));
    }
    // Initialize variable scs_ofst.
    ST_IDX scs_ofst_st_idx = lvar_store->get_scs_ofst_sym();
    WN_INSERT_BlockLast(blk_wn,
            WN_StidScalar(ST_ptr(scs_ofst_st_idx), scs_offset_wn));

    // Initialize variable n_segs_per_cs.
    BOOL single_seg_per_cs = (WN_operator(n_segs_per_cs_wn) == OPR_INTCONST
            && WN_const_val(n_segs_per_cs_wn) == 1);
    ST_IDX n_segs_per_cs_st_idx = ST_IDX_ZERO;
    if (!single_seg_per_cs)
    {
        n_segs_per_cs_st_idx = lvar_store->get_n_segs_per_cs_sym();
        WN_INSERT_BlockLast(blk_wn,
                WN_StidScalar(ST_ptr(n_segs_per_cs_st_idx),
                    n_segs_per_cs_wn));
    }
    // Initialize variable n_segs.
    ST_IDX n_segs_st_idx = lvar_store->get_n_segs_sym();
    WN_INSERT_BlockLast(blk_wn,
            WN_StidScalar(ST_ptr(n_segs_st_idx),
                single_seg_per_cs ? n_cs_wn :   // OPTIMIZATION
                WN_Mpy(Integer_type, n_cs_wn,
                    WN_LdidScalar(n_segs_per_cs_st_idx))));

    // Initialize variable warp_id.
    ST_IDX warp_id_st_idx = lvar_store->get_warp_id_sym();
    WN_INSERT_BlockLast(blk_wn,
            WN_StidScalar(ST_ptr(warp_id_st_idx),
                WN_COPY_Tree(kinfo->get_warp_id())));
    // Initialize variable id_within_warp.
    ST_IDX id_within_warp_st_idx = lvar_store->get_id_within_warp_sym();
    WN_INSERT_BlockLast(blk_wn,
            WN_StidScalar(ST_ptr(id_within_warp_st_idx),
                WN_COPY_Tree(kinfo->get_id_within_warp())));

    // Declare a loop index variable.
    ST_IDX idxv_st_idx = lvar_store->get_loop_idx_var(0);

    // Generate the loop body.
    WN *loop_body_wn = WN_CreateBlock();
    // cs_id = i / n_segs_per_cs
    ST_IDX cs_id_st_idx = ST_IDX_ZERO;
    if (!single_cs)
    {
        cs_id_st_idx = lvar_store->get_cs_id_sym();
        WN_INSERT_BlockLast(loop_body_wn,
                WN_StidScalar(ST_ptr(cs_id_st_idx),
                    // OPTIMIZATION
                    single_seg_per_cs ? WN_LdidScalar(idxv_st_idx) :
                    WN_Div(Integer_type, WN_LdidScalar(idxv_st_idx),
                        WN_LdidScalar(n_segs_per_cs_st_idx))));
    }
    // seg_id_in_cs = i % n_segs_per_cs
    // If single segment per CS, this value is 0.
    ST_IDX seg_id_in_cs_st_idx = ST_IDX_ZERO;
    if (!single_seg_per_cs)
    {
        seg_id_in_cs_st_idx = lvar_store->get_seg_id_in_cs_sym();
        WN_INSERT_BlockLast(loop_body_wn,
                WN_StidScalar(ST_ptr(seg_id_in_cs_st_idx),
                    WN_Binary(OPR_REM, Integer_type,
                        WN_LdidScalar(idxv_st_idx),
                        WN_LdidScalar(n_segs_per_cs_st_idx))));
    }
    // g_cs_ofst = gcs_sz * cs_id + gcs_ofst
    ST_IDX g_cs_ofst_st_idx = lvar_store->get_g_cs_ofst_sym();
    WN *g_cs_ofst_wn = WN_LdidScalar(gcs_ofst_st_idx);
    if (!single_cs)
    {
        g_cs_ofst_wn = WN_Add(Integer_type, g_cs_ofst_wn,
                WN_Mpy(Integer_type, WN_LdidScalar(gcs_sz_st_idx),
                    WN_LdidScalar(cs_id_st_idx)));
    }
    WN_INSERT_BlockLast(loop_body_wn,
            WN_StidScalar(ST_ptr(g_cs_ofst_st_idx), g_cs_ofst_wn));
    // s_cs_ofst = scs_sz * cs_id + scs_ofst
    ST_IDX s_cs_ofst_st_idx = lvar_store->get_s_cs_ofst_sym();
    WN *s_cs_ofst_wn = WN_LdidScalar(scs_ofst_st_idx);
    if (!single_cs)
    {
        s_cs_ofst_wn = WN_Add(Integer_type, s_cs_ofst_wn,
                WN_Mpy(Integer_type, WN_LdidScalar(scs_sz_st_idx),
                    WN_LdidScalar(cs_id_st_idx)));
    }
    WN_INSERT_BlockLast(loop_body_wn,
            WN_StidScalar(ST_ptr(s_cs_ofst_st_idx), s_cs_ofst_wn));
    // thr_ofst = <warp_sz> * seg_id_in_cs + id_within_warp
    //            - g_cs_ofst % <warp_sz>
    ST_IDX thr_ofst_st_idx = lvar_store->get_thr_ofst_sym();
    WN *thr_ofst_wn = WN_LdidScalar(id_within_warp_st_idx);
    if (!single_seg_per_cs)
    {
        Is_True(seg_id_in_cs_st_idx != ST_IDX_ZERO, (""));
        thr_ofst_wn = WN_Add(Integer_type, thr_ofst_wn,
                WN_Mpy(Integer_type, WN_Intconst(Integer_type, warp_sz),
                    WN_LdidScalar(seg_id_in_cs_st_idx)));
    }
    if (!perfect_align)
    {
        thr_ofst_wn = WN_Sub(Integer_type, thr_ofst_wn,
            WN_Binary(OPR_REM, Integer_type,
                        WN_LdidScalar(g_cs_ofst_st_idx),
                        WN_Intconst(Integer_type, warp_sz)));
    }
    WN_INSERT_BlockLast(loop_body_wn, 
            WN_StidScalar(ST_ptr(thr_ofst_st_idx), thr_ofst_wn));
    // s_A[s_cs_ofst + thr_ofst] = g_A[g_cs_ofst + thr_ofst]
    // or the other way, depending on the copy direction
    TY_IDX elem_ty_idx = sdata->get_elem_type();
    UINT elem_sz = TY_size(elem_ty_idx);
    WN *svar_access_wn = HCWN_CreateArray(WN_LdidScalar(svar_st_idx), 1);
    WN_element_size(svar_access_wn) = elem_sz;
    WN_kid1(svar_access_wn) = WN_Intconst(Integer_type, 0);
    WN_kid2(svar_access_wn) = WN_Add(Integer_type,
            WN_LdidScalar(s_cs_ofst_st_idx), WN_LdidScalar(thr_ofst_st_idx));
    WN *gvar_access_wn = HCWN_CreateArray(WN_LdidScalar(gvar_st_idx), 1);
    WN_element_size(gvar_access_wn) = elem_sz;
    WN_kid1(gvar_access_wn) = WN_Intconst(Integer_type, 0);
    WN_kid2(gvar_access_wn) = WN_Add(Integer_type,
            WN_LdidScalar(g_cs_ofst_st_idx), WN_LdidScalar(thr_ofst_st_idx));
    WN *src_access_wn, *dest_access_wn;
    if (copyin)
    {
        src_access_wn = gvar_access_wn;
        dest_access_wn = svar_access_wn;
    }
    else
    {
        src_access_wn = svar_access_wn;
        dest_access_wn = gvar_access_wn;
    }
    WN *transfer_wn = WN_Istore(TY_mtype(elem_ty_idx), 0,
            svar_ty_idx, dest_access_wn,
            WN_Iload(TY_mtype(elem_ty_idx), 0, elem_ty_idx, src_access_wn, 0),
            0);
    if (add_bndcheck)
    {
        // if (thr_ofst >= 0 && thr_ofst < cs_sz)
        WN *cond_wn = WN_LAND(
                WN_GE(Integer_type, WN_LdidScalar(thr_ofst_st_idx),
                    WN_Intconst(Integer_type, 0)),
                WN_LT(Integer_type, WN_LdidScalar(thr_ofst_st_idx),
                    WN_LdidScalar(cs_sz_st_idx)));
        WN *then_blk_wn = WN_CreateBlock();
        WN_INSERT_BlockFirst(then_blk_wn, transfer_wn);
        transfer_wn = WN_CreateIf(cond_wn, then_blk_wn, WN_CreateBlock());
    }
    WN_INSERT_BlockLast(loop_body_wn, transfer_wn);

    // Generate the loop.
    if (gen_loop == HC_KEEP_LOOP)
    {
        WN_INSERT_BlockLast(blk_wn,
                WN_CreateDO(WN_CreateIdname(0, idxv_st_idx),
                    WN_StidScalar(ST_ptr(idxv_st_idx),
                        WN_LdidScalar(warp_id_st_idx)),
                    WN_LT(Integer_type, WN_LdidScalar(idxv_st_idx),
                        WN_LdidScalar(n_segs_st_idx)),
                    WN_StidScalar(ST_ptr(idxv_st_idx),
                        WN_Add(Integer_type, WN_LdidScalar(idxv_st_idx),
                            WN_Intconst(Integer_type, n_warps))),
                    loop_body_wn, NULL));
    }
    else if (gen_loop == HC_GEN_BRANCH)
    {
        // i = warp_id
        WN_INSERT_BlockLast(blk_wn,
                WN_StidScalar(ST_ptr(idxv_st_idx),
                    WN_LdidScalar(warp_id_st_idx)));
        // if (i < n_segments)
        WN *cond_wn = WN_LT(Integer_type, WN_LdidScalar(idxv_st_idx),
                WN_LdidScalar(n_segs_st_idx));
        WN *then_blk_wn = WN_CreateBlock();
        WN_INSERT_BlockFirst(then_blk_wn, loop_body_wn);
        WN_INSERT_BlockLast(blk_wn,
                WN_CreateIf(cond_wn, then_blk_wn, WN_CreateBlock()));
    }
    else
    {
        // i = warp_id
        WN_INSERT_BlockLast(blk_wn,
                WN_StidScalar(ST_ptr(idxv_st_idx),
                    WN_LdidScalar(warp_id_st_idx)));
        WN_INSERT_BlockLast(blk_wn, loop_body_wn);
    }

    return blk_wn;
}

/*****************************************************************************
 *
 * At this stage, HC_GPU_VAR_INFO must have been created. The size of the
 * shared memory variable and its offset at the global <smem> have been
 * determined.
 *
 * We create a local shared memory pointer variable here, and initialize it.
 *
 * Return a BLOCK node that contains the code to be inserted before the
 * pragma.
 *
 ****************************************************************************/

static WN* HC_gen_code_for_shared_copyin(WN *pragma_wn, HC_GPU_DATA *sdata,
        HC_LOCAL_VAR_STORE *lvar_store)
{
    HC_GPU_VAR_INFO *svi = sdata->get_gvar_info();
    Is_True(svi != NULL, (""));

    ST_IDX svar_st_idx = svi->get_symbol();
    Is_True(svar_st_idx != ST_IDX_ZERO, (""));
    TY_IDX svar_ty_idx = ST_type(svar_st_idx);

    // Create a BLOCK node to hold the generated code.
    WN *blk_wn = WN_CreateBlock();

    if (flag_opencl){ 
      // X is a shared variable 
      // 1) Declare a shared array cl_X
      // 2) Make assignment X = cl_X
      
      TY_IDX base_ty_idx = TY_pointed(Ty_Table[svar_ty_idx]);
      TY_IDX tya_idx = Make_Array_Type(TY_mtype(base_ty_idx), 
				       1, 
				       WN_const_val(svi->get_size()) / TY_size(base_ty_idx));

      ST_IDX cl_svar_st_idx = new_local_var(gen_var_str("cl_", svar_st_idx),
					    tya_idx);
      
      set_st_attr_is_shared_var(cl_svar_st_idx);
      set_st_attr_is_shared_var(svar_st_idx);
      
      WN *cl_svar_wn = WN_CreateLda(OPR_LDA, Pointer_type, MTYPE_V, 0,
				    svar_ty_idx, cl_svar_st_idx, 0);
      
      WN_INSERT_BlockLast(blk_wn,
			  WN_StidScalar(ST_ptr(svar_st_idx),
					WN_Add(Pointer_type, cl_svar_wn,
					       WN_Intconst(Integer_type, 0))));

    } else {
      // Generate an assignment that initializes the shared memory variable
      // with <smem> + <offset>.
      ST_IDX smem_st_idx = hc_glob_var_store.get_smem_sym();
      WN *smem_wn = WN_CreateLda(OPR_LDA, Pointer_type, MTYPE_V, 0,
				 svar_ty_idx, smem_st_idx, 0);
      WN_INSERT_BlockLast(blk_wn,
			  WN_StidScalar(ST_ptr(svar_st_idx),
					WN_Add(Pointer_type, smem_wn,
					       WN_Intconst(Integer_type, svi->get_offset()))));
    }

    // Generate the code for copying the data from the global memory to the
    // shared memory.
    if (sdata->do_copyin())
    {
        WN_INSERT_BlockLast(blk_wn,
                HC_make_svar_transfer_code(sdata, TRUE, lvar_store));
    }

    return blk_wn;
}


WN* HC_lower_shared_copyin(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *sdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "Start lowering a SHARED COPYIN directive (%p) ...\n", sdata);
    }

    Is_True(sdata != NULL, (""));
    Is_True(parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK,
            ("HC_lower_shared_copyin: NULL parent block\n"));

    // Save the insertion point.
    WN *insert_pt = pragma_wn;

    /* Advance the pragmas and remove them. */

    // base pragma
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_PRAGMA,
            ("HC_lower_shared_copyin: invalid base pragma\n"));
    // the ALLOC clause
    pragma_wn = WN_next(pragma_wn);
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_XPRAGMA
            && WN_kid0(pragma_wn) != NULL,
            ("HC_lower_shared_copyin: invalid ALLOC pragma\n"));
    // the COPYIN clause
    pragma_wn = WN_next(pragma_wn);
    if (pragma_wn != NULL && WN_pragma(pragma_wn) == WN_PRAGMA_HC_COPY)
    {
        Is_True(sdata->do_copyin(), (""));
        pragma_wn = WN_next(pragma_wn);
    }
    // Now, <pragma_wn> points to the WN node after the last pragma.

    // Remove all pragmas from the insertion point (the 1st pragma)
    // to the current one (exclusive).
    WN *next_wn = NULL;
    while (insert_pt != pragma_wn)
    {
        next_wn = WN_next(insert_pt);
        WN_DELETE_FromBlock(parent_wn, insert_pt);
        insert_pt = next_wn;
    }
    // <insert_pt> == <pragma_wn>. All new code will be inserted before it.

    if (gen_code)
    {
        // Generate the code block and insert it before the pragma.
        WN_INSERT_BlockBefore(parent_wn, pragma_wn,
                HC_gen_code_for_shared_copyin(pragma_wn, sdata, lvar_store));
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the SHARED COPYIN directive%s.\n",
                gen_code ? " (omitted)" : "");
    }

    return pragma_wn;
}

WN* HC_lower_shared_copyout(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *sdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "Start lowering a SHARED COPYOUT directive (%p) ...\n",
                sdata);
    }

    Is_True(sdata != NULL, (""));
    Is_True(sdata->do_copyout(), (""));
    Is_True(parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK, (""));

    // Validate the COPYOUT field.
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_XPRAGMA, (""));
    WN *copy_wn = WN_kid0(pragma_wn);
    Is_True(copy_wn != NULL, (""));

    OPERATOR opr = WN_operator(copy_wn);
    Is_True(opr == OPR_ARRSECTION && sdata->is_arr_section(), (""));

    // Remove this pragma.
    WN *next_wn = WN_next(pragma_wn);
    WN_DELETE_FromBlock(parent_wn, pragma_wn);

    if (gen_code)
    {
        WN_INSERT_BlockBefore(parent_wn, next_wn,
                HC_make_svar_transfer_code(sdata, FALSE, lvar_store));
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the SHARED COPYOUT directive%s.\n",
                gen_code ? " (omitted)" : "");
    }

    return next_wn;
}

WN* HC_lower_shared_remove(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *sdata, BOOL gen_code)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "Start lowering a SHARED REMOVE directive (%p) ...\n", sdata);
    }

    Is_True(sdata != NULL, (""));
    Is_True(parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK,
            ("HC_lower_shared_remove: NULL parent block\n"));

    // Save the insertion point (next node).
    WN *insert_pt = WN_next(pragma_wn);

    // Validate and remove the pragma.
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_PRAGMA, (""));
    ST_IDX var_st_idx = WN_st_idx(pragma_wn);
    Is_True(var_st_idx == sdata->get_symbol(), (""));
    WN_DELETE_FromBlock(parent_wn, pragma_wn);

    // No code generated.

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the SHARED REMOVE directive.\n");
    }

    return insert_pt;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Given an ARRAY node, construct a new ARRAY that represents the same
 * access to the corresponding global/constant memory variable.
 *
 ****************************************************************************/

WN* HC_create_gvar_access_for_array(WN *access_wn, HC_GPU_DATA *gdata)
{
    HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();

    // Get the GPU memory variable, i.e. target of access redirection.
    ST_IDX gvar_st_idx = gvi->get_symbol();
    Is_True(gvar_st_idx != ST_IDX_ZERO, (""));

    // Mark this GPU memory variable being referenced.
    gvi->set_local_ref();

    // Get array dimensionality.
    UINT16 ndims = WN_num_dim(access_wn);
    Is_True(ndims >= 1, (""));

    // The new access treats the global/constant variable as a 1-D array.
    WN *new_access_wn = HCWN_CreateArray(WN_LdidScalar(gvar_st_idx), 1);
    WN_element_size(new_access_wn) = WN_element_size(access_wn);

    HC_ARRSECTION_INFO *alloc_section = gdata->get_alloc_section();
    Is_True(alloc_section != NULL, (""));

    // Compute the offset for the global array access.
    WN *offset_wn = WN_Sub(Integer_type,
            WN_COPY_Tree(WN_kid(access_wn,ndims+1)),
            WN_COPY_Tree(alloc_section->get_dim_lbnd(0)));
    for (UINT16 i = 1; i < ndims; ++i)
    {
        // offset_wn = offset_wn * curr_dim_size + (curr_dim_idx - start_idx)
        offset_wn = WN_Add(Integer_type,
                WN_Mpy(Integer_type, offset_wn, alloc_section->get_dim_sz(i)),
                WN_Sub(Integer_type,
                    WN_COPY_Tree(WN_kid(access_wn,ndims+i+1)),
                    WN_COPY_Tree(alloc_section->get_dim_lbnd(i))));
    }

    // We provide a fake dimension size because it is not necessarily in 1-D.
    WN_kid1(new_access_wn) = WN_Intconst(Integer_type, 0);

    // We do not need ILOAD because an ARRAY node only returns the access
    // address, so there should always be a parent ILOAD.
    WN_kid2(new_access_wn) = offset_wn;

    return new_access_wn;
}

/*****************************************************************************
 *
 * Given a LDID/STID node,
 * - for a scalar/struct variable, construct a ILOAD/ISTORE node that
 *   represents the same access to the corresponding global/constant variable.
 * - for a pointer-to-ARRAY variable, just replace the symbol with the
 *   global/constant variable (FIXME)
 *
 * Given an LDA node,
 *
 * - it must be an ARRAY variable, replace with a LDID of the global/constant
 *   variable.
 *
 ****************************************************************************/

WN* HC_create_gvar_access_for_scalar(WN *access_wn, HC_GPU_DATA *gdata)
{
    HC_GPU_VAR_INFO *gvi = gdata->get_gvar_info();

    // Get the GPU memory variable, i.e. target of access redirection.
    ST_IDX gvar_st_idx = gvi->get_symbol();
    Is_True(gvar_st_idx != ST_IDX_ZERO, (""));

    // Mark this GPU memory variable being referenced.
    gvi->set_local_ref();

    OPERATOR opr = WN_operator(access_wn);

    BOOL is_global_var = (gdata->get_type() == HC_GLOBAL_DATA);

    if (is_global_var) {
        Is_True(opr == OPR_LDID || opr == OPR_STID || opr == OPR_LDA, (""));
    } else {
        Is_True(opr == OPR_LDID || opr == OPR_LDA, (""));
    }

    ST_IDX st_idx = WN_st_idx(access_wn);
    TY_IDX ty_idx = ST_type(st_idx);
    TY_KIND var_ty_kind = TY_kind(ty_idx);

    if (opr == OPR_LDID || opr == OPR_STID)
    {
        if (var_ty_kind == KIND_POINTER)
        {
            Is_True(gdata->is_arr_section(), (""));
            // Just replace the symbol and the type.
            WN *result = WN_COPY_Tree(access_wn);
            WN_st_idx(result) = gvar_st_idx;
            WN_set_ty(result, ST_type(gvar_st_idx));
            return result;
        }

        Is_True(var_ty_kind == KIND_SCALAR
                || var_ty_kind == KIND_STRUCT, (""));

        return (opr == OPR_LDID) ?
            // Convert it to ILOAD.
            WN_CreateIload(OPR_ILOAD,
                    WN_rtype(access_wn), WN_desc(access_wn),
                    WN_load_offset(access_wn),
                    WN_ty(access_wn),
                    ST_type(gvar_st_idx),
                    WN_LdidScalar(gvar_st_idx),
                    WN_field_id(access_wn))
            :
            // Convert it to ISTORE.
            WN_CreateIstore(OPR_ISTORE,
                    WN_rtype(access_wn), WN_desc(access_wn),
                    WN_load_offset(access_wn),
                    ST_type(gvar_st_idx),
                    WN_COPY_Tree(WN_kid0(access_wn)),
                    WN_LdidScalar(gvar_st_idx),
                    WN_field_id(access_wn));
    }

    // LDA of ARRAY variable.
    Is_True(var_ty_kind == KIND_ARRAY, (""));
    Is_True(gdata->is_arr_section(), (""));
    Is_True(TY_kind(ST_type(gvar_st_idx)) == KIND_POINTER, (""));

    // Convert it to LDID.
    WN *result = WN_LdidScalar(gvar_st_idx);
    if (WN_offset(access_wn) != 0)
    {
        result = WN_Add(Pointer_type,
                result, WN_Intconst(Integer_type, WN_offset(access_wn)));
    }

    return result;
}

/*** DAVID CODE END ***/
