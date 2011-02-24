/** DAVID CODE BEGIN **/

#include <assert.h>

#include "wn.h"
#include "wn_util.h"
#include "const.h"

#include "hc_utils.h"
#include "cuda_utils.h"


static const char* enum_cudaMemcpyKind_entries[] = {
    "cudaMemcpyHostToHost",
    "cudaMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice"
};

#if 0
static const char* enum_cudaError_entries[] = {
    "cudaSuccess",
    "cudaErrorMissingConfiguration",
    "cudaErrorMemoryAllocation",
    "cudaErrorInitializationError",
    "cudaErrorLaunchFailure",
    "cudaErrorPriorLaunchFailure",
    "cudaErrorLaunchTimeout",
    "cudaErrorLaunchOutOfResources",
    "cudaErrorInvalidDeviceFunction",
    "cudaErrorInvalidConfiguration",
    "cudaErrorInvalidDevice",
    "cudaErrorInvalidValue",
    "cudaErrorInvalidPitchValue",
    "cudaErrorInvalidSymbol",
    "cudaErrorMapBufferObjectFailed",
    "cudaErrorUnmapBufferObjectFailed",
    "cudaErrorInvalidHostPointer",
    "cudaErrorInvalidDevicePointer",
    "cudaErrorInvalidTexture",
    "cudaErrorInvalidTextureBinding",
    "cudaErrorInvalidChannelDescriptor",
    "cudaErrorInvalidMemcpyDirection",
    "cudaErrorAddressOfConstant",
    "cudaErrorTextureFetchFailed",
    "cudaErrorTextureNotBound",
    "cudaErrorSynchronizationError",
    "cudaErrorInvalidFilterSetting",
    "cudaErrorInvalidNormSetting",
    "cudaErrorMixedDeviceExecution",
    "cudaErrorCudartUnloading",
    "cudaErrorUnknown",
    "cudaErrorNotYetImplemented",
    "cudaErrorMemoryValueTooLarge",
    "cudaErrorInvalidResourceHandle",
    "cudaErrorNotReady",
    "cudaErrorStartupFailure",
    "cudaErrorApiFailureBase"
};
#endif

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/* Declare scalar types. */

static TY_IDX int_ty_idx = TY_IDX_ZERO;         // int
static TY_IDX uint_ty_idx = TY_IDX_ZERO;        // unsigned int
static TY_IDX size_t_ty_idx = TY_IDX_ZERO;      // size_t

// cudaError_t: enum cudaError
static TY_IDX cudaError_t_ty_idx = TY_IDX_ZERO;

// enum cudaMemcpyKind
static TY_IDX enum_cudaMemcpyKind_ty_idx = TY_IDX_ZERO;


/**
 * Declare an ENUM type of name 'ty_name', and with entries of names
 * 'entry_names'.
 */
static TY_IDX
declare_enum_ty(const char *ty_name,
        int num_entries, const char **entry_names) {
    TY_IDX ty_idx;

    TY &ty = New_TY(ty_idx);
    TY_Init(ty, 4, KIND_SCALAR, MTYPE_U4, Save_Str(ty_name));

    // Make sure that the flag is set.
    Set_TY_is_enum(ty);

    EELIST_IDX eelist_idx;
    for (int i = 0; i < num_entries; ++i) {
        New_EELIST(eelist_idx);
        if (i == 0) Set_TY_eelist(ty, eelist_idx);
        EElist_Table[eelist_idx] = Save_Str(entry_names[i]);
    }

    // Add a NULL terminator.
    New_EELIST(eelist_idx);
    EElist_Table[eelist_idx] = EELIST_IDX_ZERO;

    return ty_idx;
}

/* Called by declare_cuda_types */
static void
declare_cuda_scalar_types() {
    int_ty_idx = MTYPE_To_TY(Integer_type);
    uint_ty_idx = MTYPE_To_TY(MTYPE_complement(Integer_type));
    assert(uint_ty_idx != TY_IDX_ZERO);

    // For now, we assume that 'size_t' is 'unsigned int'.
    size_t_ty_idx = uint_ty_idx;

#if 0
    cudaError_t_ty_idx = declare_enum_ty(
        "cudaError",
        NUM_enum_cudaError,
        enum_cudaError_entries
    );
#endif

    // For now, cudaError_t is UINT.
    cudaError_t_ty_idx = uint_ty_idx;

    enum_cudaMemcpyKind_ty_idx = declare_enum_ty(
        "cudaMemcpyKind",
        NUM_enum_cudaMemcpyKind,
        enum_cudaMemcpyKind_entries
    );
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

TY_IDX dim3_ty_idx = TY_IDX_ZERO;
TY_IDX uint3_ty_idx = TY_IDX_ZERO;

// OpenCL types
TY_IDX cl_mem_ty_idx = TY_IDX_ZERO;
TY_IDX cl_context_ty_idx = TY_IDX_ZERO;
TY_IDX cl_command_queue_ty_idx = TY_IDX_ZERO;
TY_IDX cl_kernel_ty_idx = TY_IDX_ZERO;
TY_IDX cl_program_ty_idx = TY_IDX_ZERO;

#if 0
static TY_IDX
get_cuda_type(const char *ty_name, UINT32 align_bytes) {
    TY_IDX ty_idx = lookup_type(ty_name);

    assert(ty_idx != TY_IDX_ZERO);
    Set_TY_align(ty_idx, align_bytes);
    Set_TY_is_cuda_runtime(ty_idx);

    return ty_idx;
}
#endif

static TY_IDX
declare_cuda_dim3() {
    /* We would not care if the type has existed, because we have the
     * type cleanup routine that would remove identical types. If the
     * user declares a different type with the same nime, the cod
     * generated will have two different struct defs with the same name,
     * which will cause error in NVCC compilation. */
    
    /**
     * struct dim3 {
     *     unsigned int x, y, z;
     * }
     */

    TY_IDX ty_idx;
    TY &ty = New_TY(ty_idx);
    TY_Init(ty, 12, KIND_STRUCT, MTYPE_M, Save_Str("dim3"));

    // Alignment is important!
    Set_TY_align(ty_idx, 4);
    // Flag it as a CUDA type.
    Set_TY_is_cuda_runtime(ty_idx);

    /* Set the fields' types. */
    
    // Determine the MTYPE for 'unsigned int'.
    FLD_HANDLE fh = New_FLD();
    FLD_Init(fh, Save_Str("x"), uint_ty_idx, 0);
    Set_TY_fld(ty, fh);

    fh = New_FLD();
    FLD_Init(fh, Save_Str("y"), uint_ty_idx, 4);

    fh = New_FLD();
    FLD_Init(fh, Save_Str("z"), uint_ty_idx, 8);
    Set_FLD_last_field(fh);

    return ty_idx;
}

static TY_IDX
declare_cuda_uint3() {
    /**
     * struct uint3 {
     *     unsigned int x, y, z;
     * }
     */

    TY_IDX ty_idx;
    TY &ty = New_TY(ty_idx);
    TY_Init(ty, 12, KIND_STRUCT, MTYPE_M, Save_Str("uint3"));

    // Alignment is important!
    Set_TY_align(ty_idx, 4);
    // Flag it as a CUDA type.
    Set_TY_is_cuda_runtime(ty_idx);

    /* Set the fields' types. */
    
    // Determine the MTYPE for 'unsigned int'.
    FLD_HANDLE fh = New_FLD();
    FLD_Init(fh, Save_Str("x"), uint_ty_idx, 0);
    Set_TY_fld(ty, fh);

    fh = New_FLD();
    FLD_Init(fh, Save_Str("y"), uint_ty_idx, 4);

    fh = New_FLD();
    FLD_Init(fh, Save_Str("z"), uint_ty_idx, 8);
    Set_FLD_last_field(fh);

    return ty_idx;
}

/**
 * Declare CUDA types, dim3, uint3, etc.
 * Called by init_cuda_includes
 */
static void
declare_cuda_types() {
    declare_cuda_scalar_types();

    dim3_ty_idx = declare_cuda_dim3();
    uint3_ty_idx = declare_cuda_uint3();

    // Declare OpenCL types
    TY &ty0 = New_TY(cl_mem_ty_idx);
    TY_Init(ty0, 4, KIND_SCALAR, MTYPE_U4, Save_Str("cl_mem"));

    TY &ty1 = New_TY(cl_context_ty_idx);
    TY_Init(ty1, 4, KIND_SCALAR, MTYPE_U4, Save_Str("cl_context"));

    TY &ty2 = New_TY(cl_command_queue_ty_idx);
    TY_Init(ty2, 4, KIND_SCALAR, MTYPE_U4, Save_Str("cl_command_queue"));

    TY &ty3 = New_TY(cl_kernel_ty_idx);
    TY_Init(ty3, 4, KIND_SCALAR, MTYPE_U4, Save_Str("cl_kernel"));
       
    TY &ty4 = New_TY(cl_program_ty_idx);
    TY_Init(ty4, 4, KIND_SCALAR, MTYPE_U4, Save_Str("cl_program"));
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

// Indexing for CUDA
ST_IDX blockIdx_st_idx = ST_IDX_ZERO;
ST_IDX threadIdx_st_idx = ST_IDX_ZERO;

// Indexing for OpenCL
ST_IDX blockIdxX_st_idx = ST_IDX_ZERO;
ST_IDX blockIdxY_st_idx = ST_IDX_ZERO;
ST_IDX blockIdxZ_st_idx = ST_IDX_ZERO;
ST_IDX threadIdxX_st_idx = ST_IDX_ZERO;
ST_IDX threadIdxY_st_idx = ST_IDX_ZERO;
ST_IDX threadIdxZ_st_idx = ST_IDX_ZERO;

/* Declare blockIdx and threadIdx */
static void
declare_block_thread_idx() {
    blockIdx_st_idx = new_extern_var("blockIdx", uint3_ty_idx);
    threadIdx_st_idx = new_extern_var("threadIdx", uint3_ty_idx);
    // ST *blockIdx_st = ST_ptr(blockIdx_st_idx);
    // ST *threadIdx_st = ST_ptr(threadIdx_st_idx);

    // Set_ST_is_const_var(blockIdx_st);
    // Set_ST_is_const_var(threadIdx_st);

    // Both variables will have declarations in cuda_runtime.h, so flag
    // them as CUDA symbols. Since no extra flag is available, it is for
    // now the combination of SCLASS_EXTERN and ST_IS_TEMP_VAR, which is
    // invalid in the original WHIRL spec.
    // Set_ST_is_temp_var(blockIdx_st);
    // Set_ST_is_temp_var(threadIdx_st);

    blockIdxX_st_idx = new_extern_var("blockIdxX", MTYPE_To_TY(MTYPE_U4));
    blockIdxY_st_idx = new_extern_var("blockIdxY", MTYPE_To_TY(MTYPE_U4));
    blockIdxZ_st_idx = new_extern_var("blockIdxZ", MTYPE_To_TY(MTYPE_U4));
    threadIdxX_st_idx = new_extern_var("threadIdxX", MTYPE_To_TY(MTYPE_U4));
    threadIdxY_st_idx = new_extern_var("threadIdxY", MTYPE_To_TY(MTYPE_U4));
    threadIdxZ_st_idx = new_extern_var("threadIdxZ", MTYPE_To_TY(MTYPE_U4));
    
    // Now we have a dedicated attribute.
    set_st_attr_is_cuda_runtime(blockIdx_st_idx);
    set_st_attr_is_cuda_runtime(threadIdx_st_idx);

    set_st_attr_is_cuda_runtime(blockIdxX_st_idx);
    set_st_attr_is_cuda_runtime(blockIdxY_st_idx);
    set_st_attr_is_cuda_runtime(blockIdxZ_st_idx);
    set_st_attr_is_cuda_runtime(threadIdxX_st_idx);
    set_st_attr_is_cuda_runtime(threadIdxY_st_idx);
    set_st_attr_is_cuda_runtime(threadIdxZ_st_idx);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Configure the given function symbol to indicate it is a CUDA runtime
 * function.
 */
static void
setup_cuda_runtime_function(ST_IDX func_st_idx) {
    ST &func_st = St_Table[func_st_idx];

    // This function will not be defined, so change the storage class.
    Set_ST_storage_class(func_st, SCLASS_EXTERN);

    // Flag it as part of the CUDA runtime.
    set_st_attr_is_cuda_runtime(func_st_idx);
}

static ST_IDX cudaMemcpy_st_idx = ST_IDX_ZERO;

/**
 * cudaError_t cudaMemcpy(void *dst, void *src,
 *     size_t count, enum cudaMemcpyKind kind)
 */
static void declare_cudaMemcpy()
{
    TY_IDX void_ptr_ty_idx = Make_Pointer_Type(MTYPE_To_TY(MTYPE_V));

    cudaMemcpy_st_idx = declare_function_va("cudaMemcpy", "cudaMemcpy.ty",
        cudaError_t_ty_idx,
        4,
        void_ptr_ty_idx,
        void_ptr_ty_idx,
        size_t_ty_idx,
        enum_cudaMemcpyKind_ty_idx
    );

    setup_cuda_runtime_function(cudaMemcpy_st_idx);
}

static ST_IDX cudaMemcpyToSymbol_st_idx = ST_IDX_ZERO;

/**
 * cudaError_t cudaMemcpyToSymbol(const char *symbol, void *src,
 *     size_t count, size_t offset, enum cudaMemcpyKind kind)
 */
static void declare_cudaMemcpyToSymbol()
{
    TY_IDX void_ptr_ty_idx = Make_Pointer_Type(MTYPE_To_TY(MTYPE_V));
    TY_IDX const_char_ty_idx = MTYPE_To_TY(MTYPE_I1);
    Set_TY_is_const(const_char_ty_idx);

    cudaMemcpyToSymbol_st_idx = declare_function_va(
        "cudaMemcpyToSymbol", "cudaMemcpyToSymbol.ty",
        cudaError_t_ty_idx,
        5,
        Make_Pointer_Type(const_char_ty_idx),
        void_ptr_ty_idx,
        size_t_ty_idx,
        size_t_ty_idx,
        enum_cudaMemcpyKind_ty_idx
    );

    setup_cuda_runtime_function(cudaMemcpyToSymbol_st_idx);
}

static ST_IDX cudaMemset_st_idx = ST_IDX_ZERO;

// cudaError_t cudaMemset(void *devPtr, int value, size_t count)
//
static void declare_cudaMemset()
{
    TY_IDX void_ptr_ty_idx = Make_Pointer_Type(MTYPE_To_TY(MTYPE_V));

    cudaMemset_st_idx = declare_function_va("cudaMemset", "cudaMemset.ty",
            cudaError_t_ty_idx,
            3, void_ptr_ty_idx, int_ty_idx, size_t_ty_idx);

    setup_cuda_runtime_function(cudaMemset_st_idx);
}

static ST_IDX cudaMalloc_st_idx = ST_IDX_ZERO;

// cudaError_t cudaMalloc(void **devPtr, size_t count)
//
static void declare_cudaMalloc()
{
    TY_IDX void_pp_ty_idx = Make_Pointer_Type(Make_Pointer_Type(
        MTYPE_To_TY(MTYPE_V)));

    cudaMalloc_st_idx = declare_function_va("cudaMalloc", "cudaMalloc.ty",
            cudaError_t_ty_idx,
            2, void_pp_ty_idx, size_t_ty_idx);

    setup_cuda_runtime_function(cudaMalloc_st_idx);
}

static ST_IDX cudaFree_st_idx = ST_IDX_ZERO;

/**
 * cudaError_t cudaFree(void *devPtr)
 */
static void
declare_cudaFree() {
    TY_IDX void_p_ty_idx = Make_Pointer_Type(MTYPE_To_TY(MTYPE_V));

    cudaFree_st_idx = declare_function_va("cudaFree", "cudaFree.ty",
        cudaError_t_ty_idx,
        1,
        void_p_ty_idx
    );

    setup_cuda_runtime_function(cudaFree_st_idx);
}

static ST_IDX syncthreads_st_idx = ST_IDX_ZERO;

/**
 * void __syncthreads()
 */
static void
declare_syncthreads() {
    syncthreads_st_idx = declare_function("__syncthreads",
        MTYPE_To_TY(MTYPE_V), 0, NULL);

    setup_cuda_runtime_function(syncthreads_st_idx);
}

// OpenCL functions

static ST_IDX clCreateBufferRet_st_idx = ST_IDX_ZERO;

/**
 *cl_mem clCreateBufferRet (cl_mem* ret
 *                          cl_context context,
 *                          cl_mem_flags flags, ----> unsigned int
 *                          size_t size,        ----> unsigned int
 *                          void *host_ptr,
 *                          cl_int *errcode_ret) ----> int *
 **/
static void 
declare_clCreateBufferRet() {
  clCreateBufferRet_st_idx = declare_function_va("clCreateBufferRet", "clCreateBufferRet.ty",
						 MTYPE_To_TY(MTYPE_V),
						 6,
						 Make_Pointer_Type(cl_mem_ty_idx),	 
						 cl_context_ty_idx,
						 MTYPE_To_TY(MTYPE_U4),
						 MTYPE_To_TY(MTYPE_U4),
						 Make_Pointer_Type(MTYPE_To_TY(MTYPE_V)),
						 Make_Pointer_Type(MTYPE_To_TY(MTYPE_I4))
						 );
  
  setup_cuda_runtime_function(clCreateBufferRet_st_idx);
}

static ST_IDX clEnqueueReadBuffer_st_idx = ST_IDX_ZERO;

/**
 *cl_int clEnqueueReadBuffer (cl_command_queue command_queue,
 *                            cl_mem buffer,
 *                            cl_bool blocking_read,
 *                            size_t offset,
 *                            size_t cb,
 *                            void *ptr,
 *                            cl_uint num_events_in_wait_list,
 *                            const cl_event *event_wait_list, --> void *
 *                            cl_event *event) --> void *
 **/
static void 
declare_clEnqueueReadBuffer() {
  clEnqueueReadBuffer_st_idx = declare_function_va("clEnqueueReadBuffer", "clEnqueueReadBuffer.ty",
						   MTYPE_To_TY(MTYPE_I4),
						   9,
						   cl_command_queue_ty_idx,
						   cl_mem_ty_idx,
						   MTYPE_To_TY(MTYPE_U4),
						   MTYPE_To_TY(MTYPE_U4),
						   MTYPE_To_TY(MTYPE_U4),
						   Make_Pointer_Type(MTYPE_To_TY(MTYPE_V)),
						   MTYPE_To_TY(MTYPE_U4),
						   Make_Pointer_Type(MTYPE_To_TY(MTYPE_V)),
						   Make_Pointer_Type(MTYPE_To_TY(MTYPE_V))
						   );
  
  setup_cuda_runtime_function(clEnqueueReadBuffer_st_idx);
}

static ST_IDX clEnqueueWriteBuffer_st_idx = ST_IDX_ZERO;

/**
 *cl_int clEnqueueWriteBuffer (cl_command_queue command_queue,
 *                             cl_mem buffer,
 *                             cl_bool blocking_write,
 *                             size_t offset,
 *                             size_t cb,
 *                             const void *ptr,
 *                             cl_uint num_events_in_wait_list,
 *                             const cl_event *event_wait_list,
 *                             const cl_event *event)
 **/
static void 
declare_clEnqueueWriteBuffer() {
  clEnqueueWriteBuffer_st_idx = declare_function_va("clEnqueueWriteBuffer", "clEnqueueWriteBuffer.ty",
						    MTYPE_To_TY(MTYPE_I4),
						    9,
						    cl_command_queue_ty_idx,
						    cl_mem_ty_idx,
						    MTYPE_To_TY(MTYPE_U4),
						    MTYPE_To_TY(MTYPE_U4),
						    MTYPE_To_TY(MTYPE_U4),
						    Make_Pointer_Type(MTYPE_To_TY(MTYPE_V)),
						    MTYPE_To_TY(MTYPE_U4),
						    Make_Pointer_Type(MTYPE_To_TY(MTYPE_V)),
						    Make_Pointer_Type(MTYPE_To_TY(MTYPE_V))
						    );
  
  setup_cuda_runtime_function(clEnqueueWriteBuffer_st_idx);
}

static ST_IDX clEnqueueWriteCleanBuffer_st_idx = ST_IDX_ZERO;

/**
 *cl_int clEnqueueWriteCleanBuffer (cl_command_queue command_queue,
 *                                  cl_mem buffer,
 *                                  cl_bool blocking_write,
 *                                  size_t offset,
 *                                  size_t cb,
 *                                  const void *ptr,
 *                                  cl_uint num_events_in_wait_list,
 *                                  const cl_event *event_wait_list,
 *                                  const cl_event *event)
 **/
static void 
declare_clEnqueueWriteCleanBuffer() {
  clEnqueueWriteCleanBuffer_st_idx = declare_function_va("clEnqueueWriteCleanBuffer", "clEnqueueWriteCleanBuffer.ty",
						    MTYPE_To_TY(MTYPE_I4),
						    9,
						    cl_command_queue_ty_idx,
						    cl_mem_ty_idx,
						    MTYPE_To_TY(MTYPE_U4),
						    MTYPE_To_TY(MTYPE_U4),
						    MTYPE_To_TY(MTYPE_U4),
						    Make_Pointer_Type(MTYPE_To_TY(MTYPE_V)),
						    MTYPE_To_TY(MTYPE_U4),
						    Make_Pointer_Type(MTYPE_To_TY(MTYPE_V)),
						    Make_Pointer_Type(MTYPE_To_TY(MTYPE_V))
						    );
  
  setup_cuda_runtime_function(clEnqueueWriteCleanBuffer_st_idx);
}

static ST_IDX clReleaseMemObj_st_idx = ST_IDX_ZERO;

/**
 * cl_int clReleaseMemObject (cl_mem memobj)
 **/
static void 
declare_clReleaseMemObj() {
  clReleaseMemObj_st_idx = declare_function_va("clReleaseMemObject", "clReleaseMemObject.ty",
					       MTYPE_To_TY(MTYPE_I4),
					       1,
					       cl_mem_ty_idx
					       );
  
  setup_cuda_runtime_function(clReleaseMemObj_st_idx);
}

static ST_IDX clCreateKernelRet_st_idx = ST_IDX_ZERO;

/**
 * cl_kernel *clCreateKernel (cl_kernel *ret,
 *                            cl_program program,
 *                            const char *kernel_name,
 *                            cl_int *errcode_ret)
 **/
static void 
declare_clCreateKernelRet() {
  TY_IDX const_char_ty_idx = MTYPE_To_TY(MTYPE_I1);
  Set_TY_is_const(const_char_ty_idx);

  clCreateKernelRet_st_idx = declare_function_va("clCreateKernelRet", "clCreateKernelRet.ty",
					      MTYPE_To_TY(MTYPE_V),
					      4,
					      Make_Pointer_Type(cl_kernel_ty_idx),
					      cl_program_ty_idx,
					      Make_Pointer_Type(const_char_ty_idx),
					      Make_Pointer_Type(MTYPE_To_TY(MTYPE_I4))
					      );
  
  setup_cuda_runtime_function(clCreateKernelRet_st_idx);
}

static ST_IDX clSetKernelArg_st_idx = ST_IDX_ZERO;

/**
 * cl_int clSetKernelArg (cl_kernel kernel,
 *                        cl_uint arg_index,
 *                        size_t arg_size,
 *                        const void *arg_value)
 **/
static void declare_clSetKernelArg()
{
 TY_IDX const_void_ty_idx = MTYPE_To_TY(MTYPE_V);
 Set_TY_is_const(const_void_ty_idx);

  clSetKernelArg_st_idx = declare_function_va("clSetKernelArg", "clSetKernelArg.ty",
					      MTYPE_To_TY(MTYPE_I4),
					      4,
					      cl_kernel_ty_idx,
					      MTYPE_To_TY(MTYPE_U4),
					      MTYPE_To_TY(MTYPE_U4),
					      Make_Pointer_Type(const_void_ty_idx)
					      );
					  
  setup_cuda_runtime_function(clSetKernelArg_st_idx);
}

static ST_IDX clEnqueueNDRangeKernel_st_idx = ST_IDX_ZERO;

/**
 * cl_int *clEnqueueNDRangeKernel (cl_command_queue command_queue,
 *                                 cl_kernel kernel,
 *                                 cl_uint work_dim,
 *                                 const size_t *global_work_offset,
 *                                 const size_t *global_work_size,
 *                                 const size_t *local_work_size,
 *                                 cl_uint num_events_in_wait_list,
 *                                 const cl_event *event_wait_list,
 *                                 cl_event *event)
 **/
static void 
declare_clEnqueueNDRangeKernel() {
  TY_IDX const_uint_ty_idx = MTYPE_To_TY(MTYPE_U4);
  Set_TY_is_const(const_uint_ty_idx);

  clEnqueueNDRangeKernel_st_idx = declare_function_va("clEnqueueNDRangeKernel", "clEnqueueNDRangeKernel.ty",
						      MTYPE_To_TY(MTYPE_I4),
						      9,
						      cl_command_queue_ty_idx,
						      cl_kernel_ty_idx,
						      MTYPE_To_TY(MTYPE_U4),
						      Make_Pointer_Type(const_uint_ty_idx),
						      Make_Pointer_Type(const_uint_ty_idx),
						      Make_Pointer_Type(const_uint_ty_idx),
						      MTYPE_To_TY(MTYPE_U4),
						      Make_Pointer_Type(MTYPE_To_TY(MTYPE_V)),
						      Make_Pointer_Type(MTYPE_To_TY(MTYPE_V))
						    );
  
  setup_cuda_runtime_function(clEnqueueNDRangeKernel_st_idx);
}

static ST_IDX clBarrier_st_idx = ST_IDX_ZERO;

/**
 * void barrier (cl_mem_fence_flags flags)
 **/
static void declare_clBarrier()
{
  clBarrier_st_idx = declare_function_va("barrier", "barrier.ty",
					 MTYPE_To_TY(MTYPE_V),
					 1,
					 MTYPE_To_TY(MTYPE_U4)
					 );
  
  setup_cuda_runtime_function(clBarrier_st_idx);
}

/**
 * Declare CUDA function prototypes.
 * Called by init_cuda_includes
 */
static void declare_cuda_functions()
{
    // Declare CUDA functions
    declare_cudaMemcpy();
    declare_cudaMemcpyToSymbol();
    declare_cudaMemset();
    declare_cudaMalloc();
    declare_cudaFree();
    declare_syncthreads();

    // Declare OpenCL function
    declare_clCreateBufferRet();  
    declare_clEnqueueReadBuffer();
    declare_clEnqueueWriteBuffer();
    declare_clEnqueueWriteCleanBuffer();
    declare_clReleaseMemObj();
    declare_clCreateKernelRet();
    declare_clSetKernelArg();
    declare_clEnqueueNDRangeKernel();
    declare_clBarrier();
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

static BOOL cuda_includes_initialized = FALSE;

void init_cuda_includes()
{
    if (cuda_includes_initialized) return;

    // scalar and aggregate types
    declare_cuda_types();

    // must be called AFTER declaring all CUDA types.
    declare_block_thread_idx();

    declare_cuda_functions();

    cuda_includes_initialized = TRUE;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

WN* call_cudaMemcpy(WN *dst, WN *src, WN *count, enum cudaMemcpyKind kind)
{
    // Get the function symbol.
    assert(cudaMemcpy_st_idx != ST_IDX_ZERO);
    ST_IDX func_st_idx = cudaMemcpy_st_idx;

    // Get the parameter types.
    TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
    TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
    TY_IDX dst_ty_idx = Tylist_Table[param_tyl_idx];
    TY_IDX src_ty_idx = Tylist_Table[param_tyl_idx+1];
    TY_IDX count_ty_idx = Tylist_Table[param_tyl_idx+2];
    // TY_IDX kind_ty_idx = Tylist_Table[param_tyl_idx+3];

    // Generate a node for the copy kind.
    assert(enum_cudaMemcpyKind_ty_idx != TY_IDX_ZERO);
    WN *kind_wn = WN_Enum(kind, enum_cudaMemcpyKind_ty_idx);

    WN *wn = WN_Call(MTYPE_V, MTYPE_V, 4, func_st_idx);

    // Construct the arguments.
    WN_kid0(wn) = HCWN_Parm(TY_mtype(dst_ty_idx), dst, dst_ty_idx);
    WN_kid1(wn) = HCWN_Parm(TY_mtype(src_ty_idx), src, src_ty_idx);
    WN_kid2(wn) = HCWN_Parm(TY_mtype(count_ty_idx), count, count_ty_idx);
    // enum in C is mapped to UINT.
    assert(uint_ty_idx != TY_IDX_ZERO);
    WN_kid(wn,3) = HCWN_Parm(TY_mtype(uint_ty_idx), kind_wn, uint_ty_idx);

    return wn;
}

WN* call_cudaMemcpyToSymbol(ST_IDX st_idx, WN *src,
        WN *count, WN *offset, enum cudaMemcpyKind kind)
{
    // Get the function symbol.
    assert(cudaMemcpyToSymbol_st_idx != ST_IDX_ZERO);
    ST_IDX func_st_idx = cudaMemcpyToSymbol_st_idx;

    // Get the parameter types.
    TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
    TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
    TY_IDX sym_ty_idx = Tylist_Table[param_tyl_idx];
    TY_IDX src_ty_idx = Tylist_Table[param_tyl_idx+1];
    TY_IDX count_ty_idx = Tylist_Table[param_tyl_idx+2];
    TY_IDX offset_ty_idx = Tylist_Table[param_tyl_idx+3];

    // Create a node that holds the name of 'st_idx'.
    char *st_name = ST_name(st_idx);
    TCON tcon = Host_To_Targ_String(MTYPE_STRING, st_name, strlen(st_name));
    // const char
    TY_IDX cc_ty_idx = MTYPE_To_TY(MTYPE_I1);
    Set_TY_is_const(cc_ty_idx);
    // const char*
    TY_IDX ccs_ty_idx = Make_Pointer_Type(cc_ty_idx);
    ST *name_st = New_Const_Sym(Enter_tcon(tcon), ccs_ty_idx);
    WN *name_wn = WN_LdaZeroOffset(ST_st_idx(name_st), ccs_ty_idx);
/*
    WN *sym_wn = WN_CreateConst(OPR_CONST, Pointer_type, MTYPE_V,
        Gen_String_Sym(&tc, MTYPE_To_TY(MTYPE_STRING), FALSE));
*/

    // Generate a node for the copy kind.
    assert(enum_cudaMemcpyKind_ty_idx != TY_IDX_ZERO);
    WN *kind_wn = WN_Enum(kind, enum_cudaMemcpyKind_ty_idx);

    WN *wn = WN_Call(MTYPE_V, MTYPE_V, 5, func_st_idx);

    // Construct the arguments.
    WN_kid0(wn) = HCWN_Parm(Pointer_type, name_wn, sym_ty_idx);
    WN_kid1(wn) = HCWN_Parm(TY_mtype(src_ty_idx), src, src_ty_idx);
    WN_kid2(wn) = HCWN_Parm(TY_mtype(count_ty_idx), count, count_ty_idx);
    WN_kid(wn,3) = HCWN_Parm(TY_mtype(offset_ty_idx), offset, offset_ty_idx);
    // enum in C is mapped to UINT.
    assert(uint_ty_idx != TY_IDX_ZERO);
    WN_kid(wn,4) = HCWN_Parm(TY_mtype(uint_ty_idx), kind_wn, uint_ty_idx);

    return wn;
}

WN* call_cudaMemset(WN *dst_wn, WN *value_wn, WN *count_wn)
{
    // Get the function symbol.
    Is_True(cudaMemset_st_idx != ST_IDX_ZERO, (""));
    ST_IDX func_st_idx = cudaMemset_st_idx;

    // Get the parameter types.
    TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
    TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
    TY_IDX dst_ty_idx = Tylist_Table[param_tyl_idx];
    TY_IDX value_ty_idx = Tylist_Table[param_tyl_idx+1];
    TY_IDX count_ty_idx = Tylist_Table[param_tyl_idx+2];

    WN *wn = WN_Call(MTYPE_V, MTYPE_V, 3, func_st_idx);

    // Construct the arguments.
    WN_kid0(wn) = HCWN_Parm(TY_mtype(dst_ty_idx), dst_wn, dst_ty_idx);
    WN_kid1(wn) = HCWN_Parm(TY_mtype(value_ty_idx), value_wn, value_ty_idx);
    WN_kid2(wn) = HCWN_Parm(TY_mtype(count_ty_idx), count_wn, count_ty_idx);

    return wn;
}

WN*
call_cudaMalloc(WN *ptr_addr, WN *size) {
    // Get the function symbol.
    assert(cudaMalloc_st_idx != ST_IDX_ZERO);
    ST_IDX func_st_idx = cudaMalloc_st_idx;

    // Get the parameter types.
    TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
    TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
    TY_IDX ptr_addr_ty_idx = Tylist_Table[param_tyl_idx];
    TY_IDX size_ty_idx = Tylist_Table[param_tyl_idx+1];

    WN *wn = WN_Call(MTYPE_V, MTYPE_V, 2, func_st_idx);

    // Construct the arguments.
    WN_kid0(wn) = HCWN_Parm(Pointer_type, ptr_addr, ptr_addr_ty_idx);
    WN_kid1(wn) = HCWN_Parm(MTYPE_U4, size, size_ty_idx);

    return wn;
}

WN*
call_cudaFree(WN *ptr) {
    // Get the function symbol.
    assert(cudaFree_st_idx != ST_IDX_ZERO);
    ST_IDX func_st_idx = cudaFree_st_idx;

    // Get the parameter types.
    TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
    TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
    TY_IDX ptr_ty_idx = Tylist_Table[param_tyl_idx];

    WN *wn = WN_Call(MTYPE_V, MTYPE_V, 1, func_st_idx);

    // Construct the arguments.
    WN_kid0(wn) = HCWN_Parm(Pointer_type, ptr, ptr_ty_idx);

    return wn;
}

WN*
call_syncthreads() {
    return WN_Call(MTYPE_V, MTYPE_V, 0, syncthreads_st_idx);
}

// OpenCL function

WN* 
call_clCreateBufferRet(WN *p1, WN* p2, WN* p3, WN *p4, WN *p5, WN *p6) {
  // Get the function symbol.
  assert(clCreateBufferRet_st_idx != ST_IDX_ZERO);
  ST_IDX func_st_idx = clCreateBufferRet_st_idx;
  
  // Get the parameter types.
  TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
  TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
  TY_IDX p1_ty_idx = Tylist_Table[param_tyl_idx+0];
  TY_IDX p2_ty_idx = Tylist_Table[param_tyl_idx+1];
  TY_IDX p3_ty_idx = Tylist_Table[param_tyl_idx+2];
  TY_IDX p4_ty_idx = Tylist_Table[param_tyl_idx+3];
  TY_IDX p5_ty_idx = Tylist_Table[param_tyl_idx+4];  
  TY_IDX p6_ty_idx = Tylist_Table[param_tyl_idx+5];

  WN *wn = WN_Call(MTYPE_U4, MTYPE_V, 6, func_st_idx);
  
  // Construct the arguments.
  WN_kid0(wn) = HCWN_Parm(TY_mtype(p1_ty_idx), p1, p1_ty_idx);
  WN_kid1(wn) = HCWN_Parm(TY_mtype(p2_ty_idx), p2, p2_ty_idx);
  WN_kid2(wn) = HCWN_Parm(TY_mtype(p3_ty_idx), p3, p3_ty_idx);
  WN_kid(wn, 3) = HCWN_Parm(TY_mtype(p4_ty_idx), p4, p4_ty_idx);
  WN_kid(wn, 4) = HCWN_Parm(TY_mtype(p5_ty_idx), p5, p5_ty_idx);
  WN_kid(wn, 5) = HCWN_Parm(TY_mtype(p6_ty_idx), p6, p6_ty_idx);

  return wn;
}

WN* 
call_clEnqueueReadBuffer(WN *p1, WN* p2, WN* p3, WN *p4, WN *p5, WN *p6, WN *p7, WN *p8, WN *p9) {
  // Get the function symbol.
  assert(clEnqueueReadBuffer_st_idx != ST_IDX_ZERO);
  ST_IDX func_st_idx = clEnqueueReadBuffer_st_idx;
  
  // Get the parameter types.
  TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
  TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
  TY_IDX p1_ty_idx = Tylist_Table[param_tyl_idx+0];
  TY_IDX p2_ty_idx = Tylist_Table[param_tyl_idx+1];
  TY_IDX p3_ty_idx = Tylist_Table[param_tyl_idx+2];
  TY_IDX p4_ty_idx = Tylist_Table[param_tyl_idx+3];
  TY_IDX p5_ty_idx = Tylist_Table[param_tyl_idx+4];  
  TY_IDX p6_ty_idx = Tylist_Table[param_tyl_idx+5];
  TY_IDX p7_ty_idx = Tylist_Table[param_tyl_idx+6];
  TY_IDX p8_ty_idx = Tylist_Table[param_tyl_idx+7];  
  TY_IDX p9_ty_idx = Tylist_Table[param_tyl_idx+8];
  
  WN *wn = WN_Call(MTYPE_V, MTYPE_V, 9, func_st_idx);
  
  // Construct the arguments.
  WN_kid0(wn) = HCWN_Parm(TY_mtype(p1_ty_idx), p1, p1_ty_idx);
  WN_kid1(wn) = HCWN_Parm(TY_mtype(p2_ty_idx), p2, p2_ty_idx);
  WN_kid2(wn) = HCWN_Parm(TY_mtype(p3_ty_idx), p3, p3_ty_idx);
  WN_kid(wn, 3) = HCWN_Parm(TY_mtype(p4_ty_idx), p4, p4_ty_idx);
  WN_kid(wn, 4) = HCWN_Parm(TY_mtype(p5_ty_idx), p5, p5_ty_idx);
  WN_kid(wn, 5) = HCWN_Parm(TY_mtype(p6_ty_idx), p6, p6_ty_idx);
  WN_kid(wn, 6) = HCWN_Parm(TY_mtype(p7_ty_idx), p7, p7_ty_idx);
  WN_kid(wn, 7) = HCWN_Parm(TY_mtype(p8_ty_idx), p8, p8_ty_idx);
  WN_kid(wn, 8) = HCWN_Parm(TY_mtype(p9_ty_idx), p9, p9_ty_idx);

  return wn;
}

WN* 
call_clEnqueueWriteBuffer(WN *p1, WN* p2, WN* p3, WN *p4, WN *p5, WN *p6, WN *p7, WN *p8, WN *p9) {
  // Get the function symbol.
  assert(clEnqueueWriteBuffer_st_idx != ST_IDX_ZERO);
  ST_IDX func_st_idx = clEnqueueWriteBuffer_st_idx;
  
  // Get the parameter types.
  TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
  TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
  TY_IDX p1_ty_idx = Tylist_Table[param_tyl_idx+0];
  TY_IDX p2_ty_idx = Tylist_Table[param_tyl_idx+1];
  TY_IDX p3_ty_idx = Tylist_Table[param_tyl_idx+2];
  TY_IDX p4_ty_idx = Tylist_Table[param_tyl_idx+3];
  TY_IDX p5_ty_idx = Tylist_Table[param_tyl_idx+4];  
  TY_IDX p6_ty_idx = Tylist_Table[param_tyl_idx+5];
  TY_IDX p7_ty_idx = Tylist_Table[param_tyl_idx+6];
  TY_IDX p8_ty_idx = Tylist_Table[param_tyl_idx+7];  
  TY_IDX p9_ty_idx = Tylist_Table[param_tyl_idx+8];
  
  WN *wn = WN_Call(MTYPE_V, MTYPE_V, 9, func_st_idx);
  
  // Construct the arguments.
  WN_kid0(wn) = HCWN_Parm(TY_mtype(p1_ty_idx), p1, p1_ty_idx);
  WN_kid1(wn) = HCWN_Parm(TY_mtype(p2_ty_idx), p2, p2_ty_idx);
  WN_kid2(wn) = HCWN_Parm(TY_mtype(p3_ty_idx), p3, p3_ty_idx);
  WN_kid(wn, 3) = HCWN_Parm(TY_mtype(p4_ty_idx), p4, p4_ty_idx);
  WN_kid(wn, 4) = HCWN_Parm(TY_mtype(p5_ty_idx), p5, p5_ty_idx);
  WN_kid(wn, 5) = HCWN_Parm(TY_mtype(p6_ty_idx), p6, p6_ty_idx);
  WN_kid(wn, 6) = HCWN_Parm(TY_mtype(p7_ty_idx), p7, p7_ty_idx);
  WN_kid(wn, 7) = HCWN_Parm(TY_mtype(p8_ty_idx), p8, p8_ty_idx);
  WN_kid(wn, 8) = HCWN_Parm(TY_mtype(p9_ty_idx), p9, p9_ty_idx);

  return wn;
}

WN* 
call_clEnqueueWriteCleanBuffer(WN *p1, WN* p2, WN* p3, WN *p4, WN *p5, WN *p6, WN *p7, WN *p8, WN *p9) {
  // Get the function symbol.
  assert(clEnqueueWriteCleanBuffer_st_idx != ST_IDX_ZERO);
  ST_IDX func_st_idx = clEnqueueWriteCleanBuffer_st_idx;
  
  // Get the parameter types.
  TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
  TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
  TY_IDX p1_ty_idx = Tylist_Table[param_tyl_idx+0];
  TY_IDX p2_ty_idx = Tylist_Table[param_tyl_idx+1];
  TY_IDX p3_ty_idx = Tylist_Table[param_tyl_idx+2];
  TY_IDX p4_ty_idx = Tylist_Table[param_tyl_idx+3];
  TY_IDX p5_ty_idx = Tylist_Table[param_tyl_idx+4];  
  TY_IDX p6_ty_idx = Tylist_Table[param_tyl_idx+5];
  TY_IDX p7_ty_idx = Tylist_Table[param_tyl_idx+6];
  TY_IDX p8_ty_idx = Tylist_Table[param_tyl_idx+7];  
  TY_IDX p9_ty_idx = Tylist_Table[param_tyl_idx+8];
  
  WN *wn = WN_Call(MTYPE_V, MTYPE_V, 9, func_st_idx);
  
  // Construct the arguments.
  WN_kid0(wn) = HCWN_Parm(TY_mtype(p1_ty_idx), p1, p1_ty_idx);
  WN_kid1(wn) = HCWN_Parm(TY_mtype(p2_ty_idx), p2, p2_ty_idx);
  WN_kid2(wn) = HCWN_Parm(TY_mtype(p3_ty_idx), p3, p3_ty_idx);
  WN_kid(wn, 3) = HCWN_Parm(TY_mtype(p4_ty_idx), p4, p4_ty_idx);
  WN_kid(wn, 4) = HCWN_Parm(TY_mtype(p5_ty_idx), p5, p5_ty_idx);
  WN_kid(wn, 5) = HCWN_Parm(TY_mtype(p6_ty_idx), p6, p6_ty_idx);
  WN_kid(wn, 6) = HCWN_Parm(TY_mtype(p7_ty_idx), p7, p7_ty_idx);
  WN_kid(wn, 7) = HCWN_Parm(TY_mtype(p8_ty_idx), p8, p8_ty_idx);
  WN_kid(wn, 8) = HCWN_Parm(TY_mtype(p9_ty_idx), p9, p9_ty_idx);

  return wn;
}

WN* 
call_clReleaseMemObj(WN *p1) {
  // Get the function symbol.
  assert(clReleaseMemObj_st_idx != ST_IDX_ZERO);
  ST_IDX func_st_idx = clReleaseMemObj_st_idx;
  
  // Get the parameter types.
  TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
  TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
  TY_IDX p1_ty_idx = Tylist_Table[param_tyl_idx+0];  

  WN *wn = WN_Call(MTYPE_V, MTYPE_V, 1, func_st_idx);
  
  // Construct the arguments.
  WN_kid0(wn) = HCWN_Parm(TY_mtype(p1_ty_idx), p1, p1_ty_idx);

  return wn;
}


WN* 
call_clCreateKernelRet(WN *p1, WN* p2, WN* p3, WN *p4) {
  // Get the function symbol.
  assert(clCreateKernelRet_st_idx != ST_IDX_ZERO);
  ST_IDX func_st_idx = clCreateKernelRet_st_idx;
  
  // Get the parameter types.
  TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
  TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
  TY_IDX p1_ty_idx = Tylist_Table[param_tyl_idx+0];
  TY_IDX p2_ty_idx = Tylist_Table[param_tyl_idx+1];
  TY_IDX p3_ty_idx = Tylist_Table[param_tyl_idx+2]; 
  TY_IDX p4_ty_idx = Tylist_Table[param_tyl_idx+3];

  WN *wn = WN_Call(MTYPE_V, MTYPE_V, 4, func_st_idx);
  
  // Construct the arguments.
  WN_kid0(wn) = HCWN_Parm(TY_mtype(p1_ty_idx), p1, p1_ty_idx);
  WN_kid1(wn) = HCWN_Parm(TY_mtype(p2_ty_idx), p2, p2_ty_idx);
  WN_kid2(wn) = HCWN_Parm(TY_mtype(p3_ty_idx), p3, p3_ty_idx);
  WN_kid3(wn) = HCWN_Parm(TY_mtype(p4_ty_idx), p4, p4_ty_idx);

  return wn;
}

WN* 
call_clSetKernelArg(WN *p1, WN* p2, WN* p3, WN *p4) {
  // Get the function symbol.
  assert(clSetKernelArg_st_idx != ST_IDX_ZERO);
  ST_IDX func_st_idx = clSetKernelArg_st_idx;
  
  // Get the parameter types.
  TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
  TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
  TY_IDX p1_ty_idx = Tylist_Table[param_tyl_idx+0];
  TY_IDX p2_ty_idx = Tylist_Table[param_tyl_idx+1];
  TY_IDX p3_ty_idx = Tylist_Table[param_tyl_idx+2];
  TY_IDX p4_ty_idx = Tylist_Table[param_tyl_idx+3];;  

  WN *wn = WN_Call(MTYPE_V, MTYPE_V, 4, func_st_idx);
  
  // Construct the arguments.
  WN_kid0(wn) = HCWN_Parm(TY_mtype(p1_ty_idx), p1, p1_ty_idx);
  WN_kid1(wn) = HCWN_Parm(TY_mtype(p2_ty_idx), p2, p2_ty_idx);
  WN_kid2(wn) = HCWN_Parm(TY_mtype(p3_ty_idx), p3, p3_ty_idx);
  WN_kid(wn, 3) = HCWN_Parm(TY_mtype(p4_ty_idx), p4, p4_ty_idx);

  return wn;
}

WN* 
call_clEnqueueNDRangeKernel(WN *p1, WN* p2, WN* p3, WN *p4, WN *p5, WN *p6, WN *p7, WN *p8, WN *p9) {
  // Get the function symbol.
  assert(clEnqueueNDRangeKernel_st_idx != ST_IDX_ZERO);
  ST_IDX func_st_idx = clEnqueueNDRangeKernel_st_idx;
  
  // Get the parameter types.
  TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
  TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
  TY_IDX p1_ty_idx = Tylist_Table[param_tyl_idx+0];
  TY_IDX p2_ty_idx = Tylist_Table[param_tyl_idx+1];
  TY_IDX p3_ty_idx = Tylist_Table[param_tyl_idx+2];
  TY_IDX p4_ty_idx = Tylist_Table[param_tyl_idx+3];
  TY_IDX p5_ty_idx = Tylist_Table[param_tyl_idx+4];  
  TY_IDX p6_ty_idx = Tylist_Table[param_tyl_idx+5];
  TY_IDX p7_ty_idx = Tylist_Table[param_tyl_idx+6];
  TY_IDX p8_ty_idx = Tylist_Table[param_tyl_idx+7];  
  TY_IDX p9_ty_idx = Tylist_Table[param_tyl_idx+8];
  
  WN *wn = WN_Call(MTYPE_V, MTYPE_V, 9, func_st_idx);
  
  // Construct the arguments.
  WN_kid0(wn) = HCWN_Parm(TY_mtype(p1_ty_idx), p1, p1_ty_idx);
  WN_kid1(wn) = HCWN_Parm(TY_mtype(p2_ty_idx), p2, p2_ty_idx);
  WN_kid2(wn) = HCWN_Parm(TY_mtype(p3_ty_idx), p3, p3_ty_idx);
  WN_kid(wn, 3) = HCWN_Parm(TY_mtype(p4_ty_idx), p4, p4_ty_idx);
  WN_kid(wn, 4) = HCWN_Parm(TY_mtype(p5_ty_idx), p5, p5_ty_idx);
  WN_kid(wn, 5) = HCWN_Parm(TY_mtype(p6_ty_idx), p6, p6_ty_idx);
  WN_kid(wn, 6) = HCWN_Parm(TY_mtype(p7_ty_idx), p7, p7_ty_idx);
  WN_kid(wn, 7) = HCWN_Parm(TY_mtype(p8_ty_idx), p8, p8_ty_idx);
  WN_kid(wn, 8) = HCWN_Parm(TY_mtype(p9_ty_idx), p9, p9_ty_idx);

  return wn;
}

WN* 
call_clBarrier(WN *p1) {
  // Get the function symbol.
  assert(clBarrier_st_idx != ST_IDX_ZERO);
  ST_IDX func_st_idx = clBarrier_st_idx;
  
  // Get the parameter types.
  TY_IDX func_ty_idx = ST_pu_type(func_st_idx);
  TYLIST_IDX param_tyl_idx = TY_parms(func_ty_idx);
  TY_IDX p1_ty_idx = Tylist_Table[param_tyl_idx+0];  

  WN *wn = WN_Call(MTYPE_V, MTYPE_V, 1, func_st_idx);
  
  // Construct the arguments.
  WN_kid0(wn) = HCWN_Parm(TY_mtype(p1_ty_idx), p1, p1_ty_idx);

  return wn;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

void HC_rebuild_cuda_enum_type(WN *wn)
{
    if (! cuda_includes_initialized) return;

    for (WN_ITER* wni = WN_WALK_TreeIter(wn); wni != NULL;
            wni = WN_WALK_TreeNext(wni))
    {
        WN *wn = WN_ITER_wn(wni);
        if (WN_operator(wn) != OPR_CALL) continue;

        // Is it a cudaMemcpy?
        ST_IDX call_st_idx = WN_st_idx(wn);
        if (call_st_idx == cudaMemcpy_st_idx
                || call_st_idx == cudaMemcpyToSymbol_st_idx)
        {
            // The enum type constant is the last parameter.
            WN *param_wn = WN_kid0(WN_actual(wn,WN_kid_count(wn)-1));
            Is_True(WN_operator(param_wn) == OPR_INTCONST, (""));
            WN_set_ty(param_wn, enum_cudaMemcpyKind_ty_idx);
        }
    }
}

/*** DAVID CODE END ***/
