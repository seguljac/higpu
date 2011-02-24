/** DAVID CODE BEGIN **/

#ifndef _HICUDA_CUDA_UTILS_H_
#define _HICUDA_CUDA_UTILS_H_

#include "hc_utils.h"

extern BOOL flag_opencl;

// TODO: automatically generate these declarations?

enum cudaError
{
    cudaSuccess = 0,
    cudaErrorMissingConfiguration,
    cudaErrorMemoryAllocation,
    cudaErrorInitializationError,
    cudaErrorLaunchFailure,
    cudaErrorPriorLaunchFailure,
    cudaErrorLaunchTimeout,
    cudaErrorLaunchOutOfResources,
    cudaErrorInvalidDeviceFunction,
    cudaErrorInvalidConfiguration,
    cudaErrorInvalidDevice,
    cudaErrorInvalidValue,
    cudaErrorInvalidPitchValue,
    cudaErrorInvalidSymbol,
    cudaErrorMapBufferObjectFailed,
    cudaErrorUnmapBufferObjectFailed,
    cudaErrorInvalidHostPointer,
    cudaErrorInvalidDevicePointer,
    cudaErrorInvalidTexture,
    cudaErrorInvalidTextureBinding,
    cudaErrorInvalidChannelDescriptor,
    cudaErrorInvalidMemcpyDirection,
    cudaErrorAddressOfConstant,
    cudaErrorTextureFetchFailed,
    cudaErrorTextureNotBound,
    cudaErrorSynchronizationError,
    cudaErrorInvalidFilterSetting,
    cudaErrorInvalidNormSetting,
    cudaErrorMixedDeviceExecution,
    cudaErrorCudartUnloading,
    cudaErrorUnknown,
    cudaErrorNotYetImplemented,
    cudaErrorMemoryValueTooLarge,
    cudaErrorInvalidResourceHandle,
    cudaErrorNotReady,
    cudaErrorStartupFailure = 0x7f,
    cudaErrorApiFailureBase = 10000
};

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    NUM_enum_cudaMemcpyKind
};

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// Indexing for CUDA
extern ST_IDX blockIdx_st_idx;
extern ST_IDX threadIdx_st_idx;

// Indexing for OpenCL
extern ST_IDX blockIdxX_st_idx;
extern ST_IDX blockIdxY_st_idx;
extern ST_IDX blockIdxZ_st_idx;
extern ST_IDX threadIdxX_st_idx;
extern ST_IDX threadIdxY_st_idx;
extern ST_IDX threadIdxZ_st_idx;

extern TY_IDX dim3_ty_idx;
extern TY_IDX uint3_ty_idx;

// OpenCL types
extern TY_IDX cl_mem_ty_idx;
extern TY_IDX cl_context_ty_idx;
extern TY_IDX cl_command_queue_ty_idx;
extern TY_IDX cl_kernel_ty_idx;
extern TY_IDX cl_program_ty_idx;

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

/* Must be called at the beginning of processing. */
extern void init_cuda_includes();

/* The following functions have no side effects. */

extern WN* call_cudaMalloc(WN *ptr_addr, WN *size);

extern WN* call_cudaMemcpy(WN *dst, WN *src, WN *count,
    enum cudaMemcpyKind kind);

extern WN* call_cudaMemcpyToSymbol(ST_IDX st_idx, WN *src,
    WN *count, WN *offset, enum cudaMemcpyKind kind);

extern WN* call_cudaMemset(WN *dst_wn, WN *value_wn, WN *count_wn);

extern WN* call_cudaFree(WN *ptr);

extern WN* call_syncthreads();

/* OpenCL memory handling functions */

extern WN* call_clCreateBufferRet(WN *p1, WN* p2, WN* p3, WN *p4, 
				  WN *p5, WN *p6);
extern WN* call_clEnqueueReadBuffer(WN *p1, WN* p2, WN* p3, WN *p4, 
				    WN *p5, WN* p6, WN *p7, WN *p8, WN* p9);
extern WN* call_clEnqueueWriteBuffer(WN *p1, WN* p2, WN* p3, WN *p4, WN *p5, 
				     WN* p6, WN *p7, WN *p8, WN *p9);
extern WN* call_clEnqueueWriteCleanBuffer(WN *p1, WN* p2, WN* p3, WN *p4, 
					  WN *p5, WN* p6, WN *p7, WN *p8, WN *p9);
extern WN* call_clReleaseMemObj(WN *p1);

/* OpenCL kernel handling functions */

extern WN* call_clCreateKernelRet(WN *p1, WN* p2, WN* p3, WN *p4);

extern WN* call_clSetKernelArg(WN *p1, WN* p2, WN* p3, WN *p4);

extern WN* call_clEnqueueNDRangeKernel(WN *p1, WN* p2, WN* p3, WN *p4, 
				       WN *p5, WN* p6, WN *p7, WN *p8, WN *p9);

/* OpenCL synchronization functions */

extern WN* call_clBarrier(WN *p1);

/**
 * Access a particular field of the blockIdx variable.
 * 'field_id' is zero-based.
 */
inline WN* ldid_blockIdx(UINT32 field_id)
{
  if (flag_opencl){
    if (field_id == 0){
      return WN_LdidScalar(blockIdxX_st_idx);
    } else if (field_id == 1){
      return WN_LdidScalar(blockIdxY_st_idx);
    } else {
      return WN_LdidScalar(blockIdxZ_st_idx);
    }
  } else {
    return HCWN_LdidStructField(blockIdx_st_idx, field_id+1);
  }
}

/**
 * Access a particular field of the threadIdx variable.
 * 'field_id' is zero-based.
 */
inline WN* ldid_threadIdx(UINT32 field_id)
{
  if (flag_opencl){
    if (field_id == 0){
      return WN_LdidScalar(threadIdxX_st_idx);
    } else if (field_id == 1){
      return WN_LdidScalar(threadIdxY_st_idx);
    } else {
      return WN_LdidScalar(threadIdxZ_st_idx);
    }
  } else {
    return HCWN_LdidStructField(threadIdx_st_idx, field_id+1);
  }
}

/*****************************************************************************
 *
 * Rebuild the enum type constant for CUDA runtime calls in <wn>.
 *
 ****************************************************************************/

extern void HC_rebuild_cuda_enum_type(WN *wn);

#endif  // _HICUDA_CUDA_UTILS_H_

/*** DAVID CODE END ***/
