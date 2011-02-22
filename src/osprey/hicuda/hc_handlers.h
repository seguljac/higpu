/** DAVID CODE BEGIN **/

#ifndef _HICUDA_HC_HANDLERS_H_
#define _HICUDA_HC_HANDLERS_H_

/**
 * Routines that handle the following HiCUDA directives:
 *
 * - GLOBAL COPYIN
 * - GLOBAL COPYOUT
 * - GLOBAL FREE
 * - CONST COPYIN
 * - CONST REMOVE
 * - SHARED COPYIN
 * - SHARED COPYOUT
 * - SHARED REMOVE
 * - KERNEL
 * - KERNEL PART
 * - BARRIER
 * - LOOPBLOCK
 *
 * They are called by lower_hicuda_pragmas in driver.cxx.
 *
 * The BEGIN/END directives have been handled in the front-end.
 */

/**
 * 'parent' must be a BLOCK. 'pragma' is a PRAGMA/XPRAGMA inside
 * 'parent'. This function may consume any follow-up pragmas using
 * WN_next.
 * 
 * Return the next node to be processed in the block.
 */
extern WN* lower_hc_global_copyin(WN *parent, WN *pragma);
extern WN* lower_hc_global_copyout(WN *parent, WN *pragma);
extern WN* lower_hc_global_free(WN *parent, WN *pragma);
extern WN* lower_hc_const_copyin(WN *parent, WN *pragma);
extern WN* lower_hc_const_remove(WN *parent, WN *pragma);
extern WN* lower_hc_shared_copyin_list(WN *parent, WN *pragma);
extern WN* lower_hc_shared_copyout(WN *parent, WN *pragma);
extern WN* lower_hc_shared_remove(WN *parent, WN *pragma);

/**
 * Determine allocation of all shared variables in the current kernel.
 */
extern void allocate_svars();

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/**
 * 'parent' must be a BLOCK. 'pragma' is a PRAGMA/XPRAGMA inside
 * 'parent'. This function may consume any follow-up pragmas using
 * WN_next.
 * 
 * Return the next node to be processed in the block.
 */
extern WN* lower_hc_barrier(WN *parent, WN *pragma);

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/**
 * 'region' must be a REGION. 'pragma' is a PRAGMA/XPRAGMA in the
 * pragma block of 'region'. This function may consume any follow-up
 * pragmas using WN_next.
 *
 * Return the next pragma to be processed in the pragma block.
 */
extern WN* lower_hc_kernel(WN *region, WN *pragma);

/**
 * Outline the kernel function.
 * Called at the end of a kernel REGION.
 */
extern void outline_kernel(WN *region);

/**
 * 'region' must be a REGION. 'parent' is a BLOCK that contains 'region'.
 * 'pragma' is a PRAGMA/XPRAGMA in the pragma block of 'region'.
 * This function may consume any follow-up pragmas using WN_next.
 *
 * Return the next pragma to be processed in the pragma block.
 */
extern WN* lower_hc_kernel_part(WN *region, WN *pragma);

/**
 * Update the kernel context at the end of a kernel part REGION.
 * This includes dimension index of the virtual block/thread space,
 * and the loop partition info.
 */
extern void end_kernel_part_region();

/**
 * When a non-partitioned loop ends inside a kernel, update the state
 * of the loop stack.
 */
extern void end_loop_in_kernel(WN *loop);

/**
 * Do extra processing when inside a kernel. For now it includes:
 * - replace a variable with the corresponding shared variable
 *
 * Same interface as WN_HANDLER
 */
extern WN* kernel_processing(WN *wn, WN *parent, bool *del_wn);

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/**
 * 'region' must be a REGION. 'pragma' is a PRAGMA/XPRAGMA in the
 * pragma block of 'region'. This function may consume any follow-up
 * pragmas using WN_next.
 *
 * Return the next pragma to be processed in the pragma block.
 */
extern WN* lower_hc_loopblock(WN *region, WN *pragma);

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/**
 * Reset internal states after a PU ends, including:
 * - clear the declarations of 'dimGrid' and 'dimBlock'.
 */
extern void reset_handler_states_at_pu_end();

#endif  // _HICUDA_HC_HANDLERS_H_

/*** DAVID CODE END ***/
