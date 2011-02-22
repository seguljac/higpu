/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_PREPROCESS_H_
#define _IPA_HC_PREPROCESS_H_

/*****************************************************************************
 *
 * Interprocedural pre-processing routines
 *
 ****************************************************************************/

class IPA_NODE;

extern void HC_preprocess(IPA_NODE *node);

/*****************************************************************************
 *
 * Classify each procedure into three DISJOINT groups:
 * - K:  contains_kernel
 * - MK: may_lead_to_kernel but not contains_kernel
 * - IK: may_be_inside_kernel
 * - N:  others
 *
 * Works on IPA_Call_Graph
 *
 ****************************************************************************/

extern void IPA_HC_classify_procedure();

/*****************************************************************************
 *
 * - Make sure that no GLOBAL/CONSTANT directives in IK-procedures, and warn
 *   if there are these directives in N-procedures.
 *
 * - Make sure that LOOP_PARTITION directives are only in IK-,K-procedures.
 *   For K-procedures, we need to validate by another tree traversal.
 *
 * This is called after procedure classification.
 *
 ****************************************************************************/

extern void HC_post_validate(IPA_NODE *node);

#endif  // _IPA_HC_PREPROCESS_H_

/*** DAVID CODE END ***/
