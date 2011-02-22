/** DAVID CODE BEGIN **/

#ifndef _HICUDA_TY_CLEANUP_H_
#define _HICUDA_TY_CLEANUP_H_

#include "wn.h"

/**
 * Initialize internal data structures.
 */
extern void init_type_internal_data();

/**
 * Routines used to determine equivalence classes of types in Ty_Table and
 * adjust type references in the WN tree to the representative of the
 * type class.
 */

/**
 * Determine sets of identical types in Ty_Table.
 */
extern void find_ident_types();

/**
 * For each symbol, if its type has a set of identical types in Ty_Table,
 * adjust it to be the set's representative.
 */
extern void replace_types_in_symtab(SYMTAB_IDX level);

/**
 * Save as above, except this is a WN_HANDLER that replaces types in
 * a WHIRL node.
 */
extern WN* replace_types_in_wn(WN *wn, WN *parent, bool *del_wn);

/**
 * Free internal data structures.
 */
extern void reset_type_internal_data();

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#endif  // _HICUDA_TY_CLEANUP_H_

/*** DAVID CODE END ***/
