/** DAVID CODE BEGIN **/

#ifndef _HICUDA_HC_UTILS_H_
#define _HICUDA_HC_UTILS_H_

#include "pu_info.h"

#include "cxx_template.h"
#include "cxx_hash.h"

/**
 * Convert 'loop' to a DO_LOOP and fix all linkage issues.
 *
 * This routine does not change the init, update and check statements.
 *
 * Return the replaced DO_LOOP node if it is legal and successful,
 * and false otherwise.
 */
extern WN* convert_to_doloop(WN *loop, WN *parent);

extern WN* get_loop_step(WN *doloop);

struct doloop_info {
    WN *init_val;
    WN *step_val;
    // past-end bound (i.e. through LT or GT comparison)
    WN *end_val;

    bool flip_sign;

    WN *tripcount;
};

/**
 * Construct a TRIPLET that represents the range of values the loop index
 * variable could take. The starting index is always no greater than the
 * ending index. This is only possible when the sign of the loop step is
 * known. Otherwise, NULL is returned.
 */
extern WN* get_loop_idx_range(struct doloop_info *li);

extern void clean_doloop_info(struct doloop_info *li);

/**
 * Normalize a DO_LOOP.
 *
 * The DO_LOOP after this routine will have the following properties:
 * - The check statement is either LT or GT.
 * - The LHS of the comparison is the index variable.
 * - The ADD expression in the update statement is <idx_var> + <step>.
 *
 * The init, end and step values do not have to be constant. The tripcount
 * will also be determined: 0 for empty loop and NULL for infinite loop.
 *
 * For now, the loop can always be normalized, and 'info' will be filled
 * and true returned. Otherwise, false is returned.
 *
 * This routine is intended to be a preprocessing stage for kernel directive
 * handling. There are two approaches to handle a kernel directive:
 * 1) normalize the loop (which may introduce a new induction variable),
 *    tile the loop and outline,
 *    induction variable elimination on the kernel function
 * 2) gather loop information,
 *    directly tile the loop without normalization (no new induction var),
 *    induction variable elimination on the kernel function
 *
 * For now, I will adopt the 2nd approach because the advantage of the 1st
 * approach is that some of loops have already been normalized, which will
 * facilitate some optimizations. However, only the loops that will be
 * tiled are normalized while other loops are still left as they are. Also,
 * I do not want to pass information like which loops are normalized to the
 * optimization routine of the kernel function. Therefore, the advantage of
 * the 1st approach does not exist. Now, the 2nd approach can produce
 * cleaner code at the CUDA level, which may be inspected by the users, and
 * it does not introduce a new induction variable.
 */
extern bool normalize_loop(WN *doloop, struct doloop_info *info);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Declare a function symbol, including its type and PU entry.
 *
 * 'st_name' - name of the function symbol
 * 'ty_name' - name of the function prototype
 * 'ret_ty' - return type of the function
 * 'nparams' - number of parameters
 *
 * Return the function symbol index.
 *
 * NOTE: if a symbol of name 'st_name' already exists, it is turned
 * into a function symbol.
 */
extern ST_IDX declare_function_va(const char *st_name, const char *ty_name,
        TY_IDX ret_ty, int nparams, ...);
extern ST_IDX declare_function_va(STR_IDX st_name, const char *ty_name,
        TY_IDX ret_ty, int nparams, ...);

/**
 * Save as above, except parameter types are passed in as an array.
 *
 * When nparams is 0, params can be NULL.
 */
extern ST_IDX declare_function(const char *st_name, const char *ty_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params);
extern ST_IDX declare_function(STR_IDX func_str_idx, const char *ty_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params);
extern ST_IDX declare_function(ST_IDX func_st_idx, const char *ty_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params);

extern TY_IDX new_func_type(const char *ty_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params);

/**
 * A convenience routine for declaring a CUDA kernel function.
 */
extern void declare_kernel(ST_IDX func_st_idx,
        TY_IDX ret_ty, int nparams, TY_IDX *params);

/**
 * Save as above, except the type name is automatically constructed from
 * the symbol name by appending '.ty'.
 */
extern ST_IDX declare_function(const char *st_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params);
extern ST_IDX declare_function(STR_IDX st_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params);
extern ST_IDX declare_function(ST_IDX func_st_idx,
        TY_IDX ret_ty, int nparams, TY_IDX *params);

/**
 * Create a local variable (no matter if it exists or not).
 */
extern ST_IDX new_local_var(const char *name, TY_IDX ty);
extern ST_IDX new_local_var(STR_IDX str_idx, TY_IDX ty);
extern ST_IDX new_formal_var(const char *name, TY_IDX ty);
extern ST_IDX new_formal_var(STR_IDX str_idx, TY_IDX ty);
extern ST_IDX new_extern_var(const char *name, TY_IDX ty);
extern ST_IDX new_extern_var(STR_IDX str_idx, TY_IDX ty_idx);
extern ST_IDX new_global_var(STR_IDX str_idx, TY_IDX ty_idx);
extern ST_IDX new_global_var(const char *name, TY_IDX ty_idx);

/**
 * Search for a symbol in a given symbol table level.
 */
extern ST_IDX lookup_symbol(const char *st_name, UINT8 st_level);
extern ST_IDX lookup_symbol(STR_IDX st_name, UINT8 st_level);

/**
 * Search for a function symbol with the given name in the global scope.
 */
extern ST_IDX lookup_function(const char *fname);

/**
 * Search for a local variable with the given name.
 */
extern ST* lookup_localvar(const char *vname);
extern ST_IDX lookup_localvar(STR_IDX name_str_idx);

/**
 * Search for a type with the given name.
 */
extern TY_IDX lookup_type(const char *tname);

extern ST_IDX lookup_extern_var(const char *vname);

/**
 * Replace all occurences of 'from_st_idx' with 'to_st_idx' in 'wn'.
 *
 * Return the number of occurences.
 */
extern int replace_symbol(WN *wn, ST_IDX from_st_idx, ST_IDX to_st_idx);

/**
 * Make a copy of the given symbol in the current scope, but the
 * new symbol's level (ST_IDX_level) is 'dst_level'.
 */
extern ST_IDX ST_copy(ST_IDX st_idx, SYMTAB_IDX dst_level);

/**
 * Go through all symbols in the current lexical level, and change
 * their level to the given one. All ST_IDXs in INITO_TAB and ST_ATTR_TAB
 * are also fixed.
 */
extern void fix_lexical_level(SYMTAB_IDX level);

/**
 * Make a copy of each local symbol referenced in 'wn' in the current scope,
 * but the new symbol's level (ST_IDX_level) is 'dst_level'.
 *
 * When transferring the symbol, also transfer its attributes in ST_ATTR_TAB.
 *
 * For those symbols in 'params', their copied symbols will be stored in
 * the corresponding location in 'new_params'.
 */
extern void transfer_symbols(WN *wn,
    SYMTAB_IDX dst_level, int nparams, ST_IDX *params, ST_IDX *new_params);

/**
 * Verify two things about the given REGION 'region_wn':
 *
 * 1) There is no jump in FUNC_ENTRY 'parent_func_wn' from outside
 *    'region_wn' to within 'region_wn'.
 *
 * 2) The only exit of 'region_wn' is the fall through.
 *
 * Return true iff both of them are true.
 *
 * 'region_wn' is assumed to have REGION_EXITs set up properly.
 */
extern bool verify_region_labels(WN *region_wn, WN *parent_func_wn);

/**
 * Insert all labels in REGION 'region_wn' (at level 'from_level') to the
 * current LABEL_TAB.
 *
 * 'region_wn' is assumed to pass the label verification (i.e.
 * verify_region_labels returns true.
 */
extern void transfer_labels(WN *region_wn, SYMTAB_IDX from_level);

/**
 * Generate a unique variable name '<var_name><suffix><num>'.
 *
 * If var_st_idx is 0, it is treated as empty string. Same when suffix == NULL.
 */
extern STR_IDX gen_var_str(ST_IDX var_st_idx, const char *suffix);

/**
 * Generate a unique variable name '<prefix><var_name><num>'.
 *
 * If var_st_idx is 0, it is treated as empty string. Same when prefix == NULL.
 */
extern STR_IDX gen_var_str(const char *prefix, ST_IDX var_st_idx);

/**
 * Get the array dimensionality of array type 'arr_ty_idx'.
 */
extern UINT16 num_array_dims(TY_IDX arr_ty_idx);

/**
 * Find the element type of 'ty_idx'.
 *
 * If 'ty_idx' is not KIND_ARRAY, itself is returned.
 */
extern TY_IDX arr_elem_ty(TY_IDX ty_idx);

/*****************************************************************************
 *
 * Analyze the given array type:
 * 1) Find the array element type (being returned).
 * 2) Find the dimensionality (stored in "ndims").
 *
 *****************************************************************************/

extern TY_IDX analyze_arr_ty(TY_IDX arr_ty_idx, UINT *ndims);

/**
 * Declare an array type that has 'ndims' dimensions, whose sizes are
 * specified in 'dim_sz', with element of type 'elem_ty_idx'.
 */
extern TY_IDX make_arr_type(STR_IDX name_str_idx,
    UINT16 ndims, UINT32 *dim_sz, TY_IDX elem_ty_idx);

// Declare a 1-D array type with incomplete bounds.
extern TY_IDX make_incomplete_arr_type(STR_IDX name_str_idx,
        TY_IDX elem_ty_idx);

/**
 * Set the size of dimension 'dim' in type 'arr_ty' to be 'dim_sz', which
 * is a constant value.
 *
 * Return true if the dimension is found and false otherwise.
 */
extern bool set_arr_dim_sz(TY_IDX arr_ty_idx, UINT dim, UINT dim_sz);

/** 
 * Calculate the size of the given array dimension:
 *      ubnd_val - lbnd_val + 1
 */
extern WN* array_dim_size(const ARB_HANDLE &ah);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Generate a unique loop index variable symbol in the local scope, that
 * has the suffix "_<nesting_level>".
 *
 * nesting_level starts from 0.
 */
extern ST_IDX make_loop_idx(UINT32 nesting_level, bool is_unsigned = false);

/**
 * Make an empty DO_LOOP node that has unit step and LE compare operation.
 *
 * for (i = init_wn; i <= end_wn; ++i)
 *
 * Copies of 'init_wn' and 'end_wn' will be made before being used in
 * constructing the loop header.
 */
extern WN* make_empty_doloop(ST_IDX idxv_st_idx,
    WN *init_wn, WN *end_wn, WN *step_wn = NULL);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Clean up unused symbols (global and local).
 */ 
extern void cleanup_symbols(PU_Info *pu_root);

/**
 * Find the function body corresponding to a function symbol.
 */ 
extern WN* find_func_body(const PU_Info *pu_tree, ST_IDX func_st_idx);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Replace the kid of 'parent', 'wn' with 'new_wn'. 'wn' is NOT deallocated.
 *
 * Return true if 'wn' is the kid of 'parent' and false otherwise.
 */
extern bool replace_wn_kid(WN *parent, WN *wn, WN *new_wn);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Create an STID node that assigns a value to a struct's field.
 * The field id starts with 1.
 *
 * Does not work with nested structs.
 */
extern WN* HCWN_StidStructField(TYPE_ID desc,
    ST_IDX st_idx, UINT32 field_id, WN *value);

extern WN* HCWN_StidStructField(ST_IDX st_idx, UINT32 field_id, WN *value);

/**
 * Create a LDID node that loads the given component of a struct variable.
 * The field id starts with 1.
 *
 * Does not work with nested structs.
 */
extern WN* HCWN_LdidStructField(TYPE_ID desc, ST_IDX st_idx, UINT32 field_id);

extern WN* HCWN_LdidStructField(ST_IDX st_idx, UINT32 field_id);

/**
 * 'var' must be either LDA or LDID.
 */
extern WN* HCWN_CreateArray(WN *var, UINT ndims);

/**
 * A convenience method for creating a parameter node.
 */
extern WN* HCWN_Parm(TYPE_ID rtype, WN *parm_node, TY_IDX ty_idx);

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Check if the given two LDID nodes are equivalent. They are considered
 * equivalent when the symbol and offset are equal.
 */
extern bool HCWN_are_ldids_equal(WN *ldid1, WN *ldid2);

/**
 * This routine is based on WN_SimplifyExp1, and serves two purposes:
 * - Change the interface so that the returned node will never be NULL;
 *   if no simplification occurs, the input node is returned.
 * - Add more simplification scenarios.
 */
extern WN* HCWN_SimplifyExp1(WN *wn, bool *simplified = NULL);

/**
 * 'wn1' and 'wn2' are simplified integer expressions.
 *
 * Return ceil(wn1/wn2) of Integer_type.
 * If 'perfect_div' is not NULL, it will store whether or not 'wn2' divides
 * 'wn1', i.e. ceil(wn1/wn2) = wn1/wn2.
 *
 * 'wn1' and 'wn2' are copied before use.
 */
extern WN* HCWN_Ceil(WN *wn1, WN *wn2, bool *perfect_div = NULL);

/*****************************************************************************
 *
 * Return TRUE if the given WN node contains references to non-global
 * variable symbols, and FALSE otherwise.
 *
 * Symbols in <exception_list> are not checked.
 *
 ****************************************************************************/

extern BOOL HCWN_contains_non_global_syms(WN *wn,
        DYN_ARRAY<ST_IDX> *exception_list = NULL);

/*****************************************************************************
 *
 * We do not care about CUDA symbols.
 *
 ****************************************************************************/

extern void HCWN_build_st_table(WN *wn, HASH_TABLE<ST_IDX, ST*> *st_table);

/*****************************************************************************
 *
 * For each symbol referenced in the given WN node, create a formal symbol to
 * replace it. This formal symbol is identical to the original symbol except
 * the CLASS type, and is created based on the given symbol table. A map from
 * the old symbol to the formal is stored in <new_formal_map>.
 *
 ****************************************************************************/

extern void HCWN_replace_syms(WN *wn,
        HASH_TABLE<ST_IDX,ST_IDX> *new_formal_map,
        const HASH_TABLE<ST_IDX,ST*> *st_tbl);

/*****************************************************************************
 *
 * Replace each occurrence of source symbols of <map> in <wn> with the
 * corresponding target symbols.
 *
 ****************************************************************************/

extern void HCWN_replace_syms(WN *wn,
        const HASH_TABLE<ST_IDX,ST_IDX> *map);

// This function does not require access to any symbol table.
inline BOOL HCST_is_global_symbol(ST_IDX st_idx)
{
    return (ST_IDX_level(st_idx) <= GLOBAL_SYMTAB);
}

inline BOOL HCST_is_local_symbol(ST_IDX st_idx)
{
    return (ST_IDX_level(st_idx) == CURRENT_SYMTAB);
}

inline BOOL HCST_is_scalar(ST_IDX st_idx)
{
    TY_IDX ty_idx = ST_type(st_idx);
    TY_KIND ty_kind = TY_kind(ty_idx);
    return (ty_kind == KIND_SCALAR || ty_kind == KIND_STRUCT);
}

/*****************************************************************************
 *
 * Check if the given is of ARRAY type or pointer-to-ARRAY type.
 *
 ****************************************************************************/

inline BOOL HCST_is_array(ST_IDX st_idx)
{
    TY_IDX ty_idx = ST_type(st_idx);
    TY_KIND ty_kind = TY_kind(ty_idx);

    if (ty_kind == KIND_ARRAY) return TRUE;
    if (ty_kind != KIND_POINTER) return FALSE;

    return TY_kind(TY_pointed(ty_idx)) == KIND_ARRAY;
}

/*****************************************************************************
 *
 * A pointer parameter of a call within a kernel region must be one of the
 * following two:
 * 1) LDID of an array pointer variable (with zero offset)
 * 2) LDA of an array variable (with zero offset)
 *
 * This function checks the given parameter node (the one within PARM, not the
 * PARM itself). It return the array symbol or NULL if failed.
 *
 ****************************************************************************/

inline ST_IDX HCWN_verify_pointer_param(WN *param_wn)
{
    OPERATOR opr = WN_operator(param_wn);
    if (opr != OPR_LDID && opr != OPR_LDA) return ST_IDX_ZERO;

    if (WN_offset(param_wn) != 0) return ST_IDX_ZERO;

    ST_IDX st_idx = WN_st_idx(param_wn);
    TY_IDX ty_idx = ST_type(st_idx);
    if (opr == OPR_LDID)
    {
        return (TY_kind(ty_idx) == KIND_POINTER
                && TY_kind(TY_pointed(ty_idx)) == KIND_ARRAY) ?
            st_idx : ST_IDX_ZERO;
    }
    else
    {
        return (TY_kind(ty_idx) == KIND_ARRAY) ? st_idx : ST_IDX_ZERO;
    }
}

extern void HCWN_check_parentize(const WN *wn, WN_MAP map);

extern void HCWN_check_map_id(const WN *wn);

extern BOOL HCTY_is_dyn_array(TY_IDX ty_idx);

#endif  // _HICUDA_HC_UTILS_H_

/*** DAVID CODE END ***/
