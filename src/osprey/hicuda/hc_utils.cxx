/** DAVID CODE BEGIN **/

#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

#include <map>
#include <set>

#include "wn.h"
#include "wn_util.h"
#include "wn_simp.h"
#include "pu_info.h"
#include "ir_bread.h"
#include "ir_bwrite.h"
#include "tracing.h"

#include "hc_common.h"
#include "hc_utils.h"

#include "ipc_symtab_merge.h"

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

inline bool
check_ldid(WN *wn, ST_IDX var_st_idx) {
    return (WN_operator(wn) == OPR_LDID) &&
           (WN_st_idx(wn) == var_st_idx);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

WN* HCWN_StidStructField(TYPE_ID desc,
        ST_IDX st_idx, UINT32 field_id, WN *value)
{
    ST *st = ST_ptr(st_idx);
    assert(st != NULL);

    // Get the struct type.
    TY_IDX ty_idx = ST_type(st);
    assert(TY_kind(ty_idx) == KIND_STRUCT);

    // Get the field index.
    FLD_IDX fld_idx = Ty_Table[ty_idx].Fld() + field_id - 1;
    FLD &fld = Fld_Table[fld_idx];

    return WN_Stid(desc, fld.ofst, st, fld.type, value, 0);
}

WN*
HCWN_StidStructField(ST_IDX st_idx, UINT32 field_id, WN *value) {
    ST *st = ST_ptr(st_idx);
    assert(st != NULL);

    // Get the struct type.
    TY_IDX ty_idx = ST_type(st);
    assert(TY_kind(ty_idx) == KIND_STRUCT);

    // Get the field index.
    FLD_IDX fld_idx = Ty_Table[ty_idx].Fld() + field_id - 1;
    FLD &fld = Fld_Table[fld_idx];

    // Get the field's type.
    TY_IDX fld_ty = fld.type;
    TYPE_ID fld_mtype = TY_mtype(fld.type);

    // Use the field's MTYPE as 'desc'.
    return WN_Stid(fld_mtype, fld.ofst, st, fld_ty, value, 0);
}

WN*
HCWN_LdidStructField(TYPE_ID desc, ST_IDX st_idx, UINT32 field_id) {
    ST *st = ST_ptr(st_idx);
    assert(st != NULL);

    // Get the struct type.
    TY_IDX ty_idx = ST_type(st);
    assert(TY_kind(ty_idx) == KIND_STRUCT);

    // Get the field index.
    FLD_IDX fld_idx = Ty_Table[ty_idx].Fld() + field_id - 1;
    FLD &fld = Fld_Table[fld_idx];

    return WN_Ldid(desc, fld.ofst, st_idx, fld.type, 0);
}

WN*
HCWN_LdidStructField(ST_IDX st_idx, UINT32 field_id) {
    ST *st = ST_ptr(st_idx);
    assert(st != NULL);

    // Get the struct type.
    TY_IDX ty_idx = ST_type(st);
    assert(TY_kind(ty_idx) == KIND_STRUCT);

    // Get the field index.
    FLD_IDX fld_idx = Ty_Table[ty_idx].Fld() + field_id - 1;
    FLD &fld = Fld_Table[fld_idx];

    // Get the field's type.
    TYPE_ID fld_mtype = TY_mtype(fld.type);

    // Use the field's MTYPE as 'desc'.
    return WN_Ldid(fld_mtype, fld.ofst, st_idx, fld.type, 0);
}

WN* HCWN_CreateArray(WN *var, UINT ndims)
{
    Is_True(var != NULL, ("Base address is NULL in ARRSECTION!\n"));

    OPERATOR opr = WN_operator(var);
    Is_True(opr == OPR_LDA || opr == OPR_LDID,
        ("Bad base address in ARRSECTION!\n"));

    WN *wn = WN_Create(OPR_ARRAY, Pointer_type, MTYPE_V, ndims*2+1);
    WN_kid0(wn) = var;

    return wn;
}

WN* HCWN_Parm(TYPE_ID rtype, WN *parm_node, TY_IDX ty_idx)
{
    return WN_CreateParm(rtype, parm_node, ty_idx, WN_PARM_BY_VALUE);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

WN*
get_loop_step(WN *doloop) {
    WN *upd_stmt = WN_kid(doloop,3);
    return WN_kid1(WN_kid0(upd_stmt));
}

WN*
convert_to_doloop(WN *loop, WN *parent) {
    OPCODE opcode = WN_opcode(loop);
    assert(opcode != OPC_DO_LOOP);

    assert(WN_opcode(parent) == OPC_BLOCK);

    // For now, we only handle WHILE_DO loops.
    if (opcode != OPC_WHILE_DO) return NULL;

    /* Extract the three components of a DO_LOOP */

    // The statement before the loop should be
    // the init of index variable.
    WN *init_stmt = WN_prev(loop);
    if (init_stmt == NULL) return NULL;

    // Check the validity of the init statement.
    if (WN_operator(init_stmt) != OPR_STID) return NULL;

    // Get the index variable.
    ST_IDX idx_var = WN_st_idx(init_stmt);
    if (ST_class(idx_var) != CLASS_VAR) return NULL;
    // Make sure that it is an integer variable.
    TY_IDX idx_var_ty = ST_type(idx_var);
    if (TY_kind(idx_var_ty) != KIND_SCALAR) return NULL;
    if (! MTYPE_is_integral(TY_mtype(idx_var_ty))) return NULL;

    // Get the check statement.
    WN *check_stmt = WN_kid0(loop);
    assert(check_stmt != NULL);
    // Check if it is a comparison using GE,GT,LE,LT.
    OPERATOR opr = WN_operator(check_stmt);
    if (opr != OPR_GE && opr != OPR_GT &&
        opr != OPR_LE && opr != OPR_LT) return NULL;

    // Check if LHS is the index variable and RHS is a loop-invariant
    // expression.
    WN *lhs = WN_kid0(check_stmt), *rhs = WN_kid1(check_stmt);
    if (check_ldid(rhs, idx_var)) {
        // Swap the two sides.
        lhs = rhs;
        rhs = WN_kid0(check_stmt);
        WN_kid0(check_stmt) = lhs;
        WN_kid1(check_stmt) = rhs;
        // Flip the operator.
        switch (opr) {
            case OPR_GE: WN_set_operator(check_stmt, OPR_LE); break;
            case OPR_LE: WN_set_operator(check_stmt, OPR_GE); break;
            case OPR_GT: WN_set_operator(check_stmt, OPR_LT); break;
            case OPR_LT: WN_set_operator(check_stmt, OPR_GT); break;
            default:    abort();
        }
        // TODO: check if the new RHS is loop-invariant.
    } else if (check_ldid(lhs, idx_var)) {
        // TODO: check if RHS is loop-invariant.
    } else {
        return NULL;
    }

    // Get the update statement.
    WN *loop_body = WN_kid1(loop);
    assert(loop_body != NULL && WN_opcode(loop_body) == OPC_BLOCK);
    WN *upd_stmt = WN_last(loop_body);
    // Check its validity.
    if (upd_stmt == NULL || WN_operator(upd_stmt) != OPR_STID) return NULL;
    if (WN_st_idx(upd_stmt) != idx_var) return NULL;
    WN *upd_expr = WN_kid0(upd_stmt);
    if (upd_expr == NULL || WN_operator(upd_expr) != OPR_ADD) return NULL;
    if (WN_st_idx(WN_kid0(upd_expr)) == idx_var) {
        // TODO: Check if rhs is a loop invariant expression.
    } else if (WN_st_idx(WN_kid1(upd_expr)) == idx_var) {
        // TODO: check if lhs is a loop invariant expression.
        // Swap the two components.
        WN *tmp = WN_kid0(upd_expr);
        WN_kid0(upd_expr) = WN_kid1(upd_expr);
        WN_kid1(upd_expr) = tmp;
    } else {
        return NULL;
    }

    /* Construct the DO LOOP. */

    // index variable (IDNAME)
    WN *idx_var_wn = WN_CreateIdname(0, idx_var);
    // init statement
    WN_EXTRACT_FromBlock(parent, init_stmt);
    // loop body (without the last statement)
    WN_EXTRACT_FromBlock(loop_body, upd_stmt);

    WN *doloop = WN_CreateDO(idx_var_wn,
        init_stmt, check_stmt, upd_stmt, loop_body, NULL);

    /* Fix linkages, and deallocate the original loop. */

    WN_INSERT_BlockBefore(parent, loop, doloop);
    WN_EXTRACT_FromBlock(parent, loop);

    WN_kid0(loop) = WN_kid1(loop) = NULL;
    WN_DELETE_Tree(loop);

    return doloop;
}

WN*
get_loop_idx_range(struct doloop_info *li) {
    WN *range = NULL;

    if (WN_operator(li->step_val) == OPR_INTCONST) {
        if (WN_const_val(li->step_val) > 0) {
            /* start_idx = init
             * stride = step
             * end_idx = init + [(end-1-init)/step]*step
             */
            TYPE_ID mtype = WN_rtype(li->init_val);
            WN *end_idx = WN_Add(mtype,
                WN_COPY_Tree(li->init_val),
                WN_Mpy(mtype,
                    WN_Div(mtype,
                        WN_Sub(mtype,
                            WN_Sub(mtype,
                                WN_COPY_Tree(li->end_val),
                                WN_Intconst(mtype, 1)
                            ),
                            WN_COPY_Tree(li->init_val)
                        ),
                        WN_COPY_Tree(li->step_val)
                    ),
                    WN_COPY_Tree(li->step_val)
                )
            );

            range = WN_CreateTriplet(
                WN_COPY_Tree(li->init_val),
                WN_COPY_Tree(li->step_val),
                end_idx
            );
        } else {
            /* start_idx = init + [(end+1-init)/step]*step
             * stride = -step
             * end_idx = init
             */
            TYPE_ID mtype = WN_rtype(li->init_val);
            WN *start_idx = WN_Add(mtype,
                WN_COPY_Tree(li->init_val),
                WN_Mpy(mtype,
                    WN_Div(mtype,
                        WN_Sub(mtype,
                            WN_Add(mtype,
                                WN_COPY_Tree(li->end_val),
                                WN_Intconst(mtype, 1)
                            ),
                            WN_COPY_Tree(li->init_val)
                        ),
                        WN_COPY_Tree(li->step_val)
                    ),
                    WN_COPY_Tree(li->step_val)
                )
            );

            WN *stride = WN_Intconst(WN_rtype(li->step_val),
                    - WN_const_val(li->step_val));

            range = WN_CreateTriplet(
                    start_idx, stride, WN_COPY_Tree(li->init_val));
        }
    }

    return range;
}

void
clean_doloop_info(struct doloop_info *li) {
    if (li == NULL) return;

    if (li->init_val != NULL) WN_DELETE_Tree(li->init_val);
    if (li->end_val != NULL) WN_DELETE_Tree(li->end_val);
    if (li->step_val != NULL) WN_DELETE_Tree(li->step_val);
    if (li->tripcount != NULL) WN_DELETE_Tree(li->tripcount);
}

static void calc_trip_count(struct doloop_info *info)
{
    // init_val, step_val, end_val do not alias the nodes in the loop,
    // but we need a new copy here.
    WN *init_val = WN_COPY_Tree(info->init_val);
    WN *step_val = WN_COPY_Tree(info->step_val);
    WN *end_val = WN_COPY_Tree(info->end_val);

    if (info->flip_sign) {
        init_val = WN_Neg(WN_rtype(init_val), init_val);
        step_val = WN_Neg(WN_rtype(step_val), step_val);
        end_val = WN_Neg(WN_rtype(end_val), end_val);
    }

    // Check if it is an empty loop (end_val < init_val).
    WN *wn = WN_Sub(Integer_type,
        WN_COPY_Tree(end_val), WN_COPY_Tree(init_val));
    bool isEmpty = (WN_operator(wn) == OPR_INTCONST && WN_const_val(wn) < 0);
    WN_DELETE_Tree(wn);
    if (isEmpty) {
        WN_DELETE_Tree(init_val);
        WN_DELETE_Tree(step_val);
        WN_DELETE_Tree(end_val);
        info->tripcount = WN_Intconst(MTYPE_U4, 0);
        return;
    }

    // Check if it is an infinite loop.
    if (WN_operator(step_val) == OPR_INTCONST
        && WN_const_val(step_val) <= 0) {
        WN_DELETE_Tree(init_val);
        WN_DELETE_Tree(step_val);
        WN_DELETE_Tree(end_val);
        info->tripcount = NULL;
        return;
    }

    // ceil( (end-init)/step )
    WN *tmp = WN_Sub(Integer_type, end_val, init_val);
    // This function does not consume `tmp' nor `step_val'.
    info->tripcount = HCWN_Ceil(tmp, step_val, NULL);

    WN_DELETE_Tree(tmp);
    WN_DELETE_Tree(step_val);
}

inline OPERATOR
reverse_comp(OPERATOR opr) {
    switch (opr) {
        case OPR_GE: return OPR_LE;
        case OPR_GT: return OPR_LT;
        case OPR_LE: return OPR_GE;
        case OPR_LT: return OPR_GT;
        case OPR_EQ:
        case OPR_NE: return opr;
        default:     return OPERATOR_UNKNOWN;
    }
}

inline void
swap_kids(WN *wn, int kidx1, int kidx2) {
    WN *tmp_wn = WN_kid(wn, kidx1);
    WN_kid(wn, kidx1) = WN_kid(wn, kidx2);
    WN_kid(wn, kidx2) = tmp_wn;
}

bool normalize_loop(WN *doloop, struct doloop_info *info)
{
    assert(doloop != NULL && WN_opcode(doloop) == OPC_DO_LOOP);
    assert(info != NULL);

    /* Gather information and check for legality of normalization. */

    WN *wn = NULL;

    // Get the induction variable.
    ST_IDX idxv_st_idx = WN_st_idx(WN_kid0(doloop));

    // init value
    info->init_val = WN_COPY_Tree(WN_kid0(WN_kid1(doloop)));

    // step value
    wn = WN_kid0(WN_kid(doloop,3));
    bool step_conforming = (check_ldid(WN_kid0(wn), idxv_st_idx));
    info->step_val = WN_COPY_Tree(
        (step_conforming ? WN_kid1(wn) : WN_kid0(wn)));

    // ending value
    wn = WN_kid2(doloop);
    OPERATOR opr = WN_operator(wn);
    bool check_conforming = (check_ldid(WN_kid0(wn), idxv_st_idx));
    if (check_conforming) {
        wn = WN_kid1(wn);
    } else {
        // Reverse the comparison operator.
        opr = reverse_comp(opr);
        wn = WN_kid0(wn);
    }
    info->end_val = WN_COPY_Tree(wn);
    TYPE_ID end_val_mtype = WN_rtype(wn);
    OPERATOR new_opr = OPERATOR_UNKNOWN;
    switch (opr) {
        case OPR_GE:
            new_opr = OPR_GT;
            info->end_val = WN_Sub(end_val_mtype,
                info->end_val, WN_Intconst(end_val_mtype, 1));
        case OPR_GT:
            info->flip_sign = true;
            break;
        case OPR_LE:
            new_opr = OPR_LT;
            info->end_val = WN_Add(end_val_mtype,
                info->end_val, WN_Intconst(end_val_mtype, 1));
        case OPR_LT:
            info->flip_sign = false;
            break;
        default: abort();
    }

    // Compute the trip count.
    calc_trip_count(info);

    /* Modify the DO_LOOP. */

    // check statement
    wn = WN_kid2(doloop);
    if (!check_conforming) {
        WN_set_operator(wn, opr);
        swap_kids(wn, 0, 1);
    }
    if (new_opr != OPERATOR_UNKNOWN) {
        WN_set_operator(wn, new_opr);
        WN_DELETE_Tree(WN_kid1(wn));
        WN_kid1(wn) = WN_COPY_Tree(info->end_val);
    }

    // step expression
    if (!step_conforming) {
        wn = WN_kid0(WN_kid(doloop,3));
        swap_kids(wn, 0, 1);
    }

    return true;
}

ST_IDX
declare_function_va(const char *st_name, const char *ty_name,
        TY_IDX ret_ty, int nparams, ...) {
    // Create a parameter array.
    TY_IDX params[nparams];
    va_list param_args;

    va_start(param_args, nparams);
    for (int i = 0; i < nparams; ++i) {
        params[i] = va_arg(param_args, TY_IDX);
    }
    va_end(param_args);

    return declare_function(st_name, ty_name, ret_ty, nparams, params);
}

ST_IDX
declare_function_va(STR_IDX st_name, const char *ty_name,
        TY_IDX ret_ty, int nparams, ...) {
    // Create a parameter array.
    TY_IDX params[nparams];
    va_list param_args;

    va_start(param_args, nparams);
    for (int i = 0; i < nparams; ++i) {
        params[i] = va_arg(param_args, TY_IDX);
    }
    va_end(param_args);

    return declare_function(st_name, ty_name, ret_ty, nparams, params);
}

ST_IDX
declare_function(const char *st_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params) {
    return declare_function(Save_Str(st_name), ret_ty, nparams, params);
}

ST_IDX
declare_function(const char *st_name, const char *ty_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params) {
    // Get/add the symbol name from/to the string table.
    STR_IDX st_str_idx = Save_Str(st_name);

    return declare_function(st_str_idx, ty_name, ret_ty, nparams, params);
}

#define FUNC_TY_NAME_SUFFIX ".ty"

ST_IDX
declare_function(STR_IDX st_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params) {
    char *st_name_str = &Str_Table[st_name];

    // Construct a type name, <st_name>.ty.
    int size = strlen(st_name_str) + strlen(FUNC_TY_NAME_SUFFIX) + 1;
    char ty_name_str[size];     // no need for dynamic allocation
    strcpy(ty_name_str, st_name_str);
    strcat(ty_name_str, FUNC_TY_NAME_SUFFIX);

    return declare_function(st_name, ty_name_str, ret_ty, nparams, params);
}

void
declare_kernel(ST_IDX func_st_idx,
        TY_IDX ret_ty, int nparams, TY_IDX *params) {
    declare_function(func_st_idx, ret_ty, nparams, params);

    // Identify the function as a CUDA kernel.
    Set_PU_is_kernel(Pu_Table[ST_pu(St_Table[func_st_idx])]);

    // Put the identification in the symbol as well.
    set_st_attr_is_kernel(func_st_idx);

    // The kernel does not need a prototype.
    // Clear_TY_has_prototype(ST_pu_type(func_st_idx));

    // How to remove the extern declaration?
}

ST_IDX
declare_function(ST_IDX func_st_idx,
        TY_IDX ret_ty, int nparams, TY_IDX *params) {
    char *func_str = ST_name(func_st_idx);

    // Construct a type name, <st_name>.ty.
    int size = strlen(func_str) + strlen(FUNC_TY_NAME_SUFFIX) + 1;
    char ty_str[size];     // no need for dynamic allocation
    strcpy(ty_str, func_str);
    strcat(ty_str, FUNC_TY_NAME_SUFFIX);

    return declare_function(func_st_idx, ty_str, ret_ty, nparams, params);
}

ST_IDX
declare_function(STR_IDX func_str_idx, const char *ty_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params) {
    // Search for an ST with the given name in the global SYMTAB.
    ST_IDX func_st_idx = lookup_symbol(func_str_idx, GLOBAL_SYMTAB);
    if (func_st_idx == ST_IDX_ZERO) {
        // Otherwise, create a dummy global symbol.
        ST *func_st = New_ST(GLOBAL_SYMTAB);
        Set_ST_name_idx(func_st, func_str_idx);
        func_st_idx = ST_st_idx(func_st);
    }

    return declare_function(func_st_idx, ty_name, ret_ty, nparams, params);
}

ST_IDX declare_function(ST_IDX func_st_idx, const char *ty_name,
        TY_IDX ret_ty, int nparams, TY_IDX *params)
{
    // Create a new function prototype.
    TY_IDX func_ty_idx = new_func_type(ty_name, ret_ty, nparams, params);

    // Make a PU.
    PU_IDX pu_idx;
    PU &pu = New_PU(pu_idx);
    PU_Init(pu, func_ty_idx, CURRENT_SYMTAB+1);
    // Do not inline this function.
    PU_no_inline(pu);

    // IMPORTANT: synch with the auxilliary PU table.
    UINT32 aux_idx;
    AUX_PU& aux_pu = Aux_Pu_Table.New_entry(aux_idx);
    aux_pu.construct();
    Is_True(pu_idx == aux_idx, (""));

    // Make a function symbol.
    // NOTE the storage type is TEXT not EXTERN.
    ST *func_st = ST_ptr(func_st_idx);
    ST_Init(func_st, ST_name_idx(func_st),
        CLASS_FUNC, SCLASS_TEXT, EXPORT_PREEMPTIBLE, pu_idx);

    return func_st_idx;
}

TY_IDX new_func_type(const char *ty_name,
        TY_IDX ret_ty, INT nparams, TY_IDX *params)
{
    // Create a new function prototype.
    TY_IDX func_ty_idx;
    TY &func_ty = New_TY(func_ty_idx);
    TY_Init(func_ty, 0, KIND_FUNCTION, MTYPE_UNKNOWN, Save_Str(ty_name));
    Set_TY_has_prototype(func_ty_idx);
    Set_TY_align(func_ty_idx,
        TY_align(Make_Pointer_Type(MTYPE_To_TY(MTYPE_V))));

    // Set the return type.
    TYLIST_IDX tyl_idx;
    New_TYLIST(tyl_idx);
    Tylist_Table[tyl_idx] = ret_ty;
    Set_TY_tylist(func_ty, tyl_idx);

    // Set the parameters' types.
    for (int i = 0; i < nparams; ++i) {
        New_TYLIST(tyl_idx);
        Tylist_Table[tyl_idx] = params[i];
    }

    // Set the dummy type at the end.
    New_TYLIST(tyl_idx);
    Tylist_Table[tyl_idx] = TY_IDX_ZERO;

    return func_ty_idx;
}

static ST_IDX
new_var_st(STR_IDX str_idx, TY_IDX ty, SYMTAB_IDX level,
        ST_SCLASS st_sclass, ST_EXPORT st_export) {
    // Create a new symbol.
    ST *st = New_ST(level);
    ST_Init(st, str_idx, CLASS_VAR, st_sclass, st_export, ty);

    return ST_st_idx(st);
}

static ST_IDX
new_var_st(const char *name, TY_IDX ty, SYMTAB_IDX level,
        ST_SCLASS st_sclass, ST_EXPORT st_export) {
    return new_var_st(Save_Str(name), ty, level,
        st_sclass, st_export);
}

ST_IDX new_local_var(STR_IDX str_idx, TY_IDX ty)
{
    return new_var_st(str_idx, ty, CURRENT_SYMTAB, SCLASS_AUTO,
        // is this correct?
        EXPORT_LOCAL_INTERNAL);
}

ST_IDX new_local_var(const char *name, TY_IDX ty)
{
    return new_var_st(name, ty, CURRENT_SYMTAB, SCLASS_AUTO,
        // is this correct?
        EXPORT_LOCAL_INTERNAL);
}

ST_IDX new_formal_var(STR_IDX str_idx, TY_IDX ty)
{
    return new_var_st(str_idx, ty, CURRENT_SYMTAB, SCLASS_FORMAL,
            // is this correct?
            EXPORT_LOCAL_INTERNAL);
}

ST_IDX new_formal_var(const char *name, TY_IDX ty)
{
    return new_var_st(name, ty, CURRENT_SYMTAB, SCLASS_FORMAL,
            // is this correct?
            EXPORT_LOCAL_INTERNAL);
}

ST_IDX new_extern_var(STR_IDX str_idx, TY_IDX ty_idx)
{
    return new_var_st(str_idx, ty_idx, GLOBAL_SYMTAB, SCLASS_EXTERN,
            // is this correct?
            EXPORT_PREEMPTIBLE);
}

ST_IDX new_extern_var(const char *name, TY_IDX ty)
{
    return new_var_st(name, ty, GLOBAL_SYMTAB, SCLASS_EXTERN,
            // is this correct?
            EXPORT_PREEMPTIBLE);
}

ST_IDX new_global_var(STR_IDX str_idx, TY_IDX ty_idx)
{
    return new_var_st(str_idx, ty_idx, GLOBAL_SYMTAB, SCLASS_COMMON,
            EXPORT_PREEMPTIBLE);
}

ST_IDX new_global_var(const char *name, TY_IDX ty_idx)
{
    return new_var_st(name, ty_idx, GLOBAL_SYMTAB, SCLASS_COMMON,
            EXPORT_PREEMPTIBLE);
}

ST_IDX
lookup_symbol(const char *st_name, UINT8 st_level) {
    ST_TAB *sttab = Scope_tab[st_level].st_tab;
    int size = sttab->Size();

    for (int i = 0; i < size; ++i) {
        ST &st = sttab->Entry(i);
        if (strcmp(st_name, ST_name(st)) == 0) return ST_st_idx(st);
    }

    return ST_IDX_ZERO;
}

ST_IDX
lookup_symbol(STR_IDX st_name, UINT8 st_level) {
    ST_TAB *sttab = Scope_tab[st_level].st_tab;
    int size = sttab->Size();

    for (int i = 0; i < size; ++i) {
        ST &st = sttab->Entry(i);
        if (st_name == ST_name_idx(st)) return ST_st_idx(st);
    }

    return ST_IDX_ZERO;
}

ST_IDX
lookup_function(const char *fname) {
    STR_IDX str_idx = Get_Str(fname);
    if (str_idx == STR_IDX_ZERO) return ST_IDX_ZERO;

    ST_IDX st_idx = lookup_symbol(str_idx, GLOBAL_SYMTAB);
    if (st_idx == ST_IDX_ZERO
        || ST_class(st_idx) != CLASS_FUNC) {
        return ST_IDX_ZERO;
    }

    return st_idx;
}

ST_IDX
lookup_localvar(STR_IDX name_str_idx) {
    if (name_str_idx == STR_IDX_ZERO) return ST_IDX_ZERO;

    ST_IDX st_idx = lookup_symbol(name_str_idx, CURRENT_SYMTAB);
    if (st_idx != ST_IDX_ZERO
        && ST_class(st_idx) == CLASS_VAR) return st_idx;

    return ST_IDX_ZERO;
}

/**
 * efficiency improvement
 */
ST*
lookup_localvar(const char *vname) {
    ST_TAB *local_sttab = Scope_tab[CURRENT_SYMTAB].st_tab;
    int size = local_sttab->Size();
    ST *st = NULL;

    for (int i = 0; i < size; ++i) {
        st = &local_sttab->Entry(i);
        if (ST_class(st) == CLASS_VAR &&
            strcmp(vname, ST_name(st)) == 0) return st;
    }

    return NULL;
}

ST_IDX
lookup_extern_var(const char *vname) {
    STR_IDX str_idx = Get_Str(vname);
    if (str_idx == STR_IDX_ZERO) return ST_IDX_ZERO;

    ST_IDX st_idx = lookup_symbol(str_idx, GLOBAL_SYMTAB);
    ST &st = St_Table[st_idx];
    if (ST_class(st) == CLASS_VAR
        && ST_sclass(st) == SCLASS_EXTERN) {
        return st_idx;
    }

    return ST_IDX_ZERO;
}

TY_IDX
lookup_type(const char *tname) {
    STR_IDX str_idx = Get_Str(tname);
    if (str_idx == STR_IDX_ZERO) return TY_IDX_ZERO;

    int size = Ty_tab.Size();
    TY *ty = NULL;

    for (int i = 0; i < size; ++i) {
        ty = &Ty_tab.Entry(i);
        if (str_idx == TY_name_idx(*ty)) {
            return make_TY_IDX(i);
        }
    }

    return TY_IDX_ZERO;
}

int
replace_symbol(WN *wn, ST_IDX from_st_idx, ST_IDX to_st_idx) {
    OPERATOR opr = WN_operator(wn);
    int count = 0;

    // printf("processing %s\n", OPERATOR_name(opr));

    // Check the st_idx field (only for nodes that have it).
    if (OPERATOR_has_sym(opr) && WN_st_idx(wn) == from_st_idx) {
        WN_st_idx(wn) = to_st_idx;
        count = 1;
    }

    // Handle composite nodes.
    if (opr == OPR_BLOCK) {
        // Process all nodes inside.
        WN *node = WN_first(wn);
        while (node != NULL) {
            count += replace_symbol(node, from_st_idx, to_st_idx);
            node = WN_next(node);
        }
    } else {
        // Process all kids.
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            WN *kid = WN_kid(wn,i);
            assert(kid != NULL);
            count += replace_symbol(kid, from_st_idx, to_st_idx);
        }
    }

    return count;
}

ST_IDX
ST_copy(ST_IDX st_idx, SYMTAB_IDX dst_level) {
    SYMTAB_IDX src_level = ST_IDX_level(st_idx);

    // Do nothing if the symbol is defined in the current scope.
    assert(src_level != CURRENT_SYMTAB);

    // For now, we would not worry about base address.
    assert(ST_base_idx(ST_ptr(st_idx)) == st_idx);

    // Create a new symbol in the current scope.
    ST *dst_st = New_ST();
    ST_IDX dst_st_idx = ST_st_idx(dst_st);

    // Modify the symbol's level to dst_level.
    dst_st_idx = make_ST_IDX(ST_IDX_index(dst_st_idx), dst_level);

    // Copy from the original symbol.
    memcpy(dst_st, ST_ptr(st_idx), sizeof(ST));
    Set_ST_st_idx(dst_st, dst_st_idx);
    Set_ST_base_idx(dst_st, dst_st_idx);

    // NOTE: the created symbol is not valid because its
    // ST_IDX does not point to its ST object; this is intended.

    return dst_st_idx;
}

void
fix_lexical_level(SYMTAB_IDX level) {
    SCOPE &scope = Scope_tab[CURRENT_SYMTAB];

    // Fix the ST_TAB.
    ST_TAB *st_tab = scope.st_tab;
    UINT32 size = st_tab->Size();
    ST *st = NULL;
    for (UINT32 i = 0; i < size; ++i) {
        st = &st_tab->Entry(i);
        assert(i == ST_index(st));
        Set_ST_st_idx(st, make_ST_IDX(i, level));
    }

    // Fix the INITO_TAB.
    INITO_TAB *inito_tab = scope.inito_tab;
    size = inito_tab->Size();
    INITO *inito = NULL;
    for (UINT32 i = 0; i < size; ++i) {
        inito = &inito_tab->Entry(i);
        inito->st_idx = make_ST_IDX(ST_IDX_index(inito->st_idx), level);
    }

    // Fix the ST_ATTR_TAB.
    ST_ATTR_TAB *st_attr_tab = scope.st_attr_tab;
    size = st_attr_tab->Size();
    ST_ATTR *st_attr = NULL;
    for (UINT32 i = 0; i < size; ++i) {
        st_attr = &st_attr_tab->Entry(i);
        st_attr->st_idx = make_ST_IDX(ST_IDX_index(st_attr->st_idx), level);
    }
}

/* The key is the existing symbol index; the value is either the copied symbol
 * index if +ive, or the index in the export list otherwise.
 */
typedef std::map<ST_IDX, int> ST_IDX_PAIR_MAP;

static void
transfer_symbols_walker(WN *wn, SYMTAB_IDX dst_level,
        ST_IDX_PAIR_MAP& st_idx_done, ST_IDX *new_params) {
    OPERATOR opr = WN_operator(wn);

    // Check the st_idx field (only for nodes that have it).
    if (OPERATOR_has_sym(opr)) {
        ST_IDX st_idx = WN_st_idx(wn);
        ST_IDX new_st_idx = ST_IDX_ZERO;

        ST_IDX_PAIR_MAP::iterator it = st_idx_done.find(st_idx);
        if (it != st_idx_done.end()) {
            // Found it in the map.
            if (it->second <= 0) {
                // Make a copy of the symbol and save it in 'new_params'.
                new_st_idx = ST_copy(st_idx, dst_level);
                new_params[-it->second] = new_st_idx;
                it->second = new_st_idx;
            } else {
                new_st_idx = it->second;
            }
        } else {
            /* It is not in the map, then it must be a local symbol.
             * However, some global symbols (that are not parameters) can be
             * referenced, like blockIdx, threadIdx, constants.
             * For now, we do not do any checks on this. */
            if (ST_IDX_level(st_idx) > GLOBAL_SYMTAB) {
                // Add it to the map.
                new_st_idx = ST_copy(st_idx, dst_level);
                st_idx_done[st_idx] = new_st_idx;
            }
        }

        // Replace the symbol with the new one.
        if (new_st_idx != ST_IDX_ZERO) {
            WN_st_idx(wn) = new_st_idx;

            // Also transfer the symbol's attributes.
            ST_ATTR *from_st_attr = find_st_attr(st_idx, false);
            if (from_st_attr != NULL) {
                ST_ATTR_IDX to_st_attr_idx;
                ST_ATTR &to_st_attr =
                    New_ST_ATTR(CURRENT_SYMTAB, to_st_attr_idx);
                ST_ATTR_Init(to_st_attr, new_st_idx,
                    ST_ATTR_kind(*from_st_attr), from_st_attr->Get_flags());
            }
        }

        // printf("Replace symbol %u with %u\n", st_idx, copy_st_idx);
    }

    // Handle composite nodes.
    if (opr == OPR_BLOCK) {
        // Process all nodes inside.
        WN *node = WN_first(wn);
        while (node != NULL) {
            transfer_symbols_walker(node,
                dst_level, st_idx_done, new_params);
            node = WN_next(node);
        }
    } else {
        // Process all kids.
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            WN *kid = WN_kid(wn,i);
            assert(kid != NULL);
            transfer_symbols_walker(kid, dst_level, st_idx_done, new_params);
        }
    }
}

void transfer_symbols(WN *wn, SYMTAB_IDX dst_level,
        int nparams, ST_IDX *params, ST_IDX *new_params)
{
    ST_IDX_PAIR_MAP st_idx_done;

    // Init the map with the list of exported symbols.
    if (nparams > 0) {
        assert(params != NULL && new_params != NULL);
        for (INT i = 0; i < nparams; ++i) {
            new_params[i] = ST_IDX_ZERO;
            st_idx_done[params[i]] = -i;
        }
    }

    // Walk through the WHIRL tree.
    transfer_symbols_walker(wn, dst_level, st_idx_done, new_params);

    // Check if all exported symbols are indeed referenced in the WN node.
    for (INT i = 0; i < nparams; ++i)
    {
        if (new_params[i] == ST_IDX_ZERO)
        {
            HC_dev_warn("Exported symbol %s (index %u) "
                    "is not referenced in the given WN node.",
                    ST_name(params[i]), i);
            new_params[i] = ST_copy(params[i], dst_level);
        }
    }
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

typedef std::set<LABEL_IDX> LABEL_LIST;

static void find_labels_walker(WN *wn, LABEL_LIST &ll)
{
    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_LABEL) ll.insert(WN_label_number(wn));

    if (opr == OPR_BLOCK)
    {
        // Walk over the child nodes.
        WN *node_wn = WN_first(wn);
        while (node_wn != NULL) {
            find_labels_walker(node_wn, ll);
            node_wn = WN_next(node_wn);
        }
    }
    else
    {
        // Walk over the kids.
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) find_labels_walker(WN_kid(wn,i), ll);
    }
}

/**
 * Return true if 'wn' references labels in 'll', ignoring those references
 * inside 'except_wn'.
 */
static bool find_jumpin_walker(WN *wn, WN *except_wn, LABEL_LIST &ll)
{
    if (wn == except_wn) return false;

    OPERATOR opr = WN_operator(wn);

    // Check if this node references a label.
    if (OPERATOR_has_label(opr) && opr != OPR_LABEL
            && ll.find(WN_label_number(wn)) != ll.end()) return true;
    if (OPERATOR_has_last_label(opr)
            && ll.find(WN_last_label(wn)) != ll.end()) return true;

    if (opr == OPR_BLOCK)
    {
        // Walk over the child nodes.
        WN *node_wn = WN_first(wn);
        while (node_wn != NULL) {
            if (find_jumpin_walker(node_wn, except_wn, ll)) return true;
            node_wn = WN_next(node_wn);
        }
    }
    else
    {
        // Walk over the kids.
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            if (find_jumpin_walker(WN_kid(wn,i), except_wn, ll)) return true;
        }
    }

    return false;
}

bool verify_region_labels(WN *region_wn, WN *parent_func_wn)
{
    assert(region_wn != NULL && WN_operator(region_wn) == OPR_REGION);
    assert(parent_func_wn != NULL
            && WN_operator(parent_func_wn) == OPR_FUNC_ENTRY);

    /* First, make sure that there is no "jump-out".
     * This means that these is exactly one REGION_EXIT and its label is
     * right after the region.
     */
    WN *exit_wn = WN_first(WN_region_exits(region_wn));
    if (exit_wn == NULL || WN_next(exit_wn) != NULL) return false;
    assert(WN_operator(exit_wn) == OPR_REGION_EXIT);
    WN *exit_lbl_wn = WN_next(region_wn);
    if (exit_lbl_wn == NULL
            || WN_operator(exit_lbl_wn) != OPR_LABEL) return false;
    if (WN_label_number(exit_lbl_wn) != WN_label_number(exit_wn)) return false;

    /* Second, make sure that there is no "jump-in".
     * This means that no LABEL in the region is referenced outside.
     */
    LABEL_LIST ll;

    // Find all labels inside the region.
    find_labels_walker(WN_region_body(region_wn), ll);

    // Check all jumps outside the region.
    return ! find_jumpin_walker(WN_func_body(parent_func_wn), region_wn, ll);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

typedef std::map<LABEL_IDX, LABEL_IDX> LABEL_REPLACEMENT_MAP;

/**
 * For each LABEL in 'wn', replace the label index with a new one in the
 * current symbol table.
 */
static void create_new_labels_walker(WN *wn, SYMTAB_IDX from_level,
        LABEL_REPLACEMENT_MAP &lrm)
{
    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_LABEL)
    {
        LABEL_IDX old_label_idx = WN_label_number(wn);
        LABEL &old_label =
            Scope_tab[from_level].label_tab->Entry(old_label_idx);

        // Recreate a label in the current scope.
        LABEL_IDX new_label_idx;
        LABEL &new_label = New_LABEL(CURRENT_SYMTAB, new_label_idx);
        new_label = old_label;

        // Replace the old index with the new one.
        WN_label_number(wn) = new_label_idx;

        // Store the mapping for later use.
        lrm[old_label_idx] = new_label_idx;

        printf("LABEL: %u => %u\n", old_label_idx, new_label_idx);
    }

    if (opr == OPR_BLOCK)
    {
        // Process all nodes inside.
        WN *node_wn = WN_first(wn);
        while (node_wn != NULL) {
            create_new_labels_walker(node_wn, from_level, lrm);
            node_wn = WN_next(node_wn);
        }
    }
    else
    {
        // Process all kids.
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            WN *kid = WN_kid(wn,i);
            assert(kid != NULL);
            create_new_labels_walker(kid, from_level, lrm);
        }
    }
}

/**
 * For each label reference in 'wn', replace it with the new label index
 * if it appears in the map; otherwise, turn it into a RETURN node.
 *
 * Return the new node to replace 'wn' if necessary.
 */
static WN* replace_label_refs_walker(WN *wn, LABEL_REPLACEMENT_MAP &lrm)
{
    OPERATOR opr = WN_operator(wn);

    // Check if this node references a label.
    if (OPERATOR_has_label(opr) && opr != OPR_LABEL)
    {
        LABEL_IDX lidx = WN_label_number(wn);
        LABEL_REPLACEMENT_MAP::iterator it = lrm.find(lidx);
        if (it != lrm.end()) {
            WN_label_number(wn) = it->second;
            return NULL;
        }
        
        // This node must be a REGION_EXIT, so turn it into a RETURN node.
        assert(opr == OPR_REGION_EXIT);
        return WN_CreateReturn();
    }

    if (OPERATOR_has_last_label(opr))
    {
        LABEL_IDX lidx = WN_last_label(wn);
        LABEL_REPLACEMENT_MAP::iterator it = lrm.find(lidx);
        assert(it != lrm.end());
        WN_label_number(wn) = it->second;
        return NULL;
    }

    /* Handle composite nodes. */
    if (opr == OPR_BLOCK)
    {
        // Walk over the child nodes.
        WN *node_wn = WN_first(wn);
        while (node_wn != NULL) {
            WN *new_node_wn = replace_label_refs_walker(node_wn, lrm);
            if (new_node_wn != NULL) {
                // Replace 'node_wn' with 'new_node_wn'.
                WN_INSERT_BlockBefore(wn, node_wn, new_node_wn);
                WN_DELETE_FromBlock(wn, node_wn);
                node_wn = WN_next(new_node_wn);
            } else {
                node_wn = WN_next(node_wn);
            }
        }
    }
    else
    {
        // Walk over the kids.
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            WN *kid_wn = WN_kid(wn,i);
            assert(kid_wn != NULL);
            WN *new_kid_wn = replace_label_refs_walker(kid_wn, lrm);
            if (new_kid_wn != NULL) {
                WN_DELETE_Tree(kid_wn);
                WN_kid(wn,i) = new_kid_wn;
            }
        }
    }

    return NULL;
}

void transfer_labels(WN *region_wn, SYMTAB_IDX from_level)
{
    assert(region_wn != NULL && WN_operator(region_wn) == OPR_REGION);

    WN *body_blk_wn = WN_region_body(region_wn);

    /* Collect all labels in the region and give them new indices in the
     * current symbol table.
     */
    LABEL_REPLACEMENT_MAP lrm;
    create_new_labels_walker(body_blk_wn, from_level, lrm);

    /* Replace references to the old labels inside the region.
     * We do not need to worry about references outside the region because
     * they will never reference these labels.
     */
    replace_label_refs_walker(body_blk_wn, lrm);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

UINT16
num_array_dims(TY_IDX arr_ty_idx) {
    UINT16 dimen = 0;
    TY_IDX ty_idx = arr_ty_idx;

    while (TY_kind(ty_idx) == KIND_ARRAY) {
        ARB_IDX arb_idx = Ty_Table[ty_idx].Arb();

        // accumulate the dimensionality.
        dimen += ARB_dimension(ARB_HANDLE(arb_idx));

        // move to the inner type.
        ty_idx = TY_etype(ty_idx);
    }

    return dimen;
}

TY_IDX
arr_elem_ty(TY_IDX ty_idx) {
    while (TY_kind(ty_idx) == KIND_ARRAY) ty_idx = TY_etype(ty_idx);
    return ty_idx;
}

TY_IDX analyze_arr_ty(TY_IDX arr_ty_idx, UINT *ndims)
{
    UINT16 dimen = 0;
    TY_IDX ty_idx = arr_ty_idx;

    while (TY_kind(ty_idx) == KIND_ARRAY) {
        ARB_IDX arb_idx = Ty_Table[ty_idx].Arb();
        // Accumulate the dimensionality.
        dimen += ARB_dimension(ARB_HANDLE(arb_idx));
        // Move to the inner type.
        ty_idx = TY_etype(ty_idx);
    }

    if (ndims != NULL) *ndims = dimen;    
    return ty_idx;
}

TY_IDX make_arr_type(STR_IDX name_str_idx,
        UINT16 ndims, UINT32 *dim_sz, TY_IDX elem_ty_idx)
{
    Is_True(ndims >= 1 && dim_sz != NULL, (""));

    // Determine the total array size.
    UINT arr_sz = TY_size(elem_ty_idx);
    for (UINT16 i = 0; i < ndims; ++i) arr_sz *= dim_sz[i];

    // Create the type.
    TY_IDX ty_idx;
    TY &ty = New_TY(ty_idx);
    TY_Init(ty, arr_sz, KIND_ARRAY, MTYPE_M, name_str_idx);

    // Create array bound info.
    for (UINT16 i = 0; i < ndims; ++i)
    {
        ARB_HANDLE ah = New_ARB();
        ARB_Init(ah, 0, dim_sz[i]-1, 1);
        Set_ARB_dimension(ah, ndims-i);
        if (i == 0)
        {
            Set_TY_arb(ty, ah);
            Set_ARB_flags(ah, ARB_flags(ah) | ARB_FIRST_DIMEN);
        }
        if (i == ndims-1) Set_ARB_flags(ah, ARB_flags(ah) | ARB_LAST_DIMEN);
    }

    // Set the element type.
    Set_TY_etype(ty, elem_ty_idx);

    // Is this correct?
    Set_TY_align(ty_idx, TY_align(elem_ty_idx));

    return ty_idx;
}

TY_IDX make_incomplete_arr_type(STR_IDX name_str_idx, TY_IDX elem_ty_idx)
{
    // Create the type, with array size zero.
    TY_IDX ty_idx;
    TY &ty = New_TY(ty_idx);
    TY_Init(ty, 0, KIND_ARRAY, MTYPE_M, name_str_idx);

    // Create array bound info.
    ARB_HANDLE ah = New_ARB();
    ARB_Init(ah, 0, ST_IDX_ZERO, 1);
    Set_TY_arb(ty, ah);
    Set_ARB_flags(ah, (ARB_flags(ah) & ~ARB_CONST_UBND)
            | ARB_FIRST_DIMEN | ARB_LAST_DIMEN);

    // Set the element type.
    Set_TY_etype(ty, elem_ty_idx);

    // Is this correct?
    Set_TY_align(ty_idx, TY_align(elem_ty_idx));

    return ty_idx;
}

bool
set_arr_dim_sz(TY_IDX arr_ty_idx, UINT dim, UINT dim_sz) {
    UINT16 ndims = 0;
    TY_IDX ty_idx = arr_ty_idx;

    while (TY_kind(ty_idx) == KIND_ARRAY) {
        ARB_IDX arb_idx = Ty_Table[ty_idx].Arb();
        UINT16 curr_ndims = ARB_dimension(ARB_HANDLE(arb_idx));

        if (dim < ndims + curr_ndims) {
            // Found the dimension we want to modify.
            ARB_HANDLE ah = ARB_HANDLE(arb_idx + dim - ndims);
            Set_ARB_flags(ah, ARB_flags(ah)
                | ARB_CONST_LBND | ARB_CONST_UBND | ARB_CONST_STRIDE);
            Set_ARB_lbnd_val(ah, 0);
            Set_ARB_stride_val(ah, 1);
            Set_ARB_ubnd_val(ah, dim_sz-1);

            return true;
        }

        // Accumulate dimensionality.
        ndims += curr_ndims;

        // move to the inner type.
        ty_idx = TY_etype(ty_idx);
    }

    return false;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * If `s1' or `s2' is NULL, it is treated as an empty string.
 * If both `s1' and 's2' are NULL, return STR_IDX_ZERO.
 */
static STR_IDX gen_var_str_idx(const char *s1, const char *s2,
        int suffix_num = -1)
{
    // If the client provides the suffix, we do check for uniqueness.
    if (suffix_num >= 0) return Save_Str2i(s1, s2, suffix_num, NULL);

    bool is_new = false;
    STR_IDX new_var = STR_IDX_ZERO;

    // Check if the string without suffix is unique.
    new_var = Save_Str2(s1, s2, &is_new);
    if (new_var != STR_IDX_ZERO && !is_new) {
        // Append the suffix starting from 0 and check if it is unique.
        suffix_num = 0;
        do {
            new_var = Save_Str2i(s1, s2, suffix_num++, &is_new);
        } while (new_var != STR_IDX_ZERO && !is_new);
    }

    return new_var;
}

static STR_IDX signed_loop_idx_prefix = STR_IDX_ZERO;
static STR_IDX unsigned_loop_idx_prefix = STR_IDX_ZERO;

STR_IDX gen_var_str(ST_IDX var_st_idx, const char *suffix)
{
    STR_IDX var_str_idx = (var_st_idx == ST_IDX_ZERO) ?
        STR_IDX_ZERO : ST_name_idx(St_Table[var_st_idx]);
    const char *var_str = (var_str_idx == STR_IDX_ZERO) ?
        NULL : &Str_Table[var_str_idx];

    return gen_var_str_idx(var_str, suffix);
}

STR_IDX gen_var_str(const char *prefix, ST_IDX var_st_idx)
{
    STR_IDX var_str_idx = (var_st_idx == ST_IDX_ZERO) ?
        STR_IDX_ZERO : ST_name_idx(St_Table[var_st_idx]);
    const char *var_str = (var_str_idx == STR_IDX_ZERO) ?
        NULL : &Str_Table[var_str_idx];

    return gen_var_str_idx(prefix, var_str);
}

/**
 * !!! This function does not guarantee that the variable being
 * generated is never used by user code.
 *
 * We may need to find a unique PREFIX that no other variables
 * starts with.
 */
ST_IDX
make_loop_idx(UINT32 nesting_level, bool is_unsigned) {
    STR_IDX prefix = STR_IDX_ZERO;

    if (is_unsigned) {
        if (unsigned_loop_idx_prefix == STR_IDX_ZERO) {
            unsigned_loop_idx_prefix = gen_var_str_idx(STR_IDX_ZERO, "ivu");
        }
        prefix = unsigned_loop_idx_prefix;
    } else {
        if (signed_loop_idx_prefix == STR_IDX_ZERO) {
            signed_loop_idx_prefix = gen_var_str_idx(STR_IDX_ZERO, "iv");
        }
        prefix = signed_loop_idx_prefix;
    }

    assert(prefix != STR_IDX_ZERO);
    STR_IDX str_idx = gen_var_str_idx(&Str_Table[prefix], "_", nesting_level);

    // Search for a symbol of this name.
    ST_IDX st_idx = lookup_localvar(str_idx);
    if (st_idx != ST_IDX_ZERO) return st_idx;

    return new_local_var(str_idx,
        MTYPE_To_TY(is_unsigned ? MTYPE_U4 : MTYPE_I4)
    );
}

WN*
make_empty_doloop(ST_IDX idxv_st_idx, WN *init_wn, WN *end_wn, WN *step_wn) {
    ST *idxv_st = ST_ptr(idxv_st_idx);
    TY_IDX idxv_ty_idx = ST_type(idxv_st);
    TYPE_ID idxv_mtype = TY_mtype(idxv_ty_idx);

    WN *init_stmt = WN_Stid(idxv_mtype, 0,
        idxv_st,
        idxv_ty_idx,
        WN_COPY_Tree(init_wn),  // must make a copy here
        0
    );

    WN *check_stmt = WN_LE(idxv_mtype,
        WN_Ldid(idxv_mtype, 0, idxv_st_idx, idxv_ty_idx),
        WN_COPY_Tree(end_wn)    // must make a copy here
    );

    WN *update_stmt = WN_Stid(idxv_mtype, 0,
        idxv_st,
        idxv_ty_idx,
        WN_Add(idxv_mtype,
            WN_Ldid(idxv_mtype, 0, idxv_st_idx, idxv_ty_idx),
            (step_wn == NULL ?
                WN_Intconst(idxv_mtype, 1) :
                WN_COPY_Tree(step_wn)
            )
        ),
        0
    );

    return WN_CreateDO(
        WN_CreateIdname(0, idxv_st_idx),
        init_stmt, check_stmt, update_stmt,
        WN_CreateBlock(),
        NULL
    );
}

static void
set_used_symbols(WN *wn) {
    if (wn == NULL) return;

    // Get the symbol if any.
    ST *st = WN_st(wn);
    if (st != NULL) {
        /* If the symbol has flag ST_IS_TEMP_VAR and has storage SCLASS_EXTERN,
         * which is invalid in the current WHIRL spec, then it is a CUDA symbol
         * that should not be declared in the generated code, so we do not
         * indicate that this symbol is referenced.
         */
        if (!ST_is_temp_var(st) || ST_sclass(st) != SCLASS_EXTERN) {
            Clear_ST_is_not_used(st);
        }
    }

    // Handle composite nodes.
    if (WN_operator(wn) == OPR_BLOCK) {
        WN *node = WN_first(wn);
        while (node != NULL) {
            set_used_symbols(node);
            node = WN_next(node);
        }
    } else {
        int nkids = WN_kid_count(wn);
        for (int i = 0; i < nkids; ++i) {
            set_used_symbols(WN_kid(wn,i));
        }
    }
}

static void
cleanup_pu_symbols(PU_Info *pu_tree) {
    PU_Info *pu = pu_tree, *child_pu = NULL;
    ST *st = NULL;
    UINT32 i;

    while (pu != NULL) {
        Current_PU_Info = pu;

        // Load the PU into memory.
        MEM_POOL_Push(MEM_pu_nz_pool_ptr);
        Read_Local_Info(MEM_pu_nz_pool_ptr, pu);

        // Set all local symbols to be unused.
        FOREACH_SYMBOL(CURRENT_SYMTAB, st, i) {
            Set_ST_is_not_used(st);
        }

        // Flag all symbols (global/local) being used.
        set_used_symbols(PU_Info_tree_ptr(pu));

        // Process the nested procedure, if any.
        if ((child_pu = PU_Info_child(pu)) != NULL) {
            cleanup_pu_symbols(child_pu);
        }

        // Output the PU.
        Write_PU_Info(pu);
        Free_Local_Info(pu);
        MEM_POOL_Pop(MEM_pu_nz_pool_ptr);

        pu = PU_Info_next(pu);
    }
}

void
cleanup_symbols(PU_Info *pu_root) {
    /* First, set all global symbols to be unused. */

    ST *st = NULL;
    UINT32 i;
    FOREACH_SYMBOL(GLOBAL_SYMTAB, st, i) {
        Set_ST_is_not_used(st);
    }

    /* Second, walk through all PU's to set symbols that are used. */

    cleanup_pu_symbols(pu_root);
}

WN*
find_func_body(const PU_Info *pu_tree, ST_IDX func_st_idx) {
    const PU_Info *pi = pu_tree, *child_pi = NULL;
    WN *func_wn = NULL;

    while (pi != NULL) {
        if (PU_Info_proc_sym(pi) == func_st_idx) {
            func_wn = PU_Info_tree_ptr(pi);
            assert(WN_operator(func_wn) == OPR_FUNC_ENTRY
                && WN_st_idx(func_wn) == func_st_idx);
            return func_wn;
        }

        // Check the child PUs.
        if ((child_pi = PU_Info_child(pi)) != NULL) {
            func_wn = find_func_body(child_pi, func_st_idx);
            if (func_wn != NULL) return func_wn;
        }

        pi = PU_Info_next(pi);
    }

    return NULL;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

bool
replace_wn_kid(WN *parent, WN *wn, WN *new_wn) {
    INT nkids = WN_kid_count(parent);
    bool found = false;

    for (INT i = 0; i < nkids; ++i) {
        if (WN_kid(parent,i) == wn) {
            assert(!found);
            found = true;
            WN_kid(parent,i) = new_wn;
        }
    }

    return found;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

bool
HCWN_are_ldids_equal(WN *ldid1, WN *ldid2) {
    assert(ldid1 != NULL && WN_operator(ldid1) == OPR_LDID);
    assert(ldid2 != NULL && WN_operator(ldid2) == OPR_LDID);

    return (WN_st_idx(ldid1) == WN_st_idx(ldid2))
        && (WN_load_offset(ldid1) == WN_load_offset(ldid2));
}

WN*
HCWN_SimplifyExp1(WN *wn, bool *simplified) {
    // Here, we may do more simplifications.

    WN *wn_simp = WN_SimplifyExp1(WN_opcode(wn), wn);
    if (wn_simp == NULL) {
        wn_simp = wn;
        if (simplified != NULL) *simplified = false;
    } else {
        if (simplified != NULL) *simplified = true;
    }

    return wn_simp;
}

WN* HCWN_Ceil(WN *wn1, WN *wn2, bool *perfect_div)
{
    Is_True(wn1 != NULL && wn2 != NULL, (""));

    OPERATOR opr1 = WN_operator(wn1), opr2 = WN_operator(wn2);
    Is_True(OPERATOR_is_expression(opr1), (""));
    Is_True(OPERATOR_is_expression(opr2), (""));

#if 0
    // DO NOT USE HCWN_SimplifyExp1, because it works on single-operand
    // expressions.
    WN *wn1_simp = WN_Simplify_Tree(WN_COPY_Tree(wn1));
    WN *wn2_simp = WN_Simplify_Tree(WN_COPY_Tree(wn2));
    opr1 = WN_operator(wn1);
    opr2 = WN_operator(wn2);
#endif

    WN* result = NULL;

    /* We could be very clever here. Here are the first few tricks:
     * - if w1 and w2 are constants, just compute the ceiling.
     * - if w1 == w2, return 1.
     */

    if (opr1 == OPR_INTCONST && opr2 == OPR_INTCONST)
    {
        INT64 wn1_val = WN_const_val(wn1);
        INT64 wn2_val = WN_const_val(wn2);
        if (perfect_div != NULL) *perfect_div = ((wn1_val % wn2_val) == 0);
        result = WN_Intconst(Integer_type, (wn1_val + wn2_val - 1) / wn2_val);
    }
    else if (opr1 == OPR_LDID && opr2 == OPR_LDID
            && HCWN_are_ldids_equal(wn1, wn2))
    {
        if (perfect_div != NULL) *perfect_div = true;
        result = WN_Intconst(Integer_type, 1);
    }
    else
    {
        if (perfect_div != NULL) *perfect_div = false;
        // (wn1 + wn2 - 1) / wn2
        result = WN_Div(Integer_type,
            WN_Sub(Integer_type,
                WN_Add(Integer_type, WN_COPY_Tree(wn1), WN_COPY_Tree(wn2)),
                WN_Intconst(Integer_type, 1)
            ),
            WN_COPY_Tree(wn2)
        );
    }

    return result;
}


/*****************************************************************************
 *
 * Calculate the size of the given array dimension:
 *      ubnd_val - lbnd_val + 1
 *
 ****************************************************************************/

WN* array_dim_size(const ARB_HANDLE &ah)
{
    INT64 lbnd_val = 0, ubnd_val = 0;
    ST_IDX lbnd_st = 0, ubnd_st = 0;
    WN *lbnd_wn = NULL, *ubnd_wn = NULL;
    WN *size_wn = NULL;

    if (ARB_const_lbnd(ah))
    {
        lbnd_val = ARB_lbnd_val(ah);
        if (ARB_const_ubnd(ah)) {
            // Both bounds are constant.
            ubnd_val = ARB_ubnd_val(ah);
            size_wn = WN_Intconst(Integer_type, ubnd_val - lbnd_val + 1);
        } else {
            // Only the lower bound is a constant.
            lbnd_wn = WN_Intconst(Integer_type, lbnd_val-1);
            ubnd_st = ARB_ubnd_var(ah);
            ubnd_wn = WN_Ldid(Integer_type, 0, ubnd_st, ST_type(ubnd_st));
            // Create a SUB expression.
            size_wn = WN_Sub(Integer_type, ubnd_wn, lbnd_wn);
        }
    }
    else
    {
        if (ARB_const_ubnd(ah)) {
            // Only the upper bound is a constant.
            ubnd_val = ARB_ubnd_val(ah);
            ubnd_wn = WN_Intconst(Integer_type, ubnd_val+1);
        } else {
            // Neither bound is a constant.
            ubnd_st = ARB_ubnd_var(ah);
            ubnd_wn = WN_Ldid(Integer_type, 0, ubnd_st, ST_type(ubnd_st));
            // Create UBOUND + 1.
            ubnd_wn = WN_Add(Integer_type,
                ubnd_wn, WN_Intconst(Integer_type, 1));
        }
        lbnd_st = ARB_lbnd_var(ah);
        lbnd_wn = WN_Ldid(Integer_type, 0, lbnd_st, ST_type(lbnd_st));
        // Create a SUB expression.
        size_wn = WN_Sub(Integer_type, ubnd_wn, lbnd_wn);
    }

    return size_wn;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

BOOL HCWN_contains_non_global_syms(WN *wn,
        DYN_ARRAY<ST_IDX> *exception_list)
{
    if (wn == NULL) return FALSE;

    OPERATOR opr = WN_operator(wn);
    if (OPERATOR_has_sym(opr))
    {
        ST_IDX st_idx = WN_st_idx(wn);

        // Is this symbol in the exception list?
        BOOL found = FALSE;
        for (INT i = 0; i < exception_list->Elements(); ++i) {
            if (st_idx == (*exception_list)[i]) {
                found = TRUE;
                break;
            }
        }

        if (!found && !Is_Global_Symbol(WN_st(wn))) return TRUE;
    }

    if (opr == OPR_BLOCK) {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            if (HCWN_contains_non_global_syms(kid_wn)) return TRUE;
            kid_wn = WN_next(kid_wn);
        }
    } else {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            if (HCWN_contains_non_global_syms(WN_kid(wn,i))) return TRUE;
        }
    }

    return FALSE;
}

void HCWN_build_st_table(WN *wn, HASH_TABLE<ST_IDX, ST*> *st_table)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (OPERATOR_has_sym(opr))
    {
        ST_IDX st_idx = WN_st_idx(wn);
        if (!st_attr_is_cuda_runtime(st_idx))
        {
            // Keep the signature unique.
            st_table->Enter_If_Unique(st_idx, ST_ptr(st_idx));
        }
    }

    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            HCWN_build_st_table(kid_wn, st_table);
            kid_wn = WN_next(kid_wn);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            HCWN_build_st_table(WN_kid(wn,i), st_table);
        }
    }
}

void HCWN_replace_syms(WN *wn, HASH_TABLE<ST_IDX,ST_IDX> *new_formal_map,
        const HASH_TABLE<ST_IDX, ST*> *st_tbl)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (OPERATOR_has_sym(opr))
    {
        ST_IDX old_st_idx = WN_st_idx(wn);
        ST_IDX new_st_idx = new_formal_map->Find(old_st_idx);

        // Create a new formal symbol in the appropriate mode.
        if (new_st_idx == ST_IDX_ZERO)
        {
            ST *old_st = st_tbl->Find(old_st_idx);
            Is_True(old_st != NULL, (""));
            new_st_idx = new_formal_var(ST_name_idx(old_st), ST_type(old_st));
            new_formal_map->Enter(old_st_idx, new_st_idx);
        }

        WN_st_idx(wn) = new_st_idx;
    }

    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            HCWN_replace_syms(kid_wn, new_formal_map, st_tbl);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            HCWN_replace_syms(WN_kid(wn,i), new_formal_map, st_tbl);
        }
    }
}

void HCWN_replace_syms(WN *wn, const HASH_TABLE<ST_IDX,ST_IDX> *map)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (OPERATOR_has_sym(opr))
    {
        ST_IDX old_st_idx = WN_st_idx(wn);
        ST_IDX new_st_idx = map->Find(old_st_idx);
        if (new_st_idx != ST_IDX_ZERO) WN_st_idx(wn) = new_st_idx;
    }

    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            HCWN_replace_syms(kid_wn, map);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            HCWN_replace_syms(WN_kid(wn,i), map);
        }
    }
}

void HCWN_check_parentize(const WN *wn, WN_MAP map)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            Is_True(WN_MAP_Get(map,kid_wn) == wn, (""));
            HCWN_check_parentize(kid_wn, map);
            kid_wn = WN_next(kid_wn);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            Is_True(WN_MAP_Get(map,WN_kid(wn,i)) == wn, (""));
            HCWN_check_parentize(WN_kid(wn,i), map);
        }
    }
}

void HCWN_check_map_id(const WN *wn)
{
    if (wn == NULL) return;

    OPERATOR opr = WN_operator(wn);

    if (WN_map_id(wn) >= 0)
    {
        OPERATOR_MAPCAT category = OPERATOR_mapcat(opr);
        Is_True(WN_map_id(wn) <= Current_Map_Tab->_last_map_id[category], (""));
    }

    if (opr == OPR_BLOCK)
    {
        WN *kid_wn = WN_first(wn);
        while (kid_wn != NULL) {
            HCWN_check_map_id(kid_wn);
            kid_wn = WN_next(kid_wn);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i) {
            HCWN_check_map_id(WN_kid(wn,i));
        }
    }
}

BOOL HCTY_is_dyn_array(TY_IDX ty_idx)
{
    if (TY_kind(ty_idx) != KIND_ARRAY) return FALSE;

    UINT dimen = 0;
    do
    {
        ARB_HANDLE ah = ARB_HANDLE(Ty_Table[ty_idx].Arb());

        if (!ARB_const_stride(ah)) return TRUE;
        if (!ARB_const_lbnd(ah)) return TRUE;
        if (!ARB_const_ubnd(ah)) return TRUE;

        ++dimen;
        ty_idx = TY_etype(ty_idx);

    } while (TY_kind(ty_idx) == KIND_ARRAY);

    return FALSE;
}

/*** DAVID CODE END ***/
