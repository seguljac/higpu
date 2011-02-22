/*
 * Copyright 2003, 2004, 2005, 2006 PathScale, Inc.  All Rights Reserved.
 */

/*

  Copyright (C) 2000, 2001 Silicon Graphics, Inc.  All Rights Reserved.

  This program is free software; you can redistribute it and/or modify it
  under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it would be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

  Further, this software is distributed without any warranty that it is
  free of the rightful claim of any third person regarding infringement 
  or the like.  Any license provided herein, whether implied or 
  otherwise, applies only to this software file.  Patent licenses, if 
  any, provided herein do not apply to combinations of this program with 
  other software, or any other product whatsoever.  

  You should have received a copy of the GNU General Public License along
  with this program; if not, write the Free Software Foundation, Inc., 59
  Temple Place - Suite 330, Boston MA 02111-1307, USA.

  Contact information:  Silicon Graphics, Inc., 1600 Amphitheatre Pky,
  Mountain View, CA 94043, or:

  http://www.sgi.com

  For further information regarding this notice, see:

  http://oss.sgi.com/projects/GenInfo/NoticeExplan

*/



#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <elf.h>

#include "defs.h"
#include "mempool.h"            // MEM_POOL
#include "config_ipa.h"	        // IPA_Enable_* flags
#include "ipa_cg.h"             // IPA_NODE, IPA_Call_Graph
#include "ipaa.h"		        // IPAA_NODE_INFO 
#include "ipa_section_annot.h"  // array section classes
#include "ipa_section_main.h"   // Init_IPA_Print_Arrays
#include "ipa_section_prop.h"   // Trace_IPA_Sections
#include "ipa_reshape.h"        // RESHAPE
#include "reshape.h"            // RESHAPE
#include "ipa_cost.h" 		    // Execution cost 

// ==========================================================
// ==========================================================
// INTERPROCEDURAL ARRAY SECTION ANALYSIS FOR PARALLELIZATION
// ==========================================================
// ==========================================================

/** DAVID CODE BEGIN **/
#include "ipo_defs.h"

BOOL IPA_HC_enable_local_vars_in_section = FALSE;
/*** DAVID CODE END ***/

//=====================================================================
// return the position of the formal parameter from its summary symbol
//=====================================================================

static inline INT32 Formal_position(const SUMMARY_FORMAL *formal_array, 
        const SUMMARY_SYMBOL *formal_symbol)
{
    Is_True(formal_symbol->Is_formal(), ("Expected a formal symbol"));

    return formal_array[formal_symbol->Get_findex()].Get_position();
}


/*****************************************************************************
 *
 * Add an IVAR term to the caller linex.
 *
 ****************************************************************************/

static void Add_ivar_to_caller_linex(const IPA_NODE *caller,
        COEFF coeff, const IVAR *ivar, LINEX *linex)
{
    IP_FILE_HDR& file_hdr = caller->File_Header();
    SECTION_FILE_ANNOT *caller_file_annot =
        IP_FILE_HDR_section_annot(file_hdr);

    INT32 idx = caller_file_annot->Find_ivar(caller, *ivar);
    if (idx == -1) idx = caller_file_annot->Add_ivar(caller, *ivar);

    linex->Set_term(LTKIND_IV, coeff, idx, 0);
}

// forward declaration because of recursion  
static void
Add_value_to_caller_linex(const IPA_NODE* caller,
                          COEFF coeff,
                          const SUMMARY_VALUE* value,
                          LINEX* linex);

// --------------------------------------
// Add a SUMMARY_EXPR to the caller linex
// --------------------------------------
static void
Add_expr_to_caller_linex(const IPA_NODE* caller,
                         COEFF coeff,
                         const SUMMARY_EXPR* expr,
                         LINEX* linex)
{
  OPERATOR opr = OPCODE_operator(expr->Get_opcode());
  
  if (expr->Has_const_operand()) {
    // constant +|-|* value
    if (expr->Is_expr_value(0)) {
      Is_True(opr == OPR_ADD || opr == OPR_SUB || opr == OPR_MPY,
              ("Add_expr_to_caller_linex: expected +, -, or *"));
      const SUMMARY_VALUE* value = 
        IPA_get_value_array(caller) + expr->Get_node_index(expr->Get_kid());
      if (opr == OPR_ADD || opr == OPR_SUB) {
        linex->Set_term(LTKIND_CONST, 
                        (COEFF) expr->Get_const_value() * coeff,
                        CONST_DESC, 0);
      }
      if (opr == OPR_SUB) {
        coeff = -coeff;
      }
      else if (opr == OPR_MPY) {
        coeff *= expr->Get_const_value();
      }
      Add_value_to_caller_linex(caller, coeff, value, linex);
    }
    // constant +|- expression
    else {
      Is_True(expr->Is_expr_expr(0) && (opr == OPR_ADD || opr == OPR_SUB),
              ("Add_expr_to_caller_linex: expected const +|- expr"));
      const SUMMARY_EXPR* kid_expr = 
        IPA_get_expr_array(caller) + expr->Get_node_index(expr->Get_kid());
      linex->Set_term(LTKIND_CONST, 
                      (COEFF) expr->Get_const_value() * coeff,
                      CONST_DESC, 0);
      if (opr == OPR_SUB) {
        coeff = -coeff;
      }
      Add_expr_to_caller_linex(caller, coeff, kid_expr, linex);
    }
  }
  else {
    Is_True(opr == OPR_ADD || opr == OPR_SUB,
            ("Add_expr_to_caller_linex: expected + or -"));
    // expression +|- value
    for (INT i = 0; i < 2; ++i) {
      if (expr->Is_expr_value(i)) {
        const SUMMARY_VALUE* value = 
          IPA_get_value_array(caller) + expr->Get_node_index(i);
        Add_value_to_caller_linex(caller, coeff, value, linex);
      }
      else if (expr->Is_expr_expr(i)) {
        const SUMMARY_EXPR* kid_expr = 
          IPA_get_expr_array(caller) + expr->Get_node_index(i);
        Add_expr_to_caller_linex(caller, coeff, kid_expr, linex);
      }
      else {
        Is_True(0, 
                ("Add_expr_to_caller_linex: kid %d must be value or expr", i));
      }
      if (opr == OPR_SUB) {
        coeff = -coeff;
      }
    }
  }
}

// ---------------------------------------
// Add a SUMMARY_VALUE to the caller linex
// ---------------------------------------
static void
Add_value_to_caller_linex(const IPA_NODE* caller,
                          COEFF coeff,
                          const SUMMARY_VALUE* value,
                          LINEX* linex)
{
  IPA_CONST_TYPE value_kind = value->Get_const_type();
  
  if (value_kind == VALUE_INT_CONST) {
    linex->Set_term(LTKIND_CONST, 
                    (COEFF) value->Get_int_const_value() * coeff,
                    CONST_DESC, 0);
  }
  else if (value_kind == VALUE_CONST) {
    TCON_IDX tcon_idx = ST_tcon(ST_ptr(value->Get_const_st_idx()));
    linex->Set_term(LTKIND_CONST, 
                    (COEFF) Targ_To_Host(Tcon_Table[tcon_idx]) * coeff, 
                    CONST_DESC, 0);
  }
  else if (value_kind == VALUE_FORMAL) {
    const SUMMARY_FORMAL& formal = 
      IPA_get_formal_array(caller)[value->Get_formal_index()];
    const SUMMARY_SYMBOL& symbol = 
      IPA_get_symbol_array(caller)[formal.Get_symbol_index()];

    IVAR ivar(formal.Get_position(), 0, symbol.Get_btype());
    Add_ivar_to_caller_linex(caller, coeff, &ivar, linex);
  }
  else if (value_kind == VALUE_GLOBAL) {
    const SUMMARY_SYMBOL& symbol = 
      IPA_get_symbol_array(caller)[value->Get_global_index()];
    IVAR ivar(ST_ptr(symbol.St_idx()), 0, symbol.Get_btype());
    Add_ivar_to_caller_linex(caller, coeff, &ivar, linex);
  }
  else {
    Is_True(value_kind == VALUE_EXPR,
            ("Add_value_to_caller_linex: expected a linexable value"));
    SUMMARY_EXPR* expr = IPA_get_expr_array(caller) + value->Get_expr_index();
    Add_expr_to_caller_linex(caller, coeff, expr, linex);
  }
}


/*****************************************************************************
 *
 * Update caller linex with the actual argument passed at callsite.
 *
 ****************************************************************************/
  
static void Add_actual_to_caller_linex(const IPA_NODE* caller,
        COEFF coeff, const SUMMARY_ACTUAL* actual, LINEX* linex)
{
    if (actual->Get_symbol_index() != -1)
    {
        IPAA_NODE_INFO* modref_info = caller->Mod_Ref_Info();
        const SUMMARY_SYMBOL& symbol = 
            IPA_get_symbol_array(caller)[actual->Get_symbol_index()];

        if (symbol.Is_formal())
        {
            // pass-through formal
            UINT32 position = Formal_position(
                    IPA_get_formal_array(caller), &symbol);

            if (!modref_info->Is_formal_dmod_elmt(position)
                    && !modref_info->Is_formal_imod_elmt(position)) {
                IVAR ivar(position, 0, symbol.Get_btype());
                Add_ivar_to_caller_linex(caller, coeff, &ivar, linex);
                return;
            }
        }
        else if (ST_IDX_level(symbol.St_idx()) == GLOBAL_SYMTAB)
        {
            // global variable
            ST* st = ST_ptr(symbol.St_idx());
            while (ST_base(st) != st) st = ST_base(st);

            if (!modref_info->Is_def_elmt(ST_index(st))) {
                IVAR ivar(st, 0, symbol.Get_btype());
                Add_ivar_to_caller_linex(caller, coeff, &ivar, linex);
                return;
            }
        }
    }

    Is_True(actual->Get_value_index() != -1, 
            ("Add_actual_to_caller_linex: expected a linexable argument"));

    Add_value_to_caller_linex(caller, coeff,
            IPA_get_value_array(caller) + actual->Get_value_index(), linex);
}


/*****************************************************************************
 *
 * Map a callee term to caller variables and add it to the caller linex.
 *
 ****************************************************************************/

static void Map_term_to_caller(const IPA_NODE *caller, const IPA_NODE *callee,
        const SUMMARY_CALLSITE *call, TERM *callee_term, LINEX *caller_linex)
{
    switch (callee_term->Get_type())
    {
        case LTKIND_CONST:
            caller_linex->Set_term(callee_term);
            break;

        case LTKIND_IV:
        {
            INT32 size;
            const IVAR *ivar = IPA_get_ivar_array(callee, size)
                + callee_term->Get_desc();

            if (ivar->Is_Formal()) {
                const SUMMARY_ACTUAL *actual = IPA_get_actual_array(caller) 
                    + call->Get_actual_index() + ivar->Formal_Position();

                Add_actual_to_caller_linex(caller,
                        callee_term->Get_coeff(), actual, caller_linex);
            } else {
                // ivar is a global variable.
                Add_ivar_to_caller_linex(caller,
                        callee_term->Get_coeff(), ivar, caller_linex);
            }

            break;
        }

        default:
            break;
    }
}


/*****************************************************************************
 *
 * Map the linex terms to the caller variables.
 *
 ****************************************************************************/

static void Map_linex_to_caller(
        const IPA_NODE *caller, const IPA_NODE *callee,
        const SUMMARY_CALLSITE *callsite,
        LINEX *callee_linex, LINEX *caller_linex)
{
    INT j, last_idx = callee_linex->Num_terms();

    for (j = 0; j <= last_idx; ++j) {
        Map_term_to_caller(caller, callee, callsite, 
                callee_linex->Get_term(j), caller_linex);
    }
}


/*****************************************************************************
 *
 * Map a PROJECTED_NODE from the callee's to the caller's space.
 *
 ****************************************************************************/

static void Map_projected_node_to_caller(
        const IPA_NODE* caller, const IPA_NODE* callee,
        const SUMMARY_CALLSITE* callsite,
        PROJECTED_NODE* caller_pnode, PROJECTED_NODE* callee_pnode)
{
    MEM_POOL* pool = caller_pnode->Mem_Pool();

    caller_pnode->Set_flags(callee_pnode->Get_flags());

    // map the linex's upper bound
    if (!callee_pnode->Is_messy_ub()) {
        Map_linex_to_caller(caller, callee, callsite, 
                callee_pnode->Get_upper_linex(),
                caller_pnode->Get_upper_linex());
        caller_pnode->Get_upper_linex()->Simplify();
    }

    // map lower bound
    if (!callee_pnode->Is_messy_lb()) {
        Map_linex_to_caller(caller, callee, callsite, 
                callee_pnode->Get_lower_linex(),
                caller_pnode->Get_lower_linex());
        caller_pnode->Get_lower_linex()->Simplify();
    }

    // map step
    if (!callee_pnode->Is_messy_step()) {
        Map_linex_to_caller(caller, callee, callsite, 
                callee_pnode->Get_step_linex(),
                caller_pnode->Get_step_linex());
        caller_pnode->Get_step_linex()->Simplify();
    }

    // segment length and stride are always constant: make a straight copy
    if (callee_pnode->Get_segment_length_linex()) {
        LINEX* segment_length = CXX_NEW(LINEX(pool), pool);
        callee_pnode->Get_segment_length_linex()->Copy(segment_length);
        caller_pnode->Set_segment_length_linex(segment_length);
    } 

    if (callee_pnode->Get_segment_stride_linex()) {
        LINEX* segment_stride = CXX_NEW(LINEX(pool), pool);
        callee_pnode->Get_segment_stride_linex()->Copy(segment_stride);
        caller_pnode->Set_segment_stride_linex(segment_stride);
    } 
}
  
// forward declaration because of recursion
static BOOL 
Is_caller_value_linexable(const IPA_NODE* caller, 
                          const SUMMARY_VALUE* value);

// ----------------------------------------------------
// Check if an expression can be represented as a LINEX 
// of constants, globals, and caller formals
// ----------------------------------------------------
static BOOL 
Is_caller_expr_linexable(const IPA_NODE* caller, 
                         const SUMMARY_EXPR* expr)
{
  if (expr->Is_expr_unknown()) {
    return FALSE;
  }

  OPERATOR opr = OPCODE_operator(expr->Get_opcode());

  if (expr->Has_const_operand()) {
    // constant +|-|* value
    if (expr->Is_expr_value(0) &&
        (opr == OPR_ADD || opr == OPR_SUB || opr == OPR_MPY)) {
      const SUMMARY_VALUE* value = 
        IPA_get_value_array(caller) + expr->Get_node_index(expr->Get_kid());
      return Is_caller_value_linexable(caller, value);
    }
    // constant +|- expression
    else if (expr->Is_expr_expr(0) && (opr == OPR_ADD || opr == OPR_SUB)) {
      const SUMMARY_EXPR* kid_expr = 
        IPA_get_expr_array(caller) + expr->Get_node_index(expr->Get_kid());
      return Is_caller_expr_linexable(caller, kid_expr);
    }
  }
  else if (opr == OPR_ADD || opr == OPR_SUB) {
    // expression +|- value
    for (INT i = 0; i < 2; ++i) {
      if (expr->Is_expr_value(i)) {
        const SUMMARY_VALUE* value = 
          IPA_get_value_array(caller) + expr->Get_node_index(i);
        if (!Is_caller_value_linexable(caller, value)) {
          return FALSE;
        }
      }
      else if (expr->Is_expr_expr(i)) {
        const SUMMARY_EXPR* kid_expr = 
          IPA_get_expr_array(caller) + expr->Get_node_index(i);
        if (!Is_caller_expr_linexable(caller, kid_expr)) {
          return FALSE;
        }
      }
      else {
        return FALSE;
      }
    }
    return TRUE;
  }
  
  return FALSE;
}

// -------------------------------------------------
// Check if a value can be represented as a LINEX of
// constants, globals, and caller formals
// -------------------------------------------------
static BOOL 
Is_caller_value_linexable(const IPA_NODE* caller, 
                          const SUMMARY_VALUE* value)
{
  switch (value->Get_const_type()) {

    case VALUE_INT_CONST:
      return TRUE;

    case VALUE_CONST: {
      INT64 int_val;
      TCON_IDX tcon_idx = ST_tcon(ST_ptr(value->Get_const_st_idx()));
      return Targ_Is_Integral(Tcon_Table[tcon_idx], &int_val);
    }

    case VALUE_FORMAL: {
      IPAA_NODE_INFO* caller_modref = caller->Mod_Ref_Info();      
      INT32 caller_position = 
        IPA_get_formal_array(caller)[value->Get_formal_index()].Get_position();
      return (!caller_modref->Is_formal_dmod_elmt(caller_position) &&
              !caller_modref->Is_formal_imod_elmt(caller_position));
    }
    
    case VALUE_GLOBAL: {
      INT32 symbol_index = value->Get_global_index();
      if (symbol_index != -1) {
        IPAA_NODE_INFO* caller_modref = caller->Mod_Ref_Info();      
        ST* st = ST_ptr(IPA_get_symbol_array(caller)[symbol_index].St_idx());
        while (ST_base(st) != st) {
          st = ST_base(st);
        }
        return !caller_modref->Is_def_elmt(ST_index(st));
      }
    }
    
    case VALUE_EXPR: {
      SUMMARY_EXPR* expr = IPA_get_expr_array(caller)+value->Get_expr_index();
      return Is_caller_expr_linexable(caller, expr);
    }
  }
  
  return FALSE;
}


/*****************************************************************************
 *
 * Check if a callee's formal can be mapped into the caller's space. The
 * formal can only be mapped if the actual argument is:
 *
 * 1. constant value
 * 2. pass-through formal (not modified in the caller)
 * 3. global variable (not modified in the caller)
 *
 * DAVID COMMENT: why can't the actual be a local variable in the caller?
 *
 ****************************************************************************/

static BOOL Is_callee_formal_mappable_to_caller(const IPA_NODE *caller, 
        const SUMMARY_CALLSITE *call, UINT32 position)
{
    if (position >= call->Get_param_count()) return FALSE;

    const SUMMARY_ACTUAL& actual = 
        IPA_get_actual_array(caller)[call->Get_actual_index() + position];

    if (actual.Get_symbol_index() != -1)
    {     
        IPAA_NODE_INFO *caller_modref = caller->Mod_Ref_Info();
        const SUMMARY_SYMBOL& caller_symbol = 
            IPA_get_symbol_array(caller)[actual.Get_symbol_index()];

        if (caller_symbol.Is_formal())
        {
            // pass-through formal
            INT32 caller_position = Formal_position(
                    IPA_get_formal_array(caller), &caller_symbol);

            return (!caller_modref->Is_formal_dmod_elmt(caller_position) &&
                    !caller_modref->Is_formal_imod_elmt(caller_position));
        }
        else if (ST_IDX_level(caller_symbol.St_idx()) == GLOBAL_SYMTAB)
        {
            // global variable
            ST *st = ST_ptr(caller_symbol.St_idx());
            while (ST_base(st) != st) st = ST_base(st);

            return !caller_modref->Is_def_elmt(ST_index(st));
        }
    }

    if (actual.Get_value_index() != -1) {     
        return Is_caller_value_linexable(caller,
                IPA_get_value_array(caller) + actual.Get_value_index());
    }

    return FALSE;
}


/*****************************************************************************
 *
 * Check if the callee term can be mapped into caller's variables.
 *
 ****************************************************************************/

static BOOL Is_term_mappable_to_caller(
        const IPA_NODE* caller, const IPA_NODE* callee,
        const SUMMARY_CALLSITE* call, const TERM* t)
{
    switch (t->Get_type())
    {
        case LTKIND_CONST:
            return TRUE;

        case LTKIND_IV: 
            // DAVID COMMENT: this is not read!
            if (IPA_Enable_Simple_Alias)
            {
                INT32 size;
                const IVAR& ivar =
                    IPA_get_ivar_array(callee, size)[t->Get_desc()];
                if (ivar.Is_Formal()) {
                    UINT32 position = ivar.Formal_Position();
                    return (ivar.Offset() == 0 &&
                            Is_callee_formal_mappable_to_caller(caller, call, position));
                } else {
                    // must be a global variable
                    Is_True(ST_IDX_level(ivar.St_Idx()) == GLOBAL_SYMTAB,
                            ("Map_term_to_caller: expected a global ST"));
                    UINT32 modref_key = ST_IDX_index(ST_base_idx(ST_ptr(ivar.St_Idx())));
                    return !caller->Mod_Ref_Info()->Is_def_elmt(modref_key);
                }
            }
            return FALSE;

        default:
            return FALSE;
    }
}


/*****************************************************************************
 *
 * Map the linex terms to the caller variables, if we are unable to do the
 * mapping then, return FALSE.
 *
 ****************************************************************************/

static BOOL Is_linex_mappable_to_caller(
        const IPA_NODE* caller, const IPA_NODE* callee,
        const SUMMARY_CALLSITE* callsite, LINEX* l)
{
    // Go through each term.
    for (INT j = 0; j <= l->Num_terms(); ++j) {
        if (!Is_term_mappable_to_caller(caller, callee,
                    callsite, l->Get_term(j))) return FALSE;
    }

    return TRUE; 
}


/*****************************************************************************
 *
 * Return TRUE if we can map callee region in terms of caller's variables.
 *
 ****************************************************************************/

static BOOL Is_region_mappable_to_caller(
        const IPA_NODE* caller, const IPA_NODE* callee,
        const SUMMARY_CALLSITE *callsite, PROJECTED_REGION *callee_region)
{
    if (callee_region->Is_messy_region()) return FALSE;

    // Go through each dimension.
    for (INT i = 0; i < callee_region->Get_num_dims(); ++i)
    {
        PROJECTED_NODE *p1 = callee_region->Get_projected_node(i);
        Is_True(p1 != NULL,
                ("Is_projected_region_mappable_to_caller: p1 is NULL\n"));

        LINEX *ub, *lb, *step;

        // map upper bound
        if (!p1->Is_messy_ub() && (ub = p1->Get_upper_linex()) != NULL) {
            if (!Is_linex_mappable_to_caller(caller, callee, callsite, ub)) {
                return FALSE;
            }
        }

        // map lower bound
        if (!p1->Is_messy_lb() && (lb = p1->Get_lower_linex()) != NULL) {
            if (!Is_linex_mappable_to_caller(caller, callee, callsite, lb)) {
                return FALSE;
            }
        }

        // map step
        if (!p1->Is_messy_step() && (step = p1->Get_step_linex()) != NULL) {
            if (!Is_linex_mappable_to_caller(caller, callee, callsite, step)) {
                return FALSE;
            }
        }
    }

    return TRUE;
}


/*****************************************************************************
 * 
 * Map callee annotation in terms of caller variables. If we are able to map
 * all the callee variables in terms of actuals or globals in the caller,
 * then perform the mapping; otherwise set the new caller region to messy and
 * return.
 *
 ****************************************************************************/

extern void Map_callee_region_to_caller(
        const IPA_NODE *caller, const IPA_NODE *callee,
        const SUMMARY_CALLSITE *callsite,
        PROJECTED_REGION *caller_region, PROJECTED_REGION *callee_region) 
{
    if (Is_region_mappable_to_caller(caller, callee, callsite, callee_region))
    {
        mINT16 callee_ndims = callee_region->Get_num_dims();
        mINT16 caller_ndims = caller_region->Get_num_dims();

        Is_True(callee_ndims == caller_ndims, 
                ("Dim size mismatch in Map_callee_region_to_caller\n"));

        // Map each dimension.
        for (INT i = 0; i < callee_ndims; ++i) {
            Map_projected_node_to_caller(caller, callee, callsite,
                    caller_region->Get_projected_node(i),
                    callee_region->Get_projected_node(i));
        }
    }
    else
    {
        caller_region->Set_messy_region();
    }
}


/*****************************************************************************
 *
 * Return TRUE if at the 'callsite' in the node 'caller_node', the
 * 'caller_shape' and the 'callee_shape' have the same shape. Return FALSE
 * otherwise. Use memory from 'mem_pool'.
 *
 ****************************************************************************/

static BOOL Same_Shape(
        const IPA_NODE *caller_node, const IPA_NODE *callee_node,
        const SUMMARY_CALLSITE *callsite, 
        PROJECTED_REGION *caller_shape, PROJECTED_REGION *callee_shape,
        TYPE_ID caller_mtype, TYPE_ID callee_mtype,
        MEM_POOL *mem_pool)
{
    if (caller_shape == NULL && callee_shape == NULL) return TRUE;

    if (caller_shape == NULL || callee_shape == NULL || callsite == NULL
        || caller_shape->Get_num_dims() != callee_shape->Get_num_dims()
        || caller_mtype != callee_mtype) {
        return FALSE;
    }

    PROJECTED_REGION callee_shape_in_caller(callee_shape->Get_type(),
            callee_shape->Get_depth(), callee_shape->Get_num_dims(),
            mem_pool);

    Map_callee_region_to_caller(caller_node, callee_node, callsite,
            &callee_shape_in_caller, callee_shape);

    return caller_shape->Equivalent(&callee_shape_in_caller);
} 


/*****************************************************************************
 *
 * Return the projected region of an array section passed in.
 *
 ****************************************************************************/

static PROJECTED_REGION* Get_actual_passed_region(const IPA_NODE *caller, 
        const SUMMARY_ACTUAL& actual)
{
    INT ra_idx = actual.Get_index();
    Is_True(ra_idx != -1, ("Expecting a valid region array index\n"));

    // The REGION_ARRAY has only one PROJECTED_REGION.
    INT pr_idx = IPA_get_region_array(caller)[ra_idx].Get_idx();
    Is_True(pr_idx != -1, ("Expecting a valid projected region index\n"));

    return IPA_get_proj_region_array(caller) + pr_idx;
}


/*****************************************************************************
 *
 * Union section annotations:
 *
 * 1) If caller and callee regions exist, union the annotations and check for
 * change from old annotation.
 * 
 * 2) If only caller region exists, return no change, since no union operation
 * is performed.
 *
 * 3) If only callee region exists, copy the region and return change.
 *
 * BOOL is_mod: TRUE if unioning mod regions else ref regions
 *
 ****************************************************************************/

static BOOL Union_sections(const IPA_NODE* caller, const IPA_NODE* callee, 
        const SUMMARY_CALLSITE* callsite,
        STATE* caller_annot, STATE* callee_annot, 
        BOOL is_mod,
        PROJECTED_REGION* caller_shape = NULL,
        PROJECTED_REGION* callee_shape = NULL,
        TYPE_ID caller_mtype = MTYPE_UNKNOWN, 
        TYPE_ID callee_mtype = MTYPE_UNKNOWN,  
        PROJECTED_REGION* callsite_region = NULL)
{
    PROJECTED_REGION *caller_region, *callee_region;

    if (is_mod) {
        caller_region = caller_annot->Get_projected_mod_region();
        callee_region = callee_annot->Get_projected_mod_region();
    } else {
        caller_region = caller_annot->Get_projected_ref_region();
        callee_region = callee_annot->Get_projected_ref_region();
    }

    if (callee_region == NULL) return FALSE;
    // We do not union if the caller's region is messy.
    if (caller_region != NULL
            && caller_region->Is_messy_region()) return FALSE;


    MEM_POOL *mem_pool = caller->Section_Annot()->Mem_Pool();
    BOOL created_caller_region = FALSE;

    // Initialize caller_region, if necessary.
    if (caller_region == NULL)
    {
        caller_region = CXX_NEW(PROJECTED_REGION(callee_region->Get_type(),
                    callee_region->Get_depth(), callee_region->Get_num_dims(),
                    mem_pool), mem_pool);

        if (is_mod) {
            caller_annot->Set_projected_mod_region(caller_region);
        } else {
            caller_annot->Set_projected_ref_region(caller_region);
        }

        created_caller_region = TRUE;
    }

    // If callee region is messy, caller region should be too.
    if (callee_region->Is_messy_region()) {
        caller_region->Set_messy_region();
        return TRUE;
    }

    BOOL reshape_trace = Get_Trace(TP_IPA, IPA_TRACE_RESHAPE);

    RESHAPE reshape(caller_shape, callee_shape, callee_region, 
            callsite_region, mem_pool, reshape_trace);

    // Are the callee and caller decl regions of the same shape?
    BOOL same_shape = Same_Shape(caller, callee, callsite,
            caller_shape, callee_shape, 
            caller_mtype, callee_mtype, mem_pool);

    // caller and callee see the same shape, but an array section is passed.
    if (same_shape && callsite_region != NULL) {
        reshape.Set_callee_proj_reshaped_region(callee_region);
        if (!reshape.Reshapeable_Passed_Section(reshape_trace)) {
            caller_region->Set_messy_region();
            return TRUE;
        }
    }

    // diferent shapes, so we need to reshape callee region
    if (!same_shape && caller_shape) {
        callee_region = reshape.Reshape_Callee_To_Caller(reshape_trace);
        if (callsite_region != NULL && !callee_region->Is_messy_region()) {
            if (!reshape.Reshapeable_Passed_Section(reshape_trace)) {
                callee_region->Set_messy_region();
            }
        }
    }

    if (callee_region->Is_messy_region()) {
        caller_region->Set_messy_region();
        return TRUE;
    }

    PROJECTED_REGION *new_caller_region = 
        (created_caller_region &&
         caller_region->Get_num_dims() == callee_region->Get_num_dims()) ?
        caller_region :
        CXX_NEW(PROJECTED_REGION(callee_region->Get_type(),
                    callee_region->Get_depth(), callee_region->Get_num_dims(),
                    mem_pool), mem_pool);

    /* Map (possibly reshaped) callee region into caller's space. */
    Map_callee_region_to_caller(caller, callee, callsite,
            new_caller_region, callee_region);

    // Factor the effects of the array section passed at callsite.
    if (callsite_region != NULL) {
        reshape.Reshape_Passed_Section(new_caller_region, reshape_trace); 
    }

    BOOL change = TRUE;
    if (!created_caller_region) {
        // Merge with the existing caller region.
        change = caller_region->May_Union(*new_caller_region,
                Trace_IPA_Sections);
    } else {
        if (is_mod) {
            caller_annot->Set_projected_mod_region(new_caller_region);
        } else {
            caller_annot->Set_projected_ref_region(new_caller_region);
        }
    }

    return change;
}


/*****************************************************************************
 *
 * For the caller with given 'section', set the 'caller_annot' to MESSY.
 * Clone an annot from 'callee_annot' if 'caller_annot' is NULL.
 *
 ****************************************************************************/

static BOOL Set_Caller_Annot_Messy(STATE *caller_annot, STATE *callee_annot,
        BOOL make_messy_region, MEM_POOL *pool)
{
    BOOL change = FALSE;

    PROJECTED_REGION *pr_callee_mod = (callee_annot != NULL) ?
        callee_annot->Get_projected_mod_region() : NULL;

    if (pr_callee_mod != NULL || make_messy_region)
    {
        // Mark the MOD region messy.
        PROJECTED_REGION *pr_caller_mod =
            caller_annot->Get_projected_mod_region();
        if (pr_caller_mod == NULL) {
            mUINT8 depth = 0, ndims = 0;
            if (pr_callee_mod != NULL) {
                depth = pr_callee_mod->Get_depth();
                ndims = pr_callee_mod->Get_num_dims();
            }
            pr_caller_mod = CXX_NEW(PROJECTED_REGION(MESSY_REGION,
                        depth, ndims, pool), pool);
            caller_annot->Set_projected_mod_region(pr_caller_mod);
            change = TRUE;
        } else if (!pr_caller_mod->Is_messy_region()) {
            pr_caller_mod->Set_messy_region();
            change = TRUE;
        }
    }

    PROJECTED_REGION *pr_callee_ref = (callee_annot != NULL) ?
        callee_annot->Get_projected_ref_region() : NULL;

    if (pr_callee_ref != NULL || make_messy_region)
    {
        // Mark the REF region messy.
        PROJECTED_REGION* pr_caller_ref =
            caller_annot->Get_projected_ref_region();
        if (pr_caller_ref == NULL) {
            mUINT8 depth = 0, ndims = 0;
            if (pr_callee_ref != NULL) {
                depth = pr_callee_ref->Get_depth();
                ndims = pr_callee_ref->Get_num_dims();
            }
            pr_caller_ref = CXX_NEW(PROJECTED_REGION(MESSY_REGION,
                        depth, ndims, pool), pool);
            caller_annot->Set_projected_ref_region(pr_caller_ref);
            change = TRUE;
        } else if (!pr_caller_ref->Is_messy_region()) {
            pr_caller_ref->Set_messy_region();
            change = TRUE;
        }
    }

    return change;
}


/*****************************************************************************
 *
 * Set the array section for the 'i'-th argument of the 'caller' to MESSY,
 * because the 'callee' has been passed a reshaped array section. Information
 * about the actuals of the caller at that callsite is found in the array
 * 'actuals'.
 *
 ****************************************************************************/

static BOOL Set_Caller_Actual_Messy(const IPA_NODE *caller,
        IPA_NODE_SECTION_INFO *caller_info,
        IPA_NODE_SECTION_INFO *callee_info,
        const SUMMARY_FORMAL *caller_formals,
        const SUMMARY_SYMBOL *caller_symbols,
        const SUMMARY_ACTUAL *actuals, INT i)
{
    INT symbol_index = actuals[i].Get_symbol_index();
    if (symbol_index == -1) return FALSE;

    STATE *callee_formal = NULL;
    BOOL make_messy_region = FALSE;
    if (i >= 0 && i < callee_info->Get_formal_count()) {
        callee_formal = callee_info->Get_formal(i);
    } else {
        make_messy_region = TRUE;
    }

    const SUMMARY_SYMBOL *symbol = caller_symbols + symbol_index;
    ST_IDX st_idx = symbol->St_idx();

    /* If the actual is a formal of the caller, mark the former as messy. */
    if (symbol->Is_formal()
            && ST_IDX_level(st_idx) == caller->Lexical_Level())
    {
        INT formal_pos = Formal_position(caller_formals, symbol);
        return Set_Caller_Annot_Messy(caller_info->Get_formal(formal_pos),
                callee_formal, make_messy_region, caller_info->Mem_Pool());
    }

    /* If the actual is a global array, mark the global array as messy. */
    if (ST_IDX_level(st_idx) == GLOBAL_SYMTAB
            && ST_class(st_idx) == CLASS_VAR
            && TY_kind(ST_type(st_idx)) == KIND_ARRAY)
    {
        // Give up completely if common block elements are equivalenced.
        if (ST_is_equivalenced(ST_ptr(st_idx))) {
            return caller_info->Set_Global_Array_List_To_Messy(symbol);      
        }

        GLOBAL_ARRAY_INFO *gai = caller_info->Find_Global_Array_Info(symbol);
        if (gai == NULL) gai = caller_info->Add_Global_Array_Info(symbol);

        return Set_Caller_Annot_Messy(gai->Get_state(),
                callee_formal, make_messy_region, caller_info->Mem_Pool());
    }

    return FALSE;
}


/*****************************************************************************
 *
 * Walk the global sections of the callee and merge them in with the caller.
 *
 * IPA_NODE* caller : caller node
 * IPA_NODE* callee : callee node
 *
 ****************************************************************************/

static BOOL Merge_global_sections(IPA_NODE *caller, IPA_NODE *callee,
        SUMMARY_CALLSITE *call)
{
    BOOL change = FALSE; 

    IPA_NODE_SECTION_INFO *caller_info = caller->Section_Annot();
    IPA_NODE_SECTION_INFO *callee_info = callee->Section_Annot();

    GLOBAL_ARRAY_TABLE *caller_tbl = caller_info->Global_Array_Table();
    GLOBAL_ARRAY_TABLE *callee_tbl = callee_info->Global_Array_Table();

    MEM_POOL *pool = caller_info->Mem_Pool();

    // Walk through all commons in the callee and merge them with those in the
    // caller.
    ST_IDX st;
    GLOBAL_ARRAY_LIST *callee_list = NULL, *caller_list = NULL;
    GLOBAL_ARRAY_TABLE_ITER callee_tbl_iter(callee_tbl);
    while (callee_tbl_iter.Step(&st, &callee_list)) {
        caller_list = caller_tbl->Find(st);
        // Create the list if necessary.
        if (caller_list == NULL) {
            caller_list = CXX_NEW(GLOBAL_ARRAY_LIST(st), pool);
            caller_tbl->Enter(st, caller_list);
        }
        change |= caller_list->Merge(caller, callee, call, callee_list, pool);
    }

    return change; 
}

//---------------------------------------------------------------------------
// Build the projected region that represents the shape of the caller. 
// If no region exists return NULL
//---------------------------------------------------------------------------
static PROJECTED_REGION*
Global_shape_region(SUMMARY_SYMBOL* symbol, MEM_POOL* pool)
{
  TY_IDX ty_idx = ST_type(symbol->St_idx());
  INT num_dims = TY_AR_ndims(ty_idx);
  
  PROJECTED_REGION* p = 
    CXX_NEW(PROJECTED_REGION(NON_MESSY_REGION, 0, num_dims, pool), pool);
  
  // fill in projected nodes
  for (INT i = 0; i < num_dims; ++i) {
    PROJECTED_NODE* node = p->Get_projected_node (i);
    node->Set_constant_linexs (TY_AR_ubnd_val(ty_idx,i) -
                               TY_AR_lbnd_val(ty_idx,i),
                               0,
                               1, 
                               0, 
                               0);
  }
  return p;
}


/*****************************************************************************
 *
 * For each call do
 *     For each actual parameter do
 *         if (array && formal of the caller) {
 *             perform reshape analysis
 *             merge with formal section
 *         }
 *         if (array && global) {
 *             if (section exists) {
 *                 perform reshape analysis
 *                 merge with global section
 *             }
 *             else
 *                 mark global as bottom
 *         }
 *     For each global do
 *         if (section exists in caller) {
 *             if consistent shape and no aliasing
 *                 merge with global section
 *             else
 *                 mark global as bottom
 *         }
 *         else
 *             create new section
 *
 * Return TRUE is a change in a section is noted.
 *
 ****************************************************************************/

BOOL Merge_Section(IPA_NODE *caller)
{
    if (caller->Summary_Proc()->Has_incomplete_array_info()) return FALSE;

    // Be conservative if there are indirect or opaque calls.
    if (!caller->Icall_List().empty() || !caller->Ocall_List().empty()) {
        caller->Summary_Proc()->Set_has_incomplete_array_info();
        return TRUE;
    }

    BOOL change = FALSE;

    IPA_NODE_SECTION_INFO *caller_info = caller->Section_Annot();
    SUMMARY_FORMAL *caller_formals = IPA_get_formal_array(caller);
    SUMMARY_ACTUAL *caller_actuals = IPA_get_actual_array(caller);
    SUMMARY_SYMBOL *caller_symbols = IPA_get_symbol_array(caller);

    change = Merge_Execution_Cost(caller, &IPA_array_prop_pool);

    // Walk through the calls in this procedure.
    IPA_SUCC_ITER edge_iter(IPA_Call_Graph, caller);
    for (edge_iter.First(); !edge_iter.Is_Empty(); edge_iter.Next())
    {
        IPA_EDGE *e = edge_iter.Current_Edge();
        if (e == NULL) continue;

        IPA_NODE *callee = IPA_Call_Graph->Callee(e);
        IPA_NODE_SECTION_INFO *callee_info = callee->Section_Annot();
        SUMMARY_FORMAL *callee_formals = IPA_get_formal_array(callee);

        Init_IPA_Print_Arrays(callee);

        // Propagate INCOMPLETE_ARRAY_INFO bit from callee to caller.
        if (callee->Summary_Proc()->Has_incomplete_array_info()) {
            caller->Summary_Proc()->Set_has_incomplete_array_info();
            return TRUE;
        }

        SUMMARY_CALLSITE *call = e->Summary_Callsite();
        SUMMARY_ACTUAL *actuals = caller_actuals + call->Get_actual_index();

        // Go conservative if caller actual and formal counts do not match.
        INT formal_count = callee->Summary_Proc()->Get_formal_count();
        INT actual_count = e->Num_Actuals();
        if (formal_count != actual_count) {
            for (INT i = 0; i < actual_count; i++) {
                change = Set_Caller_Actual_Messy(caller, 
                        caller_info, callee_info, 
                        caller_formals, caller_symbols, actuals, i);
            }
            continue;
        }

        // Go through each actual.
        for (INT i = 0; i < actual_count; ++i)
        {
            // Do more checks on messiness.
            BOOL is_messy = IPA_Enable_Reshape ?
                Mismatched_Types(
                        caller, callee, call, i, &IPA_array_prop_pool) :
                Try_Reshape_Callee_Formal(
                        caller, callee, call, i, &IPA_array_prop_pool);

            if (is_messy) {
                change = Set_Caller_Actual_Messy(caller, 
                        caller_info, callee_info, 
                        caller_formals, caller_symbols, actuals, i);
                continue;
            }

            INT actual_symbol_index = actuals[i].Get_symbol_index();
            // This is the case where an entire array is passed.
            if (actual_symbol_index == -1) continue;

            STATE *callee_annot = callee_info->Get_formal(i);
            PROJECTED_REGION *callee_shape =
                callee_annot->Get_projected_dcl_region();
            STATE *caller_annot = NULL;
            PROJECTED_REGION *caller_shape = NULL;

            PROJECTED_REGION *passed_region = NULL;

            // Get the machine type of the corresponding formal of callee.
            INT32 callee_f_idx =
                callee->Summary_Proc()->Get_formal_index() + i;
            TYPE_ID callee_mtype =
                callee_formals[callee_f_idx].Get_machine_type();

            SUMMARY_SYMBOL *symbol = caller_symbols + actual_symbol_index;
            ST_IDX st_idx = symbol->St_idx();

            if (symbol->Is_formal()
                    && ST_IDX_level(st_idx) == caller->Lexical_Level())
            {
                // The actual is a formal of the caller.
                INT caller_mtype = 
                    caller_formals[symbol->Get_findex()].Get_machine_type();
                INT formal_pos = Formal_position(caller_formals, symbol);

                caller_annot = caller_info->Get_formal(formal_pos);
                caller_shape = caller_annot->Get_projected_dcl_region();

                if (actuals[i].Get_pass_type() == PASS_ARRAY_SECTION) {
                    passed_region = Projected_Region_To_Memory(caller,
                            Get_actual_passed_region(caller, actuals[i]),
                            &IPA_array_prop_pool);
                }

                // mod region
                change |= Union_sections(caller, callee, call,
                        caller_annot, callee_annot, TRUE,
                        caller_shape, callee_shape, 
                        caller_mtype, callee_mtype, passed_region);

                // ref region
                change |= Union_sections(caller, callee, call, 
                        caller_annot, callee_annot, FALSE,
                        caller_shape, callee_shape,
                        caller_mtype, callee_mtype, passed_region);
            }
            else if (ST_IDX_level(st_idx) == GLOBAL_SYMTAB)
            {
                // The actual is a global variable.
                if (ST_class(st_idx) == CLASS_FUNC 
                        || ST_class(st_idx) == CLASS_BLOCK) { 
                    change = Set_Caller_Actual_Messy(caller, 
                            caller_info, callee_info,
                            caller_formals, caller_symbols, actuals, i);
                    continue;
                }

                TY_IDX ty_idx = ST_type(st_idx);
                // no need to do anything for common scalars
                if (TY_kind(ty_idx) != KIND_ARRAY) continue;

                BOOL is_messy = FALSE;

                caller_info->Global_Array_Region(
                        symbol, &is_messy, NULL, TRUE);
                caller_info->Global_Array_Region(
                        symbol, &is_messy, NULL, FALSE);

                if (is_messy) continue;

                if (actuals[i].Get_pass_type() == PASS_ARRAY_SECTION) {
                    passed_region = Projected_Region_To_Memory(caller,
                            Get_actual_passed_region(caller, actuals[i]),
                            &IPA_array_prop_pool);
                }

                caller_annot =
                    caller_info->Find_Global_Array_Sections(symbol);
                caller_shape =
                    Global_shape_region(symbol, &IPA_array_prop_pool);

                INT caller_mtype = TY_mtype(TY_etype(ty_idx));

                change |= Union_sections(caller, callee, call,
                        caller_annot, callee_annot, TRUE, 
                        caller_shape, callee_shape,
                        caller_mtype, callee_mtype, passed_region);

                change |= Union_sections(caller, callee, call, 
                        caller_annot, callee_annot, FALSE, 
                        caller_shape, callee_shape,
                        caller_mtype, callee_mtype, passed_region);
            }
        }

        // Merge global sections in the callee with those in the caller.
        change |= Merge_global_sections(caller, callee, call);
    }

    return change;
}

// ==================================
// GLOBAL_ARRAY_LIST member functions
// ==================================


/*****************************************************************************
 *
//      Find_Global_Array_Info(SUMMARY_COMMON_SHAPE *table)
//
 * Return the common shape element in the common block list If it is not found
 * then return NULL
 *
 ****************************************************************************/

GLOBAL_ARRAY_INFO* GLOBAL_ARRAY_LIST::Find_Global_Array_Info(ST_IDX st_idx)
{
    Is_True(ST_IDX_level(st_idx) == GLOBAL_SYMTAB,
            ("Find_Global_Array_Info: Symbol is NOT global!\n"));

    GLOBAL_ARRAY_LIST_ITER iter(this);
    for (iter.First(); !iter.Is_Empty(); iter.Next())
    {
        GLOBAL_ARRAY_INFO *curr_section = iter.Cur();
        ST_IDX curr_st_idx = curr_section->St_Idx();

        if (st_idx == curr_st_idx ||
                (ST_type(st_idx) == ST_type(curr_st_idx) &&
                 ST_ofst(ST_ptr(st_idx)) == ST_ofst(ST_ptr(curr_st_idx)))) {
            return curr_section;
        }
    }

    return NULL;
}


/*****************************************************************************
 *
 * Merge callee's sections for elements in the common block with the caller's
 * (i.e. this instance).
 *
 ****************************************************************************/

BOOL GLOBAL_ARRAY_LIST::Merge(const IPA_NODE* caller, const IPA_NODE* callee,
        const SUMMARY_CALLSITE* call, GLOBAL_ARRAY_LIST* callee_list,
        MEM_POOL* m)
{
    if (callee_list->Is_messy()) {
        // if caller's list is messy, no change
        if (Is_messy()) return FALSE;
        Set_is_messy(); // set caller's list to messy
        return TRUE;
    }

    BOOL change = FALSE;

    // Walk through all common elements and merge mod/ref regions.
    GLOBAL_ARRAY_LIST_ITER iter(callee_list);
    for (iter.First(); !iter.Is_Empty(); iter.Next())
    {
        GLOBAL_ARRAY_INFO *callee_info = iter.Cur();
        ST_IDX st_idx = callee_info->St_Idx();

        GLOBAL_ARRAY_INFO *caller_info = Find_Global_Array_Info(st_idx);
        if (caller_info == NULL) caller_info = Append(st_idx, m);

        STATE *caller_state = caller_info->Get_state();
        STATE *callee_state = callee_info->Get_state();

        // NOTE: dont's use if with || here, because short-circuiting may skip
        // the second call to Union_sections.
        change |= Union_sections(caller, callee, call, 
                caller_state, callee_state, TRUE);
        change |= Union_sections(caller, callee, call, 
                caller_state, callee_state, FALSE);
    }

    return change;
}

//------------------------------------------------------------------------
//        Print the GLOBAL_ARRAY_LIST
//------------------------------------------------------------------------

void GLOBAL_ARRAY_LIST::Print(FILE* fp)
{
    GLOBAL_ARRAY_LIST_ITER iter(this);
    for (iter.First(); !iter.Is_Empty(); iter.Next()) iter.Cur()->Print(fp);
}


// ======================================
// IPA_NODE_SECTION_INFO member functions
// ======================================


/*****************************************************************************
 *
 * For the SUMMARY_SYMBOL 's' that represents a global array, return MOD
 * (is_mod == TRUE) or REF (is_mod == FALSE) projected region.
 *
 * Set 'is_messy' to TRUE if the entire common block is messy. If a new
 * GLOBAL_ARRAY_INFO is created, set its MOD/REF to 'region'.
 *
 ****************************************************************************/

PROJECTED_REGION* IPA_NODE_SECTION_INFO::Global_Array_Region(
        const SUMMARY_SYMBOL* s, BOOL* is_messy,
        PROJECTED_REGION* region, BOOL is_mod)
{
    ST_IDX st_idx = s->St_idx();

    GLOBAL_ARRAY_LIST *list = Find_Global_Array_List(s);
    if (list == NULL) {
        list = Add_Global_Array_List(s);
        if (ST_is_equivalenced(ST_ptr(s->St_idx()))) list->Set_is_messy();
    }

    if (list->Is_messy()) {
        *is_messy = TRUE;
        return NULL;
    }

    *is_messy = FALSE;

    GLOBAL_ARRAY_INFO *gai = Find_Global_Array_Info(s);
    if (gai == NULL) gai = Add_Global_Array_Info(s);

    if (is_mod) {
        if (region && !gai->Get_projected_mod_region()) {
            gai->Set_projected_mod_region(region);
        }
        return gai->Get_projected_mod_region();
    } else {
        if (region && !gai->Get_projected_ref_region()) {
            gai->Set_projected_ref_region(region);
        }
        return gai->Get_projected_ref_region();
    }
}

//--------------------------------------------------------
// SUMMARY_SYMBOL* s: summary symbol for the global array
//--------------------------------------------------------
BOOL
IPA_NODE_SECTION_INFO::Set_Global_Array_List_To_Messy(const SUMMARY_SYMBOL* s)
{ 
  GLOBAL_ARRAY_LIST* list = Find_Global_Array_List(s);
  if (!list) {
    list = Add_Global_Array_List(s);
  }
  if (list->Is_messy()) {
    return FALSE;
  }
  list->Set_is_messy();
  return TRUE;
}

//-------------------------------------------------------
// print the section annotation
//-------------------------------------------------------
void
IPA_NODE_SECTION_INFO::Print(FILE *fp)
{
  fprintf(fp,"---------start printing section information-------\n");
  if (STATE_ARRAY* state = Get_formals()) {
    for (INT i = 0; i < state->Elements(); ++i) {
      fprintf(fp, "formal %d : ", i);
      (*state)[i].Print(fp);
    }
  }
  fprintf(fp,"------end printing section information-------\n");
}

//-------------------------------------------------------
//      Print the global annotation
//-------------------------------------------------------
void 
IPA_NODE_SECTION_INFO::Print_Global_Sections (FILE* fp)
{
  GLOBAL_ARRAY_TABLE_ITER tbl_iter(Global_Array_Table());

  GLOBAL_ARRAY_LIST* list;
  ST_IDX st_idx;
  
  while (tbl_iter.Step(&st_idx, &list)) {
    fprintf(fp, "Common block = %s\n", ST_name(st_idx));
    GLOBAL_ARRAY_LIST_ITER iter(list);
    for (iter.First(); !iter.Is_Empty(); iter.Next())	{
      iter.Cur()->Print(fp);
    }
  }
}

//-------------------------------------------------------
// get the euse count for the formals in this PU
//-------------------------------------------------------
INT 
IPA_NODE_SECTION_INFO::Get_formal_euse_count()
{
  INT count = 0;
  if (STATE_ARRAY* formals_state = Get_formals()) {
    for (INT i = 0; i < formals_state->Elements(); ++i)	{
      if ((*formals_state)[i].Is_euse()) {
        ++count;
      }
    }
  }
  return count;
}

//-------------------------------------------------------
// get the kill count for the formals in this PU
//-------------------------------------------------------
INT 
IPA_NODE_SECTION_INFO::Get_formal_kill_count()
{
  INT count = 0;
  if (STATE_ARRAY* formals_state = Get_formals()) {
    for (INT i = 0; i < formals_state->Elements(); ++i)	{
      if ((*formals_state)[i].Is_must_kill()) {
        ++count;
      }
    }
  }
  return count;
}


// ==================================
// GLOBAL_ARRAY_INFO member functions
// ==================================

//-------------------------------------------
// Print mod/ref sections for a global array
//-------------------------------------------

void GLOBAL_ARRAY_INFO::Print(FILE* fp)
{
    const ST& st = St_Table[St_Idx()];

    st.Print(fp, FALSE);
    Ty_Table[ST_type(st)].Print(fp);

    if (PROJECTED_REGION* mod = Get_projected_mod_region()) {
        fprintf(fp, "mod region\n");
        mod->Print(fp);
    }

    if (PROJECTED_REGION* ref = Get_projected_ref_region()) {
        fprintf(fp, "ref region\n");
        ref->Print(fp);
    }
}


// ===================================
// SECTION_FILE_ANNOT member functions
// ===================================

//------------------------------------------------------------------------
//      Find an ivar entry
//------------------------------------------------------------------------
INT32 SECTION_FILE_ANNOT::Find_ivar(const IPA_NODE* node, const IVAR& ivar)
{
    INT32 ivar_size;
    const IVAR* ivar_array = IPA_get_ivar_array(node, ivar_size);
    for (INT32 i = 0; i < ivar_size; i++) {
        if (ivar_array[i] == ivar) return i;
    } 
    return -1; 
}     

//------------------------------------------------------------------------
//      Add a local ivar entry
//------------------------------------------------------------------------
INT32 SECTION_FILE_ANNOT::Add_ivar(const IPA_NODE* node, const IVAR& ivar)
{
    if (_iv_grow == NULL)
    {
        // copy all IVARs from the summary file into _iv_grow
        _iv_grow = CXX_NEW(IVAR_ARRAY(Mem_Pool()), Mem_Pool());
        INT32 ivar_size;
        IPA_get_ivar_file_array(node->File_Header(), ivar_size);
        for (INT32 i = 0; i < ivar_size; ++i) {
            _iv_grow->AddElement(_iv_base[i]);
        }
    }

    _iv_grow->AddElement(ivar);
    return _iv_grow->Lastidx();
}

//-------------------------------------------------------
//  Print the state information
//-------------------------------------------------------

void STATE::Print(FILE *fp)
{
    if (Is_must_kill()) fprintf(fp, " must kill, ");
    if (Is_may_kill())  fprintf(fp, " may kill, ");
    if (Is_euse())      fprintf(fp, " euse, ");
    if (Is_use())       fprintf(fp, " use, ");
    if (Is_must_reduc())fprintf(fp, " must reduc, ");
    if (Is_may_reduc()) fprintf(fp, " may reduc, ");
    if (Is_scalar())    fprintf(fp, "is scalar, ");

    PROJECTED_REGION *pr = NULL;
    if (pr = Get_projected_mod_region()) pr->Print_file(fp);
    fprintf(fp, "\n");
    if (pr = Get_projected_ref_region()) pr->Print_file(fp);
    fprintf(fp, "\n");
    if (pr = Get_projected_dcl_region()) pr->Print_file(fp);
    fprintf(fp, "\n");
}

/** DAVID CODE BEGIN **/

void STATE::ShallowCopy(const STATE *other)
{
    _state = other->_state;
    _mod = other->_mod;
    _ref = other->_ref;
    _dcl = other->_dcl;
}


BOOL IPA_get_call_data_summary(IPA_EDGE *e,
        STATE_ARRAY **p_actuals, GLOBAL_ARRAY_TABLE **p_globals,
        MEM_POOL *pool)
{
    Is_True(p_actuals != NULL && p_globals != NULL, (""));

    SUMMARY_CALLSITE *call = e->Summary_Callsite();
    IPA_NODE *caller = IPA_Call_Graph->Caller(e);
    IPA_NODE *callee = IPA_Call_Graph->Callee(e);

    // Does the callee have complete array info?
    BOOL incomplete_array_info =
        callee->Summary_Proc()->Has_incomplete_array_info();

    IPA_NODE_SECTION_INFO *callee_info = callee->Section_Annot();
    SUMMARY_FORMAL *callee_formals = IPA_get_formal_array(callee);

    // Create an GLOBAL_ARRY_TABLE. Go through the globals annot of the callee
    // and map them to the caller space.
    GLOBAL_ARRAY_TABLE *globals = CXX_NEW(GLOBAL_ARRAY_TABLE(307,pool), pool);

    GLOBAL_ARRAY_TABLE *callee_tbl = callee_info->Global_Array_Table();
    ST_IDX st;
    GLOBAL_ARRAY_LIST *callee_gal = NULL;
    GLOBAL_ARRAY_TABLE_ITER callee_tbl_iter(callee_tbl);
    while (callee_tbl_iter.Step(&st, &callee_gal))
    {
        // TODO: handle messy region for COMMON.
        if (callee_gal->Is_messy()) {
            DevWarn("COMMON messy region is not supported yet.");
            continue;
        }

        GLOBAL_ARRAY_LIST *caller_gal = CXX_NEW(GLOBAL_ARRAY_LIST(st), pool);
        globals->Enter(st, caller_gal);

        // Walk through all common elements and merge mod/ref regions.
        GLOBAL_ARRAY_LIST_ITER iter(callee_gal);
        for (iter.First(); !iter.Is_Empty(); iter.Next())
        {
            GLOBAL_ARRAY_INFO *callee_gai = iter.Cur();
            GLOBAL_ARRAY_INFO *caller_gai = CXX_NEW(
                    GLOBAL_ARRAY_INFO(callee_gai->St_Idx()), pool);
            caller_gal->Append(caller_gai);

            STATE *callee_annot = callee_gai->Get_state();
            STATE *caller_annot = caller_gai->Get_state();

            // No need to map regions if it is treated as a scalar or the
            // callee does not have complete array summary.
            if (callee_annot->Is_scalar()) continue;
            if (incomplete_array_info) continue;

            PROJECTED_REGION *callee_region = NULL, *caller_region = NULL;

            // REF info
            callee_region = callee_annot->Get_projected_ref_region();
            if (callee_region != NULL)
            {
                caller_region = CXX_NEW(
                        PROJECTED_REGION(callee_region->Get_type(),
                            callee_region->Get_depth(),
                            callee_region->Get_num_dims(), pool), pool);
                if (callee_region->Is_messy_region()) {
                    caller_region->Set_messy_region();
                } else {
#if 0
                    {
                        IPA_NODE_CONTEXT nc(callee);
                        callee_region->Print(stderr);
                    }
#endif
                    Map_callee_region_to_caller(caller, callee, call,
                            caller_region, callee_region);
                }
                caller_annot->Set_projected_ref_region(caller_region);
            }

            // MOD info
            callee_region = callee_annot->Get_projected_mod_region();
            if (callee_region != NULL)
            {
                caller_region = CXX_NEW(
                        PROJECTED_REGION(callee_region->Get_type(),
                            callee_region->Get_depth(),
                            callee_region->Get_num_dims(), pool), pool);
                if (callee_region->Is_messy_region()) {
                    caller_region->Set_messy_region();
                } else {
#if 0
                    {
                        IPA_NODE_CONTEXT nc(callee);
                        callee_region->Print(stderr);
                    }
#endif
                    Map_callee_region_to_caller(caller, callee, call,
                            caller_region, callee_region);
                }
                caller_annot->Set_projected_mod_region(caller_region);
            }

            // DCL info
            callee_region = callee_annot->Get_projected_dcl_region();
            // DAVID COMMENT: somehow this could be NULL.
            if (callee_region != NULL)
            {
                caller_region = CXX_NEW(
                        PROJECTED_REGION(callee_region->Get_type(),
                            callee_region->Get_depth(),
                            callee_region->Get_num_dims(), pool), pool);
                if (callee_region->Is_messy_region()) {
                    caller_region->Set_messy_region();
                } else {
                    Map_callee_region_to_caller(caller, callee, call,
                            caller_region, callee_region);
                }
                caller_annot->Set_projected_dcl_region(caller_region);
            }
        }
    }
    INT formal_count = callee->Summary_Proc()->Get_formal_count();
    INT actual_count = e->Num_Actuals();
    Is_True(formal_count == actual_count,
            ("IPA_get_call_data_summary: %d formals but %d actuals "
             "for call <%s> in <%s>", formal_count, actual_count,
             ST_name(callee->Func_ST()), ST_name(caller->Func_ST())));

    // Go through the formals of the callee, and map each into the caller
    // space. Create the STATE_ARRAY.
    STATE_ARRAY *actuals = CXX_NEW(STATE_ARRAY(pool), pool);
    actuals->Force_Alloc_array(actual_count);
    actuals->Setidx(actual_count-1);

    for (INT i = 0; i < actual_count; ++i)
    {
        STATE *callee_annot = callee_info->Get_formal(i);
        STATE *actual = &(*actuals)[i];

        // Make a shallow copy of the formal state.
        actual->ShallowCopy(callee_annot);

        // No need to map regions if it is treated as a scalar or the callee
        // does not have complete array summary.
        if (callee_annot->Is_scalar()) continue;
        if (incomplete_array_info) continue;

        PROJECTED_REGION *callee_region = NULL, *caller_region = NULL;

        // MOD info
        callee_region = callee_annot->Get_projected_mod_region();
        if (callee_region != NULL)
        {
            caller_region = CXX_NEW(
                    PROJECTED_REGION(callee_region->Get_type(),
                        callee_region->Get_depth(),
                        callee_region->Get_num_dims(), pool), pool);
            if (callee_region->Is_messy_region()) {
                caller_region->Set_messy_region();
            } else {
#if 0
                callee_region->Print(stderr);
#endif
                Map_callee_region_to_caller(caller, callee, call,
                        caller_region, callee_region);
            }
            actual->Set_projected_mod_region(caller_region);
        }

        // REF info
        callee_region = callee_annot->Get_projected_ref_region();
        if (callee_region != NULL)
        {
            caller_region = CXX_NEW(
                    PROJECTED_REGION(callee_region->Get_type(),
                        callee_region->Get_depth(),
                        callee_region->Get_num_dims(), pool), pool);
            if (callee_region->Is_messy_region()) {
                caller_region->Set_messy_region();
            } else {
#if 0
                callee_region->Print(stderr);
#endif
                Map_callee_region_to_caller(caller, callee, call,
                        caller_region, callee_region);
            }
            actual->Set_projected_ref_region(caller_region);
        }

        // DCL info
        callee_region = callee_annot->Get_projected_dcl_region();
        Is_True(callee_region != NULL, (""));
        caller_region = CXX_NEW(PROJECTED_REGION(callee_region->Get_type(),
                    callee_region->Get_depth(),
                    callee_region->Get_num_dims(), pool), pool);
        if (callee_region->Is_messy_region()) {
            caller_region->Set_messy_region();
        } else {
            Map_callee_region_to_caller(caller, callee, call,
                    caller_region, callee_region);
        }
        actual->Set_projected_dcl_region(caller_region);
    }



    *p_actuals = actuals;
    *p_globals = globals;

    return incomplete_array_info;
}

/*** DAVID CODE END ***/

