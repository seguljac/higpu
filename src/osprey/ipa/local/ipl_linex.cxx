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


/* -*-Mode: c++;-*- (Tell emacs to use c++ mode) */
//
// Exported functions:
//
//  extern void IPL_Access_Vector_To_Projected_Region(..)
//  extern void IPL_Finalize_Projected_Regions(..)
//
//--------------------------------------------------------------------------
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <elf.h>
#include <sys/elf_whirl.h>
#include <sys/types.h>

#include "defs.h"
#include "errors.h"
#include "tracing.h"
#include "wn.h"
#include "stab.h"                   /* for symtab    */
#include "const.h"                  /* for constab   */
#include "targ_const.h"             /* for tcon      */
#include "pu_info.h"
#include "irbdata.h"
#include "ir_bwrite.h"
#include "ir_bread.h"               /* for read routines routines */
#include "ir_bcom.h"
#include "loop_info.h"
#include "if_info.h"
#include "ipl_linex.h"
#include "ipa_section.h"
#include "cxx_hash.h"
#include "wn_util.h"
/** DAVID CODE BEGIN **/
#include "wn_simp.h"
#include "ipl_lwn_util.h"
/*** DAVID CODE END ***/
#include "ipl_summary.h"
#include "ipl_summarize.h"
#include "ipl_summarize_util.h"
#include "ipl_array_bread_write.h"
#include "ipl_tlog.h"
#include "ipl_main.h"
#include "ipl_lno_util.h"
#include "wb_ipl.h" 

/** DAVID CODE BEGIN **/
#ifdef HICUDA
#include "ir_reader.h"

#include "cxx_hash.h"

#include "hc_common.h"
#include "hc_utils.h"

#include "ipa_cg.h"
#include "ipa_section_annot.h"
#include "ipa_hc_gpu_data.h"
#include "ipa_hc_gpu_data_prop.h"

#ifdef __linux__
// defined in LNO
extern WN* (*HC_extract_arr_base_p)(WN *addr_wn, ST_IDX& arr_st_idx);
#define HC_extract_arr_base (*HC_extract_arr_base_p)
// defined in IPA
extern BOOL (*IPA_get_call_data_summary_p)(IPA_EDGE*,
        STATE_ARRAY**, GLOBAL_ARRAY_TABLE**, MEM_POOL*);
#define IPA_get_call_data_summary (*IPA_get_call_data_summary_p)
#else
// defined in LNO
#pragma weak HC_extract_arr_base
// defined in IPA
#pragma weak IPA_get_call_data_summary
#endif // __linux__

#endif  // HICUDA
/*** DAVID CODE END ***/


ARRAY_SUMMARY Array_Summary;

static SUMMARY_PROCEDURE* Current_summary_procedure = NULL;
static CFG_NODE_INFO *Cfg_entry = NULL;
static INT Cfg_entry_idx = -1;

//---------------------------------------------------------------------
// process a scalar reduction node
// check to see how far out it should go
//---------------------------------------------------------------------
static void 
process_scalar_reduc_node(WN* stmt, ST* lhs_st)
{
  BOOL branch;
  INT loop_idx;
  INT stmt_idx;
  INT cd_idx;
  WN* loop = NULL;
  SUMMARY_CONTROL_DEPENDENCE* d = Get_controlling_stmt(stmt);

  while (d) {
    cd_idx = Get_cd_idx(d);
    if (d->Is_do_loop()) {
      loop = d->Get_wn();
      loop_idx = cd_idx;
      d = Get_controlling_stmt(loop); 
    }
    else {
      if (loop && d->Is_entry()) {
        cd_idx = loop_idx;
      }
      else if (d->Is_if_stmt()) {
        SUMMARY_STMT* summary_stmt = 
          Search_for_summary_stmt(loop ? loop : stmt, branch, stmt_idx);
        Is_True(summary_stmt,("process_scalar_reduc_node: NULL summary stmt"));
      }

      CFG_NODE_INFO* cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);
      if (d->Is_if_stmt() && !branch) {
        cfg_node= Array_Summary.Get_cfg_node_array(cfg_node->Get_else_index());
      }
      cfg_node->Add_scalar_reduc(Summary->Get_symbol_index(lhs_st));
      return;
    }
  }

  // it is under some messy control flow
  Cfg_entry->Add_scalar_may_reduc(Summary->Get_symbol_index(lhs_st));
}

//---------------------------------------------------------------------
// process a scalar definition node
// check to see how far out it should go
//---------------------------------------------------------------------
static void 
process_scalar_def_node(WN* stmt, ST* lhs_st)
{
  BOOL branch;
  INT loop_idx;
  INT stmt_idx;
  INT cd_idx;
  CFG_NODE_INFO* cfg_node;
  WN* loop = NULL;
  INT id = Summary->Get_symbol_index(lhs_st);
  // PV 667258: If the user is writing a "constant", it must be under a 
  // conditional that he promises will never execute.  Therefor, we can 
  // ignore it. (rcox)
  if (id == -1)
    return; 
 
  SUMMARY_CONTROL_DEPENDENCE* d =  Get_controlling_stmt(stmt);

  while (d) {
    cd_idx = Get_cd_idx(d);
    if (d->Is_do_loop()) {
      loop = d->Get_wn();
      loop_idx = cd_idx;
      d = Get_controlling_stmt(loop); 
    }
    else { 
      // if the loop variable was set then add the definition
      // to the loop node
      if (loop && d->Is_entry()) {
        cd_idx = loop_idx;
        cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);
      }
      else {
        if (d->Is_if_stmt()) {
          SUMMARY_STMT* summary_stmt = 
            Search_for_summary_stmt(stmt, branch, stmt_idx);
          Is_True(summary_stmt, ("process_scalar_def: NULL summary stmt"));
          cd_idx = Get_cd_idx(d);
          cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);
          if (!branch) {
            cd_idx = cfg_node->Get_else_index();
            cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);
          }
        }
        else {
          cd_idx = Get_cd_idx(d);
          cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);
        }
      }
      

      // if it is the entry node add it to the symbols array
      if (cfg_node->Is_entry()) {
        Summary->Get_symbol(id)->Set_dkill();
      }

      // store it as part of the cfg node information
      cfg_node->Add_scalar_def(id);
      return;
    }
  }

  // is under a messy section, add it to the entry node
  Cfg_entry->Add_scalar_may_def(id);
}

//---------------------------------------------------------------------
// process a scalar definition node
// check to see how far out it should go
//---------------------------------------------------------------------
static void process_scalar_node(WN* node, IPA_SECTION_TYPE type)
{
    ST *s = WN_st(node);
    INT id = Summary->Get_symbol_index(s);

    // don't do anything if ST is constant
    if (id == -1) return;

    SUMMARY_STMT* summary_stmt;
    CFG_NODE_INFO* cfg_node;
    WN* loop = NULL;
    BOOL branch = FALSE;
    INT loop_idx;
    INT stmt_idx;

    INT cd_idx = -1;
    WN* stmt = IPL_get_stmt_scf(node);
    SUMMARY_CONTROL_DEPENDENCE* d = Get_controlling_stmt(stmt);

    while (d)
    {
        cd_idx = Get_cd_idx(d);
        if (d->Is_do_loop())
        {
            loop = d->Get_wn();
            loop_idx = cd_idx;
            d = Get_controlling_stmt(loop); 
        }
        else
        {
            // attach it at the loop header rather than the entry point
            if ((loop) && (d->Is_entry()))
                cd_idx = loop_idx;

#if 0
            if (d->Is_if_stmt() && loop)
            {
                summary_stmt = Search_for_summary_stmt(loop, branch,
                        stmt_idx);
                FmtAssert(summary_stmt != NULL,("process_scalar_node: NULL summary stmt"));
            }
            else if (d->Is_if_stmt())
            {
                summary_stmt = Search_for_summary_stmt(node, branch,
                        stmt_idx);
                FmtAssert(summary_stmt != NULL,("process_scalar_node: NULL summary stmt"));
            }
#endif
            // get the branch information
            // if it is not part of some array then it is
            // likely to be part of some control flow node
            // and hence it belongs to the def set of that
            // control flow node
            cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);

            if (d->Is_if_stmt() && branch == FALSE)
            {
                cfg_node = 
                    Array_Summary.Get_cfg_node_array(cfg_node->Get_else_index());
            }

            switch( type )
            {
                case IPA_USE: {
                                  // if there is a kill then don't bother with setting the
                                  // euse bit
                                  SUMMARY_SYMBOL *symbol = Summary->Get_symbol(id);
                                  if (!symbol->Is_dkill())
                                      cfg_node->Add_scalar_use(id);
                              }
                              break;

                case IPA_DEF:
                              cfg_node->Add_scalar_def(id);
                              break;

                case IPA_REDUC:
                              cfg_node->Add_scalar_reduc(id);
                              break;

                default:
                              Fail_FmtAssertion("unknown scalar type %d \n", id);
                              break;
            }
            return;
        }
    }

    if (loop)
    {
        d = Get_cd_by_idx(cd_idx);
        FmtAssert(d != NULL, (" Expecting a cd node \n"));

        if (d->Is_if_stmt())
        {
            summary_stmt = Search_for_summary_stmt(loop, branch, stmt_idx);
            FmtAssert(summary_stmt != NULL,("process_scalar_node: NULL summary stmt"));      
        }

        cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);

        if (d->Is_if_stmt() && branch == FALSE)
        {
            cfg_node = 
                Array_Summary.Get_cfg_node_array(cfg_node->Get_else_index());
        }

        id = Summary->Get_symbol_index(s);

        switch( type )
        {
            case IPA_USE: {
                              SUMMARY_SYMBOL *symbol = Summary->Get_symbol(id);
                              if (!symbol->Is_dkill())
                                  cfg_node->Add_scalar_use(id);
                          } 
                          break;

            case IPA_DEF:
                          cfg_node->Add_scalar_def(id);
                          break;

            case IPA_REDUC:
                          cfg_node->Add_scalar_reduc(id);
                          break;

            default:
                          Fail_FmtAssertion("unknown scalar type %d \n", id);
                          break;

        }
    }

    else
    {
        id = Summary->Get_symbol_index(s);
        switch( type )
        {
            case IPA_USE:
                Cfg_entry->Add_scalar_may_use(id);
                break;

            case IPA_DEF:
                Cfg_entry->Add_scalar_may_def(id);
                break;

            case IPA_REDUC:
                Cfg_entry->Add_scalar_may_reduc(id);
                break;

            default:
                Fail_FmtAssertion("unknown scalar type %d \n", id);
                break;

        }
    }
}

//----------------------------------------------------------------
// return the id into the actual array
//----------------------------------------------------------------
static INT get_actual_id(INT callsite_id, INT param_pos, INT start_idx)
{
    SUMMARY_CALLSITE *callsite = Summary->Get_callsite(callsite_id);
    INT actual_id = callsite->Get_actual_index() + param_pos - start_idx;

    if (Get_Trace(TP_IPL, TT_IPL_SECTION)) {
        fprintf(TFile, "callsite_id = %d , param_pos = %d , actual_id = %d\n", 
                callsite_id, param_pos, actual_id);
    }

    return actual_id;
}


/*****************************************************************************
 *
 * Process an actual parameter that is an ARRAY. The projected region is the
 * access. No real projection is done.
 *
 ****************************************************************************/

static void process_actual_array_node(WN* wn,
        INT16 callsite_id, INT16 actual_id)
{
    // Find the base address.
    WN *array_base = WN_array_base(wn);
    while (array_base != NULL && (WN_operator(array_base) == OPR_ARRAY)) {
        array_base = WN_array_base(array_base);
    }
    FmtAssert(array_base != NULL, ("NULL array base encountered\n"));
    if (!OPCODE_has_sym(WN_opcode(array_base))) return;

    ST *array_st = WN_st(array_base);
    ST_SCLASS array_ssc = ST_sclass(array_st);

    // Currently, we cannot deal with formals from a parent PU (652403)
    if ((array_ssc == SCLASS_FORMAL || array_ssc == SCLASS_FORMAL_REF) 
            && ST_level(array_st) != CURRENT_SYMTAB) {
        Current_summary_procedure->Set_has_incomplete_array_info();
        return;
    }
    // Skip local arrays.
    if (array_ssc == SCLASS_AUTO) return;

    TY_IDX array_ty_idx = ST_type(array_st);
    if (array_ssc == SCLASS_FORMAL) array_ty_idx = TY_pointed(array_ty_idx);

    /* DAVID COMMENT: here we need to make sure the dynamic array's type is
     * a pointer to an array so that the following check is passed.
     */

    // array within structs have ST of the struct plus offset
    // currently we can't deal with them (they'll be treated as scalars)
    if (TY_kind(array_ty_idx) != KIND_ARRAY) return;

    ACCESS_ARRAY *av = (ACCESS_ARRAY*)WN_MAP_Get(IPL_info_map, wn);
    FmtAssert(av != NULL, ("Null access vector encountered\n"));

    MEM_POOL *apool = Array_Summary.Get_array_pool();
    // DAVID COMMENT: this is just an access, not the projected region.
    PROJECTED_REGION *proj_region = CXX_NEW(
            PROJECTED_REGION(av, apool, NULL), apool);

    // Get the statement for this array expression.
    WN *stmt = IPL_get_stmt_scf(wn);
    // Get the control structure surrounding the statement.
    SUMMARY_CONTROL_DEPENDENCE *d = Get_controlling_stmt(stmt);
    INT cd_idx = (d != NULL) ? Get_cd_idx(d) : -1;

    CFG_NODE_INFO *cfg_node = NULL;
    if (cd_idx != -1) {
        cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);

        if (d->Is_if_stmt()) {
            BOOL branch;
            INT stmt_idx;
            SUMMARY_STMT *summary_stmt = Search_for_summary_stmt(stmt,
                    branch, stmt_idx);
            FmtAssert(summary_stmt != NULL,
                    ("process_actual_array_node: NULL summary stmt"));

            if (branch == FALSE) {
                cfg_node = Array_Summary.Get_cfg_node_array(
                        cfg_node->Get_else_index());
            }
        }
    } else {
        cfg_node = Cfg_entry;
    }

    /* Add this projected region to the entry CFG node, which contains all the
     * call parameters.
     */
    INT sym_id = Summary->Get_symbol_index(array_st);
    cfg_node->Add_array_param(proj_region, sym_id,
            TY_size(TY_etype(array_ty_idx)),
            callsite_id, actual_id);

    INT actual_index = get_actual_id(callsite_id, actual_id, 0);
    SUMMARY_ACTUAL *actual = Summary->Get_actual(actual_index);
    actual->Set_symbol_index(sym_id);
}

//----------------------------------------------------------------
// process an actual node, see if it is an array. If yes, then
// build the linex information
//----------------------------------------------------------------
static void process_actual_node(WN* call_stmt, WN* parm_wn,
        INT callsite_id, INT param_pos)
{
    WN *actual = WN_kid0(parm_wn);

    if (WN_operator(actual) == OPR_ARRAY) {
        process_actual_array_node(actual, callsite_id, param_pos);
    }

#if _THIS_SEEMS_USELESS_
  // if actual is a formal parameter, record the passed bit
  if (OPERATOR_has_sym(WN_operator(actual))) {
    ST* st = WN_st(actual);
    if (ST_sclass(st) == SCLASS_FORMAL_REF || ST_sclass(st) == SCLASS_FORMAL) {
      INT id = Summary->Get_symbol_index(st);
      SUMMARY_CONTROL_DEPENDENCE* d = Get_controlling_stmt(call_stmt);

      if (d) {
        BOOL branch;
        if (d->Is_if_stmt()) {
          INT stmt_idx;
          SUMMARY_STMT* summary_stmt = 
            Search_for_summary_stmt(call_stmt, branch, stmt_idx);
          FmtAssert(summary_stmt, ("process_actual_node: NULL summary stmt"));
        }
        INT cd_idx = Get_cd_idx(d);
        // store it as part of the cfg node information
        CFG_NODE_INFO* cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);
        if (d->Is_if_stmt() && branch == FALSE) {
          cd_idx = cfg_node->Get_else_index();
          cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);
        }
        // in this case record passed by ref for the scalar
        // pass it the id into the actual array, which has
        // an index into the summary symbol array
        INT pid  = cfg_node->Add_scalar_ref_passed(id, callsite_id);
        INT actual_idx = get_actual_id(callsite_id, param_pos,
                                       Array_Summary.Get_actual_start_idx());
        // note the 1, to avoid zero values
        Array_Summary.Set_actual_scalar_info_map(pid+1, cd_idx, actual_idx);
      }
      else {
        // add it to the entry node as may scalar_ref_passed
        INT id = Summary->Get_symbol_index(WN_st(WN_kid0(actual)));
        INT pid = Cfg_entry->Add_scalar_ref_may_passed(id, callsite_id);
        INT actual_idx = get_actual_id(callsite_id, param_pos,
                                       Array_Summary.Get_actual_start_idx());
        Array_Summary.Set_actual_scalar_info_map(pid+1, Cfg_entry_idx, 
                                                 actual_idx);
      }
    }
  }
#endif // _THIS_SEEMS_USELESS_
}

// --------------------------------------------
// Fill in PROJECTED_NODE bounds conservatively
// --------------------------------------------
static void Min_Max_Fill_Proj_Node(PROJECTED_NODE *pn,
        DYN_ARRAY<LOOPINFO*> *loops, INT bad_loop_count)
{
    BOOL substituted_lindex = FALSE;

    pn->Fill_Out();

    INT j;

    /* Minimize the lower bound. */

    LINEX *lx_lower = pn->Get_lower_linex();
    // NOTE: lx_lower->Num_terms() may change in the loop.
    for (j = 0; j <= lx_lower->Num_terms(); j++)
    {
        TERM *tm = lx_lower->Get_term(j);
        // We only worry about terms of loop index variables.
        if (tm->Get_type() != LTKIND_LINDEX) continue;

        INT loop_array_index = loops->Lastidx() - tm->Get_desc() 
            + bad_loop_count;
        if (loop_array_index < 0 || loop_array_index > loops->Lastidx()) {
            pn->Set_messy_lb();
            break;
        } 

        LOOPINFO *l = (*loops)[loop_array_index];
        LINEX *lx_substitute = (tm->Get_coeff() < 0)
            ? l->Max_value() : l->Min_value();
        if (lx_substitute == NULL) {
            pn->Set_messy_lb();
            break; 
        } 

        lx_lower->Substitute_Lindex(tm->Get_desc(), lx_substitute);
        substituted_lindex = TRUE;

        j = -1; // Start over again
    }

    /* Maximize the upper bound. */

    LINEX *lx_upper = pn->Get_upper_linex();
    // NOTE: lx_lower->Num_terms() may change in the loop.
    for (j = 0; j <= lx_upper->Num_terms(); j++)
    {
        TERM* tm = lx_upper->Get_term(j);
        // We only worry about terms of loop index variables.
        if (tm->Get_type() != LTKIND_LINDEX) continue;

        INT loop_array_index = loops->Lastidx() - tm->Get_desc() 
            + bad_loop_count; 
        if (loop_array_index < 0 || loop_array_index > loops->Lastidx()) {
            pn->Set_messy_ub();
            break; 
        } 

        LOOPINFO *l = (*loops)[loop_array_index];
        // This line is different from the lower bound code.
        LINEX* lx_substitute = (tm->Get_coeff() < 0)
            ? l->Min_value() : l->Max_value();
        if (lx_substitute == NULL) {
            pn->Set_messy_ub();
            break; 
        } 

        lx_upper->Substitute_Lindex(tm->Get_desc(), lx_substitute);
        substituted_lindex = TRUE;

        j = -1; // Start over again
    }

    // If any substitution has occured, set the step to be 1.
    if (substituted_lindex) {
        LINEX *lx_step = pn->Get_step_linex();
        if (lx_step != NULL) { 
            lx_step->Free_terms();
            lx_step->Set_term(LTKIND_CONST, (INT32) 1, CONST_DESC, 0);
        }
    }
}


// -------------------------------------------------------------------
// Try to recognize two-strided section, e.g., those that are created
// when two- or three-dimensional arrays are linearized.
//
// Currently, the projected node must be of the form
//              [ A*I+B : A*I+C : D ]
// where A,B,C,D are constant (A != 0), I is the index variable of
// the outermost loop, and the bounds (L,U) of that loop are constant.
//
// When the condition 
//                      A*(U-L) < D 
// is satisified, we have the two-strided section
//      [ A*L+B : A*U+C : ((D % A == 0) ? A : 1) : A*(U-L)+1 : D ]
//
// DAVID COMMENT: A two-strided section consists of a strided
// subsection repeated in a strided fashion.
// -------------------------------------------------------------------
static BOOL
Proj_Node_Has_Two_Strides (PROJECTED_NODE* proj_node,
                           DYN_ARRAY<LOOPINFO*>* loops,
			   INT bad_loop_count)
{
  LOOPINFO* outer_loop = NULL;

  if (bad_loop_count != 0)
    return FALSE; 

  INT l_coeff = 0;
  INT l_offset = 0;
  INT i;
  LINEX* lx_lower = proj_node->Get_lower_linex();

  for (i = 0; i <= lx_lower->Num_terms(); ++i) {
    TERM* tm = lx_lower->Get_term(i);
    if (tm->Get_type() == LTKIND_LINDEX) {
      if (tm->Get_desc() != 0) { 
        // not the outermost loop
        return FALSE;
      }
      else {
        // accumulate A
        l_coeff += tm->Get_coeff();
      }
    }
    else if (tm->Get_type() == LTKIND_CONST) {
      // accumulate B
      l_offset += tm->Get_coeff();
    }
    else {
      // not a constant or loop-index variable
      return FALSE;
    }
  }

  if (l_coeff == 0) {
    // A must be non-zero
    return FALSE;
  }
  
  INT u_coeff = 0;
  INT u_offset = 0;
  LINEX* lx_upper = proj_node->Get_upper_linex();

  for (i = 0; i <= lx_upper->Num_terms(); ++i) {
    TERM* tm = lx_upper->Get_term(i);
    if (tm->Get_type() == LTKIND_LINDEX) {
      if (tm->Get_desc() != 0) { 
        // not the outermost loop
        return FALSE;
      }
      else {
        // accumulate A and remember outermost loop
        u_coeff += tm->Get_coeff();
        outer_loop = (*loops)[loops->Lastidx()];
      }
    }
    else if (tm->Get_type() == LTKIND_CONST) {
      // accumulate C
      u_offset += tm->Get_coeff();
    }
    else {
      // not a constant or loop-index variable
      return FALSE;
    }
  }

  if (l_coeff != u_coeff) {
    // coefficients of the loop-index variable must both be A
    return FALSE;
  }

  LINEX* lx_step = proj_node->Get_step_linex();
  if (lx_step == NULL || !lx_step->Is_const()) {
    // D must be constant
    return FALSE;
  }
  INT stride = lx_step->Get_constant_term();

  LINEX* outer_max = outer_loop->Max_value();
  if (outer_max == NULL || !outer_max->Is_const()) {
    // outermost loop bounds must be constant
    return FALSE;
  }
  INT outer_ub = outer_max->Get_constant_term();

  LINEX* outer_min = outer_loop->Min_value();
  if (!outer_min->Is_const()) {
    // outermost loop bounds must be constant
    return FALSE;
  }
  INT outer_lb = outer_min->Get_constant_term();

  // To be on the safe side, assert that:
  // A > 0, D > 0, U >= L
  if (l_coeff < 0 || stride < 0 || outer_ub < outer_lb) {
    return FALSE;
  }
  
  // Must take care of the signs of coefficients vs. Max-Min
  if (l_coeff * (outer_ub - outer_lb) < stride) {
#if 0
    printf("Double-strided section:\nlb = %d\nub= %d\nstep = %d\nsegment_length = %d\nsegment_stride = %d\n",
           l_coeff * outer_lb + l_offset,
           u_coeff * outer_ub + u_offset,
           (stride % l_coeff == 0) ? l_coeff : 1,
           l_coeff * (outer_ub - outer_lb) + 1,
           stride);
#endif
    
    proj_node->Set_constant_two_strided_section(l_coeff*outer_lb + l_offset,
      l_coeff*outer_ub + u_offset, stride % l_coeff ? 1 : l_coeff,
      l_coeff*(outer_ub-outer_lb)+1, stride);
    return TRUE;
  }

  return FALSE;
}

// ----------------------------------------------
// Fill in PROJECTED_REGION bounds conservatively
// ----------------------------------------------
static void Min_Max_Fill_Region(PROJECTED_REGION *proj_region,
        DYN_ARRAY<LOOPINFO*> *loops, INT bad_loop_count)
{
    if (proj_region->Is_messy_region()) return; 

    PROJECTED_ARRAY *pa = proj_region->Get_projected_array();
    if (pa == NULL) return;

    BOOL is_projected_region = TRUE;

    for (INT i = 0; i <= pa->Lastidx(); i++) {
        PROJECTED_NODE *pn = &(*pa)[i];
        if (!Proj_Node_Has_Two_Strides(pn, loops, bad_loop_count)) {
            Min_Max_Fill_Proj_Node(pn, loops, bad_loop_count);
        }
        if (pn->Is_unprojected()) is_projected_region = FALSE;
    }

    if (is_projected_region) {
        proj_region->Reset_is_unprojected();
    } else {
        proj_region->Set_unprojected();
    }
}

/** DAVID CODE BEGIN **/
#ifdef HICUDA

// TODO: this is copied from <hc_utils.cxx>.
UINT16 num_array_dims(TY_IDX arr_ty_idx)
{
    UINT16 dimen = 0;
    TY_IDX ty_idx = arr_ty_idx;

    while (TY_kind(ty_idx) == KIND_ARRAY)
    {
        ARB_IDX arb_idx = Ty_Table[ty_idx].Arb();

        // accumulate the dimensionality.
        dimen += ARB_dimension(ARB_HANDLE(arb_idx));

        // move to the inner type.
        ty_idx = TY_etype(ty_idx);
    }

    return dimen;
}

/* Extracted from <process_array_node>. */
static PROJECTED_REGION* project_arr_region(WN *wn, WN_MAP parent_kernel_map)
{
    // Get the array base.
    WN* array_base = WN_array_base(wn);
    while (array_base != NULL && (WN_operator(array_base) == OPR_ARRAY)) {
        array_base = WN_array_base(array_base);
    }
    Is_True(array_base != NULL, ("project_arr_region: NULL array base\n"));

    if (!OPERATOR_has_sym(WN_operator(array_base))) return NULL;

    ST *array_st = WN_st(array_base);
    if (array_st == NULL) return NULL;

    ST_SCLASS array_ssc = ST_sclass(array_st);
    // Skip local arrays.
    // if (array_ssc == SCLASS_AUTO) return NULL;

    TY_IDX array_ty_idx = ST_type(array_st);
    if (array_ssc == SCLASS_FORMAL)
    {
        Is_True(TY_kind(array_ty_idx) == KIND_POINTER, (""));
        array_ty_idx = TY_pointed(array_ty_idx);
    }
    // array within structs have ST of the struct plus offset
    // currently we can't deal with them (they'll be treated as scalars)
    if (TY_kind(array_ty_idx) != KIND_ARRAY) return NULL;

    // currently, we cannot deal with formals from a parent PU (652403)
    if (array_ssc == SCLASS_FORMAL && ST_level(array_st) != CURRENT_SYMTAB) {
        Current_summary_procedure->Set_has_incomplete_array_info();
        return NULL;
    }

    ACCESS_ARRAY *av = (ACCESS_ARRAY*)WN_MAP_Get(IPL_info_map, wn);
    Is_True(av != NULL, ("project_arr_region: NULL access vector"));

    // Cache the current kernel region for later use.
    ST_IDX kfunc_st_idx = (ST_IDX)WN_MAP32_Get(parent_kernel_map, wn);

    /* Try to project out the dimensions of the array using loopinfo's. */

    // Use the local pool to allocate the region object.
    MEM_POOL *pool = Array_Summary.Get_array_pool();
    DYN_ARRAY<LOOPINFO*> loops(pool);
    PROJECTED_REGION *proj_region = NULL;
    INT cd_idx = -1;

    // Get the statement for this array expression.
    WN *stmt = IPL_get_stmt_scf(wn);
    // Get the control structure surrounding the statement.
    SUMMARY_CONTROL_DEPENDENCE *d = Get_controlling_stmt(stmt);

    while (d != NULL)
    {
        // Stop when the WN node is outside the kernel region.
        if (WN_MAP32_Get(parent_kernel_map, d->Get_wn())
                != kfunc_st_idx) break;

        INT cd_idx = Get_cd_idx(d);
        if (d->Is_do_loop())
        {
            LOOPINFO *l =
                Array_Summary.Get_cfg_node_array(cd_idx)->Get_loopinfo();
            loops.AddElement(l);

            if (proj_region == NULL)
            {
                proj_region = CXX_NEW(PROJECTED_REGION(av, pool, l), pool);
                // DAVID COMMENT: TY_AR_ndims is incorrect!!
                if (num_array_dims(array_ty_idx) != av->Num_Vec()) {
                    proj_region->Set_messy_region();
                }
            }
            proj_region->Project(l->Get_nest_level(), l);
        }

        d = Get_controlling_stmt(d->Get_wn());
    }

    /* Use max and min values to fill in unprojected loop elements. */

    INT loop_count = 0;
    for (WN* wnn = wn; wnn != NULL
            && WN_MAP32_Get(parent_kernel_map, wnn) == kfunc_st_idx;
            wnn = LWN_Get_Parent(wnn)) {
        if (WN_operator(wnn) == OPR_DO_LOOP) loop_count++; 
    }
    INT bad_loop_count = loop_count - (loops.Lastidx() + 1);

    // When we reach here, check if the projected region has been
    // created. If it has then the array was inside a loop. Otherwise,
    // create a projected region and add it to the control structure.
    if (proj_region != NULL)
    { 
        Min_Max_Fill_Region(proj_region, &loops, bad_loop_count);

        PROJECTED_KERNEL *proj_kernel = proj_region->Get_projected_kernel();
        if (proj_kernel != NULL && proj_kernel->Get_region() != NULL) {
            Min_Max_Fill_Region(proj_kernel->Get_region(),
                    &loops, bad_loop_count);
        }
    } 
    else
    {
        proj_region = CXX_NEW(PROJECTED_REGION(av, pool, NULL), pool);
        // DAVID COMMENT: TY_AR_ndims is incorrect!!
        if (num_array_dims(array_ty_idx) != av->Num_Vec()) {
            proj_region->Set_messy_region();
        }

        if (!proj_region->Is_messy_region()) proj_region->Fill_Out();

        // DAVID COMMENT: Can d be non-null here?
        if (d != NULL) {
            d = Get_controlling_stmt(stmt);
            cd_idx = Get_cd_idx(d);
            if (d->Is_if_stmt()) {
                BOOL branch;
                INT stmt_idx;
                SUMMARY_STMT* summary_stmt = 
                    Search_for_summary_stmt(stmt, branch, stmt_idx);
                Is_True (summary_stmt, ("process_array_node: NULL summary stmt"));
                if (branch == FALSE) {
                    cd_idx = Array_Summary.Get_cfg_node_array(cd_idx)->Get_else_index();
                }
            }
        }
    }

    return proj_region;
}


/* Only different from <project_arr_region> in that the projection starts from
 * an initial PROJECTED_REGION, as opposed to ACCESS_ARRAY.
 */
static PROJECTED_REGION* project_call_arr_region(WN *call_wn,
        PROJECTED_REGION *init_pr, WN_MAP parent_kernel_map)
{
    if (init_pr == NULL) return NULL;
    // Do nothing for a messy region.
    if (init_pr->Is_messy_region()) return init_pr;

    MEM_POOL *pool = Array_Summary.Get_array_pool();

    // We must make a copy of the initial PROJECTED_REGION in the local
    // mempool; otherwise, further projection may not work.
#if 0
    PROJECTED_REGION *proj_region = CXX_NEW(
            PROJECTED_REGION((mINT16)0,(mUINT8)0,(mUINT8)0,pool), pool);
    proj_region->Copy(init_pr);
#else
    PROJECTED_REGION *proj_region = init_pr;
#endif

    // Get the current kernel region for later use.
    ST_IDX kfunc_st_idx = (ST_IDX)WN_MAP32_Get(parent_kernel_map, call_wn);

    // Get the statement for the call.
    WN *stmt = IPL_get_stmt_scf(call_wn);
    // Get the control structure surrounding the statement.
    SUMMARY_CONTROL_DEPENDENCE *d = Get_controlling_stmt(stmt);

    DYN_ARRAY<LOOPINFO*> loops(pool);
    // DAVID COMMENT: necessary?
    INT cd_idx = -1;
    while (d != NULL)
    {
        // Stop when the WN node is outside the kernel region.
        if (WN_MAP32_Get(parent_kernel_map, d->Get_wn())
                != kfunc_st_idx) break;

        INT cd_idx = Get_cd_idx(d);
        if (d->Is_do_loop())
        {
            LOOPINFO *l =
                Array_Summary.Get_cfg_node_array(cd_idx)->Get_loopinfo();
            loops.AddElement(l);

            proj_region->Project(l->Get_nest_level(), l);
        }

        d = Get_controlling_stmt(d->Get_wn());
    }

    /* Use max and min values to fill in unprojected loop elements. */

    INT loop_count = 0;
    for (WN* wnn = call_wn; wnn != NULL
            && WN_MAP32_Get(parent_kernel_map, wnn) == kfunc_st_idx;
            wnn = LWN_Get_Parent(wnn)) {
        if (WN_operator(wnn) == OPR_DO_LOOP) loop_count++;
    }
    INT bad_loop_count = loop_count - (loops.Lastidx() + 1);

    Min_Max_Fill_Region(proj_region, &loops, bad_loop_count);

    PROJECTED_KERNEL *proj_kernel = proj_region->Get_projected_kernel();
    if (proj_kernel != NULL) {
        PROJECTED_REGION *pr = proj_kernel->Get_region();
        if (pr != NULL) Min_Max_Fill_Region(pr, &loops, bad_loop_count);
    }

    return proj_region;
}

#endif  // HICUDA
/*** DAVID CODE END ***/

//---------------------------------------------------------------------
// process an array node
//---------------------------------------------------------------------
static void process_array_node(WN *wn, IPA_SECTION_TYPE type)
{
    // Get the array base.
    WN* array_base = WN_array_base(wn);
    while (array_base != NULL && (WN_operator(array_base) == OPR_ARRAY)) {
        array_base = WN_array_base(array_base);
    }
    Is_True(array_base != NULL, ("process_array_node: NULL array base\n"));

    if (!OPERATOR_has_sym(WN_operator(array_base))) return;

    ST *array_st = WN_st(array_base);
    if (array_st == NULL) return;

    ST_SCLASS array_ssc = ST_sclass(array_st);
    // Skip local arrays.
    if (array_ssc == SCLASS_AUTO) return;

    TY_IDX array_ty_idx = ST_type(array_st);
    if (array_ssc == SCLASS_FORMAL) array_ty_idx = TY_pointed(array_ty_idx);
    // array within structs have ST of the struct plus offset
    // currently we can't deal with them (they'll be treated as scalars)
    if (TY_kind(array_ty_idx) != KIND_ARRAY) return;

    // currently, we cannot deal with formals from a parent PU (652403)
    if (array_ssc == SCLASS_FORMAL && ST_level(array_st) != CURRENT_SYMTAB) {
        Current_summary_procedure->Set_has_incomplete_array_info();
        return;
    }

    ACCESS_ARRAY *av = (ACCESS_ARRAY*)WN_MAP_Get(IPL_info_map, wn);
    Is_True(av != NULL, ("process_array_node: NULL access vector"));

    /* Try to project out the dimensions of the array using loopinfo's. */

    MEM_POOL *apool = Array_Summary.Get_array_pool();
    DYN_ARRAY<LOOPINFO*> loops(apool);
    PROJECTED_REGION *proj_region = NULL;
    INT cd_idx = -1;

    // Get the statement for this array expression.
    WN *stmt = IPL_get_stmt_scf(wn);
    // Get the control structure surrounding the statement.
    SUMMARY_CONTROL_DEPENDENCE *d = Get_controlling_stmt(stmt);

    while (d != NULL)
    {
        cd_idx = Get_cd_idx(d);
        if (d->Is_do_loop())
        {
            LOOPINFO *l =
                Array_Summary.Get_cfg_node_array(cd_idx)->Get_loopinfo();
            loops.AddElement(l);

            if (proj_region == NULL) {
                proj_region = CXX_NEW(PROJECTED_REGION(av, apool, l), apool);
                if (TY_AR_ndims(array_ty_idx) != av->Num_Vec()) {
                    proj_region->Set_messy_region();
                }
            }
            proj_region->Project(l->Get_nest_level(), l);
        }

        d = Get_controlling_stmt(d->Get_wn());
    }

    /* Use max and min values to fill in unprojected loop elements. */

    INT loop_count = 0;
    for (WN* wnn = wn; wnn != NULL; wnn = LWN_Get_Parent(wnn)) {
        if (WN_operator(wnn) == OPR_DO_LOOP) loop_count++; 
    }
    INT bad_loop_count = loop_count - (loops.Lastidx() + 1);

    // When we reach here, check if the projected region has been
    // created. If it has then the array was inside a loop. Otherwise,
    // create a projected region and add it to the control structure.
    if (proj_region != NULL)
    { 
        Min_Max_Fill_Region(proj_region, &loops, bad_loop_count);

        PROJECTED_KERNEL *proj_kernel = proj_region->Get_projected_kernel();
        if (proj_kernel != NULL && proj_kernel->Get_region() != NULL) {
            Min_Max_Fill_Region(proj_kernel->Get_region(),
                    &loops, bad_loop_count);
        }
    } 
    else
    {
        proj_region = CXX_NEW(PROJECTED_REGION(av, apool, NULL), apool);
        if (TY_AR_ndims(array_ty_idx) != av->Num_Vec()) {
            proj_region->Set_messy_region();
        }

        if (!proj_region->Is_messy_region()) proj_region->Fill_Out();

        // DAVID COMMENT: Can d be non-null here?
        if (d != NULL) {
            d = Get_controlling_stmt(stmt);
            cd_idx = Get_cd_idx(d);
            if (d->Is_if_stmt()) {
                BOOL branch;
                INT stmt_idx;
                SUMMARY_STMT* summary_stmt = 
                    Search_for_summary_stmt(stmt, branch, stmt_idx);
                Is_True (summary_stmt, ("process_array_node: NULL summary stmt"));
                if (branch == FALSE) {
                    cd_idx = Array_Summary.Get_cfg_node_array(cd_idx)->Get_else_index();
                }
            }
        }
    }

    INT element_size = TY_size(TY_etype(array_ty_idx));
    INT id = Summary->Get_symbol_index(array_st);

    if (cd_idx != -1)
    {
        // Store the projected region in the outermost CFG node.
        CFG_NODE_INFO *cfg_node = Array_Summary.Get_cfg_node_array(cd_idx);
        switch (type) {
            case IPA_DEF:
                cfg_node->Add_def_array(proj_region, element_size, id);
                break;
            case IPA_USE:
                cfg_node->Add_use_array(proj_region, element_size, id);
                break;
            case IPA_REDUC:
                cfg_node->Add_array_reduc(id);
                break;
        }
    }
    else
    {
        // Store the projected region in the entry CFG node.
        switch (type) {
            case IPA_DEF:
                Cfg_entry->Add_may_def_array(proj_region, element_size, id);
                break;
            case IPA_USE:
                Cfg_entry->Add_may_use_array(proj_region, element_size, id);
                break;
            case IPA_REDUC:
                Cfg_entry->Add_array_may_reduc(id);
                break;
        }
    }
}

/** DAVID CODE BEGIN **/
#ifdef HICUDA

/*****************************************************************************
 *
 * If the given call is inside a kernel in <proc_node>, add the call's data
 * summary to the kernel's DAS.
 *
 * ASSUME: IPA_EDGEs are linked with WN nodes.
 *
 ****************************************************************************/

static void HC_add_call_data_summary(WN *call_wn,
        IPA_NODE *proc_node, WN_MAP parent_kernel_map)
{
    // Make sure that it is inside a kernel region.
    ST_IDX kfunc_st_idx = (ST_IDX)WN_MAP32_Get(parent_kernel_map, call_wn);
    if (kfunc_st_idx == ST_IDX_ZERO) return;
    // Get the kernel info.
    HC_KERNEL_INFO *ki = proc_node->get_kernel_info_by_sym(kfunc_st_idx);
    Is_True(ki != NULL, (""));

    // Get the number of actuals.
    UINT n_actuals = WN_kid_count(call_wn);

    // Handle the case when it does not have an outgoing edge
    // (e.g. a library call).
    WN_TO_EDGE_MAP *wte_map = Summary->Get_wn_to_edge_map();
    Is_True(wte_map != NULL, (""));
    IPA_EDGE *e = wte_map->Find(call_wn);
    if (e == NULL)
    {
        // Guess which one is an array section complete the access redirection
        // record (in correspondences to HC_process_symbol_in_kregion).
        // For each actual of the given call <call_wn>, if it is an LDID of a
        // pointer-to-ARRAY or an LDA of an ARRAY with zero offset, add an
        // HC_ARRAY_INFO record to the kernel's DAS.
        BOOL added = FALSE;

        for (UINT i = 0; i < n_actuals; ++i)
        {
            WN *actual_wn = WN_kid(call_wn,i);
            ST_IDX actual_st_idx = HCWN_verify_pointer_param(actual_wn);
            if (actual_st_idx == ST_IDX_ZERO) continue;

            // Add an HC_ARRAY_INFO to the kernel's DAS.
            ki->add_arr_region(actual_st_idx, call_wn, i, NULL, NULL);
            added = TRUE;
        }

        // The kernel has incomplete array info if there is at least one array
        // record added.
        if (added) ki->set_has_incomplete_array_info();

        return;
    }

    // Get the call's side effects in the caller space.
    MEM_POOL *pool = proc_node->Mem_Pool();
    STATE_ARRAY *actuals = NULL;
    GLOBAL_ARRAY_TABLE *globals = NULL;
    BOOL incomplete_array_info = IPA_get_call_data_summary(e,
            &actuals, &globals, pool);
    if (incomplete_array_info) ki->set_has_incomplete_array_info();

    // Sanity check on the number of actuals.
    Is_True(n_actuals == actuals->Elements(), (""));
    Is_True(n_actuals == e->Num_Actuals(), (""));

    // Go through each actual.
    for (UINT i = 0; i < n_actuals; ++i)
    {
        STATE *actual = &(*actuals)[i];

        WN *param_wn = WN_kid(call_wn,i);
        ST_IDX arr_st_idx = HCWN_verify_pointer_param(WN_kid0(param_wn));
        // Since the actual could be an expression, we should handle them
        // using def-use chains, which should have been done.
        if (arr_st_idx == ST_IDX_ZERO) continue;

        // Project the REF/MOD region.
        PROJECTED_REGION *ref_pr = NULL, *mod_pr = NULL;
        if (! incomplete_array_info)
        {
            Is_True(! actual->Is_scalar(), (""));
            ref_pr = project_call_arr_region(call_wn,
                    actual->Get_projected_ref_region(), parent_kernel_map);
            mod_pr = project_call_arr_region(call_wn,
                    actual->Get_projected_mod_region(), parent_kernel_map);
        }

        // Add it to the kernel's DAS.
        ki->add_arr_region(arr_st_idx, call_wn, i, ref_pr, mod_pr);
    }

    // Go through each global.
    ST_IDX st;
    GLOBAL_ARRAY_LIST *gal = NULL;
    GLOBAL_ARRAY_TABLE_ITER gati(globals);
    while (gati.Step(&st, &gal))
    {
        // TODO: handle messy regions.
        if (gal->Is_messy()) {
            DevWarn("messy region is not supported yet.");
            continue;
        }

        GLOBAL_ARRAY_LIST_ITER gali(gal);
        for (gali.First(); !gali.Is_Empty(); gali.Next())
        {
            GLOBAL_ARRAY_INFO *gai = gali.Cur();
            STATE *global = gai->Get_state();

            ST_IDX glob_st_idx = gai->St_Idx();

            // We cannot use Is_scalar to judge if this is a scalar or not.
            if (! HCST_is_array(glob_st_idx))
            {
                Is_True(global->Is_scalar(), (""));
                // FIXME: non-zero offset? what is Is_euse?
                if (global->Is_use() || global->Is_euse()) {
                    ki->add_scalar(glob_st_idx, 0, TRUE);
                }
                if (global->Is_must_kill() || global->Is_may_kill()) {
                    ki->add_scalar(glob_st_idx, 0, FALSE);
                }
            }
            else
            {
                PROJECTED_REGION *ref_pr = NULL, *mod_pr = NULL;
                if (! incomplete_array_info)
                {
                    Is_True(! global->Is_scalar(), (""));
                    // Project the REF/MOD region.
                    ref_pr = project_call_arr_region(call_wn,
                            global->Get_projected_ref_region(),
                            parent_kernel_map);
                    mod_pr = project_call_arr_region(call_wn,
                            global->Get_projected_mod_region(),
                            parent_kernel_map);
                }

                // Add it to the kernel's DAS.
                ki->add_arr_region(glob_st_idx, call_wn, -1, ref_pr, mod_pr);
            }
        }
    }
}

/*****************************************************************************
 *
 * If the given array access (ILOAD/ISTORE node) is inside a kernel region in
 * <proc_node>, add the array section accessed to the kernel's DAS.
 *
 ****************************************************************************/

static void HC_add_array_access_to_kernel_das(WN *wn,
        IPA_NODE *proc_node, WN_MAP parent_kernel_map)
{
    // Get the parent kernel region.
    ST_IDX kfunc_st_idx = (ST_IDX)WN_MAP32_Get(parent_kernel_map, wn);
    if (kfunc_st_idx == ST_IDX_ZERO) return;

    // Use the kernel symbol to get HC_KERNEL_INFO.
    HC_KERNEL_INFO *ki = proc_node->get_kernel_info_by_sym(kfunc_st_idx);
    Is_True(ki != NULL, (""));

    OPERATOR opr = WN_operator(wn);
    // TODO: ILDBIT, ISTBIT
    Is_True(opr == OPR_ILOAD || opr == OPR_ISTORE, (""));

    WN *addr_wn = (opr == OPR_ILOAD) ? WN_kid0(wn) : WN_kid1(wn);
    if (WN_operator(addr_wn) == OPR_ARRAY)
    {
        Is_True(WN_offset(wn) < WN_element_size(addr_wn), (""));

        // The address base must be a single symbol.
        WN *addr_base_wn = WN_array_base(addr_wn);
        OPERATOR addr_base_opr = WN_operator(addr_base_wn);
        Is_True((addr_base_opr == OPR_LDID || addr_base_opr == OPR_LDA)
                && WN_offset(addr_base_wn) == 0, (""));

        // Project the array section, and add it to the kernel DAS.
        ST_IDX arr_st_idx = WN_st_idx(addr_base_wn);
        PROJECTED_REGION *pr = project_arr_region(addr_wn,
                parent_kernel_map);
        if (opr == OPR_ILOAD) {
            ki->add_arr_region(arr_st_idx, wn, pr, NULL);   // REF
        } else {
            ki->add_arr_region(arr_st_idx, wn, NULL, pr);   // MOD
        }
    }
    else
    {
        // Extract the base array symbol.
        ST_IDX base_st_idx = ST_IDX_ZERO;
        WN *ofst_wn = HC_extract_arr_base(addr_wn, base_st_idx);
        WN_DELETE_Tree(ofst_wn);
        // TODO
        Is_True(base_st_idx != ST_IDX_ZERO,
                ("Failed to find the base address symbol of "
                 "an indirect access in kernel <%s>\n",
                 ST_name(ki->get_kernel_sym())));

        // For now, we mark the kernel to have incomplete array info.
        ki->set_has_incomplete_array_info();
        ki->add_arr_region(base_st_idx, wn, NULL, NULL);
    }
}

static void HC_process_messy_array_accesses_walker(WN *wn, 
        IPA_NODE *proc_node, WN_MAP parent_kernel_map)
{
    // Get the parent kernel region.
    ST_IDX kfunc_st_idx = (ST_IDX)WN_MAP32_Get(parent_kernel_map, wn);
    if (kfunc_st_idx == ST_IDX_ZERO) return;

    // Use the kernel symbol to get HC_KERNEL_INFO.
    HC_KERNEL_INFO *ki = proc_node->get_kernel_info_by_sym(kfunc_st_idx);
    Is_True(ki != NULL, (""));

    OPERATOR opr = WN_operator(wn);
    // TODO: ILDBIT, ISTBIT
    if (opr == OPR_ILOAD || opr == OPR_ISTORE)
    {
        // Determine the base array variable.
        WN *addr_wn = (opr == OPR_ILOAD) ? WN_kid0(wn) : WN_kid1(wn);
        if (WN_operator(addr_wn) == OPR_ARRAY)
        {
            addr_wn = WN_array_base(addr_wn);
        }
        ST_IDX base_st_idx = ST_IDX_ZERO;
        WN *ofst_wn = HC_extract_arr_base(addr_wn, base_st_idx);
        WN_DELETE_Tree(ofst_wn);

        // When we fail to determine the base array variable, the generated
        // code may be incorrect. Warn the user.
        if (base_st_idx == ST_IDX_ZERO)
        {
            HC_warn("Kernel <%s> has an indrect access that "
                    "the compiler cannot handle! "
                    "The kernel code generated may be incorrect.",
                    ST_name(kfunc_st_idx));
        }
        else
        {
            // Add a record to the kernel's DAS.
            // If this access has already been added, this should have no
            // effect. Otherwise, it serves an indication that there is this
            // messy access in the kernel region.
            ki->add_arr_region(base_st_idx, wn, NULL, NULL);
        }
    }

    // Handle composite nodes.
    if (opr == OPR_BLOCK)
    {
        for (WN *kid_wn = WN_first(wn); kid_wn != NULL;
                kid_wn = WN_next(kid_wn))
        {
            HC_process_messy_array_accesses_walker(kid_wn,
                    proc_node, parent_kernel_map);
        }
    }
    else
    {
        INT nkids = WN_kid_count(wn);
        for (INT i = 0; i < nkids; ++i)
        {
            WN *kid_wn = WN_kid(wn, i);
            HC_process_messy_array_accesses_walker(kid_wn,
                    proc_node, parent_kernel_map);
        }
    }
}

void HC_process_messy_array_accesses(WN *proc_wn,
        IPA_NODE *proc_node, WN_MAP parent_kernel_map)
{
    // Walk through each statement node in the function.
    for (WN_ITER *wni = WN_WALK_StmtIter(proc_wn); wni != NULL;
            wni = WN_WALK_StmtNext(wni))
    {
        HC_process_messy_array_accesses_walker(WN_ITER_wn(wni),
                proc_node, parent_kernel_map);
    }
}

#endif  // HICUDA
/*** DAVID CODE END ***/

//--------------------------------------------------------------------------
// process the stmt_scf node
//--------------------------------------------------------------------------
static void process_node(WN* wn, IPA_SECTION_TYPE type)
{
    WN *lhs, *rhs, *parent_wn;
    INT32 map;

    BOOL save_trace_sections = Trace_Sections;
    if (Get_Trace(TP_IPL, TT_IPL_VERBOSE)) Trace_Sections = TRUE; 

    switch(WN_operator(wn))
    {
        case OPR_ISTORE:
        case OPR_MSTORE:
        {
            lhs = WN_kid0(wn);
            rhs = WN_kid1(wn);
            process_node(lhs, IPA_USE);
            map = WN_MAP32_Get(IPL_reduc_map, wn);
            process_node(rhs, (map > 0 && map < 5) ? IPA_REDUC : IPA_DEF);

/** DAVID CODE BEGIN **/
#ifdef HICUDA
            IPA_NODE *proc_node = Summary->Get_ipa_node();
            if (proc_node != NULL) {
                HC_add_array_access_to_kernel_das(wn,
                        proc_node, Summary->Get_parent_kernel_map());
            }
#endif  // HICUDA
/*** DAVID CODE END ***/
        }
        break;

        case OPR_STID:
        {
            // check if it involves a reduction, if it does then set both
            // the rhs and the lhs as reduction
            rhs = WN_kid0(wn);
            ST* lhs_st = WN_st(wn);
            process_node(rhs, IPA_USE);
            map = WN_MAP32_Get(IPL_reduc_map, wn);
            if (map > 0 && map < 5)
                process_scalar_reduc_node(wn, lhs_st);
            else
                process_scalar_def_node(wn, lhs_st);
        } 
        break;

        case OPR_CALL:
        case OPR_ICALL: 
        case OPR_INTRINSIC_CALL:
        {
            // ignore fake calls
            if (WN_opcode(wn) == OPC_VCALL
                    && WN_Fake_Call_EH_Region(wn, Parent_Map)) break;

            // Index into the CALLSITE array is stored as a map.
            INT call_index = IPL_get_callsite_id(wn);
            FmtAssert(call_index != -1, ("Unknown call encountered\n"));
            SUMMARY_CALLSITE *c = Summary->Get_callsite(0) + call_index;

/** DAVID CODE BEGIN **/
#ifdef HICUDA
            IPA_NODE *proc_node = Summary->Get_ipa_node();
            if (proc_node != NULL) {
                HC_add_call_data_summary(wn, proc_node, 
                        Summary->Get_parent_kernel_map());
            }
#endif  // HICUDA
/*** DAVID CODE END ***/

            // Walk all the kids and update the actual parameter node.
            // DAVID COMMENT: here we do not call process_node.
            INT n_actuals = c->Get_param_count();
            for (INT i = 0; i < n_actuals; ++i) {
                process_actual_node(wn, WN_actual(wn,i), call_index, i);
            }
            break;
        }

        case OPR_ARRAY:
        {
            parent_wn = LWN_Get_Parent(wn);

            switch (WN_operator(parent_wn)) {
                case OPR_ILOAD:
                case OPR_MLOAD:
                    // check if it is a reduction
                    map = WN_MAP32_Get(IPL_reduc_map, parent_wn);
                    if (map != RED_NONE) process_array_node(wn, IPA_REDUC);
                    process_array_node(wn, IPA_USE);
                    break;
                case OPR_ISTORE:
                    process_array_node(wn, IPA_DEF);
                    break;
                default:
                    process_array_node(wn, type);
                    break;
            }

            // Need to look for arrays nested inside arrays 
            for (INT i = 0; i < WN_num_dim(wn); i++) 
                process_node(WN_array_index(wn, i), IPA_USE);
        } 
        break;

        case OPR_LDID:
            parent_wn = LWN_Get_Parent(wn);
            if ((WN_operator(parent_wn) != OPR_ARRAY)
                    && (WN_operator(parent_wn) != OPR_PARM)) {
                // check if is a reduction type
                map = WN_MAP32_Get(IPL_reduc_map, wn);
                if (map > 0 && map < 5)
                    process_scalar_node(wn, IPA_REDUC);
                else if (type != IPA_DEF)
                    process_scalar_node(wn, IPA_USE);
                else
                    process_scalar_node(wn, type);
            }
            break;

        case OPR_MLOAD: 
        case OPR_ILOAD:
        {
            // In the case of an iload, check if the parent node is an array
            // if so then don't bother to do anything with it.
            parent_wn = LWN_Get_Parent(wn);
            if ((WN_operator(parent_wn) != OPR_ARRAY)
                    || (WN_operator(parent_wn) != OPR_PARM)) {
                for (INT i = 0; i < WN_kid_count(wn); ++i)
                    process_node(WN_kid(wn,i), type);
            }

/** DAVID CODE BEGIN **/
#ifdef HICUDA
            IPA_NODE *proc_node = Summary->Get_ipa_node();
            if (proc_node != NULL) {
                HC_add_array_access_to_kernel_das(wn,
                        proc_node, Summary->Get_parent_kernel_map());
            }
#endif  // HICUDA
/*** DAVID CODE END ***/
        } 
        break;

        case OPR_IO:
        {
            for (INT i = 0; i < WN_kid_count(wn); ++i)
                process_node(WN_kid(wn,i), type);
        }
        break;

/** DAVID CODE BEGIN **/
#ifdef HICUDA
        // We cannot have OPR_SWITCH as it only exists in VH Whirl.
        case OPR_COMPGOTO:
        {
            // Process the switch expression.
            process_node(WN_kid0(wn), type);
        }
        break;
#endif
/*** DAVID CODE END ***/

        default:
        {
            if (OPCODE_is_expression(WN_opcode(wn))) {
                for (INT i = 0; i < WN_kid_count(wn); ++i)
                    process_node(WN_kid(wn,i), type);
            }
        } 
        break;
    }

    Trace_Sections = save_trace_sections; 
}

//-----------------------------------------------------------------------
// NAME: Process_Array_Formals
// FUNCTION: For the subprogram 'wn_func', whose formals have indices in
//   the SUMMARY_FORMAL array from 'idx_formal_first' to 'idx_formal_first'
//   + 'formal_count' - 1, add entries in the REGION array corresponding
//   to those formals which are declared as arrays.  Allocate memory from
//   'mem_pool'.
// NOTE: These summaries are needed for reshaping analysis in IPA.
//-----------------------------------------------------------------------
static void Process_Array_Formals(WN* wn_func,
        INT idx_formal_first, INT formal_count, MEM_POOL* mem_pool)
{
    for (INT i = 0; i < formal_count; i++)
    {
        ST *st_formal = WN_st(WN_formal(wn_func,i));
        FmtAssert(st_formal != NULL,
                ("Process_Array_Formals: Expecting non-NULL st_formal"));

        // Look for formals of type pointer to an array.
        TY_IDX ty_idx_formal = ST_type(st_formal);
        if (TY_kind(ty_idx_formal) != KIND_POINTER) continue;
        TY_IDX ty_idx_pformal = TY_pointed(ty_idx_formal);
        if (TY_kind(ty_idx_pformal) != KIND_ARRAY) continue;

        /* Register the formal's projected region in the entry node. */

        PROJECTED_REGION *pr = Projected_Region_From_St(wn_func, st_formal, 
                mem_pool, FALSE, NULL);

        Cfg_entry->Add_formal_array(pr,
                TY_size(TY_etype(ty_idx_pformal)),
                Summary->Get_symbol_index(st_formal),
                idx_formal_first + i);
    }
}


/*****************************************************************************
 *
 * Walk through all the DO_LOOPs in the control structure, and build the loop
 * information.
 *
 ****************************************************************************/

static void process_loops()
{
    MEM_POOL *apool = Array_Summary.Get_array_pool();
    SUMMARY_CONTROL_DEPENDENCE *cd = NULL;

    while ((cd = Get_next_cd()) != NULL)
    {
        if (!cd ->Is_do_loop()) continue;

        // Get LNO-like loop information.
        DO_LOOP_INFO_BASE *dli = 
            (DO_LOOP_INFO_BASE*)WN_MAP_Get(IPL_info_map, cd->Get_wn());

        INT cd_idx = Get_cd_idx(cd);

        // Map it into summary LOOPINFO structure.
        LOOPINFO *l = CXX_NEW(LOOPINFO(apool, cd_idx), apool);
        l->Map_do_loop_info(dli);

        // Store it as part of the cfg node information in Array_Summary.
        Array_Summary.Get_cfg_node_array(cd_idx)->Set_loopinfo(l);
    }
}

//--------------------------------------------------------------------------
// record the tlog information
//--------------------------------------------------------------------------
static void
Record_tlog(TLOG_INFO *tlog)
{
  Ipl_record_tlog("LTKIND_CONST", 0, "Count %d", tlog->Get_cterm_count());
  Ipl_record_tlog("LTKIND_LINDEX", 0, "Count %d", tlog->Get_lterm_count());
  Ipl_record_tlog("LTKIND_IV_GLOBAL", 0,"Count %d",tlog->Get_iv_gterm_count());
  Ipl_record_tlog("LTKIND_IV", 0, "Count %d",tlog->Get_iv_term_count());
  Ipl_record_tlog("LTKIND_SUBSCR", 0, "Count %d",tlog->Get_sub_term_count());
}

//-----------------------------------------------------------------------
// NAME: Has_Threadprivate_Variable
// FUNCTION: Return TRUE if the tree rooted at 'wn_tree' has a THREAD_PRIVATE
//   variable.  Return FALSE otherwise. 
//-----------------------------------------------------------------------

static BOOL Has_Threadprivate_Variable(WN* wn_tree)
{
  if (WN_operator(wn_tree) == OPR_BLOCK) { 
    for (WN* wn = WN_first(wn_tree); wn != NULL; wn = WN_next(wn))
      if (Has_Threadprivate_Variable(wn))
	return TRUE; 
  } else { 
    if (OPERATOR_has_sym(WN_operator(wn_tree)) && WN_st(wn_tree) != NULL
        && (ST_base(WN_st(wn_tree)) != WN_st(wn_tree)
        && ST_sclass(ST_base(WN_st(wn_tree))) == SCLASS_COMMON
        && ST_is_thread_private(ST_base(WN_st(wn_tree)))
        || ST_is_thread_private(WN_st(wn_tree)))) {
      return TRUE; 
    } 
    for (INT i = 0; i < WN_kid_count(wn_tree); i++) 
      if (Has_Threadprivate_Variable(WN_kid(wn_tree, i)))
	return TRUE; 
  } 
  return FALSE; 
} 


/*****************************************************************************
 *
 * Map access vectors to linex and loopinfo structures in the given function.
 * The main entry for collecting local array summary info.
 *
 ****************************************************************************/

void IPL_Access_Vector_To_Projected_Region(WN* w, SUMMARY_PROCEDURE* proc,
        INT pu_first_formal_idx, INT pu_last_formal_idx,
        INT pu_first_actual_idx, INT pu_last_actual_idx,
        INT pu_first_callsite_idx, INT pu_last_callsite_idx)
{
    WN_ITER *wni;
    WN* wn;
    INT i;
    CFG_NODE_INFO *cfg_node = NULL, *cfg_node_else = NULL;

    FmtAssert(w != NULL,
            ("NULL node in IPL_Access_Vector_To_Proj_Region\n"));

    INT max_cd_size = Get_max_cd_idx();

    INT pu_formal_count = pu_last_formal_idx - pu_first_formal_idx + 1;
    INT pu_actual_count = pu_last_actual_idx - pu_first_actual_idx + 1;
    INT pu_callsite_count = pu_last_callsite_idx - pu_first_callsite_idx + 1;

    Array_Summary.Init(pu_formal_count, pu_first_formal_idx, 
            pu_actual_count, pu_first_actual_idx, 
            pu_callsite_count, pu_first_callsite_idx, 
            max_cd_size + 1);

    WB_IPL_Set_Array_Summary(&Array_Summary);

    // Create an array of cfg_info nodes of the size of the control
    // dependence array.
    CFG_NODE_INFO_ARRAY *cfg_nodes = Array_Summary.Get_cfg_node_array();

    // If there is an entry node, then the size will always at least be 1.

    /* Count the number of IF stmts. */
    INT if_count = 0;
    for (i = 0; i <= max_cd_size; ++i) {
        SUMMARY_CONTROL_DEPENDENCE* cd = Get_cd_by_idx(i);
        if (cd->Is_if_stmt()) ++if_count;
    }

    if (max_cd_size == -1) {
        proc->Set_has_incomplete_array_info();
        return;
    }

    // For each IF we will create an ELSE node at the end.
    cfg_nodes->Force_Alloc_array(max_cd_size + 1 + if_count);
    cfg_nodes->Setidx(max_cd_size + if_count);
    INT else_idx = max_cd_size;

    INT *map = Array_Summary.Get_cd_map();

    for (i = 0; i <= max_cd_size; ++i)
    {
        /* Load the content of a CD summary node to a CFG_NODE_INFO in the
         * array summary.
         */
        SUMMARY_CONTROL_DEPENDENCE *d = Get_cd_by_idx(i);
        cfg_node = &(*cfg_nodes)[i];
        cfg_node->Init(Array_Summary.Get_array_pool());

        // This is the index into the CD summary array.
        INT real_idx = Get_cd_real_idx(d);
        // store real CD index in the cfg_node
        cfg_node->Set_cd_index(real_idx);

        // mapping from the output cds to the input cds where 
        // real_idx is the output index and i is the input index
        // map is indexed relative to the beginning of PU
        map[real_idx - proc->Get_ctrl_dep_index()] = i;

        if (d->Is_if_stmt())
        {
            // printf("cd[%d] = %s \n", i, "if node ");
            cfg_node->Set_type_if();
            cfg_node->Set_else_index(++else_idx);

            // Init the ELSE control dependence.
            cfg_node_else = &(*cfg_nodes)[else_idx];
            cfg_node_else->Init(Array_Summary.Get_array_pool());
            cfg_node_else->Set_type_else();
            cfg_node_else->Set_if_index(i);

            if (Get_cd_call_count(i, TRUE) > 0) cfg_node->Set_has_calls();
            if (Get_cd_call_count(i, FALSE) > 0) {
                cfg_node_else->Set_has_calls();
            }
        }
        else
        {
            if (d->Is_do_loop()) {
                cfg_node->Set_type_do_loop();
            } else {
                cfg_node->Set_type_entry();
                Cfg_entry = cfg_node;
                Cfg_entry_idx = i;
            }

            if (Get_cd_call_count(i) > 0) cfg_node->Set_has_calls();
        }
    }

    if (Cfg_entry == NULL) {
        proc->Set_has_incomplete_array_info();
        return;
    }

    Current_summary_procedure = proc;

    // Initialize loop info and access array maps.
    MEM_POOL *apool = Array_Summary.Get_array_pool();
    IPL_Loopinfo_Map = CXX_NEW(LOOPINFO_TO_DLI_MAP(64,apool), apool);
    IPL_Access_Array_Map = CXX_NEW(
            PROJ_REGION_TO_ACCESS_ARRAY_MAP(128,apool), apool);

    // Process loop structures, map the loop bounds to linexs.
    process_loops();

    // Check for threadprivate variables.
    if (Has_Threadprivate_Variable(w)) { 
        proc->Set_has_incomplete_array_info();
        return;
    }

    // Walk through each statement node in the function.
    for (wni = WN_WALK_StmtIter(w); wni != NULL && WN_ITER_wn(wni) != 0;
            wni = WN_WALK_StmtNext(wni)) {
        wn = WN_ITER_wn(wni);
        process_node(wn, IPA_UNKNOWN);
    }

    /* Save dimensions of array formals for reshaping analysis. */
    Process_Array_Formals(w, pu_first_formal_idx, pu_formal_count,
            Array_Summary.Get_array_pool());

    Cfg_entry = NULL;
    Cfg_entry_idx = -1;
}

//------------------------------------------------------------------------
// finalize projected regions
//------------------------------------------------------------------------
void IPL_Finalize_Projected_Regions(SUMMARY_PROCEDURE *p)
{
    INT term_offset = 0;

    if (Get_Trace(TP_PTRACE1, TP_PTRACE1_IPA)) {
        term_offset = Array_Summary_Output->Get_term_offset();
    }

    Map_asections(&Array_Summary, p);

    if (Get_Trace(TP_PTRACE1, TP_PTRACE1_IPA)) {
        Array_Summary.Record_tlogs(Array_Summary_Output->Get_term_array(),
                term_offset + 1);
        Record_tlog(Array_Summary.Get_tlog_info());
    }

    Array_Summary.Finalize();
}

