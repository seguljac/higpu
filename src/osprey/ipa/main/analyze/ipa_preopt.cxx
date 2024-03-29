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
// ====================================================================
// ====================================================================
//
// Module: ipa_preopt.cxx
// $Revision: 1.1.1.1 $
// $Date: 2005/10/21 19:00:00 $
// $Author: marcel $
// $Source: /proj/osprey/CVS/open64/osprey1.0/ipa/main/analyze/ipa_preopt.cxx,v $
//
// Revision history:
//  7-15-98 - Original Version
//
// Description:
//   Infrastructure for selectively calling PREOPT from IPA
//
// ====================================================================
// ====================================================================

#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <dlfcn.h>                      // sgidladd, dlerror

#ifndef BACK_END
#define BACK_END                        // config.h needs it
#endif
#include "be_symtab.h"                  // BE_ST
#include "config.h"                     // Run_preopt, Run_ipl
#include "dso.h"                        // load_so
#include "erglob.h"                     // ErrMsg
#include "glob.h"                       // Show_Progress
#include "ir_reader.h"                  // fdump_tree
#include "ipa_cg.h"                     // IPA_NODE, IPA_EDGE, IPA_CALL_GRAPH
#include "ipa_option.h"                 // IPA_TRACE_* flags
#include "ipa_summary.h"                // IPA_get_*_file_array 
#include "ipa_section_annot.h"          // SECTION_FILE_ANNOT
#include "ipl_array_bread_write.h"      // ARRAY_SUMMARY_OUTPUT
#include "ipl_main.h"                   // Do_Par
#include "ipl_driver.h"                 // Ipl_Init_From_Ipa
#define IPA_SUMMARY                     // SUMMARIZE<IPL>
#include "ipl_summarize.h"              // SUMMARY
#undef IPA_SUMMARY
#include "ipo_const.h"                  // IPO_propagate_globals
#include "ipo_defs.h"                   // IPA_NODE_CONTEXT
#include "optimizer.h"                  // Pre_Optimizer
#include "region_main.h"                // REGION_Initialize
#include "ipa_main.h"

/** DAVID CODE BEGIN **/
#include "wn_util.h"
#include "cxx_hash.h"

#include "cuda_utils.h"

// #include "ipo_parent.h"
// #include "ipa_preopt.h"

extern void (*transform_ptr_access_to_array_p)(WN*,
        DU_MANAGER*, ALIAS_MANAGER*);
#define transform_ptr_access_to_array (*transform_ptr_access_to_array_p)
/*** DAVID CODE END ***/

// --- from wopt.so
#pragma weak Create_Du_Manager
#pragma weak Delete_Du_Manager
#pragma weak Pre_Optimizer

// --- from ipl.so
#pragma weak Array_Summary_Output
#pragma weak Do_Par
#pragma weak Ipl_Init_From_Ipa
#pragma weak Summary
#pragma weak Trace__20ARRAY_SUMMARY_OUTPUTGP8__file_s
#pragma weak Trace__31SUMMARIZE__pt__14_XC7PROGRAML10GP8__file_s


class ST_IDX_PAIR_TO_INT32_HASH_TABLE
{
private:
  struct HASH_ELEMENT 
  {
    ST_IDX _symbol_st_idx;
    ST_IDX _func_st_idx;
    INT32  _symbol_index;
    HASH_ELEMENT* _next;
  } *_table[0x40];

  MEM_POOL* _mem_pool;

  UINT32 hash(ST_IDX st_idx, ST_IDX func_st_idx) const 
  {
    return ((ST_IDX_index(st_idx) | ST_IDX_index(func_st_idx)) & 0xff) >> 2;
  }

public:
  ST_IDX_PAIR_TO_INT32_HASH_TABLE (MEM_POOL* mem_pool) :
    _mem_pool (mem_pool)
  {
    bzero(_table, sizeof(_table));
  }
 
  void Enter (ST_IDX symbol_st_idx, ST_IDX func_st_idx, INT32 symbol_index) 
  {
    HASH_ELEMENT* p = CXX_NEW (HASH_ELEMENT, _mem_pool);
    UINT32 hash_value = hash (symbol_st_idx, func_st_idx);
    p->_symbol_st_idx = symbol_st_idx;
    p->_func_st_idx = func_st_idx;
    p->_symbol_index = symbol_index;
    p->_next = _table[hash_value];
    _table[hash_value] = p;
  }

  INT32 Find (ST_IDX symbol_st_idx, ST_IDX func_st_idx) const 
  {
    UINT32 hash_value = hash (symbol_st_idx, func_st_idx);
    HASH_ELEMENT* p = _table[hash_value];
    while (p) {
      if (p->_symbol_st_idx == symbol_st_idx && 
          p->_func_st_idx == func_st_idx) {
        return p->_symbol_index;
      }
      p = p->_next;
    }
    return -1;
  }
    
}; 


struct AUX_FILE_INFO
{
  ST_IDX_PAIR_TO_INT32_HASH_TABLE* symbol_index_map;
  INT32 max_symbol_size;
};


static MEM_POOL IPA_preopt_pool;
static BOOL IPA_preopt_initialized = FALSE;
static AUX_FILE_INFO* Aux_file_info;


/*****************************************************************************
 *
 * Build a hash-table to speed up the mapping of preopt-generated symbol
 * indices into the old file-based SYMMARY_SYMBOL array
 *
 ****************************************************************************/

static void IPA_build_symbol_index_map(const IPA_NODE *node)
{
    AUX_FILE_INFO& file_info = Aux_file_info[node->File_Index()];
    // Do this only once per file, rather than for each node.
    if (file_info.symbol_index_map != NULL) return;

    file_info.symbol_index_map = CXX_NEW(
            ST_IDX_PAIR_TO_INT32_HASH_TABLE(&IPA_preopt_pool),
            &IPA_preopt_pool);

    // Get the file-based SYMMARY_SYMBOL array.
    INT32 num_symbols;
    SUMMARY_SYMBOL *symbols = IPA_get_symbol_file_array(
            node->File_Header(), num_symbols);

    // Each file-based symbol is based on its st_idx and that of its enclosing
    // function (0 for globals).
    for (INT32 i = 0; i < num_symbols; ++i)
    {
        ST_IDX symbol_st_idx = symbols[i].St_idx();
        ST_IDX func_st_idx = (ST_IDX_level(symbol_st_idx) == GLOBAL_SYMTAB) ?
            0 : symbols[i].Get_st_idx_func();
        file_info.symbol_index_map->Enter(symbol_st_idx, func_st_idx, i);
    }
}
  

/*****************************************************************************
 *
 * Add given symbol to the array of SUMMARY_SYMBOLS in the file that contains
 * given node and adjust summary header information
 *
 ****************************************************************************/

static INT32 IPA_add_new_symbol(const IPA_NODE* node,
        const SUMMARY_SYMBOL& symbol)
{
    AUX_FILE_INFO& file_info = Aux_file_info[node->File_Index()];
    IP_FILE_HDR& file_hdr = node->File_Header();
    SUMMARY_FILE_HEADER *summary_header = IP_FILE_HDR_file_header(file_hdr);

    INT32 num_syms;
    SUMMARY_SYMBOL *old_symbols =
        IPA_get_symbol_file_array(file_hdr, num_syms);
    SUMMARY_SYMBOL* new_symbols;
    INT32 num_bytes = num_syms * sizeof(SUMMARY_SYMBOL);

    // If max_symbol_size is 0, we are extending the array for the first time
    if (file_info.max_symbol_size == 0)
    {
        file_info.max_symbol_size = num_syms * 2;
        new_symbols = (SUMMARY_SYMBOL*)MEM_POOL_Alloc(Malloc_Mem_Pool,
                num_bytes * 2);
        memcpy(new_symbols, old_symbols, num_bytes);
        Elf64_Word new_offset = summary_header->Get_symbol_offset() +
            ((char*) new_symbols - (char*) old_symbols);
        summary_header->Set_symbol_offset(new_offset);
    }
    // Reallocating when the array is extended more than once
    else if (file_info.max_symbol_size <= num_syms)
    {
        file_info.max_symbol_size = num_syms * 2;
        new_symbols = (SUMMARY_SYMBOL*)MEM_POOL_Realloc(Malloc_Mem_Pool,
                old_symbols, num_bytes, num_bytes*2);
        Elf64_Word new_offset = summary_header->Get_symbol_offset() +
            ((char*) new_symbols - (char*) old_symbols);
        summary_header->Set_symbol_offset(new_offset);
    }
    else
    {
        new_symbols = old_symbols;
    }

    new_symbols[num_syms] = symbol;
    summary_header->Set_symbol_size(num_syms + 1);

    return num_syms;
}


/*****************************************************************************
 *
 * Given a new, preopt-generated SUMMARY_SYMBOL, find the index of the
 * matching symbol in the old file-based SUMMARY_SYMBOL array.
 *
 ****************************************************************************/

static INT32 IPA_map_symbol_index(const IPA_NODE* node,
        const SUMMARY_SYMBOL& symbol)
{
    AUX_FILE_INFO& file_info = Aux_file_info[node->File_Index()];
    Is_True(file_info.symbol_index_map != NULL,
            ("Hash table for symbol to index mapping is not initialized"));

    ST_IDX symbol_st_idx = symbol.St_idx();
    ST_IDX func_st_idx = (ST_IDX_level(symbol_st_idx) == GLOBAL_SYMTAB) ?
        0 : symbol.Get_st_idx_func();

    // Look-up the symbol using its st_idx and that of the enclosing function
    // if not found, add it to the SUMMARY_SYMBOL array and enter in the
    // table.
    INT32 sym_idx = file_info.symbol_index_map->Find(
            symbol_st_idx, func_st_idx);
    if (sym_idx == -1) {
        sym_idx = IPA_add_new_symbol(node, symbol); 
        file_info.symbol_index_map->Enter(
                symbol_st_idx, func_st_idx, sym_idx);
    }

    return sym_idx;
}
  

/*****************************************************************************
 *
 * Update symbol indices in preopt-generated SUMMARY_FORMALs so that they
 * point into file-based SUMMARY_SYMBOL array.
 *
 * Update formal indices in SUMMARY_SYMBOL file array so that they point to
 * preopt-generated SUMMARY_FORMALs.
 *
 ****************************************************************************/

static void IPA_update_formal_symbol_indices(
        const IPA_NODE* node, SUMMARY* new_summary)
{
    // DAVID COMMENT: ???
    SUMMARY_FORMAL *old_formals = IPA_get_formal_array(node)
        + node->Summary_Proc()->Get_formal_index();

    SUMMARY_SYMBOL *new_symbols = new_summary->Get_symbol(0);
    SUMMARY_FORMAL *new_formals = new_summary->Get_formal(0);

    UINT num_formals = node->Num_Formals();
    for (UINT i = 0; i < num_formals; ++i)
    {
        INT32 new_sym_idx = new_formals[i].Get_symbol_index();
        if (new_sym_idx == -1) continue;

        INT32 old_sym_idx = IPA_map_symbol_index(node,
                new_symbols[new_sym_idx]);
        new_formals[i].Set_symbol_index(old_sym_idx);
        IPA_get_symbol_array(node)[old_sym_idx].Set_findex(i);
    }
}


// ---------------------------------------------------------
// Update symbol indices in preopt-generated SUMMARY_ACTUALs
// so that they point into file-based SUMMARY_SYMBOL array
// ---------------------------------------------------------
static void
IPA_update_actual_symbol_indices (const IPA_NODE* node,
                                  SUMMARY* new_summary)
{
  SUMMARY_FORMAL* old_formals = IPA_get_formal_array(node) + 
                                node->Summary_Proc()->Get_formal_index();

  SUMMARY_SYMBOL* new_symbols = new_summary->Get_symbol(0);
  SUMMARY_ACTUAL* new_actuals = new_summary->Get_actual(0);
  INT32 num_new_actuals = new_summary->Get_actual_idx() + 1;
  
  for (INT32 i = 0; i < num_new_actuals; ++i) {
    INT32 new_sym_idx = new_actuals[i].Get_symbol_index();
    if (new_sym_idx != -1) {
      INT32 old_sym_idx = IPA_map_symbol_index(node, new_symbols[new_sym_idx]);
      new_actuals[i].Set_symbol_index (old_sym_idx);
    }
  }
}


// ---------------------------------------------------------
// update symbol indices in preopt-generated SUMMARY_GLOBALs
// so that they point into file-based SUMMARY_SYMBOL array
// ---------------------------------------------------------
static void
IPA_update_global_symbol_indices (const IPA_NODE* node,
                                  SUMMARY* new_summary)
{
  SUMMARY_SYMBOL* new_symbols = new_summary->Get_symbol(0);
  SUMMARY_GLOBAL* new_globals = new_summary->Get_global(0);
  INT32 num_new_globals = new_summary->Get_global_idx() + 1;

  for (INT32 i = 0; i < num_new_globals; ++i) {
    INT32 new_sym_idx = new_globals[i].Get_symbol_index();
    Is_True (new_sym_idx != -1, ("Invalid symbol index in SUMMARY_GLOBAL"));
    INT32 old_sym_idx = IPA_map_symbol_index(node, new_symbols[new_sym_idx]);
    new_globals[i].Set_symbol_index (old_sym_idx);
  }
}

    
// -----------------------------------------------------------
// Update symbol indices in preopt-generated SUMMARY_CALLSITEs
// so that they point into file-based SUMMARY_SYMBOL array
// -----------------------------------------------------------
static void
IPA_update_callsite_symbol_indices (const IPA_NODE* node,
                                    SUMMARY* new_summary)
{
  SUMMARY_SYMBOL* new_symbols = new_summary->Get_symbol(0);
  SUMMARY_CALLSITE* new_callsites = new_summary->Get_callsite(0);
  INT32 num_new_callsites = new_summary->Get_callsite_idx() + 1;
  
  for (INT32 i = 0; i < num_new_callsites; ++i) {
    if (!new_callsites[i].Is_func_ptr()) {
      INT32 new_sym_idx = new_callsites[i].Get_symbol_index();
      Is_True(new_sym_idx != -1, ("Invalid symbol index in SUMMARY_CALLSITE"));
      INT32 old_sym_idx = IPA_map_symbol_index(node, new_symbols[new_sym_idx]);
      new_callsites[i].Set_symbol_index (old_sym_idx);
    }
  }
}


// -------------------------------------------------------
// Update symbol indices in preopt-generated SUMMARY_STIDs
// so that they point into file-based SUMMARY_SYMBOL array
// -------------------------------------------------------
static void
IPA_update_stid_symbol_indices (const IPA_NODE* node,
                                SUMMARY* new_summary)
{
  SUMMARY_SYMBOL* new_symbols = new_summary->Get_symbol(0);
  SUMMARY_STID* new_stids = new_summary->Get_global_stid(0);
  INT32 num_new_stids = new_summary->Get_global_stid_idx() + 1;
  
  for (INT32 i = 0; i < num_new_stids; ++i) {
    INT32 new_sym_idx = new_stids[i].Get_symbol_index();
    Is_True (new_sym_idx != -1, ("Invalid symbol index in SUMMARY_STID"));
    INT32 old_sym_idx = IPA_map_symbol_index(node, new_symbols[new_sym_idx]);
    new_stids[i].Set_symbol_index (old_sym_idx);
  }
}


// --------------------------------------------------------
// Update symbol indices in preopt-generated SUMMARY_VALUEs
// so that they point into file-based SUMMARY_SYMBOL array
// --------------------------------------------------------
static void
IPA_update_value_symbol_indices (const IPA_NODE* node,
                                 SUMMARY* new_summary)
{
  SUMMARY_SYMBOL* new_symbols = new_summary->Get_symbol(0);
  SUMMARY_VALUE* new_values = new_summary->Get_value(0);
  INT32 num_new_values = new_summary->Get_value_idx() + 1;
  
  for (INT32 i = 0; i < num_new_values; ++i) {
    if (new_values[i].Is_global()) {
      INT32 new_sym_idx = new_values[i].Get_global_index();
      if (new_sym_idx != -1) {
        INT32 old_sym_idx=IPA_map_symbol_index(node,new_symbols[new_sym_idx]);
        new_values[i].Set_global_index (old_sym_idx);
      }
    }
    else if (new_values[i].Is_symbol()) {
      INT32 new_sym_idx = new_values[i].Get_symbol_index();
      Is_True (new_sym_idx != -1, ("Invalid symbol index in SUMMARY_VALUE"));
      INT32 old_sym_idx = IPA_map_symbol_index(node, new_symbols[new_sym_idx]);
      new_values[i].Set_symbol_index (old_sym_idx);
    }
  }
}


// -------------------------------------------------------
// Update symbol indices in preopt-generated SUMMARY_CHIs
// so that they point into file-based SUMMARY_SYMBOL array
// -------------------------------------------------------
static void
IPA_update_chi_symbol_indices (const IPA_NODE* node,
                               SUMMARY* new_summary)
{
  SUMMARY_SYMBOL* new_symbols = new_summary->Get_symbol(0);
  SUMMARY_CHI* new_chis = new_summary->Get_chi(0);
  INT32 num_new_chis = new_summary->Get_chi_idx() + 1;
  
  for (INT32 i = 0; i < num_new_chis; ++i) {
    INT32 new_sym_idx = new_chis[i].Get_symbol_index();
    Is_True (new_sym_idx != -1, ("Invalid symbol index in SUMMARY_CHI"));
    INT32 old_sym_idx = IPA_map_symbol_index(node, new_symbols[new_sym_idx]);
    new_chis[i].Set_symbol_index (old_sym_idx);
  }
}


// ----------------------------------------------------------
// Update symbol indices in preopt-generated REGION_ARRAYS
// so that they point into file-based SUMMARY_SYMBOL array
// ----------------------------------------------------------

static void IPA_update_region_symbol_indices(const IPA_NODE* node,
        SUMMARY* new_summary, ARRAY_SUMMARY_OUTPUT* new_array_summary)
{
    // DAVID COMMENT: ???
    SUMMARY_FORMAL* old_formals = IPA_get_formal_array(node)
        + node->Summary_Proc()->Get_formal_index();

    SUMMARY_SYMBOL* new_symbols = new_summary->Get_symbol(0);
    REGION_ARRAYS* new_regions = new_array_summary->Get_region_array(0);
    INT32 num_new_regions = new_array_summary->Get_region_count() + 1;

    for (INT32 i = 0; i < num_new_regions; ++i)
    {
        INT32 new_sym_idx = new_regions[i].Get_sym_id();
        Is_True(new_sym_idx != -1, ("Invalid symbol index in REGION_ARRAYS"));
        INT32 old_sym_idx = IPA_map_symbol_index(node,
                new_symbols[new_sym_idx]);
        new_regions[i].Set_sym_id(old_sym_idx);
    }
}


// -------------------------------------------------------
// Update symbol indices in preopt-generated SCALAR_INFOs
// so that they point into file-based SUMMARY_SYMBOL array
// -------------------------------------------------------

static void IPA_update_scalar_symbol_indices(const IPA_NODE* node,
        SUMMARY* new_summary, ARRAY_SUMMARY_OUTPUT* new_array_summary)
{
    SUMMARY_SYMBOL* new_symbols = new_summary->Get_symbol(0);
    SCALAR_INFO* new_scalars = new_array_summary->Get_scalars(0);
    INT32 num_new_scalars = new_array_summary->Get_scalars_count() + 1;

    for (INT32 i = 0; i < num_new_scalars; ++i)
    {
        INT32 new_sym_idx = new_scalars[i].Get_id();
        Is_True(new_sym_idx != -1, ("Invalid symbol index in SCALAR_INFO"));
        INT32 old_sym_idx = IPA_map_symbol_index(node,
                new_symbols[new_sym_idx]);
        new_scalars[i].Set_id(old_sym_idx);
    }
}


/*****************************************************************************
 *
 * For preopt-generated TERMs update IVAR and IVAR_GLOBAL indices and their
 * symbol indices so that point into file-based arrays
 *
 ****************************************************************************/

static void IPA_update_terms(const IPA_NODE* node,
        ARRAY_SUMMARY_OUTPUT* new_array_summary,
        SECTION_FILE_ANNOT* section_file_annot)
{
    IVAR* new_ivars = new_array_summary->Get_ivar(0);
    TERM* new_terms = new_array_summary->Get_term(0);
    INT32 num_new_terms = new_array_summary->Get_term_count() + 1;

    for (INT32 i = 0; i < num_new_terms; ++i)
    {
        if (new_terms[i].Get_type() != LTKIND_IV) continue;
        
        const IVAR& ivar = new_ivars[new_terms[i].Get_desc()];
        INT ivar_idx = section_file_annot->Find_ivar(node, ivar);
        if (ivar_idx == -1) {
            ivar_idx = section_file_annot->Add_ivar(node, ivar);
        }

        new_terms[i].Set_desc(ivar_idx);
    }
}


/*****************************************************************************
 *
 * Change indices and counts in SUMMARY_PROCEDURE so that they point into
 * preopt-generated summary info arrays.
 *
 ****************************************************************************/

static void IPA_update_procedure(IPA_NODE *node, SUMMARY *new_summary)
{
    // Update the symbol index in the new SUMMARY_PROCEDURE.
    SUMMARY_SYMBOL* new_symbols = new_summary->Get_symbol(0);
    SUMMARY_PROCEDURE* new_proc = new_summary->Get_procedure(0);
    INT32 new_sym_idx = new_proc->Get_symbol_index();
    Is_True(new_sym_idx != -1,
            ("Invalid symbol index in SUMMARY_PROCEDURE"));
    INT32 old_sym_idx = IPA_map_symbol_index(node, new_symbols[new_sym_idx]);
    new_proc->Set_symbol_index(old_sym_idx);

#ifdef KEY
    new_proc->Set_size(PU_WN_BB_Cnt, PU_WN_Stmt_Cnt,
            new_proc->Get_call_count());
#endif

    // Copy everything from the new SUMMARY_PROCEDURE to the old one.
    *(node->Summary_Proc()) = *new_proc;
}


/*****************************************************************************
 *
 * One-time initialization needs to do the following:
 * - dlopen wopt.so and ipl.so
 * - initialize BE_SYMTAB resources
 * - initialize mempool for newly genereated summaries
 * - allocate and initialize auxiliary array (one elment per file)
 *
 ****************************************************************************/

/** DAVID CODE BEGIN **/
void IPA_Preopt_Initialize()
{
    if (IPA_preopt_initialized) return;
/*** DAVID CODE END ***/

    load_so("wopt.so", WOPT_Path, Show_Progress);
    load_so("ipl.so", Ipl_Path, Show_Progress);

    MEM_POOL_Initialize(&IPA_preopt_pool, "IPA preopt pool", FALSE);
    MEM_POOL_Push(&IPA_preopt_pool);

    UINT32 bytes = IP_File_header.size() * sizeof(AUX_FILE_INFO);
    Aux_file_info = (AUX_FILE_INFO*)MEM_POOL_Alloc(&IPA_preopt_pool, bytes);
    bzero(Aux_file_info, bytes);

/** DAVID CODE BEGIN **/
    IPA_preopt_initialized = TRUE;
/*** DAVID CODE END ***/
}


/*****************************************************************************
 *
 * One-time cleanup after preopt has been called on one or more nodes:
 * - free BE_SYMTAB resources
 * - eliminate functions that have become dead after preopt
 *
 ****************************************************************************/

void IPA_Preopt_Finalize()
{
    if (!IPA_preopt_initialized) return;

    // delete dead functions, but do not update modref counts
    // for global variables (651823)
    Eliminate_Dead_Func(FALSE);

    MEM_POOL_Pop(&IPA_preopt_pool);
    MEM_POOL_Delete(&IPA_preopt_pool);

    IPA_preopt_initialized = FALSE;
}

#ifdef KEY
// Computes PU size after calling preopt
// Modifies: PU_WN_BB_Cnt, PU_WN_Stmt_Cnt, PU_WN_Call_Cnt
// Modeled after Count_tree_size()
static void
Compute_PU_Size (WN * wn)
{
  if (!wn) return;

  WN * wn2;

  switch (WN_operator(wn)) {

    case OPR_BLOCK:
      wn2 = WN_first(wn);
      while (wn2) {
        Compute_PU_Size(wn2);
        wn2 = WN_next(wn2);
      }
      break;

    case OPR_REGION:
      Compute_PU_Size(WN_region_body(wn));
      break;

    case OPR_IF:
      Compute_PU_Size(WN_if_test(wn));
      if (WN_then(wn))
        Compute_PU_Size(WN_then(wn));
      if (WN_else(wn))
        Compute_PU_Size(WN_else(wn));
      break;

    case OPR_DO_LOOP:
      Compute_PU_Size(WN_start(wn));
      Compute_PU_Size(WN_step(wn));
      Compute_PU_Size(WN_end(wn));
      Compute_PU_Size(WN_do_body(wn));
      break;

    case OPR_WHILE_DO:
    case OPR_DO_WHILE:
      Compute_PU_Size(WN_while_test(wn));
      Compute_PU_Size(WN_while_body(wn));
      break;

    case OPR_SWITCH:
    case OPR_COMPGOTO:
    case OPR_XGOTO:
      {
      WN *targ_blk = WN_kid1(wn);
      wn2 = WN_first(targ_blk);
      INT t = WN_num_entries(wn) - 1;
      for ( ; t >= 0; --t, wn2 = WN_next(wn2) )
          Compute_PU_Size(wn2);
      break;
      }

    default:
      {
      INT i;
      for (i = 0; i < WN_kid_count(wn); i++) {
          wn2 = WN_kid(wn,i);
          if (wn2)
              Compute_PU_Size(wn2);
        }
      }
    }
    Count_WN_Operator (WN_operator (wn), WN_rtype (wn) , PU_WN_BB_Cnt, PU_WN_Stmt_Cnt, PU_WN_Call_Cnt);
}
#endif // KEY


/** DAVID CODE BEGIN **/

#if 0
/*****************************************************************************
 *
 * The map is from copy_wn to orig_wn.
 *
 ****************************************************************************/

static BOOL compare_and_hash_wn_tree(WN *orig_wn, WN *copy_wn,
        HASH_TABLE<WN*,WN*> *wn_map)
{
    if (orig_wn == NULL && copy_wn == NULL) return TRUE;
    if (orig_wn == NULL || copy_wn == NULL) return FALSE;

    OPERATOR opr = WN_operator(orig_wn);
    if (opr != WN_operator(copy_wn)) return FALSE;

    if (opr == OPR_LDID || opr == OPR_STID
        || opr == OPR_DO_LOOP || opr == OPR_FUNC_ENTRY
        || OPERATOR_is_call(opr))
    {
        wn_map->Enter(copy_wn, orig_wn);
    }

    /* Handle composite nodes. */
    if (opr == OPR_BLOCK) {
        WN *o = WN_first(orig_wn), *c = WN_first(copy_wn);
        while (o != NULL) {
            if (!compare_and_hash_wn_tree(o, c, wn_map)) return FALSE;
            o = WN_next(o);
            c = WN_next(c);
        }
    } else {
        INT nkids = WN_kid_count(orig_wn);
        for (INT k = 0; k < nkids; k++) {
            if (!compare_and_hash_wn_tree(
                        WN_kid(orig_wn,k), WN_kid(copy_wn,k), wn_map)) return FALSE;
        }
    }

    return TRUE;
}

static void fix_du_info(DU_MANAGER *du_mgr, HASH_TABLE<WN*,WN*> *wn_htl)
{
    WN *copy_wn, *orig_wn;

    // Go through each pair of WN nodes.
    HASH_TABLE_ITER<WN*,WN*> it(wn_htl);
    while (it.Step(&copy_wn, &orig_wn))
    {
        OPERATOR opr = WN_operator(copy_wn);

        if (opr == OPR_LDID)
        {
            // Go through the def list.
            DEF_LIST *def_list = du_mgr->Ud_Get_Def(copy_wn);
            DEF_LIST_ITER dli(def_list);
            for (DU_NODE *def = dli.First(); !dli.Is_Empty(); def = dli.Next())
            {
                WN *orig_def_wn = wn_htl->Find(def->Wn());
                Is_True(orig_def_wn != NULL,
                        ("fix_du_info: incomplete WN htable\n"));
                def->set_wn(orig_def_wn);
            }

            // Move this def list from the copy node to the original node.
            du_mgr->Ud_Put_Def(copy_wn, NULL);
            du_mgr->Ud_Put_Def(orig_wn, def_list);
        }
        else if (opr == OPR_STID || opr == OPR_FUNC_ENTRY
                || OPERATOR_is_call(opr))
        {
            // Go through the use list.
            USE_LIST *use_list = du_mgr->Du_Get_Use(copy_wn);
            USE_LIST_ITER uli(use_list);
            for (DU_NODE *use = uli.First(); !uli.Is_Empty(); use = uli.Next())
            {
                WN *orig_use_wn = wn_htl->Find(use->Wn());
                Is_True(orig_use_wn != NULL,
                        ("fix_du_info: incomplete WN htable\n"));
                use->set_wn(orig_use_wn);
            }

            // Move this use list from the copy node to the original node.
            du_mgr->Du_Put_Use(copy_wn, NULL);
            du_mgr->Du_Put_Use(orig_wn, use_list);
        }

        // We do not worry about replacing _loop_stmt in DEF_LIST.
    }
}

/*****************************************************************************
 *
 * Invoke the preoptimizer on the given procedure node, to obtain the data
 * flow and alias information (allocated using <pool>).
 *
 * IPA_NODE_CONTEXT must be created before calling this function.
 *
 ****************************************************************************/

void IPA_get_du_info(IPA_NODE *node, MEM_POOL *pool,
        DU_MANAGER **du_mgr, ALIAS_MANAGER **alias_mgr)
{
    Is_True(!node->Is_Quasi_Clone(),
            ("Quasi-clones cannot be preoptimized!"));

    // We cannot call preopt on nested PUs because IPL requires that their
    // parent PUs be processed first (652328)
    if (node->Is_Nested_PU() || node->Summary_Proc()->Is_alt_entry()) return;

    // Do one-time initialization if necessary.
    IPA_Preopt_Initialize();

    // We optimize on a copy of the WN tree.
    WN *wn = node->Whirl_Tree();
    WN *copy_wn = WN_COPY_Tree(wn);
    REGION_Initialize(copy_wn, PU_has_region(node->Get_PU()));

    MEM_POOL_Push(&MEM_local_pool);

    Set_Error_Phase("hiCUDA Data Flow Analysis");

    Run_preopt = TRUE;
    // We must not run IPL, as we did not call Ipl_Init_From_Ipa.
    Run_ipl = FALSE;

    BE_symtab_alloc_scope_level(CURRENT_SYMTAB);
    Scope_tab[CURRENT_SYMTAB].st_tab->
        Register(*Be_scope_tab[CURRENT_SYMTAB].be_st_tab);
    Scope_tab[CURRENT_SYMTAB].preg_tab->Register(Be_preg_tab);

    PU_adjust_addr_flags(Get_Current_PU_ST(), wn);

    *du_mgr = Create_Du_Manager(pool);
    *alias_mgr = Create_Alias_Manager(pool);

    copy_wn = Pre_Optimizer(PREOPT_IPA_KERNEL_DAS_PHASE,
            copy_wn, *du_mgr, *alias_mgr, NULL);
    // TODO: somehow this field is changed, so restore it.
    node->Set_Whirl_Tree(wn);

    Scope_tab[CURRENT_SYMTAB].preg_tab->Un_register(Be_preg_tab);
    Be_preg_tab.Clear();
    Scope_tab[CURRENT_SYMTAB].st_tab->
        Un_register(*Be_scope_tab[CURRENT_SYMTAB].be_st_tab);
    Be_scope_tab[CURRENT_SYMTAB].be_st_tab->Clear();

    // Compare the original and the new WHIRL tree, and construct a map
    // between corresponding nodes.
    HASH_TABLE<WN*,WN*> htl(307, &MEM_local_pool);
    Is_True(compare_and_hash_wn_tree(wn, copy_wn, &htl),
            ("IPA_get_du_info: the WN trees are different!\n"));

    // Fix the DU-chains so that they point to original WN nodes.
    fix_du_info(*du_mgr, &htl);
    (*du_mgr)->Set_Entry_Wn(wn);

    MEM_POOL_Pop(&MEM_local_pool);

    REGION_Finalize();
}
#endif


/*****************************************************************************
 *
 * The following two functions are the prelogue and epilogue of the original
 * IPA_Preoptimize.
 *
 ****************************************************************************/

static IPL_SUMMARY_PTRS* IPA_init_preopt_context(IPA_NODE *node)
{
    WN *wn = node->Whirl_Tree();

    if (Get_Trace(TP_IPA, IPA_TRACE_PREOPT_IPL)) {
        fprintf(TFile, "\n%s before preopt\n", node->Name());
        fdump_tree(TFile, wn);
    }

    // IPA preoptimization will destroy the entire tree and edges, so we must
    // reset the WN-to-IPA_EDGE map.
    IPA_Call_Graph->Reset_Callsite_Map(node);

    REGION_Initialize(wn, PU_has_region(node->Get_PU()));

    MEM_POOL_Push(&MEM_local_pool);

    // Initialize <Summary> and <Array_Summary_Output> in IPL.
    Ipl_Init_From_Ipa(Malloc_Mem_Pool);

    IPL_SUMMARY_PTRS *summary_ptrs = CXX_NEW(
            IPL_SUMMARY_PTRS(Summary, Array_Summary_Output), Malloc_Mem_Pool);

    BE_symtab_alloc_scope_level(CURRENT_SYMTAB);
    Scope_tab[CURRENT_SYMTAB].st_tab->Register(
            *Be_scope_tab[CURRENT_SYMTAB].be_st_tab);
    Scope_tab[CURRENT_SYMTAB].preg_tab->Register(Be_preg_tab);

    PU_adjust_addr_flags(Get_Current_PU_ST(), wn);

    return summary_ptrs;
}

static void IPA_fini_preopt_context(IPA_NODE *node,
        IPL_SUMMARY_PTRS *summary_ptrs, BOOL keep_edges = FALSE)
{
    WN *opt_wn = node->Whirl_Tree();

/** DAVID CODE BEGIN **/
    // Rebuild enum type constant in CUDA runtime calls.
    HC_rebuild_cuda_enum_type(opt_wn);
/*** DAVID CODE END ***/

    Scope_tab[CURRENT_SYMTAB].preg_tab->Un_register(Be_preg_tab);
    Be_preg_tab.Clear();
    Scope_tab[CURRENT_SYMTAB].st_tab->Un_register(
            *Be_scope_tab[CURRENT_SYMTAB].be_st_tab);
    Be_scope_tab[CURRENT_SYMTAB].be_st_tab->Clear();

    REGION_Finalize();

#ifdef KEY
    // Don't reset all PU stats
    PU_WN_BB_Cnt = PU_WN_Stmt_Cnt = PU_WN_Call_Cnt = 0;
    Compute_PU_Size(opt_wn);
#endif

    if (Get_Trace(TP_IPA, IPA_TRACE_PREOPT_IPL)) {
        fprintf(TFile, "\n%s after preopt\n", node->Name());
        fdump_tree(TFile, opt_wn);
        Summary->Trace(TFile);
        Array_Summary_Output->Trace(TFile);
    }

#if Is_True_On
    WN_verifier(opt_wn);
#endif

    if (Summary->Has_symbol_entry()) {
        IPA_build_symbol_index_map(node);
    }
    if (Summary->Has_formal_entry()) {
        IPA_update_formal_symbol_indices(node, Summary);
    }
    if (Summary->Has_actual_entry()) {
        IPA_update_actual_symbol_indices(node, Summary);
    }
    if (Summary->Has_global_entry()) {
        IPA_update_global_symbol_indices(node, Summary);
    }
    if (Summary->Has_callsite_entry()) {
        IPA_update_callsite_symbol_indices(node, Summary);
    }

#define _IPA_ITERATE_PREOPT_
#ifdef _IPA_ITERATE_PREOPT_
    if (Summary->Has_global_stid_entry()) {
        IPA_update_stid_symbol_indices(node, Summary);
    }
    if (Summary->Has_value_entry()) {
        IPA_update_value_symbol_indices(node, Summary);
    }
    if (Summary->Has_chi_entry()) {
        IPA_update_chi_symbol_indices(node, Summary);
    }
#endif // _IPA_ITERATE_PREOPT_

    SECTION_FILE_ANNOT *section_annot =
        IP_FILE_HDR_section_annot(node->File_Header());
    if (section_annot != NULL
            && Array_Summary_Output->Get_term_count() != -1) {
        IPA_update_terms(node, Array_Summary_Output, section_annot);
    }
    if (Array_Summary_Output->Get_region_count() != -1) {
        IPA_update_region_symbol_indices(node, Summary, Array_Summary_Output);
    }
    if (Array_Summary_Output->Get_scalars_count() != -1) {
        IPA_update_scalar_symbol_indices(node, Summary, Array_Summary_Output);
    }

    IPA_update_procedure(node, Summary);

    // Update the node info and its edges in the call graph.
    IPA_Call_Graph->Update_Node_After_Preopt(node, opt_wn,
            Summary->Has_callsite_entry() ? Summary->Get_callsite(0) : NULL,
            summary_ptrs, keep_edges);

    MEM_POOL_Pop(&MEM_local_pool);
}


/*****************************************************************************
 *
 * When constants are discovered in IPA, propagate them into PU, call PREOPT
 * to clean up things and rebuild array sections.
 *
 ****************************************************************************/

void IPA_Preoptimize(IPA_NODE *node)
{
    Is_True(!node->Is_Quasi_Clone(),
            ("Quasi-clones cannot be preoptimized!"));

    // We cannot call preopt on nested PUs because IPL requires that their
    // parent PUs be processed first (652328).
    if (node->Is_Nested_PU() || node->Summary_Proc()->Is_alt_entry()) return;

    // Do one-time initialization if necessary.
    IPA_Preopt_Initialize();

    IPA_NODE_CONTEXT context(node);

    // Generate constant assignment statements
    IPO_propagate_globals(node);            // globals
    IPA_Propagate_Constants(node, FALSE);   // formals

    IPL_SUMMARY_PTRS *summary_ptrs = IPA_init_preopt_context(node);

    Set_Error_Phase("IPA Global Optimizer");

    DU_MANAGER *du_mgr = Create_Du_Manager(MEM_pu_nz_pool_ptr);
    ALIAS_MANAGER *alias_mgr = Create_Alias_Manager(MEM_pu_nz_pool_ptr);

    Run_preopt = Run_ipl = TRUE;
#ifndef KEY
    Do_Par = TRUE;
#endif
    printf("Do_Par = %s\n", Do_Par ? "TRUE" : "FALSE");
    // Call the preopt, which then calls Perform_Procedure_Summary_Phase.
    Pre_Optimizer(PREOPT_IPA1_PHASE, node->Whirl_Tree(),
            du_mgr, alias_mgr, NULL);
    // Here, node->Whirl_Tree() points to the new procedure.

    Delete_Du_Manager(du_mgr, MEM_pu_nz_pool_ptr);
    Delete_Alias_Manager(alias_mgr, MEM_pu_nz_pool_ptr);

    IPA_fini_preopt_context(node, summary_ptrs);
}

/** DAVID CODE BEGIN **/

/*****************************************************************************
 *
 * Invoke <transform_ptr_access_to_array> and then the pre-optimizer to update
 * the SUMMARY data structures.
 *
 ****************************************************************************/

static BOOL lno_loaded = FALSE;

void HC_promote_dynamic_arrays(IPA_NODE *node)
{
    Is_True(!node->Is_Quasi_Clone(),
            ("Quasi-clones cannot be preoptimized!"));

    // We cannot call preopt on nested PUs because IPL requires that their
    // parent PUs be processed first (652328).
    if (node->Is_Nested_PU() || node->Summary_Proc()->Is_alt_entry()) return;

    // Do one-time initialization if necessary.
    IPA_Preopt_Initialize();
    if (! lno_loaded) {
        load_so("lno.so", LNO_Path, Show_Progress);
        lno_loaded = TRUE;
    }

    IPA_NODE_CONTEXT context(node);

    IPL_SUMMARY_PTRS *summary_ptrs = IPA_init_preopt_context(node);

    Set_Error_Phase("hiCUDA Pointer Promotion");
#if 0
    WN *func_wn = node->Whirl_Tree();
    INT n_formals = WN_num_formals(func_wn);
    for (INT i = 0; i < n_formals; ++i) {
        printf("formal[%d] type: %d\n", i,
                ST_type(WN_st(WN_formal(func_wn,i))));
    }
#endif
    DU_MANAGER *du_mgr = Create_Du_Manager(MEM_pu_nz_pool_ptr);
    ALIAS_MANAGER *alias_mgr = Create_Alias_Manager(MEM_pu_nz_pool_ptr);

    BOOL saved_run_preopt = Run_preopt;
    BOOL saved_run_ipl = Run_ipl;
    BOOL saved_do_par = Do_Par;

    // Get the DU-chains.
    Run_preopt = TRUE;
    Run_ipl = Do_Par = FALSE;
    Pre_Optimizer(PREOPT_IPA_PRE_PP_PHASE, node->Whirl_Tree(),
            du_mgr, alias_mgr, NULL);
    // Here, node->Whirl_Tree() points to the new procedure.

    // Do pointer promotion.
    transform_ptr_access_to_array(node->Whirl_Tree(), du_mgr, alias_mgr);

    Delete_Du_Manager(du_mgr, MEM_pu_nz_pool_ptr);
    Delete_Alias_Manager(alias_mgr, MEM_pu_nz_pool_ptr);

    du_mgr = Create_Du_Manager(MEM_pu_nz_pool_ptr);
    alias_mgr = Create_Alias_Manager(MEM_pu_nz_pool_ptr);

    // Run local summary phase again.
    Run_preopt = TRUE;
    Run_ipl = Do_Par = TRUE;
    Pre_Optimizer(PREOPT_IPA_POST_PP_PHASE, node->Whirl_Tree(),
            du_mgr, alias_mgr, NULL);
    // Here, node->Whirl_Tree() points to the new procedure.

    Run_preopt = saved_run_preopt;
    Run_ipl = saved_run_ipl;
    Do_Par = saved_do_par;

    Delete_Du_Manager(du_mgr, MEM_pu_nz_pool_ptr);
    Delete_Alias_Manager(alias_mgr, MEM_pu_nz_pool_ptr);

    IPA_fini_preopt_context(node, summary_ptrs);
}

/*****************************************************************************
 *
 * Determine data access summary for each kernel in <node>.
 *
 ****************************************************************************/

void HC_analyze_kernel_data(IPA_NODE *node)
{
    Is_True(!node->Is_Quasi_Clone(),
            ("Quasi-clones cannot be preoptimized!"));

    // We cannot call preopt on nested PUs because IPL requires that their
    // parent PUs be processed first (652328).
    if (node->Is_Nested_PU() || node->Summary_Proc()->Is_alt_entry()) return;

    // Do one-time initialization if necessary.
    IPA_Preopt_Initialize();

    IPA_NODE_CONTEXT context(node);

    IPL_SUMMARY_PTRS *summary_ptrs = IPA_init_preopt_context(node);

    Set_Error_Phase("hiCUDA Kernel Data Flow Analysis");

    DU_MANAGER *du_mgr = Create_Du_Manager(MEM_pu_nz_pool_ptr);
    ALIAS_MANAGER *alias_mgr = Create_Alias_Manager(MEM_pu_nz_pool_ptr);

    BOOL saved_run_preopt = Run_preopt;
    BOOL saved_run_ipl = Run_ipl;
    BOOL saved_do_par = Do_Par;

    Run_preopt = Run_ipl = Do_Par = TRUE;
    Pre_Optimizer(PREOPT_IPA_KERNEL_DAS0_PHASE, node->Whirl_Tree(),
            du_mgr, alias_mgr, node);
    // Here, node->Whirl_Tree() points to the new procedure.

    Run_preopt = saved_run_preopt;
    Run_ipl = saved_run_ipl;
    Do_Par = saved_do_par;

    Delete_Du_Manager(du_mgr, MEM_pu_nz_pool_ptr);
    Delete_Alias_Manager(alias_mgr, MEM_pu_nz_pool_ptr);

    IPA_fini_preopt_context(node, summary_ptrs, TRUE);
}

void HC_rebuild_kernel_scalar_das(IPA_NODE *node)
{
    Is_True(!node->Is_Quasi_Clone(),
            ("Quasi-clones cannot be preoptimized!"));

    // We cannot call preopt on nested PUs because IPL requires that their
    // parent PUs be processed first (652328).
    if (node->Is_Nested_PU() || node->Summary_Proc()->Is_alt_entry()) return;

    // Do one-time initialization if necessary.
    IPA_Preopt_Initialize();

    IPA_NODE_CONTEXT context(node);

    IPL_SUMMARY_PTRS *summary_ptrs = IPA_init_preopt_context(node);

    Set_Error_Phase("hiCUDA Kernel Scalar Data Flow Analysis");

    DU_MANAGER *du_mgr = Create_Du_Manager(MEM_pu_nz_pool_ptr);
    ALIAS_MANAGER *alias_mgr = Create_Alias_Manager(MEM_pu_nz_pool_ptr);

    BOOL saved_run_preopt = Run_preopt;
    BOOL saved_run_ipl = Run_ipl;
    BOOL saved_do_par = Do_Par;

    Run_preopt = Run_ipl = TRUE;
    // All we need to do is to disable array summary. During scalar summary,
    // the kernel's DAS gets updated.
    Do_Par = FALSE;
    node->set_collect_scalar_das_only();
    Pre_Optimizer(PREOPT_IPA_KERNEL_DAS1_PHASE, node->Whirl_Tree(),
            du_mgr, alias_mgr, node);
    // Here, node->Whirl_Tree() points to the new procedure.
    node->reset_collect_scalar_das_only();

    Run_preopt = saved_run_preopt;
    Run_ipl = saved_run_ipl;
    Do_Par = saved_do_par;

    Delete_Du_Manager(du_mgr, MEM_pu_nz_pool_ptr);
    Delete_Alias_Manager(alias_mgr, MEM_pu_nz_pool_ptr);

    IPA_fini_preopt_context(node, summary_ptrs);
}

void IPA_HC_rebuild_local_summary(IPA_NODE *node)
{
    Is_True(!node->Is_Quasi_Clone(),
            ("Quasi-clones cannot be preoptimized!"));

    // We cannot call preopt on nested PUs because IPL requires that their
    // parent PUs be processed first (652328).
    if (node->Is_Nested_PU() || node->Summary_Proc()->Is_alt_entry()) return;

    // Do one-time initialization if necessary.
    IPA_Preopt_Initialize();

    IPA_NODE_CONTEXT context(node);

    IPL_SUMMARY_PTRS *summary_ptrs = IPA_init_preopt_context(node);

    Set_Error_Phase("hiCUDA IPA Preoptimization");

    DU_MANAGER *du_mgr = Create_Du_Manager(MEM_pu_nz_pool_ptr);
    ALIAS_MANAGER *alias_mgr = Create_Alias_Manager(MEM_pu_nz_pool_ptr);

    BOOL saved_run_preopt = Run_preopt;
    BOOL saved_run_ipl = Run_ipl;
    BOOL saved_do_par = Do_Par;

    Run_preopt = TRUE; Run_ipl = TRUE;
    Do_Par = FALSE;
    Pre_Optimizer(PREOPT_IPA2_PHASE, node->Whirl_Tree(),
            du_mgr, alias_mgr, NULL);
    // Here, node->Whirl_Tree() points to the new procedure.

    Run_preopt = saved_run_preopt;
    Run_ipl = saved_run_ipl;
    Do_Par = saved_do_par;

    Delete_Du_Manager(du_mgr, MEM_pu_nz_pool_ptr);
    Delete_Alias_Manager(alias_mgr, MEM_pu_nz_pool_ptr);

    IPA_fini_preopt_context(node, summary_ptrs);
}

/*** DAVID CODE END ***/
