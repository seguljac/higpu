/*
 * Copyright (C) 2006, 2007. QLogic Corporation. All Rights Reserved.
 */

/* -*- c++ -*-
 *
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
/* ====================================================================
 * ====================================================================
 *
 * Module: ipl_summarize.cxx
 *
 * Description:
 *	operations for generating summary information.
 *
 * ====================================================================
 * ====================================================================
 */

#ifndef ipl_summarize_template_INCLUDED
#define ipl_summarize_template_INCLUDED

#ifndef wn_tree_util_INCLUDED
#include "wn_tree_util.h"
#endif

#ifndef ipl_summarize_util_INCLUDED
#include "ipl_summarize_util.h"
#endif // ipl_summarize_util_INCLUDED

#ifndef loop_info_INCLUDED
#include "loop_info.h"			// for Record_scalar_flow ()
#endif // loop_info_INCLUDED

#ifndef ipl_linex_INCLUDED
#include "ipl_linex.h"
#endif

#ifndef ipl_reorder_INCLUDED // for Ptr_to_ty_vector, local_cands
#include "ipl_reorder.h"
#endif

#include "region_util.h"                // for WN_Fake_Call_EH_Region
#include "wn_mp.h"                    // for WN_has_pragma_with_side_effect
#include "ipl_lno_util.h" 
#include "wb_ipl.h"

/** DAVID CODE BEGIN **/
#ifdef HICUDA
#include "hc_common.h"
#include "hc_utils.h"

#include "ipa_cg.h"
#include "ipa_hc_kernel.h"
#include "ipa_hc_gpu_data_prop.h"

#ifdef __linux__
extern void (*IPA_map_callsites_p)(IPA_NODE*, WN*, WN_TO_EDGE_MAP*);
#define IPA_map_callsites (*IPA_map_callsites_p)
extern void (*IPA_HC_match_gpu_data_with_kernel_das_p)(WN*,
        IPA_NODE*, MEM_POOL*);
#define IPA_HC_match_gpu_data_with_kernel_das \
    (*IPA_HC_match_gpu_data_with_kernel_das_p)
#else
#pragma weak IPA_map_callsites
#pragma weak IPA_HC_match_gpu_data_with_kernel_das
#endif  // __linux__
#endif  // HICUDA
/*** DAVID CODE END ***/


extern BOOL DoPreopt;
extern BOOL Do_Par;
extern BOOL Do_Common_Const;
extern ALIAS_MANAGER* Ipl_Al_Mgr;
extern DU_MANAGER *Ipl_Du_Mgr;
extern DYN_ARRAY<char*>* Ipl_Symbol_Names;
extern DYN_ARRAY<char*>* Ipl_Function_Names;

extern void Init_Chi_Phi_Hash_Tables(MEM_POOL*);
extern void IPL_Mark_Code(WN* func_nd);
extern void IPL_Build_Access_Vectors(WN* func_nd);
extern void Initialize_Access_Vals (DU_MANAGER*, FILE*);
extern void Finalize_Access_Vals();
extern void IPL_Initialize_Par_Code();
extern void IPL_Finalize_Par_Code();
extern void IPL_Finalize_Projected_Regions(SUMMARY_PROCEDURE *p);
extern DYN_ARRAY<char*>* Ipl_Symbol_Names;
extern DYN_ARRAY<char*>* Ipl_Function_Names; 

#ifdef KEY
static BOOL proc_has_pstatics = FALSE;
#endif

// helper functions
static inline BOOL
ST_is_formal(const ST* s)
{
  return (ST_sclass(s) == SCLASS_FORMAL || ST_sclass(s) == SCLASS_FORMAL_REF);
}

static inline BOOL
ST_is_common_block (const ST* st)
{
  return 
    ( ST_class(st) == CLASS_VAR &&
      ST_base(st) == st &&
      (ST_sclass(st) == SCLASS_COMMON || ST_sclass(st) == SCLASS_DGLOBAL) && 
      TY_kind(ST_type(st)) == KIND_STRUCT );
}

static inline BOOL
ST_is_common_element (const ST* st)
{
  return ( ST_class(st) == CLASS_VAR &&
           ST_base(st) != st &&
           ST_is_common_block(ST_base(st)) );
}

// Given an expression, find out if it contains any address of a symbol 
enum ADDR_TAKEN_ACTION
{
    RECORD_PASSED,			// set addr_passed bit for any LDA
    RECORD_SAVED,			// set addr_saved bit for any LDA
    RECORD_EXTERN			// set addr_passed bit if LDA is
					// for a symbol defined outside of
					// the current scope.
};

template <PROGRAM program>
static void
set_addr_taken_expr (const WN* expr, SUMMARIZE<program>* sum,
		     ADDR_TAKEN_ACTION action, BOOL is_iload)
{
  Is_True (OPERATOR_is_expression (WN_operator (expr)),
           ("Expecting an expression node"));

  WN_TREE_ITER<PRE_ORDER, const WN*> iter (expr);

  while (iter.Wn () != NULL) {
    const WN* wn = iter.Wn ();
    OPERATOR opr = WN_operator (wn);
	
    if (program == INLINER && ! OPERATOR_is_expression (opr)) {
      Set_Addr_Taken_Attrib (wn, sum);
      iter.WN_TREE_next_skip ();
      continue;
    }
	    
    if (OPERATOR_is_load (opr) && !OPERATOR_is_leaf (opr)) {
      // indirect load, ignore
      if (program == INLINER) {
        set_addr_taken_expr(WN_kid0 (wn), sum, action, TRUE);
        iter.WN_TREE_next_skip ();
        continue;
      }
      else {
        iter.WN_TREE_next_skip ();
        continue;
      }
    }

    if (!is_iload && opr == OPR_LDA) {
      ST* st = WN_st (wn);
      if (ST_class(st) == CLASS_VAR || ST_class(st) == CLASS_FUNC) {
        IPL_ST_INFO& st_info = Aux_Symbol(st);
        switch (action) {
        case RECORD_SAVED:
          // We don't want to set addr_saved if it is originally
          // not set in the symbol table.  However, it
          // ST_addr_saved is originally set but cleared by IPL,
          // we need to set it again.  This can happen when a
          // symbol's address is only saved by the nested PU, and 
          // IPL clears the addr_saved bit when processing the parent. 
          if (st_info.addr_saved_reset || ST_addr_saved (st)) {
            st_info.addr_saved = TRUE;
            Set_ST_addr_saved (st);
            if (st_info.summary_symbol_idx > -1) {
              sum->Get_symbol(st_info.summary_symbol_idx)->Set_addr_saved();
            }
            // propagate addr_saved atttribute from an element 
            // of a common block to the common block itself
            if (ST_is_common_element(st)) {
              ST* base_st = ST_base(st);
              IPL_ST_INFO& st_info = Aux_Symbol(base_st);
              st_info.addr_saved = TRUE;
              Set_ST_addr_saved(base_st);
              if (st_info.summary_symbol_idx > -1) {
                sum->Get_symbol(st_info.summary_symbol_idx)->Set_addr_saved(); 
              }
            }
          }
          break;

        case RECORD_EXTERN:
          if (ST_level (st) == CURRENT_SYMTAB)
            break;
          // fall through

        case RECORD_PASSED:
          st_info.addr_passed = TRUE;
          if (st_info.summary_symbol_idx > -1) {
            sum->Get_symbol(st_info.summary_symbol_idx)->Set_addr_passed();
          }
          // propagate addr_passed atttribute from an element 
          // of a common block to the common block itself
          if (ST_is_common_element(st)) {
            ST* base_st = ST_base(st);
            IPL_ST_INFO& st_info = Aux_Symbol(base_st);
            st_info.addr_passed = TRUE;
            Set_ST_addr_passed(base_st);
            if (st_info.summary_symbol_idx > -1) {
              sum->Get_symbol(st_info.summary_symbol_idx)->Set_addr_passed(); 
            }
          }
          break;
        }
      }
    }

    ++iter;
  }
} // set_addr_taken_expr


// walk the entire procedure and set the address taken attributes
template <PROGRAM program>
static void
Set_Addr_Taken_Attrib (const WN *proc_entry, SUMMARIZE<program>* sum)
{
    WN_TREE_ITER<PRE_ORDER, const WN*> iter (proc_entry);

    while (iter.Wn () != NULL) {
	const WN* wn = iter.Wn ();
	OPERATOR opr = WN_operator (wn);

	if (OPERATOR_is_store (opr)  || (opr == OPR_RETURN_VAL)) {
	    // TODO:  ignore stores to stack variables that are never
	    // passed out of the PU
	    set_addr_taken_expr (WN_kid0 (wn), sum, RECORD_SAVED, FALSE);
	    if (program == INLINER)
	        set_addr_taken_expr (WN_kid0 (wn), sum, RECORD_PASSED, FALSE);
	    iter.WN_TREE_next_skip ();
	    
	} else {
	    INT i;
	    switch (opr) {
#ifdef TODO
	    case OPR_ICALL:
		// Normally, we should ignore the function pointer of an
		// ICALL, in the same way we ignore LDA under ILOAD.
		// However, in the case of ICALL (LDA foo), we should
		// convert it to a direct call and also record this in the 
		// summary info as a direct call.  Until we do such a
		// conversion, we need to mark foo as address passed.
		// Otherwise DFE might delete foo if there is no other
		// direct call to foo.
		// By *NOT* handling ICALL in this switch statement, all
		// the PARM nodes will be handled by the OPR_PARM case
		// below, and the function pointer will be handled by the
		// OPR_LDA case below.
#endif // TODO
		
	    case OPR_IO_ITEM:
		for (i = 0; i < WN_kid_count (wn); ++i)
		    if (OPERATOR_is_expression (WN_operator (WN_kid (wn, i))))
			set_addr_taken_expr (WN_kid (wn, i), sum,
					     RECORD_PASSED, FALSE);
		iter.WN_TREE_next_skip ();
		break;
		
	    case OPR_PARM:
		set_addr_taken_expr (WN_kid0 (wn), sum, RECORD_PASSED, FALSE);
		iter.WN_TREE_next_skip ();
		break;

	    case OPR_LDA:
		if (ST_level (WN_st (wn)) != CURRENT_SYMTAB)
		    set_addr_taken_expr (wn, sum, RECORD_EXTERN, FALSE);
		++iter;
		break;

	    default:
		++iter;
	    }
	}
    }
} // Set_Addr_Taken_Attrib


struct update_symtab
{
#ifdef Is_True_On
    BOOL trace;

    update_symtab () : trace (Get_Trace (TP_IPL, TT_IPL_VERBOSE)) {}
    
#endif
    void operator() (UINT32, ST* st) const {
	IPL_ST_INFO& st_info = Aux_Symbol (st);
	if (st_info.addr_saved)
	    Set_ST_addr_saved (st);
	else if (ST_addr_saved (st) && !ST_has_nested_ref (st)) {
	    Clear_ST_addr_saved (st);
	    st_info.addr_saved_reset = TRUE;
#ifdef Is_True_On
	    if (trace)
		fprintf (TFile, "Clearing addr_saved for %s\n", ST_name (st));
#endif

	}

	if (st_info.addr_passed)
	    Set_ST_addr_passed (st);
	else if (ST_addr_passed (st) && !ST_has_nested_ref (st)) {
	    Clear_ST_addr_passed (st);
#ifdef Is_True_On
	    if (trace)
		fprintf (TFile, "Clearing addr_passed for %s\n", ST_name (st));
#endif
	}
#ifdef KEY // bug 11801
	if (ST_sclass(st) == SCLASS_PSTATIC)
	  proc_has_pstatics = TRUE;
#endif
    }
	    
}; // update_symtab


// scan all INITV for SYMOFF entries that specifies a symbol in
// CURRENT_SYMTAB
// this should be very rare, so we don't care about efficiency
static void
search_for_symoff_initv (INITV_IDX initv_idx)
{
    while (initv_idx) {
	const INITV& initv = Initv_Table[initv_idx];
	if (INITV_kind (initv) == INITVKIND_SYMOFF
#ifdef TARG_IA64
		 	|| INITV_kind (initv) == INITVKIND_SYMIPLT
#endif
			) {
	    ST_IDX st_idx = INITV_st (initv);
	    Aux_Symbol[st_idx].addr_saved = TRUE;
	} else if (INITV_kind (initv) == INITVKIND_BLOCK)
	    search_for_symoff_initv (INITV_blk (initv));
	initv_idx = INITV_next (initv);
    }
} // search_for_symoff_initv


struct search_for_static_initialization
{
    void operator() (UINT32, const INITO* inito) const {
	search_for_symoff_initv (INITO_val (*inito));
    }
};


// The preopt always set the ST_addr_saved bit too conservatively.  So we
// need to recompute them.
template <PROGRAM program>
void
Recompute_Addr_Taken (const WN *proc_entry, SUMMARIZE<program>* sum)
{
    // sometime, preopt or other backend phases might create a few new ST
    // entries 
    if (program == IPL) {
	for (SYMTAB_IDX i = GLOBAL_SYMTAB; i <= CURRENT_SYMTAB; ++i) {
	    if (ST_Table_Size (i) > Aux_Symbol_Info[i].size ()) {
		vector<IPL_ST_INFO>& aux_st = Aux_Symbol_Info[i];
		aux_st.insert (aux_st.end(),
			       ST_Table_Size (i) - aux_st.size (),
			       IPL_ST_INFO ());
	    }
	}
    }	

    // search for run-time address taken
    Set_Addr_Taken_Attrib (proc_entry, sum);

    // search for compile-time (static) address taken
    if (Scope_tab[CURRENT_SYMTAB].inito_tab->Size () > 1)
	For_all (Inito_Table, CURRENT_SYMTAB,
		 search_for_static_initialization ());

    // update(override) the symtab addr_taken attributes based on our own
    // analysis 
    //
    // KEY bug 11801: Also while we are traversing the local symbol
    // table, check for any pstatic symbol that may need to be
    // promoted in IPA. This step is required if that symbol is not
    // referenced in whirl.
    For_all (St_Table, CURRENT_SYMTAB, update_symtab ());
} // Recompute_Addr_Taken

static MEM_POOL Temp_pool;

template <PROGRAM program>
void SUMMARIZE<program>::Summarize(WN *w)
{
    static BOOL Temp_pool_initialized = FALSE;
    WN_MAP save_parent_map = Parent_Map;

    if (!Temp_pool_initialized) {
        Temp_pool_initialized = TRUE;
        MEM_POOL_Initialize(&Temp_pool, "temp pool", 0);
    }

    MEM_POOL_Popper pool(&Temp_pool);

    Init_Aux_Symbol_Info(CURRENT_SYMTAB);

#ifdef KEY
    proc_has_pstatics = FALSE;
#endif

    Recompute_Addr_Taken(w, this);

    if (!Has_alt_entry())
    {
        Parent_Map = WN_MAP_Create(&Temp_pool);
/** DAVID CODE BEGIN **/
        WN_Parentize(w, Parent_Map, Current_Map_Tab);
/*** DAVID CODE END ***/

        // init the global hash table;
        Global_hash_table = CXX_NEW(
                GLOBAL_HASH_TABLE(113,&Temp_pool), &Temp_pool);
        entry_cache = CXX_NEW(SUMMARY_ENTRY_CACHE(&Temp_pool), &Temp_pool);

        // initialize hash tables for translating CODEREP and PHI
        Chi_To_Idx_Map =
            CXX_NEW (CHI_CR_TO_INT_MAP (113, ptr_hash<CODEREP> (),
                        std::equal_to<CODEREP*> (), &Temp_pool),
                    &Temp_pool);
        Phi_To_Idx_Map =
            CXX_NEW (PHI_NODE_TO_INT_MAP (113, ptr_hash<PHI_NODE> (),
                        std::equal_to<PHI_NODE*> (), &Temp_pool),
                    &Temp_pool);
        Hashed_Chis = CXX_NEW(CHI_CR_ARRAY(&Temp_pool), &Temp_pool);
        Hashed_Phis = CXX_NEW(PHI_NODE_ARRAY(&Temp_pool), &Temp_pool); 
        Num_Chis_On_PU_Start = Get_chi_idx()+1; 
        Num_Phis_On_PU_Start = Get_phi_idx()+1;   

        // map for reducing duplicat entries for jump function descriptions
        Summary_Map = WN_MAP32_Create (&Temp_pool);

        if (DoPreopt) {
            Init_cdg(&Temp_pool);
            // map for traversing the ctrl dep and stmt structures.
            if (Do_Par) Stmt_Map = WN_MAP32_Create(&Temp_pool);
        }
    }

    Set_entry_point(w);
    Process_procedure(w);

    // If the original subroutine contained alternate entry points then we
    // need to create summary procedure nodes for each alternate entry point.
    // The entry points occuring in the subroutine are stored in a list which
    // is generated during the original summary procedure node process routine
    // The global variable Do_Altentry is also set in the process routine if
    // any entry points are encountered. Note, we need to perform a preorder
    // walk from the entry node onwards hence we will invoke the tree walker
    // for the original node and walk from there onwards.
    if (Has_alt_entry()) {
        for (INT i = 0; i <= Get_alt_entry_idx(); i++) {
            ALT_ENTRY *alt_entry = Get_alt_entry (i);
            w = alt_entry->Get_wn ();
            if ( Trace_IPA  || Show_Progress) {

                fprintf (TFile, "Summarizing alternate entry point %s \n",
                        ST_name(WN_st(w)));
            }

            INT position = alt_entry->Get_position();
            INT formal_count = alt_entry->Get_formal_count();
            Summary->Process_alt_procedure (w, position, formal_count);
        }
        _alt_entry.Free_array();
        _alt_entry.Resetidx();
    }

#ifdef KEY
    if (Get_Trace(TP_IPL, TT_IPL_SUMMARY)) Trace(TFile);
#endif

    // clean up the maps
    if (!Has_alt_entry())
    {
        WN_MAP_Delete(Parent_Map);
        Parent_Map = save_parent_map;
        WN_MAP_Delete(Summary_Map);
        if (DoPreopt && Do_Par) WN_MAP_Delete(Stmt_Map);
    }
} // SUMMARIZE<program>::Summarize


//-------------------------------------------------------------------
// DESCR:
// Set addr taken bits for global symbols
//-------------------------------------------------------------------
template <PROGRAM program>
struct process_compile_time_addr_saved
{
    SUMMARIZE<program>* const sum;
    vector<IPL_ST_INFO>& aux_st_info; 

    process_compile_time_addr_saved (SUMMARIZE<program>* const s) :
	sum (s), aux_st_info (Aux_Symbol_Info[GLOBAL_SYMTAB]) {}

    void operator() (UINT32, const INITV* initv) const {
	if (INITV_kind (*initv) != INITVKIND_SYMOFF
#ifdef TARG_IA64
			&& INITV_kind (*initv) != INITVKIND_SYMIPLT
#endif
			)
	    return;
	ST_IDX st_idx = INITV_st (*initv);
	if (ST_IDX_level (st_idx) != GLOBAL_SYMTAB)
	    return;
	const ST& st = St_Table[st_idx];
	if (ST_sym_class (st) != CLASS_VAR &&
	    ST_sym_class (st) != CLASS_FUNC)
	    return;
	
	IPL_ST_INFO& st_info = aux_st_info[ST_IDX_index (st_idx)];
	st_info.addr_saved = TRUE;
    }
}; // process_compile_time_addr_saved

#ifdef KEY
// bug 2954
#include <sys/param.h> // MAXPATHLEN
#include <unistd.h>
static unsigned int
hash_of_full_path_name (int index)
{
  bool malloced = false;
  // taken from driver/file_utils.c
  // Seems there is a memory leak in this code in file_utils.c
  char *cwd = getcwd((char *) NULL, MAXPATHLEN);
  if (cwd == NULL)
  {
    cwd = getenv("PWD");
    if (cwd == NULL)
    {
      // can't get path, we will see if it fails because of this. 
      cwd = ".";
    }
  }
  else malloced = true;

  char fullname[10 + strlen(cwd) + strlen(Src_File_Name) + 1 + 1];
  sprintf (fullname, "%d%s/%s", index, cwd, Src_File_Name);
  __gnu_cxx::hash<char*> hfn;

  if (malloced) free (cwd);

  return (unsigned int) hfn (fullname);
}

//bug# 555
template <PROGRAM program>
struct set_local_static_to_global
{
    SUMMARIZE<program>* const sum;
    const vector<IPL_ST_INFO>& aux_st_info;
                                                                                                                                                             
    set_local_static_to_global (SUMMARIZE<program>* const s) :
        sum (s),
        aux_st_info (Aux_Symbol_Info[GLOBAL_SYMTAB]) {}
                                                                                                                                                             
    void operator() (UINT32 idx, ST* st) const {
                                                                                                                                                             
        if (ST_class (st) != CLASS_VAR ||
            ST_sclass (st) != SCLASS_FSTATIC ||
            !ST_initv_in_other_st(st))
          return;
                                                                                                                                                             
        if (!ST_is_initialized (st))
          Set_ST_sclass(st, SCLASS_COMMON);
        else if (ST_init_value_zero(st))
          Set_ST_sclass(st, SCLASS_UGLOBAL);
        else
          Set_ST_sclass(st, SCLASS_DGLOBAL);
                                                                                                                                                             
        Set_ST_export(st, EXPORT_PREEMPTIBLE);
                                                                                                                                                             
        char idname[30];
	
        sprintf(idname, "_%d_%u", ST_index(st), hash_of_full_path_name(ST_index(st)));
        int newname = Save_Str2(ST_name(st), idname);
        Set_ST_name (st, newname);
    }
};
#endif

template <PROGRAM program>
struct set_global_addr_taken_attrib
{
#ifdef Is_True_On
    BOOL trace;
#endif
    SUMMARIZE<program>* const sum;
    const vector<IPL_ST_INFO>& aux_st_info;

    set_global_addr_taken_attrib (SUMMARIZE<program>* const s) :
	sum (s),
#ifdef Is_True_On
	trace (Get_Trace (TP_IPL, TT_IPL_VERBOSE)),
#endif
	aux_st_info (Aux_Symbol_Info[GLOBAL_SYMTAB]) {}

    void operator() (UINT32 idx, ST* st) const {

#ifdef _LIGHTWEIGHT_INLINER
	if (INLINE_Inlined_Pu_Call_Graph && (idx >= aux_st_info.size()))
	  return;	// This symbol was added by the inliner
#endif
        // skip special name holder symbols
        if (ST_class (st) == CLASS_NAME) 
          return;

	const IPL_ST_INFO& st_info = aux_st_info[idx];

	// update the addr_taken attributes in the symbol table
	if (st_info.addr_saved)
	    Set_ST_addr_saved (st);
#ifdef _LIGHTWEIGHT_INLINER
	else if (!INLINE_Inlined_Pu_Call_Graph && ST_addr_saved (st)) {
#else // _LIGHTWEIGHT_INLINER
	else if (ST_addr_saved (st)) {
#endif // _LIGHTWEIGHT_INLINER
	    Clear_ST_addr_saved (st);
#ifdef Is_True_On
	    if (trace)
		fprintf (TFile, "Clearing addr_saved for %s\n", ST_name (st));
#endif
	}

	if (st_info.addr_passed)
	    Set_ST_addr_passed (st);
	else
	    Clear_ST_addr_passed (st);

	// string literals often have address taken, but we don't have
	// summary info for them
	if (ST_class (st) == CLASS_CONST)
	    return;

	// now, update the summary_symbol

	if (st_info.addr_saved || st_info.addr_passed) {
	    UINT sym_idx = st_info.summary_symbol_idx;

	    Is_True (ST_class (st) == CLASS_VAR ||
		     ST_class (st) == CLASS_FUNC,
		     ("symbol is neither variable nor function"));

	    if (sym_idx == (UINT32) -1)
		sym_idx = sum->Get_symbol_index(st);

	    SUMMARY_SYMBOL *symbol = sum->Get_symbol (sym_idx);
	    
	    // record addr taken attributes for global symbols
	    if (st_info.addr_saved)
		symbol->Set_addr_saved ();

	    if (st_info.addr_passed)
		symbol->Set_addr_passed();
	}
    }
}; //  set_global_addr_taken_attrib

template <PROGRAM program>
void
SUMMARIZE<program>::Set_global_addr_taken_attrib (void)
{
    // set addr_saved for symbols whose addresses are used in compile-time
    // initialization 
    For_all (Initv_Table, process_compile_time_addr_saved<program> (this));

    // now scan all global symbols and update the SUMMARY_SYMBOL entry as
    // well as the ST attributes.
    For_all (St_Table, GLOBAL_SYMTAB,
	     set_global_addr_taken_attrib<program> (this));
#ifdef KEY
    if (program == IPL)
      For_all (St_Table, GLOBAL_SYMTAB, set_local_static_to_global<program> (this));
#endif
} // Set_global_addr_taken_attrib


//-----------------------------------------------------------
// walk the OPR_PARM opcode
//-----------------------------------------------------------
template <PROGRAM program>
void
SUMMARIZE<program>::Update_Addr_Passed_Count (WN* opr_parm)
{
  WN_ITER *wni;
  INT i;
  const ST* s = NULL;

  // walk the tree
  for ( wni = WN_WALK_TreeIter(opr_parm); wni && WN_ITER_wn(wni) != 0;
       wni = WN_WALK_TreeNext(wni) ) {

    WN* w = WN_ITER_wn(wni);
    SUMMARY_SYMBOL *symbol;

    // nested calls within COMMA nodes: ignore LDA under
    // inner calls: LDA will be counted for the inner calls once only
    if ((program == INLINER) &&
	(WN_operator(w) == OPR_CALL))
      {
	WN_WALK_Abort(wni);
	return;
      }
    if (WN_operator(w) != OPR_LDA)
      continue;

    s = WN_st(w);

    if (ST_class(s) == CLASS_CONST)
      continue;
	
    symbol = Get_symbol (Get_symbol_index (s));

    // don't do this for symbols that are part of a common block
    // in fortran 
    if (symbol->Is_local () && !symbol->Is_common())
      symbol->Incr_addr_count ();
    
    if ( Trace_CopyProp ) 
      fprintf ( TFile, "\n  Addr passed count for  %s updated to %d\n",
	       ST_name(s), symbol->Get_addr_count ());

	
    // handle splitting of common blocks
    if (ST_st_idx (s) == ST_base_idx (s) && ST_sclass (s) != SCLASS_COMMON)
      continue;
	
    if (ST_st_idx (s) != ST_base_idx (s)) {
      const ST* st = ST_base (s);
      if (ST_sclass (st) == SCLASS_COMMON) {
	// now enter it as a parameter only if the entry being
	// passed is NOT a 1 dimensional array
	// check the following:
	// 1. if the parent is an OPR_PARM node
	// 2. if it a 1-d array that is being passed
	WN* parent_w = LWN_Get_Parent(w);

	OPERATOR opc = WN_operator(parent_w);

	if (opc == OPR_PARM || opc == OPR_ARRAY) {
	  if (TY_kind (ST_type (s)) == KIND_ARRAY &&
	      TY_kind (ST_type (st)) == KIND_STRUCT) {
	    const TY& array_type = Ty_Table[ST_type (s)];
	    if (ARB_dimension (TY_arb (array_type)) == 1) {
	      WN_WALK_Abort (wni);
	      return;
	    }
	  }
	}

	s = st;
      }
    }


    i = Get_symbol_index (s);
    Get_symbol (i)->Set_parm ();
  }
} // SUMMARIZE::Update_Addr_Passed_Count


//-----------------------------------------------------------------
// add the address attributes to the symbols array
//-----------------------------------------------------------------
template <PROGRAM program>
inline void
SUMMARIZE<program>::Set_local_addr_taken_attrib ()
{
    typedef vector<IPL_ST_INFO> AUX_INFO;
    const AUX_INFO& aux_info = Aux_Symbol_Info[CURRENT_SYMTAB];

    for (AUX_INFO::const_iterator first = aux_info.begin ();
	 first != aux_info.end (); ++first) {

	const IPL_ST_INFO& st_info = *first;

	if (st_info.summary_symbol_idx == -1)
	    continue;

	if (st_info.addr_saved || st_info.addr_passed) {
	    SUMMARY_SYMBOL* symbol = Get_symbol (st_info.summary_symbol_idx);
	    if (st_info.addr_saved)
		symbol->Set_addr_saved ();
	    if (st_info.addr_passed)
		symbol->Set_addr_passed ();
	}
    }
} // Set_local_addr_taken_attrib


//-----------------------------------------------------------
// Get the symbol_crefcount_index of the st
// if not found create a new symbol_crefcount and return its index
//-----------------------------------------------------------

template <PROGRAM program>
INT 
SUMMARIZE<program>::Get_symbol_crefcount_index(INT idx) 
{
    FmtAssert(program == INLINER, 
      ("Get_symbol_crefcount_index can be called only from INLINER"));

    // the symbol in question IS a formal
    INT maxidx;
    SUMMARY_CREF_SYMBOL *sym;

    if (Has_symbol_crefcount_entry ()) { 
	INT i = 0;
	maxidx = Get_symbol_crefcount_idx ();
	while (i <= maxidx) {
	    if (Get_symbol_crefcount(i)->Get_symbol_index() == idx)
		return i;
	    else
		i++;
	} // while i <= maxidx
    } // if has_crefcount_entry
  
    // Either there are no crefcount_entries or this one wasn't found yet
    // create a NEW _symbol_crefcount
    sym = New_symbol_crefcount();  
    sym->Set_symbol_index(idx);
    return Get_symbol_crefcount_idx ();
}

//-----------------------------------------------------------
// If st is found return its symbol_crefcount_index
// else return -1
//-----------------------------------------------------------


template <PROGRAM program>
INT 
SUMMARIZE<program>::Find_symbol_crefcount_index(INT idx) 
{
  FmtAssert(program == INLINER,
    ("Find_symbol_crefcount_index can be called only from INLINER"));

// the symbol in question IS a formal
  INT i = 0;
  INT maxidx;
  if (Has_symbol_crefcount_entry ()) { 
    maxidx = Get_symbol_crefcount_idx ();
    while (i <= maxidx) {
      if (Get_symbol_crefcount(i)->Get_symbol_index() == idx)
	return i;
      else
	i++;
    } // while i < maxidx
  } // if has_crefcount_entry
  return -1; // symbol not found
}

//-----------------------------------------------------------
// If caller with st-index is found return its proc_info_index
// else return -1
//-----------------------------------------------------------


template <PROGRAM program>
INT 
SUMMARIZE<program>::Find_proc_info_index(INT32 idx) 
{
  FmtAssert(program == INLINER,
    ("Find_proc_info_index can be called only from INLINER"));

  INT i = 0;
  INT maxidx;
  if (Has_proc_info_entry ()) { 
    maxidx = Get_proc_info_idx ();
    while (i <= maxidx) {
      if (Get_proc_info(i)->Get_symbol_index() == idx)
	return i;
      else
	i++;
    } // while i < maxidx
  } // if has_proc_info_entry
  return -1; // symbol not found
}

//-----------------------------------------------------------
// return TRUE if parent of w is an ISTORE and w is the LHS
//-----------------------------------------------------------
static inline BOOL WN_is_istore_or_mstore(WN* w)
{
    WN* parent = LWN_Get_Parent(w);
    return (parent &&
            (WN_operator(parent) == OPR_ISTORE ||
             WN_operator(parent) == OPR_MSTORE) &&
            WN_kid1(parent) == w);
}

static BOOL WN_is_in_pragma (WN* wn_node)
{
    for (WN* wn = wn_node; wn != NULL; wn = LWN_Get_Parent(wn))  
        if (WN_operator(wn) == OPR_PRAGMA || WN_operator(wn) == OPR_XPRAGMA)
            return TRUE; 
    return FALSE; 
} 

// check if the current opcode is the last one in the tree
static inline BOOL
Last_Node (WN_TREE_ITER<PRE_ORDER, WN*> i)
{
    i.Skip ();
    return i.Wn () == NULL;
}

#ifdef KEY
// TODO: We don't need to replace the stidx with the summary table index.
//       We can just use the old-stidx -> new-stidx map in IPA.
template <PROGRAM program>
void
SUMMARIZE<program>::Process_eh_globals (void)
{
    if (!(PU_src_lang (Get_Current_PU()) & PU_CXX_LANG) || 
    	!Get_Current_PU().eh_info)
    	return;

    INITV_IDX i = INITV_next (INITV_next (INITO_val (Get_Current_PU().eh_info)));
    INITO_IDX idx = TCON_uval (INITV_tc_val(i));
    if (idx)	// typeinfo
    {
      INITO* ino = &Inito_Table[idx];
      INITV_IDX blk = INITO_val (*ino);
      do
      {
        INITV_IDX st_entry = INITV_blk (blk);
	ST_IDX st_idx = 0;
	if (INITV_kind (st_entry) != INITVKIND_ZERO)
	{
	  st_idx = TCON_uval (INITV_tc_val (st_entry));
	  FmtAssert (st_idx != 0, ("Invalid st idx"));
	}
	if (st_idx <= 0)
	{
	  blk = INITV_next (blk);
	  continue;
	}
	INT32 index = Get_symbol_index (&St_Table [st_idx]);
	INITV_IDX filter = INITV_next (st_entry); // for backup
	FmtAssert (index >= 0, ("Unexpected summary id for eh symbol"));
	INITV_Set_VAL (Initv_Table[st_entry], Enter_tcon (
	               Host_To_Targ (MTYPE_U4, index)), 1);
        Set_INITV_next (st_entry, filter);
	blk = INITV_next (blk);
      } while (blk);
    }

    i = INITV_next (i);
    idx = TCON_uval (INITV_tc_val (i));
    if (idx)	// eh-spec
    {
      INITO* ino = &Inito_Table[idx];
      INITV_IDX st_entry = INITV_blk (INITO_val (*ino));
      do
      {
	ST_IDX st_idx = 0;
	if (INITV_kind (st_entry) != INITVKIND_ZERO)
	{
          st_idx = TCON_uval (INITV_tc_val (st_entry));
	  FmtAssert (st_idx > 0, ("Invalid eh-spec entry"));
	}
	if (st_idx == 0)
	{
	  st_entry = INITV_next (st_entry);
	  continue;
	}
	INT32 index = Get_symbol_index (&St_Table[st_idx]);
	INITV_IDX next = INITV_next (st_entry); // for backup
	FmtAssert (index >= 0, ("Unexpected summary id for eh symbol"));
	INITV_Set_VAL (Initv_Table[st_entry], Enter_tcon (
	               Host_To_Targ (MTYPE_U4, index)), 1);
	Set_INITV_next (st_entry, next);
        st_entry = INITV_next (st_entry);
      } while (st_entry);
    }
}

template <PROGRAM program>
void
SUMMARIZE<program>::Process_eh_region (WN * wn)
{
    // !empty => try-region without any symbol worth summarizing.
    if (!WN_ereg_supp (wn) || !WN_block_empty (WN_region_pragmas (wn)))
    	return;
    FmtAssert (INITO_val (WN_ereg_supp (wn)) && 
    	       INITV_blk (INITO_val (WN_ereg_supp (wn))) &&
	       INITV_next (INITV_blk (INITO_val (WN_ereg_supp (wn)))),
	       ("No exception info attached to EH region"));
    INITV_IDX blk = INITO_val (WN_ereg_supp (wn));

    // Return if we have already summarized this block of initv's
    if (INITV_flags (Initv_Table[blk]) == INITVFLAGS_SUMMARIZED)
      return;

    Set_INITV_flags (blk, INITVFLAGS_SUMMARIZED);

    INITV_IDX types = INITV_next (INITV_blk (blk));
    for (; types; types = INITV_next (types))
    {
      int sym = 0;
      if (INITV_kind (types) != INITVKIND_ZERO)
        sym = TCON_uval (INITV_tc_val (types));
      if (sym > 0)
      {
      	INT32 index = Get_symbol_index (&St_Table[sym]);
	INITV_IDX next = INITV_next (types);	// for backup
	// We don't expect index==0 since at least Process_eh_globals is 
	// called before this.
	FmtAssert (index > 0, ("Unexpected summary id for eh symbol"));
	INITV_Set_VAL (Initv_Table[types], Enter_tcon (
		       Host_To_Targ (MTYPE_U4, index)), 1);
	Set_INITV_next (types, next);
      }
    }
}

#include <ext/hash_map>

namespace Local
{
  struct hashfn
  {
    size_t operator() (const WN * w) const
    {
      return reinterpret_cast<size_t>(w);
    }
  };

  struct eqnode
  {
    bool operator()(WN * w1, WN * w2) const
    {
      return w1 == w2;
    }
  };
};

struct branch_dir
{
  float taken, not_taken;
};
hash_map<WN*, branch_dir, Local::hashfn, Local::eqnode> if_map;

inline void get_parent_if ( WN ** p, WN ** b )
{
    WN * parent = *p;
    WN * block;
    while ( parent && WN_operator (parent) != OPR_IF )
    {
        if (WN_operator(parent) == OPR_BLOCK) block = parent;
        parent = LWN_Get_Parent(parent);
    }
    *p = parent;
    *b = block;
}

/**
 * Return the loop-nesting containing the node w. If it is inside 1 loop,
 * return 1, so on and so forth. The nesting level includes do-loop's, as well
 * as while-do's and do-while's.
 */
static inline INT get_loopnest(WN * w)
{
    WN *parent = LWN_Get_Parent(w);
    INT loopnest = 0;

    while (parent)
    {
        switch (WN_operator(parent))
        {
            case OPR_DO_LOOP:
#if !defined(_STANDALONE_INLINER) && !defined(_LIGHTWEIGHT_INLINER)
            {
                if (IPL_Ignore_Small_Loops)
                {
                    WN * loop_info = WN_do_loop_info(parent);
                    if (loop_info != NULL) {
                        // WN_loop_trip_est is just an estimate, and may only
                        // contain the value that can be held in mUINT16.
                        WN *trip = WN_loop_trip(loop_info);
                        if (trip && WN_operator(trip) == OPR_INTCONST) {
                            INT64 tripcount = WN_const_val(trip);
                            Is_True(tripcount >= 0,
                                    ("get_loopnest: negative loop trip count"));
                            if (tripcount <= IPL_Ignore_Small_Loops) break;
                        }
                    }
                }
            } // fall through
#endif // !_STANDALONE_INLINER && !_LIGHTWEIGHT_INLINER
            case OPR_WHILE_DO:
            case OPR_DO_WHILE:
                loopnest++;
                break;
            default:
                break;
        }

        parent = LWN_Get_Parent(parent);
    }

    return loopnest;
}
#endif

static BOOL is_variable_dim_array(TY_IDX ty)
{
    if (TY_kind(ty) == KIND_POINTER) ty = TY_pointed(ty);

    if (TY_kind(ty) != KIND_ARRAY) return FALSE;

    ARB_ITER arb_iter = Make_arb_iter(ARB_HANDLE(TY_arb(ty)));
    UINT dim = ARB_dimension(arb_iter);
    for (UINT i = 0; i < dim; ++i) {
        ARB_HANDLE arb(arb_iter);
        // Check the lower bound and the stride.
        if (!ARB_const_lbnd(arb) || !ARB_const_stride(arb)) return TRUE;
        // Check the upper bound.
        // C's array pointers written as a[] are not variable dim arrays.
        if (!ARB_const_ubnd(arb) && (ARB_ubnd_var(arb) != 0)) return TRUE;
        ++arb_iter;
    }

    return is_variable_dim_array(TY_etype(ty));
}

#ifdef KEY
// This function is similar to the versions available in ipc_bread.cxx
// and xstats.cxx. This function however is called by IPA summary phase
// and can potentially be smarter by analyzing the context of the WN node.
static void Count_WN(WN *wn, INT32& bbs, INT32& stmts, INT32& calls)
{
    OPERATOR opr = WN_operator(wn);
    TYPE_ID rtype = WN_rtype(wn);

    /* count nscf stmts as bbs, not stmts */
    if (OPERATOR_is_non_scf(opr)) {
        if (opr != OPR_RETURN && opr != OPR_RETURN_VAL) ++bbs;
    } else if (OPERATOR_is_stmt(opr)) {
        if (OPERATOR_is_call(opr)) {
            ++bbs;
            ++calls;
        } else if (opr == OPR_IO) {
            /* TODO:  ideally would look at values of IO_ITEMs,
             * but then have to pass more than opcode. */
            ++bbs;
            ++calls;
        } else if (! OPERATOR_is_not_executable(opr)) {
            ++stmts;
            if (MTYPE_is_complex(rtype) && OPERATOR_is_store(opr)) ++stmts;
        }
    } else if (OPERATOR_is_scf(opr)) {
        if (opr != OPR_BLOCK && opr != OPR_FUNC_ENTRY) {
            /* blocks are counted by parent node */
            ++bbs;
        }
        /* if may create two blocks if else present,
         * but can't tell just from opcode */
    } else if ((rtype == MTYPE_FQ || rtype == MTYPE_CQ) &&
            OPERATOR_is_expression(opr) &&
            !OPERATOR_is_load(opr) &&
            !OPERATOR_is_leaf(opr) ) {
        /* quad operators get turned into calls */
        ++bbs;
        ++calls;
    } else if (opr == OPR_CAND || opr == OPR_CIOR) {
        /* these may get expanded to if-then-else sequences,
         * or they may be optimized to logical expressions.
         * use the halfway average of 1 bb */
        ++bbs;
    }
}

// Keep track of whether use of label in WN has been seen.
class label_wn
{
    public:
        WN * wn;
        BOOL seen;
        label_wn() : wn (NULL), seen (FALSE) {}
};

#include <ext/hash_map>

// label equality
struct eq_oper
{
  bool operator() (INT i, INT j) const
  {
    return i == j;
  }
};

// Map from label number to label usage information
typedef hash_map<INT, label_wn, __gnu_cxx::hash<INT>, eq_oper> LABEL_WN_MAP;
#endif

/** DAVID CODE BEGIN **/
#ifdef HICUDA

template <PROGRAM program>
void SUMMARIZE<program>::process_symbol_in_kregion(WN *wn)
{
    Is_True(wn != NULL, (""));
    Is_True(_parent_kernel_map != WN_MAP_UNDEFINED, (""));

    OPERATOR opr = WN_operator(wn);
    Is_True(opr == OPR_LDID || opr == OPR_LDA || opr == OPR_STID, (""));

    // Skip pseudo registers.
    ST_IDX st_idx = WN_st_idx(wn);
    if (ST_class(st_idx) == CLASS_PREG) return;

    // Check if this symbol reference occurs within a kernel region.
    ST_IDX kfunc_st_idx = WN_MAP32_Get(_parent_kernel_map, wn);
    if (kfunc_st_idx == ST_IDX_ZERO) return;

    // Get the HC_KERNEL_INFO.
    HC_KERNEL_INFO *kinfo =
        _curr_proc_node->get_kernel_info_by_sym(kfunc_st_idx);
    Is_True(kinfo != NULL, (""));

    if (HCST_is_scalar(st_idx))
    {
        // This is an access of a scalar variable or a struct field.
        // TODO: is the handling of LDA correct?
        if (opr == OPR_LDID || opr == OPR_LDA)
        {
            // If at least one of its DEFs is outside the kernel
            // region, add this USE to the data access summary.
            DEF_LIST_ITER dli(du_mgr->Ud_Get_Def(wn));
            for (DU_NODE *def = dli.First(); !dli.Is_Empty(); def = dli.Next())
            {
                WN *def_wn = def->Wn();

                // We do not care if the variable is local and its DEF is the
                // program unit, because this occurs when a local variable is
                // not initialized in all paths.
                if (ST_sclass(St_Table[st_idx]) == SCLASS_AUTO
                        && WN_operator(def_wn) == OPR_FUNC_ENTRY) continue;

                // either outside any kernel region or inside some
                // different kernel region
                if (WN_MAP32_Get(_parent_kernel_map, def_wn)
                        != kfunc_st_idx) break;
            }

            if (!dli.Is_Empty())
            {
                kinfo->add_scalar(st_idx, WN_offset(wn), TRUE);
            }
        }

        if (opr == OPR_STID || opr == OPR_LDA)
        {
            // If at least one of its USEs is outside the kernel
            // region, add this DEF to the data access summary.
            USE_LIST_ITER uli(du_mgr->Du_Get_Use(wn));
            for (DU_NODE *use = uli.First(); !uli.Is_Empty(); use = uli.Next())
            {
                WN *use_wn = use->Wn();

                // either outside any kernel region or inside some
                // different kernel region
                if (WN_MAP32_Get(_parent_kernel_map, use_wn)
                        != kfunc_st_idx) break;
            }

            if (!uli.Is_Empty())
            {
                kinfo->add_scalar(st_idx, WN_offset(wn), FALSE);
            }
        }
    }
    else if (! _curr_proc_node->collect_scalar_das_only())
    {
        // This must be an array (or pointer-to-array) variable.
        HC_assert(HCST_is_array(st_idx),
                ("Failed to analyze symbol <%s> within kernel <%s>\n",
                 ST_name(st_idx), ST_name(kfunc_st_idx)));

        HC_assert(opr != OPR_STID,
                ("Assigning an array variable <%s> in kernel <%s> "
                 "is not allowed.\n",
                 ST_name(st_idx), ST_name(kfunc_st_idx)));

        if (opr == OPR_LDA)
        {
            Is_True(TY_kind(ST_type(st_idx)) == KIND_ARRAY, (""));
        }
        else
        {
            Is_True(TY_kind(ST_type(st_idx)) == KIND_POINTER, (""));
        }

        // This array access must be within an ILOAD/ISTORE/PARM node
        // or a pragma node.

        WN *p_wn;
        OPERATOR p_opr;
        for (p_wn = LWN_Get_Parent(wn); p_wn != NULL;
                p_wn = LWN_Get_Parent(p_wn))
        {
            p_opr = WN_operator(p_wn);
            if (p_opr == OPR_ILOAD || p_opr == OPR_ISTORE || p_opr == OPR_PARM
                    || p_opr == OPR_XPRAGMA || p_opr == OPR_PRAGMA) break;
        }

        if (p_wn == NULL)
        {

            HC_warn("Kernel <%s> references array variable <%s> in a place "
                    "that is neither an array access nor a parameter to a "
                    "function call.\n",
                    ST_name(kfunc_st_idx), ST_name(st_idx));

            kinfo->set_has_incomplete_array_info();
        }
        else if (p_opr == OPR_PARM)
        {
            WN *call_wn = LWN_Get_Parent(p_wn);
            Is_True(call_wn != NULL
                    && OPERATOR_is_call(WN_operator(call_wn)), (""));

            HC_assert(WN_kid0(p_wn) == wn && WN_offset(wn) == 0,
                    ("Array variable <%s> must be directly passed to "
                     "function <%s> in kernel <%s>.\n",
                     ST_name(st_idx), ST_name(WN_st_idx(call_wn))));
        }
        else if (p_opr == OPR_ILOAD || p_opr == OPR_ISTORE)
        {
            // Validate the address.
            WN *addr_wn = (p_opr == OPR_ILOAD) ?
                WN_kid0(p_wn) : WN_kid1(p_wn);

            if (WN_operator(addr_wn) == OPR_ARRAY)
            {
                // Make sure the offset is less than the element size.
                HC_assert(WN_offset(p_wn) < WN_element_size(addr_wn),
                        ("Invalid array access of <%s> in kernel <%s>\n",
                         ST_name(st_idx), ST_name(kfunc_st_idx)));
            }
            else
            {
                // Make sure that this address is <array symbol> + <offset>.
                // For now, we verify that by making sure each parent in the
                // middle of the chain is one of ADD, SUB, or CVT.
                WN *target_p_wn = p_wn;
                for (p_wn = LWN_Get_Parent(wn); p_wn != target_p_wn;
                        p_wn = LWN_Get_Parent(p_wn))
                {
                    p_opr = WN_operator(p_wn);
                    HC_assert(p_opr == OPR_ADD || p_opr == OPR_SUB
                            || p_opr == OPR_CVT,
                            ("Invalid array access of <%s> in kernel <%s>\n",
                             ST_name(st_idx), ST_name(kfunc_st_idx)));
                }
            }
        }
    }
}

/*****************************************************************************
 *
 * Clear all PROJECTED_REGION fields in:
 * - each HC_GPU_DATA (propagated and local)
 * - each HC_KERNEL_INFO
 *
 * This is necessary because all PROJECTED_REGION instances created in IPL
 * will be freed when Array_Summary.Finalize() is called.
 *
 ****************************************************************************/

template <PROGRAM program>
void SUMMARIZE<program>::clear_projected_regions()
{
    if (_curr_proc_node == NULL) return;

    // HC_GPU_DATA in each annotation
    IPA_HC_ANNOT_LIST *annots = _curr_proc_node->get_hc_annots();
    if (annots != NULL)
    {
        IPA_HC_ANNOT_ITER annot_iter(annots);
        for (IPA_HC_ANNOT *annot = annot_iter.First(); !annot_iter.Is_Empty();
                annot = annot_iter.Next())
        {
            HC_FORMAL_GPU_DATA_ARRAY *fgda =
                (HC_FORMAL_GPU_DATA_ARRAY*)annot->get_annot_data();
            if (fgda == NULL) continue;

            UINT n_formals = fgda->num_formals();
            for (UINT i = 0; i < n_formals; ++i)
            {
                HC_GPU_DATA *gdata = fgda->get_formal_data(i);
                if (gdata == NULL) continue;
                gdata->clear_proj_regions();
            }
        }
    }

    // local HC_GPU_DATA in the procedure
    HC_GPU_DATA_LIST *gdata_list = _curr_proc_node->get_gpu_data_list();
    UINT n_gdata = gdata_list->Elements();
    for (UINT i = 0; i < n_gdata; ++i) (*gdata_list)[i]->clear_proj_regions();

    // HC_KERNEL_INFO
    UINT n_kinfo = _curr_proc_node->num_kregions();
    if (n_kinfo > 0)
    {
        HC_KERNEL_INFO_LIST *kil = _curr_proc_node->get_kernel_info_list();
        for (UINT i = 0; i < n_kinfo; ++i) (*kil)[i]->clear_proj_regions();
    }
}

#endif  // HICUDA
/*** DAVID CODE END ***/

//-----------------------------------------------------------
// summary procedure node
// create call site entries
//-----------------------------------------------------------
template <PROGRAM program>
void SUMMARIZE<program>::Process_procedure(WN* w)
{
    SUMMARY_PROCEDURE *proc = New_procedure();
    WN* w2, *alt_wn;
    ST *st = WN_st (w);
#ifndef KEY
    INT loopnest = 0;
#endif // !KEY
    INT pu_first_formal_idx = 0;
    INT pu_last_formal_idx = 0;
    INT pu_first_actual_idx = 0;
    INT pu_first_callsite_idx = 0;
    BOOL Do_Altentry = Has_alt_entry();
    BOOL Do_parallel_stuff = Do_Par && !Do_Altentry && DoPreopt;
    BOOL Do_common_const = Do_Common_Const && !Do_Altentry && DoPreopt;
    BOOL Has_return_already = FALSE;
    BOOL Has_pdo_pragma = FALSE;
    BOOL Has_local_pragma = FALSE;
#ifdef KEY
    BOOL Do_reorder = (!Do_Altentry && Cur_PU_Feedback);
    LABEL_WN_MAP label_use_map;     
    INT icall_site[100];
    INT icall_cnt = 0;
#else
    BOOL Do_reorder = (!Do_Altentry && Cur_PU_Feedback&& IPA_Enable_Reorder);
    //and other things, such as Feedback_Enabled[PROFILE_PHASE_BEFORE_LNO]
#endif // KEY

    UINT fld_id,i, pop_loops;
    BOOL cur_pu_is_reorder_cand=FALSE;
    UINT64 loop_count;
    UINT stack_size; //just for debug
    WN* wn_tmp;
    UINT num_struct_access;//just for debug, ,Finish_PU_process_struct_access()

#ifdef KEY
    // Now that preopt has run, recompute PU size estimates
    Initialize_PU_Stats();

    // bug 11801
    if (proc_has_pstatics) proc->Set_has_pstatic();
#endif

    Trace_Modref = Get_Trace(TP_IPL, TT_IPL_MODREF);

    BOOL Direct_Mod_Ref = FALSE;
    if (!Do_parallel_stuff) proc->Set_has_incomplete_array_info();

    if (!WHIRL_Return_Val_On) proc->Set_use_lowered_return_preg();

    // check for the position in the actual index
    if (Do_parallel_stuff) {
        pu_first_formal_idx = Get_formal_idx() + 1;
        pu_first_actual_idx = Get_actual_idx() + 1;
        pu_first_callsite_idx = Get_callsite_idx() + 1;
    }

    if (WN_operator(w) == OPR_FUNC_ENTRY) {
        // Process all the file/PU scope pragmas.
        Process_pragmas(w);
        // Set the array of formals.
        INT n_formals = WN_num_formals(w);
        if (n_formals > 0) Process_formal(w, n_formals, proc);
    }

    if (Do_parallel_stuff) pu_last_formal_idx = Get_formal_idx();

    const PU& pu = Get_Current_PU();
    if (PU_mp_needs_lno(pu)) proc->Set_has_mp_needs_lno();

    // the index into the symbols array for the st to this procedure
    proc->Set_symbol_index(Get_symbol_index(st));

    if (Cur_PU_Feedback) { // was FB_PU_Has_Feedback
        FB_FREQ freq = (Cur_PU_Feedback->Query_invoke(w)).freq_invoke;

#ifdef KEY
        SUMMARY_FEEDBACK *fb = New_feedback ();
        proc->Set_feedback_index (Get_feedback_idx ());
#endif

        if (freq.Known()) {
            proc->Set_has_PU_freq ();
#ifndef KEY
            SUMMARY_FEEDBACK *fb = New_feedback ();
            proc->Set_feedback_index (Get_feedback_idx ());
#endif
            fb->Set_frequency_count (freq);
            //		printf("&&&&&&&&&&&&& %s -> %f\n",ST_name(WN_st(w)), fb->Get_frequency_count()._value);
        }
        else {
            // FB_PU_Has_Feedback = FALSE;
            DevWarn ("Unknown invoke frequency found in %s so no feedback info in this procedure will be considered", ST_name(WN_st(w)));
        }
#ifdef KEY
        // Runtime address of the current procedure based on feedback data.
        fb->Set_func_runtime_addr (Cur_PU_Feedback->Get_Runtime_Func_Addr());
#endif
    }
    else //INLINING_TUNING^
    {
        if(Feedback_Enabled[PROFILE_PHASE_BEFORE_VHO]) {
            proc->Set_Never_Invoked();
        }
    }//INLINING_TUNING$


    Set_lang(proc);

    if (PU_is_inline_function(pu)) proc->Set_may_inline();

    if (PU_no_inline(pu)) proc->Set_no_inline();

    if (PU_no_delete(pu)) proc->Set_no_delete();

    if (TY_is_varargs(Ty_Table[ST_pu_type(st)])) proc->Set_is_varargs();


    // set the index into the globals array
    proc->Set_global_index(Get_global_idx() + 1);


    // If Do_Par is set then build the access vectors, if condition
    // information and reductions.
    // TODO:
    // Map the access vectors the the CFG
    // Map the access vectors to LINEX structures

    // Use/Extend LNO's vector projection capabilities
    if (Do_parallel_stuff)
    {
        Ipl_Al_Mgr = Get_alias_mgr();	  
        Ipl_Du_Mgr = Get_du_mgr();
        IPL_Initialize_Par_Code();
        IPL_Mark_Code(w);
        WB_IPL_Set_Reduction_Map(IPL_reduc_map);
        Initialize_Access_Vals(Get_du_mgr(), TFile);
        IPL_Build_Access_Vectors(w);
        WB_IPL_Set_Access_Array_Map(IPL_info_map);
    }
    if (Do_reorder) Start_PU_process_struct_access();
    // if we are summarizing an alternate entry point then we must
    // search for the alternate entry point node in the subroutine.
    // and keep walking until we encounter it
    if (Do_Altentry) {
        alt_wn = w;
        w = Get_entry_point();
    }

    BOOL phi_index;

    // record initial phi node index
    if (DoPreopt) phi_index = Get_phi_idx();

#ifdef KEY
    Process_eh_globals();
#endif

    BOOL found = FALSE;

    // walk the tree
    for (WN_TREE_ITER<PRE_ORDER, WN*> iter(w); iter.Wn() != NULL; ++iter)
    {
        // If it is an alternate entry then we keep walking the tree until we
        // encounter the alternate entry point of interest to us.
        while (Do_Altentry && !found) {
            if ( iter.Wn () == alt_wn ) {
                found = TRUE;
            } else {
                ++iter;
            }
        }

        w2 = iter.Wn();
        st = OPERATOR_has_sym(WN_operator(w2)) ? WN_st(w2) : NULL;


        // get SUMMARY_SYMBOLs for a common block and its element
        if (st && (Do_Common_Const || Do_parallel_stuff)) {
            if (ST_is_common_element(st)) {
                Get_symbol_index(ST_base(st));
                Get_symbol_index(st);
            }
            else if (ST_is_common_block(st)) {
                Get_symbol_index(st);
            }
        }

        if (st && (is_variable_dim_array(ST_type(st)))) 
            proc->Set_has_var_dim_array();

#ifdef KEY
        // Recompute bb and statement count.
        Count_WN(w2, PU_WN_BB_Cnt, PU_WN_Stmt_Cnt, PU_WN_Call_Cnt);
#endif // KEY

        switch (WN_operator(w2))
        {
            case OPR_DO_LOOP:
#ifdef KEY
            {
                /* Check if a reference parameter is used as the loop index.
                 * If so, set the appropriate flag in the formal's summary.
                 */
                WN *index = WN_index(w2);
                Is_True(WN_operator(index) == OPR_IDNAME,
                        ("Process_procedure: Invalid do-loop index"));
                // do-loop index
                ST *id = WN_st(index);
                if (ST_sclass(id) == SCLASS_FORMAL_REF)
                {
                    INT32 id_sym_idx = Get_symbol_index(id);
                    INT i = 0, formal_id = proc->Get_formal_index();

                    for ( ; i < proc->Get_formal_count(); i++, formal_id++) {
                        SUMMARY_FORMAL *formal = Get_formal(formal_id);
                        if (id_sym_idx == formal->Get_symbol_index()) {
                            formal->Set_is_loop_index();
                            break;
                        }
                    }
                }
            } // fall-through
#endif

            case OPR_WHILE_DO:
            case OPR_DO_WHILE:
#ifndef KEY // disable buggy loopnest computation
                loopnest++;
#endif // !KEY
                if (Do_reorder) {
                    FB_Info_Loop fb_info = Cur_PU_Feedback->Query_loop(w2);
                    loop_count = (UINT64)fb_info.freq_iterate.Value();
                    loop_count_stack->Push(loop_count);
                    stack_size = loop_count_stack->Elements();  //for debug
                }

                break;

#ifdef KEY
            case OPR_IF:
            {
                // Remove the use of this flag in future
                if ( ! IPA_Enable_Branch_Heuristic || ! Cur_PU_Feedback ) break;

                FB_Info_Branch info = Cur_PU_Feedback->Query_branch ( w2 );
                if (!info.freq_taken.Known() || !info.freq_not_taken.Known())
                {
                    if_map[w2].taken = if_map[w2].not_taken = -1;
                    break;
                }
                float taken = info.freq_taken.Value() / info.Total().Value();
                float not_taken = info.freq_not_taken.Value() / info.Total().Value();
                // Check if we are inside another 'if', then adjust the 
                // probabilities.
                if ( !if_map.empty() )
                {
                    WN *block;
                    WN *parent = LWN_Get_Parent(w2);

                    get_parent_if ( &parent, &block );

                    if ( parent )
                    {
                        Is_True (block && WN_operator (block) == OPR_BLOCK, ("kid of if stmt wrong"));
                        // OPR_IF
                        if ( WN_kid1 (parent) == block )
                        {
                            taken *= if_map[parent].taken;
                            not_taken *= if_map[parent].taken;
                        }
                        else
                        {
                            Is_True ( WN_kid2 (parent) == block, ("kid of if stmt wrong"));
                            taken *= if_map[parent].not_taken;
                            not_taken *= if_map[parent].not_taken;
                        }
                    }
                }

                if_map[w2].taken = taken;
                if_map[w2].not_taken = not_taken;
            }
            break;
#endif
            case OPR_ICALL:
            case OPR_CALL:
            {
                // ignore fake call from exception handling block
                if ((WN_opcode(w2) == OPC_VCALL)
                        && (WN_Fake_Call_EH_Region(w2, Parent_Map))) break;

                proc->Incr_call_count();
#ifdef KEY
                float probability = -1;
                // Remove the use of this flag in future
                if (IPA_Enable_Branch_Heuristic && Cur_PU_Feedback && 
                        WN_operator(w2) == OPR_CALL)
                {
                    WN *block, *parent = LWN_Get_Parent(w2);

                    get_parent_if(&parent, &block);

                    if (parent)
                    { // we are inside an if stmt
                        Is_True (block && WN_operator (block) == OPR_BLOCK, ("kid of if stmt wrong"));
                        if ( WN_kid1 (parent) == block )
                            probability = if_map[parent].taken;
                        else
                            probability = if_map[parent].not_taken;
                    }
                }

                INT loopnest = get_loopnest(w2);

#if defined(KEY) && !defined(_STANDALONE_INLINER) && !defined(_LIGHTWEIGHT_INLINER)
                // This ifdef is redundant, but just for clarity.
                if (Cur_PU_Feedback && WN_operator(w2) == OPR_ICALL)
                    Process_icall(proc, w2, loopnest, probability);
#endif
#endif // KEY

#ifdef KEY
                Process_callsite(w2,
                        proc->Get_callsite_count(), loopnest, probability);
#else
                Process_callsite(w2,
                        proc->Get_callsite_count(), loopnest);
                Direct_Mod_Ref = TRUE;
#endif
                proc->Incr_callsite_count();
                Direct_Mod_Ref = TRUE;

#ifdef KEY
                if (Cur_PU_Feedback != NULL && WN_operator(w2) == OPR_ICALL)
                {
                    FB_FREQ freq = Cur_PU_Feedback->Query(w2,
                            FB_EDGE_CALL_INCOMING);
                    if (freq.Known()) {
                        icall_site[icall_cnt] = Get_callsite_idx();
                        icall_cnt++;
                    }
                }
#endif	      
                // update actual parameter count
                if (Do_common_const &&
                        !Process_control_dependence(w2, Get_callsite_idx())) {
                    proc->Set_has_unstructured_cflow();
                }
/** DAVID CODE BEGIN **/
#ifdef HICUDA
                if (_curr_proc_node != NULL)
                {
                    if (!Process_control_dependence(w2, Get_callsite_idx())) {
                        proc->Set_has_unstructured_cflow();
                    }
                }
#endif  // HICUDA
/*** DAVID CODE END ***/

                break;
            }

            case OPR_INTRINSIC_CALL:
#ifdef KEY
                Process_callsite(w2,
                        proc->Get_callsite_count(), get_loopnest(w2));
#else
                Process_callsite(w2,
                        proc->Get_callsite_count(), loopnest);
                Direct_Mod_Ref = TRUE;
#endif
                proc->Incr_callsite_count ();
                break;

            case OPR_ARRAY:
            {
                WN* base = WN_array_base(w2);
                if (OPERATOR_has_sym(WN_operator(base)))
                {
                    // for IPA:common record only stores to common arrays
                    // NOTE: This should be fixed to include the case of mixed
                    // languages, where FORTRAN common is accessed as C struct
                    if (Do_common_const
                            && WN_is_istore_or_mstore(w2)
                            && ST_is_common_element(WN_st(base)))
                    {
                        if (!Process_control_dependence(w2, 0)) {
                            proc->Set_has_unstructured_cflow();
                        }
                    }
                    // for IPA:array record all accesses to global and formal
                    // arrays
                    else if (Do_parallel_stuff && !WN_is_in_pragma(w2))
                    {
                        // We need to construct CD for local array accesses.
                        if (ST_level(WN_st(base)) == GLOBAL_SYMTAB
                                || ST_is_formal(WN_st(base))
/** DAVID CODE BEGIN **/
#ifdef HICUDA
                                || _curr_proc_node != NULL
#endif  // HICUDA
/*** DAVID CODE END ***/
                           )
                        {
                            if (!Process_control_dependence(w2, 0)) {
                                proc->Set_has_unstructured_cflow();
                            }
                        }
                    }
                }
                break;
            }

            case OPR_LDA:
            case OPR_LDID:
            case OPR_ILOAD:
#ifndef KEY
                Direct_Mod_Ref = TRUE;
#endif
                Record_ref(w2);
                if (Do_reorder && !loop_count_stack->Is_Empty()) {
                    loop_count = loop_count_stack->Top();
                    Record_struct_access(w2, loop_count);
                }

/** DAVID CODE BEGIN **/
#ifdef HICUDA
                if (_curr_proc_node != NULL
                        && WN_operator(w2) != OPR_ILOAD)
                {
                    process_symbol_in_kregion(w2);
                }
#endif  // HICUDA
/*** DAVID CODE END ***/

                break;

            case OPR_STID:
                if (Do_common_const && ST_is_common_element(st)) {
                    if (!Process_control_dependence(w2, 0)) {
                        proc->Set_has_unstructured_cflow();
                    }
                }

/** DAVID CODE BEGIN **/
#ifdef HICUDA
                if (_curr_proc_node != NULL) process_symbol_in_kregion(w2);
#endif  // HICUDA
/*** DAVID CODE END ***/

                // fall through

            case OPR_ISTORE:
            case OPR_MSTORE:
                Direct_Mod_Ref = TRUE;
                Record_mod(w2);
                if(Do_reorder && !loop_count_stack->Is_Empty()) {
                    loop_count = loop_count_stack->Top();
                    Record_struct_access(w2,loop_count);
                }
                break;

            case OPR_IO:
#ifdef KEY
                // Set mod_ref for fortran io statements like 'write',
                // otherwise a routine just containing a 'write' can get
                // deleted by ipa-dce
                Direct_Mod_Ref = TRUE;
                proc->Set_has_side_effect();
#endif
                Process_IO(w2);
                break;

            case OPR_ILDA:
            case OPR_MLOAD:
                if (Do_reorder && !loop_count_stack->Is_Empty() ) {
                    loop_count = loop_count_stack->Top();
                    Record_struct_access(w2,loop_count);
                }
                break;

                // Exceptions now come as REGIONS (not as EXC_SCOPE_BEGINS)
            case OPR_EXC_SCOPE_BEGIN:
            case OPR_EXC_SCOPE_END:
                Fail_FmtAssertion(
                        "Invalid opcode from old style Exception Processing\n");
                break;

            case OPR_REGION:
                if (WN_region_is_EH(w2))
                {
#ifdef KEY
                    Process_eh_region (w2);
#endif
                    proc->Set_exc_inline();
                }
                if (WN_region_kind(w2)== REGION_KIND_TRY) proc->Set_exc_try();
                break;

            case OPR_PRAGMA:
                if (st) {
                    if (ST_is_formal(st) && !ST_is_value_parm(st))
                        proc->Set_has_formal_pragma();

                    if (PU_has_alloca (Get_Current_PU ()) && 
                            is_variable_dim_array(ST_type(st)))
                        // Local VLAs with PRAGMA
                        proc->Set_has_formal_pragma();

                    if (ST_sclass(st) == SCLASS_FORMAL ||
                            PU_has_alloca (Get_Current_PU ())
                       ) {
                        switch (WN_pragma(w2)) {
                            case WN_PRAGMA_DISTRIBUTE:
                            case WN_PRAGMA_DISTRIBUTE_RESHAPE:
                            case WN_PRAGMA_DYNAMIC:
                            case WN_PRAGMA_REDISTRIBUTE:
                            case WN_PRAGMA_AFFINITY:
                            case WN_PRAGMA_DATA_AFFINITY:
                            case WN_PRAGMA_THREAD_AFFINITY:
                            case WN_PRAGMA_PAGE_PLACE:
                                proc->Set_has_formal_pragma();
                                break;

                            default:
                                break;
                        }
                    }

                    if (TY_is_non_pod(ST_type(st))) {
                        switch (WN_pragma(w2)) {
                            case WN_PRAGMA_LOCAL:
                            case WN_PRAGMA_LASTLOCAL:
                            case WN_PRAGMA_FIRSTPRIVATE:
                                Has_local_pragma = TRUE;
                                break;

                            default:
                                break;
                        }
                    }
                }
                // parallel pragmas can occur in 2 possible places
                // 1. just before a loop (could have interveaning statements)
                if ((WN_pragma(w2) == WN_PRAGMA_DOACROSS) ||
                        (WN_pragma(w2) == WN_PRAGMA_PARALLEL_DO)) {
                    proc->Set_has_parallel_pragma();
                }

                // 2. as a region in the region header 
                if (WN_pragma(w2) == WN_PRAGMA_PARALLEL_BEGIN) {
                    proc->Set_has_parallel_region_pragma();
                }

                if ((WN_pragma(w2) == WN_PRAGMA_DOACROSS) ||
                        (WN_pragma(w2) == WN_PRAGMA_PARALLEL_DO) ||
                        (WN_pragma(w2) == WN_PRAGMA_PARALLEL_BEGIN) ||
#ifdef KEY
                        // bug 4543
                        (WN_pragma(w2) == WN_PRAGMA_THREADPRIVATE) ||
#endif // KEY
                        (WN_pragma(w2) == WN_PRAGMA_PARALLEL_SECTIONS)) 
                    proc->Set_has_noinline_parallel_pragma();

                if ((WN_pragma(w2) == WN_PRAGMA_PDO_BEGIN) ||
                        (WN_pragma(w2) == WN_PRAGMA_PSECTION_BEGIN) ||
                        (WN_pragma(w2) == WN_PRAGMA_SINGLE_PROCESS_BEGIN)) 
                    Has_pdo_pragma = TRUE;

                if (WN_has_pragma_with_side_effect(w2))
#ifdef KEY
                    proc->Set_has_pragma_side_effect();
#else
                proc->Set_has_side_effect();
#endif

#ifdef KEY
                if (WN_pragma(w2) == WN_PRAGMA_THREADPRIVATE)
                {
                    ST * thdprv_st = ST_ptr (WN_pragma_arg2(w2));
                    Get_symbol_index (thdprv_st);  
                    WN_pragma_arg2(w2) = Get_symbol_index (thdprv_st);
                    Record_global_ref (w2, thdprv_st, OPR_PRAGMA, TRUE);

                    // increment modcount
                    INT index = Global_hash_table->Find (thdprv_st);
                    Is_True (index > 0, ("Invalid global symbol index"));
                    SUMMARY_GLOBAL * global = Get_global (index - 1);
                    global->Set_dmod ();
                    global->Inc_modcount ();
                    Get_symbol (global->Get_symbol_index ())->Set_modcount ();
                }

                // Consider any global symbol in pragma node
                if (st && ST_level (st) == GLOBAL_SYMTAB
                        && ST_class (st) == CLASS_VAR)
                {
                    // bugs 4428, 5290: summarize the symbol, and increment its
                    // ref count so that IPA DVE does not remove it.
                    Record_global_ref (w2, st, OPR_PRAGMA, TRUE);
                }
#endif

                // now, inline pragmas are part of the function body
                Process_pragma_node (w2);
                break;

                // walk the parameter statement and check for
                // addr passed opcodes only if the parameter
                // is a local/formal and it does not have the
                // addr_taken_and_saved bit set in the symbol table
                // update the addr_passed_count in that case
            case OPR_PARM:
                Update_Addr_Passed_Count(w2);
                break;

            case OPR_ALTENTRY:
                proc->Incr_altentry_count();

                if ( Do_Altentry ) {
                    // for alternate entry points record the wn 
                    // if we are looking at an alternate entry point
                    // then set is_alt_entry bit
                    proc->Set_alt_entry();
                } else {
                    // else we are implying that this subroutine has
                    // alternate entry points
                    proc->Set_has_alt_entry();
                }

                proc->Set_no_inline();

                // if we are processing the main subroutine then
                // this variable will be set to false
                // so in that case we record all the alternate
                // entry points in the subroutine
                if (!Do_Altentry)
                    Process_alt_entry (w2);
                break;

            case OPR_RETURN_VAL:
                Direct_Mod_Ref = TRUE;	// implicit store to return registers
                // fall through
            case OPR_RETURN:
                if (!proc->Has_early_returns ()) {
                    if (!Last_Node (iter))
                        proc->Set_has_early_returns ();
                }
                break;

#ifdef KEY
                // label definition
            case OPR_LABEL:
                {
                    Is_True(Do_Altentry || !WN_Label_Is_Not_Used(w2),
                            ("Label should not be marked yet"));
                    label_wn &label = label_use_map[WN_label_number(w2)];
                    Is_True(label.wn == NULL,
                            ("Process_procedure: Duplicate labels?"));
                    label.wn = w2;
                    break;
                }

                // label use
            case OPR_GOTO_OUTER_BLOCK:
                Is_True (FALSE, ("Did not expect GOTO_OUTER_BLOCK"));
            case OPR_TRUEBR:
            case OPR_FALSEBR:
            case OPR_REGION_EXIT:
            case OPR_GOTO:
            case OPR_CASEGOTO:
                label_use_map [WN_label_number (w2)].seen = TRUE;
                break;

                // bug 8479
            case OPR_ASM_STMT:
                Direct_Mod_Ref = TRUE;
                break;
#endif
        }

        /* Update the loop nesting level and the loop count stack, when this
         * node is the end of a loop.
         */
        if (WN_next(w2) == NULL) {
            WN *parent = LWN_Get_Parent(w2);
            if (parent != NULL && WN_opcode(parent) == OPC_BLOCK) {
                parent = LWN_Get_Parent(parent);
                if (parent != NULL) {
                    switch (WN_opcode(parent)) {
                        case OPC_DO_LOOP:
                        case OPC_WHILE_DO:
                        case OPC_DO_WHILE:
#ifndef KEY // disable buggy loopnest computation
                            loopnest--;
#endif // !KEY
                            if (Do_reorder) loop_count_stack->Pop();
                    }
                }
            }
        }

        // early check to see if there are any local pstatic variables
        // that would need to be promoted if inlined or cloned
        if (OPCODE_has_sym(WN_opcode(w2)) && WN_st_idx(w2) != 0) {
            ST* st2 = ST_st_idx (st) == ST_base_idx (st) ? st : ST_base (st);
            if (ST_level (st2) == CURRENT_SYMTAB) {
                // local symtab
                if (ST_sclass (st2) == SCLASS_PSTATIC) {
                    // KEY
                    Is_True (proc->Has_pstatic(),
                            ("Has_pstatic should already be set"));
                    proc->Set_has_pstatic ();
                }
            } else if (ST_sclass(st2) == SCLASS_FSTATIC &&
                    !ST_class(st2) == CLASS_CONST)
                proc->Set_has_fstatic();
        }
    }

#ifdef KEY
    {
        // Update bb and stmt count
        proc->Set_bb_count(PU_WN_BB_Cnt);
        proc->Set_stmt_count(PU_WN_Stmt_Cnt);

        // label map processing
        //
        LABEL_WN_MAP::iterator i;
        INT unused_labels = 0;
        // Mark unused labels so that IPA gets a more accurate estimate
        // of the PU size.
        for (i = label_use_map.begin(); i != label_use_map.end(); i++)
        {
            Is_True((*i).second.wn, ("Process_procedure: Undefined label?"));
            if (!(*i).second.seen) {
                unused_labels++;
                WN_Set_Label_Is_Not_Used((*i).second.wn);
            }
        }
        // For this PU, IPL has already calculated the BB count, update it
        // if required.
        if (unused_labels)
        {
            UINT16 bbs = proc->Get_bb_count();
            Is_True(bbs >= unused_labels,
                    ("Expected all labels to be included in bb count"));
            proc->Set_bb_count(bbs - unused_labels);
        }

        if_map.clear();
    }
#endif // KEY

    {
#ifdef KEY
        if_map.clear();

        /* Append a list of free slots to the current callsite_array for
         * the future use by IPA_Convert_Icalls.
         */
        FmtAssert(icall_cnt < sizeof(icall_site) / sizeof(icall_site[0]),
                ("icall array is too small."));
        for (int i = 0; i < icall_cnt; i++)
        {
            SUMMARY_CALLSITE *icall_info = Get_callsite(icall_site[i]);
            SUMMARY_CALLSITE *callsite = New_callsite();

            callsite->Set_callsite_id(proc->Get_callsite_count());
            callsite->Set_icall_slot();

            callsite->Set_param_count(icall_info->Get_param_count());
            callsite->Set_return_type(icall_info->Get_return_type());
            callsite->Set_map_id( icall_info->Get_map_id() );
            callsite->Set_loopnest( icall_info->Get_loopnest() );
            callsite->Set_probability( icall_info->Get_probability() );

            if (callsite->Get_param_count() > 0) {
                callsite->Set_actual_index(icall_info->Get_actual_index());
            }

            proc->Incr_callsite_count();
        }
#endif // KEY
    }

    /* loop_count_stack may not be empty! and loopnest may not be empty!! */

    if (proc->Get_callsite_count() > 0) {
        proc->Set_callsite_index(
                Get_callsite_idx() - proc->Get_callsite_count() + 1);
    }

    proc->Set_global_count(Get_global_idx() + 1 - proc->Get_global_index());


    if (!Do_Altentry)
    {
        if (Cur_PU_Feedback) { // was FB_PU_Has_Feedback
            INT bb_count = 0;
            INT stmt_count = 0;
#ifdef KEY
            FB_FREQ cycle_count(0.0);
#else
            FB_FREQ cycle_count(0);
#endif
            UINT16 WN_Count = 0; //INLINING_TUNING
#ifdef KEY
            FB_FREQ Cycle_Count2(0.0); //INLINING_TUNING
#else
            FB_FREQ Cycle_Count2(0); //INLINING_TUNING
#endif
            SUMMARY_FEEDBACK *fb = Get_feedback (proc->Get_feedback_index ());
            FB_FREQ freq_count = fb->Get_frequency_count();
            //	    Count_tree_size (*Cur_PU_Feedback, Get_entry_point (), bb_count, stmt_count, cycle_count, freq_count);
            Count_tree_size_tuning (*Cur_PU_Feedback, Get_entry_point (), bb_count, stmt_count, cycle_count, freq_count, WN_Count, Cycle_Count2);
            //;;printf("!!!!! PU %s(%d), bb_count = %d, stmt_count = %d, cycle=%.1f\n",ST_name(WN_st(w)), proc->Get_feedback_index (),bb_count,stmt_count,cycle_count._value);
            if (!cycle_count.Known()) {
                proc->Clear_has_PU_freq();
                DevWarn("%s has unknown frequencies so no feedback info in this procedure will be considered", ST_name(WN_st(w)));
                // FB_PU_Has_Feedback = FALSE;
            }
            else 
            {
                fb->Set_cycle_count (cycle_count);
                fb->Set_cycle_count_2(Cycle_Count2);
            }
            fb->Set_effective_bb_count (bb_count);
            fb->Set_effective_stmt_count (stmt_count);
            fb->Set_wn_count(WN_Count);
        }

        if (DoPreopt) {
            /* Copy the cached CD nodes in 'cdg' to CD summary. */
            proc->Set_ctrl_dep_index(Get_ctrl_dep_idx() + 1);
            Generate_summary_control_dependence();
            proc->Set_ctrl_dep_count(
                    Get_ctrl_dep_idx() - proc->Get_ctrl_dep_index() + 1);

            if (Get_phi_idx() > phi_index) {
                Fix_phi_node_ctrl_dep_index(Get_phi(phi_index + 1),
                        Get_phi_idx() - phi_index);
            }
        }

        if (Do_Par) IPL_Execution_Cost(w, proc, &Temp_pool, FALSE); 

        if (Do_parallel_stuff)
        {
/** DAVID CODE BEGIN **/
#ifdef HICUDA
            if (_curr_proc_node != NULL
                    && !_curr_proc_node->collect_scalar_das_only())
            {
                // Link the IPA_EDGEs in the procedure with the new WN nodes.
                _wte_map = CXX_NEW(WN_TO_EDGE_MAP(41,&Temp_pool), &Temp_pool);
                IPA_map_callsites(_curr_proc_node, w, _wte_map);
            }
#endif  // HICUDA
/*** DAVID CODE END ***/
            IPL_Access_Vector_To_Projected_Region(w, proc,
                    pu_first_formal_idx, pu_last_formal_idx,
                    pu_first_actual_idx, Get_actual_idx(),
                    pu_first_callsite_idx, Get_callsite_idx());
/** DAVID CODE BEGIN **/
#ifdef HICUDA
            if (_curr_proc_node != NULL)
            {
                if (proc->Has_incomplete_array_info())
                {
                    // We did not have a chance to summarize array data
                    // accesses for kernel regions in this procedure.
                    // Do it now.
                    HC_process_messy_array_accesses(w, _curr_proc_node,
                            _parent_kernel_map);
                }
                if (!_curr_proc_node->collect_scalar_das_only())
                {
                    // At this stage, the kernel DAS is complete.
                    // We will match data.
                    IPA_HC_match_gpu_data_with_kernel_das(w, _curr_proc_node,
                            &Temp_pool);

                    // Clear PROJECTED_REGION fields.
                    clear_projected_regions();
                }
            }
#endif  // HICUDA
/*** DAVID CODE END ***/
            Finalize_Access_Vals();
            IPL_Finalize_Projected_Regions(proc); // pop mem pools 
            IPL_Finalize_Par_Code();
        }

        if(Do_reorder) num_struct_access=Finish_PU_process_struct_access();

        // record local addr taken attributes
        Set_local_addr_taken_attrib();
    }
    else
    {
        if (Do_Par) IPL_Execution_Cost(w, proc, &Temp_pool, TRUE); 
    } 

    if (Direct_Mod_Ref) proc->Set_direct_mod_ref();

    if (Has_local_pragma && Has_pdo_pragma) proc->Set_has_pdo_pragma();
} // SUMMARIZE::Process_procedure
				 
//====================================================================
// 
// Update_call_pragmas
//
// update the call attributes if any pragmas exist
//
//====================================================================
template <PROGRAM program>
void
SUMMARIZE<program>::Update_call_pragmas (SUMMARY_CALLSITE *callsite)
{
    INT name_index;
    char* func_name;
    
    func_name = ST_name (Get_symbol (callsite->Get_symbol_index())->St_idx ());

    // go down the list from the end, and stop after the first hit
    for (INT i = Get_inline_attr_idx(); i >= 0; i--) {
	INLINE_ATTR *pragma = Get_inline_attr (i);
	char *name = pragma->Get_name ();
	BOOL is_f77 = PU_f77_lang (Get_Current_PU ());
	if (is_f77) {
	    if (Fortran_string_compare (name, func_name)) {
		if (pragma->Is_file_inline())
		    callsite->Set_must_inline();
		if (pragma->Is_no_file_inline())
		    callsite->Set_no_inline();
		return;
	    }
	} else {
	    if (func_name[0] == name[0] && strcmp (func_name, name) == 0) {
		if (pragma->Is_file_inline())
		    callsite->Set_must_inline();
		if (pragma->Is_no_file_inline())
		    callsite->Set_no_inline();
		return;
	    }
	}
    }
} // SUMMARIZE::Update_call_pragmas

#if defined(KEY) && !defined(_STANDALONE_INLINER) && !defined(_LIGHTWEIGHT_INLINER)
// If found suitable, generate a new callsite summary for the direct call
// that IPA may add for this icall. Fix other summary data as if proc now
// has another callsite.
template <PROGRAM program>
void
SUMMARIZE<program>::Process_icall (SUMMARY_PROCEDURE * proc, WN * wn,
                                   INT loopnest, float probability)
{
  Is_True (WN_operator (wn) == OPR_ICALL, ("Process_icall: ICALL expected"));

  // Tune this parameter
  const int freq_threshold = IPA_Icall_Min_Freq;

  const FB_Info_Call& info_call = Cur_PU_Feedback->Query_call(wn);
  if (!info_call.freq_entry.Known())
    return;
  if (info_call.freq_entry.Value() < freq_threshold)
    return;

  FB_Info_Icall info_icall = Cur_PU_Feedback->Query_icall(wn);
  if (info_icall.Is_uninit())
    return;

  if (info_icall.tnv._exec_counter < info_call.freq_entry.Value())
  {
    const UINT64 gap = (UINT64)info_call.freq_entry.Value() -
                               info_icall.tnv._exec_counter;
    info_icall.tnv._exec_counter += gap;
    info_icall.tnv._counters[0] += gap;
    Cur_PU_Feedback->Annot_icall (wn, info_icall);
  }

  const UINT64 exec_counter   = info_icall.tnv._exec_counter;
  const UINT64 callee_counter = info_icall.tnv._counters[0];
  const UINT64 callee_addr    = info_icall.tnv._values[0];

  if (exec_counter == 0 || callee_counter == 0)
    return;

  // For now, we have decided to proceed with ICALL transformation for
  // this icall, IPA will finally decide whether to actually transform it.
  //
  // Create a dummy callsite for a CALL. Pretend as if we are adding a call
  // to the current pu. The dummy callee is of the form "void dummy (void)".
  // NOTE: this prototype is likely to be different than the actual icall
  // prototype. Since this ST is just for temporary use in summary data, we
  // do not try to be accurate here.

  SUMMARY_CALLSITE * cs = New_callsite();
  cs->Set_callsite_id (proc->Get_callsite_count());
  cs->Set_loopnest (loopnest);
  cs->Set_probability (probability);
  cs->Set_param_count (WN_num_actuals (wn));
  cs->Set_return_type (WN_rtype (wn));

  // Get a new symbol for the dummy icall target
  static ST * st = NULL;

  if (! st)
  {
    PU_IDX pu_idx;
    PU& pu = New_PU (pu_idx);

    // a dummy placeholder for prototype
    PU_Init (pu, MTYPE_TO_TY_array[MTYPE_V], GLOBAL_SYMTAB+1);

    st = New_ST (GLOBAL_SYMTAB);
    ST_Init (st, Save_Str ("__dummy_icall_target"),
                 CLASS_FUNC, SCLASS_EXTERN, EXPORT_PREEMPTIBLE,
                 TY_IDX (pu_idx));
    vector<IPL_ST_INFO>& aux_st = Aux_Symbol_Info[GLOBAL_SYMTAB];
    aux_st.insert (aux_st.end(), 1, IPL_ST_INFO ());
  }
  cs->Set_symbol_index (Get_symbol_index (st));
 
  FB_FREQ freq ((float) callee_counter, FB_FREQ_TYPE_EXACT);
  cs->Set_callsite_freq ();
  cs->Set_frequency_count (freq);
  cs->Set_icall_target ();
  cs->Set_targ_runtime_addr (callee_addr);

  // If there are parameters in this routine then process them one at a time

  for (INT i = 0; i < cs->Get_param_count (); i++)
    Process_actual (WN_actual (wn, i));

  if (cs->Get_param_count () > 0)
    cs->Set_actual_index (Get_actual_idx () - cs->Get_param_count () + 1);

  proc->Incr_callsite_count ();
  proc->Incr_call_count ();
} // SUMMARIZE::Process_icall
#endif // KEY && !(_STANDALONE_INLINER) && !(_LIGHTWEIGHT_INLINER)

//-----------------------------------------------------------
// store the callsite information, set the actual parameter information
//-----------------------------------------------------------
template <PROGRAM program>
void SUMMARIZE<program>::Process_callsite(WN *w,
        INT id, INT loopnest, float probability)
{
    INT count;
    SUMMARY_CALLSITE *callsite = New_callsite();

    if (program == IPL) {
        // Keep a map from the call WN to the CALLSITE index.
        WN_MAP32_Set(Summary_Map, w, Get_callsite_idx() + 1);
    }

    callsite->Set_callsite_id(id);
    callsite->Set_loopnest(loopnest);
#ifdef KEY
    callsite->Set_probability(probability);
#endif
    callsite->Set_param_count(WN_num_actuals(w));
    callsite->Set_return_type(WN_rtype(w));

    if (program == INLINER) {
	callsite->Set_symbol_index (Get_symbol_index(WN_st(w)));

	WN_MAP_Set_ID (Current_Map_Tab, w);

	if (File_Pragmas)
	    Update_call_pragmas(callsite);
    
	switch (WN_operator(w)) {
	    
	case OPR_CALL:
	    // Process callsite pragma
	    // callsite pragmas SUPERSEDE callee "inline" properties
	    // callsite pragmas also SUPERSEDE COMMAND LINE options 
	    
	    //a Inconsistent pragmas: don't set must/no inline
	    if (WN_Call_Inline(w) && WN_Call_Dont_Inline(w)) {
		if (Trace_IPA || Trace_Perf)
		    fprintf (TFile, "\t user specified inconsistent inline"
			     " pragmas at callsite\n"); 
	    } else if (WN_Call_Dont_Inline(w)) {
		//b callsite: pragma noinline ==> don't inline
		callsite->Set_no_inline();
	    } else if (WN_Call_Inline(w)) {
		//c callsite pragma inline ==> inline
		callsite->Set_must_inline();
	    } else {
		WN* parent_block = LWN_Get_Parent(w);
		Is_True(parent_block, ("Cannot get parent node of call\n"));

		// ignore fake calls that are not kid of a 
		// OPR_BLOCK (generalization of PV 335012.

		if (WN_operator (parent_block) != OPR_BLOCK) {
		    // fake call -- never inline
		    callsite->Set_no_inline();
		} else if (WN_operator (parent_block) == OPR_IO_ITEM) {
		    // checks to see if the call occurs inside
    		    // a fortran write statement
		    callsite->Set_no_inline();
		}
	    }

	    break;
	default:
	    break;
	}

#if 0
	if (!INLINE_Enable_Copy_Prop)
	    return;
#endif
    } else {

	/* ipl case */
	switch (WN_operator(w)) {

	case OPR_ICALL:
	    callsite->Set_func_ptr ();
	    count = WN_kid_count(w) - 1;

	    if (OPCODE_has_sym(WN_opcode(WN_kid(w, count)))) {

                const SUMMARY_CHECK_POINT chk_pt (this);

		SUMMARY_VALUE *value = New_value ();
		INT value_idx = Get_value_idx ();
		value->Set_not_const ();

		Process_jump_function (WN_kid(w, count), value_idx);

                value = Get_value (value_idx);
		INT tmp_idx = entry_cache->Lookup (SUM_VALUE, value);
		if (tmp_idx != -1 && tmp_idx < value_idx) {
                    Restore_from_check_point(&chk_pt);
		    callsite->Set_value_index (tmp_idx);
		} else {
		    entry_cache->Insert (SUM_VALUE, value_idx);
		    callsite->Set_value_index (value_idx);
		}
	    }

	    WN_MAP_Set_ID(Current_Map_Tab, w);
	    callsite->Set_map_id (WN_map_id(w));

        // if it is a virtual function calll, set its base class and offset
        if (WN_Call_Is_Virtual(w)) {
            callsite->Set_is_virtual_call();
            WN *last = WN_kid(w, WN_kid_count(w)-1);
#ifdef TARG_IA64
            if (WN_operator_is(last, OPR_ADD)) {
                FmtAssert(WN_kid_count(last) == 2, ("Incorrect virtual call site."));
                WN *addr = WN_kid0(last);
                WN *ofst = WN_kid1(last);
                FmtAssert(WN_operator_is(addr, OPR_ILOAD) || WN_operator_is(addr, OPR_LDID),
                          ("Virtual function call does not use ILOAD or LDID."));
                FmtAssert(WN_operator_is(ofst, OPR_INTCONST),
                          ("Virtual table offset is not INTCONST."));
                callsite->Set_virtual_class(WN_ty(addr));
                callsite->Set_vtable_offset(WN_const_val(ofst));
                callsite->Set_vptr_offset(WN_load_offset(addr));
            }
            else {
                // Original WN generated by front end must be OPR_ILOAD.
                // The OPR_ILOAD may be optimized to OPR_LDID by WOPT.
                FmtAssert(WN_operator_is(last, OPR_ILOAD) || WN_operator_is(last, OPR_LDID),
                          ("Virtual function call does not use ILOAD or LDID."));
	            callsite->Set_vptr_offset(WN_load_offset(last));
                callsite->Set_virtual_class(WN_ty(last));
                callsite->Set_vtable_offset(0);
            }
#endif
#ifdef TARG_X8664
           FmtAssert(WN_operator_is(last, OPR_ILOAD), ("Virtual function call does node use ILOAD."));
           callsite->Set_vtable_offset(WN_load_offset(last));
           WN *vptr = WN_kid0(last);
           FmtAssert(WN_operator_is(vptr, OPR_ILOAD) || WN_operator_is(vptr, OPR_LDID), 
                     ("Virtual function call does not use ILOAD or LDID."));
           callsite->Set_virtual_class(WN_ty(vptr)); 
           callsite->Set_vptr_offset(WN_load_offset(vptr));
#endif
        }
		
	    break;

	case OPR_CALL:
	    callsite->Set_symbol_index (Get_symbol_index (WN_st(w)));
	    WN_MAP_Set_ID(Current_Map_Tab, w);
	    callsite->Set_map_id (WN_map_id(w));
 
	    if (Cur_PU_Feedback) { // was FB_PU_Has_Feedback
		FB_FREQ freq = Cur_PU_Feedback->Query(w, FB_EDGE_CALL_INCOMING);
		if (freq.Known()) {
		    callsite->Set_callsite_freq ();
		    callsite->Set_frequency_count (freq);
		}
		else {
		    SUMMARY_PROCEDURE *proc = Get_procedure (Get_procedure_idx());
		    proc->Clear_has_PU_freq();
		    DevWarn("%s has unknown frequencies so no feedback info in this procedure will be considered", ST_name(WN_st(w)));
		    // FB_PU_Has_Feedback = FALSE;
	        }
	    }

	    // if there are file based pragmas then update them
	    if (File_Pragmas) {
		Update_call_pragmas (callsite);
	    }

	    // Process callsite pragma
	    // callsite pragmas SUPERSEDE callee "inline" properties
	    // callsite pragmas also SUPERSEDE COMMAND LINE options 
	    
	    //a Inconsistent pragmas: don't set must/no inline
	    if (WN_Call_Inline(w) && WN_Call_Dont_Inline(w)) {
		if (Trace_IPA || Trace_Perf)
		    fprintf (TFile, "\t user specified inconsistent inline"
			     " pragmas at callsite\n"); 
	    } else if (WN_Call_Dont_Inline(w)) {
		//b callsite: pragma noinline ==> don't inline
		callsite->Set_no_inline();
	    } else if (WN_Call_Inline(w)) {
		//c callsite pragma inline ==> inline
		callsite->Set_must_inline();
	    } else {
		WN* parent_block = LWN_Get_Parent(w);
		Is_True(parent_block, ("Cannot get parent node of call\n"));
		if (WN_operator (parent_block) != OPR_BLOCK) {
		    // fake call -- never inline
		    callsite->Set_no_inline();
		}
	    }

	    break;
	
	case OPR_INTRINSIC_CALL:
	    // store things about intrinsic calls
		callsite->Set_intrinsic();
	    WN_MAP_Set_ID(Current_Map_Tab, w);
	    callsite->Set_map_id (WN_map_id(w));
	    break;

	case OPR_INTRINSIC_OP:
	    callsite->Set_intrinsic();
	    callsite->Set_map_id (-1);
	    break;
	
	default:
	    Fail_FmtAssertion ("Unsupported OPCODE %s \n",
			       OPCODE_name(WN_opcode(w)));
	    break;
	}
    }

    // if there are parameters in this routine
    // then process them one at a time

    for (INT i = 0; i < callsite->Get_param_count (); i++)
	Process_actual (WN_actual (w, i));

    if (callsite->Get_param_count () > 0)
	callsite->Set_actual_index (Get_actual_idx () -
				    callsite->Get_param_count () + 1);

} // SUMMARIZE::Process_callsite


// process formal parameters
template <PROGRAM program>
void SUMMARIZE<program>::Process_formal(WN *w,
        INT num_formals, SUMMARY_PROCEDURE *proc)
{
    for (INT i = 0; i < num_formals; i++)
    {
        SUMMARY_FORMAL *formal = New_formal();
        bzero(formal, sizeof(SUMMARY_FORMAL));

        const ST* formal_st = WN_st(WN_formal(w, i));
        INT sym_idx = Get_symbol_index(formal_st);

        // Update cross references between the two summary objects.
        formal->Set_symbol_index(sym_idx);
        Get_symbol(sym_idx)->Set_findex(Get_formal_idx());

        formal->Set_position(i);
        formal->Set_region_index(-1);
        formal->Set_machine_type(Machine_Type(WN_formal(w, i)));
        formal->Set_ty(Promoted_Parm_Type(formal_st));

        if (ST_sclass(formal_st) == SCLASS_FORMAL_REF) {
            formal->Set_is_ref_parm();
        }
        if (is_variable_dim_array(formal->Get_ty())) {
            formal->Set_is_var_dim_array();
            proc->Set_has_var_dim_array();
        }
    }

    INT position = Get_formal_idx() - num_formals + 1;
    proc->Set_formal_index(position);
    proc->Set_formal_count(num_formals);
} // SUMMARIZE::Process_formal


template <PROGRAM program>
void
SUMMARIZE<program>::Process_formal_alt (WN *w, INT kid_count)
{
    for (INT i = 0; i < kid_count; i++) {
	SUMMARY_FORMAL *formal = New_formal ();
	bzero (formal, sizeof(SUMMARY_FORMAL));
	const ST* formal_st = WN_st(WN_kid(w, i));
	INT sym_idx = Get_symbol_index (formal_st);
	formal->Set_symbol_index (sym_idx);
	Get_symbol (sym_idx)->Set_findex (Get_formal_idx());
	if (ST_sclass (formal_st) == SCLASS_FORMAL_REF)
	    formal->Set_is_ref_parm ();
	formal->Set_position (i);
        formal->Set_region_index (-1);
	formal->Set_machine_type(Machine_Type(WN_formal(w, i)));
	formal->Set_ty (ST_type(formal_st));
	if (is_variable_dim_array(formal->Get_ty())) {
	    formal->Set_is_var_dim_array();
	}
    }
} // SUMMARIZE::Process_formal_alt


static inline BOOL
Parm_Type_Equal_To_Etype(TY_IDX parm_ty, WN* array_wn)
{
  Is_True(WN_operator(array_wn) == OPR_ARRAY,
          ("Parm_Type_Equal_To_Etype: expected an array node"));

  WN* base_wn = WN_array_base(array_wn);
  if (OPERATOR_has_sym(WN_operator(base_wn))) {
    ST* array_st = WN_st(base_wn);
    TY_IDX array_ty = ST_type(array_st);
    if (TY_kind(array_ty) == KIND_POINTER) {
      array_ty = TY_pointed(array_ty);
    }
    return (TY_kind(array_ty) == KIND_ARRAY && TY_etype(array_ty) == parm_ty);
  }

  return FALSE;
}

#ifdef KEY
// This function traverses down an expression node and returns any LDA.
// Input: actual parameter
// Modifies: nothing
// Returns: LDA or NULL
//
// To generate accurate summary for actual paramter as in:
//   int a[100];
//   foo (a + global); // call foo passing the address of 'a'
//
static WN * traverse_actual (WN * w)
{
  // Lightweight inliner may have comma nodes that have not been lowered
  if (!OPERATOR_is_expression (WN_operator (w)))
    return NULL;

  if (WN_operator (w) == OPR_LDA) return w;

  for (INT i = 0; i < WN_kid_count (w); i++)
  {
    WN * lda = traverse_actual (WN_kid (w, i));
    if (lda) return lda;
  }
  return NULL;
}
#endif

template <PROGRAM program>
void 
SUMMARIZE<program>::Process_actual (WN* w)
{
  SUMMARY_ACTUAL* actual = New_actual ();
#ifdef KEY
  // Save the pointer before overwriting it.
  const WN * param = w;
#endif

  OPERATOR opr = WN_operator(w);
  if (opr == OPR_PARM) {

    WN* kid = WN_kid0(w);
    opr = WN_operator(kid);

    if (WN_Parm_By_Reference(w)) {

      if (opr == OPR_INTCONST && WN_const_val(kid) == 0) {
        // INTCONST 0 signifies an optional argument
        actual->Set_ty(0);
      }
      else if (opr == OPR_ARRAYEXP || opr == OPR_ARRSECTION) { 
        // standalone inliner will not handle complicated 
        // array expressions generated by f90
        actual->Set_ty(0);
      }
      else {
        TY_IDX parm_ty = WN_ty(w);
        if (opr == OPR_ARRAY && 
            TY_kind(parm_ty) == KIND_POINTER &&
            Parm_Type_Equal_To_Etype(TY_pointed(parm_ty), kid)) {
          // if a whole array element is passed, use the high-level array type
          ST* array_st = WN_st(WN_array_base(kid));
          actual->Set_symbol_index(Get_symbol_index(array_st));
          actual->Set_ty(ST_type(array_st));
        }	
        else {
          actual->Set_ty(parm_ty);
          if (OPERATOR_has_sym(opr)) {
            actual->Set_symbol_index(Get_symbol_index(WN_st(kid))); 
#ifdef KEY
          // bug 6229: Process any constant string instead of 
          // treating it as an indirect ref.
          ST * actual_st = WN_st (kid);
          if (opr == OPR_LDA && ST_class (actual_st) == CLASS_CONST &&
              TY_kind (ST_type (actual_st)) == KIND_ARRAY)
            w = WN_kid0(w);
#endif
          }
        }
      }
    } 
    else {
      if (WN_Parm_By_Value(w)) {
        actual->Set_is_value_parm();
      }
      actual->Set_ty(WN_ty(w)); 
      w = WN_kid0(w);
      if (OPERATOR_has_sym(opr)) {
        actual->Set_symbol_index(Get_symbol_index(WN_st(w)));
      }
#ifdef KEY
      else {
        // Traverse down actual parm looking for an LDA
        WN * lda = traverse_actual (w);
        if (lda) {
          actual->Set_symbol_index(Get_symbol_index (WN_st (lda)));
          opr = WN_operator (lda);
        }
      }
#endif
    }
  }

  if (program == INLINER)
    return;

  SUMMARY_VALUE *value;

  switch (opr) {

    case OPR_INTCONST: {

      if (WN_operator (w) == OPR_PARM)
	  w = WN_kid0(w);
      if (WN_rtype(w) == MTYPE_I4 &&
          (WN_const_val(w) == 0 || WN_const_val(w) == 1)) {
        // special case for constant 0 and 1
        actual->Set_value_index (WN_const_val (w));
        return;
      }

      value = New_value ();
      value->Set_int_const ();
      value->Set_int_const_value (WN_const_val (w));
      value->Set_mtype (WN_rtype (w));

      INT idx = entry_cache->Lookup (SUM_VALUE, value);
      if (idx != -1 && idx < Get_value_idx()) {
        actual->Set_value_index (idx);
        _value.Decidx ();
      } 
      else {
        entry_cache->Insert (SUM_VALUE, Get_value_idx ());
        actual->Set_value_index (Get_value_idx ());
      }
      return;
    }

    case OPR_LDID:
      actual->Set_pass_type(PASS_LDID);
      break;
	
    case OPR_ILOAD:
      actual->Set_pass_type(PASS_LOAD);
      break;

    case OPR_MLOAD:
      actual->Set_pass_type(PASS_MLOAD);
      break;

    case OPR_LDA:
      actual->Set_pass_type(PASS_LDA);
      break;

#ifdef KEY
    case OPR_ARRAY:
      if (WN_Parm_By_Reference(param))
        actual->Set_pass_type(PASS_ARRAY);
      break;
#endif

    default:
      actual->Set_pass_type(PASS_UNKNOWN);
      break;
  }

  const SUMMARY_CHECK_POINT chk_pt(this);

  value = New_value ();
  value->Set_not_const ();
  actual->Set_value_index (Get_value_idx ());
    
  Process_jump_function (w, Get_value_idx ());

  value = Get_value (actual->Get_value_index ());
    
  if (value->Is_not_const ()) {
    Restore_from_check_point (&chk_pt);
    actual->Set_value_index (-1);
  } 
  else {
    INT tmp_idx = entry_cache->Lookup (SUM_VALUE, value);
    if (tmp_idx != -1 && tmp_idx < actual->Get_value_index()) {
      Restore_from_check_point (&chk_pt);
      actual->Set_value_index (tmp_idx);
    } 
    else {
      entry_cache->Insert (SUM_VALUE, actual->Get_value_index ());
    }
  }
    
} // SUMMARIZE::Process_actual


//----------------------------------------------------------------------
//       check if a global symbol has a SUMMARY_SYMBOL entry
//----------------------------------------------------------------------
template <PROGRAM program>
BOOL 
SUMMARIZE<program>::Has_global_symbol_index(const ST* st)
{
  INT idx;
  SUMMARY_SYMBOL *sym;
    
  if (ST_class(st) == CLASS_CONST)
    return -1;

  // for globals, check the Globals_Index array
  if (ST_level (st) == GLOBAL_SYMTAB) {

    idx = ST_index (st);

    if (idx <= Global_index->Lastidx() && (*Global_index)[idx])
      return TRUE;
  }
  return FALSE;
}

//-----------------------------------------------------------------------
// NAME: Save_Symbol_Names
// FUNCTION: Save the name of the 'st' and, if possible, the name of its
//   function in 'Ipl_Symbol_Names' and 'Ipl_Function_Names'.  
//-----------------------------------------------------------------------

static void Save_Symbol_Name(const ST* st)
{
  static BOOL Ipl_Symbol_Function_Names_Inited = FALSE; 
  if (!Ipl_Symbol_Function_Names_Inited) { 
    Ipl_Symbol_Function_Names_Inited = TRUE; 
    Ipl_Symbol_Names = CXX_NEW(DYN_ARRAY<char*>(Malloc_Mem_Pool), 
      Malloc_Mem_Pool);
    Ipl_Function_Names = CXX_NEW(DYN_ARRAY<char*>(Malloc_Mem_Pool), 
      Malloc_Mem_Pool);
  } 
  INT idx_symbol = Ipl_Symbol_Names->Newidx();
  INT idx_function = Ipl_Function_Names->Newidx();
  char** symbol_name = &((*Ipl_Symbol_Names)[idx_symbol]);
  char** function_name = &((*Ipl_Function_Names)[idx_symbol]);
  const char* original_symbol_name = ST_name(st);
  const char* original_function_name = Scope_tab[CURRENT_SYMTAB].st == NULL
    ? "?" : ST_name(ST_st_idx(Scope_tab[CURRENT_SYMTAB].st));
  *symbol_name = CXX_NEW_ARRAY(char, strlen(original_symbol_name) + 1, 
    Malloc_Mem_Pool);
  *function_name = CXX_NEW_ARRAY(char, 
    strlen(original_function_name) + 1, Malloc_Mem_Pool);
  strcpy(*symbol_name, original_symbol_name);
  strcpy(*function_name, original_function_name);
}

//----------------------------------------------------------------------
// Return the index into the SUMMARY_SYMBOL index.
// As a side-effect, if the entry doesn't exist, then create it.
//----------------------------------------------------------------------

template <PROGRAM program>
INT32 SUMMARIZE<program>::Get_symbol_index(const ST *st)
{
    ST_CLASS sc = ST_class(st);
    ST_SCLASS ssc = ST_sclass(st);

    if (sc == CLASS_CONST) return -1;

    UINT32& index = Aux_Symbol(st).summary_symbol_idx;
    if (index != (UINT32)-1) return index;

    SUMMARY_SYMBOL *sym = New_symbol();

    if (Get_Trace(TKIND_IR, TP_IPL)) Save_Symbol_Name(st);

    /* Set the symbol and the parent function symbol. */
    sym->Set_st_idx(ST_st_idx(st));
    sym->Set_st_idx_func(ST_st_idx(Scope_tab[CURRENT_SYMTAB].st));

    /* Set symbol attributes. */

    if (ssc == SCLASS_PSTATIC || ssc == SCLASS_FSTATIC) sym->Set_static();

    if (ST_is_export_local(st)) sym->Set_local();

    if (ST_is_formal(st)) {
        sym->Set_formal();
        if (ST_is_optional_argument(st)) sym->Set_optional();
    }

    if (sc == CLASS_FUNC) { 
        sym->Set_function();
    } else if (ST_is_common_block(st)) {
        sym->Set_common_block();
    } else if (ST_is_common_element(st)) {
        sym->Set_common();
    }

    TY_IDX ty_idx = ST_type(st);

    sym->Set_btype(sc == CLASS_FUNC || sc == CLASS_BLOCK ?
            MTYPE_UNKNOWN : TY_mtype(ty_idx));

    TY_KIND ty_kind = TY_kind(ty_idx);

    // We treat a pointer to an array type as an array.
    if ((ty_kind == KIND_POINTER && TY_kind(TY_pointed(ty_idx)) == KIND_ARRAY)
            || (ty_kind == KIND_ARRAY)) {
        sym->Set_array();
    }

    if (ST_is_f90_target(st)) sym->Set_addr_f90_target();

    index = Get_symbol_idx();

    return index;
} // SUMMARIZE::Get_symbol_index


template <PROGRAM program>
void
SUMMARIZE<program>::Process_alt_entry (WN *w)
{
    ALT_ENTRY *alt = New_alt_entry ();

    alt->Set_wn (w);

    Process_formal_alt (w, WN_kid_count(w));

    INT position = Get_formal_idx () - WN_kid_count(w) + 1;
    alt->Set_position (position);
    alt->Set_formal_count (WN_kid_count(w));
} // SUMMARIZE<program>::Process_alt_entry


template <PROGRAM program>
void
SUMMARIZE<program>::Process_inline_attr (WN *pragma_node)
{
    INLINE_ATTR *inline_attr;
    TCON tc;

    switch (WN_pragma(pragma_node)) {

    case WN_PRAGMA_KAP_OPTION_INLINE:
	inline_attr = New_inline_attr ();
	tc = STC_val(WN_st(pragma_node));
	inline_attr->Set_name (Targ_String_Address(tc));
	inline_attr->Set_file_inline();
	break;

    case WN_PRAGMA_KAP_OPTION_NOINLINE:
	inline_attr = New_inline_attr ();
	tc = STC_val(WN_st(pragma_node));
	inline_attr->Set_name (Targ_String_Address(tc));
	inline_attr->Set_no_file_inline();
	break;

    default:
	break;
    }
} // SUMMARIZE<program>::Process_inline_attr

//-----------------------------------------------------------
// process a pragma WHIRL node, return TRUE if interesting pragma found
//-----------------------------------------------------------
template <PROGRAM program>
void SUMMARIZE<program>::Process_pragma_node(WN* w)
{
    switch (WN_pragma(w))
    {
        case WN_PRAGMA_KAP_OPTION_INLINE:
        case WN_PRAGMA_KAP_OPTION_NOINLINE:
            Process_inline_attr(w);
            File_Pragmas = TRUE;
            break;

        default:
            break;
    }
} // Process_pragma_node


//-----------------------------------------------------------
// process pragmas that occur in the pu pragma header list
//-----------------------------------------------------------
template <PROGRAM program>
void SUMMARIZE<program>::Process_pragmas(WN *w)
{
    WN* pragmas = WN_func_pragmas(w);
    if (pragmas == NULL) return;

    WN *p = WN_first(pragmas);
    while (p != NULL) {
        Process_pragma_node(p);
        p = WN_next(p);
    }
} // Process_pragmas


// ----------------------------------------------------------------
// Print summary info arrays: similar to IPA_Trace_Summary_Section, 
// but has simpler interface and doesn't print header info.
// ----------------------------------------------------------------
template <PROGRAM program>
void SUMMARIZE<program>::Trace(FILE* fp)
{
    if (Has_symbol_entry()) {
        Ipl_Summary_Symbol = Get_symbol(0);
        Ipl_Summary_Symbol->Print_array(fp, Get_symbol_idx()+1);
    }

    if (Has_procedure_entry())
        Get_procedure(0)->Print_array(fp, Get_procedure_idx()+1);

    if (Has_callsite_entry())
        Get_callsite(0)->Print_array(fp, Get_callsite_idx()+1);

    if (Has_feedback_entry())
        Get_feedback(0)->Print_array(fp, Get_feedback_idx()+1);

    if (Has_actual_entry())
        Get_actual(0)->Print_array(fp, Get_actual_idx()+1);

    if (Has_value_entry())
        Get_value(0)->Print_array(fp, Get_value_idx()+1);

    if (Has_expr_entry())
        Get_expr(0)->Print_array(fp, Get_expr_idx()+1);

    if (Has_phi_entry())
        Get_phi(0)->Print_array(fp, Get_phi_idx()+1);

    if (Has_chi_entry())
        Get_chi(0)->Print_array(fp, Get_chi_idx()+1);

    if (Has_stmt_entry())
        Get_stmt(0)->Print_array(fp, Get_stmt_idx()+1);

    if (Has_ctrl_dep_entry())
        Get_ctrl_dep(0)->Print_array(fp, Get_ctrl_dep_idx()+1);

    if (Has_formal_entry())
        Get_formal(0)->Print_array(fp, Get_formal_idx()+1);

    if (Has_global_entry())
        Get_global(0)->Print_array(fp, Get_global_idx()+1);

    if (Has_global_stid_entry())
        Get_global_stid(0)->Print_array(fp, Get_global_stid_idx()+1);

    if (Has_common_entry())
        Get_common(0)->Print_array(fp, Get_common_idx()+1);

    if (Has_common_shape_entry())
        Get_common_shape(0)->Print_array(fp, Get_common_shape_idx()+1);
    if (Has_struct_access_entry()) //reorder
        Get_struct_access(0)->Print_array(fp, Get_struct_access_idx()+1);
}

template <PROGRAM program>
void SUMMARIZE<program>::Record_struct_access(WN *wn, mUINT64 loop_count)
{
    /*-------------------------------------------------------------*/
    /*assumption1 : the init stmt of DO_LOOP will not use Field_id */
    /*-------------------------------------------------------------*/
    mUINT32 fld_id,flatten_flds,summary_idx;
    UINT struct_index, index=WN_ty(wn)>>8;//ty_idx of loaded object
    TY_IDX  point_idx;
    TY_TO_FLDNUM_MAP::const_iterator iter;
    TY_TO_ACCESS_MAP::const_iterator iter1;
    PTR_TO_TY_VECTOR::iterator ptr_iter;
    SUMMARY_STRUCT_ACCESS * cur_summary;
    BOOL is_pointer=FALSE;
    fld_id=WN_field_id(wn);//inc field access count
    if(fld_id<=0) return;
    if(WN_operator(wn)==OPR_ISTORE|| WN_operator(wn)==OPR_MSTORE
        ||WN_operator(wn)==OPR_MLOAD||WN_operator(wn)==OPR_LDA
        ||WN_operator(wn)==OPR_ILDA){//get the struct_index         
        point_idx=TY_pointed(Ty_tab[index]);// TY_pointed; OR find it in ptr_to_ty_vector
        struct_index=point_idx>>8;
    }
    else {
        struct_index=index;
    }
    //step 1:
    //if(ty_index in ty_to_idx_map)
    //    get cur_summary;
    //else if(not in local_cand) 
    //    return;
    iter1=Ty_to_access_map->find(struct_index);
    if (iter1!=Ty_to_access_map->end ()){// found summary
        cur_summary=iter1->second;
    }
    else {
        iter=local_cands->find(struct_index);
        if (iter==local_cands->end ())// not a candidate
            return;
        else
            flatten_flds=iter->second;
        //find struct_index in Ty_to_access_map
        //if (not found))
        //    New_struct_access(); fill in flatten_flds,
        //    fill in ty_to_idx_map for struct_index and corresponding  ptr_tys
        FmtAssert(flatten_flds!=0,
            ("in Record_struct_access(), flatten_flds!=0!\n"));
        FmtAssert(Ty_tab[struct_index].kind==KIND_STRUCT,
            ("the wn's ty_idx operated must be STRUCT"));
        iter1=Ty_to_access_map->find(struct_index);
        if (iter1!=Ty_to_access_map->end ())
            cur_summary=iter1->second;
        else{//not found summary
            cur_summary=New_struct_access(struct_index,flatten_flds);
#ifdef KEY
            Ty_to_access_map->insert(std::make_pair(struct_index,cur_summary));
#else
            Ty_to_access_map->insert(make_pair(struct_index,cur_summary));
#endif // KEY
            for(ptr_iter=Ptr_to_ty_vector->begin();
                ptr_iter!=Ptr_to_ty_vector->end();
                ptr_iter++){
                if(ptr_iter->pt_index==struct_index)
#ifdef KEY
                    Ty_to_access_map->insert(std::make_pair(ptr_iter->ty_index,cur_summary));
#else
                    Ty_to_access_map->insert(make_pair(ptr_iter->ty_index,cur_summary));
#endif // KEY
            }// fill in all such pointer_tys
        }
    }

#ifdef KEY
    Is_True(fld_id <= cur_summary->Get_flatten_flds(),
	    ("Record_struct_access: illegal field ID"));
#endif

    // process wn, Inc access_info to cur_summary
    cur_summary->Inc_fld_count(fld_id, loop_count);
    return;
}

#endif // ipl_summarize_template_INCLUDED

template <PROGRAM program>
void SUMMARIZE<program>::Process_alt_procedure(WN *w,
        INT formal_index, INT formal_count)
{
    Process_procedure(w);
    SUMMARY_PROCEDURE *proc = Get_procedure(Get_procedure_idx());
    proc->Set_formal_index (formal_index);
    proc->Set_formal_count (formal_count);
}

