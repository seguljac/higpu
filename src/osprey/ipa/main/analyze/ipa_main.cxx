/*
 * Copyright (C) 2007. QLogic Corporation. All Rights Reserved.
 */

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
#include "errors.h"
#include "mempool.h"
#include "tlog.h"                       // Generate_Tlog

#include "cgb.h"                        // CG_BROWSER, CGB_Initialize
#include "cgb_ipa.h"                    // CGB_IPA_{Initialize|Terminate}
#include "ipaa.h"                       // mod/ref analysis
#include "ipa_cg.h"			// IPA_CALL_GRAPH
#include "ipa_cprop.h"			// constant propagation
#include "ipa_inline.h"			// for IPA_INLINE
#include "ipa_option.h"                 // trace options
#include "ipa_pad.h"                    // padding related code
#include "ipa_preopt.h"                 // IPA_Preopt_Finalize
#include "ipa_section_annot.h"          // SECTION_FILE_ANNOT
#include "ipa_section_prop.h"           // IPA_ARRAY_DF_FLOW
#include "ipa_nested_pu.h"              // Build_Nested_Pu_Relations
#include "ipo_tlog_utils.h"		// Ipa_tlog

#include "ipa_chg.h"                    // Class hierarchy graph
#include "ipa_devirtual.h"              // Devirtualization

#include "ipo_defs.h"

#ifndef KEY
#include "inline_script_parser.h"
#else
extern void (*Preprocess_struct_access_p)(void);
#define Preprocess_struct_access (*Preprocess_struct_access_p)
#endif /* KEY */
#include "ipa_reorder.h"

/** DAVID CODE BEGIN **/
#ifdef HICUDA
#include "config_ipa.h"         // for IPA_HC_Included_Headers_B

#include "ir_reader.h"
#include "ipa_hc_common.h"
#include "ipa_hc_preprocess.h"
#include "ipa_hc_shape.h"
#include "ipa_hc_shape_prop.h"
#include "ipa_hc_gpu_data_prop.h"
#include "ipa_hc_access_redirection.h"
#include "ipa_hc_kernel_context_prop.h"
#include "ipa_hc_gdata_alloc.h"

// #include "hc_ic_solver.h"
#include "hc_common.h"
#include "hc_expr.h"
#include "hc_cuda_inc.h"
#include "hc_symtab.h"
#endif  // HICUDA
/*** DAVID CODE END ***/

FILE* STDOUT = stdout; 


/*****************************************************************************
 *
 * Dump the array sections in all nodes of the call graph, to the standard
 * output and the trace file (TFile).
 *
 ****************************************************************************/

static void Print_Array_Sections(char buffer[])
{
    CG_BROWSER cgb_print;
    CGB_Initialize(&cgb_print, IPA_Call_Graph);

    if (Get_Trace(TP_IPA, IPA_TRACE_SECTION_CORRECTNESS)) {
        fprintf(stdout, "%s\n", buffer);
        fprintf(TFile, "%s\n", buffer);
    }

    if (Get_Trace(TP_PTRACE1, TP_PTRACE1_IPALNO)) {
        Generate_Tlog("IPA", "Array_Section", (SRCPOS)0, "", "", "", buffer);
    }

    IPA_NODE_ITER cg_iter(cgb_print.Ipa_Cg(), PREORDER);
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
    {
        IPA_NODE *ipan = cg_iter.Current();
        if (ipan == NULL) continue;

        NODE_INDEX v = cgb_print.Find_Vertex(ipan);
        if (v == INVALID_NODE_INDEX) continue;

        if (Get_Trace(TP_IPA, IPA_TRACE_SECTION_CORRECTNESS)) { 
            fprintf(stdout, "%s\n", ipan->Name());
            fprintf(TFile, "%s\n", ipan->Name());
        }

        cgb_print.Set_Cnode(ipan);
        cgb_print.Set_Cvertex(v);

        IPA_NODE_SECTION_INFO *ipas = ipan->Section_Annot();
        SECTION_FILE_ANNOT *ipaf =
            IP_FILE_HDR_section_annot(ipan->File_Header());
        if (ipas == NULL || ipaf == NULL) continue;

        if (Get_Trace(TP_IPA, IPA_TRACE_SECTION_CORRECTNESS)) {
            SUMMARY_PROCEDURE *proc = cgb_print.Cnode()->Summary_Proc();
            if (proc != NULL && proc->Has_incomplete_array_info()) {
                fprintf(stdout, "INCOMPLETE ARRAY INFO\n");
            }
            cgb_print.Mod_Ref_Formals(stdout);
            cgb_print.Mod_Ref_Commons(stdout);
            cgb_print.Mod_Ref_Formals(TFile);
            cgb_print.Mod_Ref_Commons(TFile);
        }

        if (Get_Trace(TP_PTRACE1, TP_PTRACE1_IPALNO)) {
            cgb_print.Tlog_Mod_Ref_Formals();
            cgb_print.Tlog_Mod_Ref_Commons();
        }
    }
}

#ifndef KEY
//-----------------------------------------------------------------------
// NAME: Perform_Inline_Script_Analysis
// FUNCTION: Perform inlining analysis based on a context sensitive inlining specification file
//-----------------------------------------------------------------------
static void Perform_Inline_Script_Analysis(IPA_CALL_GRAPH* cg, MEM_POOL* pool, MEM_POOL* parser_pool)
{
    BOOL result = FALSE;
    IPA_NODE_ITER cg_iter (cg, LEVELORDER, pool);

#ifdef Enable_ISP_Verify // Additional debug information -- to be removed
    int null_caller_count = 0;
    int null_callee_count = 0;
#endif

    // traverse the call-graph, with visiting all nodes at levelorder first
    for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next()) {
      IPA_NODE* caller = cg_iter.Current();
      if(caller) {
 	IPA_NODE_CONTEXT context (caller);
	cg->Map_Callsites (caller);
		
	IPA_SUCC_ITER edge_iter (cg, caller);
        for (edge_iter.First (); !edge_iter.Is_Empty (); edge_iter.Next ()) {
	    IPA_EDGE *edge = edge_iter.Current_Edge ();
            if (edge) {
                // Restore the WHIRL node information
            	IPA_NODE* callee = cg->Callee (edge);
    		WN* call_wn = edge->Whirl_Node();

    		// Retrieve the source line number, caller/callee file name and function name
    		INT32 callsite_linenum;
		USRCPOS callsite_srcpos;
    		char  *caller_filename, *callee_filename;
    		char  *caller_funcname, *callee_funcname;

    		IP_FILE_HDR& caller_hdr = caller->File_Header ();
    		IP_FILE_HDR& callee_hdr = callee->File_Header ();

    		if (call_wn == NULL) {
       			fprintf (stderr, "Warning: no source line number found for call-edge [%s --> %s]\n",
       	       			 caller->Name(), callee->Name());
       	  		callsite_linenum = 0;
    		}
  		else {
      			USRCPOS_srcpos(callsite_srcpos) = WN_Get_Linenum (call_wn);
      			callsite_linenum = USRCPOS_linenum(callsite_srcpos);
    		}

      		caller_filename = (char *) alloca(strlen(caller_hdr.file_name)+1);
		strcpy(caller_filename, caller_hdr.file_name);
		callee_filename = (char *) alloca(strlen(callee_hdr.file_name)+1);
		strcpy(callee_filename, callee_hdr.file_name);      		
      		
#ifdef Enable_ISP_Verify // Additional debug information -- to be removed
		fprintf (stderr, "Inline script analysis for call pair");
		fprintf (stderr, "(Name: %s, Line: %d, File: %s) -> callee (Name: %s, File: %s)\n",
         		caller->Name(), callsite_linenum, caller_filename,
         		callee->Name(), callee_filename);
#endif
    		
                // Assemble the caller_key and call_key for inquiry into the inlining record
    		char *caller_key, *callee_key;
    		ISP_Fix_Filename(caller_filename);
		caller_funcname = (char *) alloca(strlen(DEMANGLE (caller->Name()))+1);
		strcpy(caller_funcname, DEMANGLE (caller->Name()));    		
    		ISP_Fix_Filename(caller_funcname);
    		
    		caller_key = (char *) alloca(strlen(caller_filename)+strlen(caller_funcname)+2);
    		strcpy(caller_key, "");
    		strcat(caller_key, caller_filename);
    		strcat(caller_key, caller_funcname);

    		ISP_Fix_Filename(callee_filename);
		callee_funcname = (char *) alloca(strlen(DEMANGLE (callee->Name()))+1);
		strcpy(callee_funcname, DEMANGLE (callee->Name()));	    		
    		ISP_Fix_Filename(callee_funcname);
    		// Assumption: the line number of integer type should not exceed 30 digits (base-10)   		
    		char callsite_linestr[30];
    		sprintf(callsite_linestr, "%d", callsite_linenum);
    		
    		callee_key = (char *) alloca(strlen(callsite_linestr)+strlen(callee_filename)+strlen(callee_funcname)+3);
    		strcpy(callee_key, "");
    		strcat(callee_key, callsite_linestr);
    		strcat(callee_key, callee_filename);
    		strcat(callee_key, callee_funcname);

    		result = Check_Inline_Script(INLINE_Script_Name, caller_key, callee_key, parser_pool);
    		
    		// Set the call edge inlining attribute according to the inlining checking results
    		if(result == TRUE) {
    		    edge->Set_Must_Inline_Attrib();
    		} else {
    		    edge->Set_Noinline_Attrib();
    		}
            }
#ifdef Enable_ISP_Verify // Additional debug information -- to be removed
	    else null_callee_count++;
#endif	
	}
      }
#ifdef Enable_ISP_Verify // Additional debug information -- to be removed
      else null_caller_count++;
#endif
    }	

#ifdef Enable_ISP_Verify // Additional debug information -- to be removed
    fprintf (stderr, "Inline script DEBUG null_caller = %d, null_callee = %d\n", null_caller_count, null_callee_count);
#endif
#ifdef Enable_ISP_Verify
    Verify_Inline_Script();
#endif
}
#endif /* KEY */


//-------------------------------------------------------------------------
// the main analysis phase at work! 
//-------------------------------------------------------------------------

void Perform_Interprocedural_Analysis()
{
    BOOL has_nested_pu = FALSE;
    BOOL run_autopar = FALSE;

    MEM_POOL_Popper pool(MEM_phase_nz_pool_ptr);

    if (IPA_Enable_Reorder) Init_merge_access();    // field reorder

    // read PU infos, update summaries, and process globals
    for (UINT i = 0; i < IP_File_header.size(); ++i)
    {
        IP_FILE_HDR& header = IP_File_header[i];

        IPA_Process_File(header);

        if (IP_FILE_HDR_has_nested_pu(header)) has_nested_pu = TRUE;
        if (IP_FILE_HDR_file_header(header)->Run_AutoPar()) run_autopar = TRUE;
    }

    if (Get_Trace(TP_IPA,IPA_TRACE_TUNING_NEW) && IPA_Enable_Reorder) {
        fprintf(TFile, "\n%s%s\tstruct_access info after merging\n%s%s\n",
                DBar, DBar, DBar, DBar);
        print_merged_access();
    }

    if (run_autopar) {
#ifndef KEY
        // enable multi_cloning and preopt for parallelization analysis
        if (!IPA_Max_Node_Clones_Set) {
            IPA_Max_Node_Clones = 5; // default number of clones per PU
        }
        if (!IPA_Enable_Preopt_Set) IPA_Enable_Preopt = TRUE;
#endif // !KEY
    } else {
        // array section analysis is done only with -ipa -pfa
        IPA_Enable_Array_Sections = FALSE;
    }

    if (IPA_Enable_Padding || IPA_Enable_Split_Common)
    {
        Temporary_Error_Phase ephase("IPA Padding Analysis");
        if (Verbose) {
            fprintf (stderr, "Common blocks padding/split analysis ...");
            fflush (stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf (TFile, "\t<<<Padding/Split analysis begins>>>\n");
        }

        Padding_Analysis(IP_File_header.size());

        if (Trace_IPA || Trace_Perf) {
            fprintf (TFile, "\t<<<Padding/Split analysis completed>>>\n");
        }
    }

    /* Create and build a call graph. */

    Temporary_Error_Phase ephase("IPA Call Graph Construction");

    if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
        fprintf(TFile, "\n%s%s\tMemory allocation information before Build_call_graph\n%s%s\n",
                DBar, DBar, DBar, DBar);
        MEM_Trace();
    }

    if (Verbose) {
        fprintf(stderr, "Building call graphs ...");
        fflush(stderr);
    }

    if (Trace_IPA || Trace_Perf) {
        fprintf(TFile, "\t<<<Call Graph Construction begins>>>\n");
    }

    Build_Call_Graph();

#ifdef KEY
    {
        /* Traverse the call graph and mark C++ nodes as PU_Can_Throw
         * (exception handling) appropriately.
         */
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, POSTORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            if (!IPA_Enable_EH_Region_Removal
                    && !IPA_Enable_Pure_Call_Opt) break;

            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            if (node->PU_Can_Throw()
                    || node->Summary_Proc()->Has_side_effect())
            {
                // Mark its callers appropriately.
                IPA_PRED_ITER preds(node->Node_Index());
                for (preds.First(); !preds.Is_Empty(); preds.Next())
                {
                    IPA_EDGE *edge = preds.Current_Edge();
                    if (edge == NULL) continue;

                    IPA_NODE *caller = IPA_Call_Graph->Caller(edge);

                    PU caller_pu = Pu_Table[ST_pu((caller)->Func_ST())];
                    if (IPA_Enable_EH_Region_Removal
                            && node->PU_Can_Throw()
                            && (PU_src_lang(caller_pu) & PU_CXX_LANG)) {
                        caller->Set_PU_Can_Throw();
                    }

                    if (node->Summary_Proc()->Has_side_effect()) {
                        caller->Summary_Proc()->Set_has_side_effect();
                    }
                }
            }
        }

        if (IPA_Enable_Source_PU_Order || Opt_Options_Inconsistent) {
            for (UINT i = 0; i < IP_File_header.size(); ++i) {
                // Store the file-id in each IPA_NODE.
                Mark_PUs_With_File_Id(
                        IP_FILE_HDR_pu_list(IP_File_header[i]), i);
            }
        }
    }
#endif

    // INLINING TOOL
    if (INLINE_Enable_Script)
    {
#ifdef KEY
        fprintf(stdout, "inline script not implemented\n");
        exit(1);
#else
        MEM_POOL script_parser_pool;
        MEM_POOL_Initialize(&script_parser_pool,
                "inlining script parser pool", FALSE);
        MEM_POOL_Push(&script_parser_pool);

        MEM_POOL_Popper inline_script_pool(MEM_local_nz_pool_ptr);
        Perform_Inline_Script_Analysis(IPA_Call_Graph,
                inline_script_pool.Pool(), &script_parser_pool);

        MEM_POOL_Pop(&script_parser_pool);
        MEM_POOL_Delete(&script_parser_pool);
#endif /* KEY */
    }
    // INLINING TOOL END

    if (has_nested_pu)
    {
        Build_Nested_Pu_Relations();
        if (Verbose) {
            fprintf (stderr, "Building Nested PU Relations...");
            fflush (stderr);
        }
    }

#ifdef Is_True_On
    CGB_IPA_Initialize(IPA_Call_Graph);
#endif

    Ipa_tlog("Must Inline", 0, "Count %d", Total_Must_Inlined);
    Ipa_tlog("Must Not-Inline", 0, "Count %d", Total_Must_Not_Inlined);

    if (Trace_IPA || Trace_Perf) {
        fprintf (TFile, "\t<<<Call Graph Construction completed>>>\n");
    }

#ifdef TODO
    if (IPA_Enable_daVinci) {
        cg_display = (daVinci *)
            CXX_NEW (daVinci (IPA_Call_Graph->Graph (), Malloc_Mem_Pool),
                    Malloc_Mem_Pool); 

        cg_display->Translate_Call_Graph ();
    }
#endif // TODO

    if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
        fprintf(TFile, "\n%s%s\tMemory allocation information after "
                "Build_call_graph\n%s%s\n", DBar, DBar, DBar, DBar );
        MEM_Trace();
    }

/** DAVID CODE BEGIN **/

    // Before we proceed to any analyses and optimizations, we must ensure
    // that there is a main function. Otherwise, nothing will be emitted by
    // ipa_link.
    {
        BOOL found_main = FALSE;

        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE* node = cg_iter.Current();
            if (node == NULL) continue;

            if (strcmp(node->Name(), "main") == 0)
            {
                // Is <main> undeletable? Is_Undeletable
                Is_True(Pred_Is_Root(node), (""));
                found_main = TRUE;
                break;
            }
        }

        HC_assert(found_main, ("The <main> function does not exist!"));
    }

/*** DAVID CODE END ***/

    // This is where we do the partitioning and looping through the
    // different partition to do the analysis

    if ((IPA_Enable_SP_Partition && (IPA_Space_Access_Mode == SAVE_SPACE_MODE))
            || IPA_Enable_GP_Partition) {
        // This is where we do the partitioning and build the partitioning
        // data structure if necessary
        // In the partitioning algorithm, it should for each IPA_NODE
        // 1) tag with a partition group, using Set_partition_group()
        // 2) whether it is INTERNAL to the partition using Set_partition_internal()
        // then for each DEFINED external DATA symbol,
        // 1) tag whether it is INTERNAL
        // 2) whether it is gp-relative
        //
        // to that partition.  The partitioning algorithm should take
        // into account whether it is doing partitioning for
        // solving 1) space problem(IPA_Enable_SP_Partition)
        //  or 2) multigot problem(IPA_Enable_GP_Partition)
        // 
        // Also, something new:
        // pobj->ipa_info.gp_status & pobj->ipa_info.partition_grp needs
        // to be set here for each object being looked at also
    }

    // need by ipa_inline
    Total_Prog_Size = Orig_Prog_Weight;

    MEM_POOL_Push (MEM_local_nz_pool_ptr);

    if (IPA_Enable_DVE || IPA_Enable_CGI)
    {
        Temporary_Error_Phase ephase("IPA Global Variable Optimization");
        if (Verbose) {
            fprintf(stderr, "Global Variable Optimization ...");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Global Variable Optimization begins>>>\n");
        }
#ifdef TODO
        if (IPA_Enable_Feedback) {
            setup_IPA_feedback_phase();
            fprintf(IPA_Feedback_dve_fd,
                    "\nDEAD VARIABLES (defined but not used)\n\n");
        }
#endif

        extern void Optimize_Global_Variables();
        Optimize_Global_Variables();

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile,
                    "\t<<<Global Variable Optimization completed>>>\n");
        }
#ifdef TODO
        if (IPA_Enable_Feedback) {
            cleanup_IPA_feedback_phase();
            fflush(IPA_Feedback_dve_fd);
        }
#endif
    }

    if (IPA_Enable_Reorder && !merged_access->empty())
    {
        IPA_reorder_legality_process(); 	
    }

    // Mark all unreachable nodes that are either EXPORT_LOCAL (file static)
    // or EXPORT_INTERNAL *AND* do not have address taken as "deletable".
    // Functions that are completely inlined to their callers are taken care
    // of later.
    if (IPA_Enable_DFE)
    {
        Temporary_Error_Phase ephase ("IPA Dead Functions Elimination");
        if (Verbose) {
            fprintf (stderr, "Dead functions elimination ...");
            fflush (stderr);
        }
#ifdef TODO
        if( IPA_Enable_Feedback ) {
            //
            // Functions completely inlined to their callers are handled in
            // routine Perform_inlining (ipa_inline.cxx) and are not marked
            // deletable until then. Thus the dfe info presented here does
            // represent truly dead source.
            //
            setup_IPA_feedback_phase();
            fprintf(IPA_Feedback_dfe_fd,
                    "\nDEAD FUNCTIONS (but not due to inlining)\n\n");
        }
#endif
        if (Trace_IPA || Trace_Perf)
            fprintf (TFile, "\t<<<Dead Functions Elimination begins>>>\n");

        Total_Dead_Function_Weight = Eliminate_Dead_Func();
        Total_Prog_Size = Orig_Prog_Weight - Total_Dead_Function_Weight;

        if (Trace_IPA || Trace_Perf)
            fprintf (TFile, "\t<<<Dead Functions Elimination completed>>>\n");
#ifdef TODO
        if( IPA_Enable_Feedback ) {
            cleanup_IPA_feedback_phase ();
            fflush(IPA_Feedback_dfe_fd);
        }
#endif
    }

    if (IPA_Enable_Devirtualization)
    {
        Temporary_Error_Phase ephase("IPA Devirtualization");
        IPA_Class_Hierarchy = Build_Class_Hierarchy();
        IPA_devirtualization();
    }

    if (IPA_Enable_Simple_Alias)
    {
        /* DAVID COMMENT: the result of this analysis is used in
         * interprocedural array section analysis.
         */
        Temporary_Error_Phase ephase("Interprocedural Alias Analysis");
        if (Verbose) {
            fprintf(stderr, "Alias analysis ...");
            fflush(stderr);
        }

        IPAA ipaa(NULL);
        ipaa.Do_Simple_IPAA(*IPA_Call_Graph);

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s%s\tMemory allocation information after IPAA\n%s%s\n",
                    DBar, DBar, DBar, DBar);
            MEM_Trace();
        }
    }
    else
    {
        /* Common block constants and aggresive node cloning can only be done
         * when alias information is available.
         */
        IPA_Enable_Cprop = FALSE;
        IPA_Enable_Common_Const = FALSE;
        IPA_Max_Node_Clones = 0;
    }

    /* Propagate information about formal parameters used as symbolic terms in
     * array section summaries. This information will later be used to trigger
     * cloning.
     */
    if (IPA_Enable_Array_Sections
            && IPA_Max_Node_Clones > 0 && IPA_Max_Clone_Bloat > 0)
    {
        Temporary_Error_Phase ephase("IPA Cloning Analysis");
        if (Verbose) {
            fprintf(stderr, "Cloning Analysis ...");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile,
                    "\t<<<Analysis of formals for cloning begins>>>\n");
        }

        IPA_FORMALS_IN_ARRAY_SECTION_DF clone_df(IPA_Call_Graph, 
                BACKWARD, MEM_local_nz_pool_ptr);
        clone_df.Init();
        clone_df.Solve();

        if (Get_Trace(TP_IPA, IPA_TRACE_CPROP_CLONING)) clone_df.Print(TFile);

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Analysis of formals for cloning ends>>>\n");
        }
    }

/** DAVID CODE BEGIN **/
    // We do not want any changes to the existing functions, so disable
    // constant propagation.
    IPA_Enable_Cprop = IPA_Enable_Cprop2 = FALSE;
/*** DAVID CODE END ***/

    // solve interprocedural constant propagation     
    if (IPA_Enable_Cprop)
    {
        Temporary_Error_Phase ephase("IPA Constant Propagation");

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s\tMemory allocation information "
                    "before IP constant propagation\n%s\n",
                    DBar, DBar);
            MEM_Trace();
        }

        if (Verbose) {
            fprintf(stderr, "Constant propagation ...");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Constant Propagation begins>>>\n");
        }

        MEM_POOL_Initialize(&Ipa_cprop_pool, "cprop pool", 0);

        // Set the upper limit for the total number of clone nodes 
        IPA_Max_Total_Clones = 
            (GRAPH_vcnt(IPA_Call_Graph->Graph()) * IPA_Max_Clone_Bloat) / 100;

        if (IPA_Enable_Common_Const) {
            static BOOL global_cprop_pool_inited = FALSE;
            if (!global_cprop_pool_inited) {
                MEM_POOL_Initialize(&Global_mem_pool,
                        "global_cprop_mem_pool", 0);
                MEM_POOL_Push(&Global_mem_pool);
                global_cprop_pool_inited = TRUE;
            }
            MEM_POOL_Initialize(&local_cprop_pool, "local_cprop_mem_pool", 0);
            MEM_POOL_Push(&local_cprop_pool);
        }

        IPA_CPROP_DF_FLOW df(FORWARD, MEM_local_nz_pool_ptr);

        df.Init();   // initialize the annotations
        df.Solve();  // solve the data flow problem

        // Convert quasi clones into real ones
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, POSTORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next()) {
            IPA_NODE* node = cg_iter.Current();
            if (node != NULL && node->Is_Quasi_Clone()) {
                IPA_Call_Graph->Quasi_To_Real_Clone(node);
            }
        }
        if (Get_Trace(TP_IPA, IPA_TRACE_CPROP_CLONING)) {
            IPA_Call_Graph->Print(TFile);
        }

#if 0
        // Optionally, we could remove quasi clones without making them real
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next()) {
            IPA_NODE* node = (IPA_NODE*) cg_iter.Current();
            if (node && node->Is_Quasi_Clone()) {
                IPA_Call_Graph->Remove_Quasi_Clone(node);
            }
        }
#endif

        if (IPA_Enable_Common_Const) {
            MEM_POOL_Pop(&local_cprop_pool);
            MEM_POOL_Delete(&local_cprop_pool);
        }

        // in the process perform cloning
        if (Trace_IPA || Trace_Perf) {
            df.Print(TFile);
            fprintf(TFile, "Constant Count = %d \n", IPA_Constant_Count);
            fprintf (TFile,"\t<<<Constant Propagation ends>>>\n");
        }
        Ipa_tlog( "Cprop", 0, "Count %d", IPA_Constant_Count);

#ifdef TODO
        // check for IPA:feedback=ON - get constant info if so
        if( IPA_Enable_Feedback ) {
            fprintf(IPA_Feedback_con_fd,"\nCONSTANTS FOUND\n\n");
            df.Print(IPA_Feedback_con_fd);
            fflush(IPA_Feedback_con_fd);
        }
#endif // TODO

        if (WN_mem_pool_ptr == &Ipa_cprop_pool) WN_mem_pool_ptr = NULL;
        MEM_POOL_Delete(&Ipa_cprop_pool);

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s%s\tMemory allocation information "
                    "after IP constant propagation\n%s%s\n",
                    DBar, DBar, DBar, DBar);
            MEM_Trace();
        }
    }

/** DAVID CODE BEGIN **/
#ifdef HICUDA
    // Solve the interprocedural array shape propagation.
    {
        Temporary_Error_Phase ephase("IP Shape Propagation");

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s\tMemory allocation information "
                    "before IP shape propagation\n%s\n",
                    DBar, DBar);
            MEM_Trace();
        }

        if (Verbose) {
            fprintf(stderr, "Shape propagation ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Shape Propagation begins>>>\n");
        }

        MEM_POOL_Initialize(&Ipa_shape_prop_pool, "shape prop pool", 0);
        MEM_POOL_Push(&Ipa_shape_prop_pool);

        // TEST IC SOLVER.
        // ic_solver_test(&Ipa_shape_prop_pool);

        IPA_HC_SHAPE_PROP_DF df(&Ipa_shape_prop_pool);

        df.Init();   // initialize the annotations
        df.Solve();  // solve the data flow problem

        df.PostProcess();

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Shape Propagation ends>>>\n");
        }

        MEM_POOL_Pop(&Ipa_shape_prop_pool);
        MEM_POOL_Delete(&Ipa_shape_prop_pool);

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s\tMemory allocation information "
                    "after IP shape propagation\n%s\n",
                    DBar, DBar);
            MEM_Trace();
        }
    }
#endif  // HICUDA
/*** DAVID CODE END ***/

#ifdef KEY
    if (IPA_Enable_Preopt) Preprocess_struct_access();
#endif // KEY

/** DAVID CODE BEGIN **/
    // This pre-optimization must be disabled because it does not generate
    // array summaries.
    // IPA_Enable_Preopt_Set = IPA_Enable_Preopt = TRUE;
/*** DAVID CODE END ***/

    // Call preopt on each node if requested.
    if (IPA_Enable_Preopt_Set && IPA_Enable_Preopt)
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, POSTORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next()) {
            IPA_NODE *node = cg_iter.Current();
            if (node != NULL) IPA_Preoptimize(node);
        }
    }

    MEM_POOL_Pop(MEM_local_nz_pool_ptr);

/** DAVID CODE BEGIN **/
#if 0
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            node->Summary_Proc()->Print(stderr, node->Node_Index());
        }
    }
#endif
/*** DAVID CODE END ***/

    // solve interprocedural array section analysis
    if (IPA_Enable_Array_Sections)
    {
        Temporary_Error_Phase ephase("IPA Array Section Analysis");
        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile, "\n%s%s\tMemory allocation information before "
                    "IP array section propagation \n%s%s\n",
                    DBar, DBar, DBar, DBar);
            MEM_Trace();
        }
        if (Verbose) {
            fprintf(stderr, "Array Section analysis ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Array section propagation begins>>>\n");
        }

        MEM_POOL_Push(MEM_local_nz_pool_ptr);

        // DAVID COMMENT: the analysis must be backward.
        IPA_ARRAY_DF_FLOW array_df(IPA_Call_Graph, BACKWARD,
                MEM_local_nz_pool_ptr);

        array_df.Init();   // initialize the annotations
        Print_Array_Sections("BEFORE PROPAGATION:");

        array_df.Solve();  // solve the data flow problem
        Print_Array_Sections("AFTER PROPAGATION:");

        if (Trace_IPA || Trace_Perf) {
            fprintf (TFile,"\t<<<Array section propagation ends>>>\n");
        }

        MEM_POOL_Pop(MEM_local_nz_pool_ptr);

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile, "\n%s%s\tMemory allocation information after "
                    "IP array section propagation \n%s%s\n",
                    DBar, DBar, DBar, DBar);
            MEM_Trace();
        }
    }

/** DAVID CODE BEGIN **/
#if 0
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            node->Summary_Proc()->Print(stderr, node->Node_Index());
        }
    }
#endif
/*** DAVID CODE END ***/

    if (IPA_Enable_Preopt) IPA_Preopt_Finalize();

/** DAVID CODE BEGIN **/
#ifdef HICUDA
    // Pre-process and validate directives.
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next()) {
            IPA_NODE *node = cg_iter.Current();
            if (node != NULL) HC_preprocess(node);
        }

        IPA_HC_classify_procedure();

        IPA_NODE_ITER cg_iter1(IPA_Call_Graph, PREORDER);
        for (cg_iter1.First(); !cg_iter1.Is_Empty(); cg_iter1.Next()) {
            IPA_NODE *node = cg_iter1.Current();
            if (node != NULL) HC_post_validate(node);
        }
    }

    // Parse KERNEL directives in each procedure.
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next()) {
            IPA_NODE *node = cg_iter.Current();
            if (node != NULL) HC_parse_kernel_directives(node);
        }
    }

    // This mempool will hold GPU data annotations, which need to last across
    // multiple phases.
    MEM_POOL_Initialize(&Ipa_gpu_data_prop_pool, "GPU data prop pool", 0);
    MEM_POOL_Push(&Ipa_gpu_data_prop_pool);

    // Solve the inter-procedural data directive propagation.
    {
        Temporary_Error_Phase ephase("IP hiCUDA Data Directive Propagation");

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s\tMemory allocation information "
                    "before IP hiCUDA data directive propagation\n%s\n",
                    DBar, DBar);
            MEM_Trace();
        }

        if (Verbose) {
            fprintf(stderr, "hiCUDA data directive propagation ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile,
                    "\t<<<hiCUDA Data Directive Propagation begins>>>\n");
        }

        IPA_HC_GPU_DATA_PROP_DF df(&Ipa_gpu_data_prop_pool);

        df.Init();   // initialize the annotations
        df.Solve();  // solve the data flow problem

        // Do data matching, lowering data directives, and parameter
        // expansion.
        df.PostProcess();

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile,
                    "\t<<<hiCUDA Data Directive Propagation ends>>>\n");
        }

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s\tMemory allocation information "
                    "after IP hiCUDA data directive propagation\n%s\n",
                    DBar, DBar);
            MEM_Trace();
        }
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA)) IPA_Call_Graph_print(TFile);

    // Analyze each kernel's grid geometry.
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            // Only work on K-procedures.
            if (node == NULL || !node->contains_kernel()) continue;

            IPA_NODE_CONTEXT context(node);

            HC_KERNEL_INFO_LIST *kil = node->get_kernel_info_list();
            UINT n_kernels = kil->Elements();
            for (UINT i = 0; i < n_kernels; ++i) {
                (*kil)[i]->process_grid_geometry();
            }
        }
    }

    // Parse SHARED directives.
    {
        Temporary_Error_Phase ephase("Parsing SHARED directives");

        if (Verbose)
        {
            fprintf(stderr, "Parsing SHARED directives ... ");
            fflush(stderr);
        }

        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;
            // Only work on K-/IK-procedures.
            if (!node->contains_kernel()
                    && !node->may_be_inside_kernel()) continue;

            // Use the common GPU data pool.
            HC_parse_shared_directives(node, &Ipa_gpu_data_prop_pool);
        }
    }

    MEM_POOL_Initialize(&Ipa_kernel_prop_pool, "kernel context pool", 0);
    MEM_POOL_Push(&Ipa_kernel_prop_pool);

    // Solve the interprocedural kerrnel context propagation.
    {
        Temporary_Error_Phase ephase("IP Kernel Context Propagation");

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s\tMemory allocation information "
                    "before IP kernel context propagation\n%s\n",
                    DBar, DBar);
            MEM_Trace();
        }

        if (Verbose) {
            fprintf(stderr, "kernel context propagation ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Kernel Context Propagation begins>>>\n");
        }

        IPA_HC_KERNEL_CONTEXT_PROP_DF df(&Ipa_kernel_prop_pool);

        df.Init();   // initialize the annotations
        df.Solve();  // solve the data flow problem

        df.PostProcess();

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Kernel Context Propagation ends>>>\n");
        }

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s\tMemory allocation information "
                    "after IP kernel context propagation\n%s\n",
                    DBar, DBar);
            MEM_Trace();
        }
    }

    // Solve the interprocedural GPU variable propagation (within kernels).
    {
        Temporary_Error_Phase ephase("IP GPU Variable Propagation");

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s\tMemory allocation information "
                    "before IP GPU variable propagation\n%s\n",
                    DBar, DBar);
            MEM_Trace();
        }

        if (Verbose) {
            fprintf(stderr, "GPU variable propagation ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<GPU Variable Propagation begins>>>\n");
        }
#if 0
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;
            // TODO: use a temp pool.
            HC_build_edge_kernel_annot(node, &Ipa_gpu_data_prop_pool);
        }
#endif
        IPA_HC_GPU_VAR_PROP_DF df(&Ipa_gpu_data_prop_pool);

        df.Init();   // initialize the annotations
        df.Solve();  // solve the data flow problem

        df.PostProcess();

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<GPU Variable Propagation ends>>>\n");
        }

        if (Get_Trace(TKIND_ALLOC, TP_IPA)) {
            fprintf(TFile,
                    "\n%s\tMemory allocation information "
                    "after IP GPU variable propagation\n%s\n",
                    DBar, DBar);
            MEM_Trace();
        }
    }

    // Handle SHARED directives.
    {
        Temporary_Error_Phase ephase("Translation of SHARED directives");

        if (Verbose)
        {
            fprintf(stderr, "translation of SHARED directives ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf)
        {
            fprintf(TFile,
                    "\t<<<Translation of SHARED directives begins>>>\n");
        }

        // Determine allocation offset of each SHARED directive.
        IPA_HC_alloc_shared_mem();

        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            // Work on K-/IK-procedures.
            if (!node->contains_kernel()
                    && !node->may_be_inside_kernel()) continue;

            HC_handle_shared_directives(node);
        }

        if (Trace_IPA || Trace_Perf)
        {
            fprintf(TFile,
                    "\t<<<Translation of SHARED directives ends>>>\n");
        }
    }

    // Handle misc directives, e.g. barrier, singular, etc.
    // TODO: validate their placement.
    {
        Temporary_Error_Phase ephase("Translation of misc directives");

        if (Verbose)
        {
            fprintf(stderr, "translation of misc directives ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf)
        {
            fprintf(TFile, "\t<<<Translation of misc directives begins>>>\n");
        }

        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            // Skip MK-procedures.
            if (!node->contains_kernel()
                    && node->may_lead_to_kernel()) continue;

            HC_handle_misc_kernel_directives(node);
        }

        if (Trace_IPA || Trace_Perf)
        {
            fprintf(TFile, "\t<<<Translation of misc directives ends>>>\n");
        }
    }

    {
        Temporary_Error_Phase ephase("Directive cleanup");

        if (Verbose)
        {
            fprintf(stderr, "cleaning up directives ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf)
        {
            fprintf(TFile, "\t<<<Directive cleanup begins>>>\n");
        }

        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            // Skip IK-procedures.
            if (node->may_be_inside_kernel()) continue;

            HC_remove_shape_dir(node);
        }

        if (Trace_IPA || Trace_Perf)
        {
            fprintf(TFile, "\t<<<Directive cleanup ends>>>\n");
        }
    }

    // Once the promoted array types are used in data access analysis, we can
    // demote them to ease the CUDA code generation phase.
    IPA_HC_demote_dyn_array_types(&Ipa_gpu_data_prop_pool);

    // Outline each kernel region and identify device functions.
    {
        Temporary_Error_Phase ephase("Kernel Outlining");

        if (Verbose) {
            fprintf(stderr, "kernel outlining ... ");
            fflush(stderr);
        }
        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Kernel Outlining begins>>>\n");
        }

        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            HC_outline_kernels(node);

            // Now, it MUST BE inside kernels.
            if (node->may_be_inside_kernel())
            {
                // IMPORTANT!
                IPA_NODE_CONTEXT context(node);
                ST *func_st = node->Func_ST();

                PU &pu = Pu_Table[ST_pu(func_st)];
                // Identify the function as a CUDA device function.
                Set_PU_is_device(pu);
                // Put the identification in the symbol as well.
                set_st_attr_is_device(ST_st_idx(func_st));
            }
        }

        if (Trace_IPA || Trace_Perf) {
            fprintf(TFile, "\t<<<Kernel Outlining ends>>>\n");
        }
    }

    MEM_POOL_Pop(&Ipa_kernel_prop_pool);
    MEM_POOL_Delete(&Ipa_kernel_prop_pool);

    MEM_POOL_Pop(&Ipa_gpu_data_prop_pool);
    MEM_POOL_Delete(&Ipa_gpu_data_prop_pool);

    // Add some tweaks for good CUDA code generation.
    {
        // Import the global symbol table of cuda_runtime.h.
        const char *crb_name = IPA_HC_Included_Headers_B;
        Is_True(crb_name != NULL, ("Missing include header WHIRL file\n"));
        hc_symtab *hcst = load_global_hc_symtab(crb_name);

        // Make global symbols and types that are declared in cuda_runtime.h
        // so that we do not generate them.
        HC_mark_cuda_runtime_symbols_and_types(hcst);

        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;

            // Flag the main function.
            ST *func_st = node->Func_ST();
            if (strcmp(ST_name(func_st), "main") == 0)
            {
                PU &func_pu = Pu_Table[ST_pu(func_st)];
                Set_PU_is_mainpu(func_pu);
                Set_PU_no_inline(func_pu);
                Set_PU_no_delete(func_pu);
            }
        }
    }

    // Finally run another around of local summary and optimizations.
#if 0
    {
        IPA_NODE_ITER cg_iter(IPA_Call_Graph, PREORDER);
        for (cg_iter.First(); !cg_iter.Is_Empty(); cg_iter.Next())
        {
            IPA_NODE *node = cg_iter.Current();
            if (node == NULL) continue;
            IPA_HC_rebuild_local_summary(node);
        }
    }
#else
    // Since we do not regenerate array summary, we will give later code the
    // impression that array summary has never been collected.
    IPA_Enable_Array_Summary = FALSE;
    IPA_Enable_Array_Sections = FALSE;
#endif

    // Clean-up.
    HCWN_delete_simp_pool();

#endif  // HICUDA
/*** DAVID CODE END ***/

/** DAVID CODE BEGIN **/
    // We must disable inlining.
    IPA_Enable_Inline = IPA_Enable_DCE = FALSE;
/*** DAVID CODE END ***/

    // solve interprocedural inlining
    if (IPA_Enable_Inline || IPA_Enable_DCE)
    {
        MEM_POOL_Popper inline_pool(MEM_local_nz_pool_ptr);

        if (Verbose) {
            fprintf(stderr, "Inlining analysis ...");
            fflush(stderr);
        }

        Temporary_Error_Phase ephase("IPA Inlining Analysis");
        if (Trace_IPA || Trace_Perf)
            fprintf (TFile, "\t<<<Inlining analysis begins>>>\n");
#ifdef TODO
        if( IPA_Enable_Feedback ) {
            setup_IPA_feedback_phase();
            fprintf(IPA_Feedback_prg_fd,"\nINLINING FAILURE INFO\n\n");
        }
#endif

        Perform_Inline_Analysis(IPA_Call_Graph, inline_pool.Pool());

        if (Trace_IPA || Trace_Perf) {
            fprintf (TFile, "\n\tTotal code expansion = %d%%, total prog WHIRL size = 0x%x \n",
                    Orig_Prog_Weight == 0 ? 0 : (Total_Prog_Size - (INT) Orig_Prog_Weight) * 100 / (INT) Orig_Prog_Weight,
                    Total_Prog_Size);
            fprintf (TFile, "\t<<<Inlining analysis completed>>>\n");
        }
#ifdef TODO
        if( IPA_Enable_Feedback ) {
            cleanup_IPA_feedback_phase ();
            fflush(IPA_Feedback_prg_fd);
        }
#endif // TODO

        Ipa_tlog( "Inline", 0, "Count %d", Total_Inlined);
        Ipa_tlog( "Not-Inline", 0, "Count %d", Total_Not_Inlined);
    }

    /* print the call graph */
#ifdef Is_True_On
    CGB_IPA_Terminate();
#endif
}

