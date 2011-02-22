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
#include "assert.h"
#include "defs.h"
#include "mempool.h"
#include "errors.h"
#include "ip_graph.h"
#include "ipa_cg.h"
#include "ipa_df.h"


/*****************************************************************************
 *
 * Add a single exit node to the given graph, and add edges connecting to the
 * exit node. Fill DFS_exit(df).
 *
 * NOTE: DFN is not updated.
 *
 ****************************************************************************/

static void pre_process_graph(GRAPH *g, DFS *df, MEM_POOL *)
{
    // Create an exit node.
    NODE_INDEX exit = g->Add_Node(NULL);
 
    // For each vertex that has no successor, add an edge from it to the exit.
    for (NODE_INDEX v = 0; v < GRAPH_vmax(g); ++v) {
        if (NODE_fcnt(&GRAPH_v_i(g,v)) == -1) continue;
        if (v != exit && g->Num_Succs(v) == 0) g->Add_Edge(v, exit, NULL);
    }

    DFS_exit(df) = exit;
}


/*****************************************************************************
 *
 * Restore the given graph to its original form, so remove the exit node.
 *
 ****************************************************************************/

static void post_process_graph(GRAPH *g, DFS *df)
{
    // We only need to delete the vertices, all incident edges will be
    // eliminated automatically.
    g->Delete_Node(DFS_exit(df));
}


/*****************************************************************************
 *
 * Initialize each node's annotation.
 *
 ****************************************************************************/

void IPA_DATA_FLOW::Init()
{
    /* NOTE: at this point the exit node has not been created. */

    DFN *d = DFS_d(df);

    for (INT32 i = DFN_first(d); i < DFN_end(d); ++i) {
        NODE_INDEX vindex = DFN_v_list_i(d,i);

        // DAVID COMMENT: why do we not init the entry node? is it a dummy
        // node in the call graph as well or an actual procedure?
        if (vindex == DFS_entry(df)) continue;

        InitializeNode(
                NODE_user(&GRAPH_v_i(IPA_Call_Graph->Graph(),vindex)));
    }
}


/*****************************************************************************
 *
 * Perform meet operation on incoming edges, transfer the result to OUT.
 *
 * The parameters passed to meet are the sets being meeted into (IN), the
 * predecessor's current data flow set (pred_set), the annotation obtained
 * from the local phase for the current vertex, and the edge annotation
 * obtained from the local phase. This is for the first iteration. For the
 * next iterations, it reflects the result of the trans operation.
 *
 * For a call graph, local node annotation corresponds to summary information
 * collected for a function or procedure, and edge annotation refers to
 * summary information collected for a call site.
 *
 * The parameters passed to trans is the result of the meet operation and the
 * previous OUT annotation and the variable change which is used to determine
 * if the data flow problem has settled.
 *
 ****************************************************************************/

void IPA_DATA_FLOW::iterative_solver(DFS *df)
{
    NODE_INDEX i;

    // Get the depth-first-numbering of the nodes.
    DFN *d = DFS_d(df);
    assert(d != NULL);

    if (DFS_di(df) == FORWARD)
    {
        /* For all vertices in depth first ordering, perform meet and trans */
        for (i = DFN_first(d); i < DFN_end(d); ++i)
        {
            INT vindex = DFN_v_list_i(d,i);
            // Do nothing for the entry/exit nodes.
            if (vindex == DFS_entry(df) || vindex == DFS_exit(df)) continue;

            /* Compute the meet of incoming edge and existing IN. For the meet
             * operation, pass the IN set, the predecessors' OUT annotation,
             * the current node annotation and current edge annotation. For a
             * call graph and the forward dataflow problem, the current node
             * annotation refers to the callee and the edge annotation refers
             * to the callsite note, the meet operation will delete the old IN
             * and return the new IN.
             */
            Meet(NULL,
                    NODE_user(&GRAPH_v_i(IPA_Call_Graph->Graph(),vindex)),
                    &DFS_change(df));

            /* For the trans operation, pass the IN and OUT set, and the
             * current node annotation (the procedure node for a call graph)
             * and change to determine if the data flow problem has settled.
             * The trans operation will delete the old OUT and return the new
             * OUT.
             */
            Trans(NULL, NULL,
                    NODE_user(&GRAPH_v_i(IPA_Call_Graph->Graph(),vindex)),
                    &DFS_change(df));
        }
    }
    else
    {
        // For a backward problem, go from last to first.
        for (i = DFN_end(d)-1; i >= DFN_first(d); --i)
        {
            INT vindex = DFN_v_list_i(d,i);

            // Do nothing for the entry/exit nodes.
            if (vindex == DFS_entry(df) || vindex == DFS_exit(df)) continue;

            Meet(NULL,
                    NODE_user(&GRAPH_v_i(IPA_Call_Graph->Graph(),vindex)),
                    &DFS_change(df));

            Trans(DFS_in_i(df,vindex), DFS_out_i(df,vindex),
                    NODE_user(&GRAPH_v_i(IPA_Call_Graph->Graph(),vindex)),
                    &DFS_change(df));
        }
    }
}


// When clone nodes are added to the graph during constant propagation
// Depth-First Numbering of nodes needs to be rebuilt.
BOOL IPA_Should_Rebuild_DFN;


/*****************************************************************************
 *
 * The actual working method that solves the dataflow problem.
 *
 ****************************************************************************/

void IPA_DATA_FLOW::dataflow(DFS *df)
{
    NODE_INDEX tmp;
    INT i;

    // Back up the root node of the call graph.
    // DAVID COMMENT: it is not necessary anymore.
    NODE_INDEX root_node = IPA_Call_Graph->Root();

    /* Add the exit node to the call graph.
     * NOTE: the new exit node has not been added to the DFN.
     */
    pre_process_graph(IPA_Call_Graph->Graph(), df, m);

    IPA_Call_Graph->Set_Root(DFS_entry(df));

    // DAVID COMMENT: ???
    if (DFS_di(df) == BACKWARD && DFS_d(df) == NULL) {
        DFS_d(df) = Depth_First_Ordering(IPA_Call_Graph->Graph(), m);
    }

    /* Create and init the IN and OUT annotations for all vertices. */
    DFS_in(df) = (void **)MEM_POOL_Alloc(m,
            sizeof(void*) * GRAPH_vmax(IPA_Call_Graph->Graph()));
    assert(DFS_in(df) != NULL);
    bzero(DFS_in(df), sizeof(void*)*GRAPH_vmax(IPA_Call_Graph->Graph()));

    DFS_out(df) = (void**)MEM_POOL_Alloc(m,
            sizeof(void*)*GRAPH_vmax(IPA_Call_Graph->Graph()));
    assert(DFS_out(df) != NULL);
    bzero(DFS_out(df), sizeof(void*)*GRAPH_vmax(IPA_Call_Graph->Graph()));

    DFS_change(df) = 1;

    while (DFS_change(df))
    {
        /* Reset change to 0. During the trans operation, if the new OUT is
         * different from the old OUT, then the problem has not settled and
         * change must be set to 1.
         */
        DFS_change(df) = 0;
        IPA_Should_Rebuild_DFN = FALSE;

        /* call the iterative dataflow solver */
        iterative_solver(df);

        if (IPA_Should_Rebuild_DFN) {
            DFS_d(df) = Depth_First_Ordering(IPA_Call_Graph->Graph(), m);
        }
    }

    /* this pass is used if any post processing is needed.             */
    /* in the case of constant propagation, the tcons need to be reset */

    DFN* dd = DFS_d(df);
    for ( i=DFN_first(dd); i< DFN_end(dd); ++i ) {
        INT vindex = DFN_v_list_i(dd,i);
        if (vindex == DFS_entry(df) || vindex == DFS_exit(df))
            continue;
        PostProcessIO(NODE_user(&GRAPH_v_i(IPA_Call_Graph->Graph(),vindex)));
    }

    // Remove the exit node from the call graph.
    post_process_graph(IPA_Call_Graph->Graph(), df);  

    // Restore the root node.
    // DAVID COMMENT: not necessary.
    IPA_Call_Graph->Set_Root(root_node);
}


/*****************************************************************************
 *
 * This routine must be invoked to start a data flow problem. It sets up all
 * the fields in the DFS data structure and solves the dataflow problem using
 * an iterative solution. MAIN ENTRY
 *
 ****************************************************************************/

void IPA_DATA_FLOW::Solve()
{
    dataflow(df);
}

/*----------------------------------------------------------------------*/
/* clone a particular node                                              */
/*----------------------------------------------------------------------*/
IPA_NODE* IPA_DATA_FLOW::Clone(IPA_NODE* n)
{
    return IPA_Call_Graph->Create_Clone(n);
}

/*----------------------------------------------------------------------*/
/* the in annotation is the current in for the vertex,                  */
/* edge_in is the in annotation for the incoming edge. vertex is the    */
/* caller, edge is the callsite                                         */
/* return the result of the meet operation, which is the out set        */
/*----------------------------------------------------------------------*/

void* IPA_DATA_FLOW::Meet(void *in, void *vertex, INT *change)
{
    IPA_NODE *n = (IPA_NODE*)vertex;
    fprintf(TFile, "Entered the MEET function\n");

    return NULL;
}

/*----------------------------------------------------------------------*/
/* return the new out set.                                              */
/*----------------------------------------------------------------------*/

void* IPA_DATA_FLOW::Trans(void *, void *, void *vertex, INT *)
{
    /* IPA_NODE *nclone; */
    IPA_NODE *n = (IPA_NODE*)vertex;
    fprintf(TFile, "Entered the trans function\n");

    /* nclone = Clone(n);  */
    return NULL;
}

/*----------------------------------------------------------------------*/
/* get the caller, given the edge                                       */
/*----------------------------------------------------------------------*/
IPA_NODE* IPA_DATA_FLOW::Get_caller(IPA_EDGE *edge)
{
    return IPA_Call_Graph->Caller(edge);
}

/*----------------------------------------------------------------------*/
/* get the callee, given the edge  */
/*----------------------------------------------------------------------*/
IPA_NODE* IPA_DATA_FLOW::Get_callee(IPA_EDGE *edge)
{
    return IPA_Call_Graph->Callee(edge);
}


/*****************************************************************************
 *
 * Constructor for IPA_DATA_FLOW
 *
 ****************************************************************************/

IPA_DATA_FLOW::IPA_DATA_FLOW(DF_DIRECTION ddf, MEM_POOL *mm)
{
    m = mm;
    d = ddf;

    df = (DFS*)MEM_POOL_Alloc(m, sizeof(DFS));
    DFS_di(df) = d;

    /* Build a depth-first-ordering of the call graph. Note that this is only
     * for a forward problem. For a backward problem, build it after the graph
     * is reversed.
     */
    DFS_d(df) = Depth_First_Ordering(IPA_Call_Graph->Graph(), m);

    // The entry node is simply the root of the call graph.
    DFS_entry(df) = IPA_Call_Graph->Root();

    /* NOTE: neither the DFS (including DFN) nor the call graph contains an
     * exit node at this point.
     */
}


//----------------------------------------------------------------------
// print the output after solving the dataflow problem
//----------------------------------------------------------------------
void 
IPA_DATA_FLOW::Print(FILE* fp)
{
  INT i;

  if (df == NULL)
    Fail_FmtAssertion("You cannot print before solving the problem!! \n");

  DFN* dd = DFS_d(df);
  for ( i=DFN_first(dd); i< DFN_end(dd); ++i ) {
    INT vindex = DFN_v_list_i(dd,i);
    if (vindex == DFS_entry(df) || vindex == DFS_exit(df))
      continue;
    if (NODE_user(&GRAPH_v_i(IPA_Call_Graph->Graph(),vindex)) != NULL)
      Print_entry(fp, DFS_out_i(df,vindex), 
		  NODE_user(&GRAPH_v_i(IPA_Call_Graph->Graph(),vindex)));
  }
}

//----------------------------------------------------------------------
// print the output for each entry
//----------------------------------------------------------------------
void 
IPA_DATA_FLOW::Print_entry ( FILE *fp, void *, void *)
{
  fprintf ( fp, "Entered the print_entry function \n" );
}
