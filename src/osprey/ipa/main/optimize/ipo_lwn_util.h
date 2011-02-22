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

/** DAVID CODE BEGIN **/

/*****************************************************************************
 *
 * This is stripped version of be/lno/lwn_util.h, with all LWN_ functions
 * replaced with IPO_LWN_ functions (except macros like LWN_Get/Set_Parent).
 *
 * This is used to resolve conflicts of calling LWN_ functions when both
 * ipa.so and lno.so are loaded into memory.
 *
 ****************************************************************************/

#ifndef _IPO_LWN_UTIL_H_
#define _IPO_LWN_UTIL_H_

#ifndef wn_INCLUDED
#include "wn.h"
#endif
#ifndef wn_util_INCLUDED
#include "wn_util.h"
#endif

#include "fb_whirl.h"

// DAVID COMMENT: This is a stand-alone utility class, in which Parent_Map can
// be any map, not necessarily the one in LNO.
extern WN_MAP Parent_Map;

#define LWN_Get_Parent(wn)	((WN*)WN_MAP_Get(Parent_Map, (WN*)wn))
#define LWN_Set_Parent(wn, p)	(WN_MAP_Set(Parent_Map, wn, (void *)p))

inline void IPO_LWN_Copy_Linenumber(const WN* from, WN* to)
{
  if (OPCODE_has_next_prev(WN_opcode(from)) &&
      OPCODE_has_next_prev(WN_opcode(to))) {
    WN_linenum(to) = WN_linenum(from);
  }
}

#ifdef _USE_OLD_FEEDBACK
inline void IPO_LWN_Set_Frequency(const WN* wn, const INT32 count)
{
  if (Cur_PU_Feedback && WN_opcode(wn) != OPC_BLOCK) 
    WN_MAP32_Set(WN_MAP_FEEDBACK, (WN *) wn, count);
}

inline void IPO_LWN_Copy_Frequency(const WN* wn, const WN* wn_from)
{
  if (Cur_PU_Feedback && WN_opcode(wn) != OPC_BLOCK) 
    WN_MAP32_Set(WN_MAP_FEEDBACK, (WN *) wn, WN_MAP32_Get(WN_MAP_FEEDBACK, wn_from));
}

inline void IPO_LWN_Scale_Frequency(const WN* wn, const float ratio)
{
  if (Cur_PU_Feedback && WN_opcode(wn) != OPC_BLOCK) 
    WN_MAP32_Set(WN_MAP_FEEDBACK, (WN *) wn, (INT32) ratio*WN_MAP32_Get(WN_MAP_FEEDBACK, wn));
}

inline void IPO_LWN_Adjust_Frequency(const WN* wn, const INT32 count)
{
  if (Cur_PU_Feedback && WN_opcode(wn) != OPC_BLOCK) 
    WN_MAP32_Set(WN_MAP_FEEDBACK, (WN *) wn, WN_MAP32_Get(WN_MAP_FEEDBACK, wn) - count);
}

extern void IPO_LWN_Set_Frequency_Tree(const WN* wn, const INT32 count);
extern void IPO_LWN_Copy_Frequency_Tree(const WN* wn, const WN *wn_from);
extern void IPO_LWN_Adjust_Frequency_Tree(const WN* wn, const INT32 count);
extern void IPO_LWN_Scale_Frequency_Tree(const WN* wn, const float ratio);
    
#else

/* to be implemented */
inline void IPO_LWN_Set_Frequency(const WN*, const INT32) {}
inline void IPO_LWN_Copy_Frequency(const WN*, const WN*) {}
inline void IPO_LWN_Scale_Frequency(const WN*, const float) {}
inline void IPO_LWN_Adjust_Frequency(const WN*, const INT32) {}

extern void IPO_LWN_Set_Frequency_Tree(const WN*, const INT32);
extern void IPO_LWN_Copy_Frequency_Tree(const WN*, const WN *);
extern void IPO_LWN_Adjust_Frequency_Tree(const WN*, const INT32);
extern void IPO_LWN_Scale_Frequency_Tree(const WN*, const float);

#endif // _USE_OLD_FEEDBACK


/* Mimic the utilities in wn_util.h */
#define IPO_LWN_ITER 		WN_ITER
#define IPO_LWN_WALK_TreeIter 	WN_WALK_TreeIter
#define IPO_LWN_WALK_TreeNext 	WN_WALK_TreeNext
#define IPO_LWN_WALK_SCFIter 	WN_WALK_SCFIter
#define IPO_LWN_WALK_SCFNext 	WN_WALK_SCFNext
#define IPO_LWN_WALK_StmtIter 	WN_WALK_StmtIter
#define IPO_LWN_WALK_StmtNext 	WN_WALK_StmtNext
#define IPO_LWN_WALK_Abort 		WN_WALK_Abort

extern WN* IPO_LWN_Get_Next_Tree_Node (const WN*);
extern WN* IPO_LWN_Get_Statement(WN *wn);

extern WN* IPO_LWN_Get_Next_Stmt_Node (const WN*);

extern WN* IPO_LWN_Get_Next_SCF_Node (const WN*);

extern void IPO_LWN_Insert_Block_Before(
       WN* block,
       WN* wn,
       WN* in
);

extern void IPO_LWN_Insert_Block_After(
       WN* block,
       WN* wn, 
       WN* in
);

extern void IPO_LWN_Delete_From_Block
  (WN *block, WN* wn);

extern void IPO_LWN_Delete_Tree_From_Block (WN* wn);


extern WN* IPO_LWN_Extract_From_Block(WN* item);
extern WN* IPO_LWN_Extract_From_Block(WN* parent, WN* item);

extern WN* IPO_LWN_Copy_Tree(
       WN *wn, 
       BOOL copy_access=FALSE,
       WN_MAP access_map=0,
       BOOL copy_version=FALSE,
       WN_MAP version_map=0,
       BOOL copy_all_nodes=FALSE
);

extern void IPO_LWN_Parentize (WN* wn);

extern BOOL IPO_LWN_Check_Parentize (const WN* wn);

extern BOOL inside_parallelizable_loop( WN *wn );

extern void IPO_LWN_Delete_Tree(WN *wn);

#ifdef LNO
class DU_MANAGER;
class ARRAY_DIRECTED_GRAPH16;

extern void IPO_LWN_Copy_Def_Use_Node(WN*, WN*, DU_MANAGER*);
extern void IPO_LWN_Copy_Def_Use(WN*, WN*, DU_MANAGER*);
extern void IPO_LWN_Delete_DU(WN *wn);
extern void IPO_LWN_Delete_LNO_dep_graph(WN *wn);
extern void IPO_LWN_Delete_CG_dep_graph(WN *wn);
extern void IPO_LWN_Delete_Name_Manager(WN *wn);
extern void IPO_LWN_Update_Def_Use_Delete_Tree(WN *wn, DU_MANAGER* = NULL);
extern void IPO_LWN_Update_Dg_Delete_Tree(WN *wn, ARRAY_DIRECTED_GRAPH16*);
#endif

#define Block_is_empty(b) (WN_first(b) == NULL)

/* The WN_Create routines */

extern WN *IPO_LWN_CreateStid(OPCODE opc, WN *orig_op, WN *value);

extern WN *IPO_LWN_CreateLdid(OPCODE opc, WN *orig_op);

extern WN *IPO_LWN_CreateDivfloor(TYPE_ID type, WN *kid0, WN *kid1);

extern WN *IPO_LWN_CreateDivceil(TYPE_ID type, WN *kid0, WN *kid1);

extern WN *IPO_LWN_CreateDO(WN *index,
		       WN *start,
		       WN *end,
		       WN *step,
		       WN *body);

extern WN *IPO_LWN_CreateLoopInfo(WN *induction,
		       WN *trip,
		       UINT16 trip_est,
		       UINT16 depth,
		       INT32 flags);

extern WN *IPO_LWN_CreateDoWhile(WN *test,
			  WN *body);

extern WN *IPO_LWN_CreateWhileDo(WN *test,
			  WN *body);

extern WN *IPO_LWN_CreateIf(WN *test,
		       WN *if_then,
		       WN *if_else);

extern WN *IPO_LWN_CreateCondbr( INT32 label_number,
			   WN *exp);

extern WN *IPO_LWN_CreateReturn();

extern WN *IPO_LWN_CreateCompgoto(INT32 num_entries,
			     WN *value,
			     WN *block,
			     WN *deflt);

#ifndef KEY
extern WN *IPO_LWN_CreateIstore(OPCODE opc,
			  WN_OFFSET offset, 
			  TY_IDX ty,
			  WN *value, 
			  WN *addr);
#else
extern WN *IPO_LWN_CreateIstore(OPCODE opc,
			  WN_OFFSET offset, 
			  TY_IDX ty,
			  WN *value, 
			  WN *addr, 
			  UINT field_id = 0);
#endif /* KEY */

extern WN *IPO_LWN_CreateMstore(WN_OFFSET offset,
			   TY_IDX ty,
			   WN *value,
			   WN *addr,
			   WN *num_bytes);

extern WN *IPO_LWN_CreateStid(OPCODE opc,
			 WN_OFFSET offset, 
			 ST* st, 
			 TY_IDX ty, 
			 WN *value);

extern WN *IPO_LWN_CreateEval(WN *exp);

extern WN *IPO_LWN_CreateExp1(OPCODE opc,
			 WN *kid0);

extern WN *IPO_LWN_CreateExp2(OPCODE opc,
			 WN *kid0,
			 WN *kid1);

#ifndef KEY
extern WN *IPO_LWN_CreateIload(OPCODE opc,
			 WN_OFFSET offset, 
			 TY_IDX ty1,
			 TY_IDX ty2,
			 WN *addr);
#else
extern WN *IPO_LWN_CreateIload(OPCODE opc,
			 WN_OFFSET offset, 
			 TY_IDX ty1,
			 TY_IDX ty2,
			 WN *addr, 
			 UINT field_id = 0);
#endif /* KEY */

extern WN *IPO_LWN_CreateMload(WN_OFFSET offset, 
			  TY_IDX ty,
			  WN *addr,
			  WN *num_bytes);

extern WN *IPO_LWN_CreateCvtl(OPCODE opc,
			 INT16 cvtl_bits,
			 WN *kid0);

#ifdef KEY
// Count number of prefetches generated for a particular loop.
extern INT Num_Prefetches;
#endif
extern WN *IPO_LWN_CreatePrefetch (WN_OFFSET offset,
                               UINT32 flag,
                               WN* addr);
extern WN *IPO_LWN_CreateParm(TYPE_ID rtype,
			 WN *parm_node,
			 TY_IDX ty, 
			 UINT32 flag);

extern BOOL Is_Descendent(WN *low, WN *high);

extern WN *IPO_LWN_Int_Type_Conversion(WN *wn, TYPE_ID to_type);
extern TYPE_ID Promote_Type(TYPE_ID mtype);

extern BOOL Tree_Equiv (WN *wn1, WN* wn2);

#ifdef LNO
extern WN *IPO_LWN_Loop_Trip_Count(const WN *loop);

extern BOOL Inside_Loop_With_Goto(WN *wn);
#endif

extern WN* IPO_LWN_Simplify_Tree(WN* wn); 

#endif  // _IPO_LWN_UTIL_H_

/*** DAVID CODE END ***/
