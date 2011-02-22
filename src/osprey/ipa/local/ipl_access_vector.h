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
 * This is a copy of be/lno/access_vector.h, except that all class names are
 * prepended with IPL_. This allows IPL_ACCESS_ARRAY to be used in ipl.so
 * concurrently with ACCESS_ARRAY in lno.so, without naming conflicts.
 *
 ****************************************************************************/

//-*-c++-*- Access Arrays and Vectors --------------
//
// Description:
//
//  These are the basic data structure used to represent array and bounds
//  expressions. They allow us to succintly represent the locations accessed
//  by array instructions and to succintly represent the loop bounds. 
//
//  IPL_ACCESS_ARRAYs are used for two purposes. For array accesses and loop step
//  sizes they represent an expression. For loop upper and lower bounds and
//  for 'ifs' they represent the constraints Ax <= b (where b is the
//  const_offset of the vector).
//
// Exported types and functions:
//
//  IPL_ACCESS_ARRAY
//
//		IPL_ACCESS_ARRAY represents an array of access_vectors. Each access_vector
//		represents a function that maps the loop variables and symbolic
//		variables into a value. Each array statement is represented by an
//		access_array with one access_vector for each dimension of the array.
//		Each loop bound is represented by an access_array with one
//		acccess_vector for each term in the min or max of the bound.
//
//	    BOOL Too_Messy
//
//		Is this array access "bad". If Too_Messy is true, the rest of this
//		structure is undefined.
//
//	    mUINT16 Num_Vec()
//	
//		How many dimensions in this vector? One for each dimension in the
//		array statement.
//
//	    IPL_ACCESS_VECTOR *Dim(UINT16 i)
//	
//		A pointer to dimension 'i' of the IPL_ACCESS_ARRAY
//
// 	    MEM_POOL *Pool() const 
//
//		What pool was used to store this
//
//	    void Print (FILE *fp, BOOL is_bound) const
//
//		Print an ACCES_ARRAY. If is_bound, treat it as a constraint.
//		Otherwise treat it as an expression.
//
// 	    IPL_ACCESS_ARRAY(UINT16 num_vec, UINT16 nest_depth, MEM_POOL *mem_pool=0)
//
//		Create an access array with num_vec dimensions nested nest_depth deep.
//		The array is initialized to TOO_MESSY.
//
// 	    IPL_ACCESS_ARRAY()
//
//		Create a TOO_MESSY IPL_ACCESS_ARRAY, no memory allocated.
//
//	    mUINT16 Non_Const_Loops() const
//
//		What's the outermost loop for which all symbolics and non-linear terms
//		are invariant in that loop and all loops in.  I.e. given a[n]
//		Non_Const_Loops is 0 if 'n' is constant in all the loops,
//		Non_Const_Loops is 1 if 'n' varies in only the outer loop, etc.
//
//		Given a[i], Non_Const_Loops is 0 since i is linear, given a[n*i],
//		Non_Const_Loops is non-zero because 'i' varies.
//
// 	    IPL_ACCESS_ARRAY(const IPL_ACCESS_ARRAY *a, MEM_POOL *pool)
//
//		Copy an array, using pool for the copy.
//
//	    void Set_Array(WN *wn, DOLOOP_STACK *stack)
//
//		Build the access array for an array reference given the array
//		statement wn and given that all the induction variables corresponding
//		to the enclosing do loops are on the stack (inner loops at higher
//		elements).
//
//		This routine assumes that the access vectors for the bounds of all the
//		enclosing loops are valid.
//
//	    void Set_LB(WN *wn, DOLOOP_STACK *stack, INT step)
//
//		Build the access array for a lower bound of a do loop given that wn is
//		an expression. step is the step of the loop.  We need this because if
//		the step is negative, the lower bound is really an upper bound. stack
//		contains the wns of all the loops including the inner.
//
//	    void Set_UB(WN *wn, DOLOOP_STACK *stack)
//
//		Build the access array for an upper bound of a do loop given that wn
//		is the compare operator of the bound. stack contains the wns of all
//		the loops including the inner.
//
// 	    void Init(const IPL_ACCESS_ARRAY *a, MEM_POOL *pool)
//
//		Create a copy of a.
//
//	    BOOL operator ==(const ACCESS_ARAY&) const
//
//		Are the two the same? If either is too_mess return FALSE since we
//		don't really know if they're equal.
//
//  IPL_ACCESS_VECTOR
//
//	  	One dimension of the IPL_ACCESS_ARRAY. This is used for one dimension of
//	  	an array statement or one bound. Each access vector contains a literal
//	  	offset, an array of coefficients (one element per enclosing do loop),
//	  	a list of linear symbolic terms (ie 2*n + 3*m + ...), and a list of
//	  	non-linear symblic terms (ie 2*m*n + ...)
//
//	    BOOL Too_Messy
//
//		Is this dimension "bad". If Too_Messy is true, the rest of this
//		structure is undefined.
//
//	    mUINT16 Nest_Depth() const
//
//		How many do loops enclose this access.
//
//	    mUINT16 Non_Const_Loops() const
//
//		What's the outermost loop for which all symbolics are invariant in
//		that loop and all loops in.  I.e. given a[n] Non_Const_Loops is 0 if
//		'n' is constant in all the loops, Non_Const_Loops is 1 if 'n' varies
//		in only the outer loop, etc.
//
//	    INT32 Loop_Coeff(UINT16 i)
//
//     	Element 'i' of array of coefficients of loop variables.  Loop_Coeff(0)
//     	is the coefficient corresponding to the outermost do loop.
//     	Loop_Coeff(Nest_Depth-1) is the coefficient corresponding to the
//     	innermost do loop.  Given, for example, the array reference a(2*i+j),
//     	where 'i' is the outermost loop and 'j' is the innermost,
//     	Loop_Coeff(0) = 2, Loop_Coeff(1) = 1, Loop_Coeff(x | x >= Nest_Depth)
//     	= 0
//
//	    BOOL Has_Loop_Coeff() const
//
//		If false, all the loop coefficients are zero. If true, they might
//		still all be zero.  This is an efficiency hack to avoid looping in the
//		common case.
//
//	    void Set_Loop_Coeff(UINT16 i,INT32 val)
//
//		Set Loop_Coeff(i) to val
//
//	    IPL_INTSYMB_LIST *Lin_Symb
//
//		This is a list of all the linear symbolic terms in the access vector.
//		Each element in the list is a pair (mINT32,symbol) So given the array
//		reference a(2*i+j+3*n), Lin_Symb would contain one element, the pair
//		(3,symbol for n)
//
//	    BOOL Contains_Lin_Symb() const
//
//		Return TRUE iff  Lin_Symb is not null and not empty
//
//	    IPL_SYMBOL *Delinearized_Symbol
//
//		If this access vector was created by delinearization, what IPL_SYMBOL was
//		factored off.  Ie if we changed [n*i+j] into [i][j],
//		Delinearized_Symbol of the access vector for 'i' will equal 'n'.
//		Dependence analysis needs this field to make sure that two references
//		were delinearized in the same way.
//
//  	IPL_SUMPROD_LIST *Non_Lin_Symb 
//
//		This is a list of all the non-linear terms in sum of product form.
//		Each element in the list is a term. Each term is an mINT32 and a list
//		of symbols.
//
//	    BOOL Contains_Non_Lin_Symb() const
//
//		Return TRUE iff  Non_Lin_Symb is not null and not empty
//
//	    BOOL Is_Const() const
//
//		Is this access_vector constant?
//
// 	    INT64 Const_Offset
//
//		The offset. Given a(i+3), Const_Offset = 3
//
//	    void Print(FILE *fp) const
//
//		Print an ACCES_VECTOR. If is_bound, treat it as a constraint.
//		Otherwise treat it as an expression.
//
//	    void Print_Analysis_Info(FILE *fp, DOLOOP_STACK &do_stack) const
//
//		Print an ACCES_VECTOR. If is_bound, treat it as a constraint.
//		Otherwise treat it as an expression. Don't print the [].
//
//	    IPL_ACCESS_VECTOR(UINT16 nest_depth, MEM_POOL *mem_pool=0) 
//
//      void Mul(INT c)
//
//		Multiply every term in this access vector by c
//
//      void Init(UINT16 lnest, MEM_POOL *mem_pool=0) 
//
//		Create a new IPL_ACCESS_VECTOR, initialized to TOO_MESSY.
//
//	    IPL_ACCESS_VECTOR() 
//		
//		Create a new IPL_ACCESS_VECTOR, initialized to TOO_MESSY.  No memory is
//		allocated.
//
//	    IPL_ACCESS_VECTOR(IPL_ACCESS_VECTOR *a, MEM_POOL *pool)
//	
//		Create a copy of a.
//
//	    IPL_ACCESS_VECTOR *Convert_Bound_To_Exp(MEM_POOL *mem_pool)
//
//		Convert from the bounds form to the expression form, ie move the
//		constant to the other side and zero the last loop variable term.
//
//	    void Set(WN *wn, DOLOOP_STACK *stack, INT8 sign, INT offset=0,
//	            BOOL allow_nonlin=FALSE)
//
//		Build the access vector given a WHIRL expression, wn, and given that
//		all the induction variables corresponding to the enclosing do loops
//		are on the stack (inner loops at higher elements).  The access_vector
//		is multiplied by sign The constant offset is initialized to offset 32
//		bits is sufficient for offset as offset is set by the compiler not the
//		program writer (i.e. it's usually 0,-1 or 1).
//
//		Allow non-linear terms iff allow_nonlin = TRUE
//
//	    void Add(WN *wn, DOLOOP_STACK *stack, INT8 sign)
//
//		Add sign*the term rooted at wn to the access vector
//
//
// 	    void Init(const IPL_ACCESS_VECTOR *a);
//
//		Create a copy of a.
//
//	    BOOL operator ==(const ACCESS_ARAY&) const
//
//		are the two the same If either is too_mess return FALSE since we don't
//		really know if they're equal.
//
//	    void Negate_Me();
//
//		Change sign on all components.
// 
//      IPL_ACCESS_VECTOR(const SYSTEM_OF_EQUATIONS *soe, const INT i,
//              const IPL_SYMBOL_LIST *syms,
//              const INT depth, const INT dim, const INT non_const_loops,
//              const INT which_array, BOOL is_lower_bound, MEM_POOL *pool);
//
//      Create an access vector from SOE.  Used in ARA.
//
//
//  IPL_SYMBOL
//
//		A symbol.
//
//	    struct st *ST_Base() const
//
//		The base pointer of the symbol table entry.
//
//	    struct st *St() const
//
//		An ST* for this symbol (pts to ST_Base() with ST_Offset())
//
//	    INT64 ST_Offset() const
//
//		The ST offset
//
//	    WN_OFFSET WN_Offset() const
//
//		The WHIRL offset
//
//	    OPERATOR< OPERATOR== OPERATOR> OPERATOR!=
//
//		Compare two symbols 
//
//	    TYPE_ID Type
//
//		The descriptor type of the ld used to load this symbol This is
//		undefined if the symbol was not loaded (for example the loop variable
//		is an IDNAME)
//
//	    IPL_SYMBOL(struct st *st, WN_OFFSET offset, TYPE_ID Type)
//
//	    void Print(FILE *fp) const char* Name(char* buf, INT bufsz) const
//	    char* Name() const
//
//      Print the name, either to a file, or into the buffer 'buf' (returning
//      buf) or into a static buffer set aside for use by Name().  If Name()
//      wants to print a string that doesn't fit in bufsz or the static area,
//      then bufsz-1 characters (plus the null terminator) go into the string
//      and a DevWarn is emitted.  Name()'s buffer is 64, so if you think
//      you'll need more, use Name(char*,INT).
//
//	    char* Prompf_Name()
//
//		Return a pointer to name to be used by PROMPF.
//
//	    IPL_SYMBOL(WN *wn)
//
//		Create the symbol described by wn
//
//	    void Init(const WN *wn)
//
//	    IPL_SYMBOL(IPL_SYMBOL *s);
//
//	    void Init(IPL_SYMBOL *s)
//
//	IPL_INTSYMB_LIST
//
//		A list of pairs (INT32 Coeff, IPL_SYMBOL Symbol)
//
// 	    void Print(FILE *fp) const
//
//	    void Init(IPL_INTSYMB_LIST *il,MEM_POOL *mem_pool);
//
//	    BOOL operator ==(const IPL_INTSYMB_LIST&) const
//
//
//	IPL_INTSYMB_NODE
//
//		One element of an IPL_INTSYMB_LIST
//
//	    INT32 Coeff
//
//		An integer multipling the symbol
//
//	    IPL_SYMBOL Symbol
//
//		The symbol.
// 
// 	    void Print(FILE *fp) const
//
//	    IPL_INTSYMB_NODE(IPL_SYMBOL symbol,INT32 coeff) 
//
//		Create an IPL_INTSYMB_NODE with Symbol = symbol and COEFF = coeff
//
//	    IPL_INTSYMB_NODE(IPL_INTSYMB_NODE *in) 
//
//		Create a copy of in
//
//	    BOOL operator ==(const IPL_INTSYMB_NODE&) const
//
//	IPL_INTSYMB_ITER
//
//	IPL_SUMPROD_LIST
//
//		A sum of products.  Each products is an integer times a list of
//		symbols.
//
//  	IPL_SUMPROD_LIST(IPL_SUMPROD_LIST *sp, MEM_POOL *mem_pool) 
//
// 	    void Print(FILE *fp) const
//
//	    void Init(IPL_SUMPROD_LIST *sp)
//
//	    BOOL operator ==(const IPL_SUMPROD_LIST&) const
//
//	    INT Negate_Me() 
//
//		Negate every term on the list, return 0 on error (overflow)
//
//	    void Merge(IPL_SUMPROD_LIST *sl)
//
//		Merge sl into this, leaving sl empty
//
//	IPL_SUMPROD_NODE
//
//		An element of IPL_SUMPROD_LIST
//
//	    IPL_SYMBOL_LIST *Prod_List
//	
//		A list of symbols to be multiplied together
//
//	    INT32 Coeff
//
//		An integer that multiplies the symbols in Prod
//
// 	    void Print(FILE *fp) const
//
// 	    IPL_SUMPROD_NODE(IPL_SYMBOL_LIST *pl,INT32 coeff) 
//
//  	    IPL_SUMPROD_NODE(IPL_SUMPROD_NODE *sp,MEM_POOL *mem_pool)
//
//		Create a copy of sp, use mem_pool for the Prod_List
//
//	    BOOL operator ==(const IPL_SUMPROD_NODE&) const
//
//	IPL_SUMPROD_ITER
//
//	IPL_SYMBOL_LIST
//
//		A list of symbols.
//
// 	    void Print(FILE *fp) const
//
//	    void (const IPL_SYMBOL_LIST *sl, MEM_POOL *mem_pool);
//
//	    void Init(const IPL_SYMBOL_LIST *sl, MEM_POOL *mem_pool);
//
//	    BOOL operator ==(const IPL_SYMBOL_LIST&) const
//
//	    BOOL Contains(const IPL_SYMBOL *s) 
//
//		Is s on the list
//
//	IPL_SYMBOL_NODE
//
//		An element of IPL_SYMBOL_LIST
//
//	    IPL_SYMBOL Symbol
//
//		A symbol
//
//	    mBOOL Is_Loop_Var
//
//		Is this symbol a loop variable
//
// 	    void Print(FILE *fp) const
//
// 	    IPL_SYMBOL_NODE(IPL_SYMBOL symbol, mBOOL is_loop_var)
//
//	    IPL_SYMBOL_NODE(IPL_SYMBOL_NODE *sl) 
//
//		Create a copy of sl
//
//	    BOOL operator ==(const IPL_SYMBOL_NODE&) const
//
//	IPL_SYMBOL_ITER
//
//	DOLOOP_STACK<WN *>
//
//	    A stack of WN pointers to DO loops. The access vector build routines
//	    expect this stack to contain all the enclosing do loops. Outer loops
//	    in position 0.
//
// EXTERNAL FUNCTIONS:
//
//  IPL_INTSYMB_LIST *Subtract(IPL_INTSYMB_LIST *list1, IPL_INTSYMB_LIST *list2,
//          MEM_POOL *mem_pool)
//
//      Return list1 - list2, store the new list in mem_pool Either INPUT list
//		might be NULL. A null return value is equivalent to an empty list.
//
//  IPL_INTSYMB_LIST *Add(IPL_INTSYMB_LIST *list1, IPL_INTSYMB_LIST *list2,
//          MEM_POOL *mem_pool)
//  IPL_INTSYMB_LIST *Mul(INT c, IPL_INTSYMB_LIST *list, MEM_POOL *mem_pool)
//
//		Like Subtract, but list1 + list2 or c*list.
//
//  IPL_ACCESS_VECTOR *Subtract(IPL_ACCESS_VECTOR *av1, IPL_ACCESS_VECTOR *av2,
//          MEM_POOL *mem_pool)
//
//		Return av1 - av2, store the new list in mem_pool. Neither operand may
//		be NULL. A new access vector is returned, allocated from mem_pool.
//
//  IPL_ACCESS_VECTOR *Add(IPL_ACCESS_VECTOR *av1, IPL_ACCESS_VECTOR *av2,
//          MEM_POOL *mem_pool)
//  IPL_ACCESS_VECTOR *Mul(INT c, IPL_ACCESS_VECTOR *av, MEM_POOL *mem_pool)
//
//		Like Subtract, but av1+av2 or c*av.
//
//  void LNO_Build_Access(WN *func_nd, MEM_POOL *pool, BOOL Hoist_Bounds)
//
//		Build the access array for all the array statments and all the do
//		loops in the function. Attach them to the code using LNO_Info_Map. If
//		'Hoist_Bounds' is TRUE, promote bounds so that access vectors which
//		are too messy are expressed in terms of promoted bounds.
//
//  void LNO_Build_Access(WN *wn, DOLOOP_STACK *stack, MEM_POOL *pool,
//          IPL_INDX_RANGE_STACK *irs=0, BOOL Hoist_Bounds)
//
//		Build the access arrays for all the array statements and all the do
//		loops descended from wn. stack must contain all the outer do loops.
//		The bounds of all the outer loops must be set.  irs, if it's set, is
//		used to get bounds on the loops using the array index expressions.
//		Promote_Access same as above. 
//
//  void LNO_Build_Do_Access(WN *wn, DOLOOP_STACK *stack, BOOL Hoist_Bounds)
//
//		Build the access arrays for the bounds of the do loop headed at wn.
//		Map the wn. 'Hoist_Bounds' has the same meaning as is in
//		LNO_Build_Access.
//
//  void LNO_Build_If_Access(WN *wn, DOLOOP_STACK *stack)
//
//		Build the access arrays for the if statement 'wn'. Map the wn.
//
//  void LNO_Build_Access_Array(WN *wn, DOLOOP_STACK *stack, MEM_POOL *pool,
//          IPL_INDX_RANGE_STACK *irs=0)
//
//  void LNO_Print_One_Access(FILE *fp, WN *wn)
//       
//      Print a single access vector.
//	
//  void LNO_Print_Access(FILE *fp, WN *func_nd)
//
//		Print all the access vectors in the routine.
//
//  BOOL Bound_Is_Too_Messy(IPL_ACCESS_ARRAY *aa)
//
//      Returns TRUE if the access array 'aa' has Too_Messy set globally or on
//      any of its dimensions. 
//
//  BOOL Hoist_Lower_Bound(WN* wn_loop, DOLOOP_STACK* stack, MEM_POOL* pool)
//
//		Hoist the lower bound of the loop 'wn_loop' and put it in a statement
//		in front of the loop. The 'stack' is a stack of loops enclosing
//		'wn_loop'.  The 'pool' is used to rebuild the access vector. Returns
//		FALSE if we could not promote the lower bound, TRUE if we could.  
//
//  BOOL Hoist_Upper_Bound(WN* wn_loop, DOLOOP_STACK* stack, MEM_POOL* pool)
//
//		Hoist the upper bound of the loop 'wn_loop' and put it in a statement
//		in front of the loop. The 'stack' is a stack of loops enclosing
//		'wn_loop'.  The 'pool' is used to rebuild the access vector. Returns
//		FALSE if we could not promote the upper bound, TRUE if we could.  
//
//	void Hoist_Bounds_One_Level(WN* wn_tree)
//
//		If bounds any loop of 'wn_tree are "Too Messy", put them into a temp
//		and recompute the access vectors in terms of the new temp value. 
//
//	extern INT Num_Mins(WN *wn)
//	
//		Returns the number of OPR_MIN nodes in 'wn' which are either 'wn'
//		itself or which are OPR_MIN nodes themselves and all parents back to
//		and including 'wn' are also OPR_MIN nodes.
//	
//	extern INT Num_Maxs(WN *wn)
//	
//		Returns the number of OPR_MAX nodes in 'wn' which are either 'wn'
//		itself or which are OPR_MAX nodes themselves and all parents back to
//		and including 'wn' are also OPR_MAX nodes.  
//

/** $Revision: 1.5 $
*** $Date: 04/12/21 14:57:11-08:00 $
*** $Author: bos@eng-25.internal.keyresearch.com $
*** $Source: /home/bos/bk/kpro64-pending/be/lno/SCCS/s.access_vector.h $
**/

#ifndef ipl_access_vector_INCLUDED
#define ipl_access_vector_INCLUDED "ipl_access_vector.h"

#ifdef _KEEP_RCS_ID
static char *access_vector_rcs_id = access_vector_INCLUDED "$Revision: 1.5 $";
#endif /* _KEEP_RCS_ID */

#ifndef wn_INCLUDED
#include "wn.h"
#endif
#ifndef cxx_memory_INCLUDED
#include "cxx_memory.h"
#endif
#ifndef cxx_base_INCLUDED
#include "cxx_base.h"
#endif
#ifndef cxx_template_INCLUDED
#include "cxx_template.h"
#endif
#ifndef stab_INCLUDED
#include "stab.h"
#endif

#include "symtab_compatible.h"

#define MAX_TLOG_CHARS 3000

class SYSTEM_OF_EQUATIONS;
typedef STACK<WN *> DOLOOP_STACK;


/*****************************************************************************
 *
 * IPL_INDX_RANGE is used to find limits on loop variable ranges based on array
 * accesses.
 * 
 * Given a[i] and a[i+2] where <a> is a 10 element array, we know that i's
 * range is at most 8, and we set below Min=0, Max=2, Mult=1
 *
 ****************************************************************************/

struct IPL_INDX_RANGE
{
    // Is there a valid range for this variable?
    mBOOL Valid;
    // Is the Min_Max_Valid (if not, conservatively assume that Min=Max)
    mBOOL Min_Max_Valid;

    INT64 Min;
    INT64 Max;
    INT64 Mult;
    INT64 Size;

    IPL_INDX_RANGE() : Valid(FALSE) {};
    void Union(INT64 offset, BOOL offset_valid, INT64 Mult, INT64 Size);
    INT64 Maxsize() const;
};

typedef STACK<IPL_INDX_RANGE> IPL_INDX_RANGE_STACK;


class IPL_SYMBOL
{
private:

    mBOOL _is_formal;

    union {
        ST *_st;
        INT _formal_number; 
    } u;

    WN_OFFSET _WN_Offset;

public:

    ST* St() const {
        FmtAssert(!_is_formal, ("IPL_SYMBOL::St(): Expecting non-formal"));
        return u._st;
    }

    ST* ST_Base() const {
        FmtAssert(!_is_formal, ("IPL_SYMBOL::ST_Base(): Expecting non-formal"));
        return ST_base(u._st);
    }

    INT64 ST_Offset() const {
        FmtAssert(!_is_formal, ("IPL_SYMBOL::ST_Offset(): Expecting non-formal"));
        return ST_ofst(u._st);
    }

    WN_OFFSET WN_Offset() const { return _WN_Offset; }

    INT Formal_Number() const {
        FmtAssert(_is_formal, ("IPL_SYMBOL::Formal_Number(): Expecting formal"));
        return u._formal_number;
    }

    BOOL Is_Formal() const { return _is_formal; }

    // This type may not represent the symbol's type because an IDNAME node
    // can be used to initialize this symbol and its DESC field is always
    // M_VOID.
    TYPE_ID Type;

    IPL_SYMBOL()
    {
        _is_formal = FALSE; u._st = NULL; _WN_Offset = 0; Type = 0;
    }

    IPL_SYMBOL(ST* st, WN_OFFSET wn_offset, TYPE_ID type)
    {
        _is_formal = FALSE;
        u._st = st;
        _WN_Offset = wn_offset;
        Type = type;
    }

    IPL_SYMBOL(INT formal_number, WN_OFFSET wn_offset, TYPE_ID type)
    {
        _is_formal = TRUE;
        u._formal_number = formal_number;
        _WN_Offset = wn_offset;
        Type = type;
    } 

    IPL_SYMBOL(const IPL_SYMBOL& symbol) { Init(&symbol); }
    IPL_SYMBOL(const IPL_SYMBOL *s) { Init(s); }

    IPL_SYMBOL(const WN *wn) { Init(wn); }

    BOOL operator == (const IPL_SYMBOL &symbol) const
    {
        if (_is_formal != symbol._is_formal) return FALSE; 

        if (_is_formal) {
            return (u._formal_number == symbol.u._formal_number
                    && _WN_Offset == symbol._WN_Offset
                    && Type == symbol.Type);
        }

        if (u._st == NULL || symbol.u._st == NULL) {
            return (u._st == symbol.u._st && _WN_Offset == symbol._WN_Offset);
        }

        return (ST_Base() == symbol.ST_Base()
                && ST_Offset() == symbol.ST_Offset()
                && WN_Offset() == symbol.WN_Offset());
    }

    BOOL operator != (const IPL_SYMBOL &symbol) const {
        return (!(*this == symbol));
    }

    IPL_SYMBOL& operator=(const IPL_SYMBOL& symbol)
    {
        Init(&symbol);
        return(*this);
    }

    void Init(const WN *wn)
    {
        FmtAssert(OPCODE_has_sym(WN_opcode(wn)),
                ("IPL_SYMBOL::Init(WN*) called with opcode %d", WN_opcode(wn)));

        _is_formal = FALSE;
        u._st = WN_st(wn);

        if (WN_operator(wn) == OPR_CONST || WN_operator(wn) == OPR_INTCONST) {
            _WN_Offset = 0;
            Type = WN_rtype(wn);
        } else {
            _WN_Offset = WN_offset(wn);
            Type = WN_desc(wn);
        }
    }

    void Init(const IPL_SYMBOL *s)
    {
        _is_formal = s->_is_formal; 
        if (_is_formal) {
            u._formal_number = s->u._formal_number;
        } else {
            u._st = s->u._st;
        }
        _WN_Offset = s->_WN_Offset;
        Type = s->Type;
    }

/** DAVID CODE BEGIN **/
    /* Construct a WHIRL node from this. */
    WN* to_wn();
/*** DAVID CODE END ***/

    /* The following functions are defined in access_vector.cxx. */

    void Print(FILE *fp) const;
    INT Print(char* bf, INT ccount) const;
    char* Name() const;
    char* Prompf_Name() const;
    char* Name(char* buf, INT bufsz) const;
};


/*****************************************************************************
 *
 * A list of pairs (Coeff, Symbol), which represents
 *      Coeff(0)*Symbol(0) + ... + ... Coeff(n)*Symbol(n)
 *
 ****************************************************************************/

class IPL_INTSYMB_NODE: public SLIST_NODE
{
    DECLARE_SLIST_NODE_CLASS(IPL_INTSYMB_NODE);

public:

    IPL_SYMBOL Symbol;    // NOTE: here it is not a pointer!
    INT32 Coeff;

    IPL_INTSYMB_NODE(IPL_SYMBOL symbol,INT32 coeff) {
        Symbol = symbol;
        Coeff = coeff;
    }

    IPL_INTSYMB_NODE(IPL_INTSYMB_NODE *in) {
        Symbol = in->Symbol;
        Coeff = in->Coeff;
    }

    ~IPL_INTSYMB_NODE() {}

    BOOL operator ==(const IPL_INTSYMB_NODE &i) const {
        return (Coeff == i.Coeff) && (Symbol == i.Symbol);
    }

/** DAVID CODE BEGIN **/
    /* Construct a WHIRL node from this. */
    WN* to_wn();
/*** DAVID CODE END ***/

    void Print(FILE* fp) const;
    INT Print(char* bf, INT ccount) const;
};

class IPL_INTSYMB_LIST: public SLIST
{
    DECLARE_SLIST_CLASS(IPL_INTSYMB_LIST, IPL_INTSYMB_NODE);

public:

    /* Make a deep copy of "il". */
    void Init(IPL_INTSYMB_LIST *il, MEM_POOL *mem_pool);

    friend IPL_INTSYMB_LIST *Subtract(IPL_INTSYMB_LIST *, IPL_INTSYMB_LIST *,
            MEM_POOL *mem_pool);
    friend IPL_INTSYMB_LIST *Add(IPL_INTSYMB_LIST *, IPL_INTSYMB_LIST *,
            MEM_POOL *mem_pool);
    friend IPL_INTSYMB_LIST *Mul(INT, IPL_INTSYMB_LIST *, MEM_POOL *mem_pool);

    void Print(FILE* fp) const;
    INT Print(char* bf, INT ccount) const;

    BOOL operator ==(const IPL_INTSYMB_LIST &symbol_list) const;

    ~IPL_INTSYMB_LIST();
};

class IPL_INTSYMB_ITER: public SLIST_ITER
{
    DECLARE_SLIST_ITER_CLASS(IPL_INTSYMB_ITER, IPL_INTSYMB_NODE ,IPL_INTSYMB_LIST);

public:

    ~IPL_INTSYMB_ITER() {}
};

class IPL_INTSYMB_CONST_ITER: public SLIST_ITER
{
    DECLARE_SLIST_CONST_ITER_CLASS(
            IPL_INTSYMB_CONST_ITER, IPL_INTSYMB_NODE ,IPL_INTSYMB_LIST);

public:

    ~IPL_INTSYMB_CONST_ITER() {}
};


/*****************************************************************************
 *
 * A list of symbols multiplied together
 *
 ****************************************************************************/

class IPL_SYMBOL_NODE: public SLIST_NODE
{
    DECLARE_SLIST_NODE_CLASS(IPL_SYMBOL_NODE);

public:

    IPL_SYMBOL Symbol;      // This is not a pointer!
    mBOOL Is_Loop_Var;

    IPL_SYMBOL_NODE(const IPL_SYMBOL_NODE *sl) {
        Symbol = sl->Symbol;
        Is_Loop_Var = sl->Is_Loop_Var;
    }

    IPL_SYMBOL_NODE(const IPL_SYMBOL& symbol, mBOOL is_loop_var) {
        Symbol = symbol;
        Is_Loop_Var = is_loop_var;
    }

    ~IPL_SYMBOL_NODE() {}

    BOOL operator ==(const IPL_SYMBOL_NODE& sn) const {
        return (Symbol == sn.Symbol); 
    }

    void Print(FILE* fp) const;
    INT Print(char* bf, INT ccount) const;
};

class IPL_SYMBOL_LIST : public SLIST
{
    DECLARE_SLIST_CLASS(IPL_SYMBOL_LIST, IPL_SYMBOL_NODE);

public:

    IPL_SYMBOL_LIST(IPL_SYMBOL_LIST *sl, MEM_POOL *mem_pool) { Init(sl,mem_pool); }
    ~IPL_SYMBOL_LIST();

    void Init(const IPL_SYMBOL_LIST *sl, MEM_POOL *mem_pool);

    void Print(FILE* fp, BOOL starsep = FALSE) const;
    INT Print(char* bf, INT ccount, BOOL starsep = FALSE) const;

    BOOL operator ==(const IPL_SYMBOL_LIST&) const;
    BOOL Contains(const IPL_SYMBOL *s);
};

class IPL_SYMBOL_ITER : public SLIST_ITER
{
    DECLARE_SLIST_ITER_CLASS(IPL_SYMBOL_ITER, IPL_SYMBOL_NODE, IPL_SYMBOL_LIST);

public:

    ~IPL_SYMBOL_ITER() {}
};

class IPL_SYMBOL_CONST_ITER : public SLIST_ITER
{
    DECLARE_SLIST_CONST_ITER_CLASS(
            IPL_SYMBOL_CONST_ITER, IPL_SYMBOL_NODE, IPL_SYMBOL_LIST);

public:

    ~IPL_SYMBOL_CONST_ITER() {}
};


/*****************************************************************************
 *
 * A list of non-linear symbolics in sum of products form
 *
 ****************************************************************************/

class IPL_SUMPROD_NODE: public SLIST_NODE
{
    DECLARE_SLIST_NODE_CLASS(IPL_SUMPROD_NODE);

public:

    IPL_SYMBOL_LIST *Prod_List;
    INT32 Coeff;  // An integer that multiplies the symbols in Prod_List


    IPL_SUMPROD_NODE(IPL_SYMBOL_LIST *pl, INT32 coeff) { 
        Prod_List = pl; Coeff = coeff;
    }

    IPL_SUMPROD_NODE(IPL_SUMPROD_NODE *sp, MEM_POOL *mem_pool) { 
        Prod_List = CXX_NEW(IPL_SYMBOL_LIST(sp->Prod_List, mem_pool), mem_pool);
        Coeff = sp->Coeff;
    }

    BOOL operator ==(const IPL_SUMPROD_NODE& sp) const {
        if (Coeff != sp.Coeff) return FALSE;
        return (*Prod_List == *sp.Prod_List);
    }

    ~IPL_SUMPROD_NODE() {}

/** DAVID CODE BEGIN **/
    /* Construct a WHIRL node from this. */
    WN* to_wn();
/*** DAVID CODE END ***/

    void Print(FILE* fp) const;
    INT Print(char* bf, INT ccount) const;
};

class IPL_SUMPROD_LIST: public SLIST
{
    DECLARE_SLIST_CLASS(IPL_SUMPROD_LIST, IPL_SUMPROD_NODE);

public:

    BOOL operator ==(const IPL_SUMPROD_LIST&) const;

    INT Negate_Me() ;
    void Merge(IPL_SUMPROD_LIST *sl);

    void Print(FILE* fp) const;
    INT Print(char* bf, INT ccount) const;
    void Init(IPL_SUMPROD_LIST *sp, MEM_POOL *mem_pool);

    IPL_SUMPROD_LIST(IPL_SUMPROD_LIST *sp, MEM_POOL *mem_pool) { Init(sp,mem_pool); }
    ~IPL_SUMPROD_LIST();
};

class IPL_SUMPROD_ITER : public SLIST_ITER
{
    DECLARE_SLIST_ITER_CLASS(IPL_SUMPROD_ITER, IPL_SUMPROD_NODE, IPL_SUMPROD_LIST);

public:

    ~IPL_SUMPROD_ITER() {}
};

class IPL_SUMPROD_CONST_ITER : public SLIST_ITER
{
    DECLARE_SLIST_CONST_ITER_CLASS(
            IPL_SUMPROD_CONST_ITER, IPL_SUMPROD_NODE, IPL_SUMPROD_LIST);

public:

    ~IPL_SUMPROD_CONST_ITER() {}
};


class IPL_ACCESS_VECTOR
{
    MEM_POOL *_mem_pool;

    mINT32 *_lcoeff;        // linear coefficients of loop variables
    mUINT16 _nest_depth;    // # of DO_LOOPs surrounding this access
    mUINT16 _non_const_loops;

    void Add_Sum(WN *wn, INT64 coeff, DOLOOP_STACK *stack,
            BOOL allow_nonlin = FALSE);

    IPL_SUMPROD_LIST *Add_Nonlin(WN *wn,IPL_SUMPROD_LIST *list, DOLOOP_STACK *stack);

    IPL_ACCESS_VECTOR(const IPL_ACCESS_VECTOR&) {}
    IPL_ACCESS_VECTOR& operator=(const IPL_ACCESS_VECTOR&);

public:

    BOOL Too_Messy;     // is this vector too messy

    mUINT16 Non_Const_Loops() const { return _non_const_loops; }

    void Set_Non_Const_Loops(const mUINT16 i) { _non_const_loops = i; }
    void Max_Non_Const_Loops(INT val) { 
        _non_const_loops = MAX(_non_const_loops, val); 
    };
    void Update_Non_Const_Loops(WN *wn, DOLOOP_STACK *stack);

    IPL_INTSYMB_LIST *Lin_Symb;     // linear symbolic terms
    IPL_SUMPROD_LIST *Non_Lin_Symb; // non-linear symbolic terms
    INT64 Const_Offset;

    mUINT16 Nest_Depth() const { return _nest_depth; }

    void Set_Nest_Depth(mUINT16 nest_depth) 
    {
        if (_lcoeff != NULL && _nest_depth < nest_depth) {
            // Expand the coeff array.
            mINT32 * newlcoeff = CXX_NEW_ARRAY(mINT32, nest_depth, _mem_pool);
            INT i = 0;
            for ( ; i < _nest_depth; ++i) newlcoeff[i] = _lcoeff[i];
            for ( ; i < nest_depth; ++i) newlcoeff[i] = 0;
            CXX_DELETE_ARRAY(_lcoeff, _mem_pool);
            _lcoeff = newlcoeff;
        }
        _nest_depth = nest_depth;
    }

    BOOL Is_Const() const;

    void Set(WN *wn, DOLOOP_STACK *stack, INT8 sign, INT offset = 0,
            BOOL allow_nonlin=FALSE);

    void Add(WN *wn, DOLOOP_STACK *stack, INT8 sign);

    void Mul(INT c);

    BOOL operator ==(const IPL_ACCESS_VECTOR &av) const;

    BOOL Contains_Lin_Symb() const {
        return (Lin_Symb != NULL && !Lin_Symb->Is_Empty());
    }

    IPL_SYMBOL *Delinearized_Symbol;

    void Add_Symbol(INT64 coeff, IPL_SYMBOL symbol, DOLOOP_STACK *stack, WN *wn);

    BOOL Contains_Non_Lin_Symb() const { 
        return (Non_Lin_Symb != NULL && !Non_Lin_Symb->Is_Empty());
    }

    void Update_Non_Const_Loops_Nonlinear(DOLOOP_STACK *stack);

    void Substitute(INT formal_number, WN* wn_sub, DOLOOP_STACK* stack,
            BOOL allow_nonlin = FALSE);

    INT32 Loop_Coeff(UINT16 i) const {
        return (_lcoeff != NULL && i < _nest_depth) ? _lcoeff[i] : 0;
    }

    void Set_Condition(WN *wn, DOLOOP_STACK *stack, BOOL negate);

    BOOL Has_Loop_Coeff() const { return _lcoeff != NULL; }
    void Set_Loop_Coeff(UINT16 i, INT32 val);

    void Print(FILE *fp, BOOL is_bound=FALSE, BOOL print_brackets=TRUE) const;
    INT Print(char* bf, INT ccount, BOOL is_bound=FALSE, 
            BOOL print_brackets=TRUE) const;
    void Print_Analysis_Info(FILE *fp, DOLOOP_STACK &do_stack, 
            BOOL is_bound=FALSE) const;

    IPL_ACCESS_VECTOR(UINT16 lnest, MEM_POOL *mem_pool = NULL) {
        Init(lnest, mem_pool);
    }
    IPL_ACCESS_VECTOR() { 
        Too_Messy = TRUE; _lcoeff=NULL; 
        Lin_Symb=NULL; Non_Lin_Symb=NULL; 
        Delinearized_Symbol = NULL;
    }

#ifdef LNO
    BOOL Can_Delinearize(WN *wn, const IPL_SYMBOL *delin_symbol);
#endif

    /* Make a deep copy of the given access vector. */
    IPL_ACCESS_VECTOR(const IPL_ACCESS_VECTOR *a, MEM_POOL *pool);

    void Init(UINT16 nest_depth, MEM_POOL *mem_pool = NULL) {
        _mem_pool = mem_pool;
        Too_Messy = TRUE;
        _nest_depth = nest_depth;
        _lcoeff = NULL; Lin_Symb = NULL; Non_Lin_Symb = NULL;
        Const_Offset = 0;
        _non_const_loops = 0;
        Delinearized_Symbol = NULL;
    }

    /* Make a deep copy of the given access vector. */
    void Init(const IPL_ACCESS_VECTOR *a, MEM_POOL *pool);

    friend IPL_ACCESS_VECTOR *Subtract(IPL_ACCESS_VECTOR *, IPL_ACCESS_VECTOR *,
            MEM_POOL *mem_pool);
    friend IPL_ACCESS_VECTOR *Add(IPL_ACCESS_VECTOR *, IPL_ACCESS_VECTOR *,
            MEM_POOL *mem_pool);
    friend IPL_ACCESS_VECTOR *Mul(INT c, IPL_ACCESS_VECTOR *av, MEM_POOL *mem_pool);
    friend IPL_ACCESS_VECTOR *Merge(IPL_ACCESS_VECTOR *, IPL_ACCESS_VECTOR *,
            MEM_POOL *mem_pool);

    void Negate_Me();

    IPL_ACCESS_VECTOR *Convert_Bound_To_Exp(MEM_POOL *mem_pool);
    IPL_ACCESS_VECTOR(const SYSTEM_OF_EQUATIONS *soe,
            const INT i, const IPL_SYMBOL_LIST *syms,
            const INT depth, const INT dim,
            const INT non_const_loops,
            const INT which_array,
            BOOL is_lower_bound, MEM_POOL *pool);

    ~IPL_ACCESS_VECTOR() { 
        if (_lcoeff) CXX_DELETE_ARRAY(_lcoeff,_mem_pool); 
        if (Lin_Symb) {
            MEM_POOL_Set_Default(_mem_pool);
            CXX_DELETE(Lin_Symb,_mem_pool);
        }
        if (Non_Lin_Symb) {
            MEM_POOL_Set_Default(_mem_pool);
            CXX_DELETE(Non_Lin_Symb,_mem_pool);
        }
    }

    BOOL Has_Formal_Parameter();

/** DAVID CODE BEGIN **/
#ifdef LNO
    /* Divide this access vector by a positive integer. If successful, this
     * access vector will hold the remainder, and the quotient is returned.
     *
     * "wn" is an address expression whose base address is a pointer to an
     * array, one of whose dimensions is "divisor".
     *
     * Memory is allocated using the mempool for this access vector.
     */
    IPL_ACCESS_VECTOR* divide_by_const(UINT divisor, WN *wn);
#endif  // LNO

    /* Perfect-Divide this access vector by a positive integer. If successful,
     * return TRUE and this vector is the quotient. Otherwise, this vector is
     * intact and FALSE is returned.
     */
    BOOL perfect_divide_by_const(UINT divisor);

    /* Construct a WHIRL node for this access vector. The DO_LOOP stack is
     * used to look up loop index variables.
     */
    WN* to_wn(DOLOOP_STACK *stack);
/*** DAVID CODE END ***/
};


class IPL_ACCESS_ARRAY
{
    IPL_ACCESS_VECTOR *_dim;
    MEM_POOL *_mem_pool;
    mUINT16 _num_vec;
    IPL_ACCESS_ARRAY(const IPL_ACCESS_ARRAY&) {}
    IPL_ACCESS_ARRAY& operator=(const IPL_ACCESS_ARRAY&);

public:

    BOOL Too_Messy;  // is this array too messy

    mUINT16 Num_Vec() const { return _num_vec; }

    IPL_ACCESS_VECTOR *Dim(UINT16 i) const {
        Is_True(i < _num_vec,("Bad index in IPL_ACCESS_ARRAY::Dim"));
        return &(_dim[i]);
    }

    IPL_ACCESS_ARRAY(UINT16 num_vec,UINT16 nest_depth,MEM_POOL *mem_pool); 
    IPL_ACCESS_ARRAY(UINT16 num_vec,IPL_ACCESS_VECTOR* dim[],MEM_POOL *mem_pool); 
    void Print(FILE *fp, BOOL is_bound=FALSE) const __attribute__((weak));
    IPL_ACCESS_ARRAY() { Too_Messy = TRUE; _dim = NULL; _num_vec=0;}
    IPL_ACCESS_ARRAY(const IPL_ACCESS_ARRAY *a, MEM_POOL *pool); 
    mUINT16 Non_Const_Loops() const;
    void Set_Array(WN *wn, DOLOOP_STACK *stack);
    void Set_LB(WN *wn, DOLOOP_STACK *stack, INT64 step);
    void Set_UB(WN *wn, DOLOOP_STACK *stack);
    void Init(const IPL_ACCESS_ARRAY *a, MEM_POOL *pool);
    ~IPL_ACCESS_ARRAY() { CXX_DELETE_ARRAY(_dim,_mem_pool); }
    MEM_POOL *Pool() const { return _mem_pool; }
    BOOL operator ==(const IPL_ACCESS_ARRAY& a) const;
    //  friend void LNO_Build_If_Access(WN *wn, DOLOOP_STACK *stack);
    INT Set_IF(WN *wn, DOLOOP_STACK *stack, BOOL negate, BOOL is_and, INT i);
    void Substitute(INT formal_number, WN* wn_sub, DOLOOP_STACK* stack,
            BOOL allow_nonlinear = FALSE);
    BOOL Has_Formal_Parameter();

private:

    INT Set_UB_r(WN *wn, DOLOOP_STACK *stack, INT i, INT sign);
    INT Set_LB_r(WN *wn, DOLOOP_STACK *stack, INT i, INT64 step);
#ifdef LNO
    void Delinearize(DOLOOP_STACK *stack, WN *wn);
    INT Delinearize(DOLOOP_STACK *stack, INT dim, WN *wn);
    INT Delinearize(DOLOOP_STACK *stack, INT dim, const IPL_SYMBOL *delin_symbol);
#endif
    void Update_Non_Const_Loops(WN *wn,DOLOOP_STACK *stack);
};

extern INT  Num_Mins(WN *wn);
extern INT  Num_Maxs(WN *wn);
extern INT  Num_Lands(WN* wn);
extern INT  Num_Liors(WN* wn);

extern INT Num_Upper_Bounds(WN* wn);            
extern INT Num_Lower_Bounds(WN* wn, IPL_ACCESS_VECTOR *step);

extern INT snprintfs(char* buf, INT ccount, INT tcount, const char* fstring);
extern INT snprintfd(char* buf, INT ccount, INT tcount, INT32 value);
extern INT snprintfll(char* buf, INT ccount, INT tcount, INT64 value);
extern INT snprintfx(char* buf, INT ccount, INT tcount, INT32 value);

#endif

/*** DAVID CODE END ***/
