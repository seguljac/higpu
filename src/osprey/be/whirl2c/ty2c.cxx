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


/* ====================================================================
 * ====================================================================
 *
 * Module: ty2c.c
 * $Revision: 1.1.1.1 $
 * $Date: 2005/10/21 19:00:00 $
 * $Author: marcel $
 * $Source: /proj/osprey/CVS/open64/osprey1.0/be/whirl2c/ty2c.cxx,v $
 *
 * Revision history:
 *  07-Oct-94 - Original Version
 *
 * Description:
 *
 *    See ty2c.h for a description of the exported functions and 
 *    variables.
 *
 * ====================================================================
 * ====================================================================
 */
#ifdef _KEEP_RCS_ID
static char *rcs_id = "$Source: /proj/osprey/CVS/open64/osprey1.0/be/whirl2c/ty2c.cxx,v $ $Revision: 1.1.1.1 $";
#endif /* _KEEP_RCS_ID */

#include "whirl2c_common.h"
#include "PUinfo.h"
#include "ty2c.h"
#include "tcon2c.h"

extern BOOL W2C_Emit_OpenCL;
extern int openCL_kernel_function;
extern int openCL_device_function;

/*------------------ macros, types, and local state -------------------*/
/*---------------------------------------------------------------------*/


/* Scalar_C_Names[] is a mapping from a scalar mtype to the builtin
 * C type we have chosen to represent it.  This mapping relies on
 * a certain ordinal numbering between the MTYPE definitions in 
 * common/tdt/mtypes.h, and any changes in that ordering must be
 * reflected in a corresponding change here.
 */
typedef struct SCALAR_C_NAME
{
   const char *real_name;
   const char *pseudo_name;
} SCALAR_C_NAME;


/* This name is used to wrap a union around a block containing
 * equivalence fields.  Should only occur when translating
 * Fortran to C.
 */
const char TY2C_Aligned_Block_Name[] = "__block";

#ifndef TARG_X8664
#define MTYPE_PREDEF MTYPE_F16
#else
#define MTYPE_PREDEF MTYPE_V16F8
#endif /* TARG_X8664 */

static char Name_Unknown_Type[] = "__UNKNOWN_TYPE";
static const SCALAR_C_NAME Scalar_C_Names[MTYPE_LAST + 1] =
   {{"void",               Name_Unknown_Type},  /* MTYPE_UNKNOWN 0 */
/** DAVID CODE BEGIN **/
#if 0
    {"char",               "_BOOLEAN"},         /* MTYPE_B = 1 */
    {"signed char",        "_INT8"},            /* MTYPE_I1 = 2 */
    {"signed short",       "_INT16"},           /* MTYPE_I2 = 3 */
    {"signed int",         "_INT32"},           /* MTYPE_I4 = 4 */
    {"signed long long",   "_INT64"},           /* MTYPE_I8 = 5 */
    {"unsigned char",      "_UINT8"},           /* MTYPE_U1 = 6 */
    {"unsigned short",     "_UINT16"},          /* MTYPE_U2 = 7 */
    {"unsigned int",       "_UINT32"},          /* MTYPE_U4 = 8 */
    {"unsigned long long", "_UINT64"},          /* MTYPE_U8 = 9 */
    {"float",              "_IEEE32"},          /* MTYPE_F4 = 10 */
    {"double",             "_IEEE64"},          /* MTYPE_F8 = 11 */
#else
    /* We emit readable types directly, skipping whirl2c.h. */
    {"char",               "char"},         /* MTYPE_B = 1 */
    {"signed char",        "signed char"},            /* MTYPE_I1 = 2 */
    {"signed short",       "short"},           /* MTYPE_I2 = 3 */
    {"signed int",         "int"},           /* MTYPE_I4 = 4 */
    {"signed long long",   "long long"},           /* MTYPE_I8 = 5 */
    {"unsigned char",      "unsigned char"},           /* MTYPE_U1 = 6 */
    {"unsigned short",     "unsigned short"},          /* MTYPE_U2 = 7 */
    {"unsigned int",       "unsigned int"},          /* MTYPE_U4 = 8 */
    {"unsigned long long", "unsigned long long"},          /* MTYPE_U8 = 9 */
    {"float",              "float"},          /* MTYPE_F4 = 10 */
    {"double",             "double"},          /* MTYPE_F8 = 11 */
#endif
/*** DAVID CODE END ***/
#ifndef TARG_X8664
    {Name_Unknown_Type,    "_IEEE80"},          /* MTYPE_F10 = 12 */
    {Name_Unknown_Type,    "_IEEE128"}  /* MTYPE_F16 = 13 = MTYPE_PREDEF */
#else
    {Name_Unknown_Type,        ""},             /* MTYPE_V16C4 = 12 */
    {Name_Unknown_Type,        "_IEEE128"},     /* MTYPE_F16 = 13 */
    {Name_Unknown_Type,        ""},             /* MTYPE_STRING = 14 */
    {Name_Unknown_Type,        ""},             /* MTYPE_FQ = 15 */
    {Name_Unknown_Type,        ""},             /* MTYPE_M = 16 */
    {"_Complex float",	   "_COMPLEX32"},	/* MTYPE_C4 = 17 */
    {"_Complex double",	   "_COMPLEX64"},		/* MTYPE_C8 = 18 */
    {Name_Unknown_Type,        ""},             /* MTYPE_CQ = 19 */
    {Name_Unknown_Type,        ""},             /* MTYPE_V = 20 */
    {Name_Unknown_Type,        ""},             /* MTYPE_BS = 21 */
    {Name_Unknown_Type,        ""},             /* MTYPE_A4 = 22 */
    {Name_Unknown_Type,        ""},             /* MTYPE_A8 = 23 */
    {"_Complex long double", "_COMPLEX80"},	/* MTYPE_C10 = 24 */
    {Name_Unknown_Type,        ""},             /* MTYPE_C16 = 25 */
    {Name_Unknown_Type,        ""},             /* MTYPE_I16 = 26 */
    {Name_Unknown_Type,        ""}		/* MTYPE_U16 = 27 */
   ,{Name_Unknown_Type,        "_CMPLX8[2]"},   /* MTYPE_V16C4 = 24 */
    {Name_Unknown_Type,        "_CMPLX16[1]"},  /* MTYPE_V16C8 = 25 */
    {"signed char[16]",        "_INT8[16]"},    /* MTYPE_V16I1 = 26 */
    {"signed short[8]",        "_INT16[8]"},    /* MTYPE_V16I2 = 27 */
    {"signed int[4]",          "_INT32[4]"},    /* MTYPE_V16I4 = 28 */
    {"signed long long int[2]","_INT64[2]"},    /* MTYPE_V16I8 = 29 */
    {"float[4]",               "_IEEE32[4]"},   /* MTYPE_V16F4 = 30 */
    {"double[2]",              "_IEEE64[2]"}    /* MTYPE_V16F8 = 31 */
#endif /* TARG_X8664 */
   }; /* Scalar_C_Names */


/* The following is a mapping from some special types to their
 * names.  This allows us to predefine them and give them a
 * name of our choice (by means of a typedef).  These variables 
 * are initialized by TY2C_initialize().  GET_SPECIAL_TYPENAME
 * returns the type name if the given mtype is one that we
 * treat specially; otherwise, it returns NULL.
 */
const char Special_Void_TypeName[] = "void";
const char Special_String_TypeName[] = "_STRING";
const char Special_Quad_TypeName[] = "_QUAD";
const char Special_Complex32_TypeName[] = "_COMPLEX32";
const char Special_Complex64_TypeName[] = "_COMPLEX64";
const char Special_ComplexQD_TypeName[] = "_COMPLEXQD";

#define GET_SPECIAL_TYPENAME(mtype)\
   ((mtype) == MTYPE_V? Special_Void_TypeName : \
    ((mtype) == MTYPE_STR? Special_String_TypeName : \
     ((mtype) == MTYPE_FQ? Special_Quad_TypeName : \
      ((mtype) == MTYPE_C4? Special_Complex32_TypeName : \
       ((mtype) == MTYPE_C8? Special_Complex64_TypeName : \
	((mtype) == MTYPE_CQ? Special_ComplexQD_TypeName : \
	 (const char *)NULL))))))

#define PTR_OR_ALIGNED_WITH_STRUCT(fld_ty, struct_align) \
   (TY_Is_Pointer(fld_ty) || TY_align(fld_ty) <= struct_align)

/* Some forward declarations */
static void TY2C_scalar(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context);
static void TY2C_array(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context);
static void TY2C_struct(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context);
static void TY2C_function(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context);
static void TY2C_pointer(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context);
static void TY2C_void(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context);
static void TY2C_invalid(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context);


/* TY2C_Handle[] maps a TY_kind to a function that translates
 * a type of the given type into C code.  Should the ordinal
 * numbering of the KIND change in "../common/com/stab.h", then
 * a corresponding change must be made here.
 */
typedef void (*TY2C_HANDLER_FUNC)(TOKEN_BUFFER, TY_IDX, CONTEXT);

static const TY2C_HANDLER_FUNC 
   TY2C_Handle[KIND_LAST/*TY_KIND*/] =
{
   &TY2C_invalid,   /* KIND_INVALID */
   &TY2C_scalar,    /* KIND_SCALAR */
   &TY2C_array,     /* KIND_ARRAY */
   &TY2C_struct,    /* KIND_STRUCT */
   &TY2C_pointer,   /* KIND_POINTER */
   &TY2C_function,  /* KIND_FUNCTION */
   &TY2C_void,      /* KIND_VOID */
};


/*------------------------- exported objects --------------------------*/
/*---------------------------------------------------------------------*/


/* These must match the names given in "stab.c" for the complex types. 
 */
const char *TY2C_Complex_Realpart_Name = "realpart";
const char *TY2C_Complex_Imagpart_Name = "imagpart";


/*--------------------- hidden utility routines -----------------------*/
/*---------------------------------------------------------------------*/

static void 
Write_Scalar_Typedef(FILE       *ofile, 
		     const char *real_type, 
		     const char *new_name)
{
   TOKEN_BUFFER tdef = New_Token_Buffer();

   Append_Token_String(tdef, "typedef");
   Append_Token_String(tdef, real_type);
   Append_Token_String(tdef, new_name);
   Append_Token_Special(tdef, ';');
   Append_Indented_Newline(tdef, 1);
   Write_And_Reclaim_Tokens(ofile, NULL, &tdef);
} /* Write_Scalar_Typedef */


static void 
Write_Typedef(FILE       *ofile, 
	      TY_IDX      ty, 
	      const char *type_name)
{
   TOKEN_BUFFER tdef = New_Token_Buffer();
   CONTEXT      context;
   
   CONTEXT_reset(context);
   
   Append_Token_String(tdef, type_name);
   TY2C_Handle[(TY_KIND)TY_kind(ty)](tdef, ty, context);
   Prepend_Token_String(tdef, "typedef");
   Append_Token_Special(tdef, ';');
   Append_Indented_Newline(tdef, 1);

   /* Put an empty line before this declaration and write it out */
   Prepend_Indented_Newline(tdef, 1);
   Write_And_Reclaim_Tokens(ofile, NULL, &tdef);
} /* Write_Typedef */


static void TY2C_prepend_qualifiers(TOKEN_BUFFER decl_tokens, TY_IDX ty,
        CONTEXT context)
{
    if (!CONTEXT_unqualified_ty2c(context))
    {
        if (TY_is_const(ty)) Prepend_Token_String(decl_tokens, "const");
        if (TY_is_volatile(ty)) Prepend_Token_String(decl_tokens, "volatile");
        if (TY_is_restrict(ty))
        {
/** DAVID CODE BEGIN **/
            // Added "__".
            Prepend_Token_String(decl_tokens, "__restrict__");
/*** DAVID CODE END ***/
        }
    }
} /* TY2C_prepend_qualifiers */


static INT64
get_field_gap(FLD_HANDLE this_fld, FLD_HANDLE next_fld, INT64 max_size,
	      BOOL is_union) 
{
   /* Return the minimum of the actual size of this_field and
    * the difference in offsets between this and the next field, 
    * unless there is no next_fld or we are dealing with a union 
    * of fields, in which case we assume the field-size must be 
    * less than or equal to the given max_size.
    */
   INT64 field_gap = TY_size(FLD_type(this_fld));
   
   if (!is_union        && 
       !next_fld.Is_Null() &&
       field_gap != FLD_ofst(next_fld)-FLD_ofst(this_fld))
   {
      field_gap = FLD_ofst(next_fld) - FLD_ofst(this_fld);
   }
   if (field_gap > max_size || (!is_union && next_fld.Is_Null ()))
   {
      field_gap = max_size;
   }
   return field_gap;
} /* get_field_gap */


static FLD_HANDLE
skip_till_next_field(FLD_HANDLE  this_fld, 
		     const INT64 struct_align,
		     const INT64 struct_size,
		     BOOL        is_union)
{
   /* Walk through the field-list until we find a field with
    * a larger offset than the first field, which is aligned
    * commensurate with its type but not with stricter (higher)
    * alignment requirements than the struct-type as a whole.  
    * I.e. until we find a field which can safely be accessed 
    * using C "->" or "." notation.
    */
   FLD_HANDLE next_fld = FLD_next(this_fld);
   
   if (!is_union)
   {
      /* Skip until we get to a next_fld which is neither packed
       * tighter than the struct itself nor is misaligned (we only 
       * allow this for pointers to handle INITVKIND_SYMOFF 
       * initializers).
       */
      while (!next_fld.Is_Null () &&
	     (FLD_ofst(this_fld) >= FLD_ofst(next_fld) ||
	      !PTR_OR_ALIGNED_WITH_STRUCT(FLD_type(next_fld), struct_align) ||
	      (!TY_Is_Pointer(FLD_type(next_fld)) &&
	       FLD_ofst(next_fld) % TY_align(FLD_type(next_fld)) != 0) ||
	      (!is_union &&
	       FLD_Is_Bitfield(next_fld, 
			       FLD_next(next_fld),
			       struct_size - FLD_ofst(next_fld)))))
      {
	 next_fld = FLD_next(next_fld);
      }
   }
   return next_fld;
} /* skip_till_next_field */


static void
TY2C_prepend_filler_field(TOKEN_BUFFER decl_tokens, INT64 byte_size)
{
   /* Declare a filler field, uniquely named "fill".
    */
   TOKEN_BUFFER field_tokens;
   
   field_tokens = New_Token_Buffer();
   Append_Token_String(field_tokens,
		       Scalar_C_Names[MTYPE_U1].pseudo_name);
   Append_Token_String(field_tokens, W2CF_Symtab_Unique_Name("fill"));
   Append_Token_Special(field_tokens, '[');
   TCON2C_translate(field_tokens, Host_To_Targ(MTYPE_I8, byte_size));
   Append_Token_Special(field_tokens, ']');
   Append_Token_Special(field_tokens, ';');
   Prepend_Indented_Newline(field_tokens, 1);
   Prepend_And_Reclaim_Token_List(decl_tokens, &field_tokens);
} /* TY2C_prepend_filler_field */


static void
TY2C_prepend_FLD_list(TOKEN_BUFFER decl_tokens,
		      FLD_HANDLE   fld,
		      const BOOL   is_union,
		      const INT64  struct_align,
		      const INT64  struct_size,
		      CONTEXT      context)
{
   /* Add the fields in the field-list (fld) in the correct order,
    * where we handle bitfields and packed structures by declaring
    * container or "filler" fields.  Such fillers must never be 
    * referenced, so we must recognize them uniformly both at places
    * of reference and here at the point of declaration.  Insert an
    * indented newline before each field.  Note that for bitfields we 
    * cannot assume anything about the structure alignment, since 
    * this is dependent on the container types, so we pass the 
    * structure alignment in here explicitly.  Assert(fld != NULL)!!
    */
   TOKEN_BUFFER fld_tokens;
   INT64        fld_gap;
   const INT64  remaining_bytes = struct_size - FLD_ofst(fld);
   
   FLD_HANDLE next_fld = skip_till_next_field(fld,
					      struct_align,
					      struct_size,
					      is_union);
   fld_gap = get_field_gap(fld, next_fld, remaining_bytes, is_union);
      
   /* Do the next field before this one, since fields are prepended 
    * (rather than appended) to the token buffer.  Note that filler-
    * fields to precede the next_fld should be prepended after having
    * prepended the next_fld.
    */
   if (!next_fld.Is_Null ())
   {
      if (is_union)
	 TY2C_prepend_FLD_list(decl_tokens, 
			       next_fld,
			       TRUE,  /* is_union */
			       struct_align,
			       struct_size,
			       context);
      else
	 TY2C_prepend_FLD_list(decl_tokens,
			       next_fld,
			       FALSE, /* is_union */
			       struct_align,
			       struct_size, 
			       context);
   }
 
   /* Add the tokens for the field declaration in a right-to-left
    * manner.  Note that we cannot redeclare bitfields accurately,
    * since the TY information is too incomplete to allow this.
    */
   if (!PTR_OR_ALIGNED_WITH_STRUCT(FLD_type(fld), struct_align) ||
       (!is_union &&
	FLD_Is_Bitfield(fld, FLD_next(fld), remaining_bytes)))
   {
      /* Fill up this space with a filler-field */
      if (fld_gap > 0)
	 TY2C_prepend_filler_field(decl_tokens, fld_gap);
   }
   else /* A regular field, at least we think so */
   {
      /* Insert filler to succede this fld, since such fillers are
       * not handled in the processing of the next field.
       */
      if (!is_union && fld_gap > TY_size(FLD_type(fld)))
      {
	 TY2C_prepend_filler_field(decl_tokens, 
				   fld_gap - TY_size(FLD_type(fld)));
      }
      
      /* Emit this_fld declaration */
      fld_tokens = New_Token_Buffer();
      Append_Token_String(fld_tokens, W2CF_Symtab_Nameof_Fld(fld));
      TY2C_translate(fld_tokens, FLD_type(fld), context);
      Append_Token_Special(fld_tokens, ';');
      Prepend_And_Reclaim_Token_List(decl_tokens, &fld_tokens);
      Prepend_Indented_Newline(decl_tokens, 1);
   } /*if*/
} /* TY2C_prepend_FLD_list */


static void
TY2C_Prepend_Alignment_Type(TOKEN_BUFFER tokens, INT64 align)
{
   CONTEXT c;

   CONTEXT_reset(c);
   switch (align)
   {
   case 1:
      TY2C_translate(tokens, Stab_Mtype_To_Ty(MTYPE_U1), c);
      break;
   case 2:
      TY2C_translate(tokens, Stab_Mtype_To_Ty(MTYPE_U2), c);
      break;
   case 4:
      TY2C_translate(tokens, Stab_Mtype_To_Ty(MTYPE_U4), c);
      break;
   case 8:
      TY2C_translate(tokens, Stab_Mtype_To_Ty(MTYPE_U8), c);
      break;
   case 16:
      TY2C_translate(tokens, Stab_Mtype_To_Ty(MTYPE_FQ), c);
      break;
   default:
      Is_True(FALSE, 
	      ("Unexpected alignment (%lld) in TY2C_Prepend_Alignment_Type()",
	       align));
   }
} /* TY2C_Prepend_Alignment_Type */


static void
TY2C_prototype_params(TOKEN_BUFFER decl_tokens, 
		      TYLIST_IDX   params, 
		      CONTEXT      context)
{
   TOKEN_BUFFER param_tokens;

   if (Tylist_Table[params] == 0)
   {
      Append_Token_String(decl_tokens, "void");
   }
   else
   {
      while (Tylist_Table[params] != 0)
      {
	 param_tokens = New_Token_Buffer();
	 TY2C_translate(param_tokens, TYLIST_item(Tylist_Table[params]),
                        context);
	 /*if (W2C_Emit_OpenCL && openCL_device_function){
	   if (TY_Is_Pointer(TYLIST_item(Tylist_Table[params]))){
	     Prepend_Token_String(param_tokens, "__private");
	     printf("%d\n", openCL_device_function_info);
	   }
	   }*/
	 Append_And_Reclaim_Token_List(decl_tokens, &param_tokens);
	 params = TYLIST_next(params);
         if (Tylist_Table[params] != 0)
	    Append_Token_Special(decl_tokens, ',');
      }
   }
} /* TY2C_prototype_params */


/*---------- hidden routines to handle each kind of type --------------*/
/*---------------------------------------------------------------------*/

/** DAVID CODE BEGIN **/

static void 
TY2C_scalar(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context) {
    Is_True(TY_Is_String(ty) || TY_mtype(ty) <= MTYPE_LAST,
            ("Illegal type in TY2C_scalar()"));
    Is_True(TY_mtype(ty) != MTYPE_UNKNOWN,
            ("Unknown type in TY2C_scalar()"));

    if (TY_Is_String(ty)) {
        Prepend_Token_String(decl_tokens,
                Scalar_C_Names[MTYPE_U1].pseudo_name);
        Append_Token_Special(decl_tokens, '[');
        Append_Token_Special(decl_tokens, ']');
    } else if (TY_is_enum(ty)) {
        // This is an ENUM type.
        Prepend_Token_String(decl_tokens, "enum");
        Prepend_Token_String(decl_tokens, TY_name(ty));
    } else if (TY_is_long_int(ty)) {
        // This is a 'long' or 'unsigned long'.
        // We don't want to add this special case to 'Scalar_C_Names'.
        Prepend_Token_String(decl_tokens, "long");
        if (MTYPE_is_unsigned(TY_mtype(ty))) {
            Prepend_Token_String(decl_tokens, "unsigned");
        }
    } else if (TY_is_char(ty)) {
        // This is a 'char' with no sign spec.
        // We don't want to add this special case to 'Scalar_C_Names'.
        Prepend_Token_String(decl_tokens, "char");
    } else {
      // Don't touch types of the form cl_* (openCL types)
      if ((W2C_Emit_OpenCL) && 
	  (strlen(TY_name(ty)) > 3) &&
	  (TY_name(ty)[0] == 'c') && 
	  (TY_name(ty)[1] == 'l') && 
	  (TY_name(ty)[2] == '_')){
	Prepend_Token_String(decl_tokens, TY_name(ty));	
      } else {
        Prepend_Token_String(decl_tokens, 
                Scalar_C_Names[TY_mtype(ty)].pseudo_name);
      }
    }

    TY2C_prepend_qualifiers(decl_tokens, ty, context);
} /* TY2C_scalar */

static void 
TY2C_array(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context) {
    if (Stab_Array_Has_Dynamic_Bounds(ty)) {
        /* The array has dynamic bounds, unknown at compile-time,
         * so declare it as a pointer type in a context that cannot
         * accept incomplete types, and declare it as an incomplete
         * array otherwise. 
         *
         * Is_True(TY_size(ty) == 0,
         *      ("Expected size of dynamic array to be zero (size=%llu)",
         *       (UINT64)TY_size(ty)));
         */
        if (CONTEXT_complete_ty2c(context)) {
            /* Declare it as a pointer type, since it must be a dynamic or 
	         * allocatable Fortran array. */
            Prepend_Token_Special(decl_tokens, '*');
            if (TY_Is_Array_Or_Function(TY_AR_etype(ty))) {
                WHIRL2C_parenthesize(decl_tokens);
            }

            // Allow incomplete element type.
            CONTEXT_reset_complete_ty2c(context);
        } else {
            /* We can have incomplete C types here, so just declare it as 
             * an incomplete array of one dimension.  Dynamic arrays are
             * indexed as though they have one dimension, so this should
             * work just fine. */
            Append_Token_Special(decl_tokens, '[');
            Append_Token_Special(decl_tokens, ']');

            // In C, the element type of an array must be complete.
            // CONTEXT_set_complete_ty2c(context);
        }
    } else {
        INT64 num_elts;
        INT32 ndims = TY_AR_ndims(ty);
 
        /* Emit the array dimensions.
         * TODO: If the stride is different from the element type size,
         * we need to adjust the num_elts. */
        for (INT32 dim = 0; dim < ndims; ++dim) {
            Append_Token_Special(decl_tokens, '[');

            /* If we need to take the stride (in bytes) into account, then
             * we must divide the following by:
             *      (TY_AR_stride_val(ty, dim)/TY_size(TY_AR_etype(ty))) */
            num_elts = TY_AR_ubnd_val(ty, dim) - TY_AR_lbnd_val(ty, dim) + 1;
            if (num_elts > 0) {
                TCON2C_translate(decl_tokens, 
                    Host_To_Targ(MTYPE_I8, num_elts));
            }

            Append_Token_Special(decl_tokens, ']');
        }

        // In C, the element type of an array must be complete.
        // CONTEXT_set_complete_ty2c(context);
    }

    // Add the element type tokens.
    TY2C_translate(decl_tokens, TY_AR_etype(ty), context);

    // Prepend the array and element qualifiers.
    TY2C_prepend_qualifiers(decl_tokens, ty, context);
} /* TY2C_array */


static void 
TY2C_struct(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context)
{
    /* We need to get the alignment right for structs containing
     * equivalence fields; i.e. overlapping fields, only one of which
     * will be declared by whirl2f.  One way to accomplish this is
     * to wrap the whole struct within a union and give the union the
     * name of the struct, the struct field an anonymous type and
     * a name "__block", and enforce the alignment with the a
     * second union field with the name "__align".  As an example,
     * consider the struct S (having an equivalence field):
     *
     *   struct S1 {....}
     *
     * We translate this into:
     *
     *   union S1 {
     *     struct {
     *       ....
     *     } __block;
     *     UINT64 __align;
     *   }
     *
     * Access expressions to this struct must accordingly denote
     * the "__block" portion of the union.
     */

    BOOL declare_complete = (CONTEXT_complete_ty2c(context) &&
        !TY_is_translated_to_c(ty));

    // Flag it as having been translated into C.
    if (declare_complete) Set_TY_is_translated_to_c(ty);

    BOOL is_equivalenced = Stab_Is_Equivalenced_Struct(ty);

    // DAVID COMMENT:
    //
    // We may attempt to retrive the type name here as it is used in multiple
    // places in the following code. However, W2CF_Symtab_Nameof_Ty may
    // deallocate these strings returned in the middle. Therefore, we must
    // retrieve on-demand.
    //
    // const char *ty_name = W2CF_Symtab_Nameof_Ty(ty);

    /* Declare all the struct fields, unless the incomplete type is all
     * that is needed. */

    if (declare_complete && !TY_flist(Ty_Table[ty]).Is_Null())
    {
        CONTEXT fld_context = context;

        // The field is unqualified and incomplete.
        CONTEXT_reset_unqualified_ty2c(fld_context);
        CONTEXT_reset_complete_ty2c(fld_context);

        if (is_equivalenced) {
            /* Prepend the end-of-union character, on a new line following
             * the list of its two fields, and force the alignment with
             * a new field. */
            Prepend_Token_Special(decl_tokens, '}');
            Prepend_Indented_Newline(decl_tokens, 1);
            Increment_Indentation();
            Prepend_Token_Special(decl_tokens, ';');
            Prepend_Token_String(decl_tokens, "__align");
            TY2C_Prepend_Alignment_Type(decl_tokens, TY_align(ty));
            Prepend_Indented_Newline(decl_tokens, 1);
            Prepend_Token_Special(decl_tokens, ';');
            Prepend_Token_String(decl_tokens, TY2C_Aligned_Block_Name);
        }
      
        // Prepend the end-of-struct character, on a new line following
        // the list of fields.
        Prepend_Token_Special(decl_tokens, '}');
        Prepend_Indented_Newline(decl_tokens, 1);

        // Prepend fields, where each field is preceeded by a newline 
        // and is indented.
        Increment_Indentation();
        TY2C_prepend_FLD_list(decl_tokens,
            TY_flist(Ty_Table[ty]),
            TY_is_union(ty),
            TY_align(ty),
            TY_size(ty),
            fld_context
        );
        Decrement_Indentation();

        // Prepend the start-of-struct character.
        Prepend_Token_Special(decl_tokens, '{');

        if (is_equivalenced) {
            // Prepend the start-of-union character, followed by a newline
            // to precede the struct and alignment fields.
            Prepend_Token_String(decl_tokens,
                TY_is_union(ty)? "union": "struct");
            Prepend_Indented_Newline(decl_tokens, 1);
            Decrement_Indentation();
            Prepend_Token_Special(decl_tokens, '{');
        }

        Prepend_Token_String(decl_tokens, W2CF_Symtab_Nameof_Ty(ty));
    }
    else if (declare_complete && TY_size(ty) == 1)
    {
        /* A special struct with no fields, which may sometimes occur
         * for C++ lowered into C. */

        Prepend_Token_Special(decl_tokens, '}');
        Prepend_Indented_Newline(decl_tokens, 1);

        Prepend_Token_Special(decl_tokens, ';');
        Prepend_Token_String(decl_tokens, W2CF_Symtab_Unique_Name("dummy"));
        TY2C_translate(decl_tokens, Stab_Mtype_To_Ty(MTYPE_U1), context);

        Increment_Indentation();
        Prepend_Indented_Newline(decl_tokens, 1);
        Prepend_Token_Special(decl_tokens, '{');
        Prepend_Token_String(decl_tokens, W2CF_Symtab_Nameof_Ty(ty));
        Decrement_Indentation();
    }
    else
    {
        Prepend_Token_String(decl_tokens, W2CF_Symtab_Nameof_Ty(ty));
    }

    Prepend_Token_String(decl_tokens,
        (TY_is_union(ty) || is_equivalenced) ? "union" : "struct");

    TY2C_prepend_qualifiers(decl_tokens, ty, context);
} /* TY2C_struct */

/*** DAVID CODE END ***/

static void 
TY2C_function(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context)
{
   TYLIST_IDX params = TY_parms(ty);

   /* A function cannot be qualified! */
   CONTEXT_reset_unqualified_ty2c(context);
   
   /* Append the parameter type list to the right of the decl_tokens */
   Append_Token_Special(decl_tokens, '(');
   if (TY_has_prototype(ty))
   {
      TY2C_prototype_params(decl_tokens, params, context);
      if (TY_is_varargs(ty))
      {
	 Append_Token_Special(decl_tokens, ',');
	 Append_Token_String(decl_tokens, "...");
      }
   }
   Append_Token_Special(decl_tokens, ')');

   /* Prepend the return type as the type of the decl_tokens */
   TY2C_translate(decl_tokens, Func_Return_Type(ty), context);
} /* TY2C_function */


/** DAVID CODE BEGIN **/

static void 
TY2C_pointer(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context) {
    // The qualifier applies to the pointer, so add it to the rhs of '*'.
    TY2C_prepend_qualifiers(decl_tokens, ty, context);
    Prepend_Token_Special(decl_tokens, '*');

    TY_IDX pty = TY_pointed(ty);

    // Not sure if the parentheses are applied to the right place.
    if (TY_Is_Array_Or_Function(pty)) WHIRL2C_parenthesize(decl_tokens);

    // The pointee is qualified and can be incomplete.
    CONTEXT_reset_unqualified_ty2c(context);
    CONTEXT_reset_complete_ty2c(context);

    TY2C_translate(decl_tokens, pty, context);
} /* TY2C_pointer */

/*** DAVID CODE END ***/


static void 
TY2C_void(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context)
{
   Prepend_Token_String(decl_tokens, Special_Void_TypeName);
   TY2C_prepend_qualifiers(decl_tokens, ty, context);
} /* TY2C_void */


static void 
TY2C_invalid(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context)
{
   Is_True(FALSE, ("Invalid TY kind (%d) for translation to C", TY_kind(ty)));
} /* TY2C_invalid */


/*------------------------ exported routines --------------------------*/
/*---------------------------------------------------------------------*/

void 
TY2C_initialize(CONTEXT context)
{
   /* Since /usr/include/whirl2c.h has declared typedefs for the 
    * complex and string types, we need to ensure that they are 
    * not declared again:
    */
   Set_TY_is_translated_to_c(Be_Type_Tbl(MTYPE_STRING));
   Set_TY_is_translated_to_c(Be_Type_Tbl(MTYPE_C4));
   Set_TY_is_translated_to_c(Be_Type_Tbl(MTYPE_C8));
   Set_TY_is_translated_to_c(Be_Type_Tbl(MTYPE_C10));
   Set_TY_is_translated_to_c(Be_Type_Tbl(MTYPE_CQ));
} /* TY2C_initialize */


void 
TY2C_finalize(void)
{
   return; /* nothing to do */
} /* TY2C_finalize */


void 
TY2C_translate(TOKEN_BUFFER decl_tokens, TY_IDX ty, CONTEXT context)
{
   /* Note that "decl_tokens" can either be the variable name or a
    * string representing an enclosing type.
    */
   const char *special_name = GET_SPECIAL_TYPENAME(TY_mtype(ty));

   if (special_name != NULL)
   {
      Prepend_Token_String(decl_tokens, special_name);
      TY2C_prepend_qualifiers(decl_tokens, ty, context);
   }
   else
   {
      Is_True((TY_KIND)TY_kind(ty) < KIND_LAST && TY_kind(ty) > KIND_INVALID, 
	      ("Unexpected TY_kind (%d) in TY2C_translate()", TY_kind(ty)));
      TY2C_Handle[(TY_KIND)TY_kind(ty)](decl_tokens, ty, context);
   }

/** DAVID CODE BEGIN **/
    // Mark that the type is used in the code.
    // Its definition (if a struct) will appear in the header file generated.
    Set_TY_is_used(ty);
/*** DAVID CODE END ***/
} /* TY2C_translate */


void 
TY2C_translate_unqualified(TOKEN_BUFFER decl_tokens, TY_IDX ty)
{
   /* Any qualification on the top-level type should be ignored.
    */
   CONTEXT context;

   CONTEXT_reset(context);
   CONTEXT_set_unqualified_ty2c(context);
   TY2C_translate(decl_tokens, ty, context);
} /* TY2C_translate_unqualified */


TY2C_FLD_INFO
TY2C_get_field_info(TY_IDX struct_ty,      /* base type */
		    TY_IDX desired_ty,     /* preferred field type */
		    MTYPE  desired_mty,    /* preferred field mtype */
		    INT64  desired_offset) /* required field mtype */
{
   /* Try to find a field at any level of nesting, which is at
    * the given offset from the base of the struct_ty and
    * (preferably) of type field_ty or (less preferably) of mtype 
    * field_mty.
    *
    * possibly (?) TODO:  enhance this to be more similar to 
    * Construct_Fld_Path in ty2f.h.
    */
   TY_IDX        this_ty;
   FLD_HANDLE       this_fld;
   FLD_HANDLE       next_fld;
   TY2C_FLD_INFO this_fld_info;
   const INT64   struct_size = TY_size(struct_ty);
   const INT64   struct_align = TY_align(struct_ty);
   BOOL          found_ty = FALSE, found_mty = FALSE;

   /* Search for a field at, or just before, the given offset */
   this_fld = TY_flist(Ty_Table[struct_ty]);
   if (!this_fld.Is_Null () && !TY_is_union(struct_ty))
   {
      /* Set "this_field" to the field closest to the given offset, such
       * that FLD_ofst(this_field) <= desired_offset and next_fld==NULL
       * or FLD_ofst(next_field) > desired_offset.
       */
      for (next_fld = skip_till_next_field(this_fld, 
					   struct_align,
					   struct_size,
					   FALSE/*is_union*/);
	   !next_fld.Is_Null() && FLD_ofst(next_fld) <= desired_offset;
	   next_fld = skip_till_next_field(next_fld, 
					   struct_align,
					   struct_size,
					   FALSE/*is_union*/))
      {
	 this_fld = next_fld;
      }
   }
   else if (!this_fld.Is_Null ())
   {
      next_fld = skip_till_next_field(this_fld, 
				      struct_align,
				      struct_size,
				      TRUE/*is_union*/);
   }
   
   /* Search for a field of the correct type at the *exact* offset.
    * If a struct/union/class field is encountered, of a different
    * type than the one we are looking for, then try recursively if
    * possible.
    */
   while (!found_ty && !this_fld.Is_Null () &&
	  FLD_ofst(this_fld) <= desired_offset)
   {
      this_ty = FLD_type(this_fld);

      if (!PTR_OR_ALIGNED_WITH_STRUCT(this_ty, struct_align) ||
	  (!TY_is_union(struct_ty) &&
	   FLD_Is_Bitfield(this_fld, FLD_next(this_fld), 
			   struct_size - FLD_ofst(this_fld))))
      {
	 /* A bitfield: Note that we cannot redeclare bitfields
	  * accurately, and we therefore cannot access them directly
	  * either.  Continue down the list of fields to see if
	  * there is a non-bitfield unioned with this one.
	  */
      }
      else if (desired_offset == FLD_ofst(this_fld) && 
	       Stab_Identical_Types(desired_ty, this_ty, 
				    FALSE, /* check_quals */
				    TRUE,  /* check_scalars */
				    TRUE)) /* ptrs_as_scalars */
      {
	 /* A perfect match, so initiate the field_info */
	 this_fld_info.found_fld = this_fld;
	 this_fld_info.select_tokens = New_Token_Buffer();
	 Append_Token_String(this_fld_info.select_tokens,
			     W2CF_Symtab_Nameof_Fld(this_fld));
	 found_ty = TRUE;
      }
      else if (TY_Is_Structured(this_ty))
      {
	 /* Try to find a nested field that matches.  Subtract the
	  * offset of this field from the offset searched for in
	  * the nested struct.
	  */
	 this_fld_info = 
	    TY2C_get_field_info(this_ty, 
				desired_ty, 
				desired_mty, 
				desired_offset-FLD_ofst(this_fld));
	 if (!this_fld_info.found_fld.Is_Null ())
	 {
	    /* prepend this name to select the path returned */
	    Prepend_Token_Special(this_fld_info.select_tokens, '.');
	    Prepend_Token_String(this_fld_info.select_tokens,
				 W2CF_Symtab_Nameof_Fld(this_fld));
	    found_ty = TRUE;
	 }
      }
      else if (desired_offset == FLD_ofst(this_fld) && 
	       desired_mty == TY_mtype(this_ty))
      {
	 /* An imperfect match, so initiate the found_fld, but keep 
	  * searching ...
	  */
	 this_fld_info.found_fld = this_fld;
	 found_mty = TRUE;
      }
      if (!found_ty)
      {
	 this_fld = next_fld;
	 if (!this_fld.Is_Null ())
	    next_fld = skip_till_next_field(this_fld,
					    struct_align,
					    struct_size,
					    TY_is_union(struct_ty));
      }
   } /* while */
   
   if (!found_ty && found_mty)
   {
      /* An imperfect match was the best we found, so initiate the
       * field_info.
       */
      this_fld_info.select_tokens = New_Token_Buffer();
      Append_Token_String(this_fld_info.select_tokens,
			  W2CF_Symtab_Nameof_Fld(this_fld_info.found_fld));
   }
   if (found_ty || found_mty)
   {
      /* If we have an equivalence field, we must dereference through
       * the artificial union we have created to declare the struct
       * of appropriate alignment.  See also "TY2C_struct()".
       */
      if (Stab_Is_Equivalenced_Struct(struct_ty))
      {
	 Prepend_Token_Special(this_fld_info.select_tokens, '.');
	 Prepend_Token_String(this_fld_info.select_tokens, 
			      TY2C_Aligned_Block_Name);
      }
   }
   else if (!found_ty)
   {
      this_fld_info.found_fld = FLD_HANDLE(0);
   }
   return this_fld_info;
} /* TY2C_get_field_info */


/** DAVID CODE BEGIN **/

BOOL
TY2C_builtin(TY_IDX ty) {
    TYPE_ID ty_mtype = TY_mtype(ty);

    if (GET_SPECIAL_TYPENAME(ty_mtype) != NULL) return true;

    // Not sure what this check covers exactly.
    if (ty_mtype <= MTYPE_LAST) {
        if (ty_mtype != MTYPE_UNKNOWN || TY_kind(ty) == KIND_INVALID) {
            const char *name = Scalar_C_Names[ty_mtype].pseudo_name;
            return (name != NULL && strcmp(name, "") != 0);
        }
    }

    return false;
} /* TY2C_builtin */

/*** DAVID CODE END ***/
