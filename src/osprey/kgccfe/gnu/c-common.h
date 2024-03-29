/* 
   Copyright 2003, 2004, 2005, 2006 PathScale, Inc.  All Rights Reserved.
   File modified October 3, 2003 by PathScale, Inc. to update Open64 C/C++ 
   front-ends to GNU 3.3.1 release.
 */

/* Definitions for c-common.c.
   Copyright (C) 1987, 1993, 1994, 1995, 1997, 1998,
   1999, 2000, 2001, 2002 Free Software Foundation, Inc.

This file is part of GCC.

GCC is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2, or (at your option) any later
version.

GCC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with GCC; see the file COPYING.  If not, write to the Free
Software Foundation, 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.  */

#ifndef GCC_C_COMMON_H
#define GCC_C_COMMON_H

#include "splay-tree.h"
#include "cpplib.h"
#ifdef KEY
#include "omp_types.h"
#endif

/** DAVID CODE BEGIN **/
#include "hicuda_types.h"
/*** DAVID CODE END ***/

/* Usage of TREE_LANG_FLAG_?:
   0: COMPOUND_STMT_NO_SCOPE (in COMPOUND_STMT).
      TREE_NEGATED_INT (in INTEGER_CST).
      IDENTIFIER_MARKED (used by search routines).
      SCOPE_BEGIN_P (in SCOPE_STMT)
      DECL_PRETTY_FUNCTION_P (in VAR_DECL)
      NEW_FOR_SCOPE_P (in FOR_STMT)
      ASM_INPUT_P (in ASM_STMT)
      STMT_EXPR_NO_SCOPE (in STMT_EXPR)
   1: C_DECLARED_LABEL_FLAG (in LABEL_DECL)
      STMT_IS_FULL_EXPR_P (in _STMT)
   2: STMT_LINENO_FOR_FN_P (in _STMT)
   3: SCOPE_NO_CLEANUPS_P (in SCOPE_STMT)
      COMPOUND_STMT_BODY_BLOCK (in COMPOUND_STMT)
   4: SCOPE_PARTIAL_P (in SCOPE_STMT)
*/

/* Reserved identifiers.  This is the union of all the keywords for C,
   C++, and Objective-C.  All the type modifiers have to be in one
   block at the beginning, because they are used as mask bits.  There
   are 27 type modifiers; if we add many more we will have to redesign
   the mask mechanism.  */

enum rid
{
  /* Modifiers: */
  /* C, in empirical order of frequency.  */
  RID_STATIC = 0,
  RID_UNSIGNED, RID_LONG,    RID_CONST, RID_EXTERN,
  RID_REGISTER, RID_TYPEDEF, RID_SHORT, RID_INLINE,
  RID_VOLATILE, RID_SIGNED,  RID_AUTO,  RID_RESTRICT,

  /* C extensions */
  RID_BOUNDED, RID_UNBOUNDED, RID_COMPLEX, RID_THREAD,

  /* C++ */
  RID_FRIEND, RID_VIRTUAL, RID_EXPLICIT, RID_EXPORT, RID_MUTABLE,

  /* ObjC */
  RID_IN, RID_OUT, RID_INOUT, RID_BYCOPY, RID_BYREF, RID_ONEWAY,

  /* C */
  RID_INT,     RID_CHAR,   RID_FLOAT,    RID_DOUBLE, RID_VOID,
  RID_ENUM,    RID_STRUCT, RID_UNION,    RID_IF,     RID_ELSE,
  RID_WHILE,   RID_DO,     RID_FOR,      RID_SWITCH, RID_CASE,
  RID_DEFAULT, RID_BREAK,  RID_CONTINUE, RID_RETURN, RID_GOTO,
  RID_SIZEOF,

  /* C extensions */
  RID_ASM,       RID_TYPEOF,   RID_ALIGNOF,  RID_ATTRIBUTE,  RID_VA_ARG,
  RID_EXTENSION, RID_IMAGPART, RID_REALPART, RID_LABEL,      RID_PTRBASE,
  RID_PTREXTENT, RID_PTRVALUE, RID_CHOOSE_EXPR, RID_TYPES_COMPATIBLE_P,

  /* Too many ways of getting the name of a function as a string */
  RID_FUNCTION_NAME, RID_PRETTY_FUNCTION_NAME, RID_C99_FUNCTION_NAME,

  /* C++ */
  RID_BOOL,     RID_WCHAR,    RID_CLASS,
  RID_PUBLIC,   RID_PRIVATE,  RID_PROTECTED,
  RID_TEMPLATE, RID_NULL,     RID_CATCH,
  RID_DELETE,   RID_FALSE,    RID_NAMESPACE,
  RID_NEW,      RID_OPERATOR, RID_THIS,
  RID_THROW,    RID_TRUE,     RID_TRY,
  RID_TYPENAME, RID_TYPEID,   RID_USING,

  /* casts */
  RID_CONSTCAST, RID_DYNCAST, RID_REINTCAST, RID_STATCAST,

  /* Objective-C */
  RID_ID,          RID_AT_ENCODE,    RID_AT_END,
  RID_AT_CLASS,    RID_AT_ALIAS,     RID_AT_DEFS,
  RID_AT_PRIVATE,  RID_AT_PROTECTED, RID_AT_PUBLIC,
  RID_AT_PROTOCOL, RID_AT_SELECTOR,  RID_AT_INTERFACE,
  RID_AT_IMPLEMENTATION,

  RID_MAX,

  RID_FIRST_MODIFIER = RID_STATIC,
  RID_LAST_MODIFIER = RID_ONEWAY,

  RID_FIRST_AT = RID_AT_ENCODE,
  RID_LAST_AT = RID_AT_IMPLEMENTATION,
  RID_FIRST_PQ = RID_IN,
  RID_LAST_PQ = RID_ONEWAY
};

#define OBJC_IS_AT_KEYWORD(rid) \
  ((unsigned int)(rid) >= (unsigned int)RID_FIRST_AT && \
   (unsigned int)(rid) <= (unsigned int)RID_LAST_AT)

#define OBJC_IS_PQ_KEYWORD(rid) \
  ((unsigned int)(rid) >= (unsigned int)RID_FIRST_PQ && \
   (unsigned int)(rid) <= (unsigned int)RID_LAST_PQ)

/* The elements of `ridpointers' are identifier nodes for the reserved
   type names and storage classes.  It is indexed by a RID_... value.  */
extern tree *ridpointers;

/* Standard named or nameless data types of the C compiler.  */

enum c_tree_index
{
    CTI_WCHAR_TYPE,
    CTI_SIGNED_WCHAR_TYPE,
    CTI_UNSIGNED_WCHAR_TYPE,
    CTI_WINT_TYPE,
    CTI_SIGNED_SIZE_TYPE, /* For format checking only.  */
    CTI_UNSIGNED_PTRDIFF_TYPE, /* For format checking only.  */
    CTI_INTMAX_TYPE,
    CTI_UINTMAX_TYPE,
    CTI_WIDEST_INT_LIT_TYPE,
    CTI_WIDEST_UINT_LIT_TYPE,

    CTI_CHAR_ARRAY_TYPE,
    CTI_WCHAR_ARRAY_TYPE,
    CTI_INT_ARRAY_TYPE,
    CTI_STRING_TYPE,
    CTI_CONST_STRING_TYPE,

    /* Type for boolean expressions (bool in C++, int in C).  */
    CTI_BOOLEAN_TYPE,
    CTI_BOOLEAN_TRUE,
    CTI_BOOLEAN_FALSE,
    /* C99's _Bool type.  */
    CTI_C_BOOL_TYPE,
    CTI_C_BOOL_TRUE,
    CTI_C_BOOL_FALSE,
    CTI_DEFAULT_FUNCTION_TYPE,

    CTI_G77_INTEGER_TYPE,
    CTI_G77_UINTEGER_TYPE,
    CTI_G77_LONGINT_TYPE,
    CTI_G77_ULONGINT_TYPE,

    /* These are not types, but we have to look them up all the time.  */
    CTI_FUNCTION_NAME_DECL,
    CTI_PRETTY_FUNCTION_NAME_DECL,
    CTI_C99_FUNCTION_NAME_DECL,
    CTI_SAVED_FUNCTION_NAME_DECLS,
    
    CTI_VOID_ZERO,

    CTI_MAX
};

#define C_RID_CODE(id)	(((struct c_common_identifier *) (id))->node.rid_code)

/* Identifier part common to the C front ends.  Inherits from
   tree_identifier, despite appearances.  */
struct c_common_identifier GTY(())
{
  struct tree_common common;
  struct cpp_hashnode GTY ((skip (""))) node;
};

#define wchar_type_node			c_global_trees[CTI_WCHAR_TYPE]
#define signed_wchar_type_node		c_global_trees[CTI_SIGNED_WCHAR_TYPE]
#define unsigned_wchar_type_node	c_global_trees[CTI_UNSIGNED_WCHAR_TYPE]
#define wint_type_node			c_global_trees[CTI_WINT_TYPE]
#define signed_size_type_node		c_global_trees[CTI_SIGNED_SIZE_TYPE]
#define unsigned_ptrdiff_type_node	c_global_trees[CTI_UNSIGNED_PTRDIFF_TYPE]
#define intmax_type_node		c_global_trees[CTI_INTMAX_TYPE]
#define uintmax_type_node		c_global_trees[CTI_UINTMAX_TYPE]
#define widest_integer_literal_type_node c_global_trees[CTI_WIDEST_INT_LIT_TYPE]
#define widest_unsigned_literal_type_node c_global_trees[CTI_WIDEST_UINT_LIT_TYPE]

#define boolean_type_node		c_global_trees[CTI_BOOLEAN_TYPE]
#define boolean_true_node		c_global_trees[CTI_BOOLEAN_TRUE]
#define boolean_false_node		c_global_trees[CTI_BOOLEAN_FALSE]

#define c_bool_type_node		c_global_trees[CTI_C_BOOL_TYPE]
#define c_bool_true_node		c_global_trees[CTI_C_BOOL_TRUE]
#define c_bool_false_node		c_global_trees[CTI_C_BOOL_FALSE]

#define char_array_type_node		c_global_trees[CTI_CHAR_ARRAY_TYPE]
#define wchar_array_type_node		c_global_trees[CTI_WCHAR_ARRAY_TYPE]
#define int_array_type_node		c_global_trees[CTI_INT_ARRAY_TYPE]
#define string_type_node		c_global_trees[CTI_STRING_TYPE]
#define const_string_type_node		c_global_trees[CTI_CONST_STRING_TYPE]

#define default_function_type		c_global_trees[CTI_DEFAULT_FUNCTION_TYPE]

/* g77 integer types, which which must be kept in sync with f/com.h */
#define g77_integer_type_node		c_global_trees[CTI_G77_INTEGER_TYPE]
#define g77_uinteger_type_node		c_global_trees[CTI_G77_UINTEGER_TYPE]
#define g77_longint_type_node		c_global_trees[CTI_G77_LONGINT_TYPE]
#define g77_ulongint_type_node		c_global_trees[CTI_G77_ULONGINT_TYPE]

#define function_name_decl_node		c_global_trees[CTI_FUNCTION_NAME_DECL]
#define pretty_function_name_decl_node	c_global_trees[CTI_PRETTY_FUNCTION_NAME_DECL]
#define c99_function_name_decl_node		c_global_trees[CTI_C99_FUNCTION_NAME_DECL]
#define saved_function_name_decls	c_global_trees[CTI_SAVED_FUNCTION_NAME_DECLS]

/* A node for `((void) 0)'.  */
#define void_zero_node                  c_global_trees[CTI_VOID_ZERO]

extern GTY(()) tree c_global_trees[CTI_MAX];

/* Mark which labels are explicitly declared.
   These may be shadowed, and may be referenced from nested functions.  */
#define C_DECLARED_LABEL_FLAG(label) TREE_LANG_FLAG_1 (label)

/* Flag strings given by __FUNCTION__ and __PRETTY_FUNCTION__ for a
   warning if they undergo concatenation.  */
#define C_ARTIFICIAL_STRING_P(NODE) TREE_LANG_FLAG_0 (NODE)

typedef enum c_language_kind
{
  clk_c = 0,      /* A dialect of C: K&R C, ANSI/ISO C89, C2000, etc.  */
  clk_cplusplus   /* ANSI/ISO C++ */
}
c_language_kind;

/* Information about a statement tree.  */

struct stmt_tree_s GTY(()) {
  /* The last statement added to the tree.  */
  tree x_last_stmt;
  /* The type of the last expression statement.  (This information is
     needed to implement the statement-expression extension.)  */
  tree x_last_expr_type;
  /* The last filename we recorded.  */
  const char *x_last_expr_filename;
  /* In C++, Nonzero if we should treat statements as full
     expressions.  In particular, this variable is no-zero if at the
     end of a statement we should destroy any temporaries created
     during that statement.  Similarly, if, at the end of a block, we
     should destroy any local variables in this block.  Normally, this
     variable is nonzero, since those are the normal semantics of
     C++.

     However, in order to represent aggregate initialization code as
     tree structure, we use statement-expressions.  The statements
     within the statement expression should not result in cleanups
     being run until the entire enclosing statement is complete.

     This flag has no effect in C.  */
  int stmts_are_full_exprs_p;
};

typedef struct stmt_tree_s *stmt_tree;

/* Global state pertinent to the current function.  Some C dialects
   extend this structure with additional fields.  */

struct c_language_function GTY(()) {
  /* While we are parsing the function, this contains information
     about the statement-tree that we are building.  */
  struct stmt_tree_s x_stmt_tree;
  /* The stack of SCOPE_STMTs for the current function.  */
  tree x_scope_stmt_stack;
};

/* When building a statement-tree, this is the last statement added to
   the tree.  */

#define last_tree (current_stmt_tree ()->x_last_stmt)

/* The type of the last expression-statement we have seen.  */

#define last_expr_type (current_stmt_tree ()->x_last_expr_type)

/* The name of the last file we have seen.  */

#define last_expr_filename (current_stmt_tree ()->x_last_expr_filename)

/* LAST_TREE contains the last statement parsed.  These are chained
   together through the TREE_CHAIN field, but often need to be
   re-organized since the parse is performed bottom-up.  This macro
   makes LAST_TREE the indicated SUBSTMT of STMT.  */

#define RECHAIN_STMTS(stmt, substmt)		\
  do {						\
    substmt = TREE_CHAIN (stmt);		\
    TREE_CHAIN (stmt) = NULL_TREE;		\
    last_tree = stmt;				\
  } while (0)

/* Language-specific hooks.  */

extern int (*lang_statement_code_p)             PARAMS ((enum tree_code));
extern void (*lang_expand_stmt)                 PARAMS ((tree));
extern void (*lang_expand_decl_stmt)            PARAMS ((tree));
extern void (*lang_expand_function_end)         PARAMS ((void));
extern tree gettags				PARAMS ((void));

/* Callback that determines if it's ok for a function to have no
   noreturn attribute.  */
extern int (*lang_missing_noreturn_ok_p)	PARAMS ((tree));

extern int yyparse				PARAMS ((void));
extern void free_parser_stacks			PARAMS ((void));

extern stmt_tree current_stmt_tree              PARAMS ((void));
extern tree *current_scope_stmt_stack           PARAMS ((void));
extern void begin_stmt_tree                     PARAMS ((tree *));
extern tree add_stmt				PARAMS ((tree));
extern void add_decl_stmt                       PARAMS ((tree));
extern tree add_scope_stmt                      PARAMS ((int, int));
extern void finish_stmt_tree                    PARAMS ((tree *));

extern int statement_code_p                     PARAMS ((enum tree_code));
extern tree walk_stmt_tree			PARAMS ((tree *,
							 walk_tree_fn,
							 void *));
extern void prep_stmt                           PARAMS ((tree));
extern void expand_stmt                         PARAMS ((tree));
extern void shadow_warning			PARAMS ((const char *,
							 tree, tree));
extern tree c_begin_if_stmt			PARAMS ((void));
extern tree c_begin_while_stmt			PARAMS ((void));
extern void c_finish_while_stmt_cond		PARAMS ((tree, tree));


/* Extra information associated with a DECL.  Other C dialects extend
   this structure in various ways.  The C front-end only uses this
   structure for FUNCTION_DECLs; all other DECLs have a NULL
   DECL_LANG_SPECIFIC field.  */

struct c_lang_decl GTY(()) {
  unsigned declared_inline : 1;
};

/* In a FUNCTION_DECL for which DECL_BUILT_IN does not hold, this is
     the approximate number of statements in this function.  There is
     no need for this number to be exact; it is only used in various
     heuristics regarding optimization.  */
#define DECL_NUM_STMTS(NODE) \
  (FUNCTION_DECL_CHECK (NODE)->decl.u1.i)

/* The variant of the C language being processed.  Each C language
   front-end defines this variable.  */

extern c_language_kind c_language;

/* Switches common to the C front ends.  */

/* Nonzero if prepreprocessing only.  */
extern int flag_preprocess_only;

/* Nonzero if an ISO standard was selected.  It rejects macros in the
   user's namespace.  */
extern int flag_iso;

/* Nonzero whenever Objective-C functionality is being used.  */
extern int flag_objc;

/* Nonzero if -undef was given.  It suppresses target built-in macros
   and assertions.  */
extern int flag_undef;

/* Nonzero means don't recognize the non-ANSI builtin functions.  */

extern int flag_no_builtin;

/* Nonzero means don't recognize the non-ANSI builtin functions.
   -ansi sets this.  */

extern int flag_no_nonansi_builtin;

/* Nonzero means give `double' the same size as `float'.  */

extern int flag_short_double;

/* Nonzero means give `wchar_t' the same size as `short'.  */

extern int flag_short_wchar;

/* Nonzero means allow Microsoft extensions without warnings or errors.  */
extern int flag_ms_extensions;

/* Nonzero means don't recognize the keyword `asm'.  */

extern int flag_no_asm;

/* Nonzero means give string constants the type `const char *', as mandated
   by the standard.  */

extern int flag_const_strings;

/* Nonzero means `$' can be in an identifier.  */

extern int dollars_in_ident;

/* Nonzero means to treat bitfields as signed unless they say `unsigned'.  */

extern int flag_signed_bitfields;
extern int explicit_flag_signed_bitfields;

/* Nonzero means warn about pointer casts that can drop a type qualifier
   from the pointer target type.  */

extern int warn_cast_qual;

/* Warn about functions which might be candidates for format attributes.  */

extern int warn_missing_format_attribute;

/* Nonzero means warn about sizeof(function) or addition/subtraction
   of function pointers.  */

extern int warn_pointer_arith;

/* Nonzero means warn for any global function def
   without separate previous prototype decl.  */

extern int warn_missing_prototypes;

/* Warn if adding () is suggested.  */

extern int warn_parentheses;

/* Warn if initializer is not completely bracketed.  */

extern int warn_missing_braces;

/* Warn about comparison of signed and unsigned values.
   If -1, neither -Wsign-compare nor -Wno-sign-compare has been specified.  */

extern int warn_sign_compare;

/* Nonzero means warn about usage of long long when `-pedantic'.  */

extern int warn_long_long;

/* Nonzero means warn about deprecated conversion from string constant to
   `char *'.  */

extern int warn_write_strings;

/* Nonzero means warn about multiple (redundant) decls for the same single
   variable or function.  */

extern int warn_redundant_decls;

/* Warn about testing equality of floating point numbers.  */

extern int warn_float_equal;

/* Warn about a subscript that has type char.  */

extern int warn_char_subscripts;

/* Warn if a type conversion is done that might have confusing results.  */

extern int warn_conversion;

/* Warn about #pragma directives that are not recognized.  */      

extern int warn_unknown_pragmas; /* Tri state variable.  */  

/* Warn about format/argument anomalies in calls to formatted I/O functions
   (*printf, *scanf, strftime, strfmon, etc.).  */

extern int warn_format;

/* Warn about Y2K problems with strftime formats.  */

extern int warn_format_y2k;

/* Warn about excess arguments to formats.  */

extern int warn_format_extra_args;

/* Warn about zero-length formats.  */

extern int warn_format_zero_length;

/* Warn about non-literal format arguments.  */

extern int warn_format_nonliteral;

/* Warn about possible security problems with calls to format functions.  */

extern int warn_format_security;


/* C/ObjC language option variables.  */


/* Nonzero means message about use of implicit function declarations;
 1 means warning; 2 means error.  */

extern int mesg_implicit_function_declaration;

/* Nonzero means allow type mismatches in conditional expressions;
   just make their values `void'.  */

extern int flag_cond_mismatch;

/* Nonzero means enable C89 Amendment 1 features.  */

extern int flag_isoc94;

/* Nonzero means use the ISO C99 dialect of C.  */

extern int flag_isoc99;

/* Nonzero means that we have builtin functions, and main is an int */

extern int flag_hosted;

/* Nonzero means add default format_arg attributes for functions not
   in ISO C.  */

extern int flag_noniso_default_format_attributes;

/* Nonzero means warn when casting a function call to a type that does
   not match the return type (e.g. (float)sqrt() or (anything*)malloc()
   when there is no previous declaration of sqrt or malloc.  */

extern int warn_bad_function_cast;

/* Warn about traditional constructs whose meanings changed in ANSI C.  */

extern int warn_traditional;

/* Nonzero means warn for a declaration found after a statement.  */

extern int warn_declaration_after_statement;

/* Nonzero means warn for non-prototype function decls
   or non-prototyped defs without previous prototype.  */

extern int warn_strict_prototypes;

/* Nonzero means warn for any global function def
   without separate previous decl.  */

extern int warn_missing_declarations;

/* Nonzero means warn about extern declarations of objects not at
   file-scope level and about *all* declarations of functions (whether
   extern or static) not at file-scope level.  Note that we exclude
   implicit function declarations.  To get warnings about those, use
   -Wimplicit.  */

extern int warn_nested_externs;

/* Warn if main is suspicious.  */

extern int warn_main;

/* Nonzero means warn about possible violations of sequence point rules.  */

extern int warn_sequence_point;

/* Nonzero means to warn about compile-time division by zero.  */
extern int warn_div_by_zero;

/* Nonzero means warn about use of implicit int.  */

extern int warn_implicit_int;

/* Warn about NULL being passed to argument slots marked as requiring
   non-NULL.  */ 
      
extern int warn_nonnull;


/* ObjC language option variables.  */


/* Open and close the file for outputting class declarations, if
   requested (ObjC).  */

extern int flag_gen_declaration;

/* Generate code for GNU or NeXT runtime environment.  */

extern int flag_next_runtime;

/* Tells the compiler that this is a special run.  Do not perform any
   compiling, instead we are to test some platform dependent features
   and output a C header file with appropriate definitions.  */

extern int print_struct_values;

/* ???.  Undocumented.  */

extern const char *constant_string_class_name;

/* Warn if multiple methods are seen for the same selector, but with
   different argument types.  Performs the check on the whole selector
   table at the end of compilation.  */

extern int warn_selector;

/* Warn if a @selector() is found, and no method with that selector
   has been previously declared.  The check is done on each
   @selector() as soon as it is found - so it warns about forward
   declarations.  */

extern int warn_undeclared_selector;

/* Warn if methods required by a protocol are not implemented in the 
   class adopting it.  When turned off, methods inherited to that
   class are also considered implemented.  */

extern int warn_protocol;


/* C++ language option variables.  */


/* Nonzero means don't recognize any extension keywords.  */

extern int flag_no_gnu_keywords;

/* Nonzero means do emit exported implementations of functions even if
   they can be inlined.  */

extern int flag_implement_inlines;

/* Nonzero means do emit exported implementations of templates, instead of
   multiple static copies in each file that needs a definition.  */

extern int flag_external_templates;

/* Nonzero means that the decision to emit or not emit the implementation of a
   template depends on where the template is instantiated, rather than where
   it is defined.  */

extern int flag_alt_external_templates;

/* Nonzero means that implicit instantiations will be emitted if needed.  */

extern int flag_implicit_templates;

/* Nonzero means that implicit instantiations of inline templates will be
   emitted if needed, even if instantiations of non-inline templates
   aren't.  */

extern int flag_implicit_inline_templates;

/* Nonzero means generate separate instantiation control files and
   juggle them at link time.  */

extern int flag_use_repository;

/* Nonzero if we want to issue diagnostics that the standard says are not
   required.  */

extern int flag_optional_diags;

/* Nonzero means we should attempt to elide constructors when possible.  */

extern int flag_elide_constructors;

/* Nonzero means that member functions defined in class scope are
   inline by default.  */

extern int flag_default_inline;

/* Controls whether compiler generates 'type descriptor' that give
   run-time type information.  */

extern int flag_rtti;

/* Nonzero if we want to conserve space in the .o files.  We do this
   by putting uninitialized data and runtime initialized data into
   .common instead of .data at the expense of not flagging multiple
   definitions.  */

extern int flag_conserve_space;

/* Nonzero if we want to obey access control semantics.  */

extern int flag_access_control;

/* Nonzero if we want to check the return value of new and avoid calling
   constructors if it is a null pointer.  */

extern int flag_check_new;

/* Nonzero if we want the new ISO rules for pushing a new scope for `for'
   initialization variables.
   0: Old rules, set by -fno-for-scope.
   2: New ISO rules, set by -ffor-scope.
   1: Try to implement new ISO rules, but with backup compatibility
   (and warnings).  This is the default, for now.  */

extern int flag_new_for_scope;

/* Nonzero if we want to emit defined symbols with common-like linkage as
   weak symbols where possible, in order to conform to C++ semantics.
   Otherwise, emit them as local symbols.  */

extern int flag_weak;

/* Nonzero to use __cxa_atexit, rather than atexit, to register
   destructors for local statics and global objects.  */

extern int flag_use_cxa_atexit;

/* Nonzero means output .vtable_{entry,inherit} for use in doing vtable gc.  */

extern int flag_vtable_gc;

/* Nonzero means make the default pedwarns warnings instead of errors.
   The value of this flag is ignored if -pedantic is specified.  */

extern int flag_permissive;

/* Nonzero means to implement standard semantics for exception
   specifications, calling unexpected if an exception is thrown that
   doesn't match the specification.  Zero means to treat them as
   assertions and optimize accordingly, but not check them.  */

extern int flag_enforce_eh_specs;

/*  The version of the C++ ABI in use.  The following values are
    allowed:

    0: The version of the ABI believed most conformant with the 
       C++ ABI specification.  This ABI may change as bugs are
       discovered and fixed.  Therefore, 0 will not necessarily
       indicate the same ABI in different versions of G++.

    1: The version of the ABI first used in G++ 3.2.

    Additional positive integers will be assigned as new versions of
    the ABI become the default version of the ABI.  */

extern int flag_abi_version;

/* Nonzero means warn about things that will change when compiling
   with an ABI-compliant compiler.  */

extern int warn_abi;

/* Nonzero means warn about implicit declarations.  */

extern int warn_implicit;

/* Nonzero means warn when all ctors or dtors are private, and the class
   has no friends.  */

extern int warn_ctor_dtor_privacy;

/* Nonzero means warn in function declared in derived class has the
   same name as a virtual in the base class, but fails to match the
   type signature of any virtual function in the base class.  */

extern int warn_overloaded_virtual;

/* Nonzero means warn when declaring a class that has a non virtual
   destructor, when it really ought to have a virtual one.  */

extern int warn_nonvdtor;

/* Nonzero means warn when the compiler will reorder code.  */

extern int warn_reorder;

/* Nonzero means warn when synthesis behavior differs from Cfront's.  */

extern int warn_synth;

/* Nonzero means warn when we convert a pointer to member function
   into a pointer to (void or function).  */

extern int warn_pmf2ptr;

/* Nonzero means warn about violation of some Effective C++ style rules.  */

extern int warn_ecpp;

/* Nonzero means warn where overload resolution chooses a promotion from
   unsigned to signed over a conversion to an unsigned of the same size.  */

extern int warn_sign_promo;

/* Nonzero means warn when an old-style cast is used.  */

extern int warn_old_style_cast;

/* Nonzero means warn when non-templatized friend functions are
   declared within a template */

extern int warn_nontemplate_friend;

/* Nonzero means complain about deprecated features.  */

extern int warn_deprecated;

/* Maximum template instantiation depth.  This limit is rather
   arbitrary, but it exists to limit the time it takes to notice
   infinite template instantiations.  */

extern int max_tinst_depth;

/* Nonzero means the expression being parsed will never be evaluated.
   This is a count, since unevaluated expressions can nest.  */

extern int skip_evaluation;

/* C types are partitioned into three subsets: object, function, and
   incomplete types.  */
#define C_TYPE_OBJECT_P(type) \
  (TREE_CODE (type) != FUNCTION_TYPE && TYPE_SIZE (type))

#define C_TYPE_INCOMPLETE_P(type) \
  (TREE_CODE (type) != FUNCTION_TYPE && TYPE_SIZE (type) == 0)

#define C_TYPE_FUNCTION_P(type) \
  (TREE_CODE (type) == FUNCTION_TYPE)

/* For convenience we define a single macro to identify the class of
   object or incomplete types.  */
#define C_TYPE_OBJECT_OR_INCOMPLETE_P(type) \
  (!C_TYPE_FUNCTION_P (type))

/* Record in each node resulting from a binary operator
   what operator was specified for it.  */
#define C_EXP_ORIGINAL_CODE(exp) ((enum tree_code) TREE_COMPLEXITY (exp))

/* Attribute table common to the C front ends.  */
extern const struct attribute_spec c_common_attribute_table[];
extern const struct attribute_spec c_common_format_attribute_table[];

/* Pointer to function to lazily generate the VAR_DECL for __FUNCTION__ etc.
   ID is the identifier to use, NAME is the string.
   TYPE_DEP indicates whether it depends on type of the function or not
   (i.e. __PRETTY_FUNCTION__).  */

extern tree (*make_fname_decl)                  PARAMS ((tree, int));

extern tree identifier_global_value		PARAMS ((tree));
extern void record_builtin_type			PARAMS ((enum rid,
							 const char *, tree));
extern tree build_void_list_node		PARAMS ((void));
extern void start_fname_decls			PARAMS ((void));
extern void finish_fname_decls			PARAMS ((void));
extern const char *fname_as_string		PARAMS ((int));
extern tree fname_decl				PARAMS ((unsigned, tree));
extern const char *fname_string			PARAMS ((unsigned));

extern void check_function_arguments		PARAMS ((tree, tree));
extern void check_function_arguments_recurse	PARAMS ((void (*) (void *,
								   tree,
								   unsigned HOST_WIDE_INT),
							 void *, tree,
							 unsigned HOST_WIDE_INT));
extern void check_function_format		PARAMS ((int *, tree, tree));
extern void set_Wformat				PARAMS ((int));
extern tree handle_format_attribute		PARAMS ((tree *, tree, tree,
							 int, bool *));
extern tree handle_format_arg_attribute		PARAMS ((tree *, tree, tree,
							 int, bool *));
extern void c_common_insert_default_attributes	PARAMS ((tree));
extern int c_common_decode_option		PARAMS ((int, char **));
extern tree c_common_type_for_mode		PARAMS ((enum machine_mode,
							 int));
extern tree c_common_type_for_size		PARAMS ((unsigned int, int));
extern tree c_common_unsigned_type		PARAMS ((tree));
extern tree c_common_signed_type		PARAMS ((tree));
extern tree c_common_signed_or_unsigned_type	PARAMS ((int, tree));
extern tree c_common_truthvalue_conversion	PARAMS ((tree));
extern void c_apply_type_quals_to_decl		PARAMS ((int, tree));
extern tree c_sizeof_or_alignof_type	PARAMS ((tree, enum tree_code, int));
extern tree c_alignof_expr			PARAMS ((tree));
/* Print an error message for invalid operands to arith operation CODE.
   NOP_EXPR is used as a special case (see truthvalue_conversion).  */
extern void binary_op_error			PARAMS ((enum tree_code));
#define my_friendly_assert(EXP, N) (void) \
 (((EXP) == 0) ? (fancy_abort (__FILE__, __LINE__, __FUNCTION__), 0) : 0)

extern tree c_expand_expr_stmt			PARAMS ((tree));
extern void c_expand_start_cond			PARAMS ((tree, int, tree));
extern void c_finish_then                       PARAMS ((void));
extern void c_expand_start_else			PARAMS ((void));
extern void c_finish_else			PARAMS ((void));
extern void c_expand_end_cond			PARAMS ((void));
/* Validate the expression after `case' and apply default promotions.  */
extern tree check_case_value			PARAMS ((tree));
extern tree fix_string_type			PARAMS ((tree));
struct varray_head_tag;
extern tree combine_strings		PARAMS ((struct varray_head_tag *));
extern void constant_expression_warning		PARAMS ((tree));
extern tree convert_and_check			PARAMS ((tree, tree));
extern void overflow_warning			PARAMS ((tree));
extern void unsigned_conversion_warning		PARAMS ((tree, tree));

/* Read the rest of the current #-directive line.  */
extern char *get_directive_line			PARAMS ((void));
#define GET_DIRECTIVE_LINE() get_directive_line ()
#define c_sizeof(T)  c_sizeof_or_alignof_type (T, SIZEOF_EXPR, 1)
#define c_alignof(T) c_sizeof_or_alignof_type (T, ALIGNOF_EXPR, 1)

/* Subroutine of build_binary_op, used for comparison operations.
   See if the operands have both been converted from subword integer types
   and, if so, perhaps change them both back to their original type.  */
extern tree shorten_compare			PARAMS ((tree *, tree *, tree *, enum tree_code *));

extern tree pointer_int_sum			PARAMS ((enum tree_code, tree, tree));
extern unsigned int min_precision		PARAMS ((tree, int));

/* Add qualifiers to a type, in the fashion for C.  */
extern tree c_build_qualified_type              PARAMS ((tree, int));

/* Build tree nodes and builtin functions common to both C and C++ language
   frontends.  */
extern void c_common_nodes_and_builtins		PARAMS ((void));

extern void disable_builtin_function		PARAMS ((const char *));

extern tree build_va_arg			PARAMS ((tree, tree));

extern void c_common_init_options		PARAMS ((enum c_language_kind));
extern bool c_common_post_options		PARAMS ((void));
extern const char *c_common_init		PARAMS ((const char *));
extern void c_common_finish			PARAMS ((void));
extern void c_common_parse_file			PARAMS ((int));
extern HOST_WIDE_INT c_common_get_alias_set	PARAMS ((tree));
extern bool c_promoting_integer_type_p		PARAMS ((tree));
extern int self_promoting_args_p		PARAMS ((tree));
extern tree strip_array_types                   PARAMS ((tree));

/* These macros provide convenient access to the various _STMT nodes.  */

/* Nonzero if this statement should be considered a full-expression,
   i.e., if temporaries created during this statement should have
   their destructors run at the end of this statement.  (In C, this
   will always be false, since there are no destructors.)  */
#define STMT_IS_FULL_EXPR_P(NODE) TREE_LANG_FLAG_1 ((NODE))

/* IF_STMT accessors. These give access to the condition of the if
   statement, the then block of the if statement, and the else block
   of the if statement if it exists.  */
#define IF_COND(NODE)           TREE_OPERAND (IF_STMT_CHECK (NODE), 0)
#define THEN_CLAUSE(NODE)       TREE_OPERAND (IF_STMT_CHECK (NODE), 1)
#define ELSE_CLAUSE(NODE)       TREE_OPERAND (IF_STMT_CHECK (NODE), 2)

/* WHILE_STMT accessors. These give access to the condition of the
   while statement and the body of the while statement, respectively.  */
#define WHILE_COND(NODE)        TREE_OPERAND (WHILE_STMT_CHECK (NODE), 0)
#define WHILE_BODY(NODE)        TREE_OPERAND (WHILE_STMT_CHECK (NODE), 1)

/* DO_STMT accessors. These give access to the condition of the do
   statement and the body of the do statement, respectively.  */
#define DO_COND(NODE)           TREE_OPERAND (DO_STMT_CHECK (NODE), 0)
#define DO_BODY(NODE)           TREE_OPERAND (DO_STMT_CHECK (NODE), 1)

/* RETURN_STMT accessors. These give the expression associated with a
   return statement, and whether it should be ignored when expanding
   (as opposed to inlining).  */
#define RETURN_STMT_EXPR(NODE)  TREE_OPERAND (RETURN_STMT_CHECK (NODE), 0)

/* EXPR_STMT accessor. This gives the expression associated with an
   expression statement.  */
#define EXPR_STMT_EXPR(NODE)    TREE_OPERAND (EXPR_STMT_CHECK (NODE), 0)

/* FOR_STMT accessors. These give access to the init statement,
   condition, update expression, and body of the for statement,
   respectively.  */
#define FOR_INIT_STMT(NODE)     TREE_OPERAND (FOR_STMT_CHECK (NODE), 0)
#define FOR_COND(NODE)          TREE_OPERAND (FOR_STMT_CHECK (NODE), 1)
#define FOR_EXPR(NODE)          TREE_OPERAND (FOR_STMT_CHECK (NODE), 2)
#define FOR_BODY(NODE)          TREE_OPERAND (FOR_STMT_CHECK (NODE), 3)

/* SWITCH_STMT accessors. These give access to the condition, body and
   original condition type (before any compiler conversions)
   of the switch statement, respectively.  */
#define SWITCH_COND(NODE)       TREE_OPERAND (SWITCH_STMT_CHECK (NODE), 0)
#define SWITCH_BODY(NODE)       TREE_OPERAND (SWITCH_STMT_CHECK (NODE), 1)
#define SWITCH_TYPE(NODE)	TREE_OPERAND (SWITCH_STMT_CHECK (NODE), 2)

/* CASE_LABEL accessors. These give access to the high and low values
   of a case label, respectively.  */
#define CASE_LOW(NODE)          TREE_OPERAND (CASE_LABEL_CHECK (NODE), 0)
#define CASE_HIGH(NODE)         TREE_OPERAND (CASE_LABEL_CHECK (NODE), 1)
#define CASE_LABEL_DECL(NODE)   TREE_OPERAND (CASE_LABEL_CHECK (NODE), 2)

/* GOTO_STMT accessor. This gives access to the label associated with
   a goto statement.  */
#define GOTO_DESTINATION(NODE)  TREE_OPERAND (GOTO_STMT_CHECK (NODE), 0)
/* True for goto created artifically by the compiler.  */
#define GOTO_FAKE_P(NODE)	(TREE_LANG_FLAG_0 (GOTO_STMT_CHECK (NODE)))

/* COMPOUND_STMT accessor. This gives access to the TREE_LIST of
   statements associated with a compound statement. The result is the
   first statement in the list. Succeeding nodes can be accessed by
   calling TREE_CHAIN on a node in the list.  */
#define COMPOUND_BODY(NODE)     TREE_OPERAND (COMPOUND_STMT_CHECK (NODE), 0)

/* ASM_STMT accessors. ASM_STRING returns a STRING_CST for the
   instruction (e.g., "mov x, y"). ASM_OUTPUTS, ASM_INPUTS, and
   ASM_CLOBBERS represent the outputs, inputs, and clobbers for the
   statement.  */
#define ASM_CV_QUAL(NODE)       TREE_OPERAND (ASM_STMT_CHECK (NODE), 0)
#define ASM_STRING(NODE)        TREE_OPERAND (ASM_STMT_CHECK (NODE), 1)
#define ASM_OUTPUTS(NODE)       TREE_OPERAND (ASM_STMT_CHECK (NODE), 2)
#define ASM_INPUTS(NODE)        TREE_OPERAND (ASM_STMT_CHECK (NODE), 3)
#define ASM_CLOBBERS(NODE)      TREE_OPERAND (ASM_STMT_CHECK (NODE), 4)

/* DECL_STMT accessor. This gives access to the DECL associated with
   the given declaration statement.  */
#define DECL_STMT_DECL(NODE)    TREE_OPERAND (DECL_STMT_CHECK (NODE), 0)

/* STMT_EXPR accessor.  */
#define STMT_EXPR_STMT(NODE)    TREE_OPERAND (STMT_EXPR_CHECK (NODE), 0)

/* Nonzero if this statement-expression does not have an associated scope.  */
#define STMT_EXPR_NO_SCOPE(NODE) \
   TREE_LANG_FLAG_0 (STMT_EXPR_CHECK (NODE))

/* LABEL_STMT accessor. This gives access to the label associated with
   the given label statement.  */
#define LABEL_STMT_LABEL(NODE)  TREE_OPERAND (LABEL_STMT_CHECK (NODE), 0)

/* COMPOUND_LITERAL_EXPR accessors.  */
#ifndef SGI_MONGOOSE
#define COMPOUND_LITERAL_EXPR_DECL_STMT(NODE)		\
  TREE_OPERAND (COMPOUND_LITERAL_EXPR_CHECK (NODE), 0)
#else
// I can not see where COMPOUND_LITERAL_EXPR_CHECK is defined in gcc-3.2.2
// It appears just a regular check to see if the node is of correct type
// We can do without it.
#define COMPOUND_LITERAL_EXPR_DECL_STMT(NODE)           \
	  TREE_OPERAND (NODE, 0)
#endif /* SGI_MONGOOSE */
#define COMPOUND_LITERAL_EXPR_DECL(NODE)			\
  DECL_STMT_DECL (COMPOUND_LITERAL_EXPR_DECL_STMT (NODE))

/* Nonzero if this SCOPE_STMT is for the beginning of a scope.  */
#define SCOPE_BEGIN_P(NODE) \
  (TREE_LANG_FLAG_0 (SCOPE_STMT_CHECK (NODE)))

/* Nonzero if this SCOPE_STMT is for the end of a scope.  */
#define SCOPE_END_P(NODE) \
  (!SCOPE_BEGIN_P (SCOPE_STMT_CHECK (NODE)))

/* The BLOCK containing the declarations contained in this scope.  */
#define SCOPE_STMT_BLOCK(NODE) \
  (TREE_OPERAND (SCOPE_STMT_CHECK (NODE), 0))

/* Nonzero for a SCOPE_STMT if there were no variables in this scope.  */
#define SCOPE_NULLIFIED_P(NODE) \
  (SCOPE_STMT_BLOCK ((NODE)) == NULL_TREE)

/* Nonzero for a SCOPE_STMT which represents a lexical scope, but
   which should be treated as non-existent from the point of view of
   running cleanup actions.  */
#define SCOPE_NO_CLEANUPS_P(NODE) \
  (TREE_LANG_FLAG_3 (SCOPE_STMT_CHECK (NODE)))

/* Nonzero for a SCOPE_STMT if this statement is for a partial scope.
   For example, in:

     S s;
     l:
     S s2;
     goto l;

   there is (implicitly) a new scope after `l', even though there are
   no curly braces.  In particular, when we hit the goto, we must
   destroy s2 and then re-construct it.  For the implicit scope,
   SCOPE_PARTIAL_P will be set.  */
#define SCOPE_PARTIAL_P(NODE) \
  (TREE_LANG_FLAG_4 (SCOPE_STMT_CHECK (NODE)))

/* Nonzero for an ASM_STMT if the assembly statement is volatile.  */
#define ASM_VOLATILE_P(NODE)			\
  (ASM_CV_QUAL (ASM_STMT_CHECK (NODE)) != NULL_TREE)

/* The VAR_DECL to clean up in a CLEANUP_STMT.  */
#define CLEANUP_DECL(NODE) \
  TREE_OPERAND (CLEANUP_STMT_CHECK (NODE), 0)
/* The cleanup to run in a CLEANUP_STMT.  */
#define CLEANUP_EXPR(NODE) \
  TREE_OPERAND (CLEANUP_STMT_CHECK (NODE), 1)

/* The filename we are changing to as of this FILE_STMT.  */
#ifndef SGI_MONGOOSE
#define FILE_STMT_FILENAME_NODE(NODE) \
  (TREE_OPERAND (FILE_STMT_CHECK (NODE), 0))
#else
// I can not see where FILE_STMT_CHECK is defined in gcc-3.2.2
// It appears just a regular check to see if the node is of correct type
// We can do without it.
#define FILE_STMT_FILENAME_NODE(NODE) \
  (TREE_OPERAND (NODE, 0))
#endif /* SGI_MONGOOSE */
#define FILE_STMT_FILENAME(NODE) \
  (IDENTIFIER_POINTER (FILE_STMT_FILENAME_NODE (NODE)))

/* The line-number at which a statement began.  But if
   STMT_LINENO_FOR_FN_P does holds, then this macro gives the
   line number for the end of the current function instead.  */
#define STMT_LINENO(NODE)			\
  (TREE_COMPLEXITY ((NODE)))

/* If nonzero, the STMT_LINENO for NODE is the line at which the
   function ended.  */
#define STMT_LINENO_FOR_FN_P(NODE)		\
  (TREE_LANG_FLAG_2 ((NODE)))

/* Nonzero if we want the new ISO rules for pushing a new scope for `for'
   initialization variables.  */
#define NEW_FOR_SCOPE_P(NODE) (TREE_LANG_FLAG_0 (NODE))

/* Nonzero if we want to create an ASM_INPUT instead of an
   ASM_OPERAND with no operands.  */
#define ASM_INPUT_P(NODE) (TREE_LANG_FLAG_0 (NODE))

#define DEFTREECODE(SYM, NAME, TYPE, LENGTH) SYM,

enum c_tree_code {
  C_DUMMY_TREE_CODE = LAST_AND_UNUSED_TREE_CODE,
#include "c-common.def"
  LAST_C_TREE_CODE
};

#undef DEFTREECODE

extern void genrtl_do_pushlevel                 PARAMS ((void));
extern void genrtl_goto_stmt                    PARAMS ((tree));
extern void genrtl_expr_stmt                    PARAMS ((tree));
extern void genrtl_expr_stmt_value              PARAMS ((tree, int, int));
extern void genrtl_decl_stmt                    PARAMS ((tree));
extern void genrtl_if_stmt                      PARAMS ((tree));
extern void genrtl_while_stmt                   PARAMS ((tree));
extern void genrtl_do_stmt                      PARAMS ((tree));
extern void genrtl_return_stmt                  PARAMS ((tree));
extern void genrtl_for_stmt                     PARAMS ((tree));
extern void genrtl_break_stmt                   PARAMS ((void));
extern void genrtl_continue_stmt                PARAMS ((void));
extern void genrtl_scope_stmt                   PARAMS ((tree));
extern void genrtl_switch_stmt                  PARAMS ((tree));
extern void genrtl_case_label                   PARAMS ((tree));
extern void genrtl_compound_stmt                PARAMS ((tree));
extern void genrtl_asm_stmt                     PARAMS ((tree, tree,
							 tree, tree,
							 tree, int));
extern void genrtl_decl_cleanup                 PARAMS ((tree));
extern int stmts_are_full_exprs_p               PARAMS ((void));
extern int anon_aggr_type_p                     PARAMS ((tree));

/* For a VAR_DECL that is an anonymous union, these are the various
   sub-variables that make up the anonymous union.  */
#define DECL_ANON_UNION_ELEMS(NODE) DECL_ARGUMENTS ((NODE))

/* In a FIELD_DECL, nonzero if the decl was originally a bitfield.  */
#define DECL_C_BIT_FIELD(NODE) \
  (DECL_LANG_FLAG_4 (FIELD_DECL_CHECK (NODE)) == 1)
#define SET_DECL_C_BIT_FIELD(NODE) \
  (DECL_LANG_FLAG_4 (FIELD_DECL_CHECK (NODE)) = 1)
#define CLEAR_DECL_C_BIT_FIELD(NODE) \
  (DECL_LANG_FLAG_4 (FIELD_DECL_CHECK (NODE)) = 0)

/* In a VAR_DECL, nonzero if the decl is a register variable with
   an explicit asm specification.  */
#define DECL_C_HARD_REGISTER(DECL)  DECL_LANG_FLAG_4 (VAR_DECL_CHECK (DECL))

extern void emit_local_var                      PARAMS ((tree));
extern void make_rtl_for_local_static           PARAMS ((tree));
extern tree expand_cond                         PARAMS ((tree));
extern tree c_expand_return			PARAMS ((tree));
extern tree do_case				PARAMS ((tree, tree));
extern tree build_stmt                          PARAMS ((enum tree_code, ...));
extern tree build_case_label                    PARAMS ((tree, tree, tree));
extern tree build_continue_stmt                 PARAMS ((void));
extern tree build_break_stmt                    PARAMS ((void));
extern tree build_return_stmt                   PARAMS ((tree));

#define COMPOUND_STMT_NO_SCOPE(NODE)	TREE_LANG_FLAG_0 (NODE)

/* Used by the C++ frontend to mark the block around the member
   initializers and cleanups.  */
#define COMPOUND_STMT_BODY_BLOCK(NODE)	TREE_LANG_FLAG_3 (NODE)

extern void c_expand_asm_operands		PARAMS ((tree, tree, tree, tree, int, const char *, int));

/* These functions must be defined by each front-end which implements
   a variant of the C language.  They are used in c-common.c.  */

extern tree build_unary_op                      PARAMS ((enum tree_code,
							 tree, int));
extern tree build_binary_op                     PARAMS ((enum tree_code,
							 tree, tree, int));
extern int lvalue_p				PARAMS ((tree));
extern tree default_conversion                  PARAMS ((tree));

/* Given two integer or real types, return the type for their sum.
   Given two compatible ANSI C types, returns the merged type.  */

extern tree common_type                         PARAMS ((tree, tree));

extern tree expand_tree_builtin                 PARAMS ((tree, tree, tree));

extern tree decl_constant_value		PARAMS ((tree));

/* Handle increment and decrement of boolean types.  */
extern tree boolean_increment			PARAMS ((enum tree_code,
							 tree));

/* Hook currently used only by the C++ front end to reset internal state
   after entering or leaving a header file.  */
extern void extract_interface_info		PARAMS ((void));

extern int case_compare                         PARAMS ((splay_tree_key,
							 splay_tree_key));

extern tree c_add_case_label                    PARAMS ((splay_tree,
							 tree, tree,
							 tree));

extern tree build_function_call			PARAMS ((tree, tree));

extern tree finish_label_address_expr		PARAMS ((tree));

/* Same function prototype, but the C and C++ front ends have
   different implementations.  Used in c-common.c.  */
extern tree lookup_label			PARAMS ((tree));

extern rtx c_expand_expr			PARAMS ((tree, rtx,
							 enum machine_mode,
							 int));

extern int c_safe_from_p                        PARAMS ((rtx, tree));

extern int c_staticp                            PARAMS ((tree));

extern int c_common_unsafe_for_reeval		PARAMS ((tree));

extern const char *init_c_lex			PARAMS ((const char *));

extern void cb_register_builtins		PARAMS ((cpp_reader *));

/* Information recorded about each file examined during compilation.  */

struct c_fileinfo
{
  int time;	/* Time spent in the file.  */
  short interface_only;		/* Flags - used only by C++ */
  short interface_unknown;
};

struct c_fileinfo *get_fileinfo			PARAMS ((const char *));
extern void dump_time_statistics		PARAMS ((void));

extern int c_dump_tree				PARAMS ((void *, tree));

#ifdef KEY
// This function takes a void *, and the type of pointer 
// depends on the 1st argument
extern tree build_omp_stmt			PARAMS ((enum omp_tree_type,
                                                         void *));
/** DAVID CODE BEGIN **/

/* The following functions are Defined in c-semantics.cxx. */

extern tree build_hicuda_stmt PARAMS ((enum hicuda_tree_type, void*));
extern void finish_hicuda_stmt PARAMS ((tree, int, const char*));

/**
 * Used the validate that KERNEL BEGIN and KERNEL END are in the same scope.
 */
extern void new_hc_kernel_scope PARAMS (());
extern void end_hc_kernel_scope PARAMS (());

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/* For use in c-parse.y */
extern tree tree_type_list;
extern tree tree_sym_list;

/**
 * Defined in c-decl.c; used in c-parse.y
 *
 * Chain all types in the global binding level to 'tree_type_list'.
 */
extern void save_global_types_and_symbols();

/**
 * Defined in c-semantics.c; used in c-parse.y
 *
 * Put all types in 'tree_type_list' in the WHIRL type table.
 */
extern void decl_all_types_and_symbols();

/*** DAVID CODE END ***/

#endif

#endif /* ! GCC_C_COMMON_H */
