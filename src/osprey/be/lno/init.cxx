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


// This file contains only Linux-specific code and should be entirely
// #ifdef'd out for Irix.

// Work around the "undefined weak symbol" bug in Linux.
//
// see comments in be/com/weak.cxx.
//
// This file define initialization of pointer variables to symbols defined
// in lno.so but referenced in be/be.so.

#ifdef __linux__

#include "defs.h"
#include "wn.h"
#include "pu_info.h"
#include "lnodriver.h"
/** DAVID CODE BEGIN **/
#include "ptr_promotion.h"
/*** DAVID CODE END ***/

extern void (*lno_main_p)(INT, char**, INT, char**);
extern void (*Lno_Init_p)();
extern void (*Lno_Fini_p)();
extern WN* (*Perform_Loop_Nest_Optimization_p)(PU_Info*, WN*, WN*, BOOL);
/** DAVID CODE BEGIN **/
class DU_MANAGER;
extern WN* (*HC_extract_arr_base_p)(WN *addr_wn, ST_IDX& arr_st_idx);
extern void (*transform_ptr_access_to_array_p)(WN*,
        DU_MANAGER*, ALIAS_MANAGER*);
/*** DAVID CODE END ***/

struct LNO_INIT
{
    LNO_INIT()
    {
        lno_main_p = lno_main;
        Lno_Init_p = Lno_Init;
        Lno_Fini_p = Lno_Fini;
        Perform_Loop_Nest_Optimization_p = Perform_Loop_Nest_Optimization;
/** DAVID CODE BEGIN **/
        HC_extract_arr_base_p = HC_extract_arr_base;
        transform_ptr_access_to_array_p = transform_ptr_access_to_array;
/*** DAVID CODE END ***/
    }
} Lno_Initializer;

#endif // __linux__
