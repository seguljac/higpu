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
 * Module: be_util.h
 * $Revision: 1.1.1.1 $
 * $Date: 2005/10/21 19:00:00 $
 * $Author: marcel $
 * $Source: /proj/osprey/CVS/open64/osprey1.0/be/com/be_util.h,v $
 *
 * Revision history:
 *  06-Dec-95 - Original Version
 *
 * Description:
 *    Utilities for all backend components.
 *
 * ====================================================================
 * ====================================================================
 */
#ifndef be_util_INCLUDED
#define be_util_INCLUDED

#ifdef _KEEP_RCS_ID
static char *be_utilrcs_id = "$Source: /proj/osprey/CVS/open64/osprey1.0/be/com/be_util.h,v $ $Revision: 1.1.1.1 $";
#endif /* _KEEP_RCS_ID */

#ifndef wn_INCLUDED
#include "wn.h"
#endif 

extern void  Reset_Current_PU_Count(void);
extern void  Advance_Current_PU_Count(void);
extern INT32 Current_PU_Count(void);
extern BOOL St_Idx_Is_Intconst(ST_IDX st_idx, INT64 *val);
extern BOOL Wn_Is_Intconst(WN *ldid, INT64 *val);

#endif /* be_util_INCLUDED */
