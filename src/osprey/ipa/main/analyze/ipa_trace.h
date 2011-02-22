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


#ifndef ipa_trace_INCLUDED
#define ipa_trace_INCLUDED

/* Trace Options -- Please put them here, and not in arbitrary places */

#define IPA_TRACE_IPA                       0x00001
#define IPA_TRACE_IPAA                      0x00002
#define IPA_TRACE_DETAIL                    0x00004
#define IPA_TRACE_CG                        0x00008
#define IPA_TRACE_IPAA_SUMMARY              0x00010
#define IPA_TRACE_STATS                     0x00020
#define IPA_TRACE_ITERATOR                  0x00040
#define IPA_TRACE_SPLIT_COMMON              0x00080
#define IPA_TRACE_SECTIONS                  0x00100
#define IPA_TRACE_CPROP_CLONING             0x00200
#define IPA_TRACE_LNO_WRITE                 0x00400
#define IPA_TRACE_SECTION_CORRECTNESS       0x00800
#define IPA_TRACE_PREOPT_IPL                0x01000 
#define IPA_TRACE_MODREF                    0x02000
#define IPA_TRACE_COMMON_CONST              0x04000
#define IPA_TRACE_RESHAPE                   0x08000
#define IPA_TRACE_EXCOST                    0x10000
#define IPA_TRACE_SIMPLIFY                  0x20000
#define IPA_TRACE_TUNING                    0x40000
#define IPA_TRACE_TUNING_NEW                0x80000
/** DAVID CODE BEGIN **/
#define IPA_TRACE_SHAPE_PROP                0x100000
#define IPA_TRACE_KERNEL_CLASSIFICATION     0x200000
/*** DAVID CODE END ***/

#endif /* ipa_trace_INCLUDED */
