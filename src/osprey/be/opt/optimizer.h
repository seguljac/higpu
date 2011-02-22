//-*-c++-*-
// ====================================================================
// ====================================================================
//
// Module: optimizer.h
// $Revision: 1.1.1.1 $
// $Date: 2005/10/21 19:00:00 $
// $Author: marcel $
// $Source: /proj/osprey/CVS/open64/osprey1.0/be/opt/optimizer.h,v $
//
// Revision history:
//  14-SEP-94 - Original Version
//
// ====================================================================
//
// Copyright (C) 2000, 2001 Silicon Graphics, Inc.  All Rights Reserved.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of version 2 of the GNU General Public License as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it would be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// Further, this software is distributed without any warranty that it
// is free of the rightful claim of any third person regarding
// infringement  or the like.  Any license provided herein, whether
// implied or otherwise, applies only to this software file.  Patent
// licenses, if any, provided herein do not apply to combinations of
// this program with other software, or any other product whatsoever.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write the Free Software Foundation,
// Inc., 59 Temple Place - Suite 330, Boston MA 02111-1307, USA.
//
// Contact information:  Silicon Graphics, Inc., 1600 Amphitheatre Pky,
// Mountain View, CA 94043, or:
//
// http://www.sgi.com
//
// For further information regarding this notice, see:
//
// http://oss.sgi.com/projects/GenInfo/NoticeExplan
//
// ====================================================================
//
// Description:
//
// The external interface for the optimizer.
//
// ====================================================================
// ====================================================================


#ifndef optimizer_INCLUDED
#define optimizer_INCLUDED      "optimizer.h"
#ifdef _KEEP_RCS_ID
static char *optimizerrcs_id =	optimizer_INCLUDED"$Revision$";
#endif /* _KEEP_RCS_ID */

#include "opt_alias_interface.h"

/* The phases of PREOPT */
typedef enum
{
    PREOPT_PHASE,           // used for -PHASE:p
    PREOPT_LNO_PHASE,       // used for -PHASE:l
    PREOPT_DUONLY_PHASE,    // called by LNO, but will disable optimization

/** DAVID CODE BEGIN **/

    // same as PREOPT_DUONLY_PHASE,
    // but invoked to obtain DU-chains for pointer promotion in IPL
    PREOPT_PRE_PP_PHASE,

    // same as PREOPT_DUONLY_PHASE,
    // but invoked to obtain DU-chains for pointer promotion in IPA
    PREOPT_IPA_PRE_PP_PHASE,

    // re-run the preoptmizer after pointer promotion in IPA
    PREOPT_IPA_POST_PP_PHASE,

    // similar to PREOPT_DUONLY_PHASE,
    // but invoked in IPA to determine kernel DAS
    PREOPT_IPA_KERNEL_DAS0_PHASE,

    // similar to PREOPT_DUONLY_PHASE,
    // but invoked in IPA (2nd time) to rebuild kernel scalar DAS
    PREOPT_IPA_KERNEL_DAS1_PHASE,

    // similar to PREOPT_IPA0_PHASE
    // invoked at the end of hiCUDA phase
    PREOPT_IPA2_PHASE,

/*** DAVID CODE END ***/

    MAINOPT_PHASE,          // used for -PHASE:w
    PREOPT_IPA0_PHASE,      // called by IPL
    PREOPT_IPA1_PHASE,      // called by main IPA
} PREOPT_PHASES;

typedef PREOPT_PHASES OPT_PHASE;

#ifdef __cplusplus
extern "C" {
#endif


/* Clients of the optimizer pass a WHIRL tree for the function, and
 * receive back a possibly optimized version of the tree.
 */
class DU_MANAGER;

/** DAVID CODE BEGIN **/
class IPA_NODE;

extern WN* Pre_Optimizer(INT32 /* PREOPT_PHASES */,
        WN*, DU_MANAGER* , ALIAS_MANAGER*, IPA_NODE*);
/*** DAVID CODE END ***/

DU_MANAGER* Create_Du_Manager(MEM_POOL *);
void Delete_Du_Manager(DU_MANAGER *, MEM_POOL *);


#ifdef __cplusplus
}
#endif
#endif /* optimizer_INCLUDED */

