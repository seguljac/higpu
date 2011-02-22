/*
 * Copyright 2004, 2005, 2006 PathScale, Inc.  All Rights Reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of version 2 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it would be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Further, this software is distributed without any warranty that it is
 * free of the rightful claim of any third person regarding infringement
 * or the like.  Any license provided herein, whether implied or
 * otherwise, applies only to this software file.  Patent licenses, if
 * any, provided herein do not apply to combinations of this program with
 * other software, or any other product whatsoever.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write the Free Software Foundation, Inc., 59
 * Temple Place - Suite 330, Boston MA 02111-1307, USA.
 */

// This file contains only Linux-specific code and should be entirely
// #ifdef'd out for Irix.
                                                                                
// Work around the "undefined weak symbol" bug in Linux.
//
// see comments in be/com/weak.cxx.
//
// This file define initialization of pointer variables to symbols defined
// in ipa.so but referenced in be/be.so.

#ifdef __linux__

#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <elf.h>
#include "defs.h"
#include "ipa_summary.h"
#include "ipl_summary.h"
#include "ipa_cg.h"
#include "ipc_file.h"
/** DAVID CODE BEGIN **/
#include "ipa_section_annot.h"
#include "ipa_hc_gpu_data_prop.h"
/*** DAVID CODE END ***/

extern SUMMARY_SYMBOL* (*IPA_get_symbol_file_array_p) (const IP_FILE_HDR&, INT32&);
extern IPA_NODE* (*Get_Node_From_PU_p) (PU_Info*);
extern IP_FILE_HDR_TABLE *IP_File_header_p;

/** DAVID CODE BEGIN **/
extern void (*IPA_map_callsites_p)(IPA_NODE*, WN*, WN_TO_EDGE_MAP*);
extern BOOL (*IPA_get_call_data_summary_p)(IPA_EDGE*,
        STATE_ARRAY**, GLOBAL_ARRAY_TABLE**, MEM_POOL*);
extern void (*IPA_HC_match_gpu_data_with_kernel_das_p)(WN*,
        IPA_NODE*, MEM_POOL*);
/*** DAVID CODE END ***/

struct IPA_INIT
{
    IPA_INIT()
    {
        IPA_get_symbol_file_array_p = IPA_get_symbol_file_array;
        Get_Node_From_PU_p = Get_Node_From_PU;
        IP_File_header_p = &IP_File_header;
/** DAVID CODE BEGIN **/
        IPA_map_callsites_p = IPA_map_callsites;
        IPA_get_call_data_summary_p = IPA_get_call_data_summary;
        IPA_HC_match_gpu_data_with_kernel_das_p =
            IPA_HC_match_gpu_data_with_kernel_das;
/*** DAVID CODE END ***/
    }
} Ipa_Initializer;

#endif // __linux__
