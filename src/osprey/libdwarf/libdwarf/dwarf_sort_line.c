/*
 * Copyright 2005, 2006 PathScale, Inc.  All Rights Reserved.
 */

/*

  Copyright (C) 2000,2002,2004 Silicon Graphics, Inc.  All Rights Reserved.

  This program is free software; you can redistribute it and/or modify it
  under the terms of version 2.1 of the GNU Lesser General Public License 
  as published by the Free Software Foundation.

  This program is distributed in the hope that it would be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

  Further, this software is distributed without any warranty that it is
  free of the rightful claim of any third person regarding infringement 
  or the like.  Any license provided herein, whether implied or 
  otherwise, applies only to this software file.  Patent licenses, if
  any, provided herein do not apply to combinations of this program with 
  other software, or any other product whatsoever.  

  You should have received a copy of the GNU Lesser General Public 
  License along with this program; if not, write the Free Software 
  Foundation, Inc., 59 Temple Place - Suite 330, Boston MA 02111-1307, 
  USA.

  Contact information:  Silicon Graphics, Inc., 1500 Crittenden Lane,
  Mountain View, CA 94043, or:

  http://www.sgi.com

  For further information regarding this notice, see:

  http://oss.sgi.com/projects/GenInfo/NoticeExplan

*/



#include "config.h"
#include "dwarf_incl.h"
#include <stdio.h>
#include <stdlib.h>
#include "dwarf_line.h"
#ifdef HAVE_ALLOCA_H
#include <alloca.h>
#endif
#include <malloc.h>

#define MINIMUM_POSSIBLE_PROLOG_LEN 10  /* 10 is  based on */
	/*  the definition of the DWARF2/3 line table prolog.
	    The value here should be >8 (accounting for
	    a 64 bit read) and  <= the length
	    of a legal DWARF2/3 line prolog, 
	    which is at least 10 bytes long (but can be longer).
	    What this constant helps avoid is reading past the end of a 
	    malloc'd buffer in _dwarf_update_line_sec().
	*/

static int
  _dwarf_update_line_sec(Dwarf_Small * line_ptr,
			 unsigned long remaining_bytes,
			 int *any_change,
			 int length_size,
			 int *err_code, Dwarf_Small ** new_line_ptr);

/* Used to construct
   a linked list of so we can sort and reorder the line info.
*/
struct a_line_area {
    Dwarf_Addr ala_address;	/* from DW_LNE_set_address */
    Dwarf_Unsigned ala_offset;	/* byte offset in buffer */
    Dwarf_Unsigned ala_length;	/* byte length in buffer */
    int ala_entry_num;		/* to guarantee stable sort */
    struct a_line_area *ala_next;
};



/* 
	returns
	DW_DLV_OK if nothing went wrong.
	DW_DLV_ERROR if could not do anything due to
		error.  the original buffer is unchanged.

	is_64_bit must be passed in by caller and tells
	if this is a 32 or 64bit pointer object section
	being processed.

	err_code must be a non-null pointer to integer.
	If DW_DLV_ERROR is returned that integer is set
	to a dwarf error code so the caller may
	print it for diagnostic purposes.

	*any_change is set here 
		set 0 if no sorting (movement) done.
		set 1 if some sorting (movement) done.
	on all returns. On error return sets to 0.
	

*/
int
_dwarf_ld_sort_lines(void *orig_buffer,
		     unsigned long buffer_len,
		     int is_64_bit, int *any_change, int *err_code)
{

    int length_size = 4;
    Dwarf_Small *orig_line_ptr;	/* our local copy of the user's input
				   buffer */
    Dwarf_Small *line_ptr;	/* starts at orig_line_ptr, gets
				   incremented thru to end of our copy
				   of the input buffer */
    Dwarf_Small *new_line_ptr;	/* output of _dwarf_update_line_sec(),
				   used to update line_ptr as we pass
				   thru compilation units in a .o
				   .debug_line */

    unsigned long remaining_bytes = buffer_len;	/* total length of
						   original area left
						   to be processed.
						   Changes as we pass
						   thru compilation
						   units in a .o
						   .debug_line */

    int sec_res;
    int lany_change = 0;
    int did_change = 0;

    if (is_64_bit)
	length_size = 8;

    *any_change = 0;
    line_ptr = malloc(buffer_len);
    if (!line_ptr) {
	*err_code = DW_DLE_ALLOC_FAIL;
	return DW_DLV_ERROR;
    }
    orig_line_ptr = line_ptr;
    memcpy(line_ptr, orig_buffer, buffer_len);


    /* 
       We must iterate thru each of a set of prologues and line data.
       We process each set in turn. If all pass, we update the
       passed-in buffer. */
    sec_res = DW_DLV_OK;

    for (sec_res = _dwarf_update_line_sec(line_ptr,
					  remaining_bytes,
					  &lany_change,
					  length_size,
					  err_code,
					  &new_line_ptr);
	 (sec_res == DW_DLV_OK) && (remaining_bytes > 0);
	 sec_res = _dwarf_update_line_sec(line_ptr,
					  remaining_bytes,
					  &lany_change,
					  length_size,
					  err_code, &new_line_ptr)) {
	long bytes_used = new_line_ptr - line_ptr;

	line_ptr = new_line_ptr;
	remaining_bytes -= bytes_used;
	if (lany_change) {
	    did_change = 1;
	}
	if (remaining_bytes > 0) {
	    continue;
	}
	break;
    }
    if (sec_res == DW_DLV_ERROR) {
	free(orig_line_ptr);
	return sec_res;
    }


    /* all passed */
    if (did_change) {
	/* So update the passed in buffer orig_buffer is caller's
	   input area. orig_line_ptr is our modified copy of input
	   area. */
	memcpy(orig_buffer, orig_line_ptr, buffer_len);
	*any_change = 1;
    }
    free(orig_line_ptr);

    return sec_res;
}


/* By setting ala_entry_num we guarantee a stable sort,
	no duplicates
   Sorting in address order.
*/
static int
cmpr(const void *lin, const void *rin)
{
    const struct a_line_area *l = lin;
    const struct a_line_area *r = rin;

    if (l->ala_address < r->ala_address) {
	return -1;
    }
    if (l->ala_address > r->ala_address) {
	return 1;
    }
    if (l->ala_entry_num < r->ala_entry_num) {
	return -1;
    }
    if (l->ala_entry_num > r->ala_entry_num) {
	return 1;
    }
    return 0;			/* should never happen. */
}


/*
	On entry:
	  line_ptr must point to first
	  byte of a line group for one (original) .o
	  
	  remaining_bytes is the size of the area pointed to
	  by line_ptr: may be larger than the
	  current original compilation unit .

	  length size is 4 for 32bit pointers, 8 for 64bit pointers
	  in the data pointed to.


	On return:
	  return DW_DLV_OK if all ok.  (ignore 
		*err_code in this case)

	  return DW_DLV_ERROR and set *err_code if an error.

	  If some line data was moved around, set *any_change to 1.
	  If error or no movement, set *any_change to 0;

	  Set *new_line_ptr to one-byte-past the end of the
	  current original compilation unit  (not necessary
	  if returning DW_DLV_ERROR, but not harmful).


	This copies the entire array to a malloc area, then
	mallocs pieces of it (another malloc) for sorting a CU entries
	and copying back.  Then at end  the whole new thing copied in.
	The result is that on error, the input is not touched.

	An alternative would be to just update a piece at a time
	and on error stop updating but leave what was done, done.
	This alternative would save some temporary malloc space.
	
	
*/
static int
_dwarf_update_line_sec(Dwarf_Small * line_ptr,
		       unsigned long remaining_bytes,
		       int *any_change,
		       int length_size,
		       int *err_code, Dwarf_Small ** new_line_ptr)
{


    /* 
       This points to the last byte of the .debug_line portion for the 
       current cu. */
    Dwarf_Small *line_ptr_end;

    /* 
       This points to the end of the statement program prologue for the 
       current cu, and serves to check that the prologue was correctly 
       decoded. */
    Dwarf_Small *check_line_ptr;

    Dwarf_Small *orig_line_ptr;

    /* These are the fields of the statement program header. */
    Dwarf_Unsigned total_length;
    Dwarf_Half version;
    Dwarf_Unsigned prologue_length;
    Dwarf_Small minimum_instruction_length;
    Dwarf_Small default_is_stmt;
    Dwarf_Sbyte line_base;
    Dwarf_Small line_range;
    Dwarf_Small opcode_base;
    struct Dwarf_Debug_s dbg_data;
    Dwarf_Debug dbg = &dbg_data;

    Dwarf_Small *opcode_length = 0;

    /* These are the state machine state variables. */
    Dwarf_Addr address;
    Dwarf_Word line;
    Dwarf_Bool is_stmt;

    struct a_line_area *area_base = 0;
    struct a_line_area *area_current = 0;
    long area_count = 0;

    Dwarf_Addr last_address = 0;
    int need_to_sort = 0;

    Dwarf_Sword i;
    Dwarf_Sword file_entry_count;
    Dwarf_Sword include_directories_count;

    /* 
       This is the current opcode read from the statement program. */
    Dwarf_Small opcode;


    /* 
       These variables are used to decode leb128 numbers. Leb128_num
       holds the decoded number, and leb128_length is its length in
       bytes. */
    Dwarf_Word leb128_num;
    Dwarf_Word leb128_length;
    Dwarf_Sword advance_line;

    /* 
       This is the operand of the latest fixed_advance_pc extended
       opcode. */
    Dwarf_Half fixed_advance_pc;

    /* This is the length of an extended opcode instr.  */
    Dwarf_Word instr_length;
    Dwarf_Small ext_opcode;



    dbg->de_copy_word = memcpy;
    /* 
       Following is a straightforward decoding of the statement
       program prologue information. */
    *any_change = 0;
    orig_line_ptr = line_ptr;
    if(remaining_bytes < MINIMUM_POSSIBLE_PROLOG_LEN) {
        /* We are at the end. Remaining should be zero bytes,
           padding.
           This is really just 'end of CU buffer'
                not an error.
	   The is no 'entry' left so report there is none.
	   We don't want to READ_UNALIGNED the total_length below
	   and then belatedly discover that we read off the end 
	   already.
        */
        return(DW_DLV_NO_ENTRY);
    }

    READ_UNALIGNED(dbg, total_length, Dwarf_Unsigned,
		   line_ptr, length_size);
    line_ptr += length_size;
    line_ptr_end = line_ptr + total_length;
    if (line_ptr_end > line_ptr + remaining_bytes) {
	*err_code = DW_DLE_DEBUG_LINE_LENGTH_BAD;
	return (DW_DLV_ERROR);
    }

    *new_line_ptr = line_ptr_end;
    READ_UNALIGNED(dbg, version, Dwarf_Half,
		   line_ptr, sizeof(Dwarf_Half));
    line_ptr += sizeof(Dwarf_Half);
    if (version != CURRENT_VERSION_STAMP) {
	*err_code = DW_DLE_VERSION_STAMP_ERROR;
	return (DW_DLV_ERROR);
    }

    READ_UNALIGNED(dbg, prologue_length, Dwarf_Unsigned,
		   line_ptr, length_size);
    line_ptr += length_size;
    check_line_ptr = line_ptr;

    minimum_instruction_length = *(Dwarf_Small *) line_ptr;
    line_ptr = line_ptr + sizeof(Dwarf_Small);

    default_is_stmt = *(Dwarf_Small *) line_ptr;
    line_ptr = line_ptr + sizeof(Dwarf_Small);

    line_base = *(Dwarf_Sbyte *) line_ptr;
    line_ptr = line_ptr + sizeof(Dwarf_Sbyte);

    line_range = *(Dwarf_Small *) line_ptr;
    line_ptr = line_ptr + sizeof(Dwarf_Small);

    opcode_base = *(Dwarf_Small *) line_ptr;
    line_ptr = line_ptr + sizeof(Dwarf_Small);

    opcode_length = (Dwarf_Small *)
	alloca(sizeof(Dwarf_Small) * opcode_base);
    for (i = 1; i < opcode_base; i++) {
	opcode_length[i] = *(Dwarf_Small *) line_ptr;
	line_ptr = line_ptr + sizeof(Dwarf_Small);
    }

    include_directories_count = 0;
    while ((*(char *) line_ptr) != '\0') {
	line_ptr = line_ptr + strlen((char *) line_ptr) + 1;
	include_directories_count++;
    }
    line_ptr++;

    file_entry_count = 0;
    while (*(char *) line_ptr != '\0') {


	/* filename = (Dwarf_Small *)line_ptr; */
	line_ptr = line_ptr + strlen((char *) line_ptr) + 1;

	/* dir_index = */
	_dwarf_decode_u_leb128(line_ptr, &leb128_length);
	line_ptr = line_ptr + leb128_length;

	/* time_last_mod = */
	_dwarf_decode_u_leb128(line_ptr, &leb128_length);
	line_ptr = line_ptr + leb128_length;

	/* file_length = */
	_dwarf_decode_u_leb128(line_ptr, &leb128_length);
	line_ptr = line_ptr + leb128_length;

	file_entry_count++;
    }
    line_ptr++;

    if (line_ptr != check_line_ptr + prologue_length) {
	*err_code = DW_DLE_LINE_PROLOG_LENGTH_BAD;
	return (DW_DLV_ERROR);
    }

    /* Initialize the state machine.  */
    address = 0;
    /* file = 1; */
    line = 1;
    /* column = 0; */
    is_stmt = default_is_stmt;
    /* basic_block = false; */
    /* end_sequence = false; */

    /* Start of statement program.  */
    while (line_ptr < line_ptr_end) {
	int type;

	Dwarf_Small *stmt_prog_entry_start = line_ptr;

	opcode = *(Dwarf_Small *) line_ptr;
	line_ptr++;
	/* 'type' is the output */
	WHAT_IS_OPCODE(type, opcode, opcode_base,
		       opcode_length, line_ptr);



	if (type == LOP_DISCARD) {
	    /* do nothing, necessary ops done */
	} else if (type == LOP_SPECIAL) {
	    opcode = opcode - opcode_base;
	    address = address + minimum_instruction_length *
		(opcode / line_range);
	    line = line + line_base + opcode % line_range;

	    /* basic_block = false; */


	} else if (type == LOP_STANDARD) {


	    switch (opcode) {


	    case DW_LNS_copy:{
		    if (opcode_length[DW_LNS_copy] != 0) {
			*err_code = DW_DLE_LINE_NUM_OPERANDS_BAD;
			return (DW_DLV_ERROR);
		    }


		    /* basic_block = false; */
		    break;
		}

	    case DW_LNS_advance_pc:{
		    Dwarf_Unsigned utmp2;

		    if (opcode_length[DW_LNS_advance_pc] != 1) {
			*err_code = DW_DLE_LINE_NUM_OPERANDS_BAD;
			return (DW_DLV_ERROR);
		    }

		    DECODE_LEB128_UWORD(line_ptr, utmp2)
			leb128_num = (Dwarf_Word) utmp2;
		    address =
			address +
			minimum_instruction_length * leb128_num;
		    break;
		}

	    case DW_LNS_advance_line:{
		    Dwarf_Signed stmp;

		    if (opcode_length[DW_LNS_advance_line] != 1) {
			*err_code = DW_DLE_LINE_NUM_OPERANDS_BAD;
			return (DW_DLV_ERROR);
		    }

		    DECODE_LEB128_SWORD(line_ptr, stmp)
			advance_line = (Dwarf_Sword) stmp;
		    line = line + advance_line;
		    break;
		}

	    case DW_LNS_set_file:{
		    Dwarf_Unsigned utmp2;

		    if (opcode_length[DW_LNS_set_file] != 1) {
			*err_code = DW_DLE_LINE_NUM_OPERANDS_BAD;
			return (DW_DLV_ERROR);
		    }

		    DECODE_LEB128_UWORD(line_ptr, utmp2)
			/* file = (Dwarf_Word)utmp2; */
			break;
		}

	    case DW_LNS_set_column:{
		    Dwarf_Unsigned utmp2;

		    if (opcode_length[DW_LNS_set_column] != 1) {
			*err_code = DW_DLE_LINE_NUM_OPERANDS_BAD;
			return (DW_DLV_ERROR);
		    }

		    DECODE_LEB128_UWORD(line_ptr, utmp2)
			/* column = (Dwarf_Word)utmp2; */
			break;
		}

	    case DW_LNS_negate_stmt:{
		    if (opcode_length[DW_LNS_negate_stmt] != 0) {
			*err_code = DW_DLE_LINE_NUM_OPERANDS_BAD;
			return (DW_DLV_ERROR);
		    }

		    is_stmt = !is_stmt;
		    break;
		}

	    case DW_LNS_set_basic_block:{
		    if (opcode_length[DW_LNS_set_basic_block] != 0) {
			*err_code = DW_DLE_LINE_NUM_OPERANDS_BAD;
			return (DW_DLV_ERROR);
		    }

		    /* basic_block = true; */
		    break;
		}

	    case DW_LNS_const_add_pc:{
		    opcode = MAX_LINE_OP_CODE - opcode_base;
		    address = address + minimum_instruction_length *
			(opcode / line_range);

		    break;
		}

	    case DW_LNS_fixed_advance_pc:{
		    if (opcode_length[DW_LNS_fixed_advance_pc] != 1) {
			*err_code = DW_DLE_LINE_NUM_OPERANDS_BAD;
			return (DW_DLV_ERROR);
		    }

		    READ_UNALIGNED(dbg, fixed_advance_pc, Dwarf_Half,
				   line_ptr, sizeof(Dwarf_Half));
		    line_ptr += sizeof(Dwarf_Half);
		    address = address + fixed_advance_pc;
		    break;
		}
	    }
	} else if (type == LOP_EXTENDED) {


	    Dwarf_Unsigned utmp3;

	    DECODE_LEB128_UWORD(line_ptr, utmp3)
		instr_length = (Dwarf_Word) utmp3;
	    ext_opcode = *(Dwarf_Small *) line_ptr;
	    line_ptr++;
	    switch (ext_opcode) {

	    case DW_LNE_end_sequence:{
		    /* end_sequence = true; */

		    address = 0;
		    /* file = 1; */
		    line = 1;
		    /* column = 0; */
		    is_stmt = default_is_stmt;
		    /* basic_block = false; */
		    /* end_sequence = false; */

		    break;
		}

	    case DW_LNE_set_address:{
		    if (instr_length - 1 == length_size) {
			struct a_line_area *area;

			READ_UNALIGNED(dbg, address, Dwarf_Addr,
				       line_ptr, length_size);
			/* Here we need to remember the offset into the 
			   buffer and check to see if address went
			   down. */
			if (address < last_address) {
			    need_to_sort = 1;
			}
			last_address = address;

			area = alloca(sizeof(struct a_line_area));
			area->ala_address = address;
			area->ala_offset = stmt_prog_entry_start -
			    orig_line_ptr;
			area->ala_entry_num = area_count;
			area->ala_next = 0;
			area->ala_length = 0;
			if (area_current) {
			    area_current->ala_next = area;
			    area_current->ala_length =
				area->ala_offset -
				area_current->ala_offset;
			}
			++area_count;
			area_current = area;
			if (area_base == 0) {
			    area_base = area;
			}

			line_ptr += length_size;
		    } else {
			*err_code = DW_DLE_LINE_SET_ADDR_ERROR;
			return (DW_DLV_ERROR);
		    }


		    break;
		}

	    case DW_LNE_define_file:{

		    break;
		}

	    default:{
		    *err_code = DW_DLE_LINE_EXT_OPCODE_BAD;
		    return (DW_DLV_ERROR);
		}
	    }

	}
    }


    if (!need_to_sort) {
	return (DW_DLV_OK);
    }

    /* so now we have something to sort. First, finish off the last
       area record: */
    area_current->ala_length = (line_ptr - orig_line_ptr)	/* final 
								   offset */
	-area_current->ala_offset;

    /* Build and sort a simple array of sections. Forcing a stable sort 
       by comparing on sequence number. We will use the sorted list to
       move sections of this part of the line table. Each 'section'
       starting with a DW_LNE_set_address opcode, on the assumption
       that such only get out of order where there was an ld-cord
       function rearrangement and that it is meaningful to restart the
       line info there. */
    {
	struct a_line_area *ala_array;
	struct a_line_area *local;
	long start_len;
	Dwarf_Small *new_area;
	long i;

	ala_array = malloc(area_count * sizeof(struct a_line_area));
	if (!ala_array) {
	    *err_code = DW_DLE_ALLOC_FAIL;
	    return DW_DLV_ERROR;
	}

	for (local = area_base, i = 0; local;
	     local = local->ala_next, ++i) {

	    ala_array[i] = *local;
	}

	qsort(ala_array, area_count, sizeof(struct a_line_area), cmpr);

	/* Now we must rearrange the pieces of the line table. */

	start_len = (check_line_ptr + prologue_length) - orig_line_ptr;
	new_area = malloc(remaining_bytes);
	if (!new_area) {
	    free(ala_array);
	    *err_code = DW_DLE_ALLOC_FAIL;
	    return DW_DLV_ERROR;
	}
	memcpy(new_area, orig_line_ptr, start_len);
	line_ptr = new_area + start_len;
	for (i = 0; i < area_count; ++i) {
	    memcpy(line_ptr, orig_line_ptr +
		   ala_array[i].ala_offset, ala_array[i].ala_length);
	    line_ptr += ala_array[i].ala_length;
	}

	memcpy(orig_line_ptr, new_area, remaining_bytes);

	free(new_area);
	free(ala_array);
	ala_array = 0;
	new_area = 0;
    }

    *any_change = 1;
    return (DW_DLV_OK);
}
