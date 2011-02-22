/** DAVID CODE BEGIN **/

#include <elf.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <search.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#include "wn.h"
#include "wn_util.h"
#include "stab.h"
#include "aux_stab.h"
#include "irbdata.h"
#include "wintrinsic.h"
#include "glob.h"
#include "pu_info.h"
#include "ir_bread.h"
#include "ir_bwrite.h"
#include "file_util.h"
#include "erglob.h"
#include "err_host.tab"

#include "be_symtab.h"
#include "region_util.h"
#include "region_main.h"

#include "hc_handlers.h"
#include "hc_stack.h"
#include "hc_utils.h"
#include "cuda_utils.h"
#include "hc_ty_cleanup.h"
#include "hc_symtab.h"
#include "hc_cuda_inc.h"
#include "driver.h"


PU_Info *pu_list = NULL;

static ST_IDX main_func_st_idx = ST_IDX_ZERO;     // main function symbol

/* ====================================================================
 *
 * Local data.
 *
 * ====================================================================
 */

#define MAX_FNAME_LENGTH 256

static char fname_in[MAX_FNAME_LENGTH+8];       // '.stage1' suffix
static char fname_out[MAX_FNAME_LENGTH+6];      // '.hc.B' suffix
static char tmp_fname_out[MAX_FNAME_LENGTH+22];

bool vflag = false;

/* For verbose printing */
static INT32 indent = 0;
static INT32 indent_inc = 2;

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

/* WN handlers (plugged into wn_walker)
 *
 * If 'parent' is a BLOCK node, return the next node that should be
 * processed after this call. The caller does not need to worry about
 * deallocating 'wn' if it is removed from the block.
 *
 * If 'parent' is not a BLOCK node, 'wn' is one of its kids. If 'wn'
 * is replaced with another node, it will be placed properly under
 * 'parent' and the new node will be returned. Otherwise, NULL is returned.
 *
 * Note: 'parent' could be NULL, if 'wn' is OPR_FUNC_ENTRY.
 *
 * '*del_wn' is true if 'wn' has been deallocated and false otherwise.
 */
typedef WN* (*WN_HANDLER)(WN* wn, WN* parent, bool *del_wn);

/* A handler called after the given node is processed.
 * This can be used to update the internal states.
 */
typedef void (*WN_POST_HANDLER)(WN *wn);

/* Set before calling wn_walker */
static WN_POST_HANDLER wn_post_handler = NULL;

/**
 * WN traversal routine
 *
 * Same interface as WN_HANDLER except the 1st arg is the actual handler.
 */
static WN* wn_walker(WN_HANDLER handler, WN *wn,
        WN *parent = NULL, bool *del_wn = NULL)
{
    assert(wn != NULL);

    OPCODE op = WN_opcode(wn);
    OPERATOR opr = OPCODE_operator(op);

    if (vflag) {
        if (opr == OPR_FUNC_ENTRY) {
            printf("%*sProcessing FUNC: %s\n", indent, "",
                ST_name(WN_st(wn)));
        } else {
            printf("%*sProcessing WN: %s\n", indent, "",
                OPCODE_name(op));
        }
        indent += indent_inc;
    }

    // Call the handler, which tells me the next node (at the same level
    // as 'wn') to be processed.
    bool del_wn2;
    WN *next_wn = handler(wn, parent, &del_wn2);
    if (del_wn != NULL) *del_wn = del_wn2;

    // Stop processing if it has been deleted.
    if (del_wn2) return next_wn;

    // Handle composite nodes.
    if (opr == OPR_BLOCK) {
        WN *node = WN_first(wn);

        // Process all nodes inside.
        while (node != NULL) {
            node = wn_walker(handler, node, wn);
        }
    } else {
        WN *kid = NULL, *new_kid = NULL;
        UINT nkids = WN_kid_count(wn);

        // Process all the kids.
        for (UINT i = 0; i < nkids; i++) {
            kid = WN_kid(wn,i);
            if (kid == NULL) continue;

            new_kid = wn_walker(handler, kid, wn);
            if (new_kid != NULL) {
                if (vflag) {
                    printf("Replacing kid %u (0x%08X) with 0x%08X\n",
                        i, (UINT)kid, (UINT)new_kid);
                }
                assert(new_kid == WN_kid(wn,i));
            }
        }
    }

    if (wn_post_handler != NULL) wn_post_handler(wn);

    if (vflag) indent -= indent_inc;

    return next_wn;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

/**
 * a WN HANDLER that tries to convert non-DO_LOOPs to DO_LOOPs
 * This should be a pre-processing stage before lowering HiCUDA pragmas.
 */
static WN* regularize_loops(WN *wn, WN *parent, bool *del_wn)
{
    if (del_wn != NULL) *del_wn = false;

    OPERATOR opr = WN_operator(wn);
    if (opr == OPR_WHILE_DO) {
        assert(parent != NULL && WN_operator(parent) == OPR_BLOCK);

        WN *doloop = convert_to_doloop(wn, parent);
        if (doloop != NULL) {
            if (del_wn != NULL) *del_wn = true;
            // Here, we do a little trick: our next node is the new DO_LOOP.
            return doloop;
        }
    }

    return (parent != NULL && WN_operator(parent) == OPR_BLOCK) ?
        WN_next(wn) : NULL;
}

/**
 * the main WN HANDLER that processes HiCUDA directives
 */
static WN* lower_hicuda_pragmas(WN *wn, WN *parent, bool *del_wn)
{
    if (del_wn != NULL) *del_wn = false;

    OPERATOR opr = WN_operator(wn);

    if (opr == OPR_PRAGMA || opr == OPR_XPRAGMA) {
        WN_PRAGMA_ID ptype = (WN_PRAGMA_ID)WN_pragma(wn);
        switch (ptype) {
            case WN_PRAGMA_HC_GLOBAL_COPYIN:
                // GLOBAL COPYIN
                if (del_wn != NULL) *del_wn = true;
                return lower_hc_global_copyin(parent, wn);
            case WN_PRAGMA_HC_GLOBAL_COPYOUT:
                // GLOBAL COPYOUT
                if (del_wn != NULL) *del_wn = true;
                return lower_hc_global_copyout(parent, wn);
            case WN_PRAGMA_HC_GLOBAL_FREE:
                // GLOBAL FREE
                if (del_wn != NULL) *del_wn = true;
                return lower_hc_global_free(parent, wn);

            case WN_PRAGMA_HC_CONST_COPYIN:
                // CONST COPYIN
                if (del_wn != NULL) *del_wn = true;
                return lower_hc_const_copyin(parent, wn);
            case WN_PRAGMA_HC_CONST_REMOVE:
                // CONST REMOVE
                if (del_wn != NULL) *del_wn = true;
                return lower_hc_const_remove(parent, wn);

            case WN_PRAGMA_HC_SHARED_COPYIN:
                // SHARED COPYIN
                if (del_wn != NULL) *del_wn = true;
                return lower_hc_shared_copyin_list(parent, wn);
            case WN_PRAGMA_HC_SHARED_COPYOUT:
                // SHARED COPYOUT
                if (del_wn != NULL) *del_wn = true;
                return lower_hc_shared_copyout(parent, wn);
            case WN_PRAGMA_HC_SHARED_REMOVE:
                // SHARED REMOVE
                if (del_wn != NULL) *del_wn = true;
                return lower_hc_shared_remove(parent, wn);

            case WN_PRAGMA_HC_BARRIER:
                // BARRIER
                if (del_wn != NULL) *del_wn = true;
                return lower_hc_barrier(parent, wn);

            default:
                // Ignore it.
                // return WN_next(wn);
                break;
        }
        // Do not do kernel processing.
    }

    if (opr == OPR_REGION && WN_region_kind(wn) == REGION_KIND_HICUDA) {
        // Get the pragma block.
        WN *pragma_blk = WN_kid1(wn);
        // Get the first pragma.
        WN *pragma = WN_first(pragma_blk);
        assert(pragma != NULL);

        // Handle each type of pragma.
        WN_PRAGMA_ID ptype = (WN_PRAGMA_ID)WN_pragma(pragma);
        switch (ptype) {
            case WN_PRAGMA_HC_KERNEL: {
                // We have only one KERNEL.
                pragma = lower_hc_kernel(wn, pragma);
                break;
            }
            case WN_PRAGMA_HC_KERNEL_PART: {
                // We have only one BLOCK_PART or THREAD_PART.
                pragma = lower_hc_kernel_part(wn, pragma);
                break;
            }
            case WN_PRAGMA_HC_LOOPBLOCK: {
                // We have only one LOOPBLOCK.
                pragma = lower_hc_loopblock(wn, pragma);
                break;
            }
            default:
                fprintf(stderr,
                    "Unknown HiCUDA pragma (1st in a REGION): %u\n",
                    ptype);
                abort();
        }
        // We must be done with all pragmas.
        assert(pragma == NULL);

        // Continue walking the region body.
        WN *body_blk = WN_kid2(wn);
        assert(wn_walker(lower_hicuda_pragmas, body_blk, wn) == NULL);

        /* For the kernel directive, we have more work to do, before the
         * stack elem is popped.
         * NOTE: we are still at the function's local scope. */
        if (ptype == WN_PRAGMA_HC_KERNEL) {
            // Figure out where each shared variable is allocated.
            allocate_svars();

            // Outline the kernel.
            outline_kernel(wn);

            // The region's body has changed.
            body_blk = WN_kid2(wn);
        } else if (ptype == WN_PRAGMA_HC_KERNEL_PART) {
            // Update the internal kernel context.
            end_kernel_part_region();
        }

        // Insert the region body before the region node in the parent block,
        // so that it will not be processed again.
        WN_INSERT_BlockBefore(parent, wn, body_blk);
        // Now, body_blk is dangling.
        WN_kid2(wn) = body_blk = NULL;

        // The next node to be processed is the one after this region.
        WN *next_wn = WN_next(wn);
        // Remove this region node.
        WN_DELETE_FromBlock(parent, wn);
        if (del_wn != NULL) *del_wn = true;

        return next_wn;
    }

    // Do extra processing when inside a kernel. This includes:
    // - replace a variable with the corresponding shared variable
    return kernel_processing(wn, parent, del_wn);
}

/**
 * A WN_POST_HANDLER that updates the current kernel's contextual states
 */
static void
post_lower_hicuda_pragmas(WN *wn) {
    OPERATOR opr = WN_operator(wn);
    switch (opr) {
        case OPR_BLOCK:
            validate_hc_data_context(wn);
            break;

        case OPR_DO_LOOP:
        case OPR_WHILE_DO:
        case OPR_DO_WHILE:
            end_loop_in_kernel(wn);
            break;

        default: break;
    }
}

/* forward declaration */
static void mark_used_functions(ST_IDX func_st_idx);

/**
 * A WN handler that searches for global function symbols, called
 * by mark_used_functions.
 */
static WN*
find_func_calls(WN *wn, WN *parent, bool *del_wn) {
    if (del_wn != NULL) *del_wn = false;

    OPERATOR opr = WN_operator(wn);

    // We don't consider function body node.
    if (opr == OPR_FUNC_ENTRY) return NULL;

    /* Here, we must make sure that the symbol index is global
     * before retrieving the symbol object because the global
     * symtab is the only active scope now. */
    ST_IDX st_idx = OPERATOR_has_sym(opr) ? WN_st_idx(wn) : ST_IDX_ZERO;
    if (st_idx != ST_IDX_ZERO
        && ST_IDX_level(st_idx) == GLOBAL_SYMTAB
        && ST_class(st_idx) == CLASS_FUNC) {
        mark_used_functions(st_idx);
    }

    assert(parent != NULL);

    return (WN_operator(parent) == OPR_BLOCK) ?
        WN_next(wn) : NULL;
}

/**
 * 'func_st_idx' must be a global function symbol.
 * Mark it being used as well as all global function symbols referenced
 * in its function body.
 */
static void
mark_used_functions(ST_IDX func_st_idx) {
    assert(func_st_idx != ST_IDX_ZERO);
    
    /* 'func_st_idx' must be a global symbol, because the global
     * symtab is the only active scope now. */
    assert(ST_IDX_level(func_st_idx) == GLOBAL_SYMTAB);

    ST *func_st = ST_ptr(func_st_idx);
    if (!ST_is_not_used(func_st)) return;
    Clear_ST_is_not_used(func_st);
    if (vflag) printf("Marking function %s\n", ST_name(func_st));

    // Find its function body.
    WN *func_body = find_func_body(pu_list, func_st_idx);
    if (func_body != NULL) {
        // Traverse the function body.
        assert(!wn_walker(find_func_calls, func_body));
    }
}

/**
 * A WN_HANDLER that marks labels referenced in 'wn'.
 */
static WN*
mark_used_labels(WN *wn, WN *parent, bool *del_wn) {
    if (del_wn != NULL) *del_wn = false;

    OPERATOR opr = WN_operator(wn);
    if (opr == OPR_FUNC_ENTRY) return NULL;

    LABEL_IDX lidx = LABEL_IDX_ZERO;

    if (OPERATOR_has_label(opr) && opr != OPR_LABEL) {
        lidx = WN_label_number(wn);
    } else if (OPERATOR_has_last_label(opr)) {
        lidx = WN_last_label(wn);
    }

    if (lidx != LABEL_IDX_ZERO) {
        Clear_LABEL_is_not_used(Label_Table[lidx]);
    }

    return (WN_operator(parent) == OPR_BLOCK) ?  WN_next(wn) : NULL;
}

/**
 * A WN_HANDLER that deletes 'wn' if it is an unused LABEL.
 */
static WN*
remove_unused_labels(WN *wn, WN *parent, bool *del_wn) {
    if (WN_operator(wn) == OPR_LABEL
        && LABEL_is_not_used(Label_Table[WN_label_number(wn)])) {
        // Delete this node.
        if (del_wn != NULL) *del_wn = true;

        if (WN_operator(parent) == OPR_BLOCK) {
            WN *next_wn = WN_next(wn);
            WN_DELETE_FromBlock(parent, wn);
            return next_wn;
        } else {
            assert(replace_wn_kid(parent, wn, NULL));
            WN_DELETE_Tree(wn);
            return NULL;
        }
    } else {
        if (del_wn != NULL) *del_wn = false;
        return (parent != NULL && WN_operator(parent) == OPR_BLOCK) ?
            WN_next(wn) : NULL;
    }
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

/* PU handlers
 * Return true if the PU being processed should be deleted.
 * The handler may insert PUs into 'pu_list' but must not delete any.
 */
typedef bool (*PU_HANDLER)(PU_Info*);

/**
 * Skeleton code to walk through all PUs.
 */
static void
pu_walker(PU_HANDLER handler) {
    // if the current PU should be removed.
    bool del_pu = false;

    PU_Info *curr_pu = pu_list, *prev_pu = NULL, *next_pu = NULL;
    PU_Info *child_pu = NULL;

    while (curr_pu != NULL) {
        Current_PU_Info = curr_pu;

        if (vflag) {
            printf("%*sProcessing PU: %s\n", indent, "",
                ST_name(PU_Info_proc_sym(curr_pu)));
            indent += indent_inc;
        }

        // Push PU memory pool (the no-zeroing/faster version).
        // MEM_POOL_Push(MEM_pu_nz_pool_ptr);

        /* Load the PU into memory, including local symbol tables, maps.
         * NOTE: the WHIRL tree is actually mapped from the input file. */

        // If the symbol table is already in memory, this means it is stored
        // in the PU_Info struct. We need to know this before calling
        // Read_Local_Info because it sets it to Subsect_InMem.
        bool in_mem = (PU_Info_state(curr_pu, WT_SYMTAB) == Subsect_InMem);

        // This allocates all local symbol tables even if section
        // status is IN_MEM.
        Read_Local_Info(MEM_pu_nz_pool_ptr, curr_pu);
        if (in_mem) {
#if 0
            printf("PU %s has symtab is in memory already!\n",
                    ST_name(PU_Info_proc_sym(curr_pu)));
#endif
            // Free the newly created symbol table because we will use
            // the one save in PU_Info.
            Delete_Scope(CURRENT_SYMTAB);

            // Restore the local symbol table (just pointers).
            SCOPE *saved_scope = (SCOPE*)PU_Info_symtab_ptr(curr_pu);
            Scope_tab[CURRENT_SYMTAB] = *saved_scope;
        }

        /* Call the hanlder. */

        del_pu = handler(curr_pu);

        /* Process the nested procedure, if any. */
        if ((child_pu = PU_Info_child(curr_pu)) != NULL) {
            // PU_Info_child(curr_pu) = pu_walker(child_pu, handler);
            fprintf(stderr, "Ignore PU %s, nested in PU %s\n",
                    ST_name(PU_Info_proc_sym(child_pu)),
                    ST_name(PU_Info_proc_sym(curr_pu)));
        }

        // Frees up all memory allocated in the PU memory pool.
        // NOTE: WN nodes are allocated in a separate pool WN_mem_pool_ptr.
        // MEM_POOL_Pop(MEM_pu_nz_pool_ptr);

        // Remove the PU if necessary.
        next_pu = PU_Info_next(curr_pu);
        if (del_pu) {
            // Fix the pointers.
            if (prev_pu == NULL) {
                pu_list = next_pu;
            } else {
                PU_Info_next(prev_pu) = next_pu;
            }
            // Existing PU_Info's (from the input file) are malloc-ed
            // in one shot in Read_Global_Info; new PU_Info's are
            // allocated in Malloc_Mem_Pool (i.e. using malloc).
            // Therefore, I don't know how to deallocate any of these.
        } else {
            prev_pu = curr_pu;
        }

        if (vflag) indent -= indent_inc;

        curr_pu = next_pu;
    }
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

/**
 * STAGE 1: main PU handler for HiCUDA pragmas
 */
static bool
handle_hicuda_pragmas(PU_Info *pu)
{
    WN *func_wn = PU_Info_tree_ptr(pu);

    // Flag the main function.
    ST *func_st = WN_st(func_wn);
    if (strcmp(ST_name(func_st), "main") == 0) {
        PU &func_pu = Pu_Table[ST_pu(func_st)];
        Set_PU_is_mainpu(func_pu);
        Set_PU_no_inline(func_pu);
        Set_PU_no_delete(func_pu);
        main_func_st_idx = PU_Info_proc_sym(pu);
    }

    // Create the local symbol tables for BE.
    BE_symtab_alloc_scope_level(CURRENT_SYMTAB);
    Scope_tab[CURRENT_SYMTAB].st_tab->Register(
            *Be_scope_tab[CURRENT_SYMTAB].be_st_tab);

    REGION_Initialize(func_wn, PU_has_region(Pu_Table[ST_pu(func_st)]));

    // a pre-processing stage: try turning loops into DO_LOOPs
    wn_walker(regularize_loops, func_wn);

    // main stage
    wn_walker(lower_hicuda_pragmas, func_wn);

    // Analyze all constant memory requests and determine the size of the
    // per-PU const variable.
    struct hc_cvar_life *hcls = get_cvar_live_ranges();
    if (hcls != NULL) {
        analyze_cvar_live_ranges(hcls);
        reset_cvar_live_ranges(hcls);
    }

    reset_handler_states_at_pu_end();

    REGION_Finalize();

    // Deallocate the BE local symbol table.
    Scope_tab[CURRENT_SYMTAB].st_tab->Un_register(
            *Be_scope_tab[CURRENT_SYMTAB].be_st_tab);
    Be_scope_tab[CURRENT_SYMTAB].be_st_tab->Clear();

    // Save the local symbol table (just pointers) in the PU.
    SCOPE *saved_scope = TYPE_MEM_POOL_ALLOC(SCOPE, MEM_pu_nz_pool_ptr);
    *saved_scope = Scope_tab[CURRENT_SYMTAB];
    Set_PU_Info_symtab_ptr(pu, saved_scope);

    // No need to write PU to file nor free the local symtab.

    // Do not delete this PU.
    return false;
}

static bool
replace_types_in_pu(PU_Info *pu) {
    WN *func_wn = PU_Info_tree_ptr(pu);

    // Only do work if it is used.
    if (ST_is_not_used(WN_st(func_wn))) return false;

    // Modify the local symtab.
    replace_types_in_symtab(CURRENT_SYMTAB);

    // Go through the WHIRL tree.
    wn_walker(replace_types_in_wn, func_wn);

    return false;
}

static bool
cleanup_labels(PU_Info *pu) {
    WN *func_wn = PU_Info_tree_ptr(pu);

    // Only do work if it is used.
    if (ST_is_not_used(WN_st(func_wn))) return false;

    // Mark all labels unused.
    LABEL_TAB *ltab = Scope_tab[CURRENT_SYMTAB].label_tab;
    UINT size = ltab->Size();
    for (UINT i = 0; i < size; ++i) {
        Set_LABEL_is_not_used(ltab->Entry(i));
    }

    // Go through the WN tree to mark those referenced.
    wn_walker(mark_used_labels, func_wn);

    // Remove unused labels.
    wn_walker(remove_unused_labels, func_wn);

    // Do not delete this PU.
    return false;
}

/**
 * LAST STAGE: selectively write used PUs to the output file,
 */
static bool
write_pu(PU_Info *pu) {
    // saved_scope will be freed when the mempool is popped.
    Set_PU_Info_symtab_ptr(pu, NULL);

    // Make sure the local symbol table is consistent.
    Verify_SYMTAB(CURRENT_SYMTAB);

    // Get the function node.
    WN *func_wn = PU_Info_tree_ptr(pu);
    assert(WN_operator(func_wn) == OPR_FUNC_ENTRY);

    // Check if it is referenced.
    bool pu_not_used = ST_is_not_used(WN_st(func_wn));

    // Write the PU if it is used.
    if (!pu_not_used) {
        Write_PU_Info(pu);
    } else {
        printf("PU %s is not used.\n", ST_name(WN_st(func_wn)));
    }

    // We don't need the local symbol table anymore.
    Free_Local_Info(pu);

    // Delete it if the PU is not used.
    return pu_not_used;
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

static PU_Info* init_global_info(char *fin_name)
{
    // Map the input file content into memory.
    Open_Input_Info(fin_name);

    // Initialize the scope array.
    Initialize_Symbol_Tables(FALSE);
    New_Scope(GLOBAL_SYMTAB, Malloc_Mem_Pool, FALSE);

    // Load the global symbol table and PU_Info list.
    PU_Info *pi = Read_Global_Info(NULL);

    // Initialized predefined global symbols and types,
    // e.g. MTYPE_To_TY (MTYPE_TO_TY_Array)
    // NOTE: must come after Read_Global_Info, in which the string table
    // is initialized.
    Initialize_Special_Global_Symbols();

    // Initialize the BE symbol table.
    BE_symtab_initialize_be_scopes();
    BE_symtab_alloc_scope_level(GLOBAL_SYMTAB);
    Scope_tab[GLOBAL_SYMTAB].st_tab->Register(
            *Be_scope_tab[GLOBAL_SYMTAB].be_st_tab);

    return pi;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

static void usage()
{
    fprintf(stderr, 
        "USAGE:\n"
        "\n"
        "\t [-v] [-TARG=abi:n<32|64>] <Whirl_File_Name>\n"
        "\n"
        "The <Whirl_File_Name> is a mandatory command line argument.\n"
        "The optional -v flag controls verbose trace messages.\n"
        "\n"
    );
}

/**
 * Generate the output file name and put it in fname_out, based on
 * the input file name fname_in.
 *
 * Assume that fname_in contains a valid string.
 */
static void set_output_fname()
{
    // Search backward for the '.' char.
    char *p = fname_in + strlen(fname_in) - 1;
    while (p >= fname_in && *p != '.') --p;

    if (p < fname_in) {
        // No '.' in the fname_in, so append '.hc.B'.
        strcpy(fname_out, fname_in);
        strcat(fname_out, ".hc.B");
    } else {
        // Insert '.hc' before the '.'.
        int offset = p - fname_in;
        char *pout = fname_out;
        strncpy(pout, fname_in, offset);
        pout += offset;
        strncpy(pout, ".hc", 3);
        pout += 3;
        strcpy(pout, p);
    }

    if (vflag) printf("Output file: %s\n", fname_out);
}

/**
 * Parse the command line: set up options and get the input file name.
 * Return true if successful and false otherwise.
 *
 * 'fname_in' contains the input file name if it is specified.
 */
static bool process_cmd_line(int argc, char **argv)
{
    char *arg = NULL;
    bool seen_input_file = false;

    for (int i = 1; i < argc; ++i)
    {
        arg = argv[i];
        if (arg == NULL || strlen(arg) == 0) continue;
        
        if (arg[0] == '-')
        {
            // Process as common option group.
            if (Process_Command_Line_Group(&arg[1], Common_Option_Groups)) {
                continue;
            }
            // Verbose flag.
            if (strcmp(arg, "-v") == 0) { vflag = true; continue; }
            // Give a warning.
            fprintf(stderr, "WARNING: unrecognized command option %s\n", arg);
        }
        else
        {
            // Must be the input file name.
            if (seen_input_file) {
                fprintf(stderr,
                    "ERROR: too many input files on command line\n");
                return false;
            }
	        if (strlen(arg) > MAX_FNAME_LENGTH) {
	            strncpy(fname_in, arg, MAX_FNAME_LENGTH);
	            fname_in[MAX_FNAME_LENGTH] = '\0';
	            fprintf (stderr, "WARNING: input filename truncated to "
		            "(max=%d chars): \"%s\"\n", MAX_FNAME_LENGTH, arg);
            } else {
	            strcpy(fname_in, arg);
            }
            seen_input_file = true;
        }
    }

    // There is exactly one input file.
    if (!seen_input_file) {
        ErrMsg(EC_No_Sources);
        return false;
    }

    // Make sure that the file exists.
    struct stat statbuf;
    if (stat(fname_in, &statbuf) != 0) {
        fprintf(stderr, "ERROR: input file (%s) does not exist\n",
            fname_in);
        return false;
    }

    return true;
}   /* process_cmd_line */

static void load_components(INT argc, char **argv)
{
    INT phase_argc;
    char **phase_argv;

    // Now, we just need the IPL.
    // Get_Phase_Args(PHASE_IPL, 
}


/**
 * ====================================================================
 * Main entry point and driver for the hicuda program.
 * ====================================================================
 */
int main(INT argc, char *argv[], char *envp[])
{
    // Copied from be/be/driver.cxx.
    Handle_Signals();

    // Initialize mempools.
    MEM_Initialize();

    // Copied from be/be/driver.cxx.
    Cur_PU_Name = NULL;
    
    // THIS LINE DOES NOT APPEAR IN BE.
    Set_Error_Tables(Phases, host_errlist);
    Init_Error_Handler(100);
    Set_Error_Line(ERROR_LINE_UNKNOWN);
    Set_Error_File(NULL);
    Set_Error_Phase("HiCUDA Compiler");

    // Need this for common command-line args, defined in
    // common/com/config.cxx
    Preconfigure();

    // Process the command line.
    if (!process_cmd_line(argc, argv)) {
        usage();
        return 1;
    }

    // Must happen after parsing the command line options.
    Configure();

    // Copied from be/be/driver.cxx.
    // NOTE: it includes setting up Enable_Cfold_Aggressive.
    Configure_Source(NULL);

    // Set optimization flags.
    // Enable_Cfold_Aggressive = TRUE;

    Init_Operator_To_Opcode_Table();



    /* Before processing the input file, we need to get the global symbol
     * table of cuda_runtime.h, because we don't want to overwrite
     * Irb_File_Name. */

    // newly allocated char array
    Irb_File_Name = HC_get_cuda_runtime_B();
    if (Irb_File_Name == NULL) return 1;

    // Load the WHIRL file for cuda_runtime.h.
    hc_symtab *hcst = load_global_hc_symtab(Irb_File_Name);

    // The file name will be freed along with 'hcst'.
    Irb_File_Name = NULL;

    /* Start processing the input file. */

    // Load the input file into memory.
    Irb_File_Name = fname_in;
    pu_list = init_global_info(Irb_File_Name);

    // Declare CUDA symbols and types that will be used when processing
    // HiCUDA statements.
    init_cuda_includes();

    // Set the data context.
    init_hc_data_context();

    // This mempool lives across all PUs because local symbol tables
    // are allocated in it and cannot be freed until the end.
    MEM_POOL_Push(MEM_pu_nz_pool_ptr);

    /* STAGE 1: handle all HiCUDA pragmas. */
    wn_post_handler = post_lower_hicuda_pragmas;
    pu_walker(handle_hicuda_pragmas);
    wn_post_handler = NULL;

    /* STAGE 2: flag used global symbols in each PU. */

    // Set all global function symbols to be unused, and those
    // being referenced will be flagged later on.
    ST *st = NULL;
    UINT32 i;
    FOREACH_SYMBOL(GLOBAL_SYMTAB, st, i) {
        if (ST_class(st) == CLASS_FUNC) {
            Set_ST_is_not_used(st);
        }
    }

    // Start marking used symbols from the main function.
    assert(main_func_st_idx != ST_IDX_ZERO);
    mark_used_functions(main_func_st_idx);

    /* STAGE 3: clean up the type table. */

    // Determine sets of identical types.
    init_type_internal_data();
    find_ident_types();

    // Modify the global symtab.
    replace_types_in_symtab(GLOBAL_SYMTAB);

    // Modify each local symtab.
    pu_walker(replace_types_in_pu);

    // Clean up internal data.
    reset_type_internal_data();

    /* STAGE 4: mark global symbols and types that will be declared in
     * cuda_runtime.h. */

    mark_cuda_runtime_symbols(hcst);
    mark_cuda_runtime_types(hcst);

    // Deallocate the memory allocated by the WHIRL file.
    MEM_POOL_FREE(Malloc_Mem_Pool, (void*)hcst->fname);
    free_hc_symtab(hcst); hcst = NULL;

    /* STAGE 5: clean up labels in each PU. */

    pu_walker(cleanup_labels);

    /* LAST STAGE: write used PUs to the output file.
     * We must do this because many unused function symbols have flag
     * ST_IS_NOT_USED set, while violates the WHIRL spec unless we
     * remove their function bodies. */

    // Set up the output file, stored in fname_out.
    set_output_fname();

    // Write to a temp output file first so that we would not ruin an existing
    // output file if the program does not run successfully.
    sprintf(tmp_fname_out, "%s$%u", fname_out, (UINT32)getpid());
    remove(tmp_fname_out);
    if (vflag) printf("Temp output file: %s\n", tmp_fname_out);

    // Open the temp output file for writing.
    Open_Output_Info(tmp_fname_out);

    // Verify and write the used PUs.
    pu_walker(write_pu);

    // Deallocate the BE global symbol table.
    Scope_tab[GLOBAL_SYMTAB].st_tab->Un_register(
            *Be_scope_tab[GLOBAL_SYMTAB].be_st_tab);
    Be_scope_tab[GLOBAL_SYMTAB].be_st_tab->Clear();
    BE_symtab_free_be_scopes();

    // Verify and write the global symbol table and PU list.
    Verify_GLOBAL_SYMTAB();
    Write_Global_Info(pu_list);

    // Close the output file.
    Close_Output_Info();

    // Validate and clean up the data context.
    finish_hc_data_context();

    // ?? Who frees up pu_list structure? It's malloc-ed in Read_Global_Info.

    // All local stuff can be freed now.
    MEM_POOL_Pop(MEM_pu_nz_pool_ptr);

    // Free the pool that stores new WHIRL nodes.
    WN_Mem_Pop();

    // Clean up the memory for mapping the input file.
    Free_Input_Info();

    // Move the temp output file to the real one.
    remove(fname_out);
    rename(tmp_fname_out, fname_out);

    // Who frees up Scope_tab?

    return 0;
} /* main */


/* Dummy definitions to satisify references from routines that got pulled in
 * by the header files but are never called
 */
void Signal_Cleanup (INT sig) { }

#ifndef BACK_END

char * Host_Format_Parm (INT kind, MEM_PTR parm) { return NULL; }

INT8 Debug_Level = 0;

#endif

/*** DAVID CODE END ***/
