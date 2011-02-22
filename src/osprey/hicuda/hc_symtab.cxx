/** DAVID CODE BEGIN **/

#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <elf.h>
#include <sys/elf_whirl.h>
#include <assert.h>

#include "defs.h"
#include "erglob.h"
#include "errors.h"
#include "opcode.h"
#include "mempool.h"
#include "wn.h"
#include "wn_map.h"
#include "strtab.h"
#include "symtab.h"
#include "pu_info.h"
#include "ir_bread.h"
#include "cxx_memory.h"

#include "hc_symtab.h"

static void*
open_input_file(const char *fin_name, INT *mapped_sz) {
    Set_Error_Phase("Reading CUDA WHIRL file");

    void *fhandle = WN_open_input((char*)fin_name, (off_t*)mapped_sz);

    if (fhandle == (void*)REVISION_MISMATCH) {
        ErrMsg(EC_IR_Revision, "<EMPTY>", fin_name);
    } else if (fhandle == (void*)ABI_MISMATCH) {
        ErrMsg(EC_IR_Revision,
            "abi of whirl file doesn't match abi from command-line", fin_name);
    } else if (fhandle == (void*)READER_ERROR) {
        ErrMsg(EC_IR_Open, fin_name, errno);
    }

    return fhandle;
}

static hc_symtab*
init_hc_symtab() {
    hc_symtab *hcst = TYPE_MEM_POOL_ALLOC(hc_symtab, Malloc_Mem_Pool);

    // For now, we only have the global level.
    hcst->max_scope = 2;
    hcst->scope_tab = TYPE_MEM_POOL_ALLOC_N(SCOPE, Malloc_Mem_Pool,
        hcst->max_scope);
    bzero(hcst->scope_tab, hcst->max_scope * sizeof(SCOPE));

    hcst->pu_tab = new PU_TAB(Malloc_Mem_Pool);
    hcst->ty_tab = new TY_TAB(Malloc_Mem_Pool);
    hcst->fld_tab = new FLD_TAB(Malloc_Mem_Pool);
    hcst->tylist_tab = new TYLIST_TAB(Malloc_Mem_Pool);
    hcst->eelist_tab = new EELIST_TAB(Malloc_Mem_Pool);
    hcst->arb_tab = new ARB_TAB(Malloc_Mem_Pool);
    hcst->tcon_tab = new TCON_TAB(Malloc_Mem_Pool);
    hcst->initv_tab = new INITV_TAB(Malloc_Mem_Pool);
    hcst->blk_tab = new BLK_TAB(Malloc_Mem_Pool);

    hcst->str_tab = hcst->tcon_str_tab = NULL;

    return hcst;
}

static void
new_scope(hc_symtab *hcst, SYMTAB_IDX level, MEM_POOL *pool) {
    if (level >= hcst->max_scope) {
        UINT size = hcst->max_scope * sizeof(SCOPE);
        hcst->max_scope *= 2;
        hcst->scope_tab = (SCOPE*)MEM_POOL_Realloc(Malloc_Mem_Pool,
            hcst->scope_tab, size, size*2);
    }

    ST_TAB *st_tab = new ST_TAB(pool);
    INITO_TAB *inito_tab = new INITO_TAB(pool);
    ST_ATTR_TAB *st_attr_tab = new ST_ATTR_TAB(pool);

    LABEL_TAB *label_tab = NULL;
    PREG_TAB *preg_tab = NULL;
    if (level > GLOBAL_SYMTAB) {
        label_tab = new LABEL_TAB(pool);
        preg_tab = new PREG_TAB(pool);
    }

    hcst->scope_tab[level].Init(st_tab, label_tab, preg_tab, inito_tab,
        st_attr_tab, pool);
}

static void
free_scope(hc_symtab *hcst, SYMTAB_IDX level) {
    assert(level < hcst->max_scope);

    SCOPE &scope = hcst->scope_tab[level];

    delete scope.st_tab; scope.st_tab = NULL;
    delete scope.inito_tab; scope.inito_tab = NULL;
    delete scope.st_attr_tab; scope.st_attr_tab = NULL;
    delete scope.label_tab; scope.label_tab = NULL;
    delete scope.preg_tab; scope.preg_tab = NULL;
}

static INT
wn_get_global_symtab(hc_symtab *hcst, void *fhandle) {
    OFFSET_AND_SIZE shdr = get_section(fhandle, SHT_MIPS_WHIRL, WT_GLOBALS);
    if (shdr.offset == 0) return -1;

    const char *base = (char*)fhandle + shdr.offset;

    const GLOBAL_SYMTAB_HEADER_TABLE *gsymtab =
        (GLOBAL_SYMTAB_HEADER_TABLE*)base;

    UINT64 size = shdr.size;

    if (gsymtab->size < sizeof(gsymtab)
        || gsymtab->entries < GLOBAL_SYMTAB_TABLES
        || gsymtab->size > size) return -1;

    for (UINT i = 0; i < GLOBAL_SYMTAB_TABLES; ++i) {
	    if (gsymtab->header[i].offset + gsymtab->header[i].size > size) {
	        return -1;
        }
    }

    for (UINT i = 0; i < GLOBAL_SYMTAB_TABLES; ++i) {
        const SYMTAB_HEADER& hdr = gsymtab->header[i];
        const char *addr = base + hdr.offset;

        switch (hdr.type) {
            case SHDR_FILE:
                // Ignore FILE_INFO.
                break;
            case SHDR_ST:
                hcst->scope_tab[GLOBAL_SYMTAB].st_tab->Transfer(
                    (ST*)addr, hdr.size / hdr.entsize);
                break;
            case SHDR_TY:
                hcst->ty_tab->Transfer((TY*)addr, hdr.size / hdr.entsize);
                break;
            case SHDR_PU:
                hcst->pu_tab->Transfer((PU*)addr, hdr.size / hdr.entsize);
                break;
            case SHDR_FLD:
                hcst->fld_tab->Transfer((FLD*)addr, hdr.size / hdr.entsize);
                break;
            case SHDR_ARB:
                hcst->arb_tab->Transfer((ARB*)addr, hdr.size / hdr.entsize);
                break;
            case SHDR_TYLIST:
                hcst->tylist_tab->Transfer((TYLIST*)addr,
                    hdr.size / hdr.entsize);
                break;
            case SHDR_EELIST:
                hcst->eelist_tab->Transfer((EELIST*)addr,
                    hdr.size / hdr.entsize);
                break;
            case SHDR_TCON:
                hcst->tcon_tab->Transfer((TCON*)addr, hdr.size / hdr.entsize);
                break;
            case SHDR_STR:
                hcst->tcon_str_tab = init_tcon_str_tab(addr, hdr.size);
                break;
            case SHDR_INITO:
                hcst->scope_tab[GLOBAL_SYMTAB].inito_tab->Transfer(
                    (INITO*)addr, hdr.size / hdr.entsize);
                break;
            case SHDR_INITV:
                hcst->initv_tab->Transfer((INITV*)addr,
                    hdr.size / hdr.entsize);
                break;
            case SHDR_BLK:
                hcst->blk_tab->Transfer((BLK*)addr, hdr.size / hdr.entsize);
                break;
            case SHDR_ST_ATTR:
                hcst->scope_tab[GLOBAL_SYMTAB].st_attr_tab->Transfer(
                    (ST_ATTR*)addr, hdr.size / hdr.entsize);
                break;
        }
    }

    return 0;
}

static void
read_global_info(hc_symtab *hcst) {
    Set_Error_Phase("Reading CUDA WHIRL file");

    void *fhandle = hcst->fhandle;

    /* Initialize the string table. */

    OFFSET_AND_SIZE shdr = get_section(fhandle, SHT_MIPS_WHIRL, WT_STRTAB);
    if (shdr.offset == 0) {
        ErrMsg(EC_IR_Scn_Read, "strtab", hcst->fname);
    }
    hcst->str_tab = init_str_tab((char*)fhandle + shdr.offset, shdr.size);

    /* Initialize the remaining tables. */

    if (wn_get_global_symtab(hcst, fhandle) == -1) {
        ErrMsg(EC_IR_Scn_Read, "global symtab", hcst->fname);
    }
}

hc_symtab*
load_global_hc_symtab(const char *fin_name) {
    // Create an empty symtab.
    hc_symtab *hcst = init_hc_symtab();
    hcst->fname = fin_name;

    // Map the input file content into memory.
    hcst->fhandle = open_input_file(fin_name, &hcst->mapped_sz);

    // Create the global scope.
    new_scope(hcst, GLOBAL_SYMTAB, Malloc_Mem_Pool);

    // Fill the global symbol table.
    read_global_info(hcst);

    return hcst;
}

void
free_hc_symtab(hc_symtab *hcst) {
    WN_free_input(hcst->fhandle, (off_t)hcst->mapped_sz);
    hcst->fhandle = NULL;

    // Free the global scope.
    free_scope(hcst, GLOBAL_SYMTAB);

    // Free the scope array.
    MEM_POOL_FREE(Malloc_Mem_Pool, hcst->scope_tab); hcst->scope_tab = NULL;

    // TODO: don't how to free other global tables.
    delete hcst->pu_tab; hcst->pu_tab = NULL;
    delete hcst->ty_tab; hcst->ty_tab = NULL;
    delete hcst->fld_tab; hcst->fld_tab = NULL;
    delete hcst->tylist_tab; hcst->tylist_tab = NULL;
    delete hcst->eelist_tab; hcst->eelist_tab = NULL;
    delete hcst->arb_tab; hcst->arb_tab = NULL;
    delete hcst->tcon_tab; hcst->tcon_tab = NULL;
    delete hcst->initv_tab; hcst->initv_tab = NULL;
    delete hcst->blk_tab; hcst->blk_tab = NULL;

    // Free the struct itself.
    MEM_POOL_FREE(Malloc_Mem_Pool, hcst);
}

/*** DAVID CODE END ***/
