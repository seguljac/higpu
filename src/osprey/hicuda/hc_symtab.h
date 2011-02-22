/** DAVID CODE BEGIN **/

#ifndef _HICUDA_SYMTAB_H_
#define _HICUDA_SYMTAB_H_

#include "symtab.h"

/**
 * Data structure that holds all symbol tables. This allows muliple WHIRL
 * files being parsed and multiple symbol tables in memory.
 *
 * The client should avoid using convenience functions in symtab*.h that
 * use the globally declared tables implicitly.
 */

typedef struct hc_symtab_t hc_symtab;

struct hc_symtab_t {
    /* file info */
    const char *fname;
    void *fhandle;
    INT mapped_sz;

    SCOPE *scope_tab;
    UINT max_scope;
    
    /* global tables */
    PU_TAB *pu_tab;
    TY_TAB *ty_tab;
    FLD_TAB *fld_tab;
    TYLIST_TAB *tylist_tab;
    EELIST_TAB *eelist_tab;
    ARB_TAB *arb_tab;
    TCON_TAB *tcon_tab;
    INITV_TAB *initv_tab;
    BLK_TAB *blk_tab;

    /* opaque handlers */
    void *str_tab;
    void *tcon_str_tab;
};

/**
 * Load the global symbol table from a given WHIRL file.
 */
extern hc_symtab* load_global_hc_symtab(const char *fin_name);

extern void free_hc_symtab(hc_symtab *hcst);

#endif  // _HICUDA_SYMTAB_H_

/*** DAVID CODE END ***/
