/** DAVID CODE BEGIN **/

#include <assert.h>

#include "cmp_symtab.h"


static int nlevels = 0;
// number of symbols in each level
static UINT32 *nsyms = NULL;


void
init_cmp_st_idx() {
    nlevels = CURRENT_SYMTAB;

    assert(nsyms == NULL);
    nsyms = (UINT32*)malloc(nlevels * sizeof(UINT32));

    for (int i = 0; i < nlevels; ++i) {
        nsyms[i] = Scope_tab[i+1].st_tab->Size();
    }
}

int
total_num_symbols() {
    int total = 0;

    for (int i = 0; i < nlevels; ++i) total += nsyms[i];

    return total;
}

void
reset_cmp_st_idx() {
    if (nlevels != 0) {
        assert(nlevels > 0 && nsyms != NULL);
        nlevels = 0;
        free(nsyms);
        nsyms = NULL;
    } else {
        assert(nsyms == NULL);
    }
}

CMP_ST_IDX
cmp_st_idx(ST_IDX st_idx) {
    if (st_idx == ST_IDX_ZERO) return CMP_ST_IDX_ZERO;

    // Get the level.
    int level = ST_IDX_level(st_idx) - GLOBAL_SYMTAB;

    CMP_ST_IDX idx = CMP_ST_IDX_ZERO + 1;
    for (int i = 0; i < level; ++i) idx += nsyms[i];

    // assume that ST_IDX_index could be 0
    assert(ST_IDX_index(st_idx) < nsyms[level]);
    idx += ST_IDX_index(st_idx);

    return idx;
}

ST_IDX
regular_st_idx(CMP_ST_IDX idx) {
    if (idx == CMP_ST_IDX_ZERO) return ST_IDX_ZERO;

    idx -= (CMP_ST_IDX_ZERO + 1);

    int level = 0;
    while (idx >= nsyms[level]) {
        idx -= nsyms[level];
        ++level;
    }

    return make_ST_IDX(idx, GLOBAL_SYMTAB + level);
}

int
bitvector_to_stlist(bit_vector *idx_bv, ST_IDX *st_list) {
    int n_on_syms = 0, n_syms = idx_bv->num_bits;

    for (int i = 0; i < n_syms; ++i) {
        if (get_bit(idx_bv, i)) {
            st_list[n_on_syms++] = regular_st_idx(i + CMP_ST_IDX_ZERO + 1);
        }
    }

    return n_on_syms;
}

/*** DAVID CODE END ***/
