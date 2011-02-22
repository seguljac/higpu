/** DAVID CODE BEGIN **/

#include <set>
#include <assert.h>

#include "hc_ty_cleanup.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/* index into Ty_Table, i.e. the higher 24 bits of TY_IDX */
typedef mUINT32 TY_TAB_INDEX;
#define TY_TAB_INDEX_ZERO 0

inline TY_TAB_INDEX
TY_tt_idx(TY_IDX ty_idx) {
    return TY_IDX_index(ty_idx);
}

inline UINT32
TY_attr(TY_IDX ty_idx) {
    return ty_idx & 0xFF;
}

inline TY_IDX
replace_tt_idx(TY_IDX ty_idx, TY_TAB_INDEX tt_idx) {
    return (tt_idx << 8) | (ty_idx & 0xFF);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/* Data structure that stores sets of identical types */
typedef set<TY_TAB_INDEX> TY_TAB_INDEX_SET;
typedef TY_TAB_INDEX_SET::iterator TY_TAB_INDEX_SET_ITER;

typedef struct ident_ty_set_t ident_ty_set;
struct ident_ty_set_t {
    TY_TAB_INDEX tt_idx;            // representative
    TY_TAB_INDEX_SET ident_set;     // a set of types identical to tt_idx
                                    // it includes tt_idx

    ident_ty_set *next;
};

static ident_ty_set *its_head = NULL;

static TY_TAB_INDEX
get_representative_ty(TY_TAB_INDEX tt_idx) {
    assert(tt_idx != TY_TAB_INDEX_ZERO);

    ident_ty_set *its = its_head;
    while (its != NULL) {
        if (its->ident_set.find(tt_idx) != its->ident_set.end()) {
            // Find the type in this set.
            assert(its->tt_idx != TY_TAB_INDEX_ZERO);
            return its->tt_idx;
        }
        its = its->next;
    }

    return TY_TAB_INDEX_ZERO;
}

static bool
are_ident_types(TY_TAB_INDEX tt_idx_1, TY_TAB_INDEX tt_idx_2) {
    assert(tt_idx_1 != TY_TAB_INDEX_ZERO
        && tt_idx_2 != TY_TAB_INDEX_ZERO);

    // Search for tt_idx_1.
    ident_ty_set *its = its_head;
    while (its != NULL) {
        if (its->ident_set.find(tt_idx_1) != its->ident_set.end()) {
            // Check if tt_idx_2 is in the same set.
            if (its->ident_set.find(tt_idx_2) != its->ident_set.end()) {
                return true;
            }
        }
        its = its->next;
    }

    return false;
}

/**
 * Return true if this pair of types does not exist in the set list,
 * and false otherwise.
 */
static bool
add_ident_types(TY_TAB_INDEX tt_idx_1, TY_TAB_INDEX tt_idx_2) {
    assert(tt_idx_1 != TY_TAB_INDEX_ZERO
        && tt_idx_2 != TY_TAB_INDEX_ZERO);

    // Search for tt_idx_1.
    ident_ty_set *its = its_head;
    while (its != NULL) {
        if (its->ident_set.find(tt_idx_1) != its->ident_set.end()) break;
        its = its->next;
    }

    // Create a new set if necessary.
    if (its == NULL) {
        its = new ident_ty_set();
        // its = (ident_ty_set*)malloc(sizeof(ident_ty_set));
        // tt_idx_1 is the representative.
        its->tt_idx = tt_idx_1;
        its->ident_set.insert(tt_idx_1);

        // Add it at the list beginning.
        its->next = its_head;
        its_head = its;
    }

    // Add tt_idx_2.
    pair<TY_TAB_INDEX_SET_ITER,bool> ret = its->ident_set.insert(tt_idx_2);

    return ret.second;
}

static void
reset_its_list() {
    ident_ty_set *tmp = NULL;

    while (its_head != NULL) {
        tmp = its_head;
        its_head = its_head->next;
        // Not sure if this frees the set.
        delete tmp;
        // free(tmp);
    }
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/* Data structure that stores a hash table of type index pairs that are
 * known to be different. In order to make sure this table is not too big,
 * we will not put those pairs with different TY_kind in the table. */

typedef struct diff_ty_pair_t diff_ty_pair;
struct diff_ty_pair_t {
    TY_TAB_INDEX tt_idx_1;
    TY_TAB_INDEX tt_idx_2;

    diff_ty_pair *next;
};

static const int ht_sz = 461;   // a prime

// initialized by reset_ty_internal_data.
diff_ty_pair* dtp_ht[ht_sz];

inline int
hash_function(TY_TAB_INDEX tt_idx_1, TY_TAB_INDEX tt_idx_2) {
    // (tt_idx_1 * tt_idx_2) mod ht_sz
    return ((tt_idx_1 % ht_sz) * (tt_idx_2 % ht_sz)) % ht_sz;
}

static bool
are_diff_types(TY_TAB_INDEX tt_idx_1, TY_TAB_INDEX tt_idx_2) {
    assert(tt_idx_1 != TY_TAB_INDEX_ZERO
        && tt_idx_2 != TY_TAB_INDEX_ZERO);

    int hash = hash_function(tt_idx_1, tt_idx_2);

    diff_ty_pair *dtp = dtp_ht[hash];
    while (dtp != NULL) {
        if ((tt_idx_1 == dtp->tt_idx_1 && tt_idx_2 == dtp->tt_idx_2)
            || (tt_idx_1 == dtp->tt_idx_2 && tt_idx_2 == dtp->tt_idx_1)) {
            return true;
        }
        dtp = dtp->next;
    }

    return false;
}

/* Does not do uniqueness check. */
static void
add_diff_types(TY_TAB_INDEX tt_idx_1, TY_TAB_INDEX tt_idx_2) {
    assert(tt_idx_1 != TY_TAB_INDEX_ZERO
        && tt_idx_2 != TY_TAB_INDEX_ZERO);

    int hash = hash_function(tt_idx_1, tt_idx_2);

    diff_ty_pair *dtp = (diff_ty_pair*)malloc(sizeof(diff_ty_pair));
    dtp->tt_idx_1 = tt_idx_1;
    dtp->tt_idx_2 = tt_idx_2;

    // Add it to the list beginning.
    dtp->next = dtp_ht[hash];
    dtp_ht[hash] = dtp;
}

static void
reset_dtp_ht() {
    diff_ty_pair *dtp = NULL, *tmp = NULL;

    for (int i = 0; i < ht_sz; ++i) {
        dtp = dtp_ht[i];
        while (dtp != NULL) {
            tmp = dtp;
            dtp = dtp->next;
            free(tmp);
        }
        dtp_ht[i] = NULL;
    }
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/**
 * Return true if they are identical and false otherwise.
 */
static bool
compare_types(TY_TAB_INDEX tt_idx_1, TY_TAB_INDEX tt_idx_2) {
    assert(tt_idx_1 != TY_TAB_INDEX_ZERO
        && tt_idx_2 != TY_TAB_INDEX_ZERO);

    if (tt_idx_1 == tt_idx_2) return true;

    TY &ty1 = Ty_tab.Entry(tt_idx_1);
    TY &ty2 = Ty_tab.Entry(tt_idx_2);

    TY_KIND ty_kind = TY_kind(ty1);
    if (ty_kind != TY_kind(ty2)) return false;

    // Make sure the size, mtype, and flags are identical.
    if (TY_size(ty1) != TY_size(ty2)) return false;
    if (TY_mtype(ty1) != TY_mtype(ty2)) return false;
    if (TY_flags(ty1) != TY_flags(ty2)) return false;
    // if (TY_name_idx(ty1) != TY_name_idx(ty2)) return false;

    // Check if they have been compared already.
    if (are_ident_types(tt_idx_1, tt_idx_2)) return true;
    if (are_diff_types(tt_idx_1, tt_idx_2)) return false;

    bool ident = false;

    switch (ty_kind) {
        case KIND_SCALAR: {
            // Compare field 'eelist' and 'scalar_flags'.
            ident = (TY_eelist(ty1) == TY_eelist(ty2))
                && (ty1.Scalar_flags() == ty2.Scalar_flags());
            break;
        }
        case KIND_ARRAY: {
            // The element type must be identical.
            TY_TAB_INDEX elem_tt_idx_1 = TY_tt_idx(TY_etype(ty1));
            TY_TAB_INDEX elem_tt_idx_2 = TY_tt_idx(TY_etype(ty2));
            if (!compare_types(elem_tt_idx_1, elem_tt_idx_2)) break;

            // For now, we do a shallow check on array bounds.
            if (ty1.Arb() == ty2.Arb()) ident = true;

            break;
        }
        case KIND_STRUCT: {
            // The type names must be identical.
            if (TY_name_idx(ty1) != TY_name_idx(ty2)) break;

            ident = true;

            // Compare field index.
            FLD_IDX fld_idx_1 = ty1.Fld(), fld_idx_2 = ty2.Fld();
            if (fld_idx_1 == fld_idx_2) break;

            // Detailed field comparison.
            bool done = false;
            do {
                FLD &fld1 = Fld_Table[fld_idx_1++];
                FLD &fld2 = Fld_Table[fld_idx_2++];

                if (fld1.name_idx != fld2.name_idx
                    || fld1.ofst != fld2.ofst
                    || fld1.bsize != fld2.bsize
                    || fld1.bofst != fld2.bofst
                    || fld1.flags != fld2.flags
                    || fld1.st != fld2.st) {
                    ident = false;
                    break;
                }

                // Compare the field types.
                TY_TAB_INDEX fld1_tt_idx = TY_tt_idx(fld1.type);
                TY_TAB_INDEX fld2_tt_idx = TY_tt_idx(fld2.type);
                if ((TY_attr(fld1.type) != TY_attr(fld2.type))
                    || !compare_types(fld1_tt_idx, fld2_tt_idx)) {
                    ident = false;
                    break;
                }

                done = (fld1.flags & FLD_LAST_FIELD);
            } while (!done);

            break;
        }
        case KIND_POINTER: {
            // Compare the pointed type.
            TY_IDX typ1 = TY_pointed(ty1), typ2 = TY_pointed(ty2);
            if (TY_attr(typ1) != TY_attr(typ2)) break;

            TY_TAB_INDEX p1_tt_idx = TY_tt_idx(typ1);
            TY_TAB_INDEX p2_tt_idx = TY_tt_idx(typ2);
            ident = compare_types(p1_tt_idx, p2_tt_idx);

            break;
        }
        case KIND_FUNCTION: {
            // For now, we always assume they are different.
            break;
        }
        default: {
            fprintf(stderr, "Meet KIND_INVALID type (%u)!\n", tt_idx_1);
            break;
        }
    }

    // Add to corresponding internal data structures.
    if (ident) {
        add_ident_types(tt_idx_1, tt_idx_2);
    } else {
        add_diff_types(tt_idx_1, tt_idx_2);
    }

    return ident;
}

void
find_ident_types() {
    INT32 size = Ty_tab.Size();

    // Compare each pair of types.
    for (INT32 i = 1; i < size; ++i) {
        for (INT32 j = i+1; j < size; ++j) {
            if (compare_types(i, j)) {
                printf("Type %u and %u are identical!\n", i, j);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

static void
replace_type(TY_IDX &ty_idx) {
    assert(ty_idx != TY_IDX_ZERO);

    TY_TAB_INDEX tt_idx = get_representative_ty(TY_tt_idx(ty_idx));
    if (tt_idx != TY_TAB_INDEX_ZERO) {
        printf("Replaced index of type %u with %u\n", ty_idx, tt_idx);
        ty_idx = replace_tt_idx(ty_idx, tt_idx);
    }
}

static void
replace_types_in_pu_tab() {
    INT32 size = Pu_Table.Size();

    // Must start from index 1.
    for (INT32 i = 1; i < size; ++i) {
        PU &pu = Pu_Table.Entry(i);
        replace_type(pu.prototype);
    }
}

static void
replace_types_in_ty_tab() {
    INT32 size = Ty_tab.Size();

    // Must start from index 1.
    for (INT32 i = 1; i < size; ++i) {
        TY &ty = Ty_tab.Entry(i);

        switch (TY_kind(ty)) {
            case KIND_ARRAY: {
                // Replace etype field.
                replace_type(ty.u2.etype);
                break;
            }
            case KIND_POINTER: {
                // Replace pointed field.
                replace_type(ty.u2.pointed);
                break;
            }
            default:
                break;
        }
    }
}

static void
replace_types_in_fld_tab() {
    INT32 size = Fld_Table.Size();

    // Must start from index 1.
    for (INT32 i = 1; i < size; ++i) {
        FLD &fld = Fld_Table.Entry(i);
        // Replace type field.
        replace_type(fld.type);
    }
}

static void
replace_types_in_tylist_tab() {
    INT32 size = Tylist_Table.Size();

    // Must start from index 1.
    for (INT32 i = 1; i < size; ++i) {
        TYLIST &tyl = Tylist_Table.Entry(i);
        if (tyl != TY_IDX_ZERO) replace_type(tyl);
    }
}

static void
replace_types_in_st_tab(SYMTAB_IDX level) {
    TY_IDX ty_idx;
    TY_TAB_INDEX tt_idx;

    UINT32 i;
    ST *st;
    FOREACH_SYMBOL(level, st, i) {
        ST_CLASS sc = ST_sym_class(st);
        if (sc == CLASS_VAR || sc == CLASS_CONST || sc == CLASS_PREG) {
            ty_idx = ST_type(st);

            // A PREG may not have a type (like .preg_ret_val).
            if (sc == CLASS_PREG && ty_idx == TY_IDX_ZERO) continue;

            tt_idx = get_representative_ty(TY_tt_idx(ty_idx));
            if (tt_idx != TY_TAB_INDEX_ZERO) {
                printf("ST_TAB(%u): %u into %u\n", level, tt_idx, ty_idx);
                Set_ST_type(st, replace_tt_idx(ty_idx, tt_idx));
            }
        }
    }
}

void
replace_types_in_symtab(SYMTAB_IDX level) {
    if (level == GLOBAL_SYMTAB) {
        replace_types_in_pu_tab();
        replace_types_in_ty_tab();
        replace_types_in_fld_tab();
        replace_types_in_tylist_tab();
    }

    replace_types_in_st_tab(level);
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

WN*
replace_types_in_wn(WN *wn, WN *parent, bool *del_wn) {
    if (del_wn != NULL) *del_wn = false;

    TY_IDX ty_idx;
    TY_TAB_INDEX tt_idx;

    OPERATOR opr = WN_operator(wn);

    if (OPERATOR_has_1ty(opr) || OPERATOR_has_2ty(opr)) {
        ty_idx = WN_ty(wn);
        tt_idx = get_representative_ty(TY_tt_idx(ty_idx));
        if (tt_idx != TY_TAB_INDEX_ZERO) {
            printf("WN(1): %u into %u\n", tt_idx, ty_idx);
            WN_set_ty(wn, replace_tt_idx(ty_idx, tt_idx));
        }

        if (OPERATOR_has_2ty(opr)) {
            ty_idx = WN_load_addr_ty(wn);
            tt_idx = get_representative_ty(TY_tt_idx(ty_idx));
            if (tt_idx != TY_TAB_INDEX_ZERO) {
                printf("WN(2): %u into %u\n", tt_idx, ty_idx);
                WN_set_load_addr_ty(wn, replace_tt_idx(ty_idx, tt_idx));
            }
        }
    }

    if (parent == NULL) return NULL;

    return (WN_operator(parent) == OPR_BLOCK) ? WN_next(wn) : NULL;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

void
init_type_internal_data() {
    its_head = NULL;
    for (int i = 0; i < ht_sz; ++i) dtp_ht[i] = NULL;
}

void
reset_type_internal_data() {
    reset_its_list();
    reset_dtp_ht();
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void
flag_cuda_runtime_types() {
}

/*** DAVID CODE END ***/
