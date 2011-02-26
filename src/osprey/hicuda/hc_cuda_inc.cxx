/** DAVID CODE BEGIN **/

#include <stdio.h>
#include <sys/stat.h>

#include "defs.h"
#include "config_targ.h"        // for Target_ABI
#include "tracing.h"            // for TDEBUG_HICUDA
#include "errors.h"

#include "hc_symtab.h"
#include "hc_cuda_inc.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct _HC_INDEX_PAIR_t
{
    UINT32 idx;         // symbol or type index in the current symtab
    UINT32 hc_idx;      // symbol or type index in the CUDA runtime symtab

    struct _HC_INDEX_PAIR_t *next;
};

typedef struct _HC_INDEX_PAIR_t HC_INDEX_PAIR;

// Add the given index pair to the list <pl> and return the new list.
// Assume that this pair has not existed in the list.
//
static HC_INDEX_PAIR* HC_add_index_pair(HC_INDEX_PAIR *pl,
        UINT32 idx, UINT32 hc_idx)
{
    HC_INDEX_PAIR *p = (HC_INDEX_PAIR*)malloc(sizeof(HC_INDEX_PAIR));
    Is_True(p != NULL, (""));

    p->idx = idx;
    p->hc_idx = hc_idx;
    // Add it to the front of the list.
    p->next = pl;

    return p;
}

// Remove the given index pair to the list <pl> and return the new list.
// Assume that this pair occurs at most once in the list.
//
static HC_INDEX_PAIR* HC_del_index_pair(HC_INDEX_PAIR *pl,
        UINT32 idx, UINT32 hc_idx)
{
    HC_INDEX_PAIR *curr, *prev;

    for (curr = pl, prev = NULL; curr != NULL; prev = curr, curr = curr->next)
    {
        if (curr->idx == idx && curr->hc_idx == hc_idx) break;
    }

    if (curr != NULL)
    {
        if (prev == NULL)
        {
            pl = curr->next;
        }
        else
        {
            prev->next = curr->next;
        }
        free(curr);
    }

    return pl;
}

// Return TRUE if the given index pair exists in the list <pl>
// and FALSE otherwise.
//
inline static BOOL HC_search_index_pair(const HC_INDEX_PAIR *pl,
        UINT32 idx, UINT32 hc_idx)
{
    for ( ; pl != NULL; pl = pl->next)
    {
        if (pl->idx == idx && pl->hc_idx == hc_idx) return TRUE;
    }

    return FALSE;
}

static void HC_free_index_pair_list(HC_INDEX_PAIR *pl)
{
    HC_INDEX_PAIR *p;

    while (pl != NULL)
    {
        p = pl;
        pl = pl->next;
        free(p);
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// forward declaration
static BOOL HC_compare_cuda_symbols(
        hc_symtab *hcst, UINT32 hcst_ss_idx, UINT32 ss_idx,
        HC_INDEX_PAIR **matched_syms_p, HC_INDEX_PAIR **matched_types_p);


// Compare the qualifiers and the alignment of the two types.
inline BOOL HC_compare_TY_IDX(TY_IDX hcst_ty_idx, TY_IDX ty_idx,
        BOOL check_align)
{
    // Check the qualifiers.
    if (TY_is_const(hcst_ty_idx) != TY_is_const(ty_idx)
            || TY_is_volatile(hcst_ty_idx) != TY_is_volatile(ty_idx)
            || TY_is_restrict(hcst_ty_idx) != TY_is_restrict(ty_idx))
    {
        return FALSE;
    }
    // Check the alignment.
    if (check_align && TY_align_exp(hcst_ty_idx) != TY_align_exp(ty_idx))
    {
        return FALSE;
    }

    return TRUE;
}

/* Parameters are table indices, not TY_IDX! */
static BOOL HC_compare_cuda_types(
        hc_symtab *hcst, UINT32 hcst_tt_idx, UINT32 tt_idx,
        HC_INDEX_PAIR **matched_syms_p, HC_INDEX_PAIR **matched_types_p)
{
    Is_True(hcst_tt_idx > 0 && tt_idx > 0, (""));

    // If we have done this matching, just return the result.
    // NOTE: we do not cache mis-matching results.
    if (HC_search_index_pair(*matched_types_p, tt_idx, hcst_tt_idx))
    {
        return TRUE;
    }

    TY &hcst_ty = hcst->ty_tab->Entry(hcst_tt_idx);
    TY &ty = Ty_tab.Entry(tt_idx);

    TY_KIND ty_kind = TY_kind(hcst_ty);

    // They must have the same kind, size, mtype, flags.
    if (ty_kind != TY_kind(ty)) return FALSE;
    if (TY_size(hcst_ty) != TY_size(ty)) return FALSE;
    if (TY_mtype(hcst_ty) != TY_mtype(ty)) return FALSE;
    if (TY_flags(hcst_ty) != TY_flags(ty)){
      if ((TY_flags(hcst_ty) & 0x7fff) == (TY_flags(ty) & 0x7fff)){
	// Exclude openCL is_used_in_kernel from comparison
      } else {
	return FALSE;
      }
    }

    // Assume that the two types are identical (and try to prove otherwise).
    BOOL ident = TRUE;
    // Load this to the database to properly handle recursive comparison.
    *matched_types_p =
        HC_add_index_pair(*matched_types_p, tt_idx, hcst_tt_idx);

    switch (ty_kind)
    {
        case KIND_VOID:
            break;

        case KIND_SCALAR:
        {
            // Compare the scalar flags, to differentiate regular char and
            // special char for example.
            if (ty.Scalar_flags() != hcst_ty.Scalar_flags())
            {
                ident = FALSE;
                break;
            }

            // The types can never have EELIST (for enum) because EELIST is
            // never generated from the input program.
            break;
        }

        case KIND_ARRAY:
        {
            // They must have the same element type.
            UINT32 hcst_elem_tt_idx = TY_IDX_index(TY_etype(hcst_ty));
            UINT32 elem_tt_idx = TY_IDX_index(TY_etype(ty));
            if (!HC_compare_cuda_types(hcst, hcst_elem_tt_idx, elem_tt_idx,
                        matched_syms_p, matched_types_p))
            {
                ident = FALSE;
                break;
            }

            // Compare the ARB entries.
            ARB_IDX arb_idx = ty.Arb();
            ARB_IDX hcst_arb_idx = hcst_ty.Arb();
            while (1)
            {
                ARB& arb = Arb_Table[arb_idx++];
                ARB& hcst_arb = hcst->arb_tab->Entry(hcst_arb_idx++);

                if (arb.flags != hcst_arb.flags
                        || arb.dimension != hcst_arb.dimension)
                {
                    ident = FALSE;
                    break;
                }

                // dimension bounds (must be constant for now)
                if (!(arb.flags & ARB_CONST_LBND)
                        || !(hcst_arb.flags & ARB_CONST_LBND)
                        || arb.Lbnd_val() != hcst_arb.Lbnd_val()
                        || !(arb.flags & ARB_CONST_UBND)
                        || !(hcst_arb.flags & ARB_CONST_UBND)
                        || arb.Ubnd_val() != hcst_arb.Ubnd_val()
                        || !(arb.flags & ARB_CONST_STRIDE)
                        || !(hcst_arb.flags & ARB_CONST_STRIDE)
                        || arb.Stride_val() != hcst_arb.Stride_val())
                {
                    ident = FALSE;
                    break;
                }

                if (arb.flags & ARB_LAST_DIMEN) break;
            }

            break;
        }

        case KIND_STRUCT:
        {
            // They must have the same name (if any).
            STR_IDX str_idx = TY_name_idx(ty);
            STR_IDX hcst_str_idx = TY_name_idx(hcst_ty);
            if (str_idx == STR_IDX_ZERO || hcst_str_idx == STR_IDX_ZERO)
            {
                if (str_idx != hcst_str_idx)
                {
                    ident = FALSE;
                    break;
                }
            }
            else
            {
                const char *hcst_ty_name =
                    get_str(hcst->str_tab, hcst_str_idx);
                if (strcmp(hcst_ty_name, &Str_Table[str_idx]) != 0)
                {
                    ident = FALSE;
                    break;
                }
            }

            // Special case: if any of the FLD_IDX is FLD_IDX_ZERO,
            // they must both be.
            FLD_IDX hcst_fld_idx = hcst_ty.Fld(), fld_idx = ty.Fld();
            if (hcst_fld_idx == FLD_IDX_ZERO || fld_idx == FLD_IDX_ZERO)
            {
                ident = (hcst_fld_idx == fld_idx);
                break;
            }

            // Compare each field of the struct.
            BOOL done = FALSE;
            do
            {
                Is_True(hcst_fld_idx != FLD_IDX_ZERO
                        && fld_idx != FLD_IDX_ZERO, (""));
                FLD &hcst_fld = hcst->fld_tab->Entry(hcst_fld_idx++);
                FLD &fld = Fld_Table[fld_idx++];

                // name
                const char *hcst_fld_name = get_str(hcst->str_tab,
                    hcst_fld.name_idx);
                if (strcmp(hcst_fld_name, &Str_Table[fld.name_idx]) != 0)
                {
                    ident = FALSE;
                    break;
                }

                // ofst, bsize, bofst, flags
                if (hcst_fld.ofst != fld.ofst
                        || hcst_fld.bsize != fld.bsize
                        || hcst_fld.bofst != fld.bofst
                        || hcst_fld.flags != fld.flags)
                {
                    ident = FALSE;
                    break;
                }

                // st
                if (hcst_fld.st != ST_IDX_ZERO && fld.st != ST_IDX_ZERO)
                {
                    // They must be global symbols.
                    Is_True(ST_IDX_level(hcst_fld.st) == GLOBAL_SYMTAB
                            && ST_IDX_level(fld.st) == GLOBAL_SYMTAB, (""));

                    if (!HC_compare_cuda_symbols(hcst,
                                ST_IDX_index(hcst_fld.st),
                                ST_IDX_index(fld.st),
                                matched_syms_p, matched_types_p))
                    {
                        printf("comparing fld symbols!\n");
                        ident = FALSE;
                        break;
                    }
                }
                else if (hcst_fld.st != fld.st)
                {
                    ident = FALSE;
                    break;
                }

                // type
                UINT32 hcst_fld_tt_idx = TY_IDX_index(hcst_fld.type);
                UINT32 fld_tt_idx = TY_IDX_index(fld.type);
                if (!HC_compare_TY_IDX(hcst_fld.type, fld.type, TRUE)
                        || !HC_compare_cuda_types(hcst, hcst_fld_tt_idx,
                            fld_tt_idx, matched_syms_p, matched_types_p))
                {
                    ident = FALSE;
                    break;
                }

                done = (fld.flags & FLD_LAST_FIELD);
            } while (!done);

            break;
        }

        case KIND_POINTER:
        {
            // They must point to the same type.
            TY_IDX hcst_pty = TY_pointed(hcst_ty), pty = TY_pointed(ty);
            if (!HC_compare_TY_IDX(hcst_pty, pty, TRUE)
                    || !HC_compare_cuda_types(hcst, TY_IDX_index(hcst_pty),
                        TY_IDX_index(pty), matched_syms_p, matched_types_p))
            {
                ident = FALSE;
                break;
            }

            break;
        }

        case KIND_FUNCTION:
        {
            // TODO: should we ensure that they have the same name?

            // Go through the list of return and parameter types.
            TYLIST_IDX tyl_idx = TY_tylist(ty);
            TYLIST_IDX hcst_tyl_idx = TY_tylist(hcst_ty);
            TY_IDX param_ty_idx, hcst_param_ty_idx;
            while (1)
            {
                param_ty_idx = Tylist_Table[tyl_idx++];
                hcst_param_ty_idx = hcst->tylist_tab->Entry(hcst_tyl_idx++);

                if (param_ty_idx == TY_IDX_ZERO
                        || hcst_param_ty_idx == TY_IDX_ZERO)
                {
                    if (param_ty_idx != hcst_param_ty_idx) ident = FALSE;
                    break;
                }

                if (!HC_compare_cuda_types(hcst,
                            TY_IDX_index(hcst_param_ty_idx),
                            TY_IDX_index(param_ty_idx),
                            matched_syms_p, matched_types_p))
                {
                    ident = FALSE;
                    break;
                }
            }

            // Compare the function attributes.
            if (ty.Pu_flags() != hcst_ty.Pu_flags())
            {
                ident = FALSE;
                break;
            }

            break;
        }

        default:
            Fail_FmtAssertion("COMPARE_CUDA_TYPES: meet KIND_INVALID!\n");
            break;
    }

    if (!ident)
    {
        // Remove it from the database.
        *matched_types_p =
            HC_del_index_pair(*matched_types_p, tt_idx, hcst_tt_idx);
    }

    return ident;
}

/* Parameters are table indices, not ST_IDX! */
static BOOL HC_compare_cuda_symbols(
        hc_symtab *hcst, UINT32 hcst_ss_idx, UINT32 ss_idx,
        HC_INDEX_PAIR **matched_syms_p, HC_INDEX_PAIR **matched_types_p)
{
    Is_True(hcst_ss_idx > 0 && ss_idx > 0, (""));

    // If we have done this matching, just return the result.
    // NOTE: we do not cache mis-matching results.
    if (HC_search_index_pair(*matched_syms_p, ss_idx, hcst_ss_idx))
    {
        return TRUE;
    }

    ST &hcst_st = hcst->scope_tab[GLOBAL_SYMTAB].st_tab->Entry(hcst_ss_idx);
    ST &st = Scope_tab[GLOBAL_SYMTAB].st_tab->Entry(ss_idx);

    // They must have the same class, storage class, export type, flags.
    ST_CLASS hcst_sc = ST_sym_class(hcst_st);
    if (hcst_sc != ST_sym_class(st)) return FALSE;
    if (ST_sclass(hcst_st) != ST_sclass(st)) return FALSE;
    if (ST_export(hcst_st) != ST_export(st)) return FALSE;
    if (hcst_st.flags != st.flags) return FALSE;
    if (hcst_st.flags_ext != st.flags_ext) return FALSE;

    // Assume that the two symbols are identical (and try to prove otherwise).
    BOOL ident = TRUE;
    // Before we recursively compare the base symbol, assume that the given
    // two symbols are identical. Otherwise, this may end up in an infinite
    // recursion if the current symbol is a field of a struct symbol.
    *matched_syms_p =
        HC_add_index_pair(*matched_syms_p, ss_idx, hcst_ss_idx);

    switch (hcst_sc)
    {
        case CLASS_VAR:
        {
            // They must have the same name.
            const char *hcst_st_name = get_str(hcst->str_tab,
                ST_name_idx(hcst_st));
            if (strcmp(hcst_st_name, ST_name(st)) != 0)
            {
                ident = FALSE;
                break;
            }

            // They must have the same type.
            TY_IDX hcst_ty_idx = ST_type(hcst_st), ty_idx = ST_type(st);
            // NOTE: we do not care about alignment here.
            if (!HC_compare_TY_IDX(hcst_ty_idx, ty_idx, FALSE)
                    || !HC_compare_cuda_types(
                        hcst, TY_IDX_index(hcst_ty_idx), TY_IDX_index(ty_idx),
                        matched_syms_p, matched_types_p))
            {
                ident = FALSE;
                break;
            }

            break;
        }

        case CLASS_FUNC:
        {
            // They must have the same name.
            const char *hcst_st_name = get_str(hcst->str_tab,
                ST_name_idx(hcst_st));
            if (strcmp(hcst_st_name, ST_name(st)) != 0)
            {
                ident = FALSE;
                break;
            }

            // They must have the same prototype.
            PU& pu = Pu_Table[ST_pu(st)];
            PU& hcst_pu = hcst->pu_tab->Entry(ST_pu(hcst_st));
            // level
            // TODO: compare flags?
            if (PU_lexical_level(pu) != PU_lexical_level(hcst_pu))
            {
                ident = FALSE;
                break;
            }
            // prototype
            TY_IDX ty_idx = PU_prototype(pu);
            TY_IDX hcst_ty_idx = PU_prototype(hcst_pu);
            if (!HC_compare_TY_IDX(hcst_ty_idx, ty_idx, TRUE)
                    || !HC_compare_cuda_types(
                        hcst, TY_IDX_index(hcst_ty_idx), TY_IDX_index(ty_idx),
                        matched_syms_p, matched_types_p))
            {
                ident = FALSE;
                break;
            }

            break;
        }

        case CLASS_CONST:
        {
            // For now, they are always different.
            ident = FALSE;
            break;
        }

        case CLASS_PREG:
        {
            // For now, they are always different.
            ident = FALSE;
            break;
        }

        case CLASS_BLOCK:
        {
            // For now, they are always different.
            ident = FALSE;
            break;
        }

        case CLASS_NAME:
        {
            // For now, they are always different.
            ident = FALSE;
            break;
        }

        default:
            Fail_FmtAssertion("COMPARE_CUDA_SYMBOLS: meet CLASS_UNK!\n");
            break;
    }

    // Check base_idx and offset.
    if (ident)
    {
        ST_IDX base_idx = ST_base_idx(st);
        ST_IDX hcst_base_idx = ST_base_idx(hcst_st);
        BOOL st_alias = (ST_st_idx(st) != base_idx);
        BOOL hcst_st_alias = (ST_st_idx(hcst_st) != hcst_base_idx);
        if (hcst_st_alias != st_alias)
        {
            ident = FALSE;
        }
        else if (hcst_st_alias)
        {
            // Both symbols have a different base symbol.
            Is_True(ST_IDX_level(hcst_base_idx) == GLOBAL_SYMTAB, (""));
            Is_True(ST_IDX_level(base_idx) == GLOBAL_SYMTAB, (""));

            if (ST_ofst(hcst_st) != ST_ofst(st)
                    || !HC_compare_cuda_symbols(hcst,
                        ST_IDX_index(hcst_base_idx), ST_IDX_index(base_idx),
                        matched_syms_p, matched_types_p))
            {
                ident = FALSE;
            }
        }
        else
        {
            Is_True(ST_ofst(hcst_st) == 0 && ST_ofst(st) == 0, (""));
        }
    }

    if (!ident)
    {
        // Remove it from the database.
        *matched_syms_p =
            HC_del_index_pair(*matched_syms_p, ss_idx, hcst_ss_idx);
    }

    return ident;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void HC_mark_cuda_runtime_symbols_and_types(hc_symtab *hcst)
{
    Is_True(hcst != NULL, (""));

    // We maintain a database of matched types and symbols to avoid redundant
    // work and to properly handle recursive definition of data types, e.g.,
    // linked list node definition.
    HC_INDEX_PAIR *matched_types = NULL, *matched_syms = NULL;

    UINT32 tt_sz = TY_Table_Size();
    UINT32 hcst_tt_sz = hcst->ty_tab->Size();
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Comparing %u x %u types ...\n", tt_sz, hcst_tt_sz);
    }

    // This is a summary of the above database.
    BOOL is_cuda_type[tt_sz];

    // Go through each type to check if it is a CUDA type.
    //
    // We do not set the type flag immediately because it may affect the check
    // of other types and symbols.
    //
    for (UINT32 i = 1; i < tt_sz; ++i)
    {
        is_cuda_type[i] = FALSE;
        for (UINT32 j = 1; j < hcst_tt_sz; ++j)
        {
            if (HC_compare_cuda_types(hcst, j, i,
                        &matched_syms, &matched_types))
            {
                if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
                {
                    fprintf(TFile, "CUDA type: (%u, %u)\n", j, i);
                }
                is_cuda_type[i] = TRUE;
                break;
            }
        }
    }

    // For each global symbol, check if it belongs to CUDA runtime.
    UINT32 st_sz = ST_Table_Size(GLOBAL_SYMTAB);
    UINT32 hcst_st_sz = hcst->scope_tab[GLOBAL_SYMTAB].st_tab->Size();
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Comparing %u x %u global symbols ...\n",
                st_sz, hcst_st_sz);
    }

    BOOL is_cuda_symbol[st_sz];

    // Go through each symbol to check if it is a CUDA symbol.
    //
    // We do not set the CUDA attribute immediately because it may affect
    // the check of other symbols.
    //
    for (UINT32 i = 1; i < st_sz; ++i)
    {
        is_cuda_symbol[i] = FALSE;
        for (UINT32 j = 1; j < hcst_st_sz; ++j)
        {
            if (HC_compare_cuda_symbols(hcst, j, i,
                        &matched_syms, &matched_types))
            {
                if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
                {
                    fprintf(TFile, "CUDA symbol: (%u, %u)\n", j, i);
                }
                is_cuda_symbol[i] = TRUE;
                break;
            }
        }
    }

    // Now set the attribute TY_IS_CUDA_RUNTIME.
    for (UINT32 i = 1; i < tt_sz; ++i)
    {
        if (is_cuda_type[i]) Set_TY_is_cuda_runtime(Ty_tab.Entry(i));
    }

    // Now set the CUDA attribute.
    for (UINT32 i = 1; i < st_sz; ++i)
    {
        if (is_cuda_symbol[i])
        {
            set_st_attr_is_cuda_runtime(make_ST_IDX(i, GLOBAL_SYMTAB));
        }
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        UINT32 n_matched_types = 0, n_matched_syms = 0;
        HC_INDEX_PAIR *idxp;
        for (idxp = matched_types; idxp != NULL; idxp = idxp->next)
        {
            ++n_matched_types;
        }
        for (idxp = matched_syms; idxp != NULL; idxp = idxp->next)
        {
            ++n_matched_syms;
        }

        fprintf(TFile, "%d matched types, %d matched symbols\n",
                n_matched_types, n_matched_syms);
    }

    // Clean up.
    HC_free_index_pair_list(matched_syms);
    HC_free_index_pair_list(matched_types);
}

/*** DAVID CODE END ***/

