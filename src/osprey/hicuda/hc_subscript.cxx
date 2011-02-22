/** DAVID CODE BEGIN **/

#include <assert.h>
#include <set>

#include "wn.h"
#include "wn_util.h"

#include "hc_subscript.h"
#include "hc_stack.h"
#include "hc_utils.h"

// use to hold error messages (delayed abort)
char hc_subscript_errmsg[256];

// TODO: create a convenience function write_errmsg

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

struct subscript_context ssinfo = {
    false,
    NULL, 0, NULL
};

void
reset_subscript_context() {
    ssinfo.valid = false;
    ssinfo.var_ldids = NULL;
    ssinfo.nvars = 0;
    ssinfo.var_coeffs = NULL;
}

typedef std::set<WN*> WN_SET;

/**
 * Add all distinct variable LDID's in 'expr' into the given list.
 */
static void
find_var_ldids(WN *expr, WN_SET *var_ldids) {
    assert(expr != NULL && var_ldids != NULL);

    OPERATOR opr = WN_operator(expr);
    assert(OPERATOR_is_expression(opr));

    if (opr == OPR_LDID) {
        // Add this LDID to the list.
        WN_SET::iterator it = var_ldids->begin();
        while (it != var_ldids->end()) {
            if (expr == *it || HCWN_are_ldids_equal(expr, *it)) return;
            it++;
        }

        var_ldids->insert(WN_COPY_Tree(expr));
    }

    // Check the kids.
    INT nkids = WN_kid_count(expr);
    for (INT i = 0; i < nkids; ++i) {
        WN *kid = WN_kid(expr,i);
        if (kid != NULL) find_var_ldids(kid, var_ldids);
    }
}

/* We can go through the expression tree once to determine if each WN node
 * is constant or not, using a stack and a map to cache the results. Then
 * we do not need to call 'is_constant_term' so many times.
 */

/**
 * Determine if the given expression contains references to variables in
 * 'sinfo.var_ldids'. Return true if no and false otherwise.
 */
static bool
is_constant_term(WN *expr) {
    assert(expr != NULL && ssinfo.valid);

    OPERATOR opr = WN_operator(expr);
    assert(OPERATOR_is_expression(opr));

    if (opr == OPR_LDID) {
        for (UINT i = 0; i < ssinfo.nvars; ++i) {
            if (HCWN_are_ldids_equal(expr, ssinfo.var_ldids[i])) return false;
        }
    }

    // Check the kids.
    INT nkids = WN_kid_count(expr);
    for (INT i = 0; i < nkids; ++i) {
        WN *kid = WN_kid(expr,i);
        if (kid != NULL && !is_constant_term(kid)) return false;
    }

    return true;
}

/**
 * Recursively break down the subscript into linear components of
 * loop index variables.
 *
 * This function can definitely be improved for higher efficiency,
 * but most subscripts are simple so we don't bother.
 */
static bool
analyze_subscript_core(WN *expr, WN *coeff) {
    if (is_constant_term(expr)) {
        // Add it to the constant term.
        WN *const_term = ssinfo.var_coeffs[ssinfo.nvars];
        WN *term = WN_Mpy(WN_rtype(expr),
            WN_COPY_Tree(coeff), WN_COPY_Tree(expr));
        const_term = WN_Add(WN_rtype(const_term), const_term, term);
        ssinfo.var_coeffs[ssinfo.nvars] = const_term;

        return true;
    }

    bool successful;

    OPERATOR opr = WN_operator(expr);
    switch (opr) {
        case OPR_ADD: {
            return analyze_subscript_core(WN_kid0(expr), coeff)
                && analyze_subscript_core(WN_kid1(expr), coeff);
        }

        case OPR_SUB: {
            successful = analyze_subscript_core(WN_kid0(expr), coeff);
            if (!successful) return false;

            WN *neg_coeff = WN_Neg(WN_rtype(coeff), WN_COPY_Tree(coeff));
            successful = analyze_subscript_core(WN_kid1(expr), neg_coeff);
            WN_DELETE_Tree(neg_coeff);

            return successful;
        }

        case OPR_MPY: {
            if (is_constant_term(WN_kid0(expr))) {
                WN *new_coeff = WN_Mpy(WN_rtype(coeff),
                    WN_COPY_Tree(WN_kid0(expr)), WN_COPY_Tree(coeff));
                successful = analyze_subscript_core(WN_kid1(expr), new_coeff);
                WN_DELETE_Tree(new_coeff);
                return successful;
            }
            if (is_constant_term(WN_kid1(expr))) {
                WN *new_coeff = WN_Mpy(WN_rtype(coeff),
                    WN_COPY_Tree(WN_kid1(expr)), WN_COPY_Tree(coeff));
                successful = analyze_subscript_core(WN_kid0(expr), new_coeff);
                WN_DELETE_Tree(new_coeff);
                return successful;
            }
            return false;
        }

        case OPR_LDID: {
            // Find out which variable in the list it loads. There must be one
            // because otherwise it would have been a constant term.
            UINT idx = 0;
            while (idx < ssinfo.nvars
                && !HCWN_are_ldids_equal(expr, ssinfo.var_ldids[idx])) idx++;
            assert(idx < ssinfo.nvars);

            // Add it to the corresponding coefficient.
            WN *wn = ssinfo.var_coeffs[idx];
            wn = WN_Add(WN_rtype(wn), wn, WN_COPY_Tree(coeff));
            ssinfo.var_coeffs[idx] = wn;

            return true;
        }

        default:
            break;
    }

    return false;
}

bool
analyze_subscript(WN *expr) {
    assert(expr != NULL && ssinfo.valid);

    /* The efficiency could be improved in this case. */
    if (ssinfo.var_ldids == NULL) {
        assert(ssinfo.nvars == 0);

        // Go through the given expression to find all variables it accesses.
        WN_SET var_ldids;
        find_var_ldids(expr, &var_ldids);

        // Turn the list into an array.
        ssinfo.nvars = var_ldids.size();
        ssinfo.var_ldids = (WN**)malloc(ssinfo.nvars*sizeof(WN*));
        UINT i = 0;
        WN_SET::iterator it = var_ldids.begin();
        while (it != var_ldids.end()) {
            ssinfo.var_ldids[i++] = (*it++);
        }
        var_ldids.clear();

        ssinfo.var_coeffs = (WN**)malloc((ssinfo.nvars+1)*sizeof(WN*));
    }

    // Init each coefficient to be 0.
    for (UINT i = 0; i <= ssinfo.nvars; ++i) {
        ssinfo.var_coeffs[i] = WN_Intconst(Integer_type, 0);
    }

    // initial context coefficient
    WN *coeff = WN_Intconst(Integer_type, 1);

    // Call the recusive function.
    bool successful = analyze_subscript_core(expr, coeff);
    if (!successful) {
        for (UINT i = 0; i <= ssinfo.nvars; ++i) {
            WN_DELETE_Tree(ssinfo.var_coeffs[i]);
            ssinfo.var_coeffs[i] = NULL;
        }
    }

    WN_DELETE_Tree(coeff);

    return successful;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

WN*
idx_range_size(WN *hc_triplet) {
    // kid1 is the stride.
    WN *start_idx_wn = WN_kid0(hc_triplet);
    WN *end_idx_wn = WN_kid2(hc_triplet);
    assert(start_idx_wn != NULL && end_idx_wn != NULL);

    /* First, try algebraic simplification. */

    ssinfo.valid = true;
    
    // Analyze the start index.
    WN *size_wn = NULL;
    if (analyze_subscript(start_idx_wn)) {
        // Now, we have the list of variables found and their coefficients.
        // Save the coefficients.
        WN* start_coeffs[ssinfo.nvars+1];
        for (UINT i = 0; i <= ssinfo.nvars; ++i) {
            start_coeffs[i] = ssinfo.var_coeffs[i];
        }

        // Analyze the end index.
        if (analyze_subscript(end_idx_wn)) {
            // We can figure out the size by subtracting each component.
            WN *const_term = WN_Sub(Integer_type,
                ssinfo.var_coeffs[ssinfo.nvars], start_coeffs[ssinfo.nvars]);
            size_wn = WN_Add(Integer_type,
                const_term, WN_Intconst(Integer_type, 1));
            for (UINT i = 0; i < ssinfo.nvars; ++i) {
                size_wn = WN_Add(Integer_type,
                    size_wn,
                    WN_Mpy(WN_rtype(ssinfo.var_ldids[i]),
                        WN_Sub(WN_rtype(ssinfo.var_ldids[i]),
                            ssinfo.var_coeffs[i],
                            start_coeffs[i]
                        ),
                        WN_COPY_Tree(ssinfo.var_ldids[i])
                    )
                );
            }
        } else {
            // Deallocate the coefficients for start index.
            for (UINT i = 0; i <= ssinfo.nvars; ++i) {
                WN_DELETE_Tree(start_coeffs[i]);
            }
        }
    }

    // Clean up.
    for (UINT i = 0; i < ssinfo.nvars; ++i) {
        WN_DELETE_Tree(ssinfo.var_ldids[i]);
    }
    free(ssinfo.var_ldids);
    free(ssinfo.var_coeffs);
    reset_subscript_context();

    if (size_wn != NULL) return size_wn;

    /* Now, fall back to the basic calcuation method. */

    if (WN_operator(start_idx_wn) == OPR_INTCONST) {
        if (WN_operator(end_idx_wn) == OPR_INTCONST) {
            return WN_Intconst(Integer_type,
                WN_const_val(end_idx_wn) - WN_const_val(start_idx_wn) + 1);
        } else {
            // end_idx - (start_idx-1)
            return WN_Sub(Integer_type,
                WN_COPY_Tree(end_idx_wn),
                WN_Intconst(Integer_type, WN_const_val(start_idx_wn)-1)
            );
        }
    } else {
        if (WN_operator(end_idx_wn) == OPR_INTCONST) {
            // (end_idx+1) - start_idx
            return WN_Sub(Integer_type,
                WN_Intconst(Integer_type, WN_const_val(end_idx_wn)+1),
                WN_COPY_Tree(start_idx_wn)
            );
        } else {
            return WN_Add(Integer_type,
                WN_Sub(Integer_type,
                    WN_COPY_Tree(end_idx_wn),
                    WN_COPY_Tree(start_idx_wn)
                ),
                WN_Intconst(Integer_type, 1)
            );
        }
    }
}

/**
 * Determine if the given variable is non-negative, to our best knowledge.
 */
static TRISTATE_BOOL
is_var_non_negative(WN *var_ldid,
        struct loop_part_info **loops = NULL, UINT nloops = 0) {
    // Check the machine type of the LDID.
    if (MTYPE_is_unsigned(TY_mtype(WN_rtype(var_ldid)))) return TB_TRUE;

    ST_IDX var_st_idx = WN_st_idx(var_ldid);

    // If it is LDID of a scalar, check the symbol's type.
    if (WN_load_offset(var_ldid) == 0
        && MTYPE_is_unsigned(TY_mtype(ST_type(var_st_idx)))) {
        return TB_TRUE;
    }

    if (loops == NULL || nloops == 0) return TB_NOTSURE;

    // The loop bounds can give us more information.
    for (UINT i = 0; i < nloops; ++i) {
        if (loops[i]->idxv_st_idx == var_st_idx) {
            // TODO: the loop info struct should provide bound info.
        }
    }

    return TB_NOTSURE;
}

TBOOL
are_subscripts_equal(WN *ss1, WN *ss2) {
    TBOOL result = TB_NOTSURE;

    /* Normalize both subscripts. */

    ssinfo.valid = true;

    // Analyze 'ss1'.
    if (analyze_subscript(ss1)) {
        /* Now, we have found the list of variables and their coefficients
         * in 'ss1'. We will reuse this context.
         */

        // Save the coefficients.
        WN* ss1_coeffs[ssinfo.nvars+1];
        for (UINT i = 0; i <= ssinfo.nvars; ++i) {
            ss1_coeffs[i] = ssinfo.var_coeffs[i];
        }

        // Analyze 'ss2'.
        if (analyze_subscript(ss2)) {
            result = TB_TRUE;
            for (UINT i = 0; i <= ssinfo.nvars; ++i) {
                WN *c1 = ss1_coeffs[i], *c2 = ssinfo.var_coeffs[i];

                // We can't do comparison if they are not constants.
                // TODO: handle floating-point constants
                if (WN_operator(c1) != OPR_INTCONST
                    || WN_operator(c2) != OPR_INTCONST) {
                    result = TB_NOTSURE;
                    break;
                }

                if (WN_const_val(c1) != WN_const_val(c2)) {
                    result = TB_FALSE;
                    break;
                }
            }
        } else {
            // Deallocate the coefficients for start index.
            for (UINT i = 0; i <= ssinfo.nvars; ++i) {
                WN_DELETE_Tree(ss1_coeffs[i]);
            }
        }
    }

    // Clean up.
    for (UINT i = 0; i < ssinfo.nvars; ++i) {
        WN_DELETE_Tree(ssinfo.var_ldids[i]);
    }
    free(ssinfo.var_ldids);
    free(ssinfo.var_coeffs);
    reset_subscript_context();

    return result;
}

TBOOL
compare_subscripts(WN *ss1, WN *ss2,
        struct loop_part_info **loops, UINT nloops) {
    TBOOL result = TB_NOTSURE;

    /* Normalize both subscripts. */

    ssinfo.valid = true;

    // Analyze 'ss1'.
    if (analyze_subscript(ss1)) {
        /* Now, we have found the list of variables and their coefficients
         * in 'ss1'. We will reuse this context.
         */

        // Save the coefficients.
        WN* ss1_coeffs[ssinfo.nvars+1];
        for (UINT i = 0; i <= ssinfo.nvars; ++i) {
            ss1_coeffs[i] = ssinfo.var_coeffs[i];
        }

        // Analyze 'ss2'.
        if (analyze_subscript(ss2)) {
            // ss1 <= ss2 if each coefficient of ss1 <= that of ss2.
            bool all_le = true, all_gt = true;
            UINT i;
            for (i = 0; i <= ssinfo.nvars; ++i) {
                WN *c1 = ss1_coeffs[i], *c2 = ssinfo.var_coeffs[i];

                // We can't do comparison if they are not constants.
                // TODO: handle floating-point constants
                if (WN_operator(c1) != OPR_INTCONST
                    || WN_operator(c2) != OPR_INTCONST) break;

                if (WN_const_val(c1) == WN_const_val(c2)) {
                    all_gt = false;
                } else {
                    // Need to check the variable sign.
                    TBOOL is_non_negative = (i == ssinfo.nvars) ? TB_TRUE :
                        is_var_non_negative(ssinfo.var_ldids[i], loops, nloops);
                    if (is_non_negative == TB_NOTSURE) break;

                    if (WN_const_val(c1) < WN_const_val(c2)) {
                        if (is_non_negative == TB_TRUE) {
                            all_gt = false;
                        } else {
                            all_le = false;
                        }
                    } else {
                        if (is_non_negative == TB_TRUE) {
                            all_le = false;
                        } else {
                            all_gt = false;
                        }
                    }
                }
            }

            if (i > ssinfo.nvars) {
                if (all_le) result = TB_TRUE;
                if (all_gt) result = TB_FALSE;
            }
        } else {
            // Deallocate the coefficients for start index.
            for (UINT i = 0; i <= ssinfo.nvars; ++i) {
                WN_DELETE_Tree(ss1_coeffs[i]);
            }
        }
    }

    // Clean up.
    for (UINT i = 0; i < ssinfo.nvars; ++i) {
        WN_DELETE_Tree(ssinfo.var_ldids[i]);
    }
    free(ssinfo.var_ldids);
    free(ssinfo.var_coeffs);
    reset_subscript_context();

    return result;
}

TBOOL
is_triplet_covered(WN *t1, WN *t2,
        struct loop_part_info **loops, UINT nloops) {
    assert(WN_operator(t1) == OPR_TRIPLET && WN_operator(t2) == OPR_TRIPLET);

    WN *stride1 = WN_kid1(t1), *stride2 = WN_kid1(t2);
    assert(WN_operator(stride1) == OPR_INTCONST && WN_const_val(stride1) >= 1);
    assert(WN_operator(stride2) == OPR_INTCONST && WN_const_val(stride2) >= 1);

    // For now, we only worry about unit stride.
    if (WN_const_val(stride1) != 1
        || WN_const_val(stride2) != 1) return TB_NOTSURE;

    // t1's starting index <= t2's starting index
    TRISTATE_BOOL tb_start = compare_subscripts(WN_kid0(t1), WN_kid0(t2),
        loops, nloops);
    if (tb_start != TB_TRUE) return tb_start;

    // t1's ending index >= t2's ending index
    TRISTATE_BOOL tb_end = compare_subscripts(WN_kid2(t2), WN_kid2(t1),
        loops, nloops);
    if (tb_end != TB_TRUE) return tb_end;

    return TB_TRUE;
}

TBOOL
is_region_covered(WN *r1, WN *r2,
        struct loop_part_info **loops, UINT nloops) {
    assert(WN_operator(r1) == OPR_ARRSECTION
        && WN_operator(r2) == OPR_ARRSECTION);

    UINT ndims = WN_num_dim(r1);
    assert(ndims == (UINT)WN_num_dim(r2));

    // Go through each dimension.
    for (UINT i = 0; i < ndims; ++i) {
        TRISTATE_BOOL tb = is_triplet_covered(
            WN_kid(r1,ndims+i+1), WN_kid(r2,ndims+i+1), loops, nloops);
        if (tb != TB_TRUE) return tb;
    }

    return TB_TRUE;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

WN*
project_region(WN *arr_section,
        ST_IDX *loop_idxvs, WN **loop_ranges, UINT nloops) {
    /* Normalize start/end index of each dimension. */

    WN* idxv_ldids[nloops];
    for (UINT i = 0; i < nloops; ++i) {
        idxv_ldids[i] = WN_LdidScalar(loop_idxvs[i]);
    }

    UINT ndims = (UINT)WN_num_dim(arr_section);
    UINT nsubs = (ndims << 1);      // number of subscripts
    WN *norm_subscripts[nsubs][nloops+1];

    ssinfo.var_ldids = idxv_ldids;
    ssinfo.nvars = nloops;
    ssinfo.valid = true;

    UINT ss_idx = 0;
    for (UINT i = 0; i < ndims; ++i) {
        WN *triplet = WN_kid(arr_section,ndims+i+1);

        // start index
        ssinfo.var_coeffs = &(norm_subscripts[ss_idx++][0]);
        if (!analyze_subscript(WN_kid0(triplet))) {
            snprintf(hc_subscript_errmsg, sizeof(hc_subscript_errmsg),
                "The starting index of dimension %u is too messy!", i);
            break;
        }

        // end index
        ssinfo.var_coeffs = &(norm_subscripts[ss_idx++][0]);
        if (!analyze_subscript(WN_kid2(triplet))) {
            snprintf(hc_subscript_errmsg, sizeof(hc_subscript_errmsg),
                "The ending index of dimension %u is too messy!", i);
            break;
        }
    }

    // Clean up.
    for (UINT i = 0; i < nloops; ++i) WN_DELETE_Tree(idxv_ldids[i]);
    reset_subscript_context();

    // Error handling
    if (ss_idx < (nsubs)) {
        for (UINT i = 0; i < ss_idx; ++i) {
            for (UINT j = 0; j <= nloops; ++j) {
                WN_DELETE_Tree(norm_subscripts[i][j]);
            }
        }
        return NULL;
    }

    /* Do region projection w.r.t. each loop index variable.
     * The coefficients are replaced with an expression that multiples
     * the index variable (or its min/max bound).
     */

    UINT idxv_idx;
    for (idxv_idx = 0; idxv_idx < nloops; ++idxv_idx) {
        /* Find out what subscripts reference this variable. Subscripts
         * from different dimensions cannot reference the same variable.
         */
        for (ss_idx = 0; ss_idx < nsubs; ++ss_idx) {
            WN *coeff = norm_subscripts[ss_idx][idxv_idx];
            assert(coeff != NULL);

            if (WN_operator(coeff) != OPR_INTCONST
                || WN_const_val(coeff) != 0) break;
        }
        if (ss_idx == nsubs) continue;

        // Save the first occurrence and continue searching.
        UINT ss_idx_1 = ss_idx++;
        if (ss_idx_1 % 2 == 0) {
            // This is a start index, we allow the end index to reference
            // the variable too.
            ss_idx++;
        }

        // The remaining subscripts must not reference the variable.
        for ( ; ss_idx < nsubs; ++ss_idx) {
            WN *coeff = norm_subscripts[ss_idx][idxv_idx];
            assert(coeff != NULL);

            if (WN_operator(coeff) != OPR_INTCONST
                || WN_const_val(coeff) != 0) break;
        }
        // Meet coupled subscripts.
        if (ss_idx < nsubs) {
            snprintf(hc_subscript_errmsg, sizeof(hc_subscript_errmsg),
                "Variable %s is used in multiple dimensions.",
                ST_name(loop_idxvs[idxv_idx]));
            break;
        }

        /* Given an index range that references the variable and the
         * variable's range, we can eliminate this variable by doing
         * projection.
         *
         * (a*i, b*i) => (a*min(i), b*max(i))
         *
         * We must know the sign of a and b.
         */

        // For now, the range of the index variable must have a unit stride,
        // otherwise we quit.
        WN *range = loop_ranges[idxv_idx];
        WN *stride = WN_kid1(range);
        assert(stride != NULL);
        // We do not support non-unit stride for block range.
        if (WN_operator(stride) != OPR_INTCONST || WN_const_val(stride) != 1) {
            snprintf(hc_subscript_errmsg, sizeof(hc_subscript_errmsg),
                "Non-unit stride in block range of loop %s",
                ST_name(loop_idxvs[idxv_idx]));
             break;
        }

        // Get the min and max expression of the loop index variable.
        WN *idxv_min = WN_kid0(range), *idxv_max = WN_kid2(range);
        assert(idxv_min != NULL && idxv_max != NULL);

        WN *coeff = NULL;
        if (ss_idx_1 % 2 == 0) {
            // This is a start index, we want to minimize it.
            coeff = norm_subscripts[ss_idx_1][idxv_idx];
            assert(coeff != NULL);
            if (WN_operator(coeff) != OPR_INTCONST) break;
            norm_subscripts[ss_idx_1][idxv_idx] = WN_Mpy(WN_rtype(coeff),
                coeff,
                WN_COPY_Tree((WN_const_val(coeff) > 0) ? idxv_min: idxv_max)
            );

            ss_idx_1++;
        }

        // Do projection for the end index.
        coeff = norm_subscripts[ss_idx_1][idxv_idx];
        assert(coeff != NULL);
        if (WN_operator(coeff) != OPR_INTCONST) break;
        norm_subscripts[ss_idx_1][idxv_idx] = WN_Mpy(WN_rtype(coeff),
            coeff,
            WN_COPY_Tree((WN_const_val(coeff) > 0) ? idxv_max: idxv_min)
        );
    }
    // Error handling
    if (idxv_idx < nloops) {
        for (UINT i = 0; i < nsubs; ++i) {
            for (UINT j = 0; j <= nloops; ++j) {
                WN_DELETE_Tree(norm_subscripts[i][j]);
            }
        }
        return NULL;
    }

    /* Construct a new ARRSECTION node that contains the merged section.
     * We simply add the "coefficients" together for each subscript.
     */
    WN *merged_arr_section = WN_CreateArrSection(
        WN_COPY_Tree(WN_kid0(arr_section)), ndims);

    ss_idx = 0;
    for (UINT i = 0; i < ndims; ++i) {
        // Determine the start idx.
        WN *start_idx = WN_Intconst(Integer_type, 0);
        for (UINT j = 0; j <= nloops; ++j) {
            WN *comp = norm_subscripts[ss_idx][j];
            assert(comp != NULL);
            start_idx = WN_Add(Integer_type, start_idx, comp);
            norm_subscripts[ss_idx][j] = NULL;
        }
        ss_idx++;

        // Determine the end idx.
        WN *end_idx = WN_Intconst(Integer_type, 0);
        for (UINT j = 0; j <= nloops; ++j) {
            WN *comp = norm_subscripts[ss_idx][j];
            assert(comp != NULL);
            end_idx = WN_Add(Integer_type, end_idx, comp);
            norm_subscripts[ss_idx][j] = NULL;
        }
        ss_idx++;

        WN *triplet = WN_CreateTriplet(
            start_idx, WN_Intconst(Integer_type, 1), end_idx);

        WN_kid(merged_arr_section,ndims+i+1) = triplet;

        // Determine the dimension size.
        WN_kid(merged_arr_section,i+1) = idx_range_size(triplet);
    }

    return merged_arr_section;
}

/*** DAVID CODE END ***/
