/** DAVID CODE BEGIN **/

#include "defs.h"
#include "tracing.h"            // for TDEBUG_HICUDA
#include "wn.h"
#include "wn_util.h"
#include "ir_reader.h"

#include "hc_common.h"
#include "hc_kernel.h"
#include "hc_utils.h"
#include "hc_expr.h"
#include "cuda_utils.h"

#include "ipa_hc_kernel.h"
#include "ipa_hc_misc.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

HC_DOLOOP_INFO::HC_DOLOOP_INFO(const WN *doloop_wn)
{
    Is_True(doloop_wn != NULL && WN_opcode(doloop_wn) == OPC_DO_LOOP, (""));

    /* Gather information and check for legality of normalization. */

    // Get the loop index variable.
    _idxv_st_idx = WN_st_idx(WN_kid0(doloop_wn));

    // init value
    _init_wn = WN_COPY_Tree(WN_kid0(WN_kid1(doloop_wn)));

    // step value
    WN *wn = WN_kid0(WN_kid(doloop_wn,3));
    Is_True(WN_operator(wn) == OPR_ADD, (""));
    WN *step_wn = WN_kid1(wn);
    if (! is_ldid_scalar(WN_kid0(wn), _idxv_st_idx))
    {
        Is_True(is_ldid_scalar(WN_kid1(wn), _idxv_st_idx), (""));
        step_wn = WN_kid0(wn);
    }
    Is_True(WN_operator(step_wn) == OPR_INTCONST,
            ("The step of loop %s is not a compiler-known constant!\n",
             ST_name(_idxv_st_idx)));
    _step = WN_const_val(step_wn);
    Is_True(_step != 0,
            ("The step of loop %s is 0!\n", ST_name(_idxv_st_idx)));

    // ending value
    wn = WN_kid2(doloop_wn);
    OPERATOR opr = WN_operator(wn);
    _end_wn = WN_kid1(wn);
    if (! is_ldid_scalar(WN_kid0(wn), _idxv_st_idx))
    {
        Is_True(is_ldid_scalar(WN_kid1(wn), _idxv_st_idx), (""));
        // Reverse the comparison operator.
        opr = reverse_comp(opr);
        _end_wn = WN_kid0(wn);
    }
    _end_wn = WN_COPY_Tree(_end_wn);

    // Normalize <op> to be LE to GE.
    if (opr == OPR_GT)
    {
        opr = OPR_GE;
        _end_wn = WN_Add(WN_rtype(_end_wn),
                _end_wn, WN_Intconst(WN_rtype(_end_wn), 1));
    }
    else if (opr == OPR_LT)
    {
        opr = OPR_LE;
        _end_wn = WN_Sub(WN_rtype(_end_wn),
                _end_wn, WN_Intconst(WN_rtype(_end_wn), 1));
    }

    // <op> must be LE if <_step> is +ive, and GE otherwise.
    Is_True((_step > 0 && opr == OPR_LE) || (_step < 0 && opr == OPR_GE),
            ("Loop %s is potentially infinite!\n", ST_name(_idxv_st_idx)));

    // Check if <_step> and (<_end_wn> - <_init_wn>) have the same sign.
    // If not, the loop is empty.
    wn = WN_Sub(Integer_type, WN_COPY_Tree(_end_wn), WN_COPY_Tree(_init_wn));
    HCWN_simplify_expr(&wn);
    if (WN_operator(wn) != OPR_INTCONST)
    {
        _empty_loop = 2;
    }
    // TODO: smarter check?
    else if ((_step > 0 && WN_const_val(wn) < 0)
            || (_step < 0 && WN_const_val(wn) > 0))
    {
        _empty_loop = 1;
    }
    else
    {
        _empty_loop = 0;
    }
}

HC_DOLOOP_INFO::~HC_DOLOOP_INFO()
{
    Is_True(_init_wn != NULL, (""));
    WN_DELETE_Tree(_init_wn); _init_wn = NULL;
    Is_True(_end_wn != NULL, (""));
    WN_DELETE_Tree(_end_wn); _end_wn = NULL;
}

WN* HC_DOLOOP_INFO::get_tripcount_expr() const
{
    if (_empty_loop == 2) return NULL;

    if (_empty_loop == 1) return WN_Intconst(Integer_type, 0);

    // tripcount = floor( (<end> - <init>) / <step> ) + 1
    // NOTE: <end> is inclusive.

    // <end> - <init> is the target for simplification.
    WN *range_wn = WN_Sub(Integer_type,
            WN_COPY_Tree(_end_wn), WN_COPY_Tree(_init_wn));
    HCWN_simplify_expr(&range_wn);

    return WN_Add(Integer_type,
            WN_Div(Integer_type, range_wn, WN_Intconst(Integer_type, _step)),
            WN_Intconst(Integer_type, 1));
}

WN* HC_DOLOOP_INFO::gen_tripcount(ST_IDX st_idx) const
{
    Is_True(ST_IDX_level(st_idx) > GLOBAL_SYMTAB, (""));

    ST *st = ST_ptr(st_idx);
    Is_True(TY_mtype(ST_type(st)) == Integer_type, (""));

    WN *blk_wn = WN_CreateBlock();

    if (_empty_loop != 0)
    {
        // Generate the empty loop expression and insert it to the block.
        WN_INSERT_BlockFirst(blk_wn,
                WN_StidScalar(st, WN_Intconst(Integer_type, 0)));
    }
    if (_empty_loop != 1)
    {
        if (_step > 0)
        {
            // Generate the non-empty loop expression.
            WN *nonempty_loop_wn = WN_StidScalar(st,
                    WN_Add(Integer_type,
                        WN_Div(Integer_type,
                            WN_Sub(Integer_type,
                                WN_COPY_Tree(_end_wn),
                                WN_COPY_Tree(_init_wn)),
                            WN_Intconst(Integer_type, _step)),
                        WN_Intconst(Integer_type, 1)));

            if (_empty_loop == 0)
            {
                // non-empty loop
                WN_INSERT_BlockLast(blk_wn, nonempty_loop_wn);
            }
            else if (_empty_loop == 2)
            {
                // Generate a guard.
                WN *guard_wn = WN_GE(Integer_type,
                        WN_COPY_Tree(_end_wn), WN_COPY_Tree(_init_wn));
                WN *if_blk_wn = WN_CreateBlock();
                WN_INSERT_BlockLast(if_blk_wn, nonempty_loop_wn);
                WN_INSERT_BlockLast(blk_wn,
                        WN_CreateIf(guard_wn, if_blk_wn, WN_CreateBlock()));
            }
        }
        else
        {
            // Generate the non-empty loop expression.
            WN *nonempty_loop_wn = WN_StidScalar(st,
                    WN_Add(Integer_type,
                        WN_Div(Integer_type,
                            WN_Sub(Integer_type,
                                WN_COPY_Tree(_init_wn),
                                WN_COPY_Tree(_end_wn)),
                            WN_Intconst(Integer_type, -_step)),
                        WN_Intconst(Integer_type, 1)));

            if (_empty_loop == 0)
            {
                // non-empty loop
                WN_INSERT_BlockLast(blk_wn, nonempty_loop_wn);
            }
            else if (_empty_loop == 2)
            {
                // Generate a guard.
                WN *guard_wn = WN_LE(Integer_type,
                        WN_COPY_Tree(_end_wn), WN_COPY_Tree(_init_wn));
                WN *if_blk_wn = WN_CreateBlock();
                WN_INSERT_BlockLast(if_blk_wn, nonempty_loop_wn);
                WN_INSERT_BlockLast(blk_wn,
                        WN_CreateIf(guard_wn, if_blk_wn, WN_CreateBlock()));
            }
        }
    }

    return blk_wn;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

BOOL HC_lower_loop_part_region(WN *region, HC_LOOP_PART_INFO *lpi,
        HC_LOCAL_VAR_STORE *lvar_store, MEM_POOL *pool)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile,
                "Start lowering a LOOP_PARTITION directive (%p) ...\n", lpi);
    }

    Is_True(region != NULL && WN_operator(region) == OPR_REGION, (""));

    // TODO: validate the pragma

    HC_LPI_DIST_TYPE blk_distr = lpi->get_block_clause();
    HC_LPI_DIST_TYPE thr_distr = lpi->get_thread_clause();
    HC_KERNEL_CONTEXT *kc = lpi->get_kernel_context();
    HC_KERNEL_INFO *kinfo = kc->get_kernel_info();

    // The thread distribution type must not be blocking.
    Is_True(thr_distr != HC_LPI_DT_BLOCK, (""));

    /* Analyze the loop we are about to partition. */

    // The loop must be the first node in the region's body.
    WN *parent_blk = WN_kid2(region);
    WN *loop = WN_first(parent_blk);
    Is_True(loop != NULL && WN_operator(loop) == OPR_DO_LOOP, (""));

    // Normalize the loop.
    HC_DOLOOP_INFO li(loop);

    // Get the loop index variable.
    ST_IDX idxv_st_idx = li.get_idx_var();
    lpi->set_idxv(idxv_st_idx);
    TYPE_ID mtype = TY_mtype(ST_type(idxv_st_idx));

    // We do not handle an empty loop.
    if (li.is_empty_loop() == 1)
    {
        HC_warn("Ignore the empty loop <%s>!", ST_name(idxv_st_idx));
        return TRUE;
    }

    // fields of the loop partition info
    WN *blk_range = NULL;

    /* The modified loop will be
     *     for (i = i_init; i <op> i_end; i += i_step)
     *
     * Handle the generation of <i_init>, <i_end> and <i_step> case by case.
     */

    UINT vgrid_dim_idx = kc->get_vgrid_dim_idx();
    UINT vblk_dim_idx = kc->get_vblk_dim_idx();

    ST_IDX init_st_idx = new_local_var(
        gen_var_str(idxv_st_idx, "_init"), MTYPE_To_TY(mtype));
    ST_IDX end_st_idx = new_local_var(
        gen_var_str(idxv_st_idx, "_end"), MTYPE_To_TY(mtype));
    ST_IDX step_st_idx = new_local_var(
        gen_var_str(idxv_st_idx, "_step"), MTYPE_To_TY(mtype));

    // For now, we only derive special properties if the directive has
    // over_thread clause and its over_tblock clause is not CYCLIC. This is
    // because we know that these properties are used to derive whether or not
    // a starting index of some dimension of a smem region is a multiple of
    // the warp size, and we only need to generate properties for those
    // directives with meaningful index ranges.
    //
    // In both cases we care about, the expression that has a special property
    // is i - <threadIdx> * <step> - <init>. However, when the over_tblock
    // clause is present, it should have an extra - <tblk_stride> term. Since
    // including this term in the expression will render this property useless
    // in the above-mentioned derivation, we simply check if there is perfect
    // distribution among threads. If so, <tblk_stride> is a multiple of
    // <num_threads> * <step>, which is the target factor, and therefore does
    // not need to be included in the expression. Otherwise, no special
    // property is generated.
    //
    // If we cannot derive the special property, we fall back to the regular
    // loop property (i.e. that depends solely on the step value).
    //
    BOOL gen_special_prop = FALSE;

    // Do we still keep the loop?
    BOOL keep_loop = TRUE;

    if (blk_distr == HC_LPI_DT_NONE)
    {
        /* over_thread only */
        Is_True(thr_distr != HC_LPI_DT_NONE, (""));

        // Determine the number of iterations assigned to each thread,
        // and check if the division is perfect.
        WN *thread_sz = NULL;
        bool perfect_thr_div = false;
        WN *tc_wn = li.get_tripcount_expr();
        if (tc_wn != NULL)
        {
            thread_sz = HCWN_Ceil(tc_wn, kinfo->get_vblk_dim(vblk_dim_idx),
                    &perfect_thr_div);
            WN_DELETE_Tree(tc_wn); tc_wn = NULL;
        }

        // Does every thread execute at most one iteration?
        BOOL to_opt = (thread_sz != NULL
                && WN_operator(thread_sz) == OPR_INTCONST
                && WN_const_val(thread_sz) == 1);
        WN_DELETE_Tree(thread_sz); thread_sz = NULL;

        // regular case:
        //
        // i_init = <init> + <thread_idx> * <step>;
        // i_end = <end>;
        // i_step = <num_threads> * <step>;
        //
        // optimized case:
        //
        // i = <init> + <thread_idx> * <step>;
        //

        // <init> + <thread_idx> * <step>
        WN *new_init_wn = WN_Add(mtype,
                WN_COPY_Tree(li.get_init_expr()),
                WN_Mpy(Integer_type,
                    WN_COPY_Tree(kinfo->get_vthr_idx(vblk_dim_idx)),
                    WN_Intconst(Integer_type, li.get_step())));

        if (to_opt)
        {
            /* Replace the loop with a single index assignment. */
            keep_loop = FALSE;

            WN *body_blk = WN_kid(loop,4);
            WN_kid(loop,4) = NULL;

            // For non-perfect division, add a guard for the loop body.
            if (!perfect_thr_div)
            {
                // We use the existing end-condition check.
                WN *guard = WN_CreateIf(WN_COPY_Tree(WN_kid2(loop)),
                        body_blk, WN_CreateBlock());
                body_blk = WN_CreateBlock();
                WN_INSERT_BlockFirst(body_blk, guard);
            }

            // Insert the assignment at the beginning of the loop body.
            WN_INSERT_BlockFirst(body_blk,
                    WN_StidScalar(ST_ptr(idxv_st_idx), new_init_wn));

            // Insert the loop's body before the loop.
            WN_INSERT_BlockBefore(parent_blk, loop, body_blk);
            // Remove the current loop.
            WN_DELETE_FromBlock(parent_blk, loop);
        }
        else
        {
            // i_init
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(init_st_idx), new_init_wn));

            // i_end
            WN *new_end_wn = WN_COPY_Tree(li.get_end_expr());
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(end_st_idx), new_end_wn));

            // i_step
            WN *new_step_wn = WN_Mpy(mtype,
                    WN_COPY_Tree(kinfo->get_vblk_dim(vblk_dim_idx)),
                    WN_Intconst(Integer_type, li.get_step()));
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(step_st_idx), new_step_wn));
        }

        // TODO: as an optimization, if we could figure out the loop's
        // tripcount, we could unroll the loop automatically.

        gen_special_prop = TRUE;
    }
    else if (blk_distr == HC_LPI_DT_CYCLIC)
    {
        /* over_tblock(CYCLIC) [over_thread] */

        BOOL thr_distr_present = (thr_distr == HC_LPI_DT_CYCLIC);

        // Determine the number of iterations assigned to each thread/tblock,
        // and check if the division is perfect.
        WN *iter_count_wn = NULL;
        bool perfect_div = false;
        WN *tc_wn = li.get_tripcount_expr();
        if (tc_wn != NULL)
        {
            // number of tblocks or threads
            WN *tmp_wn = WN_COPY_Tree(kinfo->get_vgrid_dim(vgrid_dim_idx));
            if (thr_distr_present)
            {
                tmp_wn = WN_Mpy(Integer_type, tmp_wn,
                        WN_COPY_Tree(kinfo->get_vblk_dim(vblk_dim_idx)));
            }

            iter_count_wn = HCWN_Ceil(tc_wn, tmp_wn, &perfect_div);

            WN_DELETE_Tree(tmp_wn);
            WN_DELETE_Tree(tc_wn); tc_wn = NULL;
        }

        // Does every thread/tblock execute at most one iteration?
        BOOL to_opt = (iter_count_wn != NULL
                && WN_operator(iter_count_wn) == OPR_INTCONST
                && WN_const_val(iter_count_wn) == 1);
        WN_DELETE_Tree(iter_count_wn); iter_count_wn = NULL;

        // regular case:
        //
        // i_init = <init> + <idx> * <step>;
        // i_end = <end>;
        // i_step = <stride> * <step>
        //
        // if no over_thread
        //     <idx> = <tblock_idx>
        //     <stride> = <num_tblocks>
        // else
        //     <idx> = <tblock_idx> * <num_threads> + <thread_idx>
        //     <stride> = <num_tblocks> * <num_threads>
        // endif
        //
        // optimized case:
        //
        // i = <init> + <idx> * <step>;
        //

        // <idx> and <stride>
        WN *idx_wn = WN_COPY_Tree(kinfo->get_vblk_idx(vgrid_dim_idx));
        WN *stride_wn = WN_COPY_Tree(kinfo->get_vgrid_dim(vgrid_dim_idx));
        if (thr_distr == HC_LPI_DT_CYCLIC)
        {
            idx_wn = WN_Add(Integer_type,
                    WN_Mpy(Integer_type, idx_wn,
                        WN_COPY_Tree(kinfo->get_vblk_dim(vblk_dim_idx))),
                    WN_COPY_Tree(kinfo->get_vthr_idx(vblk_dim_idx)));
            stride_wn = WN_Mpy(Integer_type, stride_wn,
                    WN_COPY_Tree(kinfo->get_vblk_dim(vblk_dim_idx)));
        }

        // <init> + <idx> * <step>
        WN *new_init_wn = WN_Add(mtype,
                WN_COPY_Tree(li.get_init_expr()),
                WN_Mpy(Integer_type, idx_wn,
                    WN_Intconst(Integer_type, li.get_step())));

        if (to_opt)
        {
            /* Replace the loop with a single index assignment. */
            keep_loop = FALSE;

            WN *body_blk = WN_kid(loop,4);
            WN_kid(loop,4) = NULL;

            // For non-perfect division, add a guard for the loop body.
            if (!perfect_div)
            {
                // We use the existing end-condition check.
                WN *guard = WN_CreateIf(WN_COPY_Tree(WN_kid2(loop)),
                        body_blk, WN_CreateBlock());
                body_blk = WN_CreateBlock();
                WN_INSERT_BlockFirst(body_blk, guard);
            }

            // Insert the assignment at the beginning of the loop body.
            WN_INSERT_BlockFirst(body_blk,
                    WN_StidScalar(ST_ptr(idxv_st_idx), new_init_wn));

            // Insert the loop's body before the loop.
            WN_INSERT_BlockBefore(parent_blk, loop, body_blk);
            // Remove the current loop.
            WN_DELETE_FromBlock(parent_blk, loop);
        }
        else
        {
            // i_init
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(init_st_idx), new_init_wn));

            // i_end
            WN *new_end_wn = WN_COPY_Tree(li.get_end_expr());
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(end_st_idx), new_end_wn));

            // i_step
            WN *new_step_wn = WN_Mpy(mtype, stride_wn,
                    WN_Intconst(Integer_type, li.get_step()));
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(step_st_idx), new_step_wn));
        }

        // TODO: as an optimization, if we could figure out the loop's
        // tripcount, we could unroll the loop automatically.
    }
    else
    {
        /* over_tblock(BLOCK) [over_thread]
         *
         * tripcount = ...
         * tblk_stride = ceil(tripcount / <num_tblocks>) * <step>;
         * i_init = <init> + <tblock_idx> * tblk_stride;
         * i_end = i_init + tblock_stride - <step>;
         * // THIS LINE IS NECESSARY WHEN tripcount % <num_tblocks> != 0
         * if (! i_end <op> <end>) i_end = <end>;
         * i_step = <step>;
         *
         * if no over_thread
         *     <idx> = <tblock_idx> * tblk_stride
         *     <step_stride> = 1
         * else
         *     <idx> = <tblock_idx> * tblk_stride + <thread_idx> * <step>
         *     <step_stride> = <num_threads>
         * endif
         */

        BOOL thr_distr_present = (thr_distr == HC_LPI_DT_CYCLIC);

        // Determine the number of iterations assigned to each thread block,
        // and check if the division is perfect.
        WN *tblock_sz = NULL;
        bool perfect_tblk_div = false;
        WN *tc_wn = li.get_tripcount_expr();
        if (tc_wn != NULL)
        {
            tblock_sz = HCWN_Ceil(tc_wn, kinfo->get_vgrid_dim(vgrid_dim_idx),
                    &perfect_tblk_div);
            WN_DELETE_Tree(tc_wn); tc_wn = NULL;
        }

        // Determine the number of iterations assigned to each thread,
        // and check if the division is perfect.
        WN *thread_sz = NULL;
        bool perfect_thr_div = false;
        if (tblock_sz != NULL)
        {
            WN *n_threads_per_tblk_wn = thr_distr_present ?
                WN_COPY_Tree(kinfo->get_vblk_dim(vblk_dim_idx)) :
                WN_Intconst(Integer_type, 1);
            thread_sz = HCWN_Ceil(tblock_sz, n_threads_per_tblk_wn,
                    &perfect_thr_div);
            WN_DELETE_Tree(n_threads_per_tblk_wn);
        }
        // <tblock_sz> is still intact here.

        // Does every thread execute at most one iteration?
        BOOL to_opt = (thread_sz != NULL
                && WN_operator(thread_sz) == OPR_INTCONST
                && WN_const_val(thread_sz) == 1);
        WN_DELETE_Tree(thread_sz); thread_sz = NULL;

        if (to_opt)
        {
            /* Replace the loop with a single index assignment. */
            keep_loop = FALSE;

            WN *body_blk = WN_kid(loop,4);
            WN_kid(loop,4) = NULL;

            // For non-perfect division, add a guard for the loop body.
            if (! (perfect_tblk_div && perfect_thr_div))
            {
                // We use the existing end-condition check.
                WN *guard = WN_CreateIf(WN_COPY_Tree(WN_kid2(loop)),
                        body_blk, WN_CreateBlock());
                body_blk = WN_CreateBlock();
                WN_INSERT_BlockFirst(body_blk, guard);
            }

            // Insert a STID of the actual index variable at the beginning of
            // the loop body.
            WN *idx_wn = WN_Mpy(Integer_type, tblock_sz,
                    WN_COPY_Tree(kinfo->get_vblk_idx(vgrid_dim_idx)));
            tblock_sz = NULL;    // now it is consumed
            if (thr_distr_present)
            {
                idx_wn = WN_Add(mtype, idx_wn,
                        WN_COPY_Tree(kinfo->get_vthr_idx(vblk_dim_idx)));
            }
            WN *new_init_wn = WN_Add(mtype,
                    WN_COPY_Tree(li.get_init_expr()),
                    WN_Mpy(Integer_type, idx_wn,
                        WN_Intconst(Integer_type, li.get_step())));
            WN_INSERT_BlockFirst(body_blk,
                    WN_StidScalar(ST_ptr(idxv_st_idx), new_init_wn));

            // Insert the loop's body before the loop.
            WN_INSERT_BlockBefore(parent_blk, loop, body_blk);
            // Remove the current loop.
            WN_DELETE_FromBlock(parent_blk, loop);
        }
        else
        {
            // Compute the loop trip-count first.
            ST_IDX tc_st_idx = lvar_store->get_tripcount();
            WN_INSERT_BlockBefore(parent_blk, loop,
                    li.gen_tripcount(tc_st_idx));

            // Compute the tblock stride.
            ST_IDX tbstride_st_idx = lvar_store->get_tblock_stride();
            WN *tbstride_wn = WN_Mpy(Integer_type,
                    WN_Intconst(Integer_type, li.get_step()),
                    WN_Add(Integer_type, WN_Intconst(Integer_type, 1),
                        WN_Div(Integer_type,
                            WN_Sub(Integer_type,
                                WN_LdidScalar(tc_st_idx),
                                WN_Intconst(Integer_type, 1)),
                            WN_COPY_Tree(
                                kinfo->get_vgrid_dim(vgrid_dim_idx)))));
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(tbstride_st_idx), tbstride_wn));

            // i_init
            WN *new_init_wn = WN_Add(mtype,
                    WN_COPY_Tree(li.get_init_expr()),
                    WN_Mpy(Integer_type,
                        WN_COPY_Tree(kinfo->get_vblk_idx(vgrid_dim_idx)),
                        WN_LdidScalar(tbstride_st_idx)));
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(init_st_idx), new_init_wn));

            // i_end
            WN *new_end_wn = WN_Add(mtype, WN_LdidScalar(init_st_idx), 
                    WN_Sub(Integer_type, WN_LdidScalar(tbstride_st_idx),
                        WN_Intconst(Integer_type, li.get_step())));
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(end_st_idx), new_end_wn));

            // Guard for i_end
            if (! perfect_tblk_div)
            {
                WN *cond_wn = (li.get_step() > 0) ?
                    // <op> is LE, so ~<op> is GT.
                    WN_GT(Integer_type, WN_LdidScalar(end_st_idx),
                            WN_COPY_Tree(li.get_end_expr())) :
                    // <op> is GE, so ~<op> is LT.
                    WN_LT(Integer_type, WN_LdidScalar(end_st_idx),
                            WN_COPY_Tree(li.get_end_expr()));

                WN *body_wn = WN_CreateBlock();
                WN_INSERT_BlockFirst(body_wn,
                        WN_StidScalar(ST_ptr(end_st_idx),
                            WN_COPY_Tree(li.get_end_expr())));

                WN_INSERT_BlockBefore(parent_blk, loop,
                        WN_CreateIf(cond_wn, body_wn, WN_CreateBlock()));
            }

            // i_init = i_init + <thread_idx> * <step_expr>
            if (thr_distr_present)
            {
                WN *thr_ofst_wn = WN_Mpy(Integer_type,
                        WN_COPY_Tree(kinfo->get_vthr_idx(vblk_dim_idx)),
                        WN_Intconst(Integer_type, li.get_step()));

                WN_INSERT_BlockBefore(parent_blk, loop,
                        WN_StidScalar(ST_ptr(init_st_idx),
                            WN_Add(Integer_type,
                                WN_LdidScalar(init_st_idx), thr_ofst_wn)));
            }

            // i_step
            WN *new_step_wn = WN_Intconst(Integer_type, li.get_step());
            if (thr_distr_present)
            {
                new_step_wn = WN_Mpy(Integer_type, new_step_wn,
                        WN_COPY_Tree(kinfo->get_vblk_dim(vblk_dim_idx)));
            }
            WN_INSERT_BlockBefore(parent_blk, loop,
                    WN_StidScalar(ST_ptr(step_st_idx), new_step_wn));
        }

        // Clean up.
        if (tblock_sz != NULL) { WN_DELETE_Tree(tblock_sz); }

        gen_special_prop = (thr_distr_present && perfect_thr_div);
    }

    /* Modify the existing loop if necessary (i.e. no optimization). */

    if (keep_loop)
    {
        // i = i_init
        WN_DELETE_Tree(WN_kid1(loop));
        WN_kid1(loop) = WN_StidScalar(ST_ptr(idxv_st_idx),
                WN_LdidScalar(init_st_idx));

        // i <op> i_end
        WN_DELETE_Tree(WN_kid2(loop));
        WN_kid2(loop) = (li.get_step() > 0) ?
            WN_LE(Integer_type,
                    WN_LdidScalar(idxv_st_idx), WN_LdidScalar(end_st_idx)) :
            WN_GE(Integer_type,
                    WN_LdidScalar(idxv_st_idx), WN_LdidScalar(end_st_idx));

        // i += i_step
        WN_DELETE_Tree(WN_kid3(loop));
        WN_kid3(loop) = WN_StidScalar(ST_ptr(idxv_st_idx),
                WN_Add(mtype,
                    WN_LdidScalar(idxv_st_idx), WN_LdidScalar(step_st_idx)));
    }

    // Determine the range of the loop index variable, for directives with an
    // over_thread clause.
    //
    if (thr_distr != HC_LPI_DT_NONE)
    {
        WN *idxv_lbnd_wn = NULL, *idxv_ubnd_wn = NULL;

        if (blk_distr == HC_LPI_DT_CYCLIC)
        {
            // The index variable range is conservatively assumed to be the
            // entire iteration space.
            idxv_lbnd_wn = WN_COPY_Tree(li.get_init_expr());
            idxv_ubnd_wn = WN_COPY_Tree(li.get_end_expr());
        }
        else
        {
            // The index variable range is:
            //      [i - <thread_idx> * <step>,
            //       i - <thread_idx> * <step> + (<num_threads> - 1) * <step>]
            idxv_lbnd_wn = WN_Sub(Integer_type,
                    WN_LdidScalar(idxv_st_idx), 
                    WN_Mpy(Integer_type,
                        WN_COPY_Tree(kinfo->get_vthr_idx(vblk_dim_idx)),
                        WN_Intconst(Integer_type, li.get_step())));
            idxv_ubnd_wn = WN_Add(Integer_type,
                    WN_COPY_Tree(idxv_lbnd_wn),
                    WN_Mpy(Integer_type,
                        WN_Sub(Integer_type,
                            WN_COPY_Tree(kinfo->get_vblk_dim(vblk_dim_idx)),
                            WN_Intconst(Integer_type, 1)),
                        WN_Intconst(Integer_type, li.get_step())));
        }

        // Swap the upper and lower bound of the range when the step is -ive.
        if (li.get_step() > 0)
        {
            lpi->set_idxv_range(idxv_lbnd_wn, idxv_ubnd_wn);
        }
        else
        {
            lpi->set_idxv_range(idxv_ubnd_wn, idxv_lbnd_wn);
        }
    }

    INT factor = li.get_step();
    if (factor < 0) factor = -factor;
    if (gen_special_prop)
    {
        // The property can only be generated if the number of threads is a
        // constant.
        WN *n_threads_wn = kinfo->get_vblk_dim(vblk_dim_idx);
        gen_special_prop = (WN_operator(n_threads_wn) == OPR_INTCONST);
        factor *= WN_const_val(n_threads_wn);
    }

    if (factor > 1)
    {
        // Start with i - <init>
        WN *expr_wn = WN_Sub(Integer_type, WN_LdidScalar(idxv_st_idx),
                WN_COPY_Tree(li.get_init_expr()));
        if (gen_special_prop)
        {
            // i - <thread_idx> * <step> - <init>
            expr_wn = WN_Sub(Integer_type, expr_wn,
                    WN_Mpy(Integer_type,
                        WN_COPY_Tree(kinfo->get_vthr_idx(vblk_dim_idx)),
                        WN_Intconst(Integer_type, li.get_step())));
        }

        lpi->set_idxv_prop(CXX_NEW(HC_EXPR_PROP(expr_wn, factor), pool));
    }

    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the LOOP_PARTITION directive.\n");
    }

    return keep_loop;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

WN* HC_lower_barrier(WN *pragma_wn, WN *parent_wn, BOOL gen_code)
{
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Start lowering a BARRIER directive ...\n");
    }

    Is_True(parent_wn != NULL && WN_opcode(parent_wn) == OPC_BLOCK, (""));
    Is_True(pragma_wn != NULL && WN_opcode(pragma_wn) == OPC_PRAGMA, (""));

    // Get the next node to be processed.
    WN *next_wn = WN_next(pragma_wn);

    if (gen_code)
    {
        // Insert a call to __syncthreads before the pragma.
        WN_INSERT_BlockBefore(parent_wn, pragma_wn, call_syncthreads());
    }

    // Remove the pragma node.
    WN_DELETE_FromBlock(parent_wn, pragma_wn);
 
    if (Get_Trace(TKIND_DEBUG, TDEBUG_HICUDA))
    {
        fprintf(TFile, "Finished lowering the BARRIER directive.\n");
    }

    return next_wn;
}

/*** DAVID CODE END ***/
