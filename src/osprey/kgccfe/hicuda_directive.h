/** DAVID CODE BEGIN **/

#ifndef _GCCFE_HICUDA_DIRECIVES_H_
#define _GCCFE_HICUDA_DIRECIVES_H_

#ifdef _LANGUAGE_C_PLUS_PLUS
extern "C" {
#endif

extern struct idx_range_list* build_idx_range(tree start_idx, tree end_idx);
extern void chain_idx_range_list(
        struct idx_range_list *irl1, struct idx_range_list *irl2);

extern struct arr_region* build_arr_region(tree var,
        struct idx_range_list *irl);

extern struct copy_arr_region* build_copy_arr_region(struct arr_region *ar,
        bool nobndcheck);

/**
 * Link the two list 'ar1' and 'ar2' together (ar1 before ar2).
 * Both 'ar1' and 'ar2' can be NULL.
 * Return the new list's head.
 */
extern struct arr_region* chain_arr_region_list(
        struct arr_region *arl1, struct arr_region *arl2);

extern struct hicuda_global_shared* build_hicuda_global_shared(
        struct arr_region *decl, struct copy_arr_region *copy,
        bool clear_region);

extern struct hicuda_const* build_hicuda_const(struct arr_region *copyin);

extern struct free_data_list* build_free_data(tree var);
extern void chain_free_data_list(struct free_data_list *fdl1,
        struct free_data_list *fdl2);

extern struct dim_sz_list* build_dim_sz(tree dim_sz);
extern void chain_dim_sz_list(struct dim_sz_list *l1, struct dim_sz_list *l2);
extern struct hicuda_shape* build_hicuda_shape(tree var,
        struct dim_sz_list *l);
extern void chain_shape_list(struct hicuda_shape *sl1,
        struct hicuda_shape *sl2);

extern void expand_hc_global(struct hicuda_global_shared *dir);
extern void expand_hc_global_free(struct free_data_list *dir);
extern void expand_hc_shared(struct hicuda_global_shared *dir);
extern void expand_hc_shared_remove(struct free_data_list *dir);
extern void expand_hc_const(struct hicuda_const *dir);
extern void expand_hc_const_remove(struct free_data_list *dir);
extern void expand_hc_shape(struct hicuda_shape *dir);

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

extern void expand_hc_barrier();

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

extern struct virtual_space* build_virtual_space(tree dim_size);

/**
 * Chain 'vs2' at the end of 'vs1'.
 * Both 'vs1' and 'vs2' cannot be NULL.
 */
extern void chain_virtual_space(
    struct virtual_space *vs1, struct virtual_space *vs2);

extern struct hicuda_kernel* build_hicuda_kernel(tree kname,
    struct virtual_space *block, struct virtual_space *thread);

extern void expand_hc_kernel_begin(struct hicuda_kernel *dir);
extern void expand_hc_kernel_end();

extern struct hicuda_kernel_part* build_hicuda_kernel_part(
    enum kernel_part_distr_type block, enum kernel_part_distr_type thread);

extern void expand_hc_kernel_part_begin(struct hicuda_kernel_part *dir);
extern void expand_hc_kernel_part_end();

extern struct hicuda_loopblock* build_hicuda_loopblock(
    tree tile_sz, struct arr_region *copyin_ar);

extern void expand_hc_loopblock_begin(struct hicuda_loopblock *dir);
extern void expand_hc_loopblock_end();

#ifdef _LANGUAGE_C_PLUS_PLUS
}
#endif

#endif  // _GCCFE_HICUDA_DIRECIVES_H_

/*** DAVID CODE END ***/
