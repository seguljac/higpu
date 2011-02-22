/** DAVID CODE BEGIN **/

#ifndef _GCCFE_WFE_HIDUCA_DIRECIVES_H_
#define _GCCFE_WFE_HICUDA_DIRECIVES_H_

extern void wfe_expand_hc_global_shared(struct hicuda_global_shared *dir,
        bool is_global_dir);
extern void wfe_expand_hc_global_free(struct free_data_list *dir);
extern void wfe_expand_hc_shared_remove(struct free_data_list *dir);

extern void wfe_expand_hc_const(struct hicuda_const *dir);
extern void wfe_expand_hc_const_remove(struct free_data_list *dir);

extern void wfe_expand_hc_shape(struct hicuda_shape *dir);

extern void wfe_expand_hc_barrier();

extern void wfe_expand_hc_kernel_begin(struct hicuda_kernel *dir);
extern void wfe_expand_hc_kernel_end();

extern void wfe_expand_hc_kernel_part_begin(struct hicuda_kernel_part *dir);
extern void wfe_expand_hc_kernel_part_end();

extern void wfe_expand_hc_loopblock_begin(struct hicuda_loopblock *dir);
extern void wfe_expand_hc_loopblock_end();

#endif  // _GCCFE_WFE_HIDUCA_DIRECIVES_H_

/*** DAVID CODE END ***/
