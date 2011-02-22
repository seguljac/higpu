/** DAVID CODE BEGIN **/

#include <stdio.h>
#include <assert.h>

#include "defs.h"
#include "glob.h"
#include "config.h"
#include "wn.h"
#include "wn_util.h"

#include "gnu_config.h"
#include "system.h"

#include "srcpos.h"
#include "tree.h"

#include "wfe_expr.h"
#include "wfe_misc.h"
#include "hicuda_types.h"
#include "hicuda_directive.h"
#include "wfe_hicuda_directives.h"
// #include "wfe_omp_check_stack.h"

#include "errors.h"
#include "const.h"

struct idx_range_list* build_idx_range(tree start_idx, tree end_idx)
{
    struct idx_range_list *ir =
        (struct idx_range_list*)malloc(sizeof(struct idx_range_list));

    ir->start_idx = start_idx;
    ir->end_idx = end_idx;

    ir->next = NULL;

    return ir;
}

void chain_idx_range_list(struct idx_range_list *irl1,
        struct idx_range_list *irl2)
{
    if (irl1 == NULL) return;

    while (irl1->next != NULL) irl1 = irl1->next;
    irl1->next = irl2;
}

struct arr_region* build_arr_region(tree var, struct idx_range_list *irl)
{
    struct arr_region *result =
        (struct arr_region*)malloc(sizeof(struct arr_region));

    result->var = var;
    result->irl = irl;
    result->next = NULL;

    return result;
}

struct copy_arr_region* build_copy_arr_region(struct arr_region *ar,
        bool nobndcheck)
{
    struct copy_arr_region *result = (struct copy_arr_region*)
        malloc(sizeof(struct copy_arr_region));
    
    result->region = ar;
    result->nobndcheck = nobndcheck;

    return result;
}

struct arr_region* chain_arr_region_list(struct arr_region *arl1,
        struct arr_region *arl2)
{
    if (arl1 == NULL) return arl2;

    struct arr_region *curr = arl1;
    while (curr->next != NULL) curr = curr->next;
    curr->next = arl2;

    return arl1;
}

struct hicuda_global_shared* build_hicuda_global_shared(
        struct arr_region *decl, struct copy_arr_region *copy,
        bool clear_region)
{
    struct hicuda_global_shared *result = (struct hicuda_global_shared*)
        malloc(sizeof(struct hicuda_global_shared));

    Is_True(decl != NULL || copy != NULL, (""));
    result->decl = decl;
    result->copy = copy;
    result->clear_region = clear_region;

    return result;
}

struct hicuda_const* build_hicuda_const(struct arr_region *copyin)
{
    struct hicuda_const *result = (struct hicuda_const*)
        malloc(sizeof(struct hicuda_const));

    Is_True(copyin != NULL, (""));
    result->copyin = copyin;

    return result;
}

struct free_data_list* build_free_data(tree var)
{
    struct free_data_list *fdl = (struct free_data_list*)
        malloc(sizeof(struct free_data_list));

    fdl->var = var;
    fdl->next = NULL;

    return fdl;
}

void chain_free_data_list(struct free_data_list *fdl1,
        struct free_data_list *fdl2)
{
    if (fdl1 == NULL) return;

    while (fdl1->next != NULL) fdl1 = fdl1->next;
    fdl1->next = fdl2;
}

struct dim_sz_list* build_dim_sz(tree dim_sz)
{
    struct dim_sz_list *result = (struct dim_sz_list*)
        malloc(sizeof(struct dim_sz_list));
    result->dim_sz = dim_sz;
    result->next = NULL;
    return result;
}

void chain_dim_sz_list(struct dim_sz_list *l1, struct dim_sz_list *l2)
{
    if (l1 == NULL) return;

    while (l1->next != NULL) l1 = l1->next;
    l1->next = l2;
}

struct hicuda_shape* build_hicuda_shape(tree var, struct dim_sz_list *l)
{
    struct hicuda_shape *result = (struct hicuda_shape*)
        malloc(sizeof(struct hicuda_shape));
    result->var = var;
    result->shape = l;
    result->next = NULL;
    return result;
}

void chain_shape_list(struct hicuda_shape *sl1, struct hicuda_shape *sl2)
{
    if (sl1 == NULL) return;

    while (sl1->next != NULL) sl1 = sl1->next;
    sl1->next = sl2;
}

void expand_hc_global(struct hicuda_global_shared *dir)
{
    wfe_expand_hc_global_shared(dir, TRUE);
}

void expand_hc_global_free(struct free_data_list *dir)
{
    wfe_expand_hc_global_free(dir);
}

void expand_hc_shared(struct hicuda_global_shared *dir)
{
    wfe_expand_hc_global_shared(dir, FALSE);
}

void expand_hc_shared_remove(struct free_data_list *dir)
{
    wfe_expand_hc_shared_remove(dir);
}

void expand_hc_const(struct hicuda_const *dir)
{
    wfe_expand_hc_const(dir);
}

void expand_hc_const_remove(struct free_data_list *dir)
{
    wfe_expand_hc_const_remove(dir);
}

void expand_hc_shape(struct hicuda_shape *dir)
{
    wfe_expand_hc_shape(dir);
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void expand_hc_barrier()
{
    wfe_expand_hc_barrier();
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

struct virtual_space* build_virtual_space(tree dim_size)
{
    struct virtual_space *result =
        (struct virtual_space*)malloc(sizeof(struct virtual_space));

    result->dim_size = dim_size;
    result->next = NULL;

    return result;
}

void chain_virtual_space(struct virtual_space *vs1, struct virtual_space *vs2)
{
    Is_True(vs1 != NULL, (""));

    while (vs1->next != NULL) vs1 = vs1->next;
    vs1->next = vs2;
}

struct hicuda_kernel* build_hicuda_kernel(tree kname,
        struct virtual_space *block, struct virtual_space *thread)
{
    struct hicuda_kernel *result = (struct hicuda_kernel*)
        malloc(sizeof(struct hicuda_kernel));

    result->kname = kname;
    result->block = block;
    result->thread = thread;

    return result;
}

void expand_hc_kernel_begin(struct hicuda_kernel *dir)
{
    wfe_expand_hc_kernel_begin(dir);
}

void expand_hc_kernel_end()
{
    wfe_expand_hc_kernel_end();
}

struct hicuda_kernel_part* build_hicuda_kernel_part(
        enum kernel_part_distr_type block, enum kernel_part_distr_type thread)
{
    struct hicuda_kernel_part *result = (struct hicuda_kernel_part*)
        malloc(sizeof(struct hicuda_kernel_part));

    result->block = block;
    result->thread = thread;

    return result;
}

void expand_hc_kernel_part_begin(struct hicuda_kernel_part *dir)
{
    wfe_expand_hc_kernel_part_begin(dir);
}

void expand_hc_kernel_part_end()
{
    wfe_expand_hc_kernel_part_end();
}

struct hicuda_loopblock* build_hicuda_loopblock(tree tile_sz,
        struct arr_region *copyin_ar) 
{
    struct hicuda_loopblock *result = (struct hicuda_loopblock*)
        malloc(sizeof(struct hicuda_loopblock));

    result->tile_sz = tile_sz;
    result->copyin = copyin_ar;

    return result;
}

void expand_hc_loopblock_begin(struct hicuda_loopblock *dir)
{
    wfe_expand_hc_loopblock_begin(dir);
}

void expand_hc_loopblock_end()
{
    wfe_expand_hc_loopblock_end();
}

/*** DAVID CODE END ***/
