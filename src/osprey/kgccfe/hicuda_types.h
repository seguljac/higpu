/** DAVID CODE BEGIN **/

#ifndef _GCCFE_HICUDA_TYPES_H_
#define _GCCFE_HICUDA_TYPES_H_

// HiCUDA directive types
enum hicuda_tree_type {
    /* iteration space directives */
    HICUDA_KERNEL_DIR_BEGIN,
    HICUDA_KERNEL_DIR_END,
    HICUDA_KERNEL_PART_DIR_BEGIN,
    HICUDA_KERNEL_PART_DIR_END,
    HICUDA_LOOPBLOCK_DIR_BEGIN,
    HICUDA_LOOPBLOCK_DIR_END,
    /* data directives */
    HICUDA_GLOBAL_COPYIN_DIR,
    HICUDA_GLOBAL_COPYOUT_DIR,
    HICUDA_GLOBAL_FREE_DIR,
    HICUDA_CONST_COPYIN_DIR,
    HICUDA_CONST_REMOVE_DIR,
    HICUDA_SHARED_COPYIN_DIR,
    HICUDA_SHARED_COPYOUT_DIR,
    HICUDA_SHARED_REMOVE_DIR,
    HICUDA_SHAPE_DIR,
    /* barrier directive */
    HICUDA_BARRIER_DIR,
    /* number of directives */
    NUM_HICUDA_DIRS
};

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

// The virtual block/thread space (a list of dimension sizes)
struct virtual_space {
    tree dim_size;
    struct virtual_space *next;
};

// Data structure for the kernel directive
struct hicuda_kernel {
    tree kname;                             // kernel name
    struct virtual_space *block;            // block space geometry
    struct virtual_space *thread;           // thread space geometry
};

// The distribution strategy for kernel partitioning
enum kernel_part_distr_type {
    HC_KERNEL_PART_BLOCK,
    HC_KERNEL_PART_CYCLIC,
    HC_KERNEL_PART_NONE
};

// Data structure for the kernel partition directive
struct hicuda_kernel_part {
    enum kernel_part_distr_type block;
    enum kernel_part_distr_type thread;
};

// Data structure for the loopblock directive
struct hicuda_loopblock {
    tree tile_sz;                       // tile size
    struct arr_region *copyin;          // optional copyin clause list
};

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

struct idx_range_list
{
    // If both start_idx and end_idx are NULL, it is a full range.
    // If only end_idx is NULL, it is a single index.
    tree start_idx;
    tree end_idx;

    struct idx_range_list *next;
};

// e.g. A[0:N][0:M]
// chainable
struct arr_region
{
    tree var;                   // array variable identifer
    struct idx_range_list *irl; // NULL for a scalar variable

    struct arr_region *next;
};

// copyin/copyout array region
struct copy_arr_region
{
    struct arr_region *region;      // could be NULL in a ALLOC directive
                                    // not chained
    bool nobndcheck;                // for a shared directive
};

/* Data structure for a global or shared directive */
struct hicuda_global_shared
{
    struct arr_region *decl;        // NULL iff it is a COPYOUT directive
                                    // not chained
    struct copy_arr_region *copy;   // could be NULL in a ALLOC directive

    bool clear_region;              // now only meaningful for GLOBAL
};

/* Data structure for the const directive */
struct hicuda_const
{
    struct arr_region *copyin;      // must not be NULL (chained)
};

/* Data structure for GLOBAL FREE, SHARED REMOVE or CONST REMOVE */
struct free_data_list
{
    tree var;
    struct free_data_list *next;
};

struct dim_sz_list
{
    tree dim_sz;                // an expression
    struct dim_sz_list *next;
};

/* Data structure for the shape directive */
struct hicuda_shape
{
    tree var;                   // array variable identifier
    struct dim_sz_list *shape;  // a list of array dimension sizes

    struct hicuda_shape *next;  // chainable
};

#endif  // _GCCFE_HICUDA_TYPES_H_

/*** DAVID CODE END ***/
