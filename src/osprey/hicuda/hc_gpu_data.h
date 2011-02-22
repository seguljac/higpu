/** DAVID CODE BEGIN **/

#ifndef _HC_GPU_DATA_H_
#define _HC_GPU_DATA_H_

#include "defs.h"
#include "wn.h"

#include "cxx_template.h"

class HC_GPU_DATA;
class HC_LOCAL_VAR_STORE;


/*****************************************************************************
 *
 * This is embedded in HC_GPU_DATA, which provides the procedure context for
 * WN nodes in this class.
 *
 ****************************************************************************/

class HC_GPU_VAR_INFO
{
private:

    // the GPU memory variable symbol
    //
    // For a GLOBAL directive, this should be filled once the directive is
    // lowered to CUDA code. For a propagated directive, this stores the local
    // global memory variable.
    //
    // For a CONSTANT directive, this is left ST_IDX_ZERO because the constant
    // memory variable is at the offset <_offset> of the global <cmem> which
    // can be retrieved from <hc_glob_var_store>. This field is only filled
    // during access redirection, when a local variable will be created to be
    // initialized with <cmem> + <_offset>.
    //
    ST_IDX _st_idx;

    // the size of the GPU memory variable in bytes
    // (# of array elements (or 1 for scalar) * elem_size)
    //
    // THIS IS A NEW INSTANCE.
    //
    WN *_size_wn;

    // offset of <cmem> in bytes
    //
    // This is meaningful for a constant memory variable only, and is always
    // -1 for a global memory variable.
    //
    INT _offset;

    // This GPU memory variable is used in access redirection in the local
    // procedure.
#define HC_GVAR_LOCAL_REF   0x01

    UINT _flags;

public:

    HC_GPU_VAR_INFO()
    {
        _st_idx = ST_IDX_ZERO;
        _size_wn = NULL;
        _offset = -1;
        _flags = 0;
    }
    ~HC_GPU_VAR_INFO() {}

    ST_IDX get_symbol() const { return _st_idx; }
    void set_symbol(ST_IDX st_idx)
    {
        Is_True(st_idx != ST_IDX_ZERO, (""));
        _st_idx = st_idx;
    }

    INT get_offset() const { return _offset; }
    void set_offset(INT offset)
    {
        Is_True(offset >= 0, (""));
        _offset = offset;
    }

    // Returns a reference. DON'T DESTROY IT.
    WN* get_size() const
    {
        Is_True(_size_wn != NULL, (""));
        return _size_wn;
    }
    void set_size(WN *size_wn)
    {
        Is_True(_size_wn == NULL && size_wn != NULL, (""));
        _size_wn = size_wn;
    }

    BOOL has_local_ref() const { return (_flags & HC_GVAR_LOCAL_REF); }
    void set_local_ref() { _flags |= HC_GVAR_LOCAL_REF; }
};

/*****************************************************************************
 *
 * The following functions make two assumptions:
 *
 * 1) The correct procedure context has been set up already, mostly the one
 *    contains the given data directive.
 *
 * 2) The relevant HC_GPU_VAR_INFO has been created.
 *
 ****************************************************************************/

extern WN* HC_lower_global_copyin(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code);

extern WN* HC_lower_global_copyout(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code);

extern WN* HC_lower_global_free(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, BOOL gen_code);


extern void HC_create_local_cvar(HC_GPU_DATA *gdata);
extern WN* HC_create_cvar_init_stmt(const HC_GPU_VAR_INFO *gvi);

extern WN* HC_lower_const_copyin(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code);

extern WN* HC_lower_const_remove(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *gdata, BOOL gen_code);


extern void HC_declare_svar(HC_GPU_DATA *sdata);

extern WN* HC_lower_shared_copyin(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *sdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code);

extern WN* HC_lower_shared_copyout(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *sdata, HC_LOCAL_VAR_STORE *lvar_store, BOOL gen_code);

extern WN* HC_lower_shared_remove(WN *pragma_wn, WN *parent_wn,
        HC_GPU_DATA *sdata, BOOL gen_code);


extern WN* HC_create_gvar_access_for_array(WN *wn, HC_GPU_DATA *gdata);

extern WN* HC_create_gvar_access_for_scalar(WN *wn, HC_GPU_DATA *gdata);

#endif  // _HC_GPU_DATA_H_

/*** DAVID CODE END ***/
