/** DAVID CODE BEGIN **/

#ifndef _IPA_HC_ANNOT_H_
#define _IPA_HC_ANNOT_H_

/*****************************************************************************
 *
 * Base classes for an annotation to be propagated interprocedurally.
 * 
 * An annotation consists of information tagged to each formal and its call
 * context.
 *
 ****************************************************************************/

#include "defs.h"

#include "cxx_base.h"

class IPA_EDGE;

// forward declaration
class IPA_CALL_CONTEXT_LIST;

typedef HASH_TABLE<IPA_EDGE*, IPA_CALL_CONTEXT_LIST*>
IPA_CALLER_TABLE;
typedef HASH_TABLE_ITER<IPA_EDGE*, IPA_CALL_CONTEXT_LIST*>
IPA_CALLER_TABLE_ITER;

class IPA_CALL_CONTEXT : public SLIST_NODE
{
    static UINT _internal_id;

private:

    UINT _id;       // start from 1

    IPA_CALLER_TABLE *_table;

    MEM_POOL *_pool;

    DECLARE_SLIST_NODE_CLASS(IPA_CALL_CONTEXT);

public:

    IPA_CALL_CONTEXT(MEM_POOL *pool) {
        _pool = pool;
        _id = (++_internal_id);
        _table = CXX_NEW(IPA_CALLER_TABLE(307, pool), pool);
    }

    // copy constructor
    IPA_CALL_CONTEXT(const IPA_CALL_CONTEXT *orig);

    ~IPA_CALL_CONTEXT() {}

    UINT get_id() const { return _id; }

    void add(IPA_EDGE *e, IPA_CALL_CONTEXT *caller_context);

    BOOL contains_edge(IPA_EDGE *e) const
    {
        Is_True(e != NULL, (""));
        return (_table != NULL && _table->Find(e) != NULL);
    }

    /**
     * This function sets all incoming edges in this context TO_BE_DELETED.
     * It is used in the process of cloning and edge fixing.
     */
    void mark_edges_to_be_deleted();

    IPA_CALLER_TABLE* get_caller_table() const { return _table; }

    BOOL equals(const IPA_CALL_CONTEXT *other) const;

    void print(FILE *fp) const;
};

class IPA_CALL_CONTEXT_LIST : public SLIST
{
private:

    IPA_CALL_CONTEXT_LIST(const IPA_CALL_CONTEXT_LIST&);
    IPA_CALL_CONTEXT_LIST& operator = (const IPA_CALL_CONTEXT_LIST&);

    DECLARE_SLIST_CLASS(IPA_CALL_CONTEXT_LIST, IPA_CALL_CONTEXT);

public:

    ~IPA_CALL_CONTEXT_LIST() {}
};

class IPA_CALL_CONTEXT_ITER : public SLIST_ITER
{
private:

    DECLARE_SLIST_ITER_CLASS(IPA_CALL_CONTEXT_ITER,
            IPA_CALL_CONTEXT, IPA_CALL_CONTEXT_LIST);
};


/*****************************************************************************
 *
 * Base class that represents the acutal annotation information
 *
 ****************************************************************************/

class HC_ANNOT_DATA
{
public:

    virtual BOOL is_dummy() const = 0;

    /**
     * ASSUME: both annotation data are in the same procedure context, which
     * must have been set up before calling this function.
     *
     * This check should be consistent with the fact that two dummy annotation
     * data are equal.
     */
    virtual BOOL equals(const HC_ANNOT_DATA* other) const = 0;

    /**
     * ASSUME: the appropriate procedure context must be set up before calling
     * this function.
     */
    virtual void print(FILE *fp) const = 0;
};

/*****************************************************************************
 *
 * It includes two flags used in IP propagation:
 *
 * 1) HC_AD_DUMMY: set this annotation provides no useful annotation
 * information (used in starting up the propagation and cloning)
 *
 * 2) HC_AD_PROCESSED: set when this annotation has been processed.
 *
 ****************************************************************************/

class IPA_HC_ANNOT : public SLIST_NODE
{
private:

#define HCA_PROCESSED 0x01
#define HCA_DUMMY     0x02

    UINT _flags;

    HC_ANNOT_DATA *_annot;      // This could be NULL (i.e. dummy)

    IPA_CALL_CONTEXT *_context; // This is never NULL

    MEM_POOL *_pool;

    void set_dummy() { _flags |= HCA_DUMMY; }

public:

    IPA_HC_ANNOT(HC_ANNOT_DATA *annot, BOOL is_dummy, MEM_POOL *pool)
    {
        _pool = pool;

        // By default, the annotation is not processed.
        _flags = 0;
        if (is_dummy) set_dummy();

        _annot = annot;
        _context = CXX_NEW(IPA_CALL_CONTEXT(pool), pool);
    }

    ~IPA_HC_ANNOT() {}

    BOOL is_processed() const { return (_flags & HCA_PROCESSED); }
    BOOL set_processed() { _flags |= HCA_PROCESSED; }

    BOOL is_dummy() const { return (_flags & HCA_DUMMY); }

    HC_ANNOT_DATA* get_annot_data() const { return _annot; }

    IPA_CALL_CONTEXT* get_call_context() const
    {
        Is_True(_context != NULL, (""));
        return _context;
    }

    void clear_call_context() { _context = NULL; }

    // This function does not make a copy of <caller_context>.
    void add_call_context(IPA_EDGE *e, IPA_CALL_CONTEXT *caller_context)
    {
        Is_True(caller_context != NULL, (""));
        _context->add(e, caller_context);
    }

    /**
     * ASSUME: both annotation data are in the same procedure context, which
     * must have been set up before calling this function.
     *
     * This check takes into account the dumminess.
     */
    BOOL equals(const IPA_HC_ANNOT *other) const;

    /**
     * ASSUME: the appropriate procedure context must be set up before calling
     * this function.
     */
    void print(FILE *fp) const;
};

class IPA_HC_ANNOT_LIST : public SLIST
{
private:

    MEM_POOL *_pool;

    IPA_HC_ANNOT_LIST(const IPA_HC_ANNOT_LIST&);
    IPA_HC_ANNOT_LIST& operator = (const IPA_HC_ANNOT_LIST&);

    DECLARE_SLIST_CLASS(IPA_HC_ANNOT_LIST, IPA_HC_ANNOT);

public:

    /* DO NOT USE THE DEFAULT CONSTRUCTOR!!
     *
     * We cannot disable the default constructor due to the macro above.
     */
    IPA_HC_ANNOT_LIST(MEM_POOL *pool) : SLIST() { _pool = pool; }
    ~IPA_HC_ANNOT_LIST() {}

    /**
     * Add the annotation data (in the callee space) to this list (in the
     * callee), which comes from the given edge with the given caller context.
     *
     * Return TRUE if a new IPA_HC_ANNOT is added, or FALSE otherwise.
     *
     * This function inserts a COPY of <caller_context>, which is created
     * using this list's mem pool.
     */
    BOOL add(IPA_EDGE *e, IPA_CALL_CONTEXT *caller_context,
            HC_ANNOT_DATA *adata);

    /**
     * Add a dummy annotation (with no call context), which is used to start
     * up the data flow framework.
     */
    void add_dummy();

    /**
     * Search for the annotation data associated with the given call context.
     * Return NULL if not found.
     */
    HC_ANNOT_DATA* find_annot_data(const IPA_CALL_CONTEXT *context);

    /**
     * ASSUME: the appropriate procedure context must be set up before calling
     * this function.
     */
    void print(FILE *fp);
};

class IPA_HC_ANNOT_ITER : public SLIST_ITER
{
private:

    DECLARE_SLIST_ITER_CLASS(IPA_HC_ANNOT_ITER,
            IPA_HC_ANNOT, IPA_HC_ANNOT_LIST);
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Create clones and fix the edges in the call graph.
 *
 ****************************************************************************/

extern void IPA_HC_clone_and_fix_call_edges(MEM_POOL *m);

/*****************************************************************************
 *
 * A convenience method that prints hiCUDA annotations of each node.
 *
 ****************************************************************************/

extern void IPA_print_hc_annots(FILE *fp);

#endif  // _IPA_HC_ANNOT_H_

/*** DAVID CODE END ***/
