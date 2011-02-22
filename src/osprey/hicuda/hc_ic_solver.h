/** DAVID CODE BEGIN **/

#ifndef _HC_IC_SOLVER_H_
#define _HC_IC_SOLVER_H_

#include "cxx_memory.h"
#include "cxx_base.h"
#include "cxx_template.h"

/*****************************************************************************
 *
 * Interval-coloring solver based on the heuristics proposed by Clementson and
 * Elphick. The client first adds nodes and edges to an interference graph
 * (maintained internally) and then invokes the solve method. The client can
 * then probe the offset of each node and the total internal length.
 *
 * HC_IG_NODE_INFO:
 *      information associated with a node of the interference graph (IG)
 *      The client should inherit this class.
 *
 * HC_IG_NODE:
 *      a wrapper around HC_IG_NODE_INFO, representing a node of the IG.
 *
 * HC_IG_NODE_LN:
 *      a further wrapper of HC_IG_NODE so that it can be chained in different
 *      linked lists
 *
 * HC_IC_SOLVER: interval-coloring solver
 *
 ****************************************************************************/

class HC_IG_NODE_INFO
{
protected:

    UINT _weight;

public:

    HC_IG_NODE_INFO(UINT w) { _weight = w; }
    ~HC_IG_NODE_INFO() {}

    UINT get_weight() const { return _weight; }

    virtual BOOL equals(const HC_IG_NODE_INFO& other) const = 0;
};

class HC_IG_NODE;
typedef DYN_ARRAY<HC_IG_NODE*> HC_IG_NODE_ARRAY;

class HC_IG_NODE
{
private:

    const HC_IG_NODE_INFO *_node_info;

    HC_IG_NODE_ARRAY *_neighbours;

    // chromatic degree: sum of weights for all neighbours
    // updated as neighbours are added.
    UINT _chro_deg;

    // starting offset to be assigned in interval coloring
    // -1 indicates that it has not been assigned.
    INT _offset;

    MEM_POOL *_pool;

public:

    HC_IG_NODE(const HC_IG_NODE_INFO *info, MEM_POOL *pool)
    {
        Is_True(info != NULL, (""));
        _node_info = info;
        Is_True(pool != NULL, (""));
        _pool = pool;

        _neighbours = NULL;
        _chro_deg = 0;
        _offset = -1;
    }

    ~HC_IG_NODE() {}

    const HC_IG_NODE_INFO* get_node_info() const { return _node_info; }

    void add_neighbour(HC_IG_NODE* n)
    {
        Is_True(n != NULL, (""));

        if (_neighbours == NULL)
        {
            _neighbours = CXX_NEW(HC_IG_NODE_ARRAY(_pool), _pool);
        }
        _neighbours->AddElement(n);

        _chro_deg += n->get_weight();
    }
    UINT get_num_neighbours() const
    {
        return (_neighbours == NULL) ? 0 : _neighbours->Elements();
    }
    HC_IG_NODE* get_neighbour(UINT idx) const
    {
        return (idx >= get_num_neighbours()) ? NULL : (*_neighbours)[idx];
    }

    void set_offset(INT offset)
    {
        Is_True(offset >= 0, (""));
        _offset = offset;
    }
    void reset_offset() { _offset = -1; }
    INT get_offset() const { return _offset; }

    UINT get_chro_deg() const { return _chro_deg; }
    UINT get_weight() const { return _node_info->get_weight(); }

    /**
     * Return -ive number if this node should come after <other> in decreasing
     * LF1 order, 0 if equal, or +ive number if come before.
     */
    INT compare_lf1(const HC_IG_NODE& other) const;
};

class HC_IG_NODE_LN : public SLIST_NODE
{
private:

    HC_IG_NODE *_ig_node;

    DECLARE_SLIST_NODE_CLASS(HC_IG_NODE_LN);

public:

    HC_IG_NODE_LN(HC_IG_NODE *node)
    {
        Is_True(node != NULL, (""));
        _ig_node = node;
    }

    HC_IG_NODE* get_ig_node() const { return _ig_node; }
};

class HC_FLEX_NODE_INFO;

class HC_IC_SOLVER
{
private:

    // a list of nodes
    // (sorted in Largest First by chromaticity (LF1) order before solving the
    // interval coloring problem)
    HC_IG_NODE_LN *_node_head;

    MEM_POOL *_pool;

    void sort_nodes();

    INT compute_max_interval();

    HC_IG_NODE_LN* build_neighbour_timeline(HC_IG_NODE *n);

    INT assign_offset(HC_IG_NODE *n, HC_IG_NODE_LN *timeline,
            INT curr_max, INT threshold_max, HC_FLEX_NODE_INFO *fni,
            INT *offset_min, INT *offset_max);

    HC_IG_NODE* search_ig_node(const HC_IG_NODE_INFO *node) const;

public:

    HC_IC_SOLVER(MEM_POOL *pool)
    {
        Is_True(pool != NULL, (""));
        _pool = pool;

        _node_head = NULL;
    }
    ~HC_IC_SOLVER() {}

    void add_node(const HC_IG_NODE_INFO *ni);
    void connect(const HC_IG_NODE_INFO *ni1, const HC_IG_NODE_INFO *ni2);

    // Return the total interval length.
    INT solve();

    INT get_offset(const HC_IG_NODE_INFO *ni) const;

    // Given a subset of the nodes, build a new interference graph that
    // involves only these nodes.
    //
    HC_IC_SOLVER* sub_ic_solver(HC_IG_NODE_INFO **nodes, UINT n_nodes) const;

    // Determine the interval for the given subgraph, assuming that the
    // interval-coloring problem has been solved.
    //
    INT compute_subgraph_offset(HC_IG_NODE_INFO **nodes, UINT n_nodes) const;
};

void ic_solver_test(MEM_POOL *pool);

#endif  // _HC_IC_SOLVER_H_

/*** DAVID CODE END ***/
