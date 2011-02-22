/** DAVID CODE BEGIN **/

#include "hc_ic_solver.h"

INT HC_IG_NODE::compare_lf1(const HC_IG_NODE& other) const
{
    UINT w1 = get_weight(), w2 = other.get_weight();
    UINT cd1 = get_chro_deg(), cd2 = other.get_chro_deg();

    if (w1 != w2) return w1 - w2;
    return cd1 - cd2;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void HC_IC_SOLVER::add_node(const HC_IG_NODE_INFO *ni)
{
    HC_IG_NODE_LN *nln;
    HC_IG_NODE *n;

    Is_True(ni != NULL, (""));

    // Make sure this node info is unique.
    for (nln = _node_head; nln != NULL; nln = nln->Next())
    {
        n = nln->get_ig_node();
        Is_True(! ni->equals(*n->get_node_info()), (""));
    }

    // Create the wrappers.
    n = CXX_NEW(HC_IG_NODE(ni, _pool), _pool);
    nln = CXX_NEW(HC_IG_NODE_LN(n), _pool);

    // Add the node to the list.
    nln->Set_Next(_node_head);
    _node_head = nln;
}

void HC_IC_SOLVER::connect(const HC_IG_NODE_INFO *ni1,
        const HC_IG_NODE_INFO *ni2)
{
    Is_True(ni1 != NULL && ni2 != NULL, (""));
    Is_True(! ni1->equals(*ni2), (""));

    // Make sure that both nodes exist in the IG.
    HC_IG_NODE *n1 = NULL, *n2 = NULL;
    for (HC_IG_NODE_LN *nln = _node_head;
            nln != NULL && (n1 == NULL || n2 == NULL); nln = nln->Next())
    {
        HC_IG_NODE *n = nln->get_ig_node();
        const HC_IG_NODE_INFO& ni = *(n->get_node_info());

        if (ni1->equals(ni))
        {
            n1 = n;
        }
        else if (ni2->equals(ni))
        {
            n2 = n;
        }
    }
    Is_True(n1 != NULL && n2 != NULL, (""));

    n1->add_neighbour(n2);
    n2->add_neighbour(n1);
}

INT HC_IC_SOLVER::get_offset(const HC_IG_NODE_INFO *ni) const
{
    // Make sure that this node exists in the IG.
    for (HC_IG_NODE_LN *nln = _node_head; nln != NULL; nln = nln->Next())
    {
        HC_IG_NODE *n = nln->get_ig_node();
        if (ni->equals(*n->get_node_info())) return n->get_offset();
    }

    Is_True(FALSE, (""));
    return -1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * Sort the nodes in LF1 order: decreasing vertex weight, with decreasing
 * chromatic degree sub-order
 *
 ****************************************************************************/

void HC_IC_SOLVER::sort_nodes()
{
    // Rebuild the list.
    HC_IG_NODE_LN *new_list = NULL;
    while (_node_head != NULL)
    {
        HC_IG_NODE_LN *nln = _node_head;
        _node_head = nln->Next();

        // Insert <nln> to <new_list>.
        HC_IG_NODE_LN *prev = NULL, *curr = new_list;
        while (curr != NULL
                && curr->get_ig_node()->compare_lf1(*nln->get_ig_node()) > 0)
        {
            prev = curr;
            curr = curr->Next();
        }
        if (prev == NULL)
        {
            new_list = nln;
        }
        else
        {
            prev->Set_Next(nln);
        }
        nln->Set_Next(curr);
    }

    _node_head = new_list;
}

/*****************************************************************************
 *
 * Compute the max interval for the current offset assignments.
 *
 ****************************************************************************/

INT HC_IC_SOLVER::compute_max_interval()
{
    INT max = 0;

    for (HC_IG_NODE_LN *nln = _node_head; nln != NULL; nln = nln->Next())
    {
        HC_IG_NODE *n = nln->get_ig_node();
        INT n_ofst = n->get_offset();
        if (n_ofst >= 0)
        {
            INT new_max = n_ofst + n->get_weight();
            if (max < new_max) max = new_max;
        }
    }

    return max;
}

/*****************************************************************************
 *
 * Build a list of nodes that are connected to the given node and have the
 * offset assigned, in non-decreasing order of the offset.
 *
 ****************************************************************************/

HC_IG_NODE_LN* HC_IC_SOLVER::build_neighbour_timeline(HC_IG_NODE *n)
{
    HC_IG_NODE_LN *nlist = NULL;

    // Go through the neighbours of <n>.
    UINT n_neighbours = n->get_num_neighbours();
    for (UINT i = 0; i < n_neighbours; ++i)
    {
        HC_IG_NODE *nb = n->get_neighbour(i);
        INT nb_ofst = nb->get_offset();
        if (nb_ofst < 0) continue;

        // Add this node to the list to maintain non-decreasing offset.
        HC_IG_NODE_LN *prev = NULL, *curr = nlist;
        while (curr != NULL && curr->get_ig_node()->get_offset() < nb_ofst)
        {
            prev = curr;
            curr = curr->Next();
        }
        HC_IG_NODE_LN *nln = CXX_NEW(HC_IG_NODE_LN(nb), _pool);
        nln->Set_Next(curr);
        if (prev == NULL)
        {
            nlist = nln;
        }
        else
        {
            prev->Set_Next(nln);
        }
    }

    return nlist;
}

class HC_FLEX_NODE_INFO
{
public:
    INT _offset_min;
    INT _offset_max;
    INT _weight;
};

/*****************************************************************************
 *
 * Determine where to place the given node in its neighour timeline, so that
 * no overlapping occurs.
 *
 * Return the new max interval (or -1 if the placement is impossible) and the
 * min/max offset in <offset_min> and <offset_max>. If the given node is
 * inserted at last, the min and max offset will be the same if <max> is -1,
 * or some range bounded by <max> otherwise.
 *
 * If a flexible node is given, consider placing it in the timeline as well.
 * Update the offset min/max if necessary. <curr_max> has already taken this
 * flexible node into account.
 *
 ****************************************************************************/

INT HC_IC_SOLVER::assign_offset(HC_IG_NODE *n, HC_IG_NODE_LN *timeline,
        INT curr_max, INT threshold_max, HC_FLEX_NODE_INFO *fni,
        INT *offset_min, INT *offset_max)
{
    Is_True(threshold_max == -1 || curr_max <= threshold_max, (""));

    // Go through each node in the timeline and consider placing <n> right
    // before this node.
    HC_IG_NODE_LN *prev = NULL, *curr = timeline;
    HC_IG_NODE *prev_n = NULL, *curr_n = NULL;
    INT n_ofst_min = 0;
    while (curr != NULL)
    {
        curr_n = curr->get_ig_node();

        // Is there a big enough gap between <prev> and <curr>?
        INT n_ofst_max = curr_n->get_offset() - n->get_weight();
        if (n_ofst_min <= n_ofst_max)
        {
            if (fni == NULL)
            {
                *offset_min = n_ofst_min;
                *offset_max = n_ofst_max;
                return curr_max;
            }

            // If there is a flexible node, does this assignment conflict with
            // this node?
            if (n_ofst_min + n->get_weight() <= fni->_offset_max)
            {
                *offset_min = *offset_max = n_ofst_min;
                if (n_ofst_min + n->get_weight() > fni->_offset_min)
                {
                    fni->_offset_min = n_ofst_min + n->get_weight();
                }
                return curr_max;
            }
            if (fni->_offset_min + fni->_weight <= n_ofst_max)
            {
                fni->_offset_max = fni->_offset_min;
                *offset_max = n_ofst_max;
                *offset_min = fni->_offset_min + fni->_weight;
                if (n_ofst_min > *offset_min) *offset_min = n_ofst_min;
                return curr_max;
            }
        }

        prev = curr;
        prev_n = curr_n;
        curr = curr->Next();

        if (n_ofst_min < prev_n->get_offset() + prev_n->get_weight())
        {
            n_ofst_min = prev_n->get_offset() + prev_n->get_weight();
        }
    }

    // Insert after the last node (stored in <prev>).
    if (fni != NULL)
    {
        if (n_ofst_min + n->get_weight() <= fni->_offset_max)
        {
            *offset_min = *offset_max = n_ofst_min;
            if (n_ofst_min + n->get_weight() > fni->_offset_min)
            {
                fni->_offset_min = n_ofst_min + n->get_weight();
            }
            return curr_max;
        }
        if (fni->_offset_min + fni->_weight > n_ofst_min)
        {
            n_ofst_min = fni->_offset_min + fni->_weight;
        }
    }

    INT new_max = n_ofst_min + n->get_weight();
    if (threshold_max != -1 && new_max > threshold_max) return -1;

    *offset_min = n_ofst_min;
    if (new_max <= curr_max) 
    {
        *offset_max = curr_max - n->get_weight();
        return curr_max;
    }
    else
    {
        // TODO: this part does not follow the original algorithm.
        // The returned max interval should be a range.
        *offset_max = n_ofst_min;
        return new_max;
    }
}

class HC_IC_INTERCHANGE
{
public:

    // the interchange target and its new offset
    HC_IG_NODE *_other_n;
    INT _other_n_offset;

    // the current node's offset
    INT _n_offset;

    // the max interval in this configuration
    INT _max_interval;
};

INT HC_IC_SOLVER::solve()
{
    if (_node_head == NULL) return 0;

    // Sort nodes in LF1 order.
    sort_nodes();

    // Start off by assigning 0 offset to the first node.
    HC_IG_NODE *n = _node_head->get_ig_node();
    n->set_offset(0);
    INT max_interval = n->get_weight();

    // Assign node offset in this order.
    HC_IG_NODE_LN *nln = _node_head->Next();
    while (nln != NULL)
    {
        n = nln->get_ig_node();

        // store the running best interchange configuration
        HC_IC_INTERCHANGE iic;
        iic._other_n = NULL; iic._other_n_offset = 0;

        // This is the initial attempt (with no interchange).
        HC_IG_NODE_LN *ntl = build_neighbour_timeline(n);
        INT n_ofst_min = 0, n_ofst_max = 0;
        iic._max_interval = assign_offset(n, ntl, max_interval, -1, NULL,
                &n_ofst_min, &n_ofst_max);
        Is_True(iic._max_interval > 0, (""));
        iic._n_offset = n_ofst_min;

        if (iic._max_interval > max_interval)
        {
            // Start the interchange heuristic.
            HC_FLEX_NODE_INFO fni;

            HC_IG_NODE_LN *ntl_prev = NULL, *ntl_curr = ntl;
            while (ntl_curr != NULL)
            {
                // Remove each node in the neighbour timeline and try building
                // a flexible node for <n>.
                if (ntl_prev == NULL)
                {
                    ntl = ntl_curr->Next();
                }
                else
                {
                    ntl_prev->Set_Next(ntl_curr->Next());
                }

                HC_IG_NODE *ntl_curr_n = ntl_curr->get_ig_node();

                // Revoke the offset assigned to <ntl_curr_n>.
                INT old_curr_ofst = ntl_curr_n->get_offset();
                ntl_curr_n->reset_offset();

                // Compute the max interval in this fallback configuration.
                INT fallback_max = compute_max_interval();

                // Re-run the offset assigning process on <n>.
                INT new_max1 = assign_offset(n, ntl, fallback_max,
                        iic._max_interval-1, NULL,
                        &fni._offset_min, &fni._offset_max);
                if (new_max1 > 0)
                {
                    // Build a neighbour timeline for <ntl_curr_n>.
                    HC_IG_NODE_LN *curr_ntl =
                        build_neighbour_timeline(ntl_curr_n);

                    // Assign offset to <ntl_curr_n> with a flexible node <n>.
                    fni._weight = n->get_weight();
                    INT curr_ofst_min = 0, curr_ofst_max = 0;
                    INT new_max2 = assign_offset(ntl_curr_n, curr_ntl,
                            new_max1, iic._max_interval-1, &fni,
                            &curr_ofst_min, &curr_ofst_max);
                    if (new_max2 > 0)
                    {
                        // This is a better interchange configuration.
                        iic._other_n = ntl_curr_n;
                        iic._other_n_offset = curr_ofst_min;
                        iic._n_offset = fni._offset_min;
                        iic._max_interval = new_max2;
                    }
                }

                // Restore the offset assigned to <ntl_curr_n>.
                ntl_curr_n->set_offset(old_curr_ofst);

                // Add the node back to the timeline.
                if (ntl_prev == NULL)
                {
                    ntl = ntl_curr;
                }
                else
                {
                    ntl_prev->Set_Next(ntl_curr);
                }

                ntl_prev = ntl_curr;
                ntl_curr = ntl_curr->Next();
            }
        }

        // Commit the best interchange configuration.
        max_interval = iic._max_interval;
        n->set_offset(iic._n_offset);
        if (iic._other_n != NULL)
        {
            iic._other_n->set_offset(iic._other_n_offset);
        }

        nln = nln->Next();
    }

    return max_interval;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

HC_IG_NODE* HC_IC_SOLVER::search_ig_node(const HC_IG_NODE_INFO *ni) const
{
    for (HC_IG_NODE_LN *curr = _node_head; curr != NULL; curr = curr->Next())
    {
        HC_IG_NODE *n = curr->get_ig_node();
        if (n->get_node_info()->equals(*ni)) return n;
    }

    return NULL;
}

HC_IC_SOLVER* HC_IC_SOLVER::sub_ic_solver(
        HC_IG_NODE_INFO **subgraph, UINT n_nodes) const
{
    HC_IC_SOLVER *sub_ics = CXX_NEW(HC_IC_SOLVER(_pool), _pool);

    // Add the vertices.
    for (UINT i = 0; i < n_nodes; ++i)
    {
        Is_True(search_ig_node(subgraph[i]) != NULL, (""));
        sub_ics->add_node(subgraph[i]);
    }

    // Add the edges.
    for (UINT i = 0; i < n_nodes; ++i)
    {
        HC_IG_NODE *n = search_ig_node(subgraph[i]);
        const HC_IG_NODE_INFO *ni = n->get_node_info();

        // Go through its neighbours.
        UINT n_nbs = n->get_num_neighbours();
        for (UINT j = 0; j < n_nbs; ++j)
        {
            const HC_IG_NODE_INFO *nni = n->get_neighbour(j)->get_node_info();
            // Is this neighour part of the sub-graph?
            if (sub_ics->search_ig_node(nni) != NULL)
            {
                sub_ics->connect(ni, nni);
            }
        }
    }

    return sub_ics;
}

INT HC_IC_SOLVER::compute_subgraph_offset(
        HC_IG_NODE_INFO **nodes, UINT n_nodes) const
{
    INT max = 0;

    for (UINT i = 0; i < n_nodes; ++i)
    {
        HC_IG_NODE *n = search_ig_node(nodes[i]);
        Is_True(n != NULL, (""));
        
        INT n_ofst = n->get_offset();
        Is_True(n_ofst >= 0, (""));

        INT new_max = n_ofst + n->get_weight();
        if (max < new_max) max = new_max;
    }

    return max;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

class HC_IG_TEST_NODE_INFO : public HC_IG_NODE_INFO
{
public:

    HC_IG_TEST_NODE_INFO(UINT weight)
    : HC_IG_NODE_INFO(weight)
    {
    }

    virtual BOOL equals(const HC_IG_NODE_INFO& other) const
    {
        return (this == &other);
    }
};

void ic_solver_test(MEM_POOL *pool)
{
    MEM_POOL_Push(pool);
    {
        HC_IC_SOLVER ic_solver(pool);

        // Populate the nodes.
        HC_IG_TEST_NODE_INFO* nodes[8];
        nodes[0] = CXX_NEW(HC_IG_TEST_NODE_INFO(2), pool);
        nodes[1] = CXX_NEW(HC_IG_TEST_NODE_INFO(1), pool);
        nodes[2] = CXX_NEW(HC_IG_TEST_NODE_INFO(3), pool);
        nodes[3] = CXX_NEW(HC_IG_TEST_NODE_INFO(1), pool);
        nodes[4] = CXX_NEW(HC_IG_TEST_NODE_INFO(2), pool);
        nodes[5] = CXX_NEW(HC_IG_TEST_NODE_INFO(2), pool);
        nodes[6] = CXX_NEW(HC_IG_TEST_NODE_INFO(1), pool);
        nodes[7] = CXX_NEW(HC_IG_TEST_NODE_INFO(3), pool);

        // Add nodes to the solver.
        for (INT i = 0; i < 8; ++i) ic_solver.add_node(nodes[i]);

        // Add interference edges.
        ic_solver.connect(nodes[0], nodes[1]);
        ic_solver.connect(nodes[0], nodes[4]);
        ic_solver.connect(nodes[0], nodes[5]);
        ic_solver.connect(nodes[0], nodes[7]);
        ic_solver.connect(nodes[1], nodes[2]);
        ic_solver.connect(nodes[1], nodes[5]);
        ic_solver.connect(nodes[2], nodes[3]);
        ic_solver.connect(nodes[2], nodes[6]);
        ic_solver.connect(nodes[2], nodes[7]);
        ic_solver.connect(nodes[3], nodes[4]);
        ic_solver.connect(nodes[3], nodes[5]);
        ic_solver.connect(nodes[3], nodes[7]);
        ic_solver.connect(nodes[4], nodes[5]);
        ic_solver.connect(nodes[5], nodes[6]);
        ic_solver.connect(nodes[6], nodes[7]);

        // Solve interval coloring problem.
        INT max_interval = ic_solver.solve();
        printf("MAX_INTERVAL = %d\n", max_interval);
        for (INT i = 0; i < 8; ++i)
        {
            printf("V%d starts at %d\n",
                    i+1, ic_solver.get_offset(nodes[i]));
        }
    }
    MEM_POOL_Pop(pool);
}

/*** DAVID CODE END ***/
