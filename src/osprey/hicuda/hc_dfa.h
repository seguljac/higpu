/** DAVID CODE BEGIN **/

#ifndef _HICUDA_DFA_H_
#define _HICUDA_DFA_H_

/* Data Flow Analysis (DFA) */

#include "bitvector.h"


enum dfs_state_t {
    DFS_NYS,    // not yet started
    DFS_WIP,    // work in progress
    DFS_DONE    // finished
};

typedef enum dfs_state_t DFS_STATE;


// Types of DFA
enum dfa_type {
    DFA_LIVE_VAR,
    NUM_DFA     // number of DFA types
};

typedef enum dfa_type DFA_TYPE;

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

struct dfa_info_t {
    bit_vector *in;             // info at the beginning of a block
    bit_vector *out;            // info at the end of a block
    bit_vector *gen;            // GEN set
    bit_vector *kill;           // KILL set
};

typedef struct dfa_info_t dfa_info;

inline void
init_dfa_info(dfa_info *di) {
    di->in = di->out = di->gen = di->kill = NULL;
}

/* Free memory and reset fields */
extern void reset_dfa_info(dfa_info *di);

/**
 * Return the two properties of an DFA analysis.
 * ANY/ALL and FORWARD/BACKWARD
 * TRUE for ANY, FALSE for ALL
 * TRUE for FORWARD, FALSE for BACKWARD
 */
extern void get_analysis_property(DFA_TYPE type, bool *isAny, bool *direction);

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

struct hc_bblist_t;

extern void dfa_solver(struct hc_bblist_t *list, DFA_TYPE type, int nbits);

#endif  // _HICUDA_DFA_H_

/*** DAVID CODE END ***/
