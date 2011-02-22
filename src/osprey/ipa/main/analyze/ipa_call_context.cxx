/** DAVID CODE BEGIN **/

#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "defs.h"
#include "wn.h"
#include "wn_simp.h"
#include "wn_util.h"

#include "ipa_cg.h"
#include "ipa_summary.h"


INT IPA_CALL_CONTEXT::_internal_id = 0;

void IPA_CALL_CONTEXT::add(IPA_EDGE *e, IPA_CALL_CONTEXT *caller_context)
{
    Is_True(e != NULL, ("IPA_CALL_CONTEXT::add: NULL edge\n"));

    IPA_CALL_CONTEXT_LIST *l = _table->Find(e);
    if (l == NULL) {
        l = CXX_NEW(IPA_CALL_CONTEXT_LIST, _pool);
        _table->Enter(e, l);
    }
    l->Append(caller_context);
}

void IPA_CALL_CONTEXT::print(FILE *fp)
{
    fprintf(fp, "[CC%d", _id);
    
    IPA_EDGE *e;
    IPA_CALL_CONTEXT_LIST *cl;
    IPA_CALLER_TABLE_ITER ti(_table);
    while (ti.Step(&e, &cl)) {
        fprintf(fp, " {E%d", e->Edge_Index());
        IPA_CALL_CONTEXT_ITER ci(cl);
        for (IPA_CALL_CONTEXT *cc = ci.First(); !ci.Is_Empty();
                cc = ci.Next()) fprintf(fp, " CC%d", cc->_id);
        fprintf(fp, "}");
    }

    fprintf(fp, "]");
}

/*** DAVID CODE END ***/
