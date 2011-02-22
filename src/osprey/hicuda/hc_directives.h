/** DAVID CODE BEGIN **/

#ifndef _HC_DIRECTIVES_H_
#define _HC_DIRECTIVES_H_

/*****************************************************************************
 *
 * The file defines how hiCUDA directives are represented as WN nodes, i.e.
 * PRAGMA and XPRAGMA nodes. This information is shared between the front-end
 * (kgccfe) and the hiCUDA handler in IPA.
 *
 ****************************************************************************/

// #global alloc / #shared alloc
//
// base PRAGMA: host variable symbol, flags (define below) in arg1
// alloc XPRAGMA
// copyin XPRAGMA (optional)
//

#define HC_DIR_COPYIN               0x01
// For now, used solely in SHARED
#define HC_DIR_COPYIN_NOBNDCHECK    0x02
// For now, used solely in GLOBAL
#define HC_DIR_CLEAR                0x04

// #global copyout / #shared copyout
//
// copyout XPRAGMA: section, flags in st_idx

#endif  // _HC_DIRECTIVES_H_

/*** DAVID CODE END ***/
