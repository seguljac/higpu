/** DAVID CODE BEGIN **/

/**
 * NOTICE: The bitvector implementation is borrowed from my ECE540 homework.
 */

#ifndef _HICUDA_BITVECTOR_H_
#define _HICUDA_BITVECTOR_H_

#include <stdio.h>

/* Bit Vector Definitions 
 *
 * Bit vectors are stored in arrays of unsigned integers.
 * The bits are numbered beginning with 0.
 */

typedef struct bit_vector_struct {
    int num_bits;
    unsigned *bits;
} bit_vector;

typedef void (*bitprint_f)(FILE *fd, int bit);
/* Added by David */
typedef void (*bop_bits)(bit_vector *, bit_vector*);

extern bit_vector *new_bit_vector(int num_bits);
extern void set_bit(bit_vector *b, int ndx, bool value);
extern bool get_bit(bit_vector *b, int ndx);
extern void set_all_bits(bit_vector *b, bool value);
extern void subtract_bits(bit_vector *b, bit_vector *c);
extern void and_bits(bit_vector *b, bit_vector *c);
extern void or_bits(bit_vector *b, bit_vector *c);
extern void copy_bits(bit_vector *b, bit_vector *c);
extern bool bits_are_equal(bit_vector *b, bit_vector *c);
extern bool bits_are_false(bit_vector *b);
extern void fprint_bits(FILE *fd, bit_vector *b, bitprint_f bitprint);
extern void fprint_bit(FILE *fd, int bit);
extern void free_bit_vector(bit_vector *b);

/** Added by David **/
extern void negate_bits(bit_vector *b, bit_vector *c);
extern bool diff_by_one_bit(bit_vector *b, bit_vector *c);
extern bool contains(bit_vector *b, bit_vector *c);
extern void fprint_bit_vector(FILE *fd, bit_vector *b);

#endif  // _HICUDA_BITVECTOR_H_

/*** DAVID CODE END ***/
