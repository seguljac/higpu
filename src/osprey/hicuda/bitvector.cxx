/** DAVID CODE BEGIN **/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "bitvector.h"

#define WORDBITS (sizeof(unsigned) * 8)
#define NUM_WORDS(b) (((b) + (WORDBITS-1)) / WORDBITS)

/* Allocate a new bit vector and set all of the bits to false. */

bit_vector *
new_bit_vector (int num_bits)
{
    int num_words = NUM_WORDS(num_bits);
    bit_vector *result = (bit_vector *)malloc(sizeof(bit_vector));
    assert(num_bits >= 0);
    result->num_bits = num_bits;
    result->bits = (unsigned *)calloc(num_words, sizeof(unsigned));
    return result;
}



/*  Change the value of one bit in a vector.  The specified bit is set to
    the given bool value.  */

void
set_bit (bit_vector *b, int ndx, bool value)
{
    unsigned mask, old_bits;
    int word = ndx / WORDBITS;

    assert(ndx < b->num_bits);

    mask = 1 << (ndx % WORDBITS);
    old_bits = b->bits[word];
    if (value == false) {
	mask = ~mask;
	b->bits[word] = old_bits & mask;
    } else {
	b->bits[word] = old_bits | mask;
    }
}



/*  Retrieve the value of a bit in a vector.  */

bool
get_bit (bit_vector *b, int ndx)
{
    unsigned mask;
    int word = ndx / WORDBITS;

    if (ndx >= b->num_bits) {
        printf("%d >= %d\n", ndx, b->num_bits);
    }
    assert(ndx < b->num_bits);

    mask = 1 << (ndx % WORDBITS);
    return ((b->bits[word] & mask) != 0);
}



/*  Set all of the bits in a vector to one value.  */

void
set_all_bits (bit_vector *b, bool value)
{
    int i;
    unsigned new_bits = 0;
    int num_words = NUM_WORDS(b->num_bits);

    if (value) new_bits = ~new_bits;

    for (i = 0; i < num_words; i++) {
	b->bits[i] = new_bits;
    }
}



/*  Subtract bit vectors.  The result replaces the first vector (b = b - c).  */

void
subtract_bits (bit_vector *b, bit_vector *c)
{
    int i;
    unsigned old_bits, mask;
    int num_words = NUM_WORDS(b->num_bits);

    assert(b->num_bits == c->num_bits);

    for (i = 0; i < num_words; i++) {
	old_bits = b->bits[i];
	mask = ~(c->bits[i]);
	b->bits[i] = old_bits & mask;
    }
}

/* Negate a bit vector. The result replaces the first vector (b = ~c).
 * This could be done using set_all_bits and subtract_bits. */

void
negate_bits(bit_vector *b, bit_vector *c) {
    unsigned i, num_words = NUM_WORDS(b->num_bits);

    assert(b->num_bits == c->num_bits);

    for (i = 0; i < num_words; ++i) {
        b->bits[i] = ~ c->bits[i];
    }
}

/*  Boolean AND of two bit vectors.  The result replaces the first vector.  */

void
and_bits (bit_vector *b, bit_vector *c)
{
    int num_words = NUM_WORDS(b->num_bits);
    int i;

    assert(b->num_bits == c->num_bits);

    for (i = 0; i < num_words; i++) {
	b->bits[i] = (b->bits[i] & c->bits[i]);
    }
}



/*  Boolean OR of two bit vectors.  The result replaces the first vector.  */

void
or_bits (bit_vector *b, bit_vector *c)
{
    int num_words = NUM_WORDS(b->num_bits);
    int i;

    assert(b->num_bits == c->num_bits);

    for (i = 0; i < num_words; i++) {
	b->bits[i] = (b->bits[i] | c->bits[i]);
    }
}



/*  Copy bit vector "c" to bit vector "b".  */

void
copy_bits (bit_vector *b, bit_vector *c)
{
    int num_words = NUM_WORDS(b->num_bits);
    int i;

    assert(b->num_bits == c->num_bits);

    for (i = 0; i < num_words; i++) {
	b->bits[i] = c->bits[i];
    }
}



/*  Test if two bit vectors are equal.  */

bool
bits_are_equal (bit_vector *b, bit_vector *c)
{
    int full_words = b->num_bits / WORDBITS;
    int excess_bits = b->num_bits % WORDBITS;
    unsigned mask = (((unsigned)1) << excess_bits) - 1;
    int i;

    assert(b->num_bits == c->num_bits);

    /* check everything but the last word */
    for (i = 0; i < full_words; i++) {
	if (b->bits[i] != c->bits[i]) return false;
    }

    /* check the remaining bits */
    if (excess_bits) {
	if ((b->bits[full_words] & mask) != (c->bits[full_words] & mask)) {
	    return false;
	}
    }

    return true;
}



bool
bits_are_false (bit_vector *b)
{
    int full_words = b->num_bits / WORDBITS;
    int excess_bits = b->num_bits % WORDBITS;
    unsigned mask = (((unsigned)1) << excess_bits) - 1;
    int i;

    /* check everything but the last word */
    for (i = 0; i < full_words; i++) {
	if (b->bits[i] != 0) return false;
    }

    /* check the remaining bits */
    if (excess_bits) {
	if ((b->bits[full_words] & mask) != 0) {
	    return false;
	}
    }

    return true;
}


/* count bits of given value */

unsigned
count_bits (bit_vector *b, bool v)
{
  unsigned full_words = b->num_bits / WORDBITS;
  unsigned excess_bits = b->num_bits % WORDBITS;
  unsigned mask = (((unsigned)1) << excess_bits) - 1;
  unsigned i,x;
  unsigned count = 0;

  for (i=0; i<full_words;i++) {
    if (b->bits[i]) {
      for (x=0;x < WORDBITS;x++)
        if (((b->bits[i] & (1<<x)) >> x) == v)
          count++;
    }
  }

  /* check remaining bits */
  if (excess_bits) {
    if ((b->bits[full_words] & mask) != 0) {
      for (x=0;x < excess_bits;x++)
        if (((b->bits[i] & (1<<x)) >> x) == v)
          count++;
    }
  }

  return count;
}

/*  Print the elements of a bit vector that are true.  The bitprint argument
    specifies a function to print a single bit.  This allows you to print
    sensible output when the bits represent something other than integers.
    The fprint_bit function may be used to print them as integers.  */

void
fprint_bits (FILE *fd, bit_vector *b, bitprint_f bitprint)
{
    int i;

    for (i = 0; i < b->num_bits; i++) {
	if (get_bit(b, i)) {
	    fputc(' ', fd);
	    bitprint(fd, i);
	}
    }
}


void
fprint_bit (FILE *fd, int bit)
{
    fprintf(fd, "%d", bit);
}



/*  Deallocate a bit vector.  */

void
free_bit_vector (bit_vector *b)
{
    free(b->bits);
    free(b);
}

/**
 * Check if vector 'b' and 'c' differ by only one bit.
 * Returns false if b and c are the same or differ by more than
 * one '1'.
 */
bool
diff_by_one_bit(bit_vector *b, bit_vector *c) {
	int full_words = b->num_bits / WORDBITS;
	int excess_bits = b->num_bits % WORDBITS;
	unsigned mask = (((unsigned)1) << excess_bits) - 1;
	unsigned temp;
	int numDiffBits = 0;
	int i;

	assert(b->num_bits == c->num_bits);

    // Check everything but the last word.
    for (i = 0; i < full_words; i++) {
		// Perform XOR.
		temp = (b->bits[i] ^ c->bits[i]);
		if (temp == 0) continue;
		if (numDiffBits > 0) return false;
		// Check if it has more than one '1'.
		if (temp & (temp-1)) return false;
		numDiffBits = 1;
    }

    // Check the remaining bits.
    if (excess_bits) {
		// Perform XOR.
		temp = ((b->bits[i]&mask) ^ (c->bits[i]&mask));
		if (temp != 0) {
			if (numDiffBits > 0) return false;
			// Check if it has more than one '1'.
			if (temp & (temp-1)) return false;
			numDiffBits = 1;
		}
    }

	return numDiffBits;
}

/**
 * Check if 'b' contains 'c', i.e. for every bit 1 in 'c', the corresponding
 * bit in 'b' is also 1.
 */
bool
contains(bit_vector *b, bit_vector *c) {
	int full_words = b->num_bits / WORDBITS;
	int excess_bits = b->num_bits % WORDBITS;
	unsigned mask = (((unsigned)1) << excess_bits) - 1;
	int i;
	
	assert(b->num_bits == c->num_bits);

    // Check everything but the last word.
    for (i = 0; i < full_words; i++) {
		if (c->bits[i] & ~b->bits[i]) return false;
    }

    // Check the remaining bits.
    if (excess_bits) {
		if ((c->bits[i]&mask) & ~(b->bits[i]&mask)) return false;
	}

	return true;
}

/**
 * Print the given bit vector in a binary sequence.
 */
void
fprint_bit_vector(FILE *fd, bit_vector *b) {
    int i, limit = b->num_bits;
    for (i = 0; i < limit; ++i) fputc(get_bit(b, i) ? '1' : '0', fd);
}

/*** DAVID CODE END ***/
