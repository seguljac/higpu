#!/bin/bash
#
# For this test program, the compiler should generate a specific error msg.
#

../../common/check-err.sh HTA004 "Mismatch in dimensionality"
exit $?

