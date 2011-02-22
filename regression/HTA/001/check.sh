#!/bin/bash
#
# For this test program, the compiler should generate a specific error msg.
#

../../common/check-err.sh HTA001 "dimension #0 of array <a> must be a constant
or a scalar variable"
exit $?

