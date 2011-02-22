#!/bin/bash
#
# For this test program, the compiler should generate a specific error msg.
#

../../common/check-err.sh HTF001 "Missing the kernel_end directive"
exit $?

