#!/bin/bash
#
# For this test program, the compiler should correctly generate the CUDA files.
#

../../common/check-out-files.sh HTF009 func1.c func2.c HTF009.c
exit $?
