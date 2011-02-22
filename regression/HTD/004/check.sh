#!/bin/bash
#
# For this test program, the compiler should generate a specific error msg.
#

../../common/check-err.sh HTD004 "Potential kernel nesting in procedure <fooA>"
exit $?

