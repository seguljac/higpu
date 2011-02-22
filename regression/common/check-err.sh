#!/bin/bash
#
# Generic check script for test cases in which the compiler should generate a
# specific error message.
#
# Parameter(s): TEST_CASE ERRMSG
#

TEST_NAME=$1
PROG=${TEST_NAME}.c
ERRMSG=$2

HICUDA_OPT=
TMP_FILE="tmp.$$"
hicuda ${HICUDA_OPT} ${PROG} &> ${TMP_FILE}

grep "${ERRMSG}" ${TMP_FILE} > /dev/null
EXIT_CODE=$?

echo -n "${TEST_NAME}: "
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "PASS"
else
    echo "FAIL"
fi

rm ${TMP_FILE}

exit ${EXIT_CODE}
