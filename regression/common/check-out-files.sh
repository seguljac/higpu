#!/bin/bash
#
# Generic check script for test cases where the compiler is expected to
# correctly generate the CUDA files.
#
# The script compares the output CUDA files with the reference ones.
#
# Parameter 1: TEST_NAME
# Parameter 2 and on: explicit spec of source files
#                     (if omitted, a single file TEST_NAME.c is assumed).
#

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <TEST_NAME> [<source file list>]"
    exit 1
fi

TEST_NAME=$1

PROGS=""
shift
while (($#)); do
    PROGS="${PROGS} $1"
    shift
done
if [[ "x${PROGS}" == "x" ]]; then
    PROGS="${TEST_NAME}.c"
fi

HICUDA_OPT=
hicuda -o ${TEST_NAME} ${HICUDA_OPT} ${PROGS} &> /dev/null
EXIT_CODE=$?

echo -n "${TEST_NAME}: "

# Check if there is an output directory.
HC_OUT_DIR="${TEST_NAME}.cuda"
if [[ ! -e "${HC_OUT_DIR}" ]]; then
    echo "FAIL (no output directory)"
    exit 1
fi

# Check the exit code.
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "FAIL (exit code = ${EXIT_CODE})"
    rm -r ${HC_OUT_DIR}*
    exit 1
fi

# Determine if this machine is 32-bit or 64-bit.
# We may have different reference outputs for different machine targets.
MACHINE_TYPE=`uname -m | sed -e s/i.86/i386/`
if [[ "${MACHINE_TYPE}" == "i386" ]]; then
    WORD_SZ=32
else
    WORD_SZ=64
fi

# Compare the header file against the reference.
H_FILE=${HC_OUT_DIR}/${TEST_NAME}.cu.h
H_FILE_REF=${TEST_NAME}.cu.h.ref
if [[ ! -f "${H_FILE_REF}" ]]; then
    H_FILE_REF=${H_FILE_REF}.${WORD_SZ}
fi
diff ${H_FILE} ${H_FILE_REF} > /dev/null
EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "FAIL (header file)"
    rm -r ${HC_OUT_DIR}*
    exit 1
fi

# Compare the CUDA file against the reference.
CU_FILE=${HC_OUT_DIR}/${TEST_NAME}.cu
CU_FILE_REF=${TEST_NAME}.cu.ref
if [[ ! -f "${CU_FILE_REF}" ]]; then
    CU_FILE_REF=${CU_FILE_REF}.${WORD_SZ}
fi
# First, remove the first 3 lines of each file (that contains the timestamp).
sed '1,3d' ${CU_FILE} > ${CU_FILE}.tmp
sed '1,3d' ${CU_FILE_REF} > ${CU_FILE_REF}.tmp
diff ${CU_FILE}.tmp ${CU_FILE_REF}.tmp > /dev/null
EXIT_CODE=$?
rm ${CU_FILE}.tmp ${CU_FILE_REF}.tmp
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "FAIL (CUDA file)"
    rm -r ${HC_OUT_DIR}*
    exit 1
fi

echo "PASS"

rm -r ${HC_OUT_DIR}*

exit 0
