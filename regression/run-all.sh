#!/bin/bash
#
# Run all regression tests.
#

config_direction()
{
    echo "You must configure the hiCUDA compiler environment."
    echo "(see README.TXT in the source directory)"
    echo
}

# Is HICUDA_ROOT defined?
if [[ "x${HICUDA_ROOT}" == "x" ]]; then
    echo
    echo "*** HICUDA_ROOT is not defined."
    echo
    config_direction
    exit 1
fi

# Is "hicuda" in PATH?
HICUDA_PATH=`which hicuda`
EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo
    echo "*** The hiCUDA compiler is not in PATH."
    echo
    config_direction
    exit 1
fi

echo
echo "Compiler path: ${HICUDA_PATH}"
echo
echo "Start regression tests ..."
echo "----------------------------------"

PASS_COUNT=0
FAIL_COUNT=0

for HT_DIR in HT*; do
    if [[ -d "${HT_DIR}" ]]; then
        cd ${HT_DIR}
        for TEST_DIR in *; do
            if [[ -d "${TEST_DIR}" ]]; then
                cd ${TEST_DIR}
                if [[ -f "./check.sh" ]]; then
                    ./check.sh
                    EXIT_CODE=$?
                    if [[ ${EXIT_CODE} -eq 0 ]]; then
                        PASS_COUNT=$((PASS_COUNT+1))
                    else
                        FAIL_COUNT=$((FAIL_COUNT+1))
                    fi
                fi
                cd ..
            fi
        done
        cd ..
        echo "----------------------------------"
    fi
done

echo "${PASS_COUNT} tests passed, ${FAIL_COUNT} tests failed."
echo

exit 0
