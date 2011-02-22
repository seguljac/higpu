#!/bin/bash
#
# Clean up the environment for running the hiCUDA compiler.
# NOTE: this script must be 'source'd.
#

if [ -d ${HICUDA_ROOT} ]; then

    # Remove 'bin' from PATH.
    RMPATH="${HICUDA_ROOT}/bin/rmpath.sh";
    TMP="${HICUDA_ROOT}/bin";
    if [ -f "${RMPATH}" ]; then
        source ${RMPATH} ${TMP};
    fi

    unset HICUDA_ROOT;

else

    echo "HICUDA_ROOT is corrupted! Abort.";

fi

