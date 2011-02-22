#!/bin/bash
#
# Set up the environment for running the hiCUDA compiler.
# NOTE: this script must be 'source'd.
#

HICUDA_ROOT="##HICUDA_ROOT_STUB##";

if [ ! -d "${HICUDA_ROOT}" ]; then

    echo "${HICUDA_ROOT} does not exist! Abort.";

else

    export HICUDA_ROOT;

    # Add 'bin' to PATH.
    RMPATH="${HICUDA_ROOT}/bin/rmpath.sh";
    TMP="${HICUDA_ROOT}/bin";
    if [ -f "${RMPATH}" ]; then
        source ${RMPATH} ${TMP};
    fi
    export PATH="${TMP}:${PATH}";

fi

