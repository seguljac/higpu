#!/bin/csh
#
# Clean up the environment for running the hiCUDA compiler.
# NOTE: this script must be 'source'd.
#

if ( -d "${HICUDA_ROOT}" ) then

    # Remove 'bin' from PATH.
    set rmpath = "${HICUDA_ROOT}/bin/rmpath.csh";
    set tmp = "${HICUDA_ROOT}/bin";
    if ( -f "${rmpath}" ) then
        source "${rmpath}" "${tmp}";
    endif

    unset rmpath tmp;
    unsetenv HICUDA_ROOT;

else

    echo "HICUDA_ROOT is corrupted! Abort.";

endif

