#!/bin/csh
#
# Set up the environment for running the hiCUDA compiler.
# NOTE: this script must be 'source'd.
#

set hicuda_root = "##HICUDA_ROOT_STUB##";

if ( ! -d "${hicuda_root}" ) then

    echo "${hicuda_root} does not exist! Abort.";

else

    setenv HICUDA_ROOT "${hicuda_root}";

    # Add 'bin' to PATH.
    set rmpath = "${hicuda_root}/bin/rmpath.csh";
    set tmp = "${hicuda_root}/bin";
    if ( -f "${rmpath}" ) then
        source "${rmpath}" "${tmp}";
    endif

    setenv PATH "${tmp}:${PATH}";

    unset rmpath tmp;

endif

unset hicuda_root;

