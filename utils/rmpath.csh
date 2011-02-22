#!/bin/csh
#
# Remove a given path from the environment variable PATH
# in the current C shell.
#
# NOTE: this script must be 'source'd.
#
# Revision: 1.1.1
# Last updated: 09/10/21
#

if ( $#argv != 1 ) then
    echo "Remove a path from the environment variable PATH."
    echo "source rmpath.csh <path>"
else
    set target_path = "${1}"
    # Is this a valid directory?
    if ( ! -d "${target_path}" ) then
        echo "${target_path} is not a valid directory! Abort."
    else
        # Turn this path into an absolute one.
        set curr_dir = "${cwd}"
        cd "${target_path}"
        set target_path = "`pwd`"

        # Go through each path in PATH.
        set new_path = ()
        foreach p ($path)
            set insert = 1
            if ( "${p}" != "." && -d "${p}" ) then
                # Turn this path into an absolute one.
                cd "${p}"
                set abs_p = "`pwd`"
                if ( "${abs_p}" == "${target_path}" ) then
                    set insert = 0
                endif
            endif

            if ( ${insert} == 1 ) then
                set new_path = ( ${new_path} "${p}" )
            endif
        end

        set path = ( ${new_path} )

        cd "${curr_dir}"

        unset curr_dir new_path insert p abs_p
    endif

    unset target_path

    # Three replace commands (middle, front, end)
    # NOTE: the order matters.
    # set target_path = "${1}";
    # set CMD1 = "s;:${target_path}:;:;g";
    # set CMD2 = "s;^${target_path}:;;g";
    # set CMD3 = "s;:${target_path}^;;g";
    # set CMD4 = '$';
    # set CMD4 = "s;^${target_path}${CMD4};;";
    # set CMD="echo ${PATH} | sed -e '${CMD1}' -e '${CMD2}' -e '${CMD3}' -e '${CMD4}'";
    # echo ${CMD};

    # setenv PATH `eval ${CMD}`;

    # unset CMD1 CMD2 CMD3 CMD4 CMD;
endif

