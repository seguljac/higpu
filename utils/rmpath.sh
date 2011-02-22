#!/bin/bash
#
# Remove a given path from a given environment variable (e.g. PATH,
# LD_LIBRARY_PATH), in the current shell.
# NOTE: this script must be run using 'source'.
#
# Revision: 1.1.0
# Last updated: 08/06/15
#

HELP="source rmpath <path> [<env_var>]";
DESC="Remove a path from an environment variable
      (e.g. PATH, LD_LIBRARY_PATH).\nIf <env_var> is not specified,
      PATH is assumed."

if [[ $# != 1 && $# != 2 ]]; then
    echo -e ${DESC};
    echo ${HELP};
else
    ENVVAR="PATH";
    if [[ "x${2}" != "x" ]]; then
        ENVVAR=${2};
    fi

    # Three replace commands (middle, front, end)
    # NOTE: the order matters.
    CMD1="s;:${1}:;:;g";
    CMD2="s;^${1}:;;g";
    CMD3="s;:${1}^;;g";
    CMD4="s;^${1}$;;";

    CMD="echo \${${ENVVAR}} |
         sed -e '${CMD1}' -e '${CMD2}' -e '${CMD3}' -e '${CMD4}'";
    # echo ${CMD};

    export ${ENVVAR}=`eval ${CMD}`;
fi
