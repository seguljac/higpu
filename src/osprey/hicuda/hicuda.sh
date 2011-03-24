#! /bin/bash
#
# Driver script for the hicuda compiler.
#
# Last updated: 09/11/07
#

# Make sure that we have HICUDA_ROOT set.
if [ ! -d "${HICUDA_ROOT}" ]; then
    echo "HICUDA_ROOT is either not set or invalid! Abort."
    exit 1
fi

######################################################################

OPT_BASE_NAME="-o"
OPT_INC_DIR="-I"
OPT_KEEP_STR="-keep"
OPT_VERBOSE_STR="-v"
OPT_ABI32_STR="-m32"
OPT_ABI64_STR="-m64"
OPT_EMIT_OPENCL_STR="-opencl"

usage()
{
    echo "$0"
    echo "    [-o <base name>] [-I<include path>]*" \
         "[${OPT_ABI32_STR}|${OPT_ABI64_STR}]" \
         "[${OPT_KEEP_STR}] [${OPT_VERBOSE_STR}] [${OPT_EMIT_OPENCL_STR}]"
    echo "    <input file(s)>"
    echo
}

HICUDA_WEBSITE="http://sourceforge.net/projects/hicuda/"

# Parse the arguments.
BASE_NAME=""
INPUT_FILES=""
INC_DIRS=""
OPT_ABI_32=1
if [ `uname -m` == "x86_64" ]; then
    OPT_ABI_32=0
fi
OPT_KEEP=0
OPT_VERBOSE=0
OPT_EMIT_OPENCL=0

WARNING=0
EXPECT_BASE_NAME=0  # 0: not seen -o, 1: seen -o, 2: seen BASE_NAME
while [ "x" != "x$1" ]; do
    if [[ ${1:0:1} != "-" ]]; then
        if [ "${EXPECT_BASE_NAME}" -eq "1" ]; then
            BASE_NAME=$1
            EXPECT_BASE_NAME=2
        elif [ -f "$1" ]; then
            INPUT_FILES="${INPUT_FILES} $1"
        else
            echo "*** Unknown argument: $1"
            WARNING=1
        fi
    else
        # This is an option.
        case "$1" in
        "${OPT_KEEP_STR}" ) OPT_KEEP=1;
        ;;
        "${OPT_VERBOSE_STR}" ) OPT_VERBOSE=1;
        ;;
        "${OPT_ABI32_STR}" ) OPT_ABI_32=1
        ;;
        "${OPT_ABI64_STR}" ) OPT_ABI_32=0
        ;;
        "${OPT_EMIT_OPENCL_STR}" ) OPT_EMIT_OPENCL=1
        ;;
        "${OPT_BASE_NAME}" )
            if [ "${EXPECT_BASE_NAME}" -eq "0" ]; then
                EXPECT_BASE_NAME=1
            else
                echo
                echo "*** Duplicate -o options!"
                echo
                usage
                exit 1
            fi
        ;;
        * ) if [ ${1:0:2} == "${OPT_INC_DIR}" ]; then
                INC_DIRS="${INC_DIRS} $1";
            else
                echo "Unknown option: $1";
                WARNING=1;
            fi
        esac
    fi

    shift
done

# Ensure argument completeness.
if [ "${EXPECT_BASE_NAME}" -eq "1" ]; then
    echo;
    echo "*** Missing base name after ${OPT_BASE_NAME}!";
    echo;
    usage;
    exit 1;
fi

# Make sure that we have at least one input file.
if [ "x" == "x${INPUT_FILES}" ]; then
    echo;
    echo "*** Missing input file!";
    echo;
    usage;
    exit 1;
fi

if [ "${WARNING}" -eq "1" ]; then
    echo;
    usage;
fi

if [ "x" == "x${BASE_NAME}" ]; then
    BASE_NAME="a.out"
fi

# output directory of WHIRL2C (used later)
IPA_TMPDIR="${BASE_NAME}.ipakeep/"

###########################################################################

# 1st parameter: INPUT_FILES
# 2nd parameter: IPA_TMPDIR
#
cleanup_src()
{
    # <input_file>.o <input_file>.B <input_file>.i
    for f in ${INPUT_FILES}; do
        f_base_name=${f%.*};
        rm -f ${f_base_name}.o ${f_base_name}.B ${f_base_name}.i
    done

    # IPA temp directory.
    rm -rf ${IPA_TMPDIR}
}

# Prepare opencc environment.
export TOOLROOT="${HICUDA_ROOT}";
export PATH="${HICUDA_ROOT}/bin:${PATH}";

###########################################################################
#
# Pre-process the source files.
#
###########################################################################

# Extract all standard includes from the source files.
TMP_FILE=`mktemp`
awk '{if (($1 ~ /^#include$/) && ($2 ~ /^<.*\.h>/)) {print $1 " " $2}}' \
        ${INPUT_FILES} > ${TMP_FILE}
# Remove any ^M and ^Z characters just in case this source comes from Windows.
sed -i 's/'"$(printf '\015')"'$//' ${TMP_FILE}
sed -i 's/'"$(printf '\032')"'$//' ${TMP_FILE}

# Add two that we always want to include because they are in whirl2c.h.
echo "#include <math.h>" >> ${TMP_FILE}
echo "#include <string.h>" >> ${TMP_FILE}

# Remove duplicates from the include list.
BASE_INC_NAME="${BASE_NAME}_$$"
TMP_INC_FILE="/tmp/${BASE_INC_NAME}.c"
awk '{\
    if (!($0 in stored_lines)) {\
        print;\
        stored_lines[$0] = 1;\
    }}' ${TMP_FILE} > ${TMP_INC_FILE}
rm ${TMP_FILE}

# Compile the include list file into WHIRL using opencc in the installation
# directory of the hiCUDA compiler.
OPENCC_FLAGS="-D_BSD_SOURCE -std=c99 -D_POSIX_C_SOURCE=200112L -fe -keep -keep-all-types -gnu3"
if [ ${OPT_ABI_32} -eq 1 ]; then
    OPENCC_FLAGS="-m32 ${OPENCC_FLAGS}"
else
    OPENCC_FLAGS="-m64 ${OPENCC_FLAGS}"
fi
if [ ${OPT_EMIT_OPENCL} -eq 1 ]; then
    OPENCC_FLAGS="-opencl ${OPENCC_FLAGS}"
fi

# These are the two files to be produced.
TMP_INC_B="${BASE_INC_NAME}.B"
TMP_INC_I="${BASE_INC_NAME}.i"

if [[ ${OPT_VERBOSE} -eq 1 ]]; then
    echo "opencc ${INC_DIRS} ${OPENCC_FLAGS} ${TMP_INC_FILE}"
fi
opencc ${INC_DIRS} ${OPENCC_FLAGS} ${TMP_INC_FILE}
EXIT_CODE=$?

# Immediately remove the files that are not needed anymore.
rm -f ${TMP_INC_I}

if [ ${EXIT_CODE} -ne 0 ]; then
    echo
    echo "=================================================================="
    echo "    Failed to pre-process the source files."
    echo "    The error message may be shown above, starting with ###."
    echo
    echo "    Please report this problem to:"
    echo "          ${HICUDA_WEBSITE}"
    echo "=================================================================="
    echo

    rm -f ${TMP_INC_FILE} ${TMP_INC_B}
    exit ${EXIT_CODE}
fi

###########################################################################
#
# Invoke opencc
#
###########################################################################

IPA_FLAGS="-IPA:array_summary=on -IPA:compile=off -IPA:link=off"
IPA_FLAGS="${IPA_FLAGS} -IPA:inc_B=${TMP_INC_B}"
if [ ${OPT_EMIT_OPENCL} -eq 1 ]; then
    IPA_FLAGS="-IPA:opencl=on ${IPA_FLAGS}"
fi
OPENCC_FLAGS="-keep -hicuda -gnu3 -ipa ${IPA_FLAGS}"
OPENCC_FLAGS="${OPENCC_FLAGS} -o ${BASE_NAME}"
if [ ${OPT_VERBOSE} -eq 1 ]; then
    OPENCC_FLAGS="${OPENCC_FLAGS} -show -vhc"
fi
if [ ${OPT_ABI_32} -eq 1 ]; then
    OPENCC_FLAGS="-m32 ${OPENCC_FLAGS}"
else
    OPENCC_FLAGS="-m64 ${OPENCC_FLAGS}"
fi
if [ ${OPT_EMIT_OPENCL} -eq 1 ]; then
    OPENCC_FLAGS="-opencl ${OPENCC_FLAGS}"
fi

# Incorporate the standard include paths for GCC on the local system.
# This allows the custom C preprocessor included in the hiCUDA compiler to
# behave the same way as the local C preprocessor (if any).
#
OPENCC_INC_DIRS="${INC_DIRS}"
TMP_FILE=`mktemp`
which cpp > /dev/null
if [ $? -eq 0 ]; then
    cpp -v < /dev/null >& ${TMP_FILE}
fi
INC_START=0
while read LINE
do
    if [ ${INC_START} -eq 0 ]; then
        if [ "${LINE:0:8}" == "#include" ]; then
            INC_START=1
        fi
    else
        if [ "${LINE:0:3}" == "End" ]; then
            INC_START=0
        fi
    fi

    if [ ${INC_START} -eq 1 ]; then
        if [ "${LINE:0:1}" == "/" ]; then
            OPENCC_INC_DIRS="${OPENCC_INC_DIRS} -I${LINE}"
        fi
    fi
done < ${TMP_FILE}
rm ${TMP_FILE}

if [[ ${OPT_VERBOSE} -eq 1 ]]; then
    echo "opencc ${OPENCC_INC_DIRS} ${OPENCC_FLAGS} ${INPUT_FILES}"
fi
opencc ${OPENCC_INC_DIRS} ${OPENCC_FLAGS} ${INPUT_FILES}
EXIT_CODE=$?

# Remove the empty executable file generated.
if [[ -f ${BASE_NAME} && ! -s ${BASE_NAME} ]]; then
    rm ${BASE_NAME}
fi

# Remove the include header WHIRL file generated.
if [[ ${OPT_KEEP} -eq 0 ]]; then
    rm ${TMP_INC_B}
fi

if [ ${EXIT_CODE} -ne 0 ]; then
    echo
    echo "=================================================================="
    echo "    CUDA code generation FAILED (in opencc)."
    echo "    The error message should be right above, starting with ###."
    echo
    echo "    If you believe that there is a bug in the hiCUDA compiler, "
    echo "    please submit a report to:"
    echo "          ${HICUDA_WEBSITE}"
    echo "=================================================================="
    echo

    if [ ${OPT_KEEP} -eq 0 ]; then
        cleanup_src ${INPUT_FILES} ${IPA_TMPDIR}
    fi
    exit ${EXIT_CODE}
fi

###########################################################################
#
# Invoke whirl2c
#
###########################################################################

cd ${IPA_TMPDIR}

WHIRL2C="${HICUDA_ROOT}/lib/gcc-lib/x86_64-open64-linux/4.1/whirl2c"

ABI_NAME="n32"
if [ ${OPT_ABI_32} -eq 0 ]; then
    ABI_NAME="n64"
fi
WHIRL2C_FLAGS="-TARG:abi=${ABI_NAME} -TENV:read_global=symtab.I"
if [[ ${OPT_VERBOSE} -eq 0 ]]; then
    WHIRL2C_FLAGS="${WHIRL2C_FLAGS} -CLIST:show=off"
fi
if [[ ${OPT_EMIT_OPENCL} -eq 1 ]]; then
    WHIRL2C_FLAGS="${WHIRL2C_FLAGS} -CLIST:emit_opencl=on"
fi

# Remove from the header list two header files that are already in whirl2c.h.
sed -i '/#include <math.h>/d' ${TMP_INC_FILE}
sed -i '/#include <string.h>/d' ${TMP_INC_FILE}

WHIRL2C_FLAGS="${WHIRL2C_FLAGS} -CLIST:inc_file=${TMP_INC_FILE} -td1024"
WHIRL2C_FLAGS="${WHIRL2C_FLAGS} -fB,1.I"

if [[ ${OPT_VERBOSE} -eq 1 ]]; then
    echo
    echo "${WHIRL2C} ${WHIRL2C_FLAGS} ${BASE_NAME}.c"
fi
${WHIRL2C} ${WHIRL2C_FLAGS} ${BASE_NAME}.c
EXIT_CODE=$?

# Remove the temp include header file.
if [[ ${OPT_KEEP} -eq 0 ]]; then
    rm ${TMP_INC_FILE}
fi

# Must do so before error handling.
cd ..

if [ ${EXIT_CODE} -ne 0 ]; then
    echo
    echo "=================================================================="
    if [ ${OPT_EMIT_OPENCL} -eq 1 ]; then
	echo "    OpenCL code generation FAILED (in whirl2c)."
    else
	echo "    CUDA code generation FAILED (in whirl2c)."
    fi
    echo "    The error message should be right above, starting with ###."
    echo
    echo "    If you believe that there is a bug in the hiCUDA compiler, "
    echo "    please submit a report to:"
    echo "          ${HICUDA_WEBSITE}"
    echo "=================================================================="
    echo

    if [ ${OPT_KEEP} -eq 0 ]; then
        cleanup_src ${INPUT_FILES} ${IPA_TMPDIR}
    fi
    exit ${EXIT_CODE}
fi

# Move the trace file for WHIRL2C to the current directory.
if [[ ${OPT_VERBOSE} -eq 1 ]]; then
    mv ${IPA_TMPDIR}/${BASE_NAME}.t ${BASE_NAME}.w2c.t
fi

###########################################################################
#
# Place the output files in the appropriate location.
#
###########################################################################

if [ ${OPT_EMIT_OPENCL} -eq 1 ]; then
    OUTPUT_DIR_BASE="${BASE_NAME}.higpu_opencl"
else
    OUTPUT_DIR_BASE="${BASE_NAME}.higpu_cuda"
fi

# OUTPUT_DIR is a symbolic link to the actual folder named
#       OUTPUT_DIR.OUTPUT_DIR_VER
# This allows the user to compile a hiCUDA program multiple times without
# removing the output folder.
OUTPUT_DIR_VER=0
while [ -d "${OUTPUT_DIR_BASE}.${OUTPUT_DIR_VER}" ]; do
    let OUTPUT_DIR_VER+=1
done
OUTPUT_DIR="${OUTPUT_DIR_BASE}.${OUTPUT_DIR_VER}"

mkdir ${OUTPUT_DIR}

# Move the .cu and .cu.h files to the target directory.
if [ ${OPT_EMIT_OPENCL} -eq 1 ]; then
    mv ${IPA_TMPDIR}/${BASE_NAME}.cu ${OUTPUT_DIR}/${BASE_NAME}.c
    mv ${IPA_TMPDIR}/${BASE_NAME}.cu.h ${OUTPUT_DIR}/${BASE_NAME}.h
    mv ${IPA_TMPDIR}/*.cl ${OUTPUT_DIR}
    mv ${IPA_TMPDIR}/*.cl.h ${OUTPUT_DIR}
else
    mv ${IPA_TMPDIR}/${BASE_NAME}.cu ${OUTPUT_DIR}
    mv ${IPA_TMPDIR}/${BASE_NAME}.cu.h ${OUTPUT_DIR}
fi

# Copy the common whirl2c.h to the target directory.
cp ${HICUDA_ROOT}/misc/whirl2c.h ${OUTPUT_DIR}

if [ ${OPT_EMIT_OPENCL} -eq 1 ]; then
    # Copy the common whirl2c_OpenCL.h to the target directory.
    cp ${HICUDA_ROOT}/misc/whirl2c_OpenCL.h ${OUTPUT_DIR}
fi

# If OUTPUT_DIR_BASE does not exist or is just a symbolic link, let it point
# to the the real output directory.
if [ -h "${OUTPUT_DIR_BASE}" ]; then
    rm ${OUTPUT_DIR_BASE}
fi
if [ ! -e "${OUTPUT_DIR_BASE}" ]; then
    ln -s ${OUTPUT_DIR} ${OUTPUT_DIR_BASE}
    OUTPUT_DIR=${OUTPUT_DIR_BASE}
fi

echo
echo "================================================================="
if [ ${OPT_EMIT_OPENCL} -eq 1 ]; then
    echo "    Successful OpenCL code generation. Output files in: "
else
    echo "    Successful CUDA code generation. Output files in: "
fi
echo "        ${OUTPUT_DIR}"
echo "================================================================="
echo

###########################################################################
#
# Clean up the intermediate files (if -keep is not present).
#
###########################################################################

if [ ${OPT_KEEP} -eq 0 ]; then
    cleanup_src ${INPUT_FILES} ${IPA_TMPDIR}
fi

exit 0

