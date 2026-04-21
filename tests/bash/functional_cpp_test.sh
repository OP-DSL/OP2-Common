#!/bin/bash

# Source approprite script (scripts/source_gnuz) and run below command:
# COMPILE_OP2=TRUE COMPILE_TESTS=TRUE RUN_TESTS=TRUE ./functional_cpp_test.sh

set -e

export TEST_APP="functional_cpp";

COMPILE_OP2=${COMPILE_OP2:-FALSE}
COMPILE_TESTS=${COMPILE_TESTS:-FALSE}
RUN_TESTS=${RUN_TESTS:-FALSE}

TEST_CONSTS=${TEST_CONSTS:-TRUE}
TEST_DAT_REDUC=${TEST_DAT_REDUC:-TRUE}
TEST_ARG_GBL=${TEST_ARG_GBL:-TRUE}
TEST_STRIDES=${TEST_STRIDES:-TRUE}

source ./test_core.sh

# Compile OP2 -----------------------------------------------------------------------------
if [[ "$COMPILE_OP2" = "TRUE" ]]; then
    echo "Compiling OP2..." | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    cd $LIB_LOC
    if [ -f ../makefiles/.config.mk ]; then
      echo "Cleaning OP2..."
      make clean
    fi
    make config; 
    make -j24;
fi

# Compile and run const tests ------------------------------------------------------------
if [[ "$TEST_CONSTS" = "TRUE" ]]; then

    cd $SCRIPT_RUN_LOC/../functional/const

    if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;
    fi

    if [[ "$RUN_TESTS" = "TRUE" ]]; then
        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "OMP_NUM_THREADS=6" "const_tests_openmp" "" "passed"
        validate "" "const_tests_cuda" "" "passed"
        validate "" "const_tests_c_cuda" "" "passed"
        validate "mpirun -np 8" "const_tests_par_mpi_seq" "" "passed"
        validate "mpirun -np 8" "const_tests_par_mpi_genseq" "" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "const_tests_par_mpi_openmp" "" "passed"
        validate "mpirun -np 4" "const_tests_par_mpi_cuda" "" "passed"
        validate "mpirun -np 4" "const_tests_par_mpi_c_cuda" "" "passed"
        validate "" "const_tests_seq" "" "passed"
        validate "" "const_tests_genseq" "" "passed"

        validate "OMP_NUM_THREADS=6" "const_tests_soa_openmp" "" "passed"
        validate "" "const_tests_soa_cuda" "" "passed"
        validate "" "const_tests_soa_c_cuda" "" "passed"
        validate "mpirun -np 8" "const_tests_soa_par_mpi_seq" "" "passed"
        validate "mpirun -np 8" "const_tests_soa_par_mpi_genseq" "" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "const_tests_soa_par_mpi_openmp" "" "passed"
        validate "mpirun -np 4" "const_tests_soa_par_mpi_cuda" "" "passed"
        validate "mpirun -np 4" "const_tests_soa_par_mpi_c_cuda" "" "passed"
        validate "" "const_tests_soa_seq" "" "passed"
        validate "" "const_tests_soa_genseq" "" "passed"
    fi
fi

# Compile and run data reduction tests --------------------------------------------------
if [[ "$TEST_DAT_REDUC" = "TRUE" ]]; then

    cd $SCRIPT_RUN_LOC/../functional/dat_reductions

    if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;
    fi

    if [[ "$RUN_TESTS" = "TRUE" ]]; then
        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "OMP_NUM_THREADS=6" "reduc_tests_openmp" "" "passed"
        validate "" "reduc_tests_cuda" "" "passed"
        validate "" "reduc_tests_c_cuda" "" "passed"
        validate "mpirun -np 8" "reduc_tests_par_mpi_seq" "" "passed"
        validate "mpirun -np 8" "reduc_tests_par_mpi_genseq" "" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "reduc_tests_par_mpi_openmp" "" "passed"
        validate "mpirun -np 4" "reduc_tests_par_mpi_cuda" "" "passed"
        validate "mpirun -np 4" "reduc_tests_par_mpi_c_cuda" "" "passed"
        validate "" "reduc_tests_seq" "" "passed"
        validate "" "reduc_tests_genseq" "" "passed"

        validate "OMP_NUM_THREADS=6" "reduc_tests_soa_openmp" "" "passed"
        validate "" "reduc_tests_soa_cuda" "" "passed"
        validate "" "reduc_tests_soa_c_cuda" "" "passed"
        validate "mpirun -np 8" "reduc_tests_soa_par_mpi_seq" "" "passed"
        validate "mpirun -np 8" "reduc_tests_soa_par_mpi_genseq" "" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "reduc_tests_soa_par_mpi_openmp" "" "passed"
        validate "mpirun -np 4" "reduc_tests_soa_par_mpi_cuda" "" "passed"
        validate "mpirun -np 4" "reduc_tests_soa_par_mpi_c_cuda" "" "passed"
        validate "" "reduc_tests_soa_seq" "" "passed"
        validate "" "reduc_tests_soa_genseq" "" "passed"
    fi
fi

# Compile and run arg global tests ------------------------------------------------------
if [[ "$TEST_ARG_GBL" = "TRUE" ]]; then

    cd $SCRIPT_RUN_LOC/../functional/gbl

    if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;
    fi

    if [[ "$RUN_TESTS" = "TRUE" ]]; then
        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "OMP_NUM_THREADS=6" "gbl_tests_openmp" "" "passed"
        validate "" "gbl_tests_cuda" "" "passed"
        validate "" "gbl_tests_c_cuda" "" "passed"
        validate "mpirun -np 8" "gbl_tests_par_mpi_seq" "" "passed"
        validate "mpirun -np 8" "gbl_tests_par_mpi_genseq" "" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "gbl_tests_par_mpi_openmp" "" "passed"
        validate "mpirun -np 4" "gbl_tests_par_mpi_cuda" "" "passed"
        validate "mpirun -np 4" "gbl_tests_par_mpi_c_cuda" "" "passed"
        validate "" "gbl_tests_seq" "" "passed"
        validate "" "gbl_tests_genseq" "" "passed"

        validate "OMP_NUM_THREADS=6" "gbl_tests_soa_openmp" "" "passed"
        validate "" "gbl_tests_soa_cuda" "" "passed"
        validate "" "gbl_tests_soa_c_cuda" "" "passed"
        validate "mpirun -np 8" "gbl_tests_soa_par_mpi_seq" "" "passed"
        validate "mpirun -np 8" "gbl_tests_soa_par_mpi_genseq" "" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "gbl_tests_soa_par_mpi_openmp" "" "passed"
        validate "mpirun -np 4" "gbl_tests_soa_par_mpi_cuda" "" "passed"
        validate "mpirun -np 4" "gbl_tests_soa_par_mpi_c_cuda" "" "passed"
        validate "" "gbl_tests_soa_seq" "" "passed"
        validate "" "gbl_tests_soa_genseq" "" "passed"
    fi
fi

# Compile and run stride tests ------------------------------------------------------
if [[ "$TEST_STRIDES" = "TRUE" ]]; then

    cd $SCRIPT_RUN_LOC/../functional/strides

    if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;
    fi

    if [[ "$RUN_TESTS" = "TRUE" ]]; then
        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "OMP_NUM_THREADS=6" "stride_tests_openmp" "" "passed"
        validate "" "stride_tests_cuda" "" "passed"
        validate "" "stride_tests_c_cuda" "" "passed"
        validate "mpirun -np 8" "stride_tests_par_mpi_seq" "" "passed"
        validate "mpirun -np 8" "stride_tests_par_mpi_genseq" "" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "stride_tests_par_mpi_openmp" "" "passed"
        validate "mpirun -np 4" "stride_tests_par_mpi_cuda" "" "passed"
        validate "mpirun -np 4" "stride_tests_par_mpi_c_cuda" "" "passed"
        validate "" "stride_tests_seq" "" "passed"
        validate "" "stride_tests_genseq" "" "passed"

        validate "OMP_NUM_THREADS=6" "stride_tests_soa_openmp" "" "passed"
        validate "" "stride_tests_soa_cuda" "" "passed"
        validate "" "stride_tests_soa_c_cuda" "" "passed"
        validate "mpirun -np 8" "stride_tests_soa_par_mpi_seq" "" "passed"
        validate "mpirun -np 8" "stride_tests_soa_par_mpi_genseq" "" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "stride_tests_soa_par_mpi_openmp" "" "passed"
        validate "mpirun -np 4" "stride_tests_soa_par_mpi_cuda" "" "passed"
        validate "mpirun -np 4" "stride_tests_soa_par_mpi_c_cuda" "" "passed"
        validate "" "stride_tests_soa_seq" "" "passed"
        validate "" "stride_tests_soa_genseq" "" "passed"
    fi
fi

if [[ "$RUN_TESTS" = "TRUE" ]]; then
    check_all_tests
fi