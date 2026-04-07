#!/bin/bash

set -e

export TEST_APP="functional";

COMPILE_OP2=${COMPILE_OP2:-FALSE}

TEST_CONSTS=${TEST_CONSTS:-TRUE}
TEST_DAT_REDUC=${TEST_DAT_REDUC:-TRUE}
TEST_ARG_GBL=${TEST_ARG_GBL:-TRUE}

source ./test_core.sh

if [[ "$COMPILE_OP2" = "TRUE" ]]; then
    echo "Compiling OP2..." | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    cd $LIB_LOC
    make clean; 
    make config; 
    make -j24;
fi

if [[ "$TEST_CONSTS" = "TRUE" ]]; then

    cd $SCRIPT_RUN_LOC/../${TEST_APP}/const

    echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    
    make clean; 
    make;

    echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    validate "OMP_NUM_THREADS=6" "const_tests_openmp" "" "passed"
    validate "" "const_tests_cuda" "" "passed"
    validate "mpirun -np 8" "const_tests_par_mpi_seq" "" "passed"
    validate "mpirun -np 8" "const_tests_par_mpi_genseq" "" "passed"
    validate "OMP_NUM_THREADS=6 mpirun -np 8" "const_tests_par_mpi_openmp" "" "passed"
    validate "mpirun -np 4" "const_tests_par_mpi_cuda" "" "passed"
    validate "" "const_tests_seq" "" "passed"
    validate "" "const_tests_genseq" "" "passed"
fi


if [[ "$TEST_DAT_REDUC" = "TRUE" ]]; then

    cd $SCRIPT_RUN_LOC/../${TEST_APP}/dat_reductions

    echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    
    make clean; 
    make;

    echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    validate "OMP_NUM_THREADS=6" "reduc_tests_openmp" "" "passed"
    validate "" "reduc_tests_cuda" "" "passed"
    validate "mpirun -np 8" "reduc_tests_par_mpi_seq" "" "passed"
    validate "mpirun -np 8" "reduc_tests_par_mpi_genseq" "" "passed"
    validate "OMP_NUM_THREADS=6 mpirun -np 8" "reduc_tests_par_mpi_openmp" "" "passed"
    validate "mpirun -np 4" "reduc_tests_par_mpi_cuda" "" "passed"
    validate "" "reduc_tests_seq" "" "passed"
    validate "" "reduc_tests_genseq" "" "passed"
fi


if [[ "$TEST_ARG_GBL" = "TRUE" ]]; then

    cd $SCRIPT_RUN_LOC/../${TEST_APP}/gbl

    echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    
    make clean; 
    make;

    echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    validate "OMP_NUM_THREADS=6" "gbl_tests_openmp" "" "passed"
    validate "" "gbl_tests_cuda" "" "passed"
    validate "mpirun -np 8" "gbl_tests_par_mpi_seq" "" "passed"
    validate "mpirun -np 8" "gbl_tests_par_mpi_genseq" "" "passed"
    validate "OMP_NUM_THREADS=6 mpirun -np 8" "gbl_tests_par_mpi_openmp" "" "passed"
    validate "mpirun -np 4" "gbl_tests_par_mpi_cuda" "" "passed"
    validate "" "gbl_tests_seq" "" "passed"
    validate "" "gbl_tests_genseq" "" "passed"
fi


check_all_tests
