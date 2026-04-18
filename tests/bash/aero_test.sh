#!/bin/bash

# Source approprite script (scripts/source_gnuz) and run below command:
# COMPILE_OP2=TRUE COMPILE_TESTS=TRUE RUN_TESTS=TRUE ./aero_test.sh

set -e

export TEST_APP="aero";

COMPILE_OP2=${COMPILE_OP2:-FALSE}
COMPILE_TESTS=${COMPILE_TESTS:-FALSE}
RUN_TESTS=${RUN_TESTS:-FALSE}

TEST_AERO_CPP=${TEST_AERO_CPP:-TRUE}
TEST_AERO_FORTRAN=${TEST_AERO_FORTRAN:-FALSE}

TEST_PLAIN=${TEST_PLAIN:-TRUE}
TEST_HDF5=${TEST_HDF5:-TRUE}

source ./test_core.sh

# Compile OP2 -----------------------------------------------------------------------------
if [[ "$COMPILE_OP2" = "TRUE" ]]; then
    echo "Compiling OP2..." | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    cd $LIB_LOC
    make clean; 
    make config; 
    make -j24;
fi

# Compile and run C++ aero plain tests --------------------------------------------------
if [[ "$TEST_AERO_CPP" = "TRUE" ]] && [[ "$TEST_PLAIN" = "TRUE" ]]; then

    cd $APPS_LOC/c/${TEST_APP}/aero_plain

    if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;

        if [[ -f FE_grid.dat ]]; then rm FE_grid.dat; fi
        wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/aero/FE_grid.dat
    fi

    if [[ "$RUN_TESTS" = "TRUE" ]]; then
        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "OMP_NUM_THREADS=6" "aero_openmp" "" "PASSED"
        validate "" "aero_cuda" "" "PASSED"
        validate "" "aero_c_cuda" "" "PASSED"
        validate "mpirun -np 16" "aero_par_mpi_seq" "" "PASSED"
        validate "mpirun -np 16" "aero_par_mpi_genseq" "" "PASSED"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "aero_par_mpi_openmp" "" "PASSED"
        validate "mpirun -np 4" "aero_par_mpi_cuda" "" "PASSED"
        validate "mpirun -np 4" "aero_par_mpi_c_cuda" "" "PASSED"
        validate "" "aero_seq" "" "PASSED"
        validate "" "aero_genseq" "" "PASSED"
    fi
fi

# Compile and run C++ aero HDF5 tests --------------------------------------------------
if [[ "$TEST_AERO_CPP" = "TRUE" ]] && [[ "$TEST_HDF5" = "TRUE" ]]; then

    cd $APPS_LOC/c/${TEST_APP}/aero_hdf5

    if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;

        if [[ -f FE_grid.h5 ]]; then rm FE_grid.h5; fi
        wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/aero/FE_grid.h5
    fi

    if [[ "$RUN_TESTS" = "TRUE" ]]; then
        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "mpirun -np 16" "aero_mpi_seq" "" "PASSED"
        validate "mpirun -np 16" "aero_mpi_genseq" "" "PASSED"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "aero_mpi_openmp" "" "PASSED"
        validate "mpirun -np 4" "aero_mpi_cuda" "" "PASSED"
        validate "mpirun -np 4" "aero_mpi_c_cuda" "" "PASSED"
        validate "OMP_NUM_THREADS=6" "aero_openmp" "" "PASSED"
        validate "" "aero_cuda" "" "PASSED"
        validate "" "aero_c_cuda" "" "PASSED"
        validate "" "aero_seq" "" "PASSED"
        validate "" "aero_genseq" "" "PASSED"
    fi
fi

# Compile and run Fortran aero tests --------------------------------------------------
# TODO : Need to implement Fortran versions of the aero apps first
if [ "$TEST_AERO_FORTRAN" = "TRUE" ]; then

    cd $APPS_LOC/fortran/${TEST_APP}/

    if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        # make clean; 
        # make;
    fi

    if [[ "$RUN_TESTS" = "TRUE" ]]; then
        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    fi

fi

if [[ "$RUN_TESTS" = "TRUE" ]]; then
    check_all_tests
fi