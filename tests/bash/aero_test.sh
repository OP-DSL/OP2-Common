#!/bin/bash

set -e

export TEST_APP="aero";

COMPILE_OP2=${COMPILE_OP2:-FALSE}

TEST_AERO_CPP=${TEST_AERO_CPP:-TRUE}
TEST_AERO_FORTRAN=${TEST_AERO_FORTRAN:-FALSE}

TEST_PLAIN=${TEST_PLAIN:-TRUE}
TEST_HDF5=${TEST_HDF5:-TRUE}

source ./test_core.sh

if [[ "$COMPILE_OP2" = "TRUE" ]]; then
    echo "Compiling OP2..." | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    cd $LIB_LOC
    make clean; 
    make config; 
    make -j24;
fi

if [[ "$TEST_AERO_CPP" = "TRUE" ]] && [[ "$TEST_PLAIN" = "TRUE" ]]; then

    cd $APPS_LOC/c/${TEST_APP}/aero_plain

    echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    
    make clean; 
    make;

    if [[ -f FE_grid.dat ]]; then rm FE_grid.dat; fi
    wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/aero/FE_grid.dat

    echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    validate "OMP_NUM_THREADS=32" "aero_openmp" "" "PASSED"
    validate "" "aero_cuda" "" "PASSED"
    validate "mpirun -np 64" "aero_par_mpi_seq" "" "PASSED"
    validate "mpirun -np 64" "aero_par_mpi_genseq" "" "PASSED"
    validate "OMP_NUM_THREADS=8 mpirun -np 8" "aero_par_mpi_openmp" "" "PASSED"
    validate "mpirun -np 1" "aero_par_mpi_cuda" "" "PASSED"
    validate "" "aero_seq" "" "PASSED"
    validate "" "aero_genseq" "" "PASSED"
fi

if [[ "$TEST_AERO_CPP" = "TRUE" ]] && [[ "$TEST_HDF5" = "TRUE" ]]; then

    cd $APPS_LOC/c/${TEST_APP}/aero_hdf5

    echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    
    make clean; 
    make;

    if [[ -f FE_grid.h5 ]]; then rm FE_grid.h5; fi
    wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/aero/FE_grid.h5

    echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    validate "mpirun -np 64" "aero_mpi_seq" "" "PASSED"
    validate "mpirun -np 64" "aero_mpi_genseq" "" "PASSED"
    validate "OMP_NUM_THREADS=8 mpirun -np 8" "aero_mpi_openmp" "" "PASSED"
    validate "mpirun -np 1" "aero_mpi_cuda" "" "PASSED"
    validate "OMP_NUM_THREADS=32" "aero_openmp" "" "PASSED"
    validate "" "aero_cuda" "" "PASSED"
    validate "" "aero_seq" "" "PASSED"
    validate "" "aero_genseq" "" "PASSED"
fi

if [ "$TEST_AERO_FORTRAN" = "TRUE" ]; then

    cd $APPS_LOC/fortran/${TEST_APP}/

    echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    
    make clean; 
    make;

    echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

fi

check_all_tests
