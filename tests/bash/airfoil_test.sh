#!/bin/bash

set -e

export TEST_APP="airfoil";

TEST_AIRFOIL_CPP=TRUE
TEST_AIRFOIL_FORTRAN=FALSE

TEST_PLAIN=TRUE
TEST_HDF5=TRUE
TEST_TEMPDATS=TRUE

source ./test_core.sh

cd $LIB_LOC
make clean; 
make config; 
make -j24;


if [[ "$TEST_AIRFOIL_CPP" = "TRUE" ]] && [[ "$TEST_PLAIN" = "TRUE" ]]; then

    for p in "${precision[@]}"; do

        cd $APPS_LOC/c/${TEST_APP}/airfoil_plain/$p

        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;

        if [ -f "new_grid.dat" ]; then rm new_grid.dat; fi
        wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid.dat

        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "OMP_NUM_THREADS=32" "airfoil_openmp" "" "PASSED"
        validate "" "airfoil_cuda" "" "PASSED"
        validate "mpirun -np 64" "airfoil_par_mpi_seq" "" "PASSED"
        validate "mpirun -np 64" "airfoil_par_mpi_genseq" "" "PASSED"
        validate "OMP_NUM_THREADS=8 mpirun -np 8" "airfoil_par_mpi_openmp" "" "PASSED"
        validate "mpirun -np 1" "airfoil_par_mpi_cuda" "" "PASSED"
        validate "" "airfoil_seq" "" "PASSED"
        validate "" "airfoil_genseq" "" "PASSED"
    done
fi


if [[ "$TEST_AIRFOIL_CPP" = "TRUE" ]] && [[ "$TEST_TEMPDATS" = "TRUE" ]]; then

    cd $APPS_LOC/c/${TEST_APP}/airfoil_tempdats/dp

    echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    
    make clean; 
    make;

    if [ -f "new_grid.dat" ]; then rm new_grid.dat; fi
    wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid.dat

    echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    validate "OMP_NUM_THREADS=32" "airfoil_openmp" "" "PASSED"
    validate "" "airfoil_cuda" "" "PASSED"
    validate "mpirun -np 64" "airfoil_par_mpi_seq" "" "PASSED"
    validate "mpirun -np 64" "airfoil_par_mpi_genseq" "" "PASSED"
    validate "OMP_NUM_THREADS=8 mpirun -np 8" "airfoil_par_mpi_openmp" "" "PASSED"
    validate "mpirun -np 1" "airfoil_par_mpi_cuda" "" "PASSED"
    validate "" "airfoil_seq" "" "PASSED"
    validate "" "airfoil_genseq" "" "PASSED"
fi


if [[ "$TEST_AIRFOIL_CPP" = "TRUE" ]] && [[ "$TEST_HDF5" = "TRUE" ]]; then

    for p in "${precision[@]}"; do

        cd $APPS_LOC/c/${TEST_APP}/airfoil_hdf5/$p

        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;

        wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid_$p.h5
        if is_file_available "new_grid_$p.h5"; then
            mv new_grid_$p.h5 new_grid.h5
        else
            continue
        fi

        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "OMP_NUM_THREADS=32" "airfoil_openmp" "" "PASSED"
        validate "" "airfoil_cuda" "" "PASSED"
        validate "mpirun -np 64" "airfoil_mpi_seq" "" "PASSED"
        validate "mpirun -np 64" "airfoil_mpi_genseq" "" "PASSED"
        validate "OMP_NUM_THREADS=8 mpirun -np 8" "airfoil_mpi_openmp" "" "PASSED"
        validate "mpirun -np 1" "airfoil_mpi_cuda" "" "PASSED"
        validate "" "airfoil_seq" "" "PASSED"
        validate "" "airfoil_genseq" "" "PASSED"
    done
fi


if [[ "$TEST_AIRFOIL_FORTRAN" = "TRUE" ]] && [[ "$TEST_PLAIN" = "TRUE" ]]; then

    cd $APPS_LOC/fortran/${TEST_APP}

    echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    
    make clean; 
    make;

    if [ -f "new_grid.dat" ]; then rm new_grid.dat; fi
    wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid.dat

    wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid_dp.h5
    if is_file_available "new_grid_dp.h5"; then
        mv new_grid_dp.h5 new_grid.h5
    fi

    echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    validate "" "airfoil_plain_seq" "" "PASSED"
    validate "" "airfoil_plain_genseq" "" "PASSED"

    validate "" "airfoil_arg_ptrs_seq" "" "PASSED"
    validate "" "airfoil_arg_ptrs_genseq" "" "PASSED"

    validate "mpirun -np 64" "airfoil_hdf5_mpi_seq" "" "PASSED"
    validate "mpirun -np 64" "airfoil_hdf5_mpi_genseq" "" "PASSED"
fi

check_all_tests
