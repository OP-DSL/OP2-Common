#!/bin/bash

# Source approprite script (scripts/source_gnuz) and run below command:
# COMPILE_OP2=TRUE COMPILE_TESTS=TRUE RUN_TESTS=TRUE ./airfoil_test.sh

set -e

export TEST_APP="airfoil";

COMPILE_OP2=${COMPILE_OP2:-FALSE}
COMPILE_TESTS=${COMPILE_TESTS:-FALSE}
RUN_TESTS=${RUN_TESTS:-FALSE}

TEST_AIRFOIL_CPP=TRUE
TEST_AIRFOIL_FORTRAN=TRUE

TEST_PLAIN=TRUE
TEST_HDF5=TRUE
TEST_TEMPDATS=TRUE

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

# Compile and run C++ airfoil plain tests --------------------------------------------------
if [[ "$TEST_AIRFOIL_CPP" = "TRUE" ]] && [[ "$TEST_PLAIN" = "TRUE" ]]; then

    for p in "${precision[@]}"; do

        cd $APPS_LOC/c/${TEST_APP}/airfoil_plain/$p

        if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
            echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
            
            make clean; 
            make;

            if [ -f "new_grid.dat" ]; then rm new_grid.dat; fi
            wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid.dat
        fi

        if [[ "$RUN_TESTS" = "TRUE" ]]; then
            echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
            echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

            validate "OMP_NUM_THREADS=6" "airfoil_openmp" "" "PASSED"
            validate "" "airfoil_cuda" "" "PASSED"
            validate "" "airfoil_c_cuda" "" "PASSED"
            validate "mpirun -np 16" "airfoil_par_mpi_seq" "" "PASSED"
            validate "mpirun -np 16" "airfoil_par_mpi_genseq" "" "PASSED"
            validate "OMP_NUM_THREADS=6 mpirun -np 8" "airfoil_par_mpi_openmp" "" "PASSED"
            validate "mpirun -np 4" "airfoil_par_mpi_cuda" "" "PASSED"
            validate "mpirun -np 4" "airfoil_par_mpi_c_cuda" "" "PASSED"
            validate "" "airfoil_seq" "" "PASSED"
            validate "" "airfoil_genseq" "" "PASSED"
        fi
    done
fi

# Compile and run C++ airfoil plain tempdats tests -----------------------------------------
if [[ "$TEST_AIRFOIL_CPP" = "TRUE" ]] && [[ "$TEST_TEMPDATS" = "TRUE" ]]; then

    cd $APPS_LOC/c/${TEST_APP}/airfoil_tempdats/dp

    if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;

        if [ -f "new_grid.dat" ]; then rm new_grid.dat; fi
        wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid.dat
    fi

    if [[ "$RUN_TESTS" = "TRUE" ]]; then
        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "OMP_NUM_THREADS=6" "airfoil_openmp" "" "PASSED"
        validate "" "airfoil_cuda" "" "PASSED"
        validate "" "airfoil_c_cuda" "" "PASSED"
        validate "mpirun -np 16" "airfoil_par_mpi_seq" "" "PASSED"
        validate "mpirun -np 16" "airfoil_par_mpi_genseq" "" "PASSED"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "airfoil_par_mpi_openmp" "" "PASSED"
        validate "mpirun -np 4" "airfoil_par_mpi_cuda" "" "PASSED"
        validate "mpirun -np 4" "airfoil_par_mpi_c_cuda" "" "PASSED"
        validate "" "airfoil_seq" "" "PASSED"
        validate "" "airfoil_genseq" "" "PASSED"
    fi
fi

# Compile and run C++ airfoil plain hdf5 tests ---------------------------------------------------
if [[ "$TEST_AIRFOIL_CPP" = "TRUE" ]] && [[ "$TEST_HDF5" = "TRUE" ]]; then

    for p in "${precision[@]}"; do

        cd $APPS_LOC/c/${TEST_APP}/airfoil_hdf5/$p

        if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
            echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
            
            make clean; 
            make;

            wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid_$p.h5
            if is_file_available "new_grid_$p.h5"; then
                mv new_grid_$p.h5 new_grid.h5
            else
                continue
            fi
        fi

        if [[ "$RUN_TESTS" = "TRUE" ]]; then
            echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
            echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

            validate "OMP_NUM_THREADS=6" "airfoil_openmp" "" "PASSED"
            validate "" "airfoil_cuda" "" "PASSED"
            validate "" "airfoil_c_cuda" "" "PASSED"
            validate "mpirun -np 16" "airfoil_mpi_seq" "" "PASSED"
            validate "mpirun -np 16" "airfoil_mpi_genseq" "" "PASSED"
            validate "OMP_NUM_THREADS=6 mpirun -np 8" "airfoil_mpi_openmp" "" "PASSED"
            validate "mpirun -np 4" "airfoil_mpi_cuda" "" "PASSED"
            validate "mpirun -np 4" "airfoil_mpi_c_cuda" "" "PASSED"
            validate "" "airfoil_seq" "" "PASSED"
            validate "" "airfoil_genseq" "" "PASSED"
        fi
    done
fi

# Compile and run Fortran airfoil tests -------------------------------------------------------
if [[ "$TEST_AIRFOIL_FORTRAN" = "TRUE" ]]; then

    cd $APPS_LOC/fortran/${TEST_APP}

    if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
        echo "Compiling App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        
        make clean; 
        make;

        if [ -f "new_grid.dat" ]; then rm new_grid.dat; fi
        wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid.dat

        wget https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid_dp.h5
        if is_file_available "new_grid_dp.h5"; then
            mv new_grid_dp.h5 new_grid.h5
        fi
    fi

    if [[ "$RUN_TESTS" = "TRUE" ]]; then
        echo "Running tests on App: $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "" "airfoil_plain_seq" "" "PASSED"
        validate "" "airfoil_plain_genseq" "" "PASSED"

        validate "OMP_NUM_THREADS=6" "airfoil_plain_openmp" "" "PASSED"
        # validate "" "airfoil_plain_cuda" "" "PASSED"
        validate "" "airfoil_plain_c_cuda" "" "PASSED"

        # validate "" "airfoil_arg_ptrs_seq" "" "PASSED"
        # validate "" "airfoil_arg_ptrs_genseq" "" "PASSED"

        # validate "OMP_NUM_THREADS=6" "airfoil_arg_ptrs_openmp" "" "PASSED"
        # validate "" "airfoil_arg_ptrs_cuda" "" "PASSED"

        validate "" "airfoil_hdf5_seq" "" "PASSED"
        validate "" "airfoil_hdf5_genseq" "" "PASSED"

        validate "OMP_NUM_THREADS=6" "airfoil_hdf5_openmp" "" "PASSED"
        # validate "" "airfoil_hdf5_cuda" "" "PASSED"
        validate "" "airfoil_hdf5_c_cuda" "" "PASSED"

        validate "mpirun -np 16" "airfoil_hdf5_mpi_seq" "" "PASSED"
        validate "mpirun -np 16" "airfoil_hdf5_mpi_genseq" "" "PASSED"

        validate "OMP_NUM_THREADS=6 mpirun -np 8" "airfoil_hdf5_mpi_openmp" "" "PASSED"
        # validate "mpirun -np 4" "airfoil_hdf5_mpi_cuda" "" "PASSED"
        validate "mpirun -np 4" "airfoil_hdf5_mpi_c_cuda" "" "PASSED"
    fi
fi

if [[ "$RUN_TESTS" = "TRUE" ]]; then
    check_all_tests
fi
