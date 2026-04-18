#!/bin/bash

set -e

# Source approprite script (scripts/source_gnuz) and run below command:
# COMPILE_OP2=TRUE COMPILE_TESTS=TRUE RUN_TESTS=TRUE ./mgcfd_test.sh

export TEST_APP="euler3d";

COMPILE_OP2=${COMPILE_OP2:-FALSE}
COMPILE_TESTS=${COMPILE_TESTS:-FALSE}
RUN_TESTS=${RUN_TESTS:-FALSE}

RUN_M6_WING=${RUN_M6_WING:-TRUE}
RUN_ROTOR_37_1M=${RUN_ROTOR_37_1M:-TRUE}

M6_WING_PATH=${M6_WING_PATH:-/home/zl/mgcfd-meshes/M6_wing}
ROTOR_37_1M_PATH=${ROTOR_37_1M_PATH:-/home/zl/mgcfd-meshes/Rotor37_1M}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/../../MG-CFD-app-OP2"

REPO_URL="https://github.com/warwick-hpsc/MG-CFD-app-OP2"
BRANCH="master"

source ./test_core.sh

unset OP_AUTO_SOA

# Compile OP2 -----------------------------------------------------------------------------
if [[ "$COMPILE_OP2" = "TRUE" ]]; then
    echo "Compiling OP2..." | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    cd $LIB_LOC
    make clean; 
    make config; 
    make -j24;
fi

# Clone MG-CFD -----------------------------------------------------------------------------
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning MG-CFD repo..."
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
else
    echo "Repo already exists at $REPO_DIR"
    cd "$REPO_DIR"
    
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "$BRANCH" ]; then
        echo "Switching to branch $BRANCH"
        git fetch origin
        git checkout "$BRANCH"
    fi
fi

# Compile MG-CFD app -----------------------------------------------------------------------------
APP_FOLDER=$PWD

if [[ "$COMPILE_TESTS" = "TRUE" ]]; then
    echo "Compiling MG-CFD app..." | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    make clean
    make
fi

# Run tests -------------------------------------------------------------------------------------
if [[ "$RUN_TESTS" = "TRUE" ]]; then
    if [[ "$RUN_M6_WING" = "TRUE" ]]; then
        if [[ ! -d M6_wing ]]; then
            wget https://warwick.ac.uk/fac/sci/dcs/research/systems/hpsc/software/m6_wing.tar.gz
            tar -xvf m6_wing.tar.gz
            rm m6_wing.tar.gz
            # cp -r "$M6_WING_PATH" .
        fi

        cd M6_wing
        echo "Running tests on $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "" "../euler3d_seq" "-i input.dat -v" "passed"
        validate "" "../euler3d_genseq" "-i input.dat -v" "passed"
        validate "OMP_NUM_THREADS=6" "../euler3d_openmp" "-i input.dat -v" "passed"
        validate "" "../euler3d_cuda" "-i input.dat -v" "passed"
        validate "" "../euler3d_c_cuda" "-i input.dat -v" "passed"
        validate "mpirun -np 16" "../euler3d_mpi_seq" "-i input.dat -v" "passed"
        validate "mpirun -np 16" "../euler3d_mpi_genseq" "-i input.dat -v" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "../euler3d_mpi_openmp" "-i input.dat -v" "passed"
        validate "mpirun -np 4" "../euler3d_mpi_cuda" "-i input.dat -v" "passed"
        validate "mpirun -np 4" "../euler3d_mpi_c_cuda" "-i input.dat -v" "passed"
    fi

    cd $APP_FOLDER

    if [[ "$RUN_ROTOR_37_1M" = "TRUE" ]]; then
        if [[ ! -d Rotor37_1M ]]; then
            wget https://warwick.ac.uk/fac/sci/dcs/research/systems/hpsc/software/rotor37_1m.tar.gz
            tar -xvf rotor37_1m.tar.gz
            rm rotor37_1m.tar.gz
            # cp -r "$ROTOR_37_1M_PATH" .
        fi

        cd Rotor37_1M
        echo "Running tests on $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

        validate "" "../euler3d_seq" "-i input.dat -v" "passed"
        validate "" "../euler3d_genseq" "-i input.dat -v" "passed"
        validate "OMP_NUM_THREADS=6" "../euler3d_openmp" "-i input.dat -v" "passed"
        validate "" "../euler3d_cuda" "-i input.dat -v" "passed"
        validate "" "../euler3d_c_cuda" "-i input.dat -v" "passed"
        validate "mpirun -np 16" "../euler3d_mpi_seq" "-i input.dat -v" "passed"
        validate "mpirun -np 16" "../euler3d_mpi_genseq" "-i input.dat -v" "passed"
        validate "OMP_NUM_THREADS=6 mpirun -np 8" "../euler3d_mpi_openmp" "-i input.dat -v" "passed"
        validate "mpirun -np 4" "../euler3d_mpi_cuda" "-i input.dat -v" "passed"
        validate "mpirun -np 4" "../euler3d_mpi_c_cuda" "-i input.dat -v" "passed"
    fi

    check_all_tests
fi