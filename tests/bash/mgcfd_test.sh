#!/bin/bash

set -e

export TEST_APP="euler3d";

COMPILE_OP2=${COMPILE_OP2:-FALSE}

RUN_M6_WING=${RUN_M6_WING:-TRUE}
RUN_ROTOR_37_1M=${RUN_ROTOR_37_1M:-TRUE}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/../../MG-CFD-app-OP2"

REPO_URL="https://github.com/warwick-hpsc/MG-CFD-app-OP2"
BRANCH="OP2_refactor"

source ./test_core.sh

unset OP_AUTO_SOA

if [[ "$COMPILE_OP2" = "TRUE" ]]; then
    echo "Compiling OP2..." | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    cd $LIB_LOC
    make clean; 
    make config; 
    make -j24;
fi

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

# Compile MG-CFD app
APP_FOLDER=$PWD
make clean
make

if [[ "$RUN_M6_WING" = "TRUE" ]]; then
    if [[ ! -d M6_wing ]]; then
        wget https://warwick.ac.uk/fac/sci/dcs/research/systems/hpsc/software/m6_wing.tar.gz
        tar -xvf m6_wing.tar.gz
        rm m6_wing.tar.gz
    fi

    cd M6_wing
    echo "Running tests on $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    validate "" "../euler3d_seq" "-i input.dat -v" "passed"
    validate "" "../euler3d_genseq" "-i input.dat -v" "passed"
    validate "" "../euler3d_openmp" "-i input.dat -v" "passed"
    validate "" "../euler3d_cuda" "-i input.dat -v" "passed"
    validate "mpirun -np 64" "../euler3d_mpi_seq" "-i input.dat -v" "passed"
    validate "mpirun -np 64" "../euler3d_mpi_genseq" "-i input.dat -v" "passed"
    validate "OMP_NUM_THREADS=8 mpirun -np 8" "../euler3d_mpi_openmp" "-i input.dat -v" "passed"
    validate "mpirun -np 4" "../euler3d_mpi_cuda" "-i input.dat -v" "passed"
fi

cd $APP_FOLDER

if [[ "$RUN_ROTOR_37_1M" = "TRUE" ]]; then
    if [[ ! -d Rotor37_1M ]]; then
        wget https://warwick.ac.uk/fac/sci/dcs/research/systems/hpsc/software/rotor37_1m.tar.gz
        tar -xvf rotor37_1m.tar.gz
        rm rotor37_1m.tar.gz
    fi

    cd Rotor37_1M
    echo "Running tests on $PWD" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    validate "" "../euler3d_seq" "-i input.dat -v" "passed"
    validate "" "../euler3d_genseq" "-i input.dat -v" "passed"
    validate "" "../euler3d_openmp" "-i input.dat -v" "passed"
    validate "" "../euler3d_cuda" "-i input.dat -v" "passed"
    validate "mpirun -np 64" "../euler3d_mpi_seq" "-i input.dat -v" "passed"
    validate "mpirun -np 64" "../euler3d_mpi_genseq" "-i input.dat -v" "passed"
    validate "OMP_NUM_THREADS=8 mpirun -np 8" "../euler3d_mpi_openmp" "-i input.dat -v" "passed"
    validate "mpirun -np 4" "../euler3d_mpi_cuda" "-i input.dat -v" "passed"
fi


check_all_tests
