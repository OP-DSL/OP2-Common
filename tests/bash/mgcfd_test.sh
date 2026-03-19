#!/bin/bash

set -e

export TEST_APP="euler3d";

source ./test_core.sh

unset OP_AUTO_SOA

cd $LIB_LOC
make clean; 
make config; 
make -j24;

cd ../../MG-CFD-app-OP2
APP_FOLDER=$PWD
make clean
make


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


cd $APP_FOLDER
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


check_all_tests
