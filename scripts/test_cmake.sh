#!/bin/bash
#set -e

function validate {
  $1 > perf_out
  echo
  echo $1
  grep "Max total runtime" perf_out;grep "PASSED" perf_out
  rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;rm perf_out
}


#
#Set up directory vars and Build OP2 C libs and apps with cmake
#
. ./source_intel
export CURRENT_DIR=$PWD
export OP2_INSTALL_PATH=$PWD
#./ruby.sh #build libs and apps with cmake for ruby.oerc.ox.ac.uk
./octon.sh #build libs and apps with cmake for octon.arc.ox.ac.uk

#
# Change Dir to OP2 C apps
#
cd $OP2_INSTALL_PATH
cd ../apps/c
export OP2_APPS_DIR=$PWD
export OP2_C_APPS_BIN_DIR=$OP2_APPS_DIR/bin
cd $OP2_C_APPS_BIN_DIR
echo "In directory $PWD"

#
# Now run C apps
#

#-------------------------------------------------------------------------------
# test Arifoil DP- with plain text file I/O
#-------------------------------------------------------------------------------

#<<COMMENT
echo " "
echo "----------------Testing airfoil_dp_seq ----------------------------------"
echo " "
validate "./airfoil_dp_seq"

echo " "
echo "----------------Testing airfoil_dp_cuda ---------------------------------"
echo " "
validate "./airfoil_dp_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "----------------Testing airfoil_dp_openmp--------------------------------"
echo " "
export OMP_NUM_THREADS=20
validate "./airfoil_dp_openmp OP_PART_SIZE=256"

echo " "
echo "----------------Testing airfoil_dp_mpi ----------------------------------"
echo " "
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_dp_mpi"

echo " "
echo "----------------Testing airfoil_dp_mpi_cuda 1 mpi proc ------------------"
echo " "
validate "./airfoil_dp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "----------------Testing airfoil_dp_mpi_cuda 2 mpi procs------------------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_dp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "----------------Testing airfoil_dp_mpi_openmp 1 mpi proc ----------------"
echo " "
export OMP_NUM_THREADS=20
validate "./airfoil_dp_mpi_openmp OP_PART_SIZE=256"

echo " "
echo "----------------Testing airfoil_dp_mpi_openmp 10 mpi procs---------------"
echo " "
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_dp_mpi_openmp OP_PART_SIZE=256"



#-------------------------------------------------------------------------------
# test Arifoil SP- with plain text file I/O
#-------------------------------------------------------------------------------

echo " "
echo "-----------------Testing airfoil_sp_seq----------------------------------"
echo " "
validate "./airfoil_sp_seq ;"

echo " "
echo "-----------------Testing airfoil_sp_cuda---------------------------------"
echo " "
validate "./airfoil_sp_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "-----------------Testing airfoil_sp_openmp-------------------------------"
echo " "
export OMP_NUM_THREADS=20
validate "./airfoil_sp_openmp OP_PART_SIZE=256"

echo " "
echo "-----------------Testing airfoil_sp_mpi----------------------------------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_sp_mpi"

echo " "
echo "-----------------Testing airfoil_sp_mpi_cuda 1 mpi proc -----------------"
echo " "
validate "./airfoil_sp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "-----------------Testing airfoil_sp_mpi_cuda 2 mpi procs-----------------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_sp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "-----------------Testing airfoil_sp_mpi_openmp 1 mpi proc ---------------"
echo " "
export OMP_NUM_THREADS=20
validate "./airfoil_sp_mpi_openmp OP_PART_SIZE=256"

echo " "
echo "-----------------Testing airfoil_sp_mpi_openmp 10 mpi procs--------------"
echo " "
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_sp_mpi_openmp OP_PART_SIZE=256"

#-------------------------------------------------------------------------------
# test Arifoil DP- with hdf5 file I/O
#-------------------------------------------------------------------------------

echo " "
echo "-----------------Testing airfoil_hdf5_dp_seq-----------------------------"
echo " "
validate "./airfoil_hdf5_dp_seq"

echo " "
echo "-----------------Testing airfoil_hdf5_dp_cuda ---------------------------"
echo " "
validate "./airfoil_hdf5_dp_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "-----------------Testing airfoil_hdf5_dp_openmp -------------------------"
echo " "
export OMP_NUM_THREADS=20
validate "./airfoil_hdf5_dp_openmp OP_PART_SIZE=256"

echo " "
echo "-----------------Testing airfoil_hdf5_dp_mpi-----------------------------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_hdf5_dp_mpi"

echo " "
echo "-----------------Testing airfoil_hdf5_dp_mpi_cuda 1 mpi proc ------------"
echo " "
validate "./airfoil_hdf5_dp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "-----------------Testing airfoil_hdf5_dp_mpi_cuda 2 mpi procs------------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_hdf5_dp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "-----------------Testing airfoil_hdf5_dp_mpi_openmp 1 mpi proc ----------"
echo " "
export OMP_NUM_THREADS=20
validate "./airfoil_hdf5_dp_mpi_openmp OP_PART_SIZE=256"

echo " "
echo "-----------------Testing airfoil_hdf5_dp_mpi_openmp 10 mpi procs---------"
echo " "
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_hdf5_dp_mpi_openmp OP_PART_SIZE=256"


#-------------------------------------------------------------------------------
# test Arifoil_tempdats DP
#-------------------------------------------------------------------------------

echo " "
echo "-----------------Testing airfoil_tempdats_seq---------------------------"
echo " "
validate "./airfoil_tempdats_seq"

echo " "
echo "-----------------Testing airfoil_tempdats_cuda---------------------------"
echo " "
validate "./airfoil_tempdats_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "-----------------Testing airfoil_tempdats_openmp-------------------------"
echo " "
export OMP_NUM_THREADS=20
validate "./airfoil_tempdats_openmp OP_PART_SIZE=256"

echo " "
echo "-----------------Testing airfoil_tempdats_mpi ---------------------------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_tempdats_mpi"

echo " "
echo "-----------------Testing airfoil_tempdats_mpi_cuda 1 mpi proc -----------"
echo " "
validate "./airfoil_tempdats_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "-----------------Testing airfoil_tempdats_mpi_cuda 2 mpi procs-----------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_tempdats_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

echo " "
echo "-----------------Testing airfoil_tempdats_mpi_openmp 1 mpi proc----------"
echo " "
export OMP_NUM_THREADS=20
validate "./airfoil_tempdats_mpi_openmp OP_PART_SIZE=256"

echo " "
echo "-----------------Testing airfoil_tempdats_mpi_openmp 1 mpi procs---------"
echo " "
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_tempdats_mpi_openmp OP_PART_SIZE=256"



#-------------------------------------------------------------------------------
# test Jac DP - with plain text file I/O
#-------------------------------------------------------------------------------

echo " "
echo "------------------Testing jac1_dp_seq------------------------------------"
echo " "
validate "./jac1_dp_seq"

echo " "
echo "------------------Testing jac1_dp_cuda ----------------------------------"
echo " "
validate "./jac1_dp_cuda"

echo " "
echo "------------------Testing jac1_dp_openmp --------------------------------"
echo " "
export OMP_NUM_THREADS=20
validate "./jac1_dp_openmp"

echo " "
echo "------------------Testing jac1_dp_mpi------------------------------------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./jac1_dp_mpi"

#-------------------------------------------------------------------------------
# test Jac SP - with plain text file I/O
#-------------------------------------------------------------------------------

echo " "
echo "------------------Testing jac1_sp_seq------------------------------------"
echo " "
validate "./jac1_sp_seq"

echo " "
echo "------------------Testing jac1_sp_cuda-----------------------------------"
echo " "
validate "./jac1_sp_cuda"

echo " "
echo "------------------Testing jac1_sp_openmp---------------------------------"
echo " "
export OMP_NUM_THREADS=20
validate "./jac1_sp_openmp"

echo " "
echo "------------------Testing jac1_sp_mpi------------------------------------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./jac1_sp_mpi"


#-------------------------------------------------------------------------------
# test Jac2 DP - with plain text file I/O
#-------------------------------------------------------------------------------

echo " "
echo "------------------Testing jac2_seq --------------------------------------"
echo " "
validate "./jac2_seq"

echo " "
echo "------------------Testing jac2_cuda--------------------------------------"
echo " "
validate "./jac2_cuda"

echo " "
echo "------------------Testing jac2_openmp------------------------------------"
echo " "
export OMP_NUM_THREADS=20
validate "./jac2_openmp"

echo " "
echo "------------------Testing jac2_mpi---------------------------------------"
echo " "
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./jac2_mpi"

<<COMMENT


#-------------------------------------------------------------------------------
# test Aero DP- with plain text file I/O
#-------------------------------------------------------------------------------

echo " "
echo "-------------------Testing aero_dp_seq-----------------------------------"
echo " "
./aero_dp_seq > perf_out
grep "rms = 5.6" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out


echo " "
echo "-------------------Testing aero_dp_cuda----------------------------------"
echo " "
./aero_dp_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_dp_openmp--------------------------------"
echo " "
export OMP_NUM_THREADS=20
./aero_dp_openmp OP_PART_SIZE=256 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_dp_mpi-----------------------------------"
echo " "
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./aero_dp_mpi > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_dp_mpi_cuda 1 mpi proc ------------------"
echo " "
./aero_dp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_dp_mpi_cuda 2 mpi procs------------------"
echo " "
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./numawrap20 ./aero_dp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_dp_mpi_openmp 1 mpi proc ----------------"
echo " "
./aero_dp_mpi_openmp OP_PART_SIZE=256 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_dp_mpi_openmp 10 mpi procs---------------"
echo " "
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./aero_dp_mpi_openmp OP_PART_SIZE=256 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out




#-------------------------------------------------------------------------------
# test Aero DP- with hdf5 file I/O
#-------------------------------------------------------------------------------

echo " "
echo "-------------------Testing aero_hdf5_dp_seq-----------------------------"
echo " "
./aero_hdf5_dp_seq > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out


echo " "
echo "-------------------Testing aero_hdf5_dp_cuda ----------------------------"
echo " "
./aero_hdf5_dp_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_hdf5_dp_openmp --------------------------"
echo " "
export OMP_NUM_THREADS=20
./aero_hdf5_dp_openmp OP_PART_SIZE=256 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_hdf5_dp_mpi------------------------------"
echo " "
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./aero_hdf5_dp_mpi > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_hdf5_dp_mpi_cuda 1 mpi proc -------------"
echo " "
./aero_hdf5_dp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_hdf5_dp_mpi_cuda 2 mpi procs-------------"
echo " "
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./numawrap20 ./aero_hdf5_dp_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_hdf5_dp_mpi_openmp 1 mpi proc -----------"
echo " "
./aero_hdf5_dp_mpi_openmp OP_PART_SIZE=256 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

echo " "
echo "-------------------Testing aero_hdf5_dp_mpi_openmp 10 mpi procs----------"
echo " "
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./aero_hdf5_dp_mpi_openmp OP_PART_SIZE=256 > perf_out
grep "iter: 200" perf_out;grep "Max total runtime" perf_out;tail -n 1  perf_out

COMMENT