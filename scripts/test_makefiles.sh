#!/bin/bash

#
#Test OP2 example applications using the basic makefiles build
#

#exit script if any error is encountered during the build or
#application executions.
set -e

#set python3 as default
alias python=python3

function validate {
  $1 > perf_out
  echo
  echo $1
  grep "Max total runtime" perf_out;grep "PASSED" perf_out
  rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;rm perf_out
}

#Octon
#export NV_ARCH=Kepler
#export CUDA_VISIBLE_DEVICES=1,2
#export INTEL_SOURCE=source_intel
#export PGI_SOURCE=source_pgi_19
#export CLANG_SOURCE=source_clang

#Kos
#export NV_ARCH=Pascal
#export CUDA_VISIBLE_DEVICES=0,1
#export INTEL_SOURCE=source_intel_18
#export PGI_SOURCE=source_pgi_20_hydra
#export CLANG_SOURCE=source_clang_kos

#Telos
export NV_ARCH=Volta
export CUDA_VISIBLE_DEVICES=0
export INTEL_SOURCE=source_intel_18
export PGI_SOURCE=source_pgi_20_hydra
export CLANG_SOURCE=source_clang_kos


export CURRENT_DIR=$PWD
cd ../op2
export OP2_INSTALL_PATH=$PWD
cd $OP2_INSTALL_PATH
cd ../apps
export OP2_APPS_DIR=$PWD
export OP2_C_APPS_BIN_DIR=$OP2_APPS_DIR/c/bin
cd ../translator/c/python/
export OP2_C_CODEGEN_DIR=$PWD
cd ../../fortran/python/
export OP2_FORT_CODEGEN_DIR=$PWD
cd $OP2_INSTALL_PATH/c


#<<COMMENT0
#<<COMMENT_PGI


echo " "
echo " "
echo "**********************************************************************"
echo "***********************> Building C back-end libs with Intel Compilers"
echo "**********************************************************************"
. $CURRENT_DIR/$INTEL_SOURCE
make clean; make

echo " "
echo " "
echo "=======================> Building Airfoil Plain DP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_plain/dp
$OP2_C_CODEGEN_DIR/op2.py airfoil.cpp
$OP2_C_CODEGEN_DIR/op2.py airfoil_mpi.cpp
make clean;make



echo " "
echo " "
echo "=======================> Building Airfoil Plain SP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_plain/sp
$OP2_C_CODEGEN_DIR/op2.py airfoil.cpp
$OP2_C_CODEGEN_DIR/op2.py airfoil_mpi.cpp
make clean;make
echo " "
echo " "
echo "=======================> Building Airfoil HDF5 DP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_hdf5/dp
$OP2_C_CODEGEN_DIR/op2.py airfoil.cpp
make clean;make
echo " "
echo " "
echo "=======================> Building Airfoil HDF5 SP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_hdf5/sp
$OP2_C_CODEGEN_DIR/op2.py airfoil.cpp
make clean;make
echo " "
echo " "
echo "=======================> Building Airfoil TEMPDATS DP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_tempdats/dp/
$OP2_C_CODEGEN_DIR/op2.py airfoil.cpp
$OP2_C_CODEGEN_DIR/op2.py airfoil_mpi.cpp
make clean;make


echo " "
echo " "
echo "=======================> Building Aero Plain DP with Intel Compilers"
cd $OP2_APPS_DIR/c/aero/aero_plain/dp/
$OP2_C_CODEGEN_DIR/op2.py aero.cpp
$OP2_C_CODEGEN_DIR/op2.py aero_mpi.cpp
make clean;make
echo " "
echo " "
echo "=======================> Building Aero HDF5 DP with Intel Compilers"
cd $OP2_APPS_DIR/c/aero/aero_hdf5/dp/
$OP2_C_CODEGEN_DIR/op2.py aero.cpp
make clean;make



echo " "
echo " "
echo "=======================> Building Jac1 Plain DP with Intel Compilers"
cd $OP2_APPS_DIR/c/jac1/dp/
$OP2_C_CODEGEN_DIR/op2.py jac.cpp
make clean;make
echo " "
echo " "
echo "=======================> Building Jac1 Plain SP with Intel Compilers"
cd $OP2_APPS_DIR/c/jac1/sp/
$OP2_C_CODEGEN_DIR/op2.py jac.cpp
make clean;make
echo " "
echo " "
echo "=======================> Building Jac2 with Intel Compilers"
cd $OP2_APPS_DIR/c/jac2
$OP2_C_CODEGEN_DIR/op2.py jac.cpp
make clean;make


echo " "
echo " "
echo "=======================> Building Reduction with Intel Compilers"
cd $OP2_APPS_DIR/c/reduction
$OP2_C_CODEGEN_DIR/op2.py reduction.cpp
$OP2_C_CODEGEN_DIR/op2.py reduction_mpi.cpp
make clean;make

#<<COMMENT0


echo " "
echo " "
echo "**********************************************************************"
echo "**************************** Running C Apps built with Intel Compilers"
echo "**********************************************************************"

echo " "
echo " "
echo "=======================> Running Airfoil Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_plain/dp
validate "./airfoil_seq"
validate "./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
export OMP_NUM_THREADS=20
validate "./airfoil_openmp OP_PART_SIZE=256"
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_vec"
validate "./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda -renumber OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
export OMP_NUM_THREADS=20
validate "./airfoil_mpi_openmp OP_PART_SIZE=256"
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_mpi_openmp OP_PART_SIZE=256"


echo " "
echo " "
echo "=======================> Running Airfoil HDF5 DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_hdf5/dp
#validate "./airfoil_seq"
validate "./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
export OMP_NUM_THREADS=20
validate "./airfoil_openmp OP_PART_SIZE=256"
validate "./airfoil_openmp OP_PART_SIZE=256 -renumber"
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_genseq"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_genseq -renumber"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_vec"
validate "./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda_hyb OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
#validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_mpi_cuda_hyb OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

export OMP_NUM_THREADS=20
validate "./airfoil_mpi_openmp OP_PART_SIZE=256"
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_mpi_openmp OP_PART_SIZE=256"

#COMMENT0

echo " "
echo " "
echo "=======================> Running Airfoil HDF5 SP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_hdf5/sp
#validate "./airfoil_seq"
validate "./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
export OMP_NUM_THREADS=20
validate "./airfoil_openmp OP_PART_SIZE=256"
validate "./airfoil_openmp OP_PART_SIZE=256 -renumber"
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_genseq"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_genseq -renumber"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_vec"
validate "./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda_hyb OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

export OMP_NUM_THREADS=20
validate "./airfoil_mpi_openmp OP_PART_SIZE=256"
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_mpi_openmp OP_PART_SIZE=256"


echo "=======================> Running Convertmesh built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_hdf5/dp
make convert_mesh_seq convert_mesh_mpi

if [ -f "./test.h5" ]
then
  rm test.h5
fi
./convert_mesh_seq
./convert_mesh_seq
rm ./test.h5
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./convert_mesh_mpi
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./convert_mesh_mpi
./convert_mesh_seq
rm ./test.h5
./convert_mesh_seq
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./convert_mesh_mpi


echo " "
echo " "
echo "=======================> Running Airfoil Tempdats DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_tempdats/dp
validate "./airfoil_seq"
validate "./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
export OMP_NUM_THREADS=20
validate "./airfoil_openmp OP_PART_SIZE=256"
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi"
validate "./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
export OMP_NUM_THREADS=20
validate "./airfoil_mpi_openmp OP_PART_SIZE=256"
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_mpi_openmp OP_PART_SIZE=256"

#COMMENT0


echo " "
echo " "
echo "=======================> Running Aero Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/aero/aero_plain/dp
./aero_seq
./aero_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./aero_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./aero_mpi
./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./aero_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./aero_mpi_openmp OP_PART_SIZE=256


echo " "
echo " "
echo "=======================> Running Aero Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/aero/aero_hdf5/dp
./aero_seq
./aero_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./aero_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./aero_mpi
./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2  ./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./aero_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 12 ./aero_mpi_openmp OP_PART_SIZE=256



echo " "
echo " "
echo "=======================> Running Jac1 Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/jac1/dp/
validate "./jac_seq"
validate "./jac_cuda"
export OMP_NUM_THREADS=20
validate "./jac_openmp"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./jac_mpi"


echo " "
echo " "
echo "=======================> Running Jac1 Plain SP built with Intel Compilers"
cd $OP2_APPS_DIR/c/jac1/sp/
validate "./jac_seq"
validate "./jac_cuda"
export OMP_NUM_THREADS=20
validate "./jac_openmp"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./jac_mpi"

echo " "
echo " "
echo "=======================> Running Jac2 Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/jac2/
validate "./jac_seq"
validate "./jac_cuda"
export OMP_NUM_THREADS=20
validate "./jac_openmp"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./jac_mpi"


echo " "
echo " "
echo "=======================> Running Reduction built with Intel Compilers"
cd $OP2_APPS_DIR/c/reduction/
validate "./reduction_seq"
validate "./reduction_cuda"
validate "./reduction_vec"
export OMP_NUM_THREADS=20
validate "./reduction_openmp"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./reduction_mpi"
validate "./reduction_mpi_cuda"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./reduction_mpi_cuda"
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./reduction_mpi_openmp"


################################################################################
################################################################################
echo " "
echo " "
echo "**********************************************************************"
echo "******************* Building Fortan back-end libs with Intel Compilers"
echo "**********************************************************************"
cd $OP2_INSTALL_PATH/fortran
pwd
. $CURRENT_DIR/$INTEL_SOURCE
make clean; make

echo " "
echo " "
echo "=======================> Building Airfoil Fortran Plain DP with Intel Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_plain/dp
$OP2_FORT_CODEGEN_DIR/op2_fortran.py airfoil.F90
export PART_SIZE_ENV=128
make clean; make


echo " "
echo " "
echo "=======================> Building Airfoil Fortran HDF5 DP with Intel Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_hdf5/dp
pwd
$OP2_FORT_CODEGEN_DIR/op2_fortran.py airfoil_hdf5.F90
export PART_SIZE_ENV=128
make clean; make


echo " "
echo " "
echo "=======================> Building Airfoil Fortran DP ARG PTR Version With Intel Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_arg_ptrs/dp
pwd
$OP2_FORT_CODEGEN_DIR/op2_fortran.py airfoil.F90
export PART_SIZE_ENV=128
make clean; make



echo " "
echo " "
echo "**********************************************************************"
echo "********************** Running Fortran Apps built with Intel Compilers"
echo "**********************************************************************"

echo " "
echo " "
echo "=======================> Running Airfoil Fortran Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_plain/dp
pwd
export PART_SIZE_ENV=128
validate "./airfoil_seq OP_MAPS_BASE_INDEX=0"
validate "./airfoil_vec OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=20
validate "./airfoil_openmp  OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV


echo " "
echo " "
echo "=======================> Running Airfoil Fortran Arg Pointers DP built with Intel Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_arg_ptrs/dp
pwd
export PART_SIZE_ENV=128
validate "./airfoil_seq OP_MAPS_BASE_INDEX=0"
validate "./airfoil_genseq OP_MAPS_BASE_INDEX=0"
validate "./airfoil_vec OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=20
validate "./airfoil_openmp  OP_MAPS_BASE_INDEX=0"




echo " "
echo " "
echo "=======================> Running Airfoil Fortran HDF5 DP built with Intel Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_hdf5/dp
pwd
export PART_SIZE_ENV=128
validate "./airfoil_hdf5_seq OP_MAPS_BASE_INDEX=0"
validate "./airfoil_hdf5_vec OP_MAPS_BASE_INDEX=0"
validate "./airfoil_hdf5_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_hdf5_mpi OP_MAPS_BASE_INDEX=0"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_hdf5_mpi_vec OP_MAPS_BASE_INDEX=0"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_hdf5_mpi_genseq OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=20
validate "./airfoil_hdf5_mpi_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_hdf5_mpi OP_MAPS_BASE_INDEX=0"
#_openmp_$PART_SIZE_ENV

#COMMENT0

###################################################################################
###################################################################################

#COMMENT_PGI

echo " "
echo " "
echo "**********************************************************************"
echo "********************* Building Fortan back-end libs with PGI Compilers"
echo "**********************************************************************"
cd $OP2_INSTALL_PATH/fortran
. $CURRENT_DIR/$PGI_SOURCE
pwd

make clean; make

echo " "
echo " "
echo "=======================> Building Airfoil Fortran Plain DP with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_plain/dp
pwd
$OP2_FORT_CODEGEN_DIR/op2_fortran.py airfoil.F90
export PART_SIZE_ENV=128
make clean; make

echo " "
echo " "
echo "=======================> Building Airfoil Fortran DP ARG PTR Version With PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_arg_ptrs/dp
pwd
$OP2_FORT_CODEGEN_DIR/op2_fortran.py airfoil.F90
export PART_SIZE_ENV=128
make clean; make




echo " "
echo " "
echo "=======================> Building Airfoil Fortran HDF5 DP with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_hdf5/dp
pwd
$OP2_FORT_CODEGEN_DIR/op2_fortran.py airfoil_hdf5.F90
export PART_SIZE_ENV=128
make clean; make

echo " "
echo " "
echo "**********************************************************************"
echo "************************ Running Fortran Apps built with PGI Compilers"
echo "**********************************************************************"

echo " "
echo " "
echo "=======================> Running Airfoil Fortran Plain DP built with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_plain/dp
pwd
export PART_SIZE_ENV=128
validate "./airfoil_seq OP_MAPS_BASE_INDEX=0"
validate "./airfoil_cuda OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=28
validate "./airfoil_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV

echo " "
echo " "
echo "=======================> Running Airfoil Fortran Arg Pointers DP built with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_arg_ptrs/dp
pwd
export PART_SIZE_ENV=128
validate "./airfoil_seq OP_MAPS_BASE_INDEX=0"
validate "./airfoil_cuda OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=28
validate "./airfoil_openmp  OP_MAPS_BASE_INDEX=0"

echo " "
echo " "
echo "=======================> Running Airfoil Fortran HDF5 DP built with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_hdf5/dp
pwd
export PART_SIZE_ENV=128
validate "./airfoil_hdf5_seq OP_MAPS_BASE_INDEX=0"
validate "./airfoil_hdf5_cuda OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=28
validate "./airfoil_hdf5_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV
validate "./airfoil_hdf5_openacc OP_PART_SIZE=128 OP_BLOCK_SIZE=192 OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 28 ./airfoil_hdf5_mpi OP_MAPS_BASE_INDEX=0"
validate "./airfoil_hdf5_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 OP_MAPS_BASE_INDEX=0"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_hdf5_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=20
validate "./airfoil_hdf5_mpi_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 14 ./airfoil_hdf5_mpi_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV
validate "./airfoil_hdf5_mpi_openacc OP_PART_SIZE=128 OP_BLOCK_SIZE=192 OP_MAPS_BASE_INDEX=0"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_hdf5_mpi_openacc OP_PART_SIZE=128 OP_BLOCK_SIZE=192 OP_MAPS_BASE_INDEX=0"


###################################################################################
###################################################################################

#COMMENT0

echo " "
echo " "
echo "**********************************************************************"
echo "********************* Building C/C++ back-end libs with Clang Compilers"
echo "**********************************************************************"
cd $OP2_INSTALL_PATH/c
. $CURRENT_DIR/$CLANG_SOURCE
pwd

make clean; make

echo " "
echo " "
echo "=======================> Building Aero Plain DP with Clang Compilers"
cd $OP2_APPS_DIR/c/aero/aero_plain/dp/
$OP2_C_CODEGEN_DIR/op2.py aero.cpp
$OP2_C_CODEGEN_DIR/op2.py aero_mpi.cpp
make clean;make; make aero_openmp4;
echo " "
echo " "
echo "=======================> Building Aero HDF5 DP with Clang Compilers"
cd $OP2_APPS_DIR/c/aero/aero_hdf5/dp/
$OP2_C_CODEGEN_DIR/op2.py aero.cpp
make clean;make;make aero_openmp4;



echo " "
echo " "
echo "=======================> Running Aero Plain DP built with Clang Compilers"
cd $OP2_APPS_DIR/c/aero/aero_plain/dp
./aero_seq
./aero_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./aero_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./aero_mpi
./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./aero_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./aero_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=20
#./aero_mpi_openmp4 OP_PART_SIZE=256

#COMMENT0

echo " "
echo " "
echo "=======================> Running Aero HDF5 DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/aero/aero_hdf5/dp
./aero_seq
./aero_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./aero_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./aero_mpi
./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./aero_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 12 ./aero_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=20
#./aero_mpi_openmp4 OP_PART_SIZE=256

echo "All tests Passed !"
echo "End of Test Script !"
