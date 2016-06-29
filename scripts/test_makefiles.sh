#!/bin/bash

#
#Test OP2 example applications using the basic makefiles build
#

#exit script if any error is encountered during the build or
#application executions.
#set -e

function validate {
  $1 > perf_out
  echo
  echo $1
  grep "Max total runtime" perf_out;grep "PASSED" perf_out
  rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;rm perf_out
}


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

echo " "
echo " "
echo "**********************************************************************"
echo "***********************> Building C back-end libs with Intel Compilers"
echo "**********************************************************************"
. $CURRENT_DIR/source_intel
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
echo "=======================> Building Airfoil TEMPDATS DP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_tempdats/dp/
$OP2_C_CODEGEN_DIR/op2.py airfoil.cpp
make clean;make

echo " "
echo " "
echo "=======================> Building Aero Plain DP with Intel Compilers"
cd $OP2_APPS_DIR/c/aero/aero_plain/dp/
#$OP2_C_CODEGEN_DIR/op2.py aero.cpp
make clean;make
echo " "
echo " "
echo "=======================> Building Aero HDF5 DP with Intel Compilers"
cd $OP2_APPS_DIR/c/aero/aero_hdf5/dp/
#$OP2_C_CODEGEN_DIR/op2.py aero.cpp
make clean;make

#COMMENT1

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
make clean;make

#COMMENT1

#<<COMMENT1

echo " "
echo " "
echo "**********************************************************************"
echo "**************************** Running C Apps built with Intel Compilers"
echo "**********************************************************************"

echo " "
echo " "
echo "=======================> Running Airfoil Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_plain/dp
#validate "./airfoil_seq"
validate "./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
export OMP_NUM_THREADS=20
validate "./airfoil_openmp OP_PART_SIZE=256"
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_vec"
validate "./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
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
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_genseq"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_vec"
validate "./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda_hyb OP_PART_SIZE=128 OP_BLOCK_SIZE=192"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_mpi_cuda_hyb OP_PART_SIZE=128 OP_BLOCK_SIZE=192"

export OMP_NUM_THREADS=20
validate "./airfoil_mpi_openmp OP_PART_SIZE=256"
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_mpi_openmp OP_PART_SIZE=256"


echo " "
echo " "
echo "=======================> Running Airfoil Tempdats DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_tempdats/dp
#validate "./airfoil_seq"
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


<<COMMENT1
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
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./numawrap20 ./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
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
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./numawrap20 ./aero_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./aero_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 12 ./aero_mpi_openmp OP_PART_SIZE=256

COMMENT1

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
. $CURRENT_DIR/source_intel
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

COMMENT2

###################################################################################
###################################################################################

#TOBE REMOVED
export LD_LIBRARY_PATH+=:/opt/compilers/intel/intelPS-2015/composer_xe_2015.2.164/compiler/lib/intel64/

echo " "
echo " "
echo "**********************************************************************"
echo "********************* Building Fortan back-end libs with PGI Compilers"
echo "**********************************************************************"
cd $OP2_INSTALL_PATH/fortran
. $CURRENT_DIR/source_pgi_16.4
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
export OMP_NUM_THREADS=20
validate "./airfoil_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV

echo " "
echo " "
echo "=======================> Running Airfoil Fortran HDF5 DP built with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_hdf5/dp
pwd
export PART_SIZE_ENV=128
validate "./airfoil_hdf5_seq OP_MAPS_BASE_INDEX=0"
validate "./airfoil_hdf5_cuda OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=20
validate "./airfoil_hdf5_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV
export OMP_NUM_THREADS=1
validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_hdf5_mpi OP_MAPS_BASE_INDEX=0"
validate "./airfoil_hdf5_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 OP_MAPS_BASE_INDEX=0"
validate "$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_hdf5_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192 OP_MAPS_BASE_INDEX=0"
export OMP_NUM_THREADS=20
validate "./airfoil_hdf5_mpi_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV
export OMP_NUM_THREADS=2
validate "$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_hdf5_mpi_openmp OP_MAPS_BASE_INDEX=0"
#_$PART_SIZE_ENV


