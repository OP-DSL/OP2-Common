#!/bin/bash

#
#Test OP2 example applications using the basic makefiles build
#

#exit script if any error is encountered during the build or
#application executions.
set -e

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

echo " "
echo " "
echo "**********************************************************************"
echo "***********************> Building C back-end libs with Intel Compilers"
echo "**********************************************************************"
. $CURRENT_DIR/source_intel
make clean; make

#<<COMMENT1

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

#COMMENT1


echo " "
echo " "
echo "**********************************************************************"
echo "**************************** Running C Apps built with Intel Compilers"
echo "**********************************************************************"

echo " "
echo " "
echo "=======================> Running Airfoil Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_plain/dp
./airfoil_seq
./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./airfoil_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_vec
./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./airfoil_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 12 ./airfoil_mpi_openmp OP_PART_SIZE=256

#<<COMMENT1
echo " "
echo " "
echo "=======================> Running Airfoil HDF5 DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_hdf5/dp
./airfoil_seq
./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./airfoil_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_genseq
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_vec
./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda_hyb OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_mpi_cuda_hyb OP_PART_SIZE=128 OP_BLOCK_SIZE=192

export OMP_NUM_THREADS=20
./airfoil_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 12 ./airfoil_mpi_openmp OP_PART_SIZE=256


echo " "
echo " "
echo "=======================> Running Airfoil Tempdats DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_tempdats/dp
./airfoil_seq
./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./airfoil_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi
./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./airfoil_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 12 ./airfoil_mpi_openmp OP_PART_SIZE=256

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
$MPI_INSTALL_PATH/bin/mpirun -np 12 ./aero_mpi_openmp OP_PART_SIZE=256


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


echo " "
echo " "
echo "=======================> Running Jac1 Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/jac1/dp/
./jac_seq
./jac_cuda
export OMP_NUM_THREADS=20
./jac_openmp
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./jac_mpi


echo " "
echo " "
echo "=======================> Running Jac1 Plain SP built with Intel Compilers"
cd $OP2_APPS_DIR/c/jac1/sp/
./jac_seq
./jac_cuda
export OMP_NUM_THREADS=20
./jac_openmp
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./jac_mpi

echo " "
echo " "
echo "=======================> Running Jac2 Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/jac2/
./jac_seq
./jac_cuda
export OMP_NUM_THREADS=20
./jac_openmp
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./jac_mpi

#<<COMMENT1
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
./airfoil_seq
./airfoil_vec
export OMP_NUM_THREADS=20
./airfoil_openmp_$PART_SIZE_ENV



echo " "
echo " "
echo "=======================> Running Airfoil Fortran HDF5 DP built with Intel Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_hdf5/dp
pwd
export PART_SIZE_ENV=128
./airfoil_hdf5_seq
#./airfoil_hdf5_vec
./airfoil_hdf5_openmp_$PART_SIZE_ENV
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_hdf5_mpi
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_hdf5_mpi_vec
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_hdf5_mpi_genseq
export OMP_NUM_THREADS=20
./airfoil_hdf5_mpi_openmp_$PART_SIZE_ENV
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_hdf5_mpi_openmp_$PART_SIZE_ENV

#COMMENT1
###################################################################################
###################################################################################
echo " "
echo " "
echo "**********************************************************************"
echo "********************* Building Fortan back-end libs with PGI Compilers"
echo "**********************************************************************"
cd $OP2_INSTALL_PATH/fortran
. $CURRENT_DIR/source_pgi_15.1
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
./airfoil_seq
./airfoil_cuda
export OMP_NUM_THREADS=20
./airfoil_openmp_$PART_SIZE_ENV

echo " "
echo " "
echo "=======================> Running Airfoil Fortran HDF5 DP built with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_hdf5/dp
pwd
export PART_SIZE_ENV=128
./airfoil_hdf5_seq
./airfoil_hdf5_cuda
export OMP_NUM_THREADS=20
./airfoil_hdf5_openmp_$PART_SIZE_ENV
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_hdf5_mpi
./airfoil_hdf5_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_hdf5_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=20
./airfoil_hdf5_mpi_openmp_$PART_SIZE_ENV
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./airfoil_hdf5_mpi_openmp_$PART_SIZE_ENV


