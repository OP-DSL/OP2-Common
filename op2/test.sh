#!/bin/bash
set -e

export OP2_INSTALL_PATH=/home/mudalige/OP2-GIT/OP2-Common/op2
export OP2_APPS_DIR=/home/mudalige/OP2-GIT/OP2-Common/apps
cd $OP2_INSTALL_PATH/c

##<<COMMENT1

echo " "
echo " "
echo "=======================> Building C back-end libs with Intel Compilers"
. source_intel
make clean; make 

########
echo " "
echo " "
echo "=======================> Building Airfoil Plain DP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_plain/dp
make clean;make

echo " "
echo " "
echo "=======================> Building Airfoil Plain SP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_plain/sp
make clean;make
echo " "
echo " "
echo "=======================> Building Airfoil HDF5 DP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_hdf5/dp
make clean;make
echo " "
echo " "
echo "=======================> Building Airfoil TEMPDATS DP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_tempdats/dp/
make clean;make

########
echo " "
echo " "
echo "=======================> Building Aero Plain DP with Intel Compilers"
cd $OP2_APPS_DIR/c/aero/aero_plain/dp/
make clean;make
echo " "
echo " "
echo "=======================> Building Aero HDF5 DP with Intel Compilers"
cd $OP2_APPS_DIR/c/aero/aero_hdf5/dp/
make clean;make

########
echo " "
echo " "
echo "=======================> Building Jac1 Plain DP with Intel Compilers"
cd $OP2_APPS_DIR/c/jac1/dp/
make clean;make
echo " "
echo " "
echo "=======================> Building Jac1 Plain SP with Intel Compilers"
cd $OP2_APPS_DIR/c/jac1/sp/
make clean;make
echo " "
echo " "
echo "=======================> Building Jac2 with Intel Compilers"
cd $OP2_APPS_DIR/c/jac2
make clean;make




################################################################################
echo " "
echo " "
echo "=======================> Running C Apps built with Intel Compilers"

########
echo " "
echo " "
echo "=======================> Running Airfoil Plain DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_plain/dp
#./airfoil_seq
./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=24
./airfoil_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 22 ./airfoil_mpi
./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=24
./airfoil_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 11 ./airfoil_mpi_openmp OP_PART_SIZE=256

echo " "
echo " "
echo "=======================> Running Airfoil HDF5 DP built with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_hdf5/dp
#./airfoil_seq
./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=24
./airfoil_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 22 ./airfoil_mpi
./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=24
./airfoil_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 11 ./airfoil_mpi_openmp OP_PART_SIZE=256




###COMMENT1






################################################################################
################################################################################

cd $OP2_INSTALL_PATH/fortran
echo " "
echo " "
echo "=======================> Building Fortan back-end libs with PGI Compilers"
. source_pgi
make clean; make 

########
echo " "
echo " "
echo "=======================> Building Airfoil Fortran Plain DP with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_plain/dp
export PART_SIZE_ENV=128
make clean; make 

########
echo " "
echo " "
echo "=======================> Building Airfoil Fortran HDF5 DP with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_hdf5/dp
export PART_SIZE_ENV=128
make clean; make 


################################################################################
echo " "
echo " "
echo "=======================> Running Fortran Apps built with PGI Compilers"

########
echo " "
echo " "
echo "=======================> Running Airfoil Fortran Plain DP built with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_plain/dp
export PART_SIZE_ENV=128
#./airfoil_seq
./airfoil_cuda 
export OMP_NUM_THREADS=24
./airfoil_openmp_$PART_SIZE_ENV


########
echo " "
echo " "
echo "=======================> Running Airfoil Fortran HDF5 DP built with PGI Compilers"
cd $OP2_APPS_DIR/fortran/airfoil/airfoil_hdf5/dp
export PART_SIZE_ENV=128
#./airfoil_hdf5_seq
./airfoil_hdf5_cuda 
export OMP_NUM_THREADS=24
./airfoil_hdf5_openmp_$PART_SIZE_ENV
export OMP_NUM_THREADS=1
$MPI_INSTALL_PATH/bin/mpirun -np 22 ./airfoil_hdf5_mpi
./airfoil_hdf5_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./airfoil_hdf5_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
export OMP_NUM_THREADS=24
./airfoil_hdf5_mpi_openmp OP_PART_SIZE=256
export OMP_NUM_THREADS=2
$MPI_INSTALL_PATH/bin/mpirun -np 11 ./airfoil_hdf5_mpi_openmp OP_PART_SIZE=256


