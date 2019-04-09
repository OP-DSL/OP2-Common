#!/bin/bash

#
#Test OP2 example applications using the basic makefiles build
#

#exit script if any error is encountered during the build or
#application executions.
set -e

function validate {
  $1 > perf_out
  echo
  echo $1
  grep "Max total runtime" perf_out;grep "PASSED" perf_out
  rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;rm perf_out
}

#export NV_ARCH=Kepler


#export CURRENT_DIR=$PWD
#cd ../op2
#export OP2_INSTALL_PATH=$PWD
#cd $OP2_INSTALL_PATH
#cd ../apps
#export OP2_APPS_DIR=$PWD
#export OP2_C_APPS_BIN_DIR=$OP2_APPS_DIR/c/bin
#cd ../translator/c/python/
#export OP2_C_CODEGEN_DIR=$PWD
#cd ../../fortran/python/
#export OP2_FORT_CODEGEN_DIR=$PWD
#cd $OP2_INSTALL_PATH/c



echo " "
echo " "
echo "**********************************************************************"
echo "***********************> Building C back-end libs with Intel Compilers"
echo "**********************************************************************"
source /home/sikba/source_intel
cd /home/sikba/warwick/op2/OP2-Common/op2/c/
#make clean; make mpi_seq;
make mpi_seq -B

echo " "
echo " "
echo "=======================> Building reproducible Airfoil HDF5 DP with Intel Compilers"
cd /home/sikba/warwick/op2/OP2-Common/apps/c/airfoil/airfoil_reproducible/airfoil_hdf5/dp/
#$OP2_C_CODEGEN_DIR/op2.py airfoil.cpp
make clean;make airfoil_mpi_genseq



echo " "
echo " "
echo "=======================> Running reproducible Airfoil HDF5 DP built with Intel Compilers"
cd /home/sikba/warwick/op2/OP2-Common/apps/c/airfoil/airfoil_reproducible/airfoil_hdf5/dp/
#validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_genseq"
#validate "$MPI_INSTALL_PATH/bin/mpirun -np 20 ./airfoil_mpi_genseq -renumber"



nproc_from=10
nproc_to=20
nproc_step=3

#validate "$MPI_INSTALL_PATH/bin/mpirun -np 1 ./airfoil_mpi_genseq"
$MPI_INSTALL_PATH/bin/mpirun -np 1 ./airfoil_mpi_genseq
cp repr_comp_p_q.h5 repr_comp_p_q_ref.h5
cp repr_comp_p_res.h5 repr_comp_p_res_ref.h5

for nproc in $(eval echo "{$nproc_from..$nproc_to..$nproc_step}")
do
 #   validate "$MPI_INSTALL_PATH/bin/mpirun -np $nproc ./airfoil_mpi_genseq"
    $MPI_INSTALL_PATH/bin/mpirun -np $nproc ./airfoil_mpi_genseq
    h5diff repr_comp_p_q.h5 repr_comp_p_q_ref.h5
    rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
    h5diff repr_comp_p_res.h5 repr_comp_p_res_ref.h5
    rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
    #h5diff repr_comp_p_res.h5 file_name_10procs_3rd_invalid.h5
    #rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
done

rm -f repr_comp_p_q.h5 repr_comp_p_q_ref.h5
rm -f repr_comp_p_res.h5 repr_comp_p_res_ref.h5

echo "All tests Passed !"
