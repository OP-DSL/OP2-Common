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


#export AMOS=TRUE
export DEMOS=TRUE
#export TELOS=TRUE
#export KOS=TRUE


if [[ -v KOS ]]; then
#Kos
export NV_ARCH=Pascal
export CUDA_VISIBLE_DEVICES=0,1
export INTEL_SOURCE=source_intel_18
export PGI_SOURCE=source_pgi_nvidia-hpc-21.7
export PGI_SOURCE=source_pgi_20_hydra
export CLANG_SOURCE=source_clang_kos
elif [[ -v TELOS ]]; then
#Telos
export NV_ARCH=Volta
export CUDA_VISIBLE_DEVICES=0
export INTEL_SOURCE=source_intel_18
#export PGI_SOURCE=source_pgi_20_hydra
export PGI_SOURCE=source_pgi_nvidia-hpc-21.7
export CLANG_SOURCE=source_clang_kos
elif [[ -v DEMOS ]]; then
#Demos
export NV_ARCH=Hopper
export CUDA_VISIBLE_DEVICES=0
#export INTEL_SOURCE=source_intel_18
export PGI_SOURCE=source_nvhpc-21.7_demos
#export CLANG_SOURCE=source_clang_kos
export GNU_SOURCE=source_gnu_demos
fi 

export CURRENT_DIR=$PWD
cd ../op2
export OP2_INSTALL_PATH=$PWD
cd $OP2_INSTALL_PATH
cd ../apps
export OP2_APPS_DIR=$PWD
export OP2_C_APPS_BIN_DIR=$OP2_APPS_DIR/c/bin
cd ../translator-v2/
export OP2_CODEGEN_DIR=$PWD
cd $OP2_INSTALL_PATH


echo " "
echo " "
echo "**********************************************************************"
echo "***********************> Building back-end libs with GNU Compilers    "
echo "**********************************************************************"
. $CURRENT_DIR/$GNU_SOURCE
make clean; make clean_config; make config; make;


echo " "
echo " "
echo "=======================> Building Airfoil Plain DP with Intel Compilers"
cd $OP2_APPS_DIR/c/airfoil/airfoil_plain/dp
python3 $OP2_CODEGEN_DIR/op2-translator airfoil.cpp
python3 $OP2_CODEGEN_DIR/op2-translator airfoil_mpi.cpp
make clean;make

exit
