#! /bin/bash

rm -rf ./build
export OP2_INSTALL_PATH=/home/mudalige/OP2-GIT/OP2-Common/op2/
export SCOTCH_DIR=/opt/PT-Scotch-intel
export PARMETIS_INCLUDE_DIR=/opt/Parmetis-4-intel/
export PARMETIS_LIB_DIR=/opt/Parmetis-4-intel/
export HDF5_ROOT=/opt/hdf5-1.8.10-intel/
CC=icc CXX=icpc ./cmake.local -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=\"-O3 -xAVX -vec-report\" -DCMAKE_C_FLAGS_RELEASE=\"-O3 -xAVX -vec-report\" -DHDF5_DIR=/opt/hdf5-1.8.10-intel/ -DCUDA_NVCC_FLAGS=\" -gencode arch=compute_35,code=sm_35 -Xptxas -dlcm=ca -Xptxas=-v\"
