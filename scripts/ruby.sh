#! /bin/bash

#Example script to set environmental variables to build op2 using cmake.
#The values here are for ruby.oerc.ox.ac.uk

export OP2_INSTALL_PATH=/home/mudalige/OP2-GIT/OP2-Common/op2/
export SCOTCH_DIR=/opt/PT-Scotch-intel
export PARMETIS_INCLUDE_DIR=/opt/Parmetis-4-intel/
export PARMETIS_LIB_DIR=/opt/Parmetis-4-intel/
export HDF5_ROOT=/opt/hdf5-1.8.10-intel/
cd $OP2_INSTALL_PATH/c
rm -rf ./build
CC=icc CXX=icpc ./cmake.local -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=\"-O3 -xAVX -DMPICH_IGNORE_CXX_SEEK\" -DCMAKE_C_FLAGS_RELEASE=\"-O3 -xAVX -DMPICH_IGNORE_CXX_SEEK\" -DHDF5_DIR=/opt/hdf5-1.8.10-intel/ -DCUDA_NVCC_FLAGS=\" -gencode arch=compute_35,code=sm_35 -Xptxas -dlcm=ca -Xptxas=-v\"


# build apps with cmake
cd $OP2_INSTALL_PATH
cd ../apps/c
rm -rf ./build
#-DCUDA_NVCC_FLAGS="-gencode arch=compute_20,code=sm_21 -Xptxas -dlcm=ca -Xptxas=-v"
CC=icc CXX=icpc ./cmake.local -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=\"-O3 -xAVX -DMPICH_IGNORE_CXX_SEEK\" -DCMAKE_C_FLAGS_RELEASE=\"-O3 -xAVX -DMPICH_IGNORE_CXX_SEEK\"  -DCUDA_PROPAGATE_HOST_FLAGS=OFF -DCUDA_NVCC_FLAGS=\"-gencode arch=compute_35,code=sm_35 -Xptxas -dlcm=ca -Xptxas=-v\"