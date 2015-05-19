#! /bin/bash

#Example script to set environmental variables to build op2 using cmake.
#The values here are for octon.arc.ox.ac.uk

. source_intel
cd $OP2_INSTALL_PATH/c
rm -rf ./build
echo $OP2_INSTALL_PATH
export SCOTCH_DIR=$PTSCOTCH_INSTALL_PATH
export PARMETIS_INCLUDE_DIR=$PARMETIS_INSTALL_PATH
export PARMETIS_LIB_DIR=$PARMETIS_INSTALL_PATH
export HDF5_ROOT=$HDF5_INSTALL_PATH
CC=icc CXX=icpc ./cmake.local -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=\"-O3 -xAVX -DMPICH_IGNORE_CXX_SEEK\" -DCMAKE_C_FLAGS_RELEASE=\"-O3 -xAVX -DMPICH_IGNORE_CXX_SEEK\" -DHDF5_DIR=$HDF5_INSTALL_PATH -DCUDA_NVCC_FLAGS=\" -gencode arch=compute_35,code=sm_35 -Xptxas -dlcm=ca -Xptxas=-v\"

cd  ../../apps/c
rm -rf ./build
CC=icc CXX=icpc ./cmake.local -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=\"-O3 -xAVX -vec-report -DMPICH_IGNORE_CXX_SEEK\" -DCMAKE_C_FLAGS_RELEASE=\"-O3 -xAVX -vec-report -DMPICH_IGNORE_CXX_SEEK\"  -DCUDA_PROPAGATE_HOST_FLAGS=OFF -DCUDA_NVCC_FLAGS=\"-gencode arch=compute_35,code=sm_35 -Xptxas -dlcm=ca -Xptxas=-v\"