#! /bin/bash

rm -rf ./build
export SCOTCH_DIR=/opt/PT-Scotch-intel
export PARMETIS_INCLUDE_DIR=/opt/Parmetis-4-intel/
export PARMETIS_LIB_DIR=/opt/Parmetis-4-intel/
export HDF5_ROOT=/opt/hdf5-1.8.10-intel/
#export HDF5_DIR=/opt/hdf5-1.8.10-intel/
#export OP2_HDF5_INCLUDE_DIRS=/opt/hdf5-1.8.10-intel/include/
CC=icc CXX=icpc ./cmake.local -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=\"-O3 -xAVX -vec-report\" -DCMAKE_C_FLAGS_RELEASE=\"-O3 -xAVX -vec-report\" -DHDF5_DIR=/opt/hdf5-1.8.10-intel/
