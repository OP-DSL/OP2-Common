#! /bin/bash
rm -rf ./build
#HDF5_DIR==/opt/hdf5-1.8.10-intel/
#HDF5_DIR=/home/ireguly/hdf5-1.8.12/hdf5/
#export LD_LIBRARY_PATH=/home/ireguly/hdf5-1.8.12/hdf5/lib/:$LD_LIBRARY_PATH
CC=icc CXX=icpc ./cmake.local -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=\"-O3 -xAVX -vec-report\" -DCMAKE_C_FLAGS_RELEASE=\"-O3 -xAVX -vec-report\"  -DCUDA_PROPAGATE_HOST_FLAGS=OFF
