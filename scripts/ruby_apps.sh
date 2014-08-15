#! /bin/bash
rm -rf ./build
#-DCUDA_NVCC_FLAGS="-gencode arch=compute_20,code=sm_21 -Xptxas -dlcm=ca -Xptxas=-v"
CC=icc CXX=icpc ./cmake.local -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=\"-O3 -xAVX -vec-report\" -DCMAKE_C_FLAGS_RELEASE=\"-O3 -xAVX -vec-report\"  -DCUDA_PROPAGATE_HOST_FLAGS=OFF -DCUDA_NVCC_FLAGS=\"-gencode arch=compute_35,code=sm_35 -Xptxas -dlcm=ca -Xptxas=-v\"
