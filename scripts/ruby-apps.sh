#! /bin/bash
#. ../../scripts/source_intel
rm -rf ./build
#-DCUDA_NVCC_FLAGS="-gencode arch=compute_20,code=sm_21 -Xptxas -dlcm=ca -Xptxas=-v"
CC=icc CXX=icpc ./cmake.local -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=\"-O3 -xAVX -vec-report -DMPICH_IGNORE_CXX_SEEK\" -DCMAKE_C_FLAGS_RELEASE=\"-O3 -xAVX -vec-report -DMPICH_IGNORE_CXX_SEEK\"  -DCUDA_PROPAGATE_HOST_FLAGS=OFF -DCUDA_NVCC_FLAGS=\"-gencode arch=compute_35,code=sm_35 -Xptxas -dlcm=ca -Xptxas=-v\"
