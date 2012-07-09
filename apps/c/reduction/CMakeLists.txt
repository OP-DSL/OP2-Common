project(OP2-reduce)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

include(../common.cmake)

set(KERNEL_HEADERS count.h addto.h addto_id.h)

# sequential version
op2_application(reduce_seq LIBS op2_seq SOURCES reduce.cpp)

# x86 with kernel files from op2.m
op2_application(reduce_openmp LIBS op2_openmp
    SOURCES reduce_op.cpp reduce_kernels.cpp ${KERNEL_HEADERS})

# CUDA with kernel files from op2.m
op2_application(reduce_cuda LIBS op2_cuda
    SOURCES reduce_op.cpp reduce_kernels.cu ${KERNEL_HEADERS})
