#! PRE
OP2_C_COMPILER = intel
OP2_C_CUDA_COMPILER = nvhpc
OP2_F_COMPILER = nvhpc

NV_ARCH = Volta
CUDA_INSTALL_PATH = /lustre/sw/nvidia/hpcsdk-219/Linux_x86_64/21.9/cuda

FC = /lustre/sw/nvidia/hpcsdk-219/Linux_x86_64/21.9/compilers/bin/nvfortran
NVCC = /lustre/sw/nvidia/hpcsdk-219/Linux_x86_64/21.9/compilers/bin/nvcc

OP2_LIB_FOR_EXTRA += -L/lustre/sw/intel/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64
OP2_LIB_FOR_EXTRA_MPI += -L/lustre/sw/intel/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64

#! POST
export MPICC_CC = $(CC)
export MPICXX_CXX = $(CXX)
export MPIF90_F90 = $(FC)
