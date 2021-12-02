#! PRE
OP2_C_COMPILER := intel
OP2_F_COMPILER := intel

#! POST
export MPICC_CC := $(CC)
export MPICXX_CXX := $(CXX)
export MPIF90_F90 := $(FC)
