IC = nvcc
FC = pgfortran

IC_OPT = -lcudart -O3 -use_fast_math
FC_OPT = -Mcuda=cuda3.1 -Minform=inform -fast -O4 -Mcuda=fastmath

# the following generates corresponding intermediate C CUDA code
# -Mcuda=keepgpu

all: cppfunctions cfunctions airfoil_comp link

cfunctions: op_seq.h op_seq.c debug.c
	$(IC) $(IC_OPT) -c  op_seq.c debug.c

cppfunctions: cutil_inline.h op_support.h op_support.cpp
	$(IC) $(IC_OPT) -c  op_support.cpp

airfoil_comp: OP2Profiling.F95 constantVarsCuda.CUF cudaConfigurationParams.F95 constvars.F95 op2_c.F95 airfoil_seqfun.F95 update_cuda_module.CUF update_cuda_module.CUF bres_calc_cuda_module.CUF res_calc_cuda_module.CUF adt_calc_cuda_module.CUF input.F95 output.F95 save_soln_cuda_module.CUF airfoil.F95
	$(FC) $(FC_OPT) -c OP2Profiling.F95 constantVarsCuda.CUF cudaConfigurationParams.F95 constvars.F95 op2_c.F95 airfoil_seqfun.F95 input.F95 output.F95 update_cuda_module.CUF bres_calc_cuda_module.CUF res_calc_cuda_module.CUF save_soln_cuda_module.CUF adt_calc_cuda_module.CUF airfoil.F95

link: OP2Profiling.F95 constantVarsCuda.o cudaConfigurationParams.o op_support.o op_seq.o debug.o constvars.o op2_c.o airfoil_seqfun.o input.o output.o update_cuda_module.o bres_calc_cuda_module.o res_calc_cuda_module.o save_soln_cuda_module.o adt_calc_cuda_module.o airfoil.o
	$(FC) $(FC_OPT) OP2Profiling.F95 constantVarsCuda.o cudaConfigurationParams.o op_support.o op_seq.o debug.o constvars.o op2_c.o airfoil_seqfun.o input.o output.o update_cuda_module.o bres_calc_cuda_module.o res_calc_cuda_module.o save_soln_cuda_module.o adt_calc_cuda_module.o airfoil.o -o airfoil_cuda


clean:
	rm -f *.o
	rm -f *.mod
	rm -f airfoil_cuda