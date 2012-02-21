#
# The following environment variables are required:
#
# CUDA_INSTALL_PATH - CUDA base directory
# OP2_COMPILER      - Compiler toolchain: gnu (gcc, gfortran) or pgi (icc, pgfortran)
# OP2               - OP2 base directory
#
# Note: the CUDA library can only be built with the PGI toolchain

C_OP2 = $(OP2)/c
C_OP2_INC = $(C_OP2)/include
C_OP2_SRC = $(C_OP2)/src

F_OP2 = $(OP2)/fortran
F_OP2_INC = $(F_OP2)/include
F_OP2_SRC = $(F_OP2)/src
F_OP2_OBJ = $(F_OP2)/obj
F_OP2_LIB = $(F_OP2)/lib
F_OP2_MOD = $(F_OP2)/mod

F_OP2_USER = $(F_OP2_SRC)/user
F_OP2_BACKEND = $(F_OP2_SRC)/backend

#
# compiler settings
#

ifeq ($(OP2_COMPILER),gnu)
  CC		= gcc
	CFLAGS = -g -O3 -std=c99 -Wall -Werror -fPIC -pedantic -pipe
	FCC = gfortran
	FFLAGS = -J$(F_OP2_MOD) -O4 -Wall -Werror -pedantic -pipe
	ALLTARGETS = f_reference f_openmp_rt_support f_openmp
else
ifeq ($(OP2_COMPILER),pgi)
  CC		= icc
  CFLAGS	= -g -O3 -std=c99 -Wall -Werror -vec-report -xSSE4.2
	FCC = pgfortran
	FFLAGS = -module $(F_OP2_MOD) -Mcuda=cuda4.0 -Minform=inform -O4 -DOP2_WITH_CUDAFOR
	ALLTARGETS = f_reference f_openmp_rt_support f_openmp f_rt_support f_cuda
else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif

CUDA_INC = $(CUDA_INSTALL_PATH)/include

NVCC = nvcc

NVCCFLAGS = -O4

.PHONY: clean

# reference is not included in the all directive (missing gfortran on cx1) and is treated as a special case
all: $(ALLTARGETS)

# does not depend on reference.h because we don't need the op_par_loop templated implementation, but our own implementation
f_reference: $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/core/op_lib_core.c \
						 $(C_OP2_SRC)/sequential/op_seq.c \
						 $(F_OP2_SRC)/backend/op2_for_C_wrappers.c \
						 $(F_OP2_SRC)/backend/op2_for_declarations.F90 \
						 $(F_OP2_SRC)/backend/op2_for_reference.f90 \
						 $(F_OP2_SRC)/backend/op2_C_reference.c
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_lib_core.c -o $(F_OP2_OBJ)/op_lib_core.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/sequential/op_seq.c -o $(F_OP2_OBJ)/op_seq.o
	$(CC) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_declarations.F90 -o $(F_OP2_OBJ)/op2_for_declarations.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_reference.f90 -o $(F_OP2_OBJ)/op2_for_reference.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_C_reference.c -o $(F_OP2_OBJ)/op2_C_reference.o
	ar -r $(F_OP2_LIB)/libop2_for_reference.a \
				$(F_OP2_OBJ)/op_lib_core.o \
				$(F_OP2_OBJ)/op_seq.o \
				$(F_OP2_OBJ)/op2_for_reference.o \
				$(F_OP2_OBJ)/op2_for_C_wrappers.o \
				$(F_OP2_OBJ)/op2_for_declarations.o \
				$(F_OP2_OBJ)/op2_C_reference.o

f_cuda: $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/core/op_lib_core.c \
	$(C_OP2_INC)/op_cuda_rt_support.h $(C_OP2_SRC)/cuda/op_cuda_rt_support.c \
	$(C_OP2_SRC)/cuda/op_cuda_decl.c \
	$(F_OP2_SRC)/backend/op2_for_plan_cuda.f90
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_lib_core.c -o $(F_OP2_OBJ)/op_lib_core.o
	$(NVCC) -Xcompiler -fno-strict-aliasing $(NVCCFLAGS) -I$(C_OP2_INC)  -I$(CUDA_INC) -c $(C_OP2_SRC)/cuda/op_cuda_decl.c -o $(F_OP2_OBJ)/op_cuda_decl.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -I$(CUDA_INC) -c $(C_OP2_SRC)/cuda/op_cuda_rt_support.c -o $(F_OP2_OBJ)/op_cuda_rt_support.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_declarations.F90 -o $(F_OP2_OBJ)/op2_for_declarations.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_plan_cuda.f90 -o $(F_OP2_OBJ)/op2_for_plan_cuda.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/cudaConfigurationParams.F90 -o $(F_OP2_OBJ)/cudaConfigurationParams.o
	ar -r $(F_OP2_LIB)/libop2_for_cuda.a \
	$(F_OP2_OBJ)/op_lib_core.o \
	$(F_OP2_OBJ)/op_cuda_decl.o \
	$(F_OP2_OBJ)/op_cuda_rt_support.o \
	$(F_OP2_OBJ)/op2_for_C_wrappers.o \
	$(F_OP2_OBJ)/op2_for_declarations.o \
	$(F_OP2_OBJ)/op2_for_plan_cuda.o \
	$(F_OP2_OBJ)/cudaConfigurationParams.o

f_openmp: $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/core/op_lib_core.c \
		$(C_OP2_SRC)/openmp/op_openmp_decl.c \
		$(F_OP2_SRC)/backend/op2_for_declarations.F90
		$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_lib_core.c -o $(F_OP2_OBJ)/op_lib_core.o
		$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
		$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/openmp/op_openmp_decl.c -o $(F_OP2_OBJ)/op_openmp_decl.o
		$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_declarations.F90 -o $(F_OP2_OBJ)/op2_for_declarations.o
		ar -r $(F_OP2_LIB)/libop2_for_openmp.a \
		$(F_OP2_OBJ)/op2_for_C_wrappers.o \
		$(F_OP2_OBJ)/op_lib_core.o \
		$(F_OP2_OBJ)/op_openmp_decl.o \
		$(F_OP2_OBJ)/op2_for_declarations.o

f_openmp_rt_support: $(C_OP2_INC)/op_rt_support.h $(C_OP2_SRC)/core/op_rt_support.c \
	$(F_OP2_SRC)/backend/op2_for_rt_wrappers.c $(F_OP2_SRC)/backend/op2_for_rt_support.f90
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_rt_support.c -o $(F_OP2_OBJ)/op_rt_support.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_rt_wrappers.c -o $(F_OP2_OBJ)/op2_for_rt_wrappers.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_rt_support.f90 -o $(F_OP2_OBJ)/op2_for_rt_support.o
	ar -r $(F_OP2_LIB)/libop2_for_openmp_rt_support.a \
				$(F_OP2_OBJ)/op_rt_support.o \
				$(F_OP2_OBJ)/op2_for_rt_wrappers.o \
				$(F_OP2_OBJ)/op2_for_rt_support.o

f_rt_support: $(C_OP2_INC)/op_rt_support.h $(C_OP2_SRC)/core/op_rt_support.c \
	$(F_OP2_SRC)/backend/op2_for_rt_wrappers.c $(F_OP2_SRC)/backend/op2_for_rt_support.f90
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_rt_support.c -o $(F_OP2_OBJ)/op_rt_support.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_rt_wrappers.c -o $(F_OP2_OBJ)/op2_for_rt_wrappers.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_rt_support.f90 -o $(F_OP2_OBJ)/op2_for_rt_support.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -I$(F_OP2_SRC)/backend/ -I$(CUDA_INC) -c $(F_OP2_SRC)/backend/op2_for_rt_wrappers_cuda.c -o $(F_OP2_OBJ)/op2_for_rt_wrappers_cuda.o
	ar -r $(F_OP2_LIB)/libop2_for_rt_support.a \
				$(F_OP2_OBJ)/op_rt_support.o \
				$(F_OP2_OBJ)/op2_for_rt_wrappers.o \
				$(F_OP2_OBJ)/op2_for_rt_support.o	\
				$(F_OP2_OBJ)/op2_for_rt_wrappers_cuda.o

clean:
	-rm -if $(F_OP2_OBJ)/*.o
	-rm -if $(F_OP2_LIB)/*.a
	-rm -