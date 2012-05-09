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

# Default to PGI compiler
OP2_COMPILER ?= pgi

ifeq ($(OP2_COMPILER),gnu)
  CC         = gcc
  CFLAGS     = -std=c99 -Wall -Werror -fPIC -pedantic -pipe
  FC         = gfortran
  FFLAGS     = -J$(F_OP2_MOD) -Wall -Werror -pedantic -pipe
  ALLTARGETS = f_reference f_openmp
else
ifeq ($(OP2_COMPILER),pgi)
  F_OP2_MOD_INTEL = $(F_OP2_MOD)/intel
  F_OP2_MOD_PGI   = $(F_OP2_MOD)/pgi
  CC              = icc -O3
  CFLAGS          = -std=c99 -Wall -Werror
  FC              = ifort -O3
  FFLAGS          = -module $(F_OP2_MOD_INTEL)
  PGCC            = nvcc
  PGCFLAGS        = -Xcompiler="-O4 -std=c99 -Wall -fPIC -pedantic -pipe"
  PGFC            = pgfortran
  PGFFLAGS        = -module $(F_OP2_MOD_PGI) -Mcuda=cuda4.0 -Minform=inform -DOP2_WITH_CUDAFOR
  ALLTARGETS      = f_reference f_openmp f_cuda
else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif

CUDA_INC = $(CUDA_INSTALL_PATH)/include

NVCC = nvcc

NVCCFLAGS = -O4

.PHONY: clean directories

# reference is not included in the all directive (missing gfortran on cx1) and
# is treated as a special case
all: $(ALLTARGETS)

directories:
	mkdir -p $(F_OP2_LIB)
	mkdir -p $(F_OP2_MOD) $(F_OP2_MOD_INTEL) $(F_OP2_MOD_PGI)
	mkdir -p $(F_OP2_OBJ)

# does not depend on reference.h because we don't need the op_par_loop templated
# implementation, but our own implementation
f_reference: $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/core/op_lib_core.c \
						 $(C_OP2_SRC)/sequential/op_seq.c \
						 $(F_OP2_SRC)/backend/op2_for_C_wrappers.c \
						 $(F_OP2_SRC)/backend/op2_for_declarations.F90 \
						 $(F_OP2_SRC)/backend/op2_for_reference.F90 \
						 $(F_OP2_SRC)/backend/op2_C_reference.c \
						 directories
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_lib_core.c -o $(F_OP2_OBJ)/op_lib_core.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/sequential/op_seq.c -o $(F_OP2_OBJ)/op_seq.o
	$(CC) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
	$(FC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_declarations.F90 -o $(F_OP2_OBJ)/op2_for_declarations.o
	$(shell cp $(F_OP2_SRC)/backend/op2_for_reference.F90 $(F_OP2_SRC)/backend/op2_for_reference.c)
	$(shell cpp $(F_OP2_SRC)/backend/op2_for_reference.c | sed s/##//g | sed s/\"@\"//g | tr "@" "\\n" > $(F_OP2_SRC)/backend/op2_cpp_reference.F90)
	$(FC) -c $(FFLAGS) $(F_OP2_SRC)/backend/op2_cpp_reference.F90 -o $(F_OP2_OBJ)/op2_for_reference.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_C_reference.c -o $(F_OP2_OBJ)/op2_C_reference.o
	ar -r $(F_OP2_LIB)/libop2_for_reference.a \
				$(F_OP2_OBJ)/op_lib_core.o \
				$(F_OP2_OBJ)/op_seq.o \
				$(F_OP2_OBJ)/op2_for_reference.o \
				$(F_OP2_OBJ)/op2_for_C_wrappers.o \
				$(F_OP2_OBJ)/op2_for_declarations.o \
				$(F_OP2_OBJ)/op2_C_reference.o

f_cuda: f_rt_support $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/core/op_lib_core.c \
				$(C_OP2_INC)/op_cuda_rt_support.h $(C_OP2_SRC)/cuda/op_cuda_rt_support.c \
				$(C_OP2_SRC)/cuda/op_cuda_decl.c \
				$(F_OP2_SRC)/backend/op2_for_plan_cuda.F90 \
				$(F_OP2_SRC)/backend/op2_for_rt_support.f90 \
				directories
	$(PGCC) $(PGCFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_lib_core.c -o $(F_OP2_OBJ)/op_lib_core.o
	$(NVCC) -Xcompiler -fno-strict-aliasing $(NVCCFLAGS) -I$(C_OP2_INC)  -I$(CUDA_INC) -c $(C_OP2_SRC)/cuda/op_cuda_decl.c -o $(F_OP2_OBJ)/op_cuda_decl.o
	$(PGCC) $(PGCFLAGS) -I$(C_OP2_INC) -I$(CUDA_INC) -c $(C_OP2_SRC)/cuda/op_cuda_rt_support.c -o $(F_OP2_OBJ)/op_cuda_rt_support.o
	$(PGCC) $(PGCFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
	$(PGFC) $(PGFFLAGS) -c $(F_OP2_SRC)/backend/op2_for_declarations.F90 -o $(F_OP2_OBJ)/op2_for_declarations.o
	$(PGFC) $(PGFFLAGS) -c $(F_OP2_SRC)/backend/op2_for_rt_support.f90 -o $(F_OP2_OBJ)/op2_for_rt_support.o
	$(PGFC) $(PGFFLAGS) -c $(F_OP2_SRC)/backend/op2_for_plan_cuda.F90 -o $(F_OP2_OBJ)/op2_for_plan_cuda.o
	$(PGFC) $(PGFFLAGS) -c $(F_OP2_SRC)/backend/cudaConfigurationParams.F90 -o $(F_OP2_OBJ)/cudaConfigurationParams.o
	ar -r $(F_OP2_LIB)/libop2_for_cuda.a \
				$(F_OP2_OBJ)/op_lib_core.o \
				$(F_OP2_OBJ)/op_cuda_decl.o \
				$(F_OP2_OBJ)/op_cuda_rt_support.o \
				$(F_OP2_OBJ)/op2_for_C_wrappers.o \
				$(F_OP2_OBJ)/op2_for_declarations.o \
				$(F_OP2_OBJ)/op2_for_rt_support.o \
				$(F_OP2_OBJ)/op2_for_plan_cuda.o \
				$(F_OP2_OBJ)/cudaConfigurationParams.o

f_openmp: f_openmp_rt_support $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/core/op_lib_core.c \
					$(C_OP2_SRC)/openmp/op_openmp_decl.c \
					$(F_OP2_SRC)/backend/op2_for_declarations.F90 \
					directories
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_lib_core.c -o $(F_OP2_OBJ)/op_lib_core.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/openmp/op_openmp_decl.c -o $(F_OP2_OBJ)/op_openmp_decl.o
	$(FC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_declarations.F90 -o $(F_OP2_OBJ)/op2_for_declarations.o
	ar -r $(F_OP2_LIB)/libop2_for_openmp.a \
				$(F_OP2_OBJ)/op2_for_C_wrappers.o \
				$(F_OP2_OBJ)/op_lib_core.o \
				$(F_OP2_OBJ)/op_openmp_decl.o \
				$(F_OP2_OBJ)/op2_for_declarations.o

f_openmp_rt_support: $(C_OP2_INC)/op_rt_support.h $(C_OP2_SRC)/core/op_rt_support.c \
										 $(F_OP2_SRC)/backend/op2_for_rt_wrappers.c \
										 $(F_OP2_SRC)/backend/op2_for_rt_support.f90 \
										 directories
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_rt_support.c -o $(F_OP2_OBJ)/op_rt_support.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_rt_wrappers.c -o $(F_OP2_OBJ)/op2_for_rt_wrappers.o
	$(FC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_rt_support.f90 -o $(F_OP2_OBJ)/op2_for_rt_support.o
	ar -r $(F_OP2_LIB)/libop2_for_openmp_rt_support.a \
				$(F_OP2_OBJ)/op_rt_support.o \
				$(F_OP2_OBJ)/op2_for_rt_wrappers.o \
				$(F_OP2_OBJ)/op2_for_rt_support.o

f_rt_support: $(C_OP2_INC)/op_rt_support.h $(C_OP2_SRC)/core/op_rt_support.c \
							$(F_OP2_SRC)/backend/op2_for_rt_wrappers.c $(F_OP2_SRC)/backend/op2_for_rt_support.f90 \
							directories
	$(PGCC) $(PGCFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_rt_support.c -o $(F_OP2_OBJ)/op_rt_support.o
	$(PGCC) $(PGCFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_rt_wrappers.c -o $(F_OP2_OBJ)/op2_for_rt_wrappers.o
	$(PGFC) $(PGFFLAGS) -c $(F_OP2_SRC)/backend/op2_for_rt_support.f90 -o $(F_OP2_OBJ)/op2_for_rt_support.o
	$(PGCC) $(PGCFLAGS) -I$(C_OP2_INC) -I$(F_OP2_SRC)/backend/ -I$(CUDA_INC) -c $(F_OP2_SRC)/backend/op2_for_rt_wrappers_cuda.c -o $(F_OP2_OBJ)/op2_for_rt_wrappers_cuda.o
	ar -r $(F_OP2_LIB)/libop2_for_rt_support.a \
				$(F_OP2_OBJ)/op_rt_support.o \
				$(F_OP2_OBJ)/op2_for_rt_wrappers.o \
				$(F_OP2_OBJ)/op2_for_rt_support.o	\
				$(F_OP2_OBJ)/op2_for_rt_wrappers_cuda.o

# the following is used when the application mixes openmp loops with reference loops
f_openmp_reference: f_openmp_rt_support $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/core/op_lib_core.c \
										$(C_OP2_SRC)/openmp/op_openmp_decl.c \
										$(F_OP2_SRC)/backend/op2_for_declarations.F90 \
										$(F_OP2_SRC)/backend/op2_for_reference.F90 \
										$(F_OP2_SRC)/backend/op2_C_reference.c \
										directories
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/core/op_lib_core.c -o $(F_OP2_OBJ)/op_lib_core.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(C_OP2_SRC)/openmp/op_openmp_decl.c -o $(F_OP2_OBJ)/op_openmp_decl.o
	$(FC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_declarations.F90 -o $(F_OP2_OBJ)/op2_for_declarations.o
	$(shell cp $(F_OP2_SRC)/backend/op2_for_reference.F90 $(F_OP2_SRC)/backend/op2_for_reference.c)
	$(shell cpp $(F_OP2_SRC)/backend/op2_for_reference.c | sed s/##//g | sed s/\"@\"//g | tr "@" "\\n" > $(F_OP2_SRC)/backend/op2_cpp_reference.F90)
	$(FC) -c $(FFLAGS) $(F_OP2_SRC)/backend/op2_cpp_reference.F90 -o $(F_OP2_OBJ)/op2_for_reference.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_C_reference.c -o $(F_OP2_OBJ)/op2_C_reference.o
	$(CC) $(CFLAGS) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_C_reference.c -o $(F_OP2_OBJ)/op2_C_reference.o
	ar -r $(F_OP2_LIB)/libop2_for_openmp_reference.a \
				$(F_OP2_OBJ)/op2_for_reference.o \
				$(F_OP2_OBJ)/op2_C_reference.o \
				$(F_OP2_OBJ)/op2_for_C_wrappers.o \
				$(F_OP2_OBJ)/op_lib_core.o \
				$(F_OP2_OBJ)/op_openmp_decl.o \
				$(F_OP2_OBJ)/op2_for_declarations.o

clean:
	-rm -if $(F_OP2_OBJ)/*.o
	-rm -if $(F_OP2_LIB)/*.a
	-rm -if $(F_OP2_SRC)/*~
	-rm -if $(F_OP2_SRC)/backend/*~
	-rm -if $(F_OP2_MOD)/*.mod
	-rm -if $(F_OP2)/*~

