#
# The following environment variables should be predefined:
#
# OP2_COMPILER (gnu,intel,etc)
#
# ... and optionally the following. If not defined, then 
# they must be locatable via PATH variables:
#
# CUDA_INSTALL_PATH
# PARMETIS_INSTALL_PATH
# PTSCOTCH_INSTALL_PATH
# HDF5_INSTALL_PATH
# MPI_INSTALL_PATH
#

#
# set paths for various files
#

LIB = lib
OBJ = obj
MOD = mod

#
# compiler settings
#

#
# Locate MPI compilers:
#
ifdef MPI_INSTALL_PATH
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/bin/mpic++)")
    MPICPP_PATH = $(MPI_INSTALL_PATH)/bin/mpic++
  else
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/intel64/bin/mpic++)")
    MPICPP_PATH = $(MPI_INSTALL_PATH)/intel64/bin/mpic++
  else
    MPICPP_PATH ?= mpic++
  endif
  endif

  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/bin/mpicxx)")
    MPICXX_PATH = $(MPI_INSTALL_PATH)/bin/mpicxx
  else
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/intel64/bin/mpicxx)")
    MPICXX_PATH = $(MPI_INSTALL_PATH)/intel64/bin/mpicxx
  else
    MPICXX_PATH ?= mpicxx
  endif
  endif

  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/bin/mpicc)")
    MPICC_PATH = $(MPI_INSTALL_PATH)/bin/mpicc
  else
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/intel64/bin/mpicc)")
    MPICC_PATH = $(MPI_INSTALL_PATH)/intel64/bin/mpicc
  else
    MPICC_PATH = ?mpicc
  endif
  endif
else
  MPICPP_PATH ?= mpic++
  MPICXX_PATH ?= mpicxx
  MPICC_PATH  ?= mpicc
endif

#
# Configure compilers:
#
ifeq ($(OP2_COMPILER),gnu)
  ifdef DEBUG
  CCFLAGS  = -std=c99 -fPIC -DUNIX -Wall -g -O0 -MMD -MP
  CXXFLAGS = -fPIC -DUNIX -Wall -g -O0 -MMD -MP #-g -Wextra
  else
  CCFLAGS  = -std=c99 -fPIC -DUNIX -Wall -g -O3 -MMD -MP
  CXXFLAGS = -fPIC -DUNIX -Wall -g -O3 -MMD -MP #-g -Wextra
  endif
  CXX      = g++
  MPICXX   = $(MPICPP_PATH)
  MPIFLAGS = $(CXXFLAGS)


  FC           = gfortran
  CC           = gcc
  CFLAGS       = -g -O2 -std=c99 -fPIC -Wall -pedantic -pipe $(INC)
  FFLAGS       = -O2 -Jmod -fPIC -Wall -pedantic -pipe -g -DOP2_ARG_POINTERS -ffixed-line-length-none -ffree-line-length-none
else
ifeq ($(OP2_COMPILER),intel)
  ifdef DEBUG
    CCFLAGS = -O0 -g -pg
  else
    CCFLAGS = -O3 -xHost
  endif
  CXX       = icpc
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICXX_PATH)
  MPIFLAGS  = $(CXXFLAGS)
else
ifeq ($(OP2_COMPILER),xl)
  ifdef DEBUG
    CCFLAGS = -O0 -g
  else
    CCFLAGS = -qarch=pwr8 -qtune=pwr8 -O3 -qhot
  endif
  CXX       = xlc++
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICXX_PATH)
  MPIFLAGS  = $(CXXFLAGS)
  OMP4FLAGS = -qsmp=omp -qoffload
else
ifeq ($(OP2_COMPILER),pgi)
  ifdef DEBUG
    CCFLAGS = -g -O0
  else
    CCFLAGS = -O3
  endif
  CXX	    = pgc++
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICPP_PATH)
  MPIFLAGS  = $(CXXFLAGS)
  # NVCXXFLAGS += -ccbin=$(MPICXX)
else
ifeq ($(OP2_COMPILER),cray)
  CCFLAGS  = -O3
  CXX      = CC
  CXXFLAGS = $(CCFLAGS)
  MPICXX   = CC
  MPIFLAGS = $(CXXFLAGS) #-fsanitize=signed-integer-overflow,unsigned-integer-overflow
else
ifeq ($(OP2_COMPILER),clang)
  ifdef DEBUG
    CCFLAGS  = -x c++ -O0 -I$(OMPTARGET_LIBS)/../include
  else
    CCFLAGS  = -x c++ -O3 -I$(OMPTARGET_LIBS)/../include
  endif
  CXX	    = clang++
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICPP_PATH)
  MPIFLAGS  = $(CXXFLAGS)
  NVCXXFLAGS = -ccbin=$(NVCXX_HOST_COMPILER)
  OMP4FLAGS = -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
else
print:
	@echo "unrecognised value for OP2_COMPILER. Set to one of: gnu, intel, xl, pgi, cray"
endif
endif
endif
endif
endif
endif

ifdef CPP_WRAPPER
	CXX = $(CPP_WRAPPER)
endif
ifdef MPICPP_WRAPPER
	MPICXX = $(MPICPP_WRAPPER)
endif

# flags for nvcc
# set NV_ARCH to select the correct one
ifndef NV_ARCH
  MESSAGE=select an NVIDA device to compile in CUDA, e.g. make NV_ARCH=KEPLER
  NV_ARCH=Kepler
endif
ifeq ($(NV_ARCH),Fermi)
  CODE_GEN_CUDA=-gencode arch=compute_20,code=sm_21
else
ifeq ($(NV_ARCH),Kepler)
  CODE_GEN_CUDA=-gencode arch=compute_35,code=sm_35
else
ifeq ($(NV_ARCH),Maxwell)
  CODE_GEN_CUDA=-gencode arch=compute_50,code=sm_50
else
ifeq ($(NV_ARCH),Pascal)
  CODE_GEN_CUDA=-gencode arch=compute_60,code=sm_60
else
ifeq ($(NV_ARCH),Volta)
  CODE_GEN_CUDA=-gencode arch=compute_70,code=sm_70
endif
endif
endif
endif
endif

NVCXX = nvcc
NVCXXFLAGS += $(CODE_GEN_CUDA) -m64 -Xptxas=-v -use_fast_math -g -O3 #-G -O0 #-O3

INC = -Iinclude

ifdef CUDA_INSTALL_PATH
  INC += -I$(CUDA_INSTALL_PATH)/include
endif

ifdef MPI_INSTALL_PATH
	MPI_INC = $(MPI_INSTALL_PATH)/include
endif

PARMETIS_VER=4
ifdef PARMETIS_INSTALL_PATH
  INC += -I$(PARMETIS_INSTALL_PATH)/include
  INC += -DHAVE_PARMETIS
endif

ifeq ($(PARMETIS_VER),4)
  INC += -DPARMETIS_VER_4
endif

ifdef PTSCOTCH_INSTALL_PATH
  INC += -I$(PTSCOTCH_INSTALL_PATH)/include
  INC += -DHAVE_PTSCOTCH
endif

ifdef HDF5_INSTALL_PATH
  INC += -I$(HDF5_INSTALL_PATH)/include -I$(HDF5_INSTALL_PATH)/gnu/9.1/include
endif

CCFLAGS += $(INC)
CXXFLAGS += $(INC)
MPIFLAGS += $(INC)
NVCXXFLAGS += $(INC)
FFLAGS += $(INC)

AR = ar rcs

OP2_LIBS = hdf5 seq cuda openmp openmp4 mpi mpi_cuda
OP2_FOR_LIBS = $(foreach lib, $(OP2_LIBS), f_$(lib))

.PHONY: all clean $(OP2_LIBS) $(OP2_FOR_LIBS)

all: $(OP2_LIBS) $(OP2_FOR_LIBS)

clean:
	-rm -rf $(OBJ)
	-rm -rf $(LIB)
	-rm -rf $(MOD)


# Base set of objects shared across multiple libraries
OP2_BASE = $(addprefix $(OBJ)/core/, \
	op_lib_core.o \
	op_rt_support.o)

OP2_FOR_BASE = $(addprefix $(OBJ)/fortran/, \
	op2_for_declarations.o \
	op2_for_reference.o \
	op2_for_rt_support.o \
	op2_for_rt_wrappers.o)

OP2_FOR_BASE_MPI = $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/, \
	op2_C_reference+mpi.o \
	op2_for_C_wrappers+mpi.o \
	op2_for_hdf5_declarations.o)

OP2_FOR_BASE += $(addprefix $(OBJ)/fortran/, \
	op2_C_reference.o \
	op2_for_C_wrappers.o)


# Complete object composition for each library
OP2_HDF5 = $(addprefix $(OBJ)/, \
	externlib/op_util.o \
	externlib/op_hdf5.o)

OP2_FOR_HDF5 = $(OP2_HDF5) $(addprefix $(OBJ)/fortran/, \
	op2_for_hdf5_declarations.o)

OP2_SEQ = $(OP2_BASE) $(addprefix $(OBJ)/, \
	core/op_dummy_singlenode.o \
	sequential/op_seq.o)

OP2_FOR_SEQ = $(OP2_SEQ) $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/, \
	op_dummy_wrappers.o)

OP2_CUDA = $(OP2_BASE) $(addprefix $(OBJ)/, \
	cuda/op_cuda_decl.o \
	cuda/op_cuda_rt_support.o)

OP2_FOR_CUDA = $(OP2_CUDA) $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/, \
	op2_for_rt_wrappers_cuda.o \
	cudaConfigurationParams.o)

OP2_OPENMP = $(OP2_BASE) $(addprefix $(OBJ)/, \
	core/op_dummy_singlenode.o \
	openmp/op_openmp_decl.o)

OP2_FOR_OPENMP = $(OP2_OPENMP) $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/, \
	op_dummy_wrappers.o)

OP2_OPENMP4 = $(OP2_BASE) $(addprefix $(OBJ)/, \
	core/op_dummy_singlenode.o \
	openmp4/op_openmp4_decl.o \
	openmp4/op_openmp4_rt_support.o)

OP2_FOR_OPENMP4 = $(OP2_OPENMP4) $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/, \
	op_dummy_wrappers.o)

OP2_MPI = $(OP2_BASE) $(addprefix $(OBJ)/, \
	mpi/op_mpi_core.o \
	mpi/op_mpi_part_core.o \
	mpi/op_mpi_decl.o \
	mpi/op_mpi_rt_support.o \
	mpi/op_mpi_hdf5.o \
	mpi/op_mpi_util.o \
	externlib/op_util.o \
	externlib/op_renumber.o)

OP2_FOR_MPI = $(OP2_MPI) $(OP2_FOR_BASE_MPI) $(addprefix $(OBJ)/fortran/, \
	op_dummy_wrappers+mpi.o)

OP2_MPI_CUDA = $(OP2_BASE) $(addprefix $(OBJ)/, \
	cuda/op_cuda_rt_support+mpi.o \
	mpi/op_mpi_core.o \
	mpi/op_mpi_part_core.o \
	mpi/op_mpi_cuda_decl.o \
	mpi/op_mpi_cuda_rt_support.o \
	mpi/op_mpi_cuda_kernels.o \
	mpi/op_mpi_hdf5.o \
	mpi/op_mpi_util.o \
	externlib/op_util.o \
	externlib/op_renumber.o)

OP2_FOR_MPI_CUDA = $(OP2_MPI_CUDA) $(OP2_FOR_BASE_MPI) $(addprefix $(OBJ)/fortran/, \
	op2_for_rt_wrappers_cuda.o)


# Directory rules
$(OBJ):
	@mkdir -p $@
	@mkdir -p $(foreach dir, $(shell find src -maxdepth 1 -mindepth 1 \
		-exec basename {} \;), $@/$(dir))

$(LIB):
	@mkdir -p $@

$(MOD):
	@mkdir -p $@


# Rules for files with a specific compilation setup
$(OBJ)/externlib/op_renumber.o: src/externlib/op_renumber.cpp | $(OBJ)
	$(MPICXX) $(MPIFLAGS) -c $< -o $@

$(OBJ)/cuda/%+mpi.o: src/cuda/%.cpp | $(OBJ)
	$(MPICXX) $(MPIFLAGS) -DOPMPI -c $< -o $@

$(OBJ)/cuda/%.o: src/cuda/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -DSET_CUDA_CACHE_CONFIG -c $< -o $@

$(OBJ)/openmp4/%.o: src/openmp4/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(OMP4FLAGS) -c $< -o $@

$(OBJ)/mpi/%.o: src/mpi/%.cpp | $(OBJ)
	$(MPICXX) $(MPIFLAGS) -c $< -o $@

$(OBJ)/mpi/%.o: src/mpi/%.cu | $(OBJ)
	$(NVCXX) $(NVCXXFLAGS) -I $(MPI_INSTALL_PATH)/include -lcudart -c $< -o $@

$(OBJ)/fortran/%+mpi.o: src/fortran/%.c | $(OBJ)
	$(CC) $(CFLAGS) -DOPMPI -c $< -o $@

# Compile OP2_Fortran_Declarations first, as other F90 sources use it as a module
$(OBJ)/fortran/op2_for_declarations.o: src/fortran/op2_for_declarations.F90 | $(OBJ) $(MOD)
	$(FC) $(FFLAGS) -c $< -o $@

# Manually run C preprocessor on op2_for_reference.F90 to generate the op_par_loop subroutines
$(OBJ)/fortran/op2_for_reference.o: src/fortran/op2_for_reference.F90 $(OBJ)/fortran/op2_for_declarations.o | $(OBJ) $(MOD)
	$(shell cpp $(INC) src/fortran/op2_for_reference.F90 | sed s/##//g | sed s/\"@\"//g | tr "@" "\\n" \
		> $(OBJ)/fortran/op2_for_reference+cpp.F90)
	$(FC) $(FFLAGS) -c $(OBJ)/fortran/op2_for_reference+cpp.F90 -o $@

$(OBJ)/fortran/%.o: src/fortran/%.F90 $(OBJ)/fortran/op2_for_declarations.o | $(OBJ) $(MOD)
	$(FC) $(FFLAGS) -c $< -o $@

# Catch-all rules
$(OBJ)/%.o: src/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ)/%.o: src/%.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

# Include auto-generated dependencies for C/CXX sources (if they exist)
-include $(shell find $(OBJ) -type f -name "*.d")


# Helper function to upper-case a string
UPPERCASE = $(shell echo "$(1)" | tr "[:lower:]" "[:upper:]")

# Template for generating the library rules libop2_$1.a and libop2_for_$1.a
# and the compatability PHONY rules $1 and f_$1
define OP2_LIB_template =
$$(LIB)/libop2_$(1).a: $$(OP2_$(call UPPERCASE,$(1))) | $$(LIB)
	$(AR) $$@ $$?

$$(LIB)/libop2_for_$(1).a: $$(OP2_FOR_$(call UPPERCASE,$(1))) | $$(LIB)
	$(AR) $$@ $$?

$(1): $$(LIB)/libop2_$(1).a
f_$(1): $$(LIB)/libop2_for_$(1).a
endef

# Expand the template for all of the libraries (seq, ..., mpi_cuda)
$(foreach lib,$(OP2_LIBS),$(eval $(call OP2_LIB_template,$(lib))))
