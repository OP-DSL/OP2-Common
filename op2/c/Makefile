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
  CCFLAGS  = -std=c99 -fPIC -DUNIX -Wall -g -O0
  CXXFLAGS = -fPIC -DUNIX -Wall -g -O0 #-g -Wextra
  else
  CCFLAGS  = -std=c99 -fPIC -DUNIX -Wall -g -O3
  CXXFLAGS = -fPIC -DUNIX -Wall -g -O3 #-g -Wextra
  endif
  CXX      = g++
  MPICXX   = $(MPICPP_PATH)
  MPIFLAGS = $(CXXFLAGS)
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

AR = ar rcs

.PHONY: all clean hdf5 seq cuda openmp openmp4 mpi_seq mpi_cuda

all: hdf5 seq cuda openmp openmp4 mpi_seq mpi_cuda

$(OBJ):
	@mkdir -p $@

$(LIB):
	@mkdir -p $@

$(OBJ)/%.o: src/core/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ)/op_renumber.o: src/externlib/op_renumber.cpp | $(OBJ)
	$(MPICXX) $(MPIFLAGS) -c $< -o $@

$(OBJ)/%.o: src/externlib/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ)/%.o: src/sequential/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ)/op_cuda_rt_support_mpi.o: src/cuda/op_cuda_rt_support.cpp | $(OBJ)
	$(MPICXX) $(MPIFLAGS) -DOPMPI -c $< -o $@

$(OBJ)/%.o: src/cuda/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -DSET_CUDA_CACHE_CONFIG -c $< -o $@

$(OBJ)/%.o: src/openmp/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ)/%.o: src/openmp4/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(OMP4FLAGS) -c $< -o $@

$(OBJ)/%.o: src/mpi/%.cpp | $(OBJ)
	$(MPICXX) $(MPIFLAGS) -c $< -o $@

$(OBJ)/%.o: src/mpi/%.cu | $(OBJ)
	$(NVCXX) $(NVCXXFLAGS) -I $(MPI_INSTALL_PATH)/include -lcudart -c $< -o $@

$(LIB)/libop2_hdf5.a: \
	$(OBJ)/op_util.o \
	$(OBJ)/op_hdf5.o \
	| $(LIB)
	$(AR) $@ $?

$(LIB)/libop2_seq.a: \
	$(OBJ)/op_lib_core.o \
	$(OBJ)/op_rt_support.o \
	$(OBJ)/op_seq.o \
	$(OBJ)/op_dummy_singlenode.o \
	| $(LIB)
	$(AR) $@ $?

$(LIB)/libop2_cuda.a: \
	$(OBJ)/op_lib_core.o \
	$(OBJ)/op_rt_support.o \
	$(OBJ)/op_cuda_decl.o \
	$(OBJ)/op_cuda_rt_support.o \
	| $(LIB)
	$(AR) $@ $?

$(LIB)/libop2_openmp.a: \
	$(OBJ)/op_lib_core.o \
	$(OBJ)/op_rt_support.o \
	$(OBJ)/op_openmp_decl.o \
	$(OBJ)/op_dummy_singlenode.o \
	| $(LIB)
	$(AR) $@ $?

$(LIB)/libop2_openmp4.a: \
	$(OBJ)/op_lib_core.o \
	$(OBJ)/op_rt_support.o \
	$(OBJ)/op_openmp4_decl.o \
	$(OBJ)/op_openmp4_rt_support.o \
	| $(LIB)
	$(AR) $@ $?

$(LIB)/libop2_mpi.a: \
	$(OBJ)/op_lib_core.o \
	$(OBJ)/op_rt_support.o \
	$(OBJ)/op_util.o \
	$(OBJ)/op_mpi_core.o \
	$(OBJ)/op_mpi_part_core.o \
	$(OBJ)/op_mpi_decl.o \
	$(OBJ)/op_mpi_rt_support.o \
	$(OBJ)/op_mpi_hdf5.o \
	$(OBJ)/op_mpi_util.o \
	$(OBJ)/op_renumber.o \
	| $(LIB)
	$(AR) $@ $?

$(LIB)/libop2_mpi_cuda.a: \
	$(OBJ)/op_lib_core.o \
	$(OBJ)/op_rt_support.o \
	$(OBJ)/op_util.o \
	$(OBJ)/op_mpi_core.o \
	$(OBJ)/op_mpi_part_core.o \
	$(OBJ)/op_mpi_cuda_decl.o \
	$(OBJ)/op_cuda_rt_support_mpi.o \
	$(OBJ)/op_mpi_cuda_rt_support.o \
	$(OBJ)/op_mpi_cuda_kernels.o \
	$(OBJ)/op_mpi_hdf5.o \
	$(OBJ)/op_mpi_util.o \
	$(OBJ)/op_renumber.o \
	| $(LIB)
	$(AR) $@ $?


hdf5: $(LIB)/libop2_hdf5.a

seq: $(LIB)/libop2_seq.a

cuda: $(LIB)/libop2_cuda.a

openmp: $(LIB)/libop2_openmp.a

openmp4: $(LIB)/libop2_openmp4.a

mpi_seq: $(LIB)/libop2_mpi.a

mpi_cuda: $(LIB)/libop2_mpi_cuda.a

clean:
	-rm -rf $(OBJ)
	-rm -rf $(LIB)
	-rm -f src/*~
	-rm -f src/sequential/*~
	-rm -f src/cuda/*~
	-rm -f src/openmp/*~
	-rm -f src/mpi/*~
	-rm -f src/externlib/*~
	-rm -f include/*~
	-rm -f *~
