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

SRC = src
INC = include
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
    MPICPP_PATH = mpic++
  endif
  endif

  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/bin/mpicxx)")
    MPICXX_PATH = $(MPI_INSTALL_PATH)/bin/mpicxx
  else
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/intel64/bin/mpicxx)")
    MPICXX_PATH = $(MPI_INSTALL_PATH)/intel64/bin/mpicxx
  else
    MPICXX_PATH = mpicxx
  endif
  endif

  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/bin/mpicc)")
    MPICC_PATH = $(MPI_INSTALL_PATH)/bin/mpicc
  else
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/intel64/bin/mpicc)")
    MPICC_PATH = $(MPI_INSTALL_PATH)/intel64/bin/mpicc
  else
    MPICC_PATH = mpicc
  endif
  endif
else
  MPICPP_PATH = mpic++
  MPICXX_PATH = mpicxx
  MPICC_PATH  = mpicc
endif

#
# Configure compilers:
#
ifeq ($(OP2_COMPILER),gnu)
  CCFLAGS  = -std=c99 -fPIC -DUNIX -Wall -g -O0
  CXX      = g++
  CXXFLAGS = -fPIC -DUNIX -Wall -g -O0 #-g -Wextra
  MPICXX   = $(MPICPP_PATH)
  MPIFLAGS = $(CXXFLAGS)
else
ifeq ($(OP2_COMPILER),intel)
  ifdef DEBUG
    CCFLAGS = -O0 -g -pg
  else
    CCFLAGS = -O3 -xHost #-DCOMM_PERF
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
  CUDA_ALIGNE_FLAG = -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  ifdef DEBUG
    CCFLAGS = -g -O0
  else
    CCFLAGS = -O3
  endif
  CXX	    = pgc++
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICPP_PATH)
  MPIFLAGS  = $(CXXFLAGS)
  # NVCCFLAGS += -ccbin=$(MPICXX)
else
ifeq ($(OP2_COMPILER),cray)
  CCFLAGS  = -O3
  CXX      = CC
  CXXFLAGS = $(CCFLAGS)
  MPICXX   = CC
  MPIFLAGS = $(CXXFLAGS)
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
  NVCCFLAGS = -ccbin=$(NVCC_HOST_COMPILER)
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
endif
endif
endif
endif

NVCCFLAGS += $(CODE_GEN_CUDA) -m64 -Xptxas=-v -use_fast_math -g -G -O0 #-O3

ifdef CUDA_INSTALL_PATH
  CUDA_INC = -I$(CUDA_INSTALL_PATH)/include
endif

PARMETIS_VER=4
ifdef PARMETIS_INSTALL_PATH
  PARMETIS_INC = -I$(PARMETIS_INSTALL_PATH)/include
  PARMETIS_LIB = -L$(PARMETIS_INSTALL_PATH)/lib
endif
PARMETIS_INC += -DHAVE_PARMETIS
PARMETIS_LIB += -lparmetis -lmetis
ifeq ($(PARMETIS_VER),4)
  PARMETIS_INC += -DPARMETIS_VER_4
endif

ifdef PTSCOTCH_INSTALL_PATH
  PTSCOTCH_INC = -I$(PTSCOTCH_INSTALL_PATH)/include
  PTSCOTCH_LIB = -L$(PTSCOTCH_INSTALL_PATH)/lib
endif
PTSCOTCH_INC += -DHAVE_PTSCOTCH
PTSCOTCH_LIB += -lptscotch -lptscotcherr

ifdef HDF5_INSTALL_PATH
  HDF5_INC = -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib
endif
HDF5_LIB += -lhdf5 -lz

.PHONY: clean mklib

all: clean core hdf5 seq openmp mpi_seq cuda mpi_cuda openmp4

mklib:
	@mkdir -p $(LIB) $(OBJ)

core: mklib $(INC)/op_lib_core.h $(SRC)/core/op_lib_core.c
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_lib_core.c -o $(OBJ)/op_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/externlib/op_util.c -o $(OBJ)/op_util.o

hdf5: mklib $(SRC)/externlib/op_hdf5.c $(INC)/op_hdf5.h
	$(MPICXX) $(CXXFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/op_hdf5.c \
	-o $(OBJ)/op_hdf5.o

	ar -r $(LIB)/libop2_hdf5.a $(OBJ)/op_hdf5.o $(OBJ)/op_util.o

seq: mklib core $(INC)/op_seq.h $(SRC)/sequential/op_seq.c $(OBJ)/op_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/sequential/op_seq.c -o $(OBJ)/op_seq.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/op_rt_support.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_dummy_singlenode.c -o $(OBJ)/op_dummy_singlenode.o
	ar -r $(LIB)/libop2_seq.a $(OBJ)/op_lib_core.o $(OBJ)/op_seq.o $(OBJ)/op_rt_support.o $(OBJ)/op_dummy_singlenode.o

cuda: mklib $(INC)/op_cuda_rt_support.h $(INC)/op_cuda_reduction.h $(SRC)/cuda/op_cuda_decl.c \
	$(SRC)/cuda/op_cuda_rt_support.c $(OBJ)/op_lib_core.o

	$(CXX) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) $(CUDA_INC) -I$(INC) \
	-c $(SRC)/cuda/op_cuda_decl.c -o $(OBJ)/op_cuda_decl.o
	$(CXX) $(CXXFLAGS) $(CUDA_ALIGNE_FLAG) -I$(CUDA_INSTALL_PATH)/include -I$(INC) -c $(SRC)/cuda/op_cuda_rt_support.c \
	-o $(OBJ)/op_cuda_rt_support.o
	$(CXX) $(CXXFLAGS) $(CUDA_ALIGNE_FLAG) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/op_rt_support.o

	ar -r $(LIB)/libop2_cuda.a $(OBJ)/op_cuda_rt_support.o \
	$(OBJ)/op_cuda_decl.o $(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o

openmp4: mklib $(INC)/op_rt_support.h $(SRC)/openmp4/op_openmp4_decl.c \
	$(SRC)/openmp4/op_openmp4_rt_support.c $(OBJ)/op_lib_core.o

	$(CXX) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(OMP4FLAGS) -I$(INC) \
	-c $(SRC)/openmp4/op_openmp4_decl.c -o $(OBJ)/op_openmp4_decl.o
	$(CXX) $(CXXFLAGS) $(OMP4FLAGS) -I$(INC) -c $(SRC)/openmp4/op_openmp4_rt_support.c \
	-o $(OBJ)/op_openmp4_rt_support.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/op_rt_support.o

	ar -r $(LIB)/libop2_openmp4.a $(OBJ)/op_openmp4_rt_support.o \
	$(OBJ)/op_openmp4_decl.o $(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o

openmp: mklib $(SRC)/openmp/op_openmp_decl.c $(OBJ)/op_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/op_rt_support.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/openmp/op_openmp_decl.c -o $(OBJ)/op_openmp_decl.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/op_rt_support.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_dummy_singlenode.c -o $(OBJ)/op_dummy_singlenode.o

	ar -r $(LIB)/libop2_openmp.a $(OBJ)/op_openmp_decl.o \
	$(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o $(OBJ)/op_dummy_singlenode.o

mpi_seq: mklib $(INC)/op_seq.h $(SRC)/mpi/op_mpi_decl.c \
	$(SRC)/mpi/op_mpi_part_core.c $(SRC)/mpi/op_mpi_core.c \
	$(INC)/op_hdf5.h $(SRC)/mpi/op_mpi_hdf5.c $(OBJ)/op_lib_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_core.c \
	-o $(OBJ)/op_mpi_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c \
	-o $(OBJ)/op_rt_support.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_decl.c \
	-o $(OBJ)/op_mpi_decl.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_rt_support.c \
	-o $(OBJ)/op_mpi_rt_support.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) $(PARMETIS_INC) $(PTSCOTCH_INC) \
	-c $(SRC)/mpi/op_mpi_part_core.c \
	-o $(OBJ)/op_mpi_part_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) $(HDF5_INC) \
	-c $(SRC)/mpi/op_mpi_hdf5.c \
	-o $(OBJ)/op_mpi_hdf5.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_util.cpp \
	-o $(OBJ)/op_mpi_util.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) $(PTSCOTCH_INC) -c $(SRC)/externlib/op_renumber.cpp \
	-o $(OBJ)/op_renumber.o

	ar -r $(LIB)/libop2_mpi.a $(OBJ)/op_mpi_core.o \
	$(OBJ)/op_mpi_part_core.o \
	$(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o \
	$(OBJ)/op_mpi_decl.o \
	$(OBJ)/op_mpi_rt_support.o \
	$(OBJ)/op_util.o \
	$(OBJ)/op_mpi_hdf5.o \
	$(OBJ)/op_mpi_util.o \
	$(OBJ)/op_renumber.o

mpi_cuda: mklib $(SRC)/mpi/op_mpi_part_core.c \
	$(SRC)/mpi/op_mpi_core.c $(INC)/op_hdf5.h $(SRC)/mpi/op_mpi_hdf5.c \
	$(OBJ)/op_lib_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_core.c \
	-o $(OBJ)/op_mpi_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c \
	-o $(OBJ)/op_rt_support.o
	$(MPICXX) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) -I$(INC) -DOPMPI -I$(CUDA_INSTALL_PATH)/include \
	-c $(SRC)/cuda/op_cuda_rt_support.c \
	-o $(OBJ)/op_cuda_rt_support.o
	$(MPICXX) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) -I$(INC) -I$(CUDA_INSTALL_PATH)/include \
	-c $(SRC)/mpi/op_mpi_cuda_rt_support.c \
	-o $(OBJ)/op_mpi_cuda_rt_support.o
	$(MPICXX) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) -I$(INC) -I$(CUDA_INSTALL_PATH)/include \
	-c $(SRC)/mpi/op_mpi_cuda_decl.c \
	-o $(OBJ)/op_mpi_cuda_decl.o
	$(MPICXX) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) -I$(INC) $(PARMETIS_INC) $(PTSCOTCH_INC) \
	-c $(SRC)/mpi/op_mpi_part_core.c \
	-o $(OBJ)/op_mpi_part_core.o
	$(MPICXX) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) -I$(INC) $(HDF5_INC) \
	-c $(SRC)/mpi/op_mpi_hdf5.c \
	-o $(OBJ)/op_mpi_hdf5.o
	nvcc $(NVCCFLAGS) -I$(INC) -I $(MPI_INSTALL_PATH)/include \
	-lcudart -c $(SRC)/mpi/op_mpi_cuda_kernels.cu \
	-o $(OBJ)/op_mpi_cuda_kernels.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_util.cpp \
	-o $(OBJ)/op_mpi_util.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) $(PTSCOTCH_INC) -c $(SRC)/externlib/op_renumber.cpp \
	-o $(OBJ)/op_renumber.o

	ar -r $(LIB)/libop2_mpi_cuda.a $(OBJ)/op_mpi_core.o \
	$(OBJ)/op_mpi_part_core.o \
	$(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o \
	$(OBJ)/op_cuda_rt_support.o \
	$(OBJ)/op_mpi_cuda_rt_support.o $(OBJ)/op_mpi_cuda_decl.o \
	$(OBJ)/op_util.o \
	$(OBJ)/op_mpi_hdf5.o \
	$(OBJ)/op_mpi_cuda_kernels.o \
	$(OBJ)/op_mpi_util.o \
	$(OBJ)/op_renumber.o


clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(LIB)/*.a
	-rm -if $(LIB)/*.so
	-rm -if $(SRC)/*~
	-rm -if $(SRC)/sequential/*~
	-rm -if $(SRC)/cuda/*~
	-rm -if $(SRC)/openmp/*~
	-rm -if $(SRC)/mpi/*~
	-rm -if $(SRC)/externlib/*~
	-rm -if $(INC)/*~
	-rm -if *~
