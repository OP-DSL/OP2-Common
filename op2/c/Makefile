#
# The following environment variables should be predefined:
#
# CUDA_INSTALL_PATH
# PARMETIS_INSTALL_PATH
# PTSCOTCH_INSTALL_PATH
# HDF5_INSTALL_PATH
# MPI_INSTALL_PATH
#
# OP2_COMPILER (gnu,intel,etc)
#

#
# set paths for various files
#

SRC = src
INC = include
LIB = lib
OBJ = obj

CUDA_INC = -I$(CUDA_INSTALL_PATH)/include

#
# compiler settings
#

ifeq ($(OP2_COMPILER),gnu)
	CC       := gcc
	CCFLAGS  := -std=c99 -fPIC -DUNIX -Wall #-g
	CXX      := g++
	CXXFLAGS := -std=c99 -fPIC -DUNIX -Wall #-g -Wextra
	MPICXX   := /usr/bin/mpiCC
	MPIFLAGS := $(CXXFLAGS)
else
ifeq ($(OP2_COMPILER),intel)
	CC       := icc
	CCFLAGS  := -O3 -vec-report -xSSE4.2 #-DCOMM_PERF #-DDEBUG -g
	CXX      := icpc
	CXXFLAGS := $(CCFLAGS)
	MPICXX   := mpiCC
	MPIFLAGS := $(CXXFLAGS)
else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif

NVCCFLAGS	= -O3 -arch=sm_20 -Xptxas -dlcm=ca -Xptxas=-v -use_fast_math #-g -G -O0

PARMETIS_INCLUDE_PATH ?= $(PARMETIS_INSTALL_PATH)
PARMETIS_LIB_PATH ?= $(PARMETIS_INSTALL_PATH)
PARMETIS_INC = -I$(PARMETIS_INCLUDE_PATH) -DHAVE_PARMETIS
PARMETIS_LIB = -L$(PARMETIS_LIB_PATH) -lparmetis -lmetis

PTSCOTCH_INCLUDE_PATH ?= $(PTSCOTCH_INSTALL_PATH)/include
PTSCOTCH_LIB_PATH ?= $(PTSCOTCH_INSTALL_PATH)/lib
PTSCOTCH_INC = -I$(PTSCOTCH_INCLUDE_PATH) -DHAVE_PTSCOTCH
PTSCOTCH_LIB = -L$(PTSCOTCH_LIB_PATH) -lptscotch -lptscotcherr

HDF5_INC = -I$(HDF5_INSTALL_PATH)/include
HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5 -lz

PETSC_INC = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
PETSC_LIB = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc

.PHONY: clean mklib

all: clean core hdf5 seq openmp cuda mpi_seq mpi_cuda

mklib:
	@mkdir -p $(LIB) $(OBJ)

core: mklib $(INC)/op_lib_core.h $(SRC)/core/op_lib_core.c $(SRC)/core/op_lib_util.cpp
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_lib_core.c -o $(OBJ)/op_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_lib_util.cpp -o $(OBJ)/op_lib_util.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/externlib/op_util.c -o $(OBJ)/op_util.o

hdf5: mklib $(SRC)/externlib/op_hdf5.c $(INC)/op_hdf5.h
	$(MPICXX) $(CXXFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/op_hdf5.c \
	$(HDF5_LIB) -o $(OBJ)/op_hdf5.o

	ar -r $(LIB)/libop2_hdf5.a $(OBJ)/op_hdf5.o $(OBJ)/op_util.o

seq: mklib core $(INC)/op_seq.h $(SRC)/sequential/op_seq.c $(OBJ)/op_lib_core.o $(OBJ)/op_lib_util.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/sequential/op_seq.c -o $(OBJ)/op_seq.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/op_rt_support.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_dummy_singlenode.c -o $(OBJ)/op_dummy_singlenode.o
        $(MPICXX) $(MPIFLAGS) -I$(INC) $(PETSC_INC) $(PETSC_LIB) -c $(SRC)/sequential/op_mat_seq.cpp -o $(OBJ)/op_mat_seq.o
	ar -r $(LIB)/libop2_seq.a $(OBJ)/op_seq.o $(OBJ)/op_lib_core.o $(OBJ)/op_mat_seq.o $(OBJ)/op_rt_support.o $(OBJ)/op_lib_util.o $(OBJ)/op_dummy_singlenode.o

cuda: mklib $(INC)/op_cuda_rt_support.h $(INC)/op_cuda_reduction.h $(SRC)/cuda/op_cuda_decl.c \
	$(SRC)/cuda/op_cuda_rt_support.c $(OBJ)/op_lib_core.o $(OBJ)/op_lib_util.o

	$(CXX) -DSET_CUDA_CACHE_CONFIG $(FLAGS) $(CUDA_INC) -I$(INC) \
	-c $(SRC)/cuda/op_cuda_decl.c -o $(OBJ)/op_cuda_decl.o
	$(CXX) $(CXXFLAGS) -I$(CUDA_INSTALL_PATH)/include -I$(INC) -c $(SRC)/cuda/op_cuda_rt_support.c \
	-o $(OBJ)/op_cuda_rt_support.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/op_rt_support.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/op_dummy_singlenode.c -o $(OBJ)/op_dummy_singlenode.o

	ar -r $(LIB)/libop2_cuda.a $(OBJ)/op_cuda_rt_support.o \
	$(OBJ)/op_cuda_decl.o $(OBJ)/op_lib_core.o $(OBJ)/op_lib_util.o $(OBJ)/op_rt_support.o  $(OBJ)/op_dummy_singlenode.o

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
	$(PARMETIS_LIB) $(PTSCOTCH_LIB) \
	-c $(SRC)/mpi/op_mpi_part_core.c \
	-o $(OBJ)/op_mpi_part_core.o $(OBJ)/op_util.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) $(HDF5_INC) $(HDF5_LIB) \
	-c $(SRC)/mpi/op_mpi_hdf5.c \
	-o $(OBJ)/op_mpi_hdf5.o $(OBJ)/op_util.o

	ar -r $(LIB)/libop2_mpi.a $(OBJ)/op_mpi_core.o \
	$(OBJ)/op_mpi_part_core.o \
	$(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o \
	$(OBJ)/op_mpi_decl.o \
	$(OBJ)/op_mpi_rt_support.o \
	$(OBJ)/op_util.o \
	$(OBJ)/op_mpi_hdf5.o

mpi_cuda: mklib $(SRC)/mpi/op_mpi_part_core.c \
	$(SRC)/mpi/op_mpi_core.c $(INC)/op_hdf5.h $(SRC)/mpi/op_mpi_hdf5.c \
	$(OBJ)/op_lib_core.o

	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_core.c \
	-o $(OBJ)/op_mpi_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c \
	-o $(OBJ)/op_rt_support.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -I$(CUDA_INSTALL_PATH)/include \
	-c $(SRC)/cuda/op_cuda_rt_support.c \
	-o $(OBJ)/op_cuda_rt_support.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -I$(CUDA_INSTALL_PATH)/include \
	-c $(SRC)/mpi/op_mpi_cuda_rt_support.c \
	-o $(OBJ)/op_mpi_cuda_rt_support.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -I$(CUDA_INSTALL_PATH)/include \
	-c $(SRC)/mpi/op_mpi_cuda_decl.c \
	-o $(OBJ)/op_mpi_cuda_decl.o

	$(MPICXX) $(MPIFLAGS) -I$(INC) $(PARMETIS_INC) $(PTSCOTCH_INC) \
	$(PARMETIS_LIB) $(PTSCOTCH_LIB) \
	-c $(SRC)/mpi/op_mpi_part_core.c \
	-o $(OBJ)/op_mpi_part_core.o $(OBJ)/op_util.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) $(HDF5_INC) $(HDF5_LIB) \
	-c $(SRC)/mpi/op_mpi_hdf5.c \
	-o $(OBJ)/op_mpi_hdf5.o $(OBJ)/op_util.o
	nvcc $(NVCCFLAGS) -I$(INC) -I $(MPI_INSTALL_PATH)/include \
	-lcudart -c $(SRC)/mpi/op_mpi_cuda_kernels.cu \
	-o $(OBJ)/op_mpi_cuda_kernels.o

	ar -r $(LIB)/libop2_mpi_cuda.a $(OBJ)/op_mpi_core.o \
	$(OBJ)/op_mpi_part_core.o \
	$(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o \
	$(OBJ)/op_cuda_rt_support.o \
	$(OBJ)/op_mpi_cuda_rt_support.o $(OBJ)/op_mpi_cuda_decl.o \
	$(OBJ)/op_util.o \
	$(OBJ)/op_mpi_hdf5.o \
	$(OBJ)/op_mpi_cuda_kernels.o

clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(SRC)/sequential/*~
	-rm -if $(SRC)/cuda/*~
	-rm -if $(SRC)/openmp/*~
	-rm -if $(SRC)/mpi/*~
	-rm -if $(SRC)/externlib/*~
	-rm -if $(INC)/*~
	-rm -if *~
