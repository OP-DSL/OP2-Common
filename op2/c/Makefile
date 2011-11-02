#
# The following environment variables should be predefined:
#
# CUDA_INSTALL_PATH
# PARMETIS_INSTALL_PATH
# PTSCOTCH_INSTALL_PATH
# HDF5_INSTALL_PATH
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
  CC 		= gcc
  CCFLAGS	= -g -std=c99 -fPIC -DUNIX -Wall
  CPP		= g++
  CPPFLAGS	= -g -fPIC -DUNIX -Wall
#  CPPFLAGS	= -O3 -fPIC -DUNIX -Wall -Wextra
  MPICPP	= /usr/bin/mpiCC
  MPIFLAGS	= $(CPPFLAGS)
else
ifeq ($(OP2_COMPILER),intel)
  CC 		= icc
  CCFLAGS	= -O2 -vec-report -xSSE2,SSE3,SSE4.1,SSE4.2 #-g -DCOMM_PERF #-DDEBUG
  CPP		= icpc
  CPPFLAGS	= -O2 -vec-report -xSSE2,SSE3,SSE4.1,SSE4.2 #-g -DCOMM_PERF #-DDEBUG
  MPICPP	= mpiCC
  MPIFLAGS	= $(CPPFLAGS)
else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif

PARMETIS_INC = -I$(PARMETIS_INSTALL_PATH) -DPARMETIS
PARMETIS_LIB = -L$(PARMETIS_INSTALL_PATH) -lparmetis \
	       -L$(PARMETIS_INSTALL_PATH) -lmetis

PTSCOTCH_INC = -I$(PTSCOTCH_INSTALL_PATH)/include -DPTSCOTCH
PTSCOTCH_LIB = -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotch \
               -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotcherr

HDF5_INC = -I$(HDF5_INSTALL_PATH)/include
HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5 -lz


.PHONY: clean

all: core seq openmp cuda hdf5 mpi_seq

core: $(INC)/op_lib_core.h $(SRC)/core/op_lib_core.c
	$(CPP) $(CPPFLAGS) -I$(INC) -c $(SRC)/core/op_lib_core.c -o $(OBJ)/core/op_lib_core.o
	$(CPP) $(CPPFLAGS) -I$(INC) -c $(SRC)/externlib/op_util.c -o $(OBJ)/externlib/op_util.o

hdf5: $(SRC)/externlib/op_hdf5.c $(INC)/op_hdf5.h
	$(MPICPP) $(CPPFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/op_hdf5.c $(HDF5_LIB) -o $(OBJ)/externlib/op_hdf5.o

	ar -r $(LIB)/libop2_hdf5.a $(OBJ)/externlib/op_hdf5.o $(OBJ)/externlib/op_util.o


seq: $(INC)/op_seq.h $(SRC)/sequential/op_seq.c $(OBJ)/core/op_lib_core.o
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/sequential/op_seq.c -o $(OBJ)/op_seq.o

	ar -r $(LIB)/libop2_seq.a $(OBJ)/op_seq.o $(OBJ)/core/op_lib_core.o


cuda: $(INC)/op_cuda_rt_support.h $(INC)/op_cuda_reduction.h $(SRC)/cuda/op_cuda_decl.c \
	$(SRC)/cuda/op_cuda_rt_support.c $(OBJ)/core/op_lib_core.o

	$(CPP) -DSET_CUDA_CACHE_CONFIG $(FLAGS) $(CUDA_INC) -I$(INC) \
	-c $(SRC)/cuda/op_cuda_decl.c -o $(OBJ)/cuda/op_cuda_decl.o
	$(CPP) $(CPPFLAGS) -I$(CUDA_INSTALL_PATH)/include -I$(INC) -c $(SRC)/cuda/op_cuda_rt_support.c \
	-o $(OBJ)/cuda/op_cuda_rt_support.o
	$(CPP) $(CPPFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/core/op_rt_support.o

	ar -r $(LIB)/libop2_cuda.a $(OBJ)/cuda/op_cuda_rt_support.o \
	$(OBJ)/cuda/op_cuda_decl.o $(OBJ)/core/op_lib_core.o $(OBJ)/core/op_rt_support.o


openmp: $(SRC)/openmp/op_openmp_decl.c $(OBJ)/core/op_lib_core.o
	$(CPP) $(CPPFLAGS) -I$(INC) -c $(SRC)/openmp/op_openmp_decl.c -o $(OBJ)/openmp/op_openmp_decl.o
	$(CPP) $(CPPFLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/core/op_rt_support.o

	ar -r $(LIB)/libop2_openmp.a $(OBJ)/openmp/op_openmp_decl.o \
	$(OBJ)/core/op_lib_core.o $(OBJ)/core/op_rt_support.o


mpi_seq:$(INC)/op_mpi_seq.h $(SRC)/mpi/op_mpi_decl.c \
	$(SRC)/mpi/op_mpi_part_core.c $(SRC)/mpi/op_mpi_core.c \
	$(INC)/op_hdf5.h $(SRC)/mpi/op_mpi_hdf5.c $(OBJ)/core/op_lib_core.o

	$(MPICPP) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_core.c \
	-o $(OBJ)/mpi/op_mpi_core.o
	$(MPICPP) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_rt_support.c \
	-o $(OBJ)/mpi/op_mpi_rt_support.o
	$(MPICPP) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_decl.c \
	-o $(OBJ)/mpi/op_mpi_decl.o
	$(MPICPP) $(MPIFLAGS) -I$(INC) $(PARMETIS_INC) $(PTSCOTCH_INC) \
	$(PARMETIS_LIB) $(PTSCOTCH_LIB) \
	-c $(SRC)/mpi/op_mpi_part_core.c \
	-o $(OBJ)/mpi/op_mpi_part_core.o $(OBJ)/externlib/op_util.o
	$(MPICPP) $(MPIFLAGS) -I$(INC) $(HDF5_INC) $(HDF5_LIB) \
	-c $(SRC)/mpi/op_mpi_hdf5.c \
	-o $(OBJ)/mpi/op_mpi_hdf5.o $(OBJ)/externlib/op_util.o

	ar -r $(LIB)/libop2_mpi.a $(OBJ)/mpi/op_mpi_core.o \
	$(OBJ)/mpi/op_mpi_part_core.o \
	$(OBJ)/core/op_lib_core.o $(OBJ)/mpi/op_mpi_rt_support.o \
	$(OBJ)/mpi/op_mpi_decl.o \
	$(OBJ)/externlib/op_util.o \
	$(OBJ)/mpi/op_mpi_hdf5.o

clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(OBJ)/core/*.o
	-rm -if $(OBJ)/cuda/*.o
	-rm -if $(OBJ)/cuda/*~
	-rm -if $(OBJ)/openmp/*.o
	-rm -if $(OBJ)/openmp/*~
	-rm -if $(OBJ)/mpi/*.o
	-rm -if $(OBJ)/mpi/*~
	-rm -if $(OBJ)/externlib/*.o
	-rm -if $(OBJ)/externlib/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(SRC)/sequential/*~
	-rm -if $(SRC)/cuda/*~
	-rm -if $(SRC)/openmp/*~
	-rm -if $(SRC)/mpi/*~
	-rm -if $(SRC)/externlib/*~
	-rm -if $(INC)/*~
	-rm -if *~
