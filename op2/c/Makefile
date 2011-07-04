
SRC = src
INC = include
LIB = lib
OBJ = obj
CUDA_INC = $(CUDA_INSTALL_PATH)/include
PARMETISLIB = $(PARMETIS_INSTALL_PATH)/libparmetis.a $(PARMETIS_INSTALL_PATH)/libmetis.a
MPI_HOME = /home/gihan/openmpi

CC = icc#gcc
CPP = icpc#g++
NVCC = nvcc
MPICPP = $(MPI_HOME)/bin/mpiCC

FLAGS = -O2 -vec-report -xSSE2,SSE3,SSE4.1,SSE4.2 #-DCOMM_PERF #-DDEBUG #-Wall -O3 -g

.PHONY: clean

all: core seq cuda openmp mpi_seq

core: $(INC)/op_lib_core.h $(SRC)/core/op_lib_core.c
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/core/op_lib_core.c -o $(OBJ)/op_lib_core.o

seq: $(INC)/op_seq.h $(SRC)/sequential/op_seq.c $(OBJ)/op_lib_core.o
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/sequential/op_seq.c -o $(OBJ)/op_seq.o
	ar -r $(LIB)/libop2_seq.a $(OBJ)/op_seq.o $(OBJ)/op_lib_core.o

cuda: $(INC)/op_cuda_rt_support.h $(INC)/op_cuda_reduction.h $(SRC)/cuda/op_cuda_decl.c $(SRC)/cuda/op_cuda_rt_support.c $(OBJ)/op_lib_core.o
	$(CPP) -fno-strict-aliasing -DSET_CUDA_CACHE_CONFIG $(FLAGS) -I$(CUDA_INC) -I$(INC) -c $(SRC)/cuda/op_cuda_decl.c -o $(OBJ)/cuda/op_cuda_decl.o
	$(CPP) -fno-strict-aliasing $(FLAGS) -I$(CUDA_INSTALL_PATH)/include -I$(INC) -c $(SRC)/cuda/op_cuda_rt_support.c -o $(OBJ)/cuda/op_cuda_rt_support.o
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/op_rt_support.o
	ar -r $(LIB)/libop2_cuda.a $(OBJ)/cuda/op_cuda_rt_support.o $(OBJ)/cuda/op_cuda_decl.o $(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o

openmp: $(SRC)/openmp/op_openmp_decl.c $(OBJ)/op_lib_core.o
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/openmp/op_openmp_decl.c -o $(OBJ)/openmp/op_openmp_decl.o
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c -o $(OBJ)/op_rt_support.o
	ar -r $(LIB)/libop2_openmp.a $(OBJ)/openmp/op_openmp_decl.o $(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o
	
mpi_seq:$(INC)/op_mpi_seq.h $(PARMETISLIB) $(SRC)/openmp/op_openmp_decl.c \
	$(SRC)/mpi/op_mpi_part_core.c $(SRC)/mpi/op_mpi_core.c $(OBJ)/op_lib_core.o
	
	$(MPICPP) $(FLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_util.c \
	-o $(OBJ)/mpi/op_mpi_util.o
	$(MPICPP) $(FLAGS) -I$(INC) -c $(SRC)/mpi/op_mpi_core.c \
	-o $(OBJ)/mpi/op_mpi_core.o
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/core/op_rt_support.c \
	-o $(OBJ)/op_rt_support.o
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/openmp/op_openmp_decl.c \
	-o $(OBJ)/openmp/op_openmp_decl.o
	$(MPICPP) $(FLAGS) -I$(INC) -I$(PARMETIS_INSTALL_PATH) \
	-c $(SRC)/mpi/op_mpi_part_core.c \
	-o $(OBJ)/mpi/op_mpi_part_core.o $(OBJ)/mpi/op_mpi_util.o $(PARMETISLIB)
	
	ar -r $(LIB)/libop2_mpi.a $(OBJ)/mpi/op_mpi_core.o \
	$(OBJ)/mpi/op_mpi_part_core.o \
	$(OBJ)/op_lib_core.o $(OBJ)/op_rt_support.o \
	$(OBJ)/openmp/op_openmp_decl.o \
	$(OBJ)/mpi/op_mpi_util.o

clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(OBJ)/cuda/*.o
	-rm -if $(OBJ)/cuda/*~
	-rm -if $(OBJ)/openmp/*.o
	-rm -if $(OBJ)/openmp/*~
	-rm -if $(OBJ)/mpi/*.o
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(SRC)/cuda/*~
	-rm -if $(SRC)/openmp/*~
	-rm -if $(SRC)/mpi/*~
	-rm -if $(INC)/*~
	-rm -if *~