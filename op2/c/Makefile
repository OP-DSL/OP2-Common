
SRC = src
INC = include
LIB = lib
OBJ = obj
CUDA_INC = $(CUDA_INSTALL_PATH)/include

CC = gcc
CPP = g++
NVCC = nvcc

FLAGS = -Wall -O3

.PHONY: clean

all: core reference runtime cuda openmp

core: $(INC)/op_lib_core.h $(SRC)/op_lib_core.c
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/op_lib_core.c -o $(OBJ)/op_lib_core.o

reference: $(INC)/op_reference.h $(SRC)/op_reference.c $(OBJ)/op_lib_core.o
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/op_reference.c -o $(OBJ)/op_reference.o
	ar -r $(LIB)/libop2_reference.a $(OBJ)/op_reference.o $(OBJ)/op_lib_core.o

runtime: $(INC)/op_rt_support.h $(SRC)/op_rt_support.c $(INC)/op_lib_core.h $(SRC)/op_lib_core.c
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/op_rt_support.c -o $(OBJ)/op_rt_support.o
	ar -r $(LIB)/libop2_rt_support.a $(OBJ)/op_rt_support.o

cuda: $(INC)/op_cuda_rt_support.h $(INC)/op_cuda_reduction.h $(SRC)/cuda/op_cuda_decl.c $(SRC)/cuda/op_cuda_rt_support.c $(OBJ)/op_lib_core.o
	$(CPP) -DSET_CUDA_CACHE_CONFIG -I$(CUDA_INC) -I$(INC) -c $(SRC)/cuda/op_cuda_decl.c -o $(OBJ)/cuda/op_cuda_decl.o
	$(CPP) -fno-strict-aliasing $(FLAGS) -I$(CUDA_INSTALL_PATH)/include -I$(INC) -c $(SRC)/cuda/op_cuda_rt_support.c -o $(OBJ)/cuda/op_cuda_rt_support.o
	ar -r $(LIB)/libop2_cuda.a $(OBJ)/cuda/op_cuda_rt_support.o $(OBJ)/cuda/op_cuda_decl.o $(OBJ)/op_lib_core.o

openmp: $(SRC)/openmp/op_openmp_decl.c $(OBJ)/op_lib_core.o
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/openmp/op_openmp_decl.c -o $(OBJ)/openmp/op_openmp_decl.o
	ar -r $(LIB)/libop2_openmp.a $(OBJ)/openmp/op_openmp_decl.o $(OBJ)/op_lib_core.o

clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(OBJ)/cuda/*.o
	-rm -if $(OBJ)/cuda/*~
	-rm -if $(OBJ)/openmp/*.o
	-rm -if $(OBJ)/openmp/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(SRC)/cuda/*~
	-rm -if $(SRC)/openmp/*~
	-rm -if $(INC)/*~
	-rm -if *~